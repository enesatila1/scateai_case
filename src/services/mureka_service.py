import logging
import replicate
import tempfile
import os
import io
import uuid
import threading
import queue
import gc
from ..config import app_config, GenerateSongRequest, BillingResponse, JobStatusResponse
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
import requests as req

logger = logging.getLogger(__name__)


class ReplicateService:
    """Service for interacting with Replicate API for music generation"""

    def __init__(self):
        # Set API token for replicate
        os.environ["REPLICATE_API_TOKEN"] = app_config.replicate_api_key
        # Increase timeout for long-running operations
        os.environ["REPLICATE_TIMEOUT"] = "600"  # 10 minutes

        self.client = replicate.Client(api_token=app_config.replicate_api_key)
        self.song_model = app_config.song_generation_model
        self.voice_model = app_config.voice_cloning_model

        # Job queue system
        self.jobs = {}  # UUID → {status, progress, result, error}
        self.job_queue = queue.Queue()
        self.worker_running = True

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    async def submit_cover_job(self, song_file, voice_sample):
        """Submit a cover generation job to the queue"""
        job_id = str(uuid.uuid4())

        # Read files
        song_content = await song_file.read()
        voice_content = await voice_sample.read()

        # Validate file sizes (100MB limit per file)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        if len(song_content) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={"error": f"Song file too large. Max {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"}
            )
        if len(voice_content) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={"error": f"Voice file too large. Max {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"}
            )

        # Initialize job
        self.jobs[job_id] = {
            "status": "pending",
            "progress": "Queued for processing",
            "result": None,
            "error": None,
            "song_filename": song_file.filename,
            "voice_filename": voice_sample.filename,
            "song_content": song_content,
            "voice_content": voice_content
        }

        # Add to queue
        self.job_queue.put(job_id)
        logger.info(f"Cover job {job_id} submitted")

        return job_id

    def get_job_status(self, job_id: str):
        """Get job status"""
        if job_id not in self.jobs:
            return None

        job = self.jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job.get("progress"),
            error=job.get("error")
        )

    def get_job_result(self, job_id: str):
        """Get job result file path"""
        if job_id not in self.jobs or self.jobs[job_id]["status"] != "completed":
            return None

        return self.jobs[job_id]["result"]

    def _worker_loop(self):
        """Background worker thread that processes jobs sequentially"""
        while self.worker_running:
            try:
                # Wait for next job (timeout to allow graceful shutdown)
                job_id = self.job_queue.get(timeout=1)

                # Process job
                logger.info(f"Processing job {job_id}")
                self.jobs[job_id]["status"] = "processing"
                self.jobs[job_id]["progress"] = "Starting processing"

                try:
                    self._process_cover_job(job_id)
                except Exception as e:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["error"] = str(e)
                    logger.error(f"Job {job_id} failed: {str(e)}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

    def _process_cover_job(self, job_id: str):
        """Process a cover generation job (runs in background thread)"""
        job = self.jobs[job_id]

        # Save files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(job["song_filename"])[1]) as song_tmp:
            song_tmp.write(job["song_content"])
            song_path = song_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(job["voice_filename"])[1]) as voice_tmp:
            voice_tmp.write(job["voice_content"])
            voice_path = voice_tmp.name

        try:
            # Step 1: Separate vocals
            job["progress"] = "Separating vocals from instrumentals..."
            separation_result = self._separate_vocals_sync(song_path)

            # Step 2: Download vocals and apply voice conversion
            job["progress"] = "Converting vocals to target voice..."

            vocals_file_output = None
            if isinstance(separation_result, dict):
                vocals_file_output = separation_result.get("vocals")

            if not vocals_file_output:
                raise Exception("Could not extract vocals from separation")

            vocals_url = str(vocals_file_output.url) if hasattr(vocals_file_output, 'url') else str(vocals_file_output)

            # Download vocals
            vocals_response = req.get(vocals_url)
            content_type = vocals_response.headers.get('content-type', '')
            suffix = ".mp3" if 'mp3' in content_type or not content_type else (".wav" if 'wav' in content_type else ".mp3")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as vocals_tmp:
                vocals_tmp.write(vocals_response.content)
                vocals_path = vocals_tmp.name

            # Apply voice conversion
            with open(vocals_path, "rb") as vocals_f, open(voice_path, "rb") as voice_f:
                output = self.client.run(
                    self.voice_model,
                    input={
                        "source_audio": vocals_f,
                        "reference_audio": voice_f
                    }
                )

            # Clean up temp vocals file immediately after voice conversion
            try:
                os.unlink(vocals_path)
            except:
                pass

            # Extract converted vocals URL
            converted_vocals_url = None
            if isinstance(output, str):
                converted_vocals_url = output
            elif hasattr(output, 'url'):
                converted_vocals_url = str(output.url)
            elif isinstance(output, dict):
                if "audio" in output:
                    converted_vocals_url = output["audio"]
                elif "url" in output:
                    converted_vocals_url = output["url"]

            if not converted_vocals_url:
                raise Exception("Failed to extract converted vocals")

            # Step 3: Mix audio
            job["progress"] = "Mixing audio tracks..."
            mixed_audio_path = self._mix_audio_tracks_sync(
                converted_vocals_url,
                separation_result["drums"],
                separation_result["bass"],
                separation_result["other"]
            )

            job["result"] = mixed_audio_path
            job["status"] = "completed"
            job["progress"] = "Completed"
            logger.info(f"Job {job_id} completed: {mixed_audio_path}")

        finally:
            # Clean up temp files
            for path in [song_path, voice_path]:
                try:
                    os.unlink(path)
                except:
                    pass

            # Aggressive garbage collection
            gc.collect()

    def _separate_vocals_sync(self, audio_path: str):
        """Separate vocals from instrumentals (sync version for worker thread)"""
        with open(audio_path, "rb") as audio_file:
            output = self.client.run(
                app_config.vocal_isolation_model,
                input={"audio": audio_file}
            )
        return output

    def _mix_audio_tracks_sync(self, converted_vocals_url: str, drums_url, bass_url, other_url):
        """Mix audio tracks (sync version for worker thread)"""
        # Download all tracks
        vocals_response = req.get(converted_vocals_url)
        drums_response = req.get(str(drums_url.url)) if hasattr(drums_url, 'url') else req.get(str(drums_url))
        bass_response = req.get(str(bass_url.url)) if hasattr(bass_url, 'url') else req.get(str(bass_url))
        other_response = req.get(str(other_url.url)) if hasattr(other_url, 'url') else req.get(str(other_url))

        # Load audio tracks
        try:
            vocals_audio = AudioSegment.from_file(io.BytesIO(vocals_response.content), format="mp3")
        except:
            vocals_audio = AudioSegment.from_file(io.BytesIO(vocals_response.content), format="wav")

        try:
            drums_audio = AudioSegment.from_file(io.BytesIO(drums_response.content), format="mp3")
        except:
            drums_audio = AudioSegment.from_file(io.BytesIO(drums_response.content), format="wav")

        try:
            bass_audio = AudioSegment.from_file(io.BytesIO(bass_response.content), format="mp3")
        except:
            bass_audio = AudioSegment.from_file(io.BytesIO(bass_response.content), format="wav")

        try:
            other_audio = AudioSegment.from_file(io.BytesIO(other_response.content), format="mp3")
        except:
            other_audio = AudioSegment.from_file(io.BytesIO(other_response.content), format="wav")

        # Align audio lengths
        max_length = max(len(vocals_audio), len(drums_audio), len(bass_audio), len(other_audio))

        if len(vocals_audio) < max_length:
            vocals_audio = vocals_audio + AudioSegment.silent(duration=(max_length - len(vocals_audio)))
        if len(drums_audio) < max_length:
            drums_audio = drums_audio + AudioSegment.silent(duration=(max_length - len(drums_audio)))
        if len(bass_audio) < max_length:
            bass_audio = bass_audio + AudioSegment.silent(duration=(max_length - len(bass_audio)))
        if len(other_audio) < max_length:
            other_audio = other_audio + AudioSegment.silent(duration=(max_length - len(other_audio)))

        # Mix tracks
        mixed = vocals_audio.overlay(drums_audio).overlay(bass_audio).overlay(other_audio)

        # Clear audio objects from memory
        del vocals_audio, drums_audio, bass_audio, other_audio
        gc.collect()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mixed_tmp:
            mixed.export(mixed_tmp.name, format="mp3", bitrate="192k")
            return mixed_tmp.name

    async def generate_song(self, request: GenerateSongRequest):
        """Generate an original song from prompt and optional lyrics using ElevenLabs Music model"""
        try:
            logger.info(f"Generating song with prompt: {request.prompt}, lyrics: {request.lyrics}")

            # Combine prompt and lyrics for better music generation
            full_prompt = request.prompt
            if request.lyrics:
                # Include vocals in the prompt and add lyrics
                full_prompt = f"{request.prompt} with vocals and singing. Lyrics: {request.lyrics}"
            else:
                # If no lyrics provided, still request vocals
                full_prompt = f"{request.prompt} with vocals and singing"

            # Prepare input for elevenlabs/music model
            input_data = {
                "prompt": full_prompt,
                "music_length_ms": 30000,  # 90 seconds (1.5 minutes)
                "output_format": "mp3_high_quality",  # High quality MP3
                "force_instrumental": False  # Allow vocals in the output
            }

            # Call Replicate API
            output = self.client.run(
                self.song_model,
                input=input_data
            )

            logger.info("Song generated successfully")

            # Handle different output types from Replicate
            audio_url = None

            if isinstance(output, str):
                # If it's a URL string
                audio_url = output
            elif hasattr(output, 'url'):
                # If it's a FileOutput object with url attribute
                audio_url = str(output.url)
            elif isinstance(output, dict):
                # If it's a dict, try to find the audio URL
                if "audio" in output:
                    audio_url = output["audio"]
                elif "url" in output:
                    audio_url = output["url"]

            if audio_url:
                return await self._download_and_return_file(audio_url, "generated_song.mp3")
            else:
                logger.error(f"Could not extract audio URL from output: {output}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to extract audio from generation"}
                )

        except Exception as e:
            logger.error(f"Error generating song: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": str(e)}
            )

    async def get_billing(self) -> BillingResponse:
        """Get account information (Replicate doesn't have a billing API, return placeholder)"""
        try:
            logger.info("Fetching account information")

            return BillingResponse(
                status="success",
                data={
                    "provider": "Replicate",
                    "billing_model": "Pay-per-use",
                    "estimated_cost_per_song": "$0.03-0.08",
                    "estimated_cost_per_cover": "$0.01-0.03",
                    "note": "Visit https://replicate.com/account to manage billing"
                }
            )

        except Exception as e:
            logger.error(f"Error fetching billing: {str(e)}")
            return BillingResponse(
                status="error",
                error=str(e)
            )

    async def _download_and_return_file(self, url: str, filename: str):
        """Download file from URL and return as FileResponse"""
        import requests

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Detect the actual content type from response headers
            content_type = response.headers.get('content-type', 'audio/wav')

            # Determine file extension based on content type
            if 'audio/mpeg' in content_type or 'mp3' in content_type:
                file_ext = '.mp3'
                media_type = 'audio/mpeg'
            elif 'audio/wav' in content_type or 'wav' in content_type:
                file_ext = '.wav'
                media_type = 'audio/wav'
            elif 'audio/ogg' in content_type or 'ogg' in content_type:
                file_ext = '.ogg'
                media_type = 'audio/ogg'
            else:
                # Default to WAV
                file_ext = '.wav'
                media_type = 'audio/wav'

            # Save to temporary file with correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Create filename with correct extension
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}{file_ext}"

            return FileResponse(
                path=tmp_path,
                media_type=media_type,
                filename=output_filename
            )

        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to download audio: {str(e)}"}
            )
