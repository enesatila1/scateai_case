import logging
import replicate
import tempfile
import os
import io
import uuid
import threading
import queue
import gc
from ..config import app_config, JobStatusResponse
from pydub import AudioSegment
import requests

logger = logging.getLogger(__name__)


class VocalService:
    """Service for cover generation with voice cloning and vocal separation"""

    def __init__(self):
        os.environ["REPLICATE_API_TOKEN"] = app_config.replicate_api_key
        os.environ["REPLICATE_TIMEOUT"] = "600"

        self.client = replicate.Client(api_token=app_config.replicate_api_key)
        self.voice_model = app_config.voice_cloning_model
        self.vocal_isolation_model = app_config.vocal_isolation_model

        # Job queue system
        self.jobs = {}
        self.job_queue = queue.Queue()
        self.worker_running = True

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    async def submit_cover_job(self, song_file, voice_sample):
        """Submit a cover generation job to the queue"""
        job_id = str(uuid.uuid4())

        song_content = await song_file.read()
        voice_content = await voice_sample.read()

        # Validate file sizes
        MAX_FILE_SIZE = 100 * 1024 * 1024
        for content, name in [(song_content, "Song"), (voice_content, "Voice")]:
            if len(content) > MAX_FILE_SIZE:
                return {
                    "error": f"{name} file too large. Max 100MB"
                }, 400

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
        """Background worker thread - processes jobs sequentially"""
        while self.worker_running:
            try:
                job_id = self.job_queue.get(timeout=1)

                logger.info(f"Processing job {job_id}")
                self.jobs[job_id]["status"] = "processing"

                try:
                    self._process_cover_job(job_id)
                except Exception as e:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["error"] = str(e)
                    logger.error(f"Job {job_id} failed: {str(e)}")

            except queue.Empty:
                continue

    def _process_cover_job(self, job_id: str):
        """Process cover generation job (runs in background thread)"""
        job = self.jobs[job_id]

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

            # Step 2: Convert vocals to target voice
            job["progress"] = "Converting vocals to target voice..."

            vocals_file_output = separation_result.get("vocals") if isinstance(separation_result, dict) else None
            if not vocals_file_output:
                raise ValueError("Could not extract vocals from separation")

            vocals_url = str(vocals_file_output.url) if hasattr(vocals_file_output, 'url') else str(vocals_file_output)

            # Download separated vocals
            vocals_response = requests.get(vocals_url)
            content_type = vocals_response.headers.get('content-type', '').lower()
            suffix = ".mp3" if 'mp3' in content_type else ".wav"

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

            # Cleanup vocals temp file
            try:
                os.unlink(vocals_path)
            except:
                pass

            # Extract converted vocals URL
            converted_vocals_url = self._extract_url(output)
            if not converted_vocals_url:
                raise ValueError("Failed to extract converted vocals")

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
            logger.info(f"Job {job_id} completed")

        finally:
            for path in [song_path, voice_path]:
                try:
                    os.unlink(path)
                except:
                    pass

            gc.collect()

    def _separate_vocals_sync(self, audio_path: str):
        """Separate vocals from instrumentals using Demucs"""
        with open(audio_path, "rb") as audio_file:
            output = self.client.run(
                self.vocal_isolation_model,
                input={"audio": audio_file}
            )
        return output

    def _mix_audio_tracks_sync(self, converted_vocals_url: str, drums_url, bass_url, other_url):
        """Mix converted vocals with instrumental tracks"""
        # Download all tracks
        vocals_response = requests.get(converted_vocals_url)
        drums_response = requests.get(str(drums_url.url) if hasattr(drums_url, 'url') else drums_url)
        bass_response = requests.get(str(bass_url.url) if hasattr(bass_url, 'url') else bass_url)
        other_response = requests.get(str(other_url.url) if hasattr(other_url, 'url') else other_url)

        # Load audio tracks with format fallback
        def load_audio(response):
            try:
                return AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
            except:
                return AudioSegment.from_file(io.BytesIO(response.content), format="wav")

        vocals_audio = load_audio(vocals_response)
        drums_audio = load_audio(drums_response)
        bass_audio = load_audio(bass_response)
        other_audio = load_audio(other_response)

        # Align audio lengths
        max_length = max(len(vocals_audio), len(drums_audio), len(bass_audio), len(other_audio))

        audios = [vocals_audio, drums_audio, bass_audio, other_audio]
        for i, audio in enumerate(audios):
            if len(audio) < max_length:
                silence = AudioSegment.silent(duration=(max_length - len(audio)))
                audios[i] = audio + silence

        vocals_audio, drums_audio, bass_audio, other_audio = audios

        # Mix tracks
        mixed = vocals_audio.overlay(drums_audio).overlay(bass_audio).overlay(other_audio)

        # Clean memory
        del vocals_audio, drums_audio, bass_audio, other_audio
        gc.collect()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mixed_tmp:
            mixed.export(mixed_tmp.name, format="mp3", bitrate="192k")
            return mixed_tmp.name

    def _extract_url(self, output):
        """Extract audio URL from various Replicate response formats"""
        if isinstance(output, str):
            return output
        if hasattr(output, 'url'):
            return str(output.url)
        if isinstance(output, dict):
            return output.get("audio") or output.get("url")
        return None
