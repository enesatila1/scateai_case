import logging
import replicate
import tempfile
import os
import io
from config import app_config, GenerateSongRequest, BillingResponse
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

    async def _separate_vocals(self, audio_path: str):
        """Separate vocals from instrumentals using Demucs"""
        try:
            print(f"🎼 Demucs'a gönderiliyor: {audio_path}")

            # Open file and keep it open during the request
            with open(audio_path, "rb") as audio_file:
                print(f"📤 Dosya açıldı, Demucs'a gönderiliyor...")

                output = self.client.run(
                    app_config.vocal_isolation_model,
                    input={
                        "audio": audio_file
                    }
                )

            print(f"✅ Vocal separation tamamlandı")
            print(f"📦 Raw output: {output}")
            return output  # Returns dict with vocals_url, drums_url, bass_url, other_url

        except Exception as e:
            print(f"❌ Error separating vocals: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def generate_cover(self, song_file, voice_sample):
        """Generate a cover using voice cloning with realistic-voice-cloning model"""

        # Read file contents
        song_content = await song_file.read()
        voice_content = await voice_sample.read()

        logger.info(f"Generating cover: song={song_file.filename}, voice={voice_sample.filename}")

        # Save files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(song_file.filename)[1]) as song_tmp:
            song_tmp.write(song_content)
            song_path = song_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(voice_sample.filename)[1]) as voice_tmp:
            voice_tmp.write(voice_content)
            voice_path = voice_tmp.name

        try:
            # Step 1: Separate vocals from instrumentals using Demucs
            print("🎵 Step 1: Separating vocals from instrumentals...")
            separation_result = await self._separate_vocals(song_path)

            # Debug: Print the actual output structure
            print(f"📊 Demucs output type: {type(separation_result)}")
            print(f"📊 Demucs output: {separation_result}")

            # separation_result contains URLs to vocals, drums, bass, other
            # We'll apply voice conversion only to vocals

            # Step 2: Download separated vocals and apply voice conversion
            print("🎵 Step 2: Converting vocals to target voice...")

            # Get vocals FileOutput object from separation result
            vocals_file_output = None

            if isinstance(separation_result, dict):
                vocals_file_output = separation_result.get("vocals")
                print(f"📊 Dict keys: {separation_result.keys()}")
                print(f"📊 Vocals object type: {type(vocals_file_output)}")

            if not vocals_file_output:
                print(f"⚠️ Could not extract vocals from: {separation_result}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not extract vocals from separation"}
                )

            # FileOutput object has a .url attribute
            vocals_url = str(vocals_file_output.url) if hasattr(vocals_file_output, 'url') else str(vocals_file_output)
            print(f"🔗 Vocals URL: {vocals_url}")

            # Download vocals temporarily
            import requests as req
            vocals_response = req.get(vocals_url)
            # Detect format from content-type or default to mp3
            content_type = vocals_response.headers.get('content-type', '')
            suffix = ".mp3" if 'mp3' in content_type or not content_type else (".wav" if 'wav' in content_type else ".mp3")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as vocals_tmp:
                vocals_tmp.write(vocals_response.content)
                vocals_path = vocals_tmp.name
                print(f"💾 Vocals saved to: {vocals_path}")

            # Apply voice conversion to vocals only
            print("🎤 Applying voice conversion to vocals...")

            # Keep file handles open during the request
            with open(vocals_path, "rb") as vocals_f, open(voice_path, "rb") as voice_f:
                print(f"📤 Voice files opened, sending to free-vc...")

                output = self.client.run(
                    self.voice_model,
                    input={
                        "source_audio": vocals_f,
                        "reference_audio": voice_f
                    }
                )

            print(f"✅ Voice conversion completed")

            print("✅ Voice conversion completed, extracting URL...")

            # Handle different output types from Replicate
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
                print(f"⚠️ Could not extract converted vocals URL: {output}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to extract converted vocals"}
                )

            print(f"🔗 Converted vocals URL: {converted_vocals_url}")

            # Step 3: Mix converted vocals with instrumentals
            print("🎼 Step 3: Mixing audio tracks...")
            mixed_audio_path = await self._mix_audio_tracks(
                converted_vocals_url,
                separation_result["drums"],
                separation_result["bass"],
                separation_result["other"]
            )

            # Return the mixed audio (it's already a local file)
            print(f"📁 Returning mixed audio: {mixed_audio_path}")
            return FileResponse(
                path=mixed_audio_path,
                media_type="audio/mpeg",
                filename="generated_cover.mp3"
            )

        except Exception as e:
            logger.error(f"Error generating cover: {str(e)}")
            print(f"❌ Error generating cover: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={"error": str(e)}
            )
        finally:
            # Clean up temporary files
            try:
                os.unlink(song_path)
            except:
                pass
            try:
                os.unlink(voice_path)
            except:
                pass

    async def _mix_audio_tracks(self, converted_vocals_url: str, drums_url, bass_url, other_url):
        """Mix converted vocals with instrumental tracks"""
        try:
            print("🎼 Mixing converted vocals with instrumentals...")
            from pydub import AudioSegment
            import requests as req

            # Download all tracks
            print("📥 Downloading tracks...")
            vocals_response = req.get(converted_vocals_url)
            drums_response = req.get(str(drums_url.url)) if hasattr(drums_url, 'url') else req.get(str(drums_url))
            bass_response = req.get(str(bass_url.url)) if hasattr(bass_url, 'url') else req.get(str(bass_url))
            other_response = req.get(str(other_url.url)) if hasattr(other_url, 'url') else req.get(str(other_url))

            # Load audio tracks
            print("🔧 Loading audio tracks...")
            # Try to auto-detect format, fallback to wav if mp3 fails
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

            # Make sure all tracks have same length (pad with silence if needed)
            max_length = max(len(vocals_audio), len(drums_audio), len(bass_audio), len(other_audio))

            if len(vocals_audio) < max_length:
                silence = AudioSegment.silent(duration=(max_length - len(vocals_audio)))
                vocals_audio = vocals_audio + silence
            if len(drums_audio) < max_length:
                silence = AudioSegment.silent(duration=(max_length - len(drums_audio)))
                drums_audio = drums_audio + silence
            if len(bass_audio) < max_length:
                silence = AudioSegment.silent(duration=(max_length - len(bass_audio)))
                bass_audio = bass_audio + silence
            if len(other_audio) < max_length:
                silence = AudioSegment.silent(duration=(max_length - len(other_audio)))
                other_audio = other_audio + silence

            # Mix all tracks
            print("🎵 Mixing tracks...")
            mixed = vocals_audio.overlay(drums_audio).overlay(bass_audio).overlay(other_audio)

            # Save mixed audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mixed_tmp:
                mixed.export(mixed_tmp.name, format="mp3", bitrate="192k")
                print(f"✅ Mixing completed: {mixed_tmp.name}")
                return mixed_tmp.name

        except Exception as e:
            print(f"❌ Error mixing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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
