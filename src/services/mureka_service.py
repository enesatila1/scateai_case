import logging
import replicate
import tempfile
import os
from config import app_config, GenerateSongRequest, BillingResponse
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)


class ReplicateService:
    """Service for interacting with Replicate API for music generation"""

    def __init__(self):
        # Set API token for replicate
        os.environ["REPLICATE_API_TOKEN"] = app_config.replicate_api_key
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

    async def generate_cover(self, song_file, voice_sample):
        """Generate a cover using voice cloning with realistic-voice-cloning model"""
        try:
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
                # Call Replicate API for voice cloning
                output = self.client.run(
                    self.voice_model,
                    input={
                        "song_input": open(song_path, "rb"),
                        "rvc_model": "custom",  # Using custom voice sample
                        "pitch_change": 0,
                        "protect": 0.5
                    }
                )

                logger.info("Cover generated successfully")

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
                    return await self._download_and_return_file(audio_url, "generated_cover.mp3")
                else:
                    logger.error(f"Could not extract audio URL from output: {output}")
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to extract audio from generation"}
                    )

            finally:
                # Clean up temporary files
                os.unlink(song_path)
                os.unlink(voice_path)

        except Exception as e:
            logger.error(f"Error generating cover: {str(e)}")
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
