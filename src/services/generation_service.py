import logging
import replicate
import tempfile
import os
from ..config import app_config, GenerateSongRequest, BillingResponse
from fastapi.responses import FileResponse, JSONResponse
import requests

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for original song generation"""

    def __init__(self):
        os.environ["REPLICATE_API_TOKEN"] = app_config.replicate_api_key
        os.environ["REPLICATE_TIMEOUT"] = "600"

        self.client = replicate.Client(api_token=app_config.replicate_api_key)
        self.song_model = app_config.song_generation_model

    async def generate_song(self, request: GenerateSongRequest):
        """Generate an original song from prompt and optional lyrics"""
        try:
            full_prompt = request.prompt
            if request.lyrics:
                full_prompt = f"{request.prompt} with vocals and singing. Lyrics: {request.lyrics}"
            else:
                full_prompt = f"{request.prompt} with vocals and singing"

            input_data = {
                "prompt": full_prompt,
                "music_length_ms": 30000,
                "output_format": "mp3_high_quality",
                "force_instrumental": False
            }

            output = self.client.run(self.song_model, input=input_data)

            audio_url = self._extract_url(output)
            if not audio_url:
                raise ValueError("Could not extract audio URL from output")

            return await self._download_and_return_file(audio_url, "generated_song.mp3")

        except Exception as e:
            logger.error(f"Song generation error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": str(e)}
            )

    def _extract_url(self, output):
        """Extract audio URL from various Replicate response formats"""
        if isinstance(output, str):
            return output
        if hasattr(output, 'url'):
            return str(output.url)
        if isinstance(output, dict):
            return output.get("audio") or output.get("url")
        return None

    async def _download_and_return_file(self, url: str, filename: str):
        """Download file from URL and return as FileResponse"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()

            if 'mp3' in content_type or 'mpeg' in content_type:
                file_ext, media_type = '.mp3', 'audio/mpeg'
            elif 'wav' in content_type:
                file_ext, media_type = '.wav', 'audio/wav'
            elif 'ogg' in content_type:
                file_ext, media_type = '.ogg', 'audio/ogg'
            else:
                file_ext, media_type = '.mp3', 'audio/mpeg'

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}{file_ext}"

            return FileResponse(
                path=tmp_path,
                media_type=media_type,
                filename=output_filename
            )

        except requests.RequestException as e:
            logger.error(f"File download error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to download audio: {str(e)}"}
            )

    async def get_billing(self) -> BillingResponse:
        """Get billing information"""
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
