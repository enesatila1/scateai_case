import requests
from typing import Optional, Dict, Any
from config import app_config

class MurekaService:
    """Service for interacting with Mureka API"""

    def __init__(self):
        self.base_url = app_config.mureka_api_url
        self.api_key = app_config.api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_song(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        model: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate an original song from prompt and optional lyrics

        Args:
            prompt: Text describing mood, genre, theme, and style
            lyrics: Optional lyrics for the song
            model: Model to use (default: auto)

        Returns:
            API response with generated song details
        """
        payload = {
            "prompt": prompt,
            "model": model
        }

        if lyrics:
            payload["lyrics"] = lyrics

        try:
            response = requests.post(
                f"{self.base_url}/song/generate",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": response.status_code}

    def generate_cover_with_voice(
        self,
        song_url: str,
        voice_sample_url: str,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a cover of a song using a specific voice identity

        Args:
            song_url: URL or reference to the song to cover
            voice_sample_url: URL to the voice sample to use
            voice_id: Optional pre-registered voice ID

        Returns:
            API response with generated cover details
        """
        payload = {
            "reference_song": song_url,
            "voice_sample": voice_sample_url
        }

        if voice_id:
            payload["voice_id"] = voice_id

        try:
            response = requests.post(
                f"{self.base_url}/song/generate",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": response.status_code}

    def get_account_billing(self) -> Dict[str, Any]:
        """
        Get account billing information and quota usage

        Returns:
            API response with billing and quota details
        """
        try:
            response = requests.get(
                f"{self.base_url}/account/billing",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": response.status_code}
