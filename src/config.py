import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

load_dotenv()

class AppConfig:
    port = int(os.getenv("PORT", 8000))
    replicate_api_key = os.getenv("REPLICATE_API_KEY", "")

    # Replicate models - Music generation
    # Using elevenlabs/music for original song generation (text to music)
    song_generation_model = "elevenlabs/music"
    # Using zsxkib/realistic-voice-cloning for voice cloning
    voice_cloning_model = "zsxkib/realistic-voice-cloning"

app_config = AppConfig()


# Request Models
class GenerateSongRequest(BaseModel):
    prompt: str
    lyrics: Optional[str] = None


# Response Models
class GenerateSongResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None


class GenerateCoverResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None


class BillingResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    error: Optional[str] = None
