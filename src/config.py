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
    # Using jagilley/free-vc for zero-shot voice conversion (direct voice change)
    voice_cloning_model = "jagilley/free-vc:e4f2ff8a1d3779a2411e119dfad7d451d5f3314a8cd7003a88f88ce4c3b18d95"
    # Using cjwbw/demucs for vocal isolation (separate vocals from instruments)
    vocal_isolation_model = "cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953"

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
