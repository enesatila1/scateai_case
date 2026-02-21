import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.mureka_service import MurekaService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/music", tags=["MUSIC"])
mureka_service = MurekaService()

class GenerateSongRequest(BaseModel):
    prompt: str
    lyrics: Optional[str] = None
    model: str = "auto"

class GenerateCoverRequest(BaseModel):
    song_url: str
    voice_sample_url: str
    voice_id: Optional[str] = None

@router.post("/generate-song")
async def generate_original_song(request: GenerateSongRequest):
    """
    Task 1: Generate an original song from text prompt and optional lyrics

    Inputs:
    - prompt: Text describing mood, genre, theme, style (e.g., "upbeat pop, energetic, female vocal")
    - lyrics: Optional lyrics for the song

    Outputs:
    - Audio file (instrumental or with AI-generated vocals)
    - Metadata about the generated song
    """
    logger.info(f"Generating song with prompt: {request.prompt}")

    result = mureka_service.generate_song(
        prompt=request.prompt,
        lyrics=request.lyrics,
        model=request.model
    )

    if "error" in result:
        logger.error(f"Error generating song: {result}")
        raise HTTPException(status_code=400, detail=result.get("error"))

    logger.info("Song generated successfully")
    return result

@router.post("/generate-cover")
async def generate_cover_with_voice(request: GenerateCoverRequest):
    """
    Task 2: Generate a cover of a song using a specific voice identity

    Inputs:
    - song_url: Reference to the song to cover
    - voice_sample_url: URL to voice sample for voice cloning
    - voice_id: Optional pre-registered voice ID

    Outputs:
    - Vocal performance using the uploaded/cloned voice
    - Preserves melody and timing
    - Similar to source voice
    """
    logger.info(f"Generating cover for song: {request.song_url}")

    result = mureka_service.generate_cover_with_voice(
        song_url=request.song_url,
        voice_sample_url=request.voice_sample_url,
        voice_id=request.voice_id
    )

    if "error" in result:
        logger.error(f"Error generating cover: {result}")
        raise HTTPException(status_code=400, detail=result.get("error"))

    logger.info("Cover generated successfully")
    return result

@router.get("/billing")
async def get_billing_info():
    """
    Get current account billing info and quota usage

    Returns:
    - Current credit balance
    - Usage information
    - Plan details
    """
    logger.info("Fetching billing information")

    result = mureka_service.get_account_billing()

    if "error" in result:
        logger.error(f"Error fetching billing: {result}")
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result
