import logging
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from config import GenerateSongRequest, GenerateSongResponse, GenerateCoverResponse, BillingResponse
from services.mureka_service import ReplicateService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/music", tags=["MUSIC"])
service = ReplicateService()


@router.post("/generate-song")
async def generate_song(request: GenerateSongRequest):
    """Generate an original song from prompt and optional lyrics

    Returns: Audio file (MP3)
    """
    return await service.generate_song(request)


@router.post("/generate-cover")
async def generate_cover(
    song_file: UploadFile = File(..., description="Audio file of the song to cover (MP3, WAV)"),
    voice_sample: UploadFile = File(..., description="Voice sample for cloning (10 seconds minimum)")
):
    """Generate a cover using voice cloning from uploaded files

    Returns: Audio file with cloned voice (MP3)
    """
    return await service.generate_cover(song_file, voice_sample)


@router.get("/billing", response_model=BillingResponse)
async def get_billing():
    """Get account billing info and quota usage"""
    return await service.get_billing()
