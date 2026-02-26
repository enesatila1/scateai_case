import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from ..config import GenerateSongRequest, BillingResponse, JobStatusResponse
from ..services.generation_service import GenerationService
from ..services.vocal_service import VocalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/music", tags=["MUSIC"])
generation_service = GenerationService()
vocal_service = VocalService()


@router.post("/generate-song")
async def generate_song(request: GenerateSongRequest):
    """Generate an original song from prompt and optional lyrics

    Returns: Audio file (MP3)
    """
    return await generation_service.generate_song(request)


@router.post("/generate-cover")
async def generate_cover(
    song_file: UploadFile = File(..., description="Audio file of the song to cover (MP3, WAV)"),
    voice_sample: UploadFile = File(..., description="Voice sample for cloning (10 seconds minimum)")
):
    """Submit a cover generation job and get job_id back immediately

    Returns: {job_id: str}
    """
    job_id = await vocal_service.submit_cover_job(song_file, voice_sample)
    if isinstance(job_id, tuple):
        raise HTTPException(status_code=job_id[1], detail=job_id[0].get("error"))
    return {"job_id": job_id, "status": "pending"}


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a cover generation job"""
    status = vocal_service.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """Get the result of a completed cover generation job"""
    result = vocal_service.get_job_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found or not completed")
    return FileResponse(
        path=result,
        media_type="audio/mpeg",
        filename="generated_cover.mp3"
    )


@router.get("/billing", response_model=BillingResponse)
async def get_billing():
    """Get account billing info and quota usage"""
    return await generation_service.get_billing()
