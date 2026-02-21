import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from ..config import GenerateSongRequest, GenerateSongResponse, GenerateCoverResponse, BillingResponse, JobStatusResponse
from ..services.mureka_service import ReplicateService

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
    """Submit a cover generation job and get job_id back immediately

    Returns: {job_id: str}
    """
    job_id = await service.submit_cover_job(song_file, voice_sample)
    return {"job_id": job_id, "status": "pending"}


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a cover generation job"""
    status = service.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """Get the result of a completed cover generation job"""
    result = service.get_job_result(job_id)
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
    return await service.get_billing()
