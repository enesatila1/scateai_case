import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/public",tags=["PUBLIC"])

@router.get("/health")
async def health_check():
    return {"status": "ok"}