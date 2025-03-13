from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.models.transcription import TranscriptSegment
from src.services.transcription_service import TranscriptionService

router = APIRouter()


@router.get("/transcriptions/meeting/{meeting_id}", response_model=List[TranscriptSegment])
async def get_meeting_transcription(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get transcription segments for a specific meeting.
    """
    service = TranscriptionService(db)
    return await service.get_meeting_transcription(meeting_id)


@router.get("/transcriptions/search", response_model=List[TranscriptSegment])
async def search_transcriptions(
    query: str = Query(..., description="Search query string"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search transcriptions by keyword or topic.
    """
    service = TranscriptionService(db)
    return await service.search_transcriptions(query) 