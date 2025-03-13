from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.models.meeting import Meeting
from src.services.meeting_service import MeetingService

router = APIRouter()


@router.get("/meetings", response_model=List[Meeting])
async def get_meetings(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a list of all meetings with pagination.
    """
    service = MeetingService(db)
    return await service.get_meetings(skip=skip, limit=limit)


@router.get("/meetings/{meeting_id}", response_model=Meeting)
async def get_meeting(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get details for a specific meeting by ID.
    """
    service = MeetingService(db)
    meeting = await service.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID {meeting_id} not found"
        )
    return meeting


@router.post("/meetings", response_model=Meeting, status_code=status.HTTP_201_CREATED)
async def create_meeting(
    meeting: Meeting,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new meeting record.
    """
    service = MeetingService(db)
    return await service.create_meeting(meeting)


@router.put("/meetings/{meeting_id}", response_model=Meeting)
async def update_meeting(
    meeting_id: UUID,
    meeting: Meeting,
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing meeting record.
    """
    service = MeetingService(db)
    updated_meeting = await service.update_meeting(meeting_id, meeting)
    if not updated_meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID {meeting_id} not found"
        )
    return updated_meeting


@router.delete("/meetings/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting(
    meeting_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a meeting record.
    """
    service = MeetingService(db)
    success = await service.delete_meeting(meeting_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID {meeting_id} not found"
        )
    return None


@router.get("/meetings/search", response_model=List[Meeting])
async def search_meetings(
    query: str = Query(..., description="Search query string"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search meetings by name or other criteria.
    """
    service = MeetingService(db)
    return await service.search_meetings(query)


@router.get("/meetings/recent", response_model=List[Meeting])
async def get_recent_meetings(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get most recent meetings.
    """
    service = MeetingService(db)
    return await service.get_recent_meetings(limit)


@router.get("/meetings/upcoming", response_model=List[Meeting])
async def get_upcoming_meetings(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get upcoming meetings.
    """
    service = MeetingService(db)
    return await service.get_upcoming_meetings(limit) 