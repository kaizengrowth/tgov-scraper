from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.models.topic import Topic, TopicCreate
from src.services.topic_service import TopicService

router = APIRouter()


@router.post("/topics", response_model=Topic, status_code=status.HTTP_201_CREATED)
async def create_topic(
    topic: TopicCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new topic subscription.
    """
    service = TopicService(db)
    return await service.create_topic(topic)


@router.get("/topics/user/{user_id}", response_model=List[Topic])
async def get_user_topics(
    user_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all topics for a specific user.
    """
    service = TopicService(db)
    return await service.get_user_topics(user_id)


@router.delete("/topics/{topic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_topic(
    topic_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a topic subscription.
    """
    service = TopicService(db)
    success = await service.delete_topic(topic_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic with ID {topic_id} not found"
        )
    return None 