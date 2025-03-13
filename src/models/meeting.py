"""
Pydantic models for meeting data
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, HttpUrl


class Meeting(BaseModel):
    """
    Model representing a government meeting
    """

    id: UUID = Field(default_factory=uuid4)
    meeting: str = Field(description="Name of the meeting")
    date: datetime = Field(description="Date and time of the meeting")
    duration: str = Field(description="Duration of the meeting")
    agenda: Optional[HttpUrl] = Field(None, description="URL to the meeting agenda")
    video: Optional[HttpUrl] = Field(None, description="URL to the meeting video")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

    def __str__(self) -> str:
        """String representation of the meeting"""
        return f"{self.meeting} - {self.date} ({self.duration})"
