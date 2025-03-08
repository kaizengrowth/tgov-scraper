"""
Pydantic models for meeting data
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class Meeting(BaseModel):
    """
    Model representing a government meeting
    """

    meeting: str = Field(description="Name of the meeting")
    date: str = Field(description="Date and time of the meeting")
    duration: str = Field(description="Duration of the meeting")
    agenda: Optional[HttpUrl] = Field(None, description="URL to the meeting agenda")
    video: Optional[HttpUrl] = Field(None, description="URL to the meeting video")

    def __str__(self) -> str:
        """String representation of the meeting"""
        return f"{self.meeting} - {self.date} ({self.duration})"
