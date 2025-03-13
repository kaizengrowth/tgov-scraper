from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class TranscriptSegment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    meeting_id: UUID = Field(description="ID of the meeting")
    speaker: str = Field(description="Speaker identifier from diarization")
    start_time: float = Field(description="Start time in seconds from beginning of recording")
    end_time: float = Field(description="End time in seconds from beginning of recording")
    text: str = Field(description="Transcribed text segment")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True
