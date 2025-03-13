from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class TopicCreate(BaseModel):
    name: str = Field(description="Topic keyword or phrase to monitor")
    user_id: UUID = Field(description="ID of the user creating this topic")

class Topic(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Topic keyword or phrase to monitor")
    user_id: UUID = Field(description="ID of the user who created this topic")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True
