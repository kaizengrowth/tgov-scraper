from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    phone: str = Field(description="User's phone number for text notifications")

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class User(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    phone: str = Field(description="User's phone number for text notifications")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True
