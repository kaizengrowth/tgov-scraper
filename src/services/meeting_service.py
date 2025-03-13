from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.meeting import Meeting
from src.services.scraper_service import ScraperService


class MeetingService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.scraper = ScraperService()

    async def get_meetings(self, skip: int = 0, limit: int = 100) -> List[Meeting]:
        # Use the existing scraper to get meeting data
        meetings = await self.scraper.fetch_meetings()
        
        # Apply pagination
        return meetings[skip:skip+limit]

    async def get_meeting(self, meeting_id: UUID) -> Optional[Meeting]:
        # For now, get all meetings and find by ID
        # In a real implementation, this would query the database
        meetings = await self.get_meetings(limit=1000)
        for meeting in meetings:
            if meeting.id == meeting_id:
                return meeting
        return None

    async def create_meeting(self, meeting: Meeting) -> Meeting:
        # Placeholder implementation
        return meeting

    async def update_meeting(self, meeting_id: UUID, meeting: Meeting) -> Optional[Meeting]:
        # Placeholder implementation
        return meeting

    async def delete_meeting(self, meeting_id: UUID) -> bool:
        # Placeholder implementation
        return True

    async def search_meetings(self, query: str) -> List[Meeting]:
        # Get all meetings
        all_meetings = await self.get_meetings(limit=1000)
        
        # Filter meetings containing the query string
        return [
            meeting for meeting in all_meetings 
            if query.lower() in meeting.meeting.lower()
        ]

    async def get_recent_meetings(self, limit: int = 10) -> List[Meeting]:
        # Get all meetings
        all_meetings = await self.get_meetings(limit=1000)
        
        # Sort by date (most recent first) and return limited number
        return sorted(
            all_meetings, 
            key=lambda m: m.date if isinstance(m.date, datetime) else datetime.min, 
            reverse=True
        )[:limit]

    async def get_upcoming_meetings(self, limit: int = 10) -> List[Meeting]:
        # Get all meetings
        all_meetings = await self.get_meetings(limit=1000)
        
        # Filter for upcoming meetings and sort by date
        now = datetime.utcnow()
        
        # We need to handle possible string dates since the original data might have string dates
        upcoming = []
        for m in all_meetings:
            if isinstance(m.date, datetime) and m.date > now:
                upcoming.append(m)
        
        return sorted(
            upcoming, 
            key=lambda m: m.date
        )[:limit]
