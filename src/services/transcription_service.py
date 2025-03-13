from typing import List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.transcription import TranscriptSegment

class TranscriptionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_meeting_transcription(self, meeting_id: UUID) -> List[TranscriptSegment]:
        # Placeholder implementation
        return []

    async def search_transcriptions(self, query: str) -> List[TranscriptSegment]:
        # Placeholder implementation
        return []
