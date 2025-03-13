from typing import List
import sys
from src.meetings import get_meetings
from src.models.meeting import Meeting

class ScraperService:
    """Service to scrape meeting data from TGOV website using existing functionality"""
    
    async def fetch_meetings(self) -> List[Meeting]:
        """
        Fetch meetings from the TGOV website using the existing get_meetings function
        
        Returns:
            List[Meeting]: A list of Meeting objects
        """
        # Use the existing get_meetings function
        meetings = await get_meetings()
        
        # The meetings returned by get_meetings should already be Meeting objects
        # If they're not in the exact format we need, we can convert them here
        
        return meetings 