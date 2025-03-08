#!/usr/bin/env python3
"""
Tests specifically for video URL extraction from the meeting scraper
"""

import asyncio
import os
from pathlib import Path
import pytest
import json

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meetings import parse_meetings, get_meetings
from src.models.meeting import Meeting


@pytest.fixture
def fixture_path():
    """Path to the fixture file"""
    return Path(__file__).parent / "fixtures" / "tgov_homepage.html"


@pytest.fixture
def real_html(fixture_path):
    """Load the real HTML from the fixture file"""
    with open(fixture_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.asyncio
async def test_video_url_extraction(real_html):
    """
    Test specifically focused on extracting video URLs from the real HTML.
    This test analyzes the extraction process in detail to diagnose issues.
    """
    # Parse meetings from the real HTML
    meetings = await parse_meetings(real_html)

    # Count meetings with and without video links
    total_meetings = len(meetings)
    meetings_with_video = sum(1 for m in meetings if m.get("video") is not None)
    meetings_without_video = sum(1 for m in meetings if m.get("video") is None)

    # Print detailed diagnostics
    print(f"\nTotal meetings found: {total_meetings}")
    print(f"Meetings with video links: {meetings_with_video}")
    print(f"Meetings without video links: {meetings_without_video}")

    # Examine the first few meetings in detail
    print("\nDetailed examination of first 5 meetings:")
    for i, meeting in enumerate(meetings[:5]):
        print(f"\nMeeting {i+1}:")
        print(f"  Title: {meeting.get('meeting')}")
        print(f"  Date: {meeting.get('date')}")
        print(f"  Agenda: {meeting.get('agenda')}")
        print(f"  Video: {meeting.get('video')}")

    # Assert that at least some meetings have video links
    # This will fail if no video links are being extracted
    assert meetings_with_video > 0, "No video links were extracted from the meetings"

    # For a more detailed analysis, let's examine the HTML structure of video cells
    from selectolax.parser import HTMLParser

    parser = HTMLParser(real_html)

    # Find all video cells in the HTML
    video_cells = parser.css("td.listItem a")
    video_links = [link for link in video_cells if "Video" in link.text()]

    print(f"\nFound {len(video_links)} video link elements in the HTML")

    # Examine the first few video links in detail
    print("\nDetailed examination of first 5 video link elements:")
    for i, link in enumerate(video_links[:5]):
        print(f"\nVideo Link {i+1}:")
        print(f"  Text: {link.text()}")
        print(f"  href: {link.attributes.get('href', 'None')}")
        print(f"  onclick: {link.attributes.get('onclick', 'None')}")

    # Assert that we found video links in the HTML
    assert len(video_links) > 0, "No video link elements found in the HTML"

    # Check if the number of video links in HTML matches the number of extracted links
    # This helps identify if we're missing some links during extraction
    print(
        f"\nComparison: {len(video_links)} video links in HTML vs {meetings_with_video} extracted"
    )

    # Return data for further analysis if needed
    return {
        "total_meetings": total_meetings,
        "meetings_with_video": meetings_with_video,
        "meetings_without_video": meetings_without_video,
        "video_links_in_html": len(video_links),
    }


if __name__ == "__main__":
    # This allows running the test directly for debugging
    import asyncio

    fixture_path = Path(__file__).parent / "fixtures" / "tgov_homepage.html"
    with open(fixture_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    result = asyncio.run(test_video_url_extraction(html_content))
    print("\nTest result summary:")
    print(json.dumps(result, indent=2))
