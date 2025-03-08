#!/usr/bin/env python3
"""
Tests for the Government Access Television Meeting Scraper
"""

import asyncio
import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meetings import fetch_page, parse_meetings, get_meetings
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


@pytest.fixture
def sample_html():
    """Sample HTML fixture with a table of meetings"""
    return """
    <html>
    <body>
        <table class="listingTable">
            <thead>
                <tr>
                    <th>Meeting</th>
                    <th>Date</th>
                    <th>Duration</th>
                    <th>Agenda</th>
                    <th>Video</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>City Council</td>
                    <td>Jan 1, 2023</td>
                    <td>1:30</td>
                    <td><a href="agenda.php?id=123">Agenda</a></td>
                    <td><a href="video.php?id=456">Video</a></td>
                </tr>
                <tr>
                    <td>Planning Commission</td>
                    <td>Jan 2, 2023</td>
                    <td>2:15</td>
                    <td><a href="agenda.php?id=789">Agenda</a></td>
                    <td><a href="video.php?id=012">Video</a></td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """


@pytest.mark.asyncio
async def test_parse_meetings(sample_html):
    """Test that meetings are correctly parsed from HTML"""
    meetings = await parse_meetings(sample_html)

    assert len(meetings) == 2

    assert meetings[0]["meeting"] == "City Council"
    assert meetings[0]["date"] == "Jan 1, 2023"
    assert meetings[0]["duration"] == "1:30"
    assert "agenda.php?id=123" in meetings[0]["agenda"]
    assert "video.php?id=456" in meetings[0]["video"]

    assert meetings[1]["meeting"] == "Planning Commission"
    assert meetings[1]["date"] == "Jan 2, 2023"
    assert meetings[1]["duration"] == "2:15"
    assert "agenda.php?id=789" in meetings[1]["agenda"]
    assert "video.php?id=012" in meetings[1]["video"]


@pytest.mark.asyncio
async def test_parse_real_html(real_html):
    """Test that meetings are correctly parsed from real HTML"""
    meetings = await parse_meetings(real_html)

    # Basic validation
    assert isinstance(meetings, list)
    assert len(meetings) > 0

    # Check that each meeting has the expected fields
    for meeting in meetings:
        assert "meeting" in meeting
        assert "date" in meeting
        assert "duration" in meeting
        # Agenda and video may be None for some meetings


@pytest.mark.asyncio
async def test_fetch_page(real_html):
    """Test that fetch_page correctly fetches HTML content"""
    # Use patch to mock the aiohttp.ClientSession
    with patch("aiohttp.ClientSession") as mock_session_class:
        # Create a mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = real_html

        # Set up the mock session
        mock_session = mock_session_class.return_value
        mock_session.get.return_value.__aenter__.return_value = mock_response

        # Call the function with a new session
        result = await fetch_page("https://test.com", mock_session)

        # Verify the result
        assert result == real_html
        mock_session.get.assert_called_once_with("https://test.com")


@pytest.mark.asyncio
async def test_get_meetings(real_html):
    """Test that get_meetings returns a list of Meeting objects"""
    with patch("src.meetings.fetch_page", return_value=real_html):
        meetings = await get_meetings()

        # Basic validation
        assert isinstance(meetings, list)
        assert len(meetings) > 0

        # Check that each meeting is a Meeting object
        for meeting in meetings:
            assert isinstance(meeting, Meeting)
            assert hasattr(meeting, "meeting")
            assert hasattr(meeting, "date")
            assert hasattr(meeting, "duration")
            assert hasattr(meeting, "agenda")  # May be None
            assert hasattr(meeting, "video")  # May be None


@pytest.mark.asyncio
async def test_integration():
    """
    Integration test that actually fetches data from the website.
    This test is marked as optional and can be skipped with -m "not integration"
    """
    pytest.skip("Skipping integration test by default")

    meetings = await get_meetings()

    # Basic validation
    assert isinstance(meetings, list)
    assert len(meetings) > 0

    # Check that each meeting is a Meeting object
    for meeting in meetings:
        assert isinstance(meeting, Meeting)
        assert meeting.meeting
        assert meeting.date
        assert meeting.duration
        # agenda and video may be None
