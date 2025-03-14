#!/usr/bin/env python3
"""
Tests for the Granicus-specific functionality.
"""

import os
import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.granicus import get_video_player
from src.models.meeting import GranicusPlayerPage


@pytest.fixture
def video_player_fixture_path():
    """Path to the video player fixture file"""
    return Path(__file__).parent / "fixtures" / "video_player.html"


@pytest.fixture
def player_html(video_player_fixture_path):
    """Load the player HTML from the fixture file"""
    with open(video_player_fixture_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.asyncio
@patch("src.granicus.aiohttp.ClientSession.get")
async def test_get_video_player_from_fixture(mock_session, player_html):
    """Test that the video player URLs are correctly extracted from the fixture HTML"""
    # Mock response for initial page fetch
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value=player_html)

    mock_session.return_value.__aenter__.return_value = mock_response

    # Use a dummy URL since we're mocking the fetch
    player_page = await get_video_player("https://example.com/player")

    # Assert that we found the expected URLs
    assert player_page is not None
    assert str(player_page.url) == "https://example.com/player"

    # The specific stream URL from our fixture
    expected_stream_url = "https://archive-stream.granicus.com/OnDemand/_definst_/mp4:archive/tulsa-ok/tulsa-ok_843d30f2-b631-4a16-8018-a2a31930be70.mp4/playlist.m3u8"
    assert str(player_page.stream_url) == expected_stream_url

    # The download URL should be constructed based on the stream URL
    expected_download_url = "http://archive-video.granicus.com/tulsa-ok/tulsa-ok_843d30f2-b631-4a16-8018-a2a31930be70.mp4"
    assert str(player_page.download_url) == expected_download_url
