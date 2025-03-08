#!/usr/bin/env python3
"""
Script to download HTML from the TGOV website and save it as a fixture for testing.
"""

import asyncio
import os
from pathlib import Path

import aiohttp


async def download_html(url: str, output_path: str) -> None:
    """
    Download HTML from a URL and save it to a file.

    Args:
        url: The URL to download from
        output_path: The path to save the HTML to
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(
                    f"Failed to download {url}, status code: {response.status}"
                )

            html = await response.text()

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save HTML to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"Downloaded HTML from {url} to {output_path}")


async def main():
    """Main function to download fixtures."""
    # URL for the TGOV website
    url = "https://tulsa-ok.granicus.com/ViewPublisher.php?view_id=4"

    # Path to save the fixture
    fixture_path = (
        Path(__file__).parent.parent / "tests" / "fixtures" / "tgov_homepage.html"
    )

    await download_html(url, str(fixture_path))


if __name__ == "__main__":
    asyncio.run(main())
