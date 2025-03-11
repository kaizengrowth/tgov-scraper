#!/usr/bin/env python3
"""
Utility functions for working with video streams from Granicus player pages.
"""

import asyncio
import re
import tempfile
import os
import io
from typing import Optional, Tuple, BinaryIO
from urllib.request import urlopen
import time
import json
import aiohttp
from selectolax.parser import HTMLParser
import m3u8
import requests
from pydub import AudioSegment
from pathlib import Path
from .huggingface import get_whisper_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_page(url: str, session: aiohttp.ClientSession) -> str:
    """
    Fetch the HTML content of a page.

    Args:
        url: The URL to fetch
        session: An aiohttp ClientSession

    Returns:
        The HTML content as a string
    """
    async with session.get(url) as response:
        if response.status != 200:
            raise Exception(f"Failed to fetch {url}, status code: {response.status}")
        return await response.text()


def download_file(url: str, output_path: Path):
    print(f"Downloading video from: {url}")
    print(f"Saving to: {output_path}")

    with requests.get(str(url)) as response:
        if response.status_code != 200:
            print(f"Failed to download, status code: {response.status_code}")
            return None

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)

                # Report progress every 5%
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    if percent % 5 == 0 and percent > 0:
                        print(
                            f"Downloaded {percent}% ({downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB)"
                        )

    print(f"Download complete: {url}")
    return output_path


async def save_audio(
    video_path: str,
    output_path: Optional[str] = None,
    max_duration: Optional[int] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    temp_dir: Optional[str] = None,
) -> str:
    """
    Extract audio from a video file using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio (WAV format)
                    If None, a temporary file will be created
        max_duration: Maximum duration in seconds to extract (None for entire video)
        sample_rate: Sample rate for the output audio (default: 16000 Hz)
        channels: Number of audio channels (default: 1 for mono)
        temp_dir: Directory to use for temporary files (if output_path is None)

    Returns:
        Path to the extracted audio file
    """
    # If no output path is provided, create a temporary file
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=temp_dir
        )
        output_path = temp_file.name
        temp_file.close()

    # Prepare ffmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i",
        video_path,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        str(channels),  # Number of channels
    ]

    # Add duration limit if specified
    if max_duration is not None:
        ffmpeg_cmd.extend(["-t", str(max_duration)])

    # Add output path
    ffmpeg_cmd.append(output_path)

    # Run ffmpeg command
    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode("utf-8") if stderr else "Unknown error"
            raise RuntimeError(f"Failed to extract audio: {error_message}")

        return output_path
    except Exception as e:
        # Clean up temporary file if we created one and an error occurred
        if output_path is None and os.path.exists(output_path):
            os.unlink(output_path)
        raise e


async def transcribe_video(video_path: Path, output_path: Path):
    model = await get_whisper_model(model_size="tiny")
    logger.info(f"Transcribing video: {video_path}")
    start_time = time.time()
    # Get the base filename without extension
    video_filename = os.path.basename(video_path)
    base_filename = os.path.splitext(video_filename)[0]
    transcription_path = os.path.join(output_path, f"{base_filename}.json")
    logger.info(f"transcription will be saved to: {transcription_path}")

    # Run the transcription
    segments, info = model.transcribe(
        video_path,
        language=None,
        task="transcribe",
        beam_size=5,
        vad_filter=False,
        vad_parameters=None,
    )

    # Process the segments
    segments_list = []

    logger.info("Processing transcription segments...")
    for segment in segments:
        # Add to segments list for JSON output
        segment_dict = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "speaker": "Unknown",  # Default speaker when diarization is not available
            "words": (
                [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]
                if segment.words
                else []
            ),
        }
        segments_list.append(segment_dict)

    # Save the detailed JSON with timing information
    transcription_data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments_list,
    }

    with open(transcription_path, "w", encoding="utf-8") as f:
        json.dump(transcription_data, f, indent=2, ensure_ascii=False)

    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    logger.info(f"Detailed JSON saved to: {transcription_path}")
    return transcription_data
