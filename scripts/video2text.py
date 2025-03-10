#!/usr/bin/env python3
"""
Script to transcribe video files using faster-whisper and save the results with speaker identification.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError
from urllib3.exceptions import IncompleteRead, ProtocolError
import time
from dotenv import load_dotenv
import subprocess
import tempfile

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Patch huggingface_hub to add retries
try:
    from huggingface_hub import file_download

    original_http_get = file_download.http_get

    def http_get_with_retry(*args, **kwargs):
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                return original_http_get(*args, **kwargs)
            except (
                ChunkedEncodingError,
                ConnectionError,
                IncompleteRead,
                ProtocolError,
            ) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Download failed with error: {e}. Retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Download failed after {max_retries} attempts: {e}")
                    raise

    # Replace the original function with our retry version
    file_download.http_get = http_get_with_retry
    logger.info("Added retry mechanism to huggingface_hub downloads")
except ImportError:
    logger.warning("Could not patch huggingface_hub for retries")

# Now import WhisperModel
from faster_whisper import WhisperModel


async def transcribe_video(
    video_path: str,
    output_dir: str,
    model_size: str = "tiny",
    device: str = "cpu",
    language: Optional[str] = None,
    verbose: bool = False,
    vad_filter: bool = False,
    beam_size: int = 5,
    force_download: bool = False,
) -> str:
    """
    Transcribe a video file using faster-whisper and save the results.
    This is a fallback method when speaker diarization is not available.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the transcription
        model_size: Size of the Whisper model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: Device to use for inference (cpu, cuda, or mps)
        language: Language code (e.g., 'en' for English)
        verbose: Whether to print verbose output
        vad_filter: Whether to use Voice Activity Detection to filter out non-speech
        beam_size: Beam size for the decoder
        force_download: Whether to force download the model even if it exists

    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the base filename without extension
    video_filename = os.path.basename(video_path)
    base_filename = os.path.splitext(video_filename)[0]

    # Define output path
    json_path = os.path.join(output_dir, f"{base_filename}.json")

    # Handle MPS device (Apple Silicon)
    compute_type = "float32"
    if device == "mps":
        logger.info("Using MPS device (Apple Silicon)")
        # MPS is not directly supported by faster-whisper, use CPU with optimized compute type
        device = "cpu"
        compute_type = "float32"  # Use float32 as float16 might not be supported

    logger.info(f"Loading Whisper model: {model_size}")
    # Load the Whisper model
    try:
        from huggingface_hub import snapshot_download

        # Force download the model if requested
        if force_download:
            logger.info(f"Force downloading model {model_size}")
            model_path = snapshot_download(
                repo_id=f"guillaumekln/faster-whisper-{model_size}",
                local_dir=f"./models/whisper/{model_size}",
                force_download=True,
            )
            model = WhisperModel(model_path, device=device, compute_type=compute_type)
        else:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root="./models/whisper",
            )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to tiny model")
        model = WhisperModel(
            "tiny",
            device=device,
            compute_type="float32",
            download_root="./models/whisper",
        )

    logger.info(f"Transcribing video: {video_path}")
    start_time = time.time()

    # Run the transcription
    segments, info = model.transcribe(
        video_path,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None,
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcription_data, f, indent=2, ensure_ascii=False)

    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    logger.info(f"Detailed JSON saved to: {json_path}")

    if verbose:
        print("\nTranscription preview:")
        preview_text = "\n".join([segment["text"] for segment in segments_list[:5]])
        print(f"{preview_text}...")

    return json_path


async def transcribe_video_with_diarization(
    video_path: str,
    output_dir: str,
    model_size: str = "tiny",
    device: str = "cpu",
    language: Optional[str] = None,
    verbose: bool = False,
    vad_filter: bool = False,
    beam_size: int = 5,
    force_download: bool = False,
    hf_token: Optional[str] = None,
    max_duration: int = 300,  # Default to 5 minutes (300 seconds)
    min_segment_duration: float = 0.5,  # Minimum segment duration for speaker assignment
    roll_call_mode: bool = False,  # Special mode for roll call scenarios
) -> str:
    """
    Transcribe a video file with speaker diarization using faster-whisper and pyannote.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the transcription
        model_size: Size of the Whisper model to use
        device: Device to use for inference
        language: Language code
        verbose: Whether to print verbose output
        vad_filter: Whether to use Voice Activity Detection
        beam_size: Beam size for the decoder
        force_download: Whether to force download the model
        hf_token: Hugging Face token for accessing diarization models
        max_duration: Maximum duration in seconds to process (default: 300 seconds / 5 minutes)
        min_segment_duration: Minimum segment duration for speaker assignment (default: 0.5 seconds)
        roll_call_mode: Special mode for roll call scenarios (default: False)

    Returns:
        Path to the saved JSON file

    Raises:
        ValueError: If speaker diarization is not available or fails
    """
    import torch
    from pyannote.audio import Pipeline
    import os
    from pathlib import Path

    # Check for HF token in environment if not provided
    if not hf_token:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError(
                "No Hugging Face token found in HUGGINGFACE_TOKEN environment variable or --hf-token parameter. "
                "Speaker diarization requires a valid Hugging Face token with access to pyannote/speaker-diarization-3.1. "
                "Please visit https://hf.co/pyannote/speaker-diarization-3.1 to accept the user conditions and "
                "set your token in the HUGGINGFACE_TOKEN environment variable or pass it with --hf-token."
            )
        else:
            logger.info("Using Hugging Face token from environment variable.")

    # Get the base filename without extension
    video_filename = os.path.basename(video_path)
    base_filename = os.path.splitext(video_filename)[0]

    # Define output path
    json_path = os.path.join(output_dir, f"{base_filename}.json")

    # First, perform regular transcription
    logger.info("Performing transcription with faster-whisper...")

    # Handle MPS device (Apple Silicon)
    compute_type = "float32"
    whisper_device = device
    if device == "mps":
        logger.info(
            "MPS device requested but faster-whisper doesn't support it. Using CPU instead."
        )
        whisper_device = "cpu"

    # Load the Whisper model
    try:
        from huggingface_hub import snapshot_download

        # Force download the model if requested
        if force_download:
            logger.info(f"Force downloading model {model_size}")
            model_path = snapshot_download(
                repo_id=f"guillaumekln/faster-whisper-{model_size}",
                local_dir=f"./models/whisper/{model_size}",
                force_download=True,
            )
            model = WhisperModel(
                model_path, device=whisper_device, compute_type=compute_type
            )
        else:
            model = WhisperModel(
                model_size,
                device=whisper_device,
                compute_type=compute_type,
                download_root="./models/whisper",
            )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to tiny model")
        model = WhisperModel(
            "tiny",
            device=whisper_device,
            compute_type=compute_type,
            download_root="./models/whisper",
        )

    # Run the transcription
    segments, info = model.transcribe(
        video_path,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None,
    )

    # Process the segments
    segments_list = []
    for segment in segments:
        # Skip segments beyond max_duration
        if segment.start > max_duration:
            break

        segment_dict = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
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

    # Now perform speaker diarization
    logger.info("Performing speaker diarization with pyannote.audio...")

    # Create audio directory if it doesn't exist
    audio_dir = "data/audio"
    os.makedirs(audio_dir, exist_ok=True)

    # Define audio output path
    audio_path = os.path.join(audio_dir, f"{base_filename}_test.wav")

    # Extract audio from video file (only the first max_duration seconds)
    logger.info(f"Extracting first {max_duration} seconds of audio to: {audio_path}")

    try:
        # Use ffmpeg to extract audio (only the first max_duration seconds)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-t",
            str(max_duration),  # Limit duration
            audio_path,
        ]
        logger.info(f"Running command: {' '.join(ffmpeg_cmd)}")

        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        logger.info("Audio extraction completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr.decode('utf-8')}")
        raise ValueError(f"Failed to extract audio from video file: {e}") from e

    # Handle device for diarization
    diarize_device = device
    if device == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using MPS device for diarization")
        else:
            logger.warning(
                "MPS requested but not available. Using CPU for diarization."
            )
            diarize_device = "cpu"

    # Check for PyAnnote models in the Hugging Face cache
    from huggingface_hub import HfFolder, hf_hub_download, try_to_load_from_cache

    # Print Hugging Face cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    logger.info(f"Hugging Face cache directory: {cache_dir}")

    # Check if the token is valid
    logger.info(f"Checking if Hugging Face token is valid...")
    try:
        token_is_valid = HfFolder.get_token() is not None
        logger.info(f"Token from HfFolder: {token_is_valid}")
    except Exception as e:
        logger.warning(f"Error checking token from HfFolder: {e}")

    # Set the token explicitly
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    # Check for PyAnnote models
    model_ids = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
    ]

    for model_id in model_ids:
        logger.info(f"Checking for model: {model_id}")
        try:
            # Try to find the model in cache
            cache_path = try_to_load_from_cache(
                repo_id=model_id,
                filename="config.yaml",
                use_auth_token=hf_token,
            )
            if cache_path:
                logger.info(f"Found model in cache: {cache_path}")
            else:
                logger.info(f"Model not found in cache, will download: {model_id}")

                # Try to download the model
                try:
                    download_path = hf_hub_download(
                        repo_id=model_id,
                        filename="config.yaml",
                        use_auth_token=hf_token,
                        force_download=True,
                    )
                    logger.info(f"Downloaded model to: {download_path}")
                except Exception as download_error:
                    logger.error(
                        f"Error downloading model {model_id}: {download_error}"
                    )
        except Exception as e:
            logger.error(f"Error checking for model {model_id}: {e}")

    # Initialize the speaker diarization pipeline
    try:
        logger.info(
            f"Initializing speaker diarization pipeline with token: {hf_token[:5]}..."
        )
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

        # Set the appropriate device
        if diarize_device == "cuda":
            logger.info("Moving pipeline to CUDA device")
            pipeline = pipeline.to(torch.device("cuda"))
        elif diarize_device == "mps" and torch.backends.mps.is_available():
            logger.info("Moving pipeline to MPS device")
            pipeline = pipeline.to(torch.device("mps"))

        # Run the pipeline on the audio file
        logger.info("Running diarization pipeline...")
        diarization = pipeline(audio_path)

        # Extract speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )

        logger.info(f"Found {len(speaker_segments)} speaker segments")

        # Assign speakers to transcription segments
        logger.info("Assigning speakers to transcription segments...")
        for segment in segments_list:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_duration = segment_end - segment_start

            # Find the speaker who speaks the most during this segment
            speaker_times = {}
            for speaker_segment in speaker_segments:
                if (
                    speaker_segment["end"] > segment_start
                    and speaker_segment["start"] < segment_end
                ):
                    # Calculate overlap
                    overlap_start = max(segment_start, speaker_segment["start"])
                    overlap_end = min(segment_end, speaker_segment["end"])
                    overlap_duration = overlap_end - overlap_start

                    if overlap_duration > 0:
                        speaker = speaker_segment["speaker"]
                        speaker_times[speaker] = (
                            speaker_times.get(speaker, 0) + overlap_duration
                        )

            # Special handling for very short segments (like "here" responses in roll calls)
            if roll_call_mode and segment_duration < min_segment_duration:
                # For very short segments, we're more sensitive to speaker changes
                # Check if this segment is likely a response in a roll call
                if segment["text"].strip().lower() in ["here", "present", "yes", "no"]:
                    # Look for the most recent speaker change
                    if speaker_times:
                        # If multiple speakers detected, prefer the one that's different from previous segment
                        if (
                            len(speaker_times) > 1
                            and len(segments_list) > 0
                            and segment["id"] > 1
                        ):
                            # Get the previous segment's speaker
                            prev_idx = segment["id"] - 2  # Adjust for 0-indexing
                            if prev_idx >= 0 and prev_idx < len(segments_list):
                                prev_speaker = segments_list[prev_idx].get(
                                    "speaker", "SPEAKER_UNK"
                                )

                                # Prefer a different speaker than the previous segment
                                for spk in speaker_times:
                                    if spk != prev_speaker:
                                        segment["speaker"] = spk
                                        break
                                else:
                                    # If no different speaker found, use the dominant one
                                    dominant_speaker = max(
                                        speaker_times, key=speaker_times.get
                                    )
                                    segment["speaker"] = dominant_speaker
                            else:
                                # If can't find previous segment, use dominant speaker
                                dominant_speaker = max(
                                    speaker_times, key=speaker_times.get
                                )
                                segment["speaker"] = dominant_speaker
                        else:
                            # Use the dominant speaker
                            dominant_speaker = max(speaker_times, key=speaker_times.get)
                            segment["speaker"] = dominant_speaker
                    else:
                        segment["speaker"] = "SPEAKER_UNK"
                else:
                    # For non-response segments, use the dominant speaker
                    if speaker_times:
                        dominant_speaker = max(speaker_times, key=speaker_times.get)
                        segment["speaker"] = dominant_speaker
                    else:
                        segment["speaker"] = "SPEAKER_UNK"
            else:
                # Standard speaker assignment for normal segments
                if speaker_times:
                    dominant_speaker = max(speaker_times, key=speaker_times.get)
                    segment["speaker"] = dominant_speaker
                else:
                    segment["speaker"] = "SPEAKER_UNK"
    except Exception as e:
        logger.error(
            f"Error during diarization pipeline initialization or processing: {e}"
        )
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

        # Fail with a clear error message
        raise ValueError(
            f"Speaker diarization failed: {e}\n\n"
            "Please ensure you have:\n"
            "1. A valid Hugging Face token with access to pyannote/speaker-diarization-3.1\n"
            "2. Accepted the user conditions at https://hf.co/pyannote/speaker-diarization-3.1\n"
            "3. Set your token in the HUGGINGFACE_TOKEN environment variable or passed it with --hf-token\n"
            "4. A working internet connection to download the model"
        ) from e

    # Save the detailed JSON with timing and speaker information
    transcription_data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments_list,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcription_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed JSON with speaker information saved to: {json_path}")

    if verbose:
        print("\nTranscription preview with speakers:")
        for i, segment in enumerate(segments_list[:5]):
            speaker = segment.get("speaker", "Unknown")
            print(f"{speaker}: {segment['text']}")
            if i == 4:  # Show only first 5 segments
                print("...")

    return json_path


async def process_directory(
    input_dir: str,
    output_dir: str,
    model_size: str = "tiny",
    device: str = "cpu",
    language: Optional[str] = None,
    verbose: bool = False,
    vad_filter: bool = False,
    beam_size: int = 5,
    force_download: bool = False,
    hf_token: Optional[str] = None,
    max_duration: int = 300,  # Default to 5 minutes (300 seconds)
    min_segment_duration: float = 0.5,  # Minimum segment duration for speaker assignment
    roll_call_mode: bool = False,  # Special mode for roll call scenarios
) -> List[str]:
    """
    Process all video files in a directory with speaker diarization.

    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save transcriptions
        model_size: Size of the Whisper model to use
        device: Device to use for inference
        language: Language code
        verbose: Whether to print verbose output
        vad_filter: Whether to use Voice Activity Detection
        beam_size: Beam size for the decoder
        force_download: Whether to force download the model
        hf_token: Hugging Face token for accessing diarization models
        max_duration: Maximum duration in seconds to process (default: 300 seconds / 5 minutes)
        min_segment_duration: Minimum segment duration for speaker assignment (default: 0.5 seconds)
        roll_call_mode: Special mode for roll call scenarios (default: False)

    Returns:
        List of paths to the saved JSON files

    Raises:
        ValueError: If speaker diarization is not available or fails
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    video_files = []

    # Find all video files in the directory
    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f"*{ext}")))

    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return []

    logger.info(f"Found {len(video_files)} video files to process")

    # Process each video file
    json_paths = []
    for video_file in video_files:
        json_path = await transcribe_video_with_diarization(
            str(video_file),
            output_dir,
            model_size=model_size,
            device=device,
            language=language,
            verbose=verbose,
            vad_filter=vad_filter,
            beam_size=beam_size,
            force_download=force_download,
            hf_token=hf_token,
            max_duration=max_duration,
            min_segment_duration=min_segment_duration,
            roll_call_mode=roll_call_mode,
        )
        json_paths.append(json_path)

    return json_paths


async def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video files using faster-whisper with speaker identification"
    )
    parser.add_argument(
        "input", help="Path to video file or directory containing video files"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/transcripts",
        help="Directory to save transcriptions (default: data/transcripts)",
    )
    parser.add_argument(
        "--model-size",
        "-m",
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        default="small",
        help="Size of the Whisper model to use (default: small)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use for inference (default: cpu, use mps for Apple Silicon)",
    )
    parser.add_argument(
        "--language",
        "-l",
        help="Language code (e.g., 'en' for English). If not specified, Whisper will auto-detect.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Use Voice Activity Detection to filter out non-speech (default: False)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for the decoder (default: 5)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download the model even if it exists (default: False)",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for accessing diarization models. If not provided, will look for HUGGINGFACE_TOKEN environment variable.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=300,
        help="Maximum duration in seconds to process (default: 300 seconds / 5 minutes)",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration for speaker assignment (default: 0.5 seconds)",
    )
    parser.add_argument(
        "--roll-call-mode",
        action="store_true",
        help="Special mode for roll call scenarios (default: False)",
    )

    args = parser.parse_args()

    # Create models directory if it doesn't exist
    os.makedirs("models/whisper", exist_ok=True)

    try:
        # Check if input is a file or directory
        input_path = args.input
        if os.path.isfile(input_path):
            # Process a single file with speaker diarization
            await transcribe_video_with_diarization(
                input_path,
                args.output_dir,
                model_size=args.model_size,
                device=args.device,
                language=args.language,
                verbose=args.verbose,
                vad_filter=args.vad_filter,
                beam_size=args.beam_size,
                force_download=args.force_download,
                hf_token=args.hf_token,
                max_duration=args.max_duration,
                min_segment_duration=args.min_segment_duration,
                roll_call_mode=args.roll_call_mode,
            )
        elif os.path.isdir(input_path):
            # Process all video files in the directory with speaker diarization
            await process_directory(
                input_path,
                args.output_dir,
                model_size=args.model_size,
                device=args.device,
                language=args.language,
                verbose=args.verbose,
                vad_filter=args.vad_filter,
                beam_size=args.beam_size,
                force_download=args.force_download,
                hf_token=args.hf_token,
                max_duration=args.max_duration,
                min_segment_duration=args.min_segment_duration,
                roll_call_mode=args.roll_call_mode,
            )
        else:
            logger.error(f"Error: Input path {input_path} does not exist")
            sys.exit(1)
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
