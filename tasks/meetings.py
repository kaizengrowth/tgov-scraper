import os
from pathlib import Path
import logging

import pandas as pd
from prefect import task

from src.aws import create_bucket_if_not_exists, is_aws_configured, upload_to_s3
from src.meetings import duration_to_minutes, get_meetings
from src.granicus import get_video_player
from src.videos import download_file

file_path = 'data/meetings.csv'  # Path where the file will be saved locally temporarily
meetings_bucket_name = 'tgov-meetings'

logger = logging.getLogger(__name__)

@task
async def create_meetings_csv():
    """Create a CSV file containing meeting data."""
    meetings = await get_meetings()
    print(f"Got meetings: {meetings}")
    meeting_dicts = [meeting.model_dump() for meeting in meetings]
    print(f"meeting_dicts: {meeting_dicts}")
    df = pd.DataFrame(meeting_dicts)
    df['duration_minutes'] = df['duration'].apply(duration_to_minutes)
    df.to_csv(file_path, index=False)

    if is_aws_configured():
        print(f"file_path: {file_path}")
        create_bucket_if_not_exists(meetings_bucket_name)
        if not upload_to_s3(file_path, meetings_bucket_name, file_path):
            raise RuntimeError("Failed to upload to S3")
        os.remove(file_path)  # Remove local file after successful upload
    else:
        output_path = 'meetings.csv'  # Local path if AWS is not configured
        df.to_csv(output_path, index=False)

@task
async def download_videos():
    """Download meeting videos based on the meetings CSV data."""
    # Define paths
    data_dir = Path("data")
    meetings_csv = data_dir / "meetings.csv"
    video_dir = data_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Check if meetings CSV exists
    if not meetings_csv.exists():
        logger.error(f"Meetings CSV file not found at {meetings_csv}")
        return False

    # Load meetings data
    try:
        df = pd.read_csv(meetings_csv)
        logger.info(f"Loaded {len(df)} meetings from CSV")
    except Exception as e:
        logger.error(f"Error loading meetings CSV: {e}")
        return False

    # Process each meeting
    success_count = 0
    skipped_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        meeting_date = row.get('date', 'unknown_date')
        meeting_type = row.get('type', 'meeting')
        video_url = row.get('video_url')

        if not video_url:
            logger.warning(f"No video URL for meeting on {meeting_date}. Skipping.")
            skipped_count += 1
            continue

        # Create clean filename
        file_name = f"{meeting_type.lower().replace(' ', '_')}_{meeting_date.replace('-', '_')}.mp4"
        video_path = video_dir / file_name

        # Skip if already downloaded
        if video_path.exists():
            logger.info(f"Video already exists for {meeting_date}. Skipping.")
            skipped_count += 1
            continue

        try:
            # Get video player info
            logger.info(f"Getting video player for {meeting_date}")
            player = await get_video_player(video_url)

            if not player or not player.download_url:
                logger.error(f"Could not retrieve download URL for {meeting_date}")
                error_count += 1
                continue

            # Download the video
            logger.info(f"Downloading video: {player.download_url}")
            success = await download_file(player.download_url, str(video_path))

            if success:
                logger.info(f"Successfully downloaded video for {meeting_date}")
                success_count += 1

                # Upload to S3 if configured
                if is_aws_configured():
                    bucket_name = "tgov-videos"
                    s3_path = f"videos/{file_name}"

                    logger.info(f"Uploading video to S3: {bucket_name}/{s3_path}")
                    create_bucket_if_not_exists(bucket_name)
                    if upload_to_s3(str(video_path), bucket_name, s3_path):
                        logger.info(f"Successfully uploaded video to S3")
                    else:
                        logger.warning(f"Failed to upload video to S3")
            else:
                logger.error(f"Failed to download video for {meeting_date}")
                error_count += 1

        except Exception as e:
            logger.error(f"Error processing meeting {meeting_date}: {e}")
            error_count += 1

    logger.info(f"Video download task completed. Success: {success_count}, Skipped: {skipped_count}, Errors: {error_count}")
    return True
