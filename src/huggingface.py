"""
Utility functions for loading Hugging Face models.
"""

import os
import logging
import time
from typing import Optional
from pathlib import Path
import torch
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")


async def get_whisper_model(
    model_size: str,
    device: str = "cpu",
    force_download: bool = False,
    download_root: str = "../models/whisper",
):
    """
    Load a Whisper model from Hugging Face.

    Args:
        model_size: Size of the model (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: Device to use for inference (cpu, cuda, or mps)
        force_download: Whether to force download the model even if it exists
        download_root: Directory to save the downloaded model

    Returns:
        Loaded WhisperModel
    """
    # Handle MPS device (Apple Silicon)
    compute_type = "float32"
    whisper_device = device
    # Ensure the model directory exists
    os.makedirs(download_root, exist_ok=True)

    # Load the Whisper model

    model = WhisperModel(
        model_size,
        device=whisper_device,
        compute_type=compute_type,
        download_root=download_root,
    )
    logger.info(f"Successfully loaded Whisper model: {model_size}")
    return model


async def load_diarization_pipeline(
    hf_token: Optional[str] = None, device: str = "cpu"
) -> "Pipeline":
    """
    Load the PyAnnote speaker diarization pipeline.

    Args:
        hf_token: Hugging Face token for accessing diarization models
        device: Device to use for inference (cpu, cuda, or mps)

    Returns:
        Loaded diarization pipeline

    Raises:
        ValueError: If token is invalid or model loading fails
    """

    # Check for HF token in environment if not provided

    # Set the token explicitly
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    # Check for PyAnnote models in the Hugging Face cache
    await verify_pyannote_models(hf_token)

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

        return pipeline
    except Exception as e:
        logger.error(f"Error during diarization pipeline initialization: {e}")
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


async def verify_pyannote_models(hf_token: str) -> bool:
    """
    Verify that PyAnnote models are available in the Hugging Face cache or can be downloaded.

    Args:
        hf_token: Hugging Face token for accessing diarization models

    Returns:
        True if all models are available, False otherwise
    """
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

    # Check for PyAnnote models
    model_ids = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
    ]

    all_models_available = True
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
                    all_models_available = False
        except Exception as e:
            logger.error(f"Error checking for model {model_id}: {e}")
            all_models_available = False

    return all_models_available
