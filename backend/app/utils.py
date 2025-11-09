"""
Helper utility functions.
"""
import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_assets_path() -> Path:
    """Get the path to the assets directory."""
    return get_project_root() / "assets"


def get_data_path() -> Path:
    """Get the path to the data directory."""
    return get_project_root() / "data"


def validate_audio_file(file_path: str) -> bool:
    """Validate that an audio file exists and is readable."""
    return os.path.exists(file_path) and os.path.isfile(file_path)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

