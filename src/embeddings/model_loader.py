"""Model loading and caching for embeddings."""

import os
import time
import threading
from pathlib import Path
from typing import Callable, Optional

from sentence_transformers import SentenceTransformer

from src.utils.config import settings


def _get_device() -> str:
    """
    Determine the best available device for model inference.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    try:
        import torch

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            return "cuda"

        # Check for MPS (Apple Silicon GPU)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"

        # Fall back to CPU
        return "cpu"
    except ImportError:
        # PyTorch not available, use CPU
        return "cpu"


_model_cache: Optional[SentenceTransformer] = None


def _check_model_exists(model_name: str, cache_dir: Path) -> bool:
    """Check if model files already exist in cache."""
    # HuggingFace/SentenceTransformer uses "models--" prefix for model directories
    # e.g., "BAAI/bge-m3" -> "models--BAAI--bge-m3"
    # Model files are stored in snapshots/<hash>/ subdirectory
    model_path_hf = cache_dir / f"models--{model_name.replace('/', '--')}"
    # Also check without prefix (for backwards compatibility)
    model_path_legacy = cache_dir / model_name.replace("/", "--")

    # Check HuggingFace format first
    for model_path in [model_path_hf, model_path_legacy]:
        if model_path.exists():
            # Check in snapshots directory (HuggingFace format)
            snapshots_dir = model_path / "snapshots"
            if snapshots_dir.exists():
                # Check any snapshot subdirectory
                for snapshot_dir in snapshots_dir.iterdir():
                    if snapshot_dir.is_dir():
                        if (snapshot_dir / "config.json").exists() or (snapshot_dir / "pytorch_model.bin").exists():
                            return True
                        if any(snapshot_dir.glob("*.safetensors")):
                            return True
            # Check directly in model path (legacy format)
            if (model_path / "config.json").exists() or (model_path / "pytorch_model.bin").exists():
                return True
            if any(model_path.glob("*.safetensors")):
                return True
    return False


def _format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def _format_speed(bytes_per_sec: float) -> str:
    """Format bytes per second to human-readable string."""
    return _format_bytes(int(bytes_per_sec)) + "/s"


def _get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def get_embedding_model(
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SentenceTransformer:
    """
    Get or load the embedding model (cached).

    Args:
        progress_callback: Optional callback for progress updates (receives status message)

    Returns:
        SentenceTransformer model instance
    """
    global _model_cache

    if _model_cache is None:
        model_name = settings.embedding_model
        # Set cache directory to avoid polluting user's home
        cache_dir = Path.home() / ".cache" / "medley-recommender" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if model needs to be downloaded
        needs_download = not _check_model_exists(model_name, cache_dir)

        # Set up download progress tracking if needed
        download_monitor = None
        download_stats = None
        # HuggingFace uses "models--" prefix
        model_cache_path = cache_dir / f"models--{model_name.replace('/', '--')}"

        if needs_download and progress_callback:
            # Get initial cache directory size
            initial_size = _get_directory_size(model_cache_path) if model_cache_path.exists() else 0

            # Track download progress
            download_stats = {
                "start_time": time.time(),
                "last_size": initial_size,
                "last_time": time.time(),
                "running": True,
            }

            def monitor_download() -> None:
                """Monitor cache directory size during download."""
                while download_stats["running"]:
                    current_size = _get_directory_size(model_cache_path)
                    current_time = time.time()

                    # Calculate speed from size change
                    size_delta = current_size - download_stats["last_size"]
                    time_delta = current_time - download_stats["last_time"]

                    if time_delta > 0.5:  # Update every 0.5 seconds
                        if size_delta > 0:
                            speed = size_delta / time_delta
                            progress_callback(
                                f"Downloading {model_name}: {_format_bytes(current_size)} "
                                f"@ {_format_speed(speed)}"
                            )
                        else:
                            # Still downloading but no size change yet
                            progress_callback(
                                f"Downloading {model_name}: {_format_bytes(current_size)}..."
                            )

                        download_stats["last_size"] = current_size
                        download_stats["last_time"] = current_time

                    time.sleep(0.2)  # Check every 200ms

            # Start monitoring thread
            download_monitor = threading.Thread(target=monitor_download, daemon=True)
            download_monitor.start()
            progress_callback(f"Starting download: {model_name}...")

        # Determine device (CUDA > MPS > CPU)
        device = _get_device()

        # Enable progress bars for huggingface_hub
        try:
            from huggingface_hub.utils import enable_progress_bars
            # Make sure progress bars are enabled
            enable_progress_bars()
        except ImportError:
            pass

        # Load model with caching
        # SentenceTransformer will show its own progress via tqdm
        # We enable it by default - it will show download progress
        try:
            _model_cache = SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir),
                device=device,
            )
        finally:
            # Stop download monitoring
            if download_monitor is not None and download_stats is not None:
                download_stats["running"] = False
                download_monitor.join(timeout=1.0)

                # Show final size
                if progress_callback:
                    final_size = _get_directory_size(model_cache_path)
                    progress_callback(f"Model downloaded: {_format_bytes(final_size)}")

        if progress_callback:
            progress_callback(f"Model loaded: {model_name}")

    return _model_cache


def clear_model_cache() -> None:
    """Clear the model cache (useful for testing)."""
    global _model_cache
    _model_cache = None


