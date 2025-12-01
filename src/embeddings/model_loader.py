"""Model loading and caching for embeddings."""

import os
from pathlib import Path
from typing import Callable, Optional

from sentence_transformers import SentenceTransformer

from src.utils.config import settings


_model_cache: Optional[SentenceTransformer] = None


def _check_model_exists(model_name: str, cache_dir: Path) -> bool:
    """Check if model files already exist in cache."""
    # SentenceTransformer models are typically stored in a subdirectory
    # Check for common model files
    model_path = cache_dir / model_name.replace("/", "--")
    if model_path.exists():
        # Check for model files
        if (model_path / "config.json").exists() or (model_path / "pytorch_model.bin").exists():
            return True
    return False


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

        if needs_download and progress_callback:
            progress_callback(f"Downloading model: {model_name}...")

        # Load model with caching
        # SentenceTransformer will show its own progress via tqdm
        # We enable it by default - it will show download progress
        _model_cache = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
        )

        if progress_callback:
            progress_callback(f"Model loaded: {model_name}")

    return _model_cache


def clear_model_cache() -> None:
    """Clear the model cache (useful for testing)."""
    global _model_cache
    _model_cache = None


