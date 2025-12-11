"""Pytest configuration and fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.utils.config import Settings


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()
        (data_dir / "audio").mkdir()
        (data_dir / "songs_embeddings").mkdir()
        (data_dir / "index").mkdir()
        (data_dir / "new_songs").mkdir()
        yield data_dir


@pytest.fixture
def sample_songs_json() -> list[dict]:
    """Sample songs data for testing."""
    return [
        {
            "song_id": "test_song_1",
            "title": "Test Song 1",
            "artist": "Test Artist",
            "youtube_url": "https://www.youtube.com/watch?v=test1",
            "lyrics": "Test lyrics for song 1\nLine 2\nLine 3",
        },
        {
            "song_id": "test_song_2",
            "title": "Test Song 2",
            "artist": "Test Artist 2",
            "youtube_url": "https://www.youtube.com/watch?v=test2",
            "lyrics": "Test lyrics for song 2\nDifferent content",
        },
    ]


@pytest.fixture
def sample_songs_file(temp_data_dir: Path, sample_songs_json: list[dict]) -> Path:
    """Create a sample songs.json file."""
    songs_file = temp_data_dir / "songs.json"
    with open(songs_file, "w") as f:
        json.dump(sample_songs_json, f, indent=2)
    return songs_file


@pytest.fixture
def test_settings(temp_data_dir: Path) -> Settings:
    """Create test settings with temporary data directory."""
    return Settings(
        database_path=str(temp_data_dir / "test.db"),
    )


