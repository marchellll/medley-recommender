"""Tests for audio processing modules."""

import pytest
from pathlib import Path

from src.audio.downloader import sanitize_filename, validate_youtube_url


def test_sanitize_filename():
    """Test filename sanitization."""
    assert sanitize_filename("test/song.mp3") == "test_song.mp3"
    assert sanitize_filename("song: name") == "song_ name"
    assert sanitize_filename("  test  ") == "test"


def test_validate_youtube_url():
    """Test YouTube URL validation."""
    assert validate_youtube_url("https://www.youtube.com/watch?v=test123") is True
    assert validate_youtube_url("https://youtu.be/test123") is True
    assert validate_youtube_url("https://example.com") is False
    assert validate_youtube_url("not a url") is False




