"""Tests for lyrics utilities."""

from src.utils.lyrics import clean_lyrics, normalize_text


def test_clean_lyrics():
    """Test lyrics cleaning."""
    lyrics = "  Line 1  \n\n  Line 2  \n\n\n  Line 3  "
    cleaned = clean_lyrics(lyrics)
    assert "Line 1" in cleaned
    assert "Line 2" in cleaned
    assert "Line 3" in cleaned
    assert cleaned.count("\n\n") <= 1  # No excessive blank lines


def test_normalize_text():
    """Test text normalization."""
    text = "  Test  Text  "
    normalized = normalize_text(text)
    assert normalized == "Test  Text"


