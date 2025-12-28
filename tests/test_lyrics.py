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


def test_clean_lyrics_removes_bom():
    """Test that BOM characters are removed."""
    # BOM at start
    lyrics = "\ufeffHello World"
    cleaned = clean_lyrics(lyrics)
    assert "\ufeff" not in cleaned
    assert cleaned == "Hello World"

    # BOM in middle
    lyrics = "Hello\ufeff World"
    cleaned = clean_lyrics(lyrics)
    assert "\ufeff" not in cleaned
    assert cleaned == "Hello World"


def test_clean_lyrics_removes_zero_width_chars():
    """Test that zero-width characters are removed."""
    # Zero-width space
    lyrics = "Hello\u200B World"
    cleaned = clean_lyrics(lyrics)
    assert "\u200B" not in cleaned
    assert cleaned == "Hello World"

    # Zero-width non-joiner
    lyrics = "Hello\u200C World"
    cleaned = clean_lyrics(lyrics)
    assert "\u200C" not in cleaned
    assert cleaned == "Hello World"

    # Zero-width joiner
    lyrics = "Hello\u200D World"
    cleaned = clean_lyrics(lyrics)
    assert "\u200D" not in cleaned
    assert cleaned == "Hello World"


def test_clean_lyrics_removes_bidirectional_marks():
    """Test that bidirectional marks are removed."""
    # Left-to-right mark
    lyrics = "Hello\u200E World"
    cleaned = clean_lyrics(lyrics)
    assert "\u200E" not in cleaned
    assert cleaned == "Hello World"

    # Right-to-left mark
    lyrics = "Hello\u200F World"
    cleaned = clean_lyrics(lyrics)
    assert "\u200F" not in cleaned
    assert cleaned == "Hello World"


def test_clean_lyrics_removes_control_chars():
    """Test that control characters are removed (except newline, tab, carriage return)."""
    # Bell character
    lyrics = "Hello\a World"
    cleaned = clean_lyrics(lyrics)
    assert "\a" not in cleaned
    assert "Hello" in cleaned and "World" in cleaned

    # Form feed
    lyrics = "Hello\f World"
    cleaned = clean_lyrics(lyrics)
    assert "\f" not in cleaned
    assert "Hello" in cleaned and "World" in cleaned


def test_clean_lyrics_preserves_newlines_tabs():
    """Test that newlines and tabs are preserved."""
    lyrics = "Line 1\nLine 2\tTabbed"
    cleaned = clean_lyrics(lyrics)
    assert "\n" in cleaned
    assert "\t" in cleaned
    assert "Line 1" in cleaned
    assert "Line 2" in cleaned
    assert "Tabbed" in cleaned


def test_clean_lyrics_preserves_combining_marks():
    """Test that combining marks (accents) are preserved."""
    # Text with accents
    lyrics = "café naïve résumé"
    cleaned = clean_lyrics(lyrics)
    assert "é" in cleaned
    assert "ï" in cleaned
    assert "é" in cleaned
    assert cleaned == "café naïve résumé"


def test_clean_lyrics_complex_hidden_chars():
    """Test cleaning with multiple hidden characters."""
    lyrics = "\ufeffHello\u200B\u200C\u200D\u200E\u200F World\u2060"
    cleaned = clean_lyrics(lyrics)
    # All hidden characters should be removed
    assert "\ufeff" not in cleaned
    assert "\u200B" not in cleaned
    assert "\u200C" not in cleaned
    assert "\u200D" not in cleaned
    assert "\u200E" not in cleaned
    assert "\u200F" not in cleaned
    assert "\u2060" not in cleaned
    # Content should remain
    assert "Hello" in cleaned
    assert "World" in cleaned


def test_clean_lyrics_normalizes_curly_apostrophe():
    """Test that curly apostrophe is normalized to regular apostrophe."""
    # Right single quotation mark (most common curly apostrophe)
    lyrics = "Don't stop believin'"
    cleaned = clean_lyrics(lyrics.replace("'", "\u2019"))  # Replace with curly apostrophe
    assert "\u2019" not in cleaned
    assert "'" in cleaned
    assert cleaned == "Don't stop believin'"

    # Left single quotation mark
    lyrics = "'Tis the season"
    cleaned = clean_lyrics(lyrics.replace("'", "\u2018"))  # Replace with left curly quote
    assert "\u2018" not in cleaned
    assert "'" in cleaned
    assert cleaned == "'Tis the season"


def test_clean_lyrics_normalizes_curly_quotes():
    """Test that curly double quotes are normalized to regular quotes."""
    lyrics = 'He said "Hello"'
    cleaned = clean_lyrics(lyrics.replace('"', "\u201C").replace('"', "\u201D"))
    assert "\u201C" not in cleaned
    assert "\u201D" not in cleaned
    assert '"' in cleaned
    assert cleaned == 'He said "Hello"'


def test_clean_lyrics_normalizes_dashes():
    """Test that em dashes and en dashes are normalized."""
    # Em dash
    lyrics = "Hello—World"
    cleaned = clean_lyrics(lyrics.replace("-", "\u2014"))
    assert "\u2014" not in cleaned
    assert "--" in cleaned
    assert cleaned == "Hello--World"

    # En dash
    lyrics = "Hello–World"
    cleaned = clean_lyrics(lyrics.replace("-", "\u2013"))
    assert "\u2013" not in cleaned
    assert "-" in cleaned
    assert cleaned == "Hello-World"


def test_clean_lyrics_normalizes_ellipsis():
    """Test that ellipsis is normalized to three dots."""
    lyrics = "Hello...World"
    cleaned = clean_lyrics(lyrics.replace("...", "\u2026"))
    assert "\u2026" not in cleaned
    assert "..." in cleaned
    assert cleaned == "Hello...World"


def test_clean_lyrics_normalizes_non_breaking_space():
    """Test that non-breaking space is normalized to regular space."""
    lyrics = "Hello World"
    cleaned = clean_lyrics(lyrics.replace(" ", "\u00A0"))
    assert "\u00A0" not in cleaned
    assert " " in cleaned
    assert cleaned == "Hello World"


def test_clean_lyrics_typographic_characters_in_lyrics():
    """Test realistic lyrics with multiple typographic characters."""
    lyrics = "Don't stop believin'—hold on to that feelin'"
    # Replace with typographic characters
    lyrics_typo = lyrics.replace("'", "\u2019").replace("-", "\u2014")
    cleaned = clean_lyrics(lyrics_typo)
    # Should be normalized back
    assert "\u2019" not in cleaned
    assert "\u2014" not in cleaned
    assert "'" in cleaned
    assert "--" in cleaned
    assert cleaned == "Don't stop believin'--hold on to that feelin'"






