"""Lyrics cleaning and normalization utilities."""

import re
import unicodedata


def clean_lyrics(lyrics: str) -> str:
    """
    Clean and normalize lyrics text.

    Args:
        lyrics: Raw lyrics text

    Returns:
        Cleaned lyrics text
    """
    if not lyrics:
        return ""

    # Normalize unicode (e.g., convert smart quotes to regular quotes)
    lyrics = unicodedata.normalize("NFKD", lyrics)

    # Remove extra whitespace but preserve line breaks
    lines = lyrics.split("\n")
    cleaned_lines = []
    for line in lines:
        # Strip leading/trailing whitespace
        cleaned_line = line.strip()
        # Replace multiple spaces with single space
        cleaned_line = re.sub(r" +", " ", cleaned_line)
        if cleaned_line:  # Keep non-empty lines
            cleaned_lines.append(cleaned_line)

    # Join lines back together
    cleaned = "\n".join(cleaned_lines)

    # Remove excessive blank lines (more than 2 consecutive)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Convert to lowercase for consistency (if needed)
    # text = text.lower()

    return text.strip()


