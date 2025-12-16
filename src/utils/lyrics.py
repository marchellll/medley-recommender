"""Lyrics cleaning and normalization utilities."""

import re
import unicodedata


def clean_lyrics(lyrics: str) -> str:
    """
    Clean and normalize lyrics text, removing unreadable/hidden characters.

    This function performs comprehensive cleaning:
    - Removes BOM (Byte Order Mark) characters
    - Normalizes typographic characters (e.g., curly quotes ' ' → ' ', em dash — → --)
    - Removes control characters (except newline, tab, carriage return)
    - Removes zero-width characters
    - Removes bidirectional marks and other invisible formatting characters
    - Normalizes unicode characters
    - Cleans whitespace while preserving line breaks

    Args:
        lyrics: Raw lyrics text

    Returns:
        Cleaned lyrics text
    """
    if not lyrics:
        return ""

    # Remove BOM (Byte Order Mark) - commonly found at start of UTF-8 files
    lyrics = lyrics.replace("\ufeff", "").replace("\uFEFF", "")

    # Normalize unicode (e.g., convert smart quotes to regular quotes)
    # NFKD decomposes characters, which helps identify hidden characters
    lyrics = unicodedata.normalize("NFKD", lyrics)

    # Normalize typographic characters to ASCII equivalents
    # Mapping of typographic characters to their ASCII equivalents
    TYPOGRAPHIC_MAP = {
        # Curly single quotes
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark (apostrophe)
        "\u201A": "'",  # Single low-9 quotation mark
        "\u201B": "'",  # Single high-reversed-9 quotation mark
        # Curly double quotes
        "\u201C": '"',  # Left double quotation mark
        "\u201D": '"',  # Right double quotation mark
        "\u201E": '"',  # Double low-9 quotation mark
        "\u201F": '"',  # Double high-reversed-9 quotation mark
        # Dashes
        "\u2013": "-",  # En dash
        "\u2014": "--",  # Em dash
        "\u2015": "--",  # Horizontal bar
        # Ellipsis
        "\u2026": "...",  # Horizontal ellipsis
        # Other typographic characters
        "\u2032": "'",  # Prime (minutes)
        "\u2033": '"',  # Double prime (seconds)
        "\u2039": "<",  # Single left-pointing angle quotation mark
        "\u203A": ">",  # Single right-pointing angle quotation mark
        "\u00A0": " ",  # Non-breaking space → regular space
        "\u2028": "\n",  # Line separator → newline
        "\u2029": "\n\n",  # Paragraph separator → double newline
    }

    # Replace typographic characters
    for typo_char, ascii_char in TYPOGRAPHIC_MAP.items():
        lyrics = lyrics.replace(typo_char, ascii_char)

    # Set of invisible/hidden characters to remove
    # These include zero-width characters, bidirectional marks, and formatting characters
    INVISIBLE_CHARS = {
        # Zero-width characters
        0x200B,  # Zero-width space
        0x200C,  # Zero-width non-joiner
        0x200D,  # Zero-width joiner
        0xFEFF,  # Zero-width no-break space (BOM)
        # Bidirectional marks
        0x200E,  # Left-to-right mark
        0x200F,  # Right-to-left mark
        0x202A,  # Left-to-right embedding
        0x202B,  # Right-to-left embedding
        0x202C,  # Pop directional formatting
        0x202D,  # Left-to-right override
        0x202E,  # Right-to-left override
        # Word joiners and invisible operators
        0x2060,  # Word joiner
        0x2061,  # Function application
        0x2062,  # Invisible times
        0x2063,  # Invisible separator
        0x2064,  # Invisible plus
        # Directional isolates
        0x2066,  # Left-to-right isolate
        0x2067,  # Right-to-left isolate
        0x2068,  # First strong isolate
        0x2069,  # Pop directional isolate
        # Symmetric swapping controls
        0x206A,  # Inhibit symmetric swapping
        0x206B,  # Activate symmetric swapping
        0x206C,  # Inhibit arabic form shaping
        0x206D,  # Activate arabic form shaping
        0x206E,  # National digit shapes
        0x206F,  # Nominal digit shapes
    }

    # Remove control characters except newline (\n), tab (\t), and carriage return (\r)
    # Control characters are in the range \x00-\x1F and \x7F-\x9F
    # We keep \n (0x0A), \t (0x09), and \r (0x0D)
    cleaned_chars = []
    for char in lyrics:
        code = ord(char)
        category = unicodedata.category(char)

        # Always keep newline, tab, and carriage return
        if char in {"\n", "\t", "\r"}:
            cleaned_chars.append(char)
        # Remove control characters (except those we keep above)
        elif category[0] == "C":
            continue
        # Remove invisible/hidden characters from our list
        elif code in INVISIBLE_CHARS:
            continue
        # Remove other formatting characters (Cf category) except soft hyphen
        elif category == "Cf" and code != 0x00AD:  # Keep soft hyphen, remove other formatting
            continue
        # Keep everything else (printable characters, combining marks for accents, etc.)
        else:
            cleaned_chars.append(char)

    lyrics = "".join(cleaned_chars)

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
    Normalize text for consistent processing, removing hidden characters.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Use clean_lyrics for comprehensive cleaning
    # (it handles both lyrics and general text well)
    text = clean_lyrics(text)

    # Convert to lowercase for consistency (if needed)
    # text = text.lower()

    return text.strip()




