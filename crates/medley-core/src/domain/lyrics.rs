/// Simplified lyrics cleaning — normalize whitespace and common typographic chars.
pub fn clean_lyrics(lyrics: &str) -> String {
    if lyrics.is_empty() {
        return String::new();
    }

    let mut text = lyrics.replace('\u{feff}', "");
    let replacements = [
        ('\u{2018}', '\''),
        ('\u{2019}', '\''),
        ('\u{201c}', '"'),
        ('\u{201d}', '"'),
        ('\u{2013}', '-'),
        ('\u{2014}', '-'),
        ('\u{00a0}', ' '),
    ];
    for (from, to) in replacements {
        text = text.replace(from, &to.to_string());
    }

    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_blank_lines() {
        assert_eq!(clean_lyrics("a\n\n  b  "), "a\nb");
    }

    #[test]
    fn removes_bom() {
        assert_eq!(clean_lyrics("\u{feff}Hello World"), "Hello World");
    }

    #[test]
    fn normalizes_typographic_quotes_and_dashes() {
        let lyrics = "Don\u{2019}t stop\u{2014}believin\u{2019}";
        assert_eq!(clean_lyrics(lyrics), "Don't stop-believin'");
    }

    #[test]
    fn normalizes_non_breaking_space() {
        let lyrics = "Hello\u{00a0}World";
        assert_eq!(clean_lyrics(lyrics), "Hello World");
    }

    #[test]
    fn preserves_newlines_between_lines() {
        let lyrics = "Line 1\nLine 2\tTabbed";
        let cleaned = clean_lyrics(lyrics);
        assert!(cleaned.contains('\n'));
        assert!(cleaned.contains('\t'));
        assert_eq!(cleaned, "Line 1\nLine 2\tTabbed");
    }

    #[test]
    fn empty_input_returns_empty() {
        assert_eq!(clean_lyrics(""), "");
    }
}
