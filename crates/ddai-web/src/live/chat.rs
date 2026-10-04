//! The chat panel's data (task 5.10): one line of the game server's chat as the web unit keeps and shows it.
//!
//! **Read-only and hostile.** Everything in a line was typed by some player (or by the server), so it is treated as
//! hostile input twice over: here every control character, line break, bidirectional override and zero-width character
//! is removed and the lengths are capped (a line cannot reorder the text around it, hide itself or flood the panel), and
//! the page only ever writes it with `textContent`, never into markup (`assets/game.js`, checked by a test). Markup
//! characters are *not* mangled here: `<b>` arrives as the six characters it is, and shows as those.
//!
//! **Not stored.** Lines live only in a bounded in-memory ring ([`ChatRing`], [`RING_LINES`] lines) in the web process;
//! nothing is written to disk or to the log, and the ring is emptied when the source changes. The bot never writes
//! chat (D-007; the only exception is the typed `/kill`, D-078): this is display only.

use std::collections::VecDeque;

/// Lines the web process remembers for a browser that connects (or reconnects) later.
pub const RING_LINES: usize = 200;
/// Longest text and name, in characters, that is kept (DDNet's own limit for a chat line is 256).
pub const MAX_TEXT_CHARS: usize = 256;
pub const MAX_NAME_CHARS: usize = 32;

/// Who a line is from and to (`CNetMsg_Sv_Chat::m_Team` and the client id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatKind {
    All,
    Team,
    WhisperSent,
    WhisperReceived,
    /// From the server itself (client id -1).
    System,
}

impl ChatKind {
    /// The word on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            ChatKind::All => "all",
            ChatKind::Team => "team",
            ChatKind::WhisperSent => "whisper_to",
            ChatKind::WhisperReceived => "whisper_from",
            ChatKind::System => "system",
        }
    }
}

/// One chat line, already sanitised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatLine {
    pub kind: ChatKind,
    /// The sender's client id, `None` for the server.
    pub id: Option<u8>,
    pub name: String,
    pub text: String,
    /// When the web unit received it, milliseconds since the Unix epoch.
    pub at_ms: u64,
}

/// Characters a chat line may not contain: controls (including `\n`, `\r`, `\t`, NUL, DEL, C1), line and paragraph
/// separators, soft hyphen, zero-width and directional marks (U+200B..U+200F), bidirectional embeddings and overrides
/// (U+202A..U+202E), word joiner and invisible operators (U+2060..U+2064), bidirectional isolates (U+2066..U+2069),
/// the byte order mark, and the interlinear annotation and object replacement characters.
fn is_forbidden(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{00AD}'
                | '\u{200B}'..='\u{200F}'
                | '\u{2028}'..='\u{202E}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{2069}'
                | '\u{FEFF}'
                | '\u{FFF9}'..='\u{FFFC}'
        )
}

/// `s` without forbidden characters, cut to `max` characters, with runs of spaces collapsed and the ends trimmed.
pub fn clean(s: &str, max: usize) -> String {
    let mut out = String::with_capacity(s.len().min(max * 4));
    let mut count = 0usize;
    let mut last_space = true; // drops leading spaces
    for c in s.chars() {
        if is_forbidden(c) {
            continue;
        }
        let space = c.is_whitespace();
        if space && last_space {
            continue;
        }
        last_space = space;
        out.push(if space { ' ' } else { c });
        count += 1;
        if count >= max {
            break;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

impl ChatLine {
    /// Builds a line from the bot's `CHAT` message fields. `None` for a line with nothing left to show after cleaning.
    pub fn from_bridge(team: i32, cid: i32, name: &str, text: &str, at_ms: u64) -> Option<ChatLine> {
        let text = clean(text, MAX_TEXT_CHARS);
        if text.is_empty() {
            return None;
        }
        let kind = if cid < 0 {
            ChatKind::System
        } else {
            match team {
                1 => ChatKind::Team,
                2 => ChatKind::WhisperSent,
                3 => ChatKind::WhisperReceived,
                _ => ChatKind::All,
            }
        };
        Some(ChatLine {
            kind,
            id: u8::try_from(cid).ok(),
            name: if cid < 0 {
                String::new()
            } else {
                clean(name, MAX_NAME_CHARS)
            },
            text,
            at_ms,
        })
    }
}

/// The last [`RING_LINES`] lines, in memory only.
#[derive(Debug, Default)]
pub struct ChatRing {
    lines: VecDeque<ChatLine>,
}

impl ChatRing {
    pub fn push(&mut self, line: ChatLine) {
        if self.lines.len() >= RING_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    pub fn snapshot(&self) -> Vec<ChatLine> {
        self.lines.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markup_is_kept_as_plain_characters_not_mangled_and_not_dropped() {
        let hostile = r#"<img src=x onerror=alert(1)><script>alert("x")</script>&amp; ' " `"#;
        assert_eq!(clean(hostile, 256), hostile);
    }

    #[test]
    fn controls_line_breaks_and_direction_tricks_are_removed() {
        // NUL, ESC, newline, CR, tab, DEL, a C1 control, RLO, LRI, zero-width space, BOM, soft hyphen, line separator.
        let evil = "a\u{0}b\u{1b}c\nd\re\tf\u{7f}g\u{85}h\u{202e}i\u{2066}j\u{200b}k\u{feff}l\u{ad}m\u{2028}n";
        assert_eq!(clean(evil, 256), "abcdefghijklmn");
        // Spaces collapse and the ends are trimmed; a tab or newline is a separator-free deletion, not a space.
        assert_eq!(clean("  a   b  ", 256), "a b");
        assert_eq!(clean("\u{202e}\u{202c}", 256), "");
    }

    #[test]
    fn length_is_counted_in_characters_never_cutting_one() {
        let long = "я".repeat(1000);
        let out = clean(&long, MAX_TEXT_CHARS);
        assert_eq!(out.chars().count(), MAX_TEXT_CHARS);
        assert!(out.chars().all(|c| c == 'я'));
        // A cut that lands after a space does not leave a trailing space.
        assert_eq!(clean("ab cd", 3), "ab");
    }

    #[test]
    fn a_line_is_classified_and_the_server_has_no_name() {
        let l = ChatLine::from_bridge(0, 5, "c5-aaaa", "hi", 7).unwrap();
        assert_eq!(
            (l.kind, l.id, l.name.as_str(), l.text.as_str(), l.at_ms),
            (ChatKind::All, Some(5), "c5-aaaa", "hi", 7)
        );
        assert_eq!(ChatLine::from_bridge(1, 5, "n", "t", 0).unwrap().kind, ChatKind::Team);
        assert_eq!(
            ChatLine::from_bridge(2, 5, "n", "t", 0).unwrap().kind,
            ChatKind::WhisperSent
        );
        assert_eq!(
            ChatLine::from_bridge(3, 5, "n", "t", 0).unwrap().kind,
            ChatKind::WhisperReceived
        );
        let sys = ChatLine::from_bridge(0, -1, "ignored name", "Kill Protection enabled", 0).unwrap();
        assert_eq!((sys.kind, sys.id, sys.name.as_str()), (ChatKind::System, None, ""));
        // An id outside 0..=255 has no sender id (and is not a panic).
        assert_eq!(ChatLine::from_bridge(0, 100000, "n", "t", 0).unwrap().id, None);
        // A line that is nothing but forbidden characters is dropped.
        assert!(ChatLine::from_bridge(0, 1, "n", "\u{200b}\n\u{202e}", 0).is_none());
    }

    #[test]
    fn the_ring_keeps_only_the_newest_lines_and_clears() {
        let mut ring = ChatRing::default();
        assert!(ring.is_empty());
        for i in 0..RING_LINES + 25 {
            ring.push(ChatLine::from_bridge(0, 1, "n", &format!("line {i}"), i as u64).unwrap());
        }
        assert_eq!(ring.len(), RING_LINES);
        let lines = ring.snapshot();
        assert_eq!(lines.first().unwrap().text, "line 25");
        assert_eq!(lines.last().unwrap().text, format!("line {}", RING_LINES + 24));
        ring.clear();
        assert!(ring.is_empty());
    }
}
