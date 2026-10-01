//! Player-name normalisation for the friend/war/ignore lists (D-021: exact matching after
//! normalisation, replacing the TS substring bug — `listed()`, `bot.ts:276-286`, `docs/research/
//! orig-bot.md` §7.3/§13.5).

/// The folded form of a name or clan used as a list key: trimmed, internal whitespace collapsed to
/// single spaces, lower-cased (`foldName`, `bot.ts:299`) and stripped of DDNet's duplicate-name
/// prefix `^\(\d+\)` (the server renames a second "nick" to "(1)nick", `DUPLICATE_PREFIX`,
/// `bot.ts:288`). The empty string matches nothing (`listed()` returns false for `nameKey === ""`).
pub fn fold_name(name: &str) -> String {
    let mut out = String::new();
    fold_name_into(name, &mut out);
    out
}

/// [`fold_name`] into a reused buffer (cleared first).
pub fn fold_name_into(name: &str, out: &mut String) {
    out.clear();
    let mut first = true;
    for word in name.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        out.push_str(word);
    }
    // `to_lowercase` allocates per call; names change rarely (see `PlayerTable`), so this is off the
    // per-snapshot path.
    let lowered = out.to_lowercase();
    out.clear();
    out.push_str(strip_duplicate_prefix(&lowered));
    let trimmed_len = out.trim_end().len();
    out.truncate(trimmed_len);
    let lead = out.len() - out.trim_start().len();
    if lead > 0 {
        out.drain(..lead);
    }
}

/// Strips one leading `(<digits>)` (the duplicate-name prefix), nothing else.
fn strip_duplicate_prefix(s: &str) -> &str {
    let Some(rest) = s.strip_prefix('(') else { return s };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return s;
    }
    match rest[digits..].strip_prefix(')') {
        Some(after) => after,
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_trims_collapses_and_lowercases() {
        assert_eq!(fold_name("  Some   Nick  "), "some nick");
        assert_eq!(fold_name("ÄÖ"), "äö");
        assert_eq!(fold_name("\tTab\u{a0}Nbsp"), "tab nbsp");
    }

    #[test]
    fn fold_strips_the_duplicate_prefix_only() {
        assert_eq!(fold_name("(1)nick"), "nick");
        assert_eq!(fold_name("(12) Nick"), "nick");
        assert_eq!(fold_name("(x)nick"), "(x)nick", "not digits: kept");
        assert_eq!(fold_name("()nick"), "()nick", "no digits: kept");
        assert_eq!(fold_name("(1"), "(1", "unclosed: kept");
        assert_eq!(fold_name("a(1)"), "a(1)", "only a leading prefix");
        // Only one prefix is stripped, like the regex.
        assert_eq!(fold_name("(1)(2)nick"), "(2)nick");
    }

    #[test]
    fn empty_and_prefix_only_names_fold_to_nothing() {
        assert_eq!(fold_name(""), "");
        assert_eq!(fold_name("   "), "");
        assert_eq!(fold_name("(3)"), "");
    }

    #[test]
    fn fold_into_reuses_the_buffer() {
        let mut buf = String::from("stale content");
        fold_name_into("  X  Y ", &mut buf);
        assert_eq!(buf, "x y");
    }
}
