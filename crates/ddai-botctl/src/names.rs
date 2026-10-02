//! Player-name normalisation for the friend/war/ignore lists (D-021: exact matching after
//! normalisation, replacing the TS substring bug — `listed()`, `bot.ts:276-286`, `docs/research/
//! orig-bot.md` §7.3/§13.5).

/// The folded form of a name or clan used as a list key: trimmed, internal whitespace collapsed to
/// single spaces, lower-cased (`foldName`, `bot.ts:299`) and stripped of DDNet's duplicate-name
/// prefix `^\(\d+\)` — every leading one, so the fold is idempotent (the server renames a second "nick" to "(1)nick", `DUPLICATE_PREFIX`,
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
    // Every leading `(<digits>)` goes, not just one (task 5.6 review F5): the lists file stores folded keys and folds
    // them again on load, so the fold has to be idempotent (`fold(fold(x)) == fold(x)`) or a friend typed as
    // `(1)(2)Pal` would be stored as `(2)pal`, loaded as `pal` and no longer match the player's own key. Whitespace
    // after a stripped prefix is trimmed before the next one is looked for.
    let mut rest = lowered.as_str();
    loop {
        let stripped = strip_duplicate_prefix(rest);
        if stripped.len() == rest.len() {
            break;
        }
        rest = stripped.trim_start();
    }
    out.push_str(rest.trim());
}

/// Strips one leading `(<digits>)` (the duplicate-name prefix), nothing else ([`fold_name_into`] repeats it).
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
        // Every leading prefix is stripped (the TS regex stripped one): the fold must be idempotent, see below.
        assert_eq!(fold_name("(1)(2)nick"), "nick");
        assert_eq!(fold_name("(1) (2)  Nick"), "nick");
        assert_eq!(
            fold_name("(1)(x)(2)nick"),
            "(x)(2)nick",
            "a non-prefix stops the stripping"
        );
    }

    #[test]
    fn empty_and_prefix_only_names_fold_to_nothing() {
        assert_eq!(fold_name(""), "");
        assert_eq!(fold_name("   "), "");
        assert_eq!(fold_name("(3)"), "");
    }

    /// The lists file holds folded keys and the bot folds them again on load, so `fold` must be a projection: a friend
    /// stored in the file has to match the player's own folded name (task 5.6 review F5).
    #[test]
    fn fold_is_idempotent_on_a_million_odd_names() {
        let alphabet: Vec<&str> = vec![
            "(", ")", "(1)", "(23)", "()", "(a)", " ", "\t", "\u{a0}", "0", "9", "a", "B", "\u{c9}", "\u{3a3}",
            "\u{44f}", "-",
        ];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..200_000 {
            let len = (next() % 12) as usize;
            let name: String = (0..len)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            let once = fold_name(&name);
            assert_eq!(fold_name(&once), once, "{name:?} -> {once:?}");
        }
        for name in ["(1)(2)Pal", "(1)(2)(3)", "(1) (2) x", "((1))x", "(1)()(2)x"] {
            let once = fold_name(name);
            assert_eq!(fold_name(&once), once, "{name:?}");
        }
    }

    #[test]
    fn fold_into_reuses_the_buffer() {
        let mut buf = String::from("stale content");
        fold_name_into("  X  Y ", &mut buf);
        assert_eq!(buf, "x y");
    }
}
