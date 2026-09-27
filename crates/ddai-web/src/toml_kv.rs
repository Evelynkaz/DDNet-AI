//! A tiny, purpose-built reader/writer for the flat `key = "string"` / `key = 123` files this
//! crate uses for secrets (`web-auth.toml`, `web-session-key.toml`).
//!
//! Real TOML supports tables, arrays, dates, multi-line strings, and a lot more escaping than we
//! need. Pulling in a full TOML crate (plus its own dependency/advisory surface) for a handful of
//! scalar fields that only this crate ever writes and reads would be disproportionate. This is
//! deliberately not a general-purpose TOML parser: it accepts exactly the subset this crate
//! itself produces (one `key = value` pair per line, `#` line comments, blank lines), and callers
//! never feed it arbitrary user-authored TOML.

use std::collections::HashMap;
use std::fmt;

/// A scalar value this tiny writer can emit.
#[derive(Debug, Clone, Copy)]
pub enum Value<'a> {
    Str(&'a str),
    Int(i64),
}

/// Renders `pairs` as `key = value` lines, one per pair, in order, prefixed by `header_comment`
/// (each line of which is emitted as its own `#`-comment line).
pub fn write_kv(header_comment: &[&str], pairs: &[(&str, Value<'_>)]) -> String {
    let mut out = String::new();
    for line in header_comment {
        out.push_str("# ");
        out.push_str(line);
        out.push('\n');
    }
    if !header_comment.is_empty() {
        out.push('\n');
    }
    for (key, value) in pairs {
        out.push_str(key);
        out.push_str(" = ");
        match value {
            Value::Str(s) => {
                out.push('"');
                out.push_str(&escape_str(s));
                out.push('"');
            }
            Value::Int(i) => {
                out.push_str(&i.to_string());
            }
        }
        out.push('\n');
    }
    out
}

fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn unescape_str(s: &str) -> Result<String, ParseError> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            _ => return Err(ParseError::BadEscape),
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("line {0}: expected 'key = value'")]
    MissingEquals(usize),
    #[error("line {0}: unterminated string literal")]
    UnterminatedString(usize),
    #[error("invalid backslash escape in string literal")]
    BadEscape,
}

/// Parses the subset of TOML that [`write_kv`] produces: comment/blank lines are skipped, every
/// other line must be `key = "string"` or `key = <integer>`. Values keep their raw textual form
/// (quotes stripped and escapes resolved for strings); callers convert to the type they expect
/// with [`Kv::str`]/[`Kv::int`].
pub fn parse_kv(text: &str) -> Result<Kv, ParseError> {
    let mut map = HashMap::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(ParseError::MissingEquals(idx + 1));
        };
        let key = key.trim().to_string();
        let value = value.trim();
        let parsed = if let Some(inner) = value.strip_prefix('"') {
            let Some(inner) = inner.strip_suffix('"') else {
                return Err(ParseError::UnterminatedString(idx + 1));
            };
            unescape_str(inner)?
        } else {
            value.to_string()
        };
        map.insert(key, parsed);
    }
    Ok(Kv(map))
}

/// The result of [`parse_kv`]: a flat map of field name to its (already unescaped) textual value.
#[derive(Debug, Default)]
pub struct Kv(HashMap<String, String>);

impl Kv {
    pub fn str(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    pub fn int(&self, key: &str) -> Option<i64> {
        self.0.get(key).and_then(|v| v.parse().ok())
    }
}

impl fmt::Display for Kv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Kv({} fields)", self.0.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_strings_and_ints() {
        let text = write_kv(
            &["managed by tests", "do not edit"],
            &[
                ("hash", Value::Str("$argon2id$v=19$m=65536,t=3,p=1$abc$def")),
                ("m_cost_kib", Value::Int(65536)),
                ("t_cost", Value::Int(3)),
            ],
        );
        let kv = parse_kv(&text).expect("parse");
        assert_eq!(kv.str("hash"), Some("$argon2id$v=19$m=65536,t=3,p=1$abc$def"));
        assert_eq!(kv.int("m_cost_kib"), Some(65536));
        assert_eq!(kv.int("t_cost"), Some(3));
    }

    #[test]
    fn roundtrips_escaped_characters() {
        let text = write_kv(&[], &[("k", Value::Str("back\\slash \"quote\" tab\tnewline\n"))]);
        let kv = parse_kv(&text).expect("parse");
        assert_eq!(kv.str("k"), Some("back\\slash \"quote\" tab\tnewline\n"));
    }

    #[test]
    fn skips_comments_and_blank_lines() {
        let kv = parse_kv("# a comment\n\n  \nkey = \"value\"\n").expect("parse");
        assert_eq!(kv.str("key"), Some("value"));
    }

    #[test]
    fn rejects_missing_equals() {
        let err = parse_kv("not_a_pair").unwrap_err();
        assert_eq!(err, ParseError::MissingEquals(1));
    }

    #[test]
    fn rejects_unterminated_string() {
        let err = parse_kv("k = \"unterminated").unwrap_err();
        assert_eq!(err, ParseError::UnterminatedString(1));
    }

    #[test]
    fn missing_key_returns_none() {
        let kv = parse_kv("k = \"v\"").expect("parse");
        assert_eq!(kv.str("nope"), None);
        assert_eq!(kv.int("nope"), None);
    }
}
