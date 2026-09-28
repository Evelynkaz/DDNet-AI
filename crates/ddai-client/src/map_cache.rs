//! The map cache on disk: `<cache_dir>/<name>_<sha256-hex>.map` (task spec acceptance criterion
//! 2) — never re-downloaded once present. Pure path/IO helpers; [`crate::session::Session`]
//! itself never touches a filesystem (sans-IO) — [`crate::driver`] calls these.
//!
//! [`is_valid_map_filename`] ports DDNet's own `str_valid_filename` (`base/str.cpp:199-258`,
//! pinned rev c9d208138f85755521f16a0096b6fe036c5c8698), which the real client runs on every
//! `NETMSG_MAP_CHANGE` name before touching the filesystem with it at all
//! (`client.cpp:1740-1744`: an invalid name disconnects with "map name is not a valid filename")
//! — a hostile server could otherwise try to smuggle a path-traversal or reserved-device name
//! into what becomes part of a filesystem path. This port covers every rule except the two
//! superscript Windows device-name variants (`COM¹`/`LPT¹`, `COM²`/`LPT²`, `COM³`/`LPT³` —
//! `str.cpp:249-251`, a Windows-only legacy alias that could never be produced by a normal `.map`
//! filename and would not be reachable on the Linux-only host this bot runs on either way);
//! everything else — control characters, the shell/Windows-hostile punctuation set, irregular
//! whitespace, leading/trailing/doubled spaces, a trailing period, and the plain-ASCII reserved
//! device names — is ported faithfully.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Plain-ASCII Windows reserved device names `str_valid_filename` also checks
/// (`str.cpp:249-251`, minus the three superscript variants — see the module docs).
const RESERVED_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT0",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// `str_valid_filename` (`base/str.cpp:199-258`) — see the module docs for the one deliberate gap.
pub fn is_valid_map_filename(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let mut prev_space = false;
    let mut prev_period = false;
    let mut first_char_seen = false;

    for c in name.chars() {
        let code = c as u32;
        if code <= 0x1F || code == 0x7F || matches!(c, '\\' | '/' | '|' | ':' | '*' | '?' | '<' | '>' | '"') {
            return false;
        }
        if c.is_whitespace() && c != ' ' {
            return false; // only regular spaces are allowed
        }
        if c == ' ' {
            if !first_char_seen {
                return false; // leading space
            }
            if prev_space {
                return false; // consecutive spaces
            }
            prev_space = true;
            prev_period = false;
        } else {
            prev_space = false;
            prev_period = c == '.';
            first_char_seen = true;
        }
    }

    if prev_space || prev_period {
        return false; // trailing space or period
    }

    // Byte-based on purpose (review finding F1): `name[..reserved.len()]` on a `&str` panics
    // the instant `reserved.len()` lands inside a multi-byte UTF-8 character (e.g. "Blöck" —
    // a real, unremarkable map name) — confirmed live: a `MAP_CHANGE` with such a name killed
    // the driver thread with no `GaveUp` event. Every `RESERVED_NAMES` entry is plain ASCII, so
    // comparing raw bytes (which can never panic, `str`'s byte slicing can) is both safe and
    // exactly equivalent for the only names this check ever needs to match.
    let bytes = name.as_bytes();
    !RESERVED_NAMES.iter().any(|reserved| {
        bytes
            .get(..reserved.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(reserved.as_bytes()))
            && matches!(bytes.get(reserved.len()), None | Some(b'.'))
    })
}

/// Lower-case hex encoding of a sha256 digest, exactly as used in the cache filename.
pub fn sha256_hex(sha256: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in sha256 {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// The full path a map named `name` with hash `sha256` would live at under `cache_dir`. Callers
/// must reject `name` with [`is_valid_map_filename`] first — this function does not re-check it
/// (kept infallible/pure so it can also be used to compute the path for a *known-valid* name
/// without threading a `Result` everywhere).
pub fn cache_path(cache_dir: &Path, name: &str, sha256: &[u8; 32]) -> PathBuf {
    cache_dir.join(format!("{name}_{}.map", sha256_hex(sha256)))
}

/// Reads a cached map's bytes, if present. Never treats a read/verify failure as a hard error —
/// a caller should just fall back to (re)downloading (a corrupt/foreign-write cache file must
/// never wedge the client) — logged by the caller, not here.
pub fn read_cached(cache_dir: &Path, name: &str, sha256: &[u8; 32]) -> Option<Vec<u8>> {
    fs::read(cache_path(cache_dir, name, sha256)).ok()
}

/// Writes `bytes` to the cache under `name`/`sha256`, atomically (write to a sibling temp file,
/// then rename) so a crash or concurrent reader never observes a partially-written cache entry.
pub fn write_cache(cache_dir: &Path, name: &str, sha256: &[u8; 32], bytes: &[u8]) -> io::Result<()> {
    fs::create_dir_all(cache_dir)?;
    let final_path = cache_path(cache_dir, name, sha256);
    let tmp_path = cache_dir.join(format!("{name}_{}.map.tmp-{}", sha256_hex(sha256), std::process::id()));
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, &final_path)
}

/// sha256 of `bytes`, exposed here so callers that only have raw bytes (not yet run through
/// `ddai_map::load_map`) can compute the cache key without pulling in their own `sha2` use —
/// mirrors exactly what `ddai_map::load_map`'s `LoadedMap::sha256` computes (whole-file digest).
pub fn sha256_of(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names_pass() {
        for name in ["Copy Love Box", "BlmapChill", "a", "map.with.dots", "ctf1"] {
            assert!(is_valid_map_filename(name), "{name} should be valid");
        }
    }

    #[test]
    fn empty_name_is_invalid() {
        assert!(!is_valid_map_filename(""));
    }

    #[test]
    fn path_separators_are_rejected() {
        for name in ["../../etc/passwd", "a/b", "a\\b", "..", "."] {
            assert!(!is_valid_map_filename(name), "{name} should be rejected");
        }
    }

    #[test]
    fn control_and_hostile_punctuation_rejected() {
        for name in ["a\0b", "a\x01b", "a|b", "a:b", "a*b", "a?b", "a<b", "a>b", "a\"b"] {
            assert!(!is_valid_map_filename(name), "{name} should be rejected");
        }
    }

    #[test]
    fn leading_trailing_and_double_spaces_rejected() {
        for name in [" leading", "trailing ", "double  space"] {
            assert!(!is_valid_map_filename(name), "{name:?} should be rejected");
        }
    }

    #[test]
    fn trailing_period_rejected() {
        assert!(!is_valid_map_filename("name."));
    }

    #[test]
    fn reserved_device_names_rejected_case_insensitively() {
        for name in ["CON", "con", "NUL", "com1", "LPT9", "con.map"] {
            assert!(!is_valid_map_filename(name), "{name} should be rejected");
        }
        // Not actually reserved: a longer name that merely starts with a reserved prefix.
        assert!(is_valid_map_filename("CONcrete"));
    }

    /// Review finding F1 (blocker): `name[..reserved.len()]` on a `&str` panics the instant
    /// `reserved.len()` (3-4 bytes for every `RESERVED_NAMES` entry) lands inside a multi-byte
    /// UTF-8 character — a completely ordinary, real map name like "Blöck" (ö is 2 bytes, so
    /// byte index 3 sits inside it) used to crash the whole check. These must merely return a
    /// (valid or rejected) answer, never panic.
    #[test]
    fn unicode_names_never_panic_the_reserved_name_check() {
        for name in [
            "Blöck",
            "Ünicode Map",
            "日本語マップ",
            "🎮 Block Party",
            "ö",
            "öö",
            "аб", // 2-byte Cyrillic, exactly reserved.len()-sized prefixes for "CON" etc.
            "CoöN",
        ] {
            let _ = is_valid_map_filename(name); // must not panic
        }
        // None of these happen to collide with a reserved *ASCII* name, so they're all valid.
        assert!(is_valid_map_filename("Blöck"));
        assert!(is_valid_map_filename("日本語マップ"));
    }

    #[test]
    fn cache_path_uses_name_and_hex_sha256() {
        let dir = Path::new("/tmp/whatever");
        let sha = [0xabu8; 32];
        let path = cache_path(dir, "Copy Love Box", &sha);
        assert_eq!(path, dir.join(format!("Copy Love Box_{}.map", "ab".repeat(32))));
    }

    #[test]
    fn write_then_read_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        let sha = sha256_of(b"hello world");
        write_cache(tmp.path(), "Copy Love Box", &sha, b"hello world").unwrap();
        let read_back = read_cached(tmp.path(), "Copy Love Box", &sha).unwrap();
        assert_eq!(read_back, b"hello world");
    }

    #[test]
    fn read_missing_returns_none_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let sha = [0u8; 32];
        assert_eq!(read_cached(tmp.path(), "nope", &sha), None);
    }

    #[test]
    fn a_different_sha256_for_the_same_name_is_a_cache_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let sha_a = sha256_of(b"version a");
        let sha_b = sha256_of(b"version b");
        write_cache(tmp.path(), "Copy Love Box", &sha_a, b"version a").unwrap();
        assert_eq!(read_cached(tmp.path(), "Copy Love Box", &sha_b), None);
        assert_eq!(read_cached(tmp.path(), "Copy Love Box", &sha_a).unwrap(), b"version a");
    }
}
