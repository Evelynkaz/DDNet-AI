// Ported from DDNet `src/game/gamecore.cpp` (`StrToInts`/`IntsToStr`, pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same byte-for-byte "int string"
// encoding DDNet uses for `CNetObj_ClientInfo`'s name/clan/skin fields.
//
//! DDNet's "int string" encoding: fixed-width names/clans/skins are packed 4 bytes per `i32`
//! (each byte biased by +128, so an all-zero `i32` never collides with a real NUL byte in the
//! middle of a "shorter than the field width" string) rather than as an ordinary
//! length/NUL-terminated string, because [`crate::generated::objects::ClientInfo`] lives inside a
//! snapshot item, which only ever contains plain `i32`s (see `crate::snapshot`), never a separate
//! variable-length string blob.

/// Decodes a fixed-width int-string field back to a `String`, exactly like DDNet's `IntsToStr`.
///
/// The wire's very last byte is always forced to decode as NUL regardless of its actual bits
/// (`IntsToStr`'s own unconditional `pStr[StrIndex - 1] = '\0';`, run *after* the `-128`
/// unbiasing below — this is what makes `StrToInts`' `pInts[NumInts - 1] &= 0xFFFFFF00` safe: that
/// wire byte is always `0x00`, which unbiases to `0x80`/`-128`, not a real NUL, so `IntsToStr`
/// papers over it explicitly rather than relying on the arithmetic). Decoding then stops at the
/// first NUL byte (guaranteed to exist, if nothing earlier) — anything after it is unused padding,
/// matching C-string semantics. If the decoded bytes up to that point are not valid UTF-8 (a
/// corrupt or hostile peer), returns an empty string, matching DDNet's own fallback
/// (`IntsToStr`: `pStr[0] = '\0'; return false;`) — never panics.
pub fn ints_to_str(ints: &[i32]) -> String {
    if ints.is_empty() {
        return String::new();
    }
    let mut bytes = Vec::with_capacity(ints.len() * 4);
    for &v in ints {
        bytes.push((((v >> 24) & 0xff) as u8).wrapping_sub(128));
        bytes.push((((v >> 16) & 0xff) as u8).wrapping_sub(128));
        bytes.push((((v >> 8) & 0xff) as u8).wrapping_sub(128));
        bytes.push(((v & 0xff) as u8).wrapping_sub(128));
    }
    *bytes.last_mut().expect("checked non-empty above") = 0;
    let nul_pos = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    bytes.truncate(nul_pos);
    String::from_utf8(bytes).unwrap_or_default()
}

/// Encodes a `String` into `num_ints` wire ints, exactly like DDNet's `StrToInts`: truncates to
/// `num_ints * 4 - 1` bytes if longer (always leaving room for the forced trailing NUL), pads the
/// rest with zero bytes (which, after the `+128` bias, are stored as `-128`, matching
/// `aBuf[c] = 0` for `c` past the input length in the C++ reference), and always zeroes the low
/// byte of the last int. Used by tests to build round-trip fixtures; not required for decode.
pub fn str_to_ints(s: &str, num_ints: usize) -> Vec<i32> {
    let max_bytes = num_ints * 4;
    let mut bytes = s.as_bytes().to_vec();
    if bytes.len() >= max_bytes {
        bytes.truncate(max_bytes.saturating_sub(1));
    }
    bytes.resize(max_bytes, 0);
    let mut out = Vec::with_capacity(num_ints);
    for i in 0..num_ints {
        let b0 = (bytes[i * 4] as i32).wrapping_add(128) & 0xff;
        let b1 = (bytes[i * 4 + 1] as i32).wrapping_add(128) & 0xff;
        let b2 = (bytes[i * 4 + 2] as i32).wrapping_add(128) & 0xff;
        let b3 = (bytes[i * 4 + 3] as i32).wrapping_add(128) & 0xff;
        out.push((b0 << 24) | (b1 << 16) | (b2 << 8) | b3);
    }
    if let Some(last) = out.last_mut() {
        *last &= 0xFFFFFF00u32 as i32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_short_ascii_name() {
        let ints = str_to_ints("Muha", 4);
        assert_eq!(ints_to_str(&ints), "Muha");
    }

    #[test]
    fn roundtrip_empty_string() {
        let ints = str_to_ints("", 4);
        assert_eq!(ints_to_str(&ints), "");
    }

    #[test]
    fn truncates_to_fit_leaving_room_for_nul() {
        // 4 ints = 16 bytes = 15 usable + forced trailing NUL.
        let ints = str_to_ints("this name is definitely too long", 4);
        let decoded = ints_to_str(&ints);
        assert!(decoded.len() <= 15, "decoded={decoded:?} len={}", decoded.len());
        assert_eq!(decoded, "this name is de");
    }

    #[test]
    fn all_zero_ints_decode_to_empty_string() {
        assert_eq!(ints_to_str(&[0, 0, 0, 0]), "");
    }

    /// Packs `decoded_bytes` (the bytes [`ints_to_str`] should produce, pre-UTF-8-check) into a
    /// single wire int by inverting the `-128` bias, for building test vectors with arbitrary
    /// (including intentionally-invalid-UTF-8) decoded bytes that `str_to_ints` cannot produce
    /// since it only ever accepts an already-valid `&str`.
    fn encode_one_int(decoded_bytes: [u8; 4]) -> i32 {
        let b: Vec<i32> = decoded_bytes
            .iter()
            .map(|&d| (d as i32).wrapping_add(128) & 0xff)
            .collect();
        (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]
    }

    #[test]
    fn invalid_utf8_decodes_to_empty_string_not_panic() {
        // 0xFF is never a valid UTF-8 lead byte on its own.
        let one_int = encode_one_int([0xFF, b'x', b'x', 0]);
        assert_eq!(ints_to_str(&[one_int]), "");
    }

    #[test]
    fn multibyte_utf8_roundtrips() {
        let ints = str_to_ints("Мух", 4); // Cyrillic, 6 bytes + NUL fits in 4 ints (16 bytes)
        assert_eq!(ints_to_str(&ints), "Мух");
    }

    #[test]
    fn never_panics_on_arbitrary_ints() {
        for seed in 0..2000i32 {
            let ints = [seed, seed.wrapping_mul(7), seed.wrapping_add(999), -seed];
            let _ = ints_to_str(&ints); // must not panic regardless of content
        }
    }
}
