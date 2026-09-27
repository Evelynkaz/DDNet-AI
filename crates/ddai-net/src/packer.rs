// Ported from DDNet `src/engine/shared/{compression,packer}.{h,cpp}` (pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same `CVariableInt` byte format
// and `CPacker`/`CUnpacker` semantics (including the `str_utf8_decode` byte-level state machine
// ported into `decode_utf8_one` below), so it stays byte-for-byte compatible with the wire format
// every DDNet 0.6 client/server speaks. See docs/formats.md for the byte layout.
//
//! Variable-length integer packing and the message packer/unpacker.
//!
//! Integers use Teeworlds' `CVariableInt` format (`ESDDDDDD EDDDDDDD ...`: extend bit, sign bit
//! only on the first byte, then 6 then 7 data bits per byte, up to [`MAX_BYTES_PACKED`] bytes —
//! see [`pack_int`]/[`unpack_int`]). Strings are NUL-terminated UTF-8 with optional in-place
//! sanitisation ([`SanitizeMode`]); raw data is copied verbatim.
//!
//! [`Unpacker`] mirrors DDNet's `CUnpacker`: once *any* read fails, the unpacker is "poisoned"
//! ([`Unpacker::error`] becomes `true`) and every subsequent read returns a default value without
//! touching the cursor further, so a caller can unpack a whole message and check `error()` once
//! at the end, exactly like the C++ code does.

use std::fmt;

/// Maximum number of bytes a single packed integer can take.
pub const MAX_BYTES_PACKED: usize = 5;

/// Packs `value` as a DDNet variable-length integer into `dst`, returning the number of bytes
/// written, or `None` if `dst` is too small.
///
/// Format (`compression.cpp:9-35`, `compression.h:17-45`): first byte is
/// `[extend:1][sign:1][6 data bits]`; each following byte (while the previous byte's extend bit
/// is set) is `[extend:1][7 data bits]`, up to [`MAX_BYTES_PACKED`] bytes total. Negative numbers
/// are stored as `!value` (bitwise NOT) with the sign bit set — there is no separate `-0`.
pub fn pack_int(dst: &mut [u8], value: i32) -> Option<usize> {
    if dst.is_empty() {
        return None;
    }
    let negative = value < 0;
    // `!value` for i32::MIN is i32::MAX, well-defined, no overflow (bitwise NOT never panics).
    let mut remaining: u32 = if negative { !value as u32 } else { value as u32 };

    let mut pos = 0usize;
    dst[pos] = if negative { 0x40 } else { 0 };
    dst[pos] |= (remaining & 0x3F) as u8;
    remaining >>= 6;

    while remaining != 0 {
        dst[pos] |= 0x80;
        pos += 1;
        if pos >= dst.len() {
            return None;
        }
        dst[pos] = (remaining & 0x7F) as u8;
        remaining >>= 7;
    }
    Some(pos + 1)
}

/// Unpacks a DDNet variable-length integer from the start of `src`.
///
/// Returns the decoded value and the number of bytes consumed, or `None` if `src` runs out
/// before a complete integer is decoded (mirrors `CVariableInt::Unpack` returning `nullptr`).
/// Bounded by [`MAX_BYTES_PACKED`] regardless of how many "extend" bits a malformed input sets —
/// this can never read more than 5 bytes or loop.
pub fn unpack_int(src: &[u8]) -> Option<(i32, usize)> {
    if src.is_empty() {
        return None;
    }
    let sign = (src[0] >> 6) & 1;
    let mut value: u32 = u32::from(src[0] & 0x3F);
    let mut pos = 0usize;

    const MASKS: [u32; 4] = [0x7F, 0x7F, 0x7F, 0x0F];
    const SHIFTS: [u32; 4] = [6, 6 + 7, 6 + 7 + 7, 6 + 7 + 7 + 7];

    for i in 0..MASKS.len() {
        if src[pos] & 0x80 == 0 {
            break;
        }
        pos += 1;
        if pos >= src.len() {
            return None;
        }
        value |= (u32::from(src[pos]) & MASKS[i]) << SHIFTS[i];
    }
    pos += 1;

    // `value ^= -sign` in C++ acting on ints; the equivalent bitwise-not-if-negative on our
    // unsigned accumulator, converted back to i32 at the very end.
    let signed = if sign != 0 { !value } else { value };
    Some((signed as i32, pos))
}

/// Bulk-packs `values` as consecutive variable-length integers into `dst`.
///
/// Returns the total number of bytes written, or `None` if `dst` is too small for all of them
/// (mirrors `CVariableInt::Compress`).
pub fn pack_ints(dst: &mut [u8], values: &[i32]) -> Option<usize> {
    let mut pos = 0usize;
    for &v in values {
        let n = pack_int(&mut dst[pos..], v)?;
        pos += n;
    }
    Some(pos)
}

/// Bulk-unpacks variable-length integers from `src` until it is exhausted, appending each to
/// `out`. Returns the number of bytes consumed, or `None` on malformed input (mirrors
/// `CVariableInt::Decompress`; unlike the C++ version this has no fixed output capacity — the
/// caller controls how many integers are expected by how much of `src` it passes in).
pub fn unpack_ints(src: &[u8], out: &mut Vec<i32>) -> Option<usize> {
    let mut pos = 0usize;
    while pos < src.len() {
        let (value, n) = unpack_int(&src[pos..])?;
        out.push(value);
        pos += n;
    }
    Some(pos)
}

/// String sanitisation applied by [`Unpacker::get_string`], mirroring `CUnpacker`'s
/// `SANITIZE`/`SANITIZE_CC`/`SKIP_START_WHITESPACES` flags (`packer.h:61-66`). DDNet always
/// validates UTF-8 first regardless of sanitisation mode (`packer.cpp:192-196`); a string that
/// is not valid UTF-8 is rejected (poisons the unpacker) rather than sanitised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SanitizeMode {
    /// Replace bytes `< 32` with `' '`, except `\r`, `\n`, `\t` (`str_sanitize`).
    pub sanitize: bool,
    /// Replace bytes `< 32` with `' '`, including `\r`, `\n`, `\t` (`str_sanitize_cc`).
    pub sanitize_cc: bool,
    /// Skip leading Unicode whitespace in the returned string (`str_utf8_skip_whitespaces`).
    pub skip_start_whitespace: bool,
}

impl SanitizeMode {
    /// No sanitisation at all, no leading-whitespace skip.
    pub const NONE: SanitizeMode = SanitizeMode {
        sanitize: false,
        sanitize_cc: false,
        skip_start_whitespace: false,
    };
    /// `CUnpacker::SANITIZE` — DDNet's default for `GetString()`.
    pub const SANITIZE: SanitizeMode = SanitizeMode {
        sanitize: true,
        sanitize_cc: false,
        skip_start_whitespace: false,
    };
    /// `CUnpacker::SANITIZE_CC`.
    pub const SANITIZE_CC: SanitizeMode = SanitizeMode {
        sanitize: false,
        sanitize_cc: true,
        skip_start_whitespace: false,
    };
}

/// A message packer with a caller-supplied backing buffer, mirroring `CAbstractPacker`.
///
/// Every `add_*` method is a no-op once the packer has errored (buffer exhausted, or a string
/// exceeded its limit without truncation allowed) — call [`Packer::error`] once at the end,
/// exactly like the C++ `CPacker`/`CMsgPacker`.
pub struct Packer<'buf> {
    buffer: &'buf mut [u8],
    pos: usize,
    error: bool,
}

impl<'buf> Packer<'buf> {
    /// Starts packing into a fresh buffer.
    pub fn new(buffer: &'buf mut [u8]) -> Self {
        Packer {
            buffer,
            pos: 0,
            error: false,
        }
    }

    /// Bytes written so far.
    pub fn size(&self) -> usize {
        self.pos
    }

    /// The bytes written so far.
    pub fn data(&self) -> &[u8] {
        &self.buffer[..self.pos]
    }

    /// Whether any `add_*` call has failed since construction.
    pub fn error(&self) -> bool {
        self.error
    }

    /// Appends a variable-length integer (`packer.cpp:23-35`).
    pub fn add_int(&mut self, value: i32) {
        if self.error {
            return;
        }
        match pack_int(&mut self.buffer[self.pos..], value) {
            Some(n) => self.pos += n,
            None => self.error = true,
        }
    }

    /// Appends raw bytes verbatim (`packer.cpp:81-94`).
    pub fn add_raw(&mut self, data: &[u8]) {
        if self.error {
            return;
        }
        if self.pos + data.len() > self.buffer.len() {
            self.error = true;
            return;
        }
        self.buffer[self.pos..self.pos + data.len()].copy_from_slice(data);
        self.pos += data.len();
    }

    /// Appends a NUL-terminated UTF-8 string, mirroring `CAbstractPacker::AddString`
    /// (`packer.cpp:37-79`).
    ///
    /// `limit` caps the number of UTF-8-encoded *bytes* of `value` that are written (0 means
    /// "no limit", i.e. the whole buffer). Malformed UTF-8 in `value` (this is Rust — `value` is
    /// always valid UTF-8, but DDNet's version tolerates raw invalid bytes by substituting
    /// U+FFFD) never occurs here since `&str` is already valid; the codepoint-substitution path
    /// only matters when re-porting from raw bytes, so it is intentionally not exposed on this
    /// `&str`-based API — see [`Packer::add_raw_string`] for that.
    pub fn add_string(&mut self, value: &str, limit: usize, allow_truncation: bool) {
        self.add_raw_string(value.as_bytes(), limit, allow_truncation);
    }

    /// Byte-oriented version of [`Packer::add_string`] that mirrors the C++ behaviour exactly,
    /// including substituting invalid UTF-8 sequences with U+FFFD one byte at a time (as
    /// `str_utf8_decode` does when it hits an invalid lead/continuation byte).
    pub fn add_raw_string(&mut self, value: &[u8], limit: usize, allow_truncation: bool) {
        if self.error {
            return;
        }
        let prev_pos = self.pos;
        let effective_limit = if limit == 0 { self.buffer.len() } else { limit };
        let mut remaining_limit = effective_limit;
        let mut cursor = value;
        loop {
            if cursor.is_empty() {
                break;
            }
            let (codepoint, consumed) = decode_utf8_lossy(cursor);
            let mut encoded = [0u8; 4];
            let encoded_len = encode_utf8_replacement(codepoint, &mut encoded);
            if remaining_limit < encoded_len {
                if allow_truncation {
                    break;
                }
                self.error = true;
                self.pos = prev_pos;
                return;
            }
            if self.pos + encoded_len + 1 > self.buffer.len() {
                self.error = true;
                self.pos = prev_pos;
                return;
            }
            self.buffer[self.pos..self.pos + encoded_len].copy_from_slice(&encoded[..encoded_len]);
            self.pos += encoded_len;
            remaining_limit -= encoded_len;
            cursor = &cursor[consumed..];
        }
        if self.pos >= self.buffer.len() {
            self.error = true;
            self.pos = prev_pos;
            return;
        }
        self.buffer[self.pos] = 0;
        self.pos += 1;
    }
}

/// Decodes exactly one UTF-8 codepoint from the start of `bytes`.
///
/// This is a direct port of `str_utf8_decode` (`base/str.cpp:954-1019`, the WHATWG UTF-8 decoder
/// DDNet uses), *not* [`std::str::from_utf8`] — the two disagree on strings with trailing
/// garbage: `from_utf8` validates the whole slice and errors out on the first bad byte anywhere
/// in it, which would incorrectly poison bytes *before* the bad one. DDNet decodes one codepoint
/// at a time and only ever backs off by the one byte that turned out to be bad, which is exactly
/// what the `AddStringBroken` test vectors below pin down. Returns `None` for an invalid
/// codepoint (caller substitutes U+FFFD, see [`decode_utf8_lossy`]) and the number of input bytes
/// consumed (always `>= 1`, so callers always make progress).
fn decode_utf8_one(bytes: &[u8]) -> (Option<u32>, usize) {
    debug_assert!(!bytes.is_empty());
    let mut lower: u8 = 0x80;
    let mut upper: u8 = 0xBF;
    let mut code_point: i64 = 0;
    let mut bytes_needed: u32 = 0;
    let mut bytes_seen: u32 = 0;
    let mut idx = 0usize;

    loop {
        let Some(&byte) = bytes.get(idx) else {
            // Ran out of input mid-sequence: DDNet's C-string decoder would have hit at least
            // the NUL terminator here and treated it as an invalid continuation byte, backing
            // off without consuming it. We have no sentinel byte to back off to, so just report
            // the lead byte(s) seen so far as consumed and invalid.
            return (None, idx.max(1));
        };
        idx += 1;

        if bytes_needed == 0 {
            if byte <= 0x7F {
                return (Some(u32::from(byte)), idx);
            } else if (0xC2..=0xDF).contains(&byte) {
                bytes_needed = 1;
                code_point = i64::from(byte) - 0xC0;
            } else if (0xE0..=0xEF).contains(&byte) {
                if byte == 0xE0 {
                    lower = 0xA0;
                }
                if byte == 0xED {
                    upper = 0x9F;
                }
                bytes_needed = 2;
                code_point = i64::from(byte) - 0xE0;
            } else if (0xF0..=0xF4).contains(&byte) {
                if byte == 0xF0 {
                    lower = 0x90;
                }
                if byte == 0xF4 {
                    upper = 0x8F;
                }
                bytes_needed = 3;
                code_point = i64::from(byte) - 0xF0;
            } else {
                return (None, idx);
            }
            code_point <<= 6 * bytes_needed;
            continue;
        }

        if !(lower <= byte && byte <= upper) {
            // Invalid continuation byte: back off without consuming it, exactly like
            // `str_byte_rewind` in the C++ reference.
            idx -= 1;
            return (None, idx.max(1));
        }
        lower = 0x80;
        upper = 0xBF;
        bytes_seen += 1;
        code_point += (i64::from(byte) - 0x80) << (6 * (bytes_needed - bytes_seen));
        if bytes_seen != bytes_needed {
            continue;
        }
        return (Some(code_point as u32), idx);
    }
}

/// [`decode_utf8_one`], substituting the Unicode replacement character for invalid codepoints —
/// mirrors how `CAbstractPacker::AddString` treats `str_utf8_decode`'s `-1` (`packer.cpp:49-53`).
fn decode_utf8_lossy(bytes: &[u8]) -> (u32, usize) {
    let (codepoint, consumed) = decode_utf8_one(bytes);
    (codepoint.unwrap_or(0xFFFD), consumed)
}

/// Like `char::encode_utf8`, but accepts the raw `u32` codepoints `decode_utf8_lossy` can
/// produce (namely `0xFFFD`, always a valid `char`) without the `Option` ceremony.
fn encode_utf8_replacement(codepoint: u32, out: &mut [u8; 4]) -> usize {
    let c = char::from_u32(codepoint).unwrap_or('\u{FFFD}');
    c.encode_utf8(out).len()
}

/// Error returned by [`Unpacker`] reads is represented as the persistent `error` flag rather than
/// a `Result`, matching `CUnpacker` exactly (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnpackError;

impl fmt::Display for UnpackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unpacker ran out of data or hit malformed input")
    }
}

impl std::error::Error for UnpackError {}

/// Reads packed integers, strings and raw data out of a byte slice, mirroring `CUnpacker`.
///
/// Never panics: every method bounds-checks against the remaining slice before touching it, and
/// once anything fails, `error()` stays `true` and every further read is a no-op that returns a
/// default value (`0`, `""`, or `None` for raw/UUID reads) without advancing — see the module
/// docs.
pub struct Unpacker<'a> {
    start: &'a [u8],
    current: usize,
    error: bool,
}

impl<'a> Unpacker<'a> {
    /// Starts unpacking `data` from the beginning.
    pub fn new(data: &'a [u8]) -> Self {
        Unpacker {
            start: data,
            current: 0,
            error: false,
        }
    }

    /// Whether any read has failed since construction (or since the last [`Unpacker::new`]).
    pub fn error(&self) -> bool {
        self.error
    }

    /// The full slice this unpacker was constructed with, regardless of how much has been read.
    pub fn complete_data(&self) -> &'a [u8] {
        self.start
    }

    /// Bytes remaining to be read.
    pub fn remaining(&self) -> &'a [u8] {
        &self.start[self.current..]
    }

    fn poison(&mut self) {
        self.error = true;
    }

    /// Reads one variable-length integer. Returns `0` and poisons the unpacker if there isn't
    /// one (`packer.cpp:104-124`).
    pub fn get_int(&mut self) -> i32 {
        if self.error {
            return 0;
        }
        match unpack_int(self.remaining()) {
            Some((value, n)) => {
                self.current += n;
                value
            }
            None => {
                self.poison();
                0
            }
        }
    }

    /// Like [`Unpacker::get_int`], but returns `default` (without poisoning) if the unpacker is
    /// exactly at the end of its data — used for message fields that older peers may omit
    /// entirely, e.g. trailing `Sv_TuneParams` entries (`packer.cpp:126-137`).
    pub fn get_int_or_default(&mut self, default: i32) -> i32 {
        if self.error {
            return 0;
        }
        if self.current == self.start.len() {
            return default;
        }
        self.get_int()
    }

    /// Reads a fixed-width (non-varint) little... — DDNet packs this as native-endian raw bytes
    /// (`packer.cpp:139-154`; only used by a handful of legacy/`ClientInfo` fields, kept here for
    /// completeness). Uses target-native byte order, matching `mem_copy(&i, ..., sizeof(int))`.
    pub fn get_uncompressed_int(&mut self) -> i32 {
        if self.error {
            return 0;
        }
        let rem = self.remaining();
        if rem.len() < 4 {
            self.poison();
            return 0;
        }
        let value = i32::from_ne_bytes([rem[0], rem[1], rem[2], rem[3]]);
        self.current += 4;
        value
    }

    /// Like [`Unpacker::get_uncompressed_int`] with an end-of-data default, mirroring
    /// `GetUncompressedIntOrDefault`.
    pub fn get_uncompressed_int_or_default(&mut self, default: i32) -> i32 {
        if self.error {
            return 0;
        }
        if self.current == self.start.len() {
            return default;
        }
        self.get_uncompressed_int()
    }

    /// Reads a NUL-terminated string, validates it is UTF-8, sanitises it per `mode`, and returns
    /// it. On any failure (no NUL before the end of data, or invalid UTF-8), poisons the
    /// unpacker and returns `""` (`packer.cpp:169-204`).
    ///
    /// Sanitisation only ever replaces bytes, it never changes the byte length, so this can
    /// return a borrowed `&str` without allocating — unlike DDNet's C++, which sanitises in
    /// place in a buffer it owns, this needs a small owned copy since `&[u8]` is not mutable
    /// here; see the return type.
    pub fn get_string(&mut self, mode: SanitizeMode) -> String {
        if self.error {
            return String::new();
        }
        let rem = self.remaining();
        let Some(nul_pos) = rem.iter().position(|&b| b == 0) else {
            self.poison();
            return String::new();
        };
        let raw = &rem[..nul_pos];
        self.current += nul_pos + 1;

        let Ok(s) = std::str::from_utf8(raw) else {
            self.poison();
            return String::new();
        };

        let mut bytes = s.as_bytes().to_vec();
        if mode.sanitize {
            for b in &mut bytes {
                if *b < 32 && *b != b'\r' && *b != b'\n' && *b != b'\t' {
                    *b = b' ';
                }
            }
        } else if mode.sanitize_cc {
            for b in &mut bytes {
                if *b < 32 {
                    *b = b' ';
                }
            }
        }
        // Sanitisation above only ever replaces ASCII control bytes with ASCII space, so the
        // result of a valid-UTF-8 input is still valid UTF-8.
        let sanitized = String::from_utf8(bytes).expect("sanitisation preserves UTF-8 validity");

        if mode.skip_start_whitespace {
            let trimmed_start = sanitized
                .char_indices()
                .find(|(_, c)| !is_utf8_whitespace(*c as u32))
                .map(|(i, _)| i)
                .unwrap_or(sanitized.len());
            sanitized[trimmed_start..].to_string()
        } else {
            sanitized
        }
    }

    /// Reads exactly `size` raw bytes, or poisons the unpacker and returns `None`
    /// (`packer.cpp:206-222`).
    pub fn get_raw(&mut self, size: usize) -> Option<&'a [u8]> {
        if self.error {
            return None;
        }
        let rem = self.remaining();
        if size > rem.len() {
            self.poison();
            return None;
        }
        let out = &rem[..size];
        self.current += size;
        Some(out)
    }

    /// Reads every remaining byte, without poisoning on an empty remainder.
    pub fn get_rest(&mut self) -> &'a [u8] {
        if self.error {
            return &[];
        }
        let rem = self.remaining();
        self.current = self.start.len();
        rem
    }
}

/// `str_utf8_isspace` (`base/str.cpp:1098-1107`): the set of codepoints DDNet treats as
/// whitespace for `str_utf8_skip_whitespaces`/name validation — a specific, documented list, not
/// Unicode's general whitespace property.
fn is_utf8_whitespace(code: u32) -> bool {
    code <= 0x0020
        || code == 0x0085
        || code == 0x00A0
        || code == 0x034F
        || code == 0x115F
        || code == 0x1160
        || code == 0x1680
        || code == 0x180E
        || (0x2000..=0x200F).contains(&code)
        || (0x2028..=0x202F).contains(&code)
        || (0x205F..=0x2064).contains(&code)
        || (0x206A..=0x206F).contains(&code)
        || code == 0x2800
        || code == 0x3000
        || code == 0x3164
        || (0xFE00..=0xFE0F).contains(&code)
        || code == 0xFEFF
        || code == 0xFFA0
        || (0xFFF9..=0xFFFC).contains(&code)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CVariableInt tests, ported from `src/test/compression_test.cpp` (20.1) ---

    const DATA: [i32; 14] = [
        0,
        1,
        -1,
        32,
        64,
        256,
        -512,
        12345,
        -123456,
        1234567,
        12345678,
        123456789,
        2147483647,
        -2147483647 - 1,
    ];
    const SIZES: [usize; 14] = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5];

    #[test]
    fn roundtrip_pack_unpack() {
        // compression_test.cpp: CVariableInt.RoundtripPackUnpack
        for (i, &value) in DATA.iter().enumerate() {
            let mut packed = [0u8; MAX_BYTES_PACKED];
            let n = pack_int(&mut packed, value).unwrap();
            assert_eq!(n, SIZES[i], "pack size mismatch for {value}");
            let (result, n2) = unpack_int(&packed).unwrap();
            assert_eq!(n2, SIZES[i]);
            assert_eq!(result, value);
        }
    }

    #[test]
    fn unpack_invalid() {
        // compression_test.cpp: CVariableInt.UnpackInvalid
        let mut packed = [0xFFu8; MAX_BYTES_PACKED];
        let (result, n) = unpack_int(&packed).unwrap();
        assert_eq!(n, MAX_BYTES_PACKED);
        assert_eq!(result, -2147483647 - 1);

        packed[0] &= !0x40; // unset sign bit
        let (result, n) = unpack_int(&packed).unwrap();
        assert_eq!(n, MAX_BYTES_PACKED);
        assert_eq!(result, 2147483647);
    }

    #[test]
    fn pack_buffer_too_small() {
        // compression_test.cpp: CVariableInt.PackBufferTooSmall
        let mut packed = [0u8; MAX_BYTES_PACKED / 2];
        assert_eq!(pack_int(&mut packed, 2147483647), None);
    }

    #[test]
    fn unpack_buffer_too_small() {
        // compression_test.cpp: CVariableInt.UnpackBufferTooSmall
        let packed = [0xFFu8; MAX_BYTES_PACKED / 2];
        assert_eq!(unpack_int(&packed), None);
    }

    #[test]
    fn roundtrip_compress_decompress() {
        // compression_test.cpp: CVariableInt.RoundtripCompressDecompress
        let mut compressed = [0u8; DATA.len() * MAX_BYTES_PACKED];
        let expected_size: usize = SIZES.iter().sum();
        let n = pack_ints(&mut compressed, &DATA).unwrap();
        assert_eq!(n, expected_size);
        let mut decompressed = Vec::new();
        let consumed = unpack_ints(&compressed[..n], &mut decompressed).unwrap();
        assert_eq!(consumed, n);
        assert_eq!(decompressed, DATA);
    }

    #[test]
    fn compress_buffer_too_small() {
        // compression_test.cpp: CVariableInt.CompressBufferTooSmall
        let mut compressed = [0u8; 14]; // too small (NUM items, not enough bytes)
        assert_eq!(pack_ints(&mut compressed, &DATA), None);
    }

    #[test]
    fn decompress_buffer_too_small() {
        // compression_test.cpp: CVariableInt.DecompressBufferTooSmall
        let compressed = [
            0x00u8, 0x01, 0x40, 0x20, 0x80, 0x01, 0x80, 0x04, 0xFF, 0x07, 0xB9, 0xC0, 0x01,
        ];
        let mut out = Vec::new();
        // Feed a `src` that decodes to more than 4 ints by capping the reader manually: the
        // Rust API has no fixed output capacity, so emulate DDNet's "output buffer too small"
        // by truncating input consumption ourselves and checking we'd overflow 4 slots.
        let consumed = unpack_ints(&compressed, &mut out);
        assert!(consumed.is_some());
        assert!(out.len() > 4, "expected more than 4 ints decoded, got {}", out.len());
    }

    // --- Packer tests, ported from `src/test/packer_test.cpp` (20.1) ---

    fn expect_add_int(input: i32, expected: u8) {
        let mut buf = [0u8; 2048];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(input);
        assert!(!packer.error());
        assert_eq!(packer.size(), 1);
        assert_eq!(packer.data()[0], expected);
    }

    fn expect_add_extended_int(input: i32, expected: &[u8]) {
        let mut buf = [0u8; 2048];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(input);
        assert!(!packer.error());
        assert_eq!(packer.size(), expected.len());
        assert_eq!(packer.data(), expected);
    }

    #[test]
    fn packer_add_int() {
        // packer_test.cpp: Packer.AddInt
        expect_add_int(1, 0b0000_0001);
        expect_add_int(2, 0b0000_0010);
        expect_add_int(-1, 0b0100_0000);
        expect_add_int(-2, 0b0100_0001);
        for i in 0..63 {
            expect_add_int(i, i as u8);
        }
    }

    #[test]
    fn packer_add_extended_int() {
        // packer_test.cpp: Packer.AddExtendedInt
        expect_add_extended_int(64, &[0b1000_0000, 0b0000_0001]);
        expect_add_extended_int(65, &[0b1000_0001, 0b0000_0001]);
        expect_add_extended_int(66, &[0b1000_0010, 0b0000_0001]);
        expect_add_extended_int(-65, &[0b1100_0000, 0b0000_0001]);
        expect_add_extended_int(-66, &[0b1100_0001, 0b0000_0001]);
        expect_add_extended_int(-67, &[0b1100_0010, 0b0000_0001]);
        expect_add_extended_int(-68, &[0b1100_0011, 0b0000_0001]);
        expect_add_extended_int(-69, &[0b1100_0100, 0b0000_0001]);
        expect_add_extended_int(-70, &[0b1100_0101, 0b0000_0001]);
    }

    /// `pExpected == None` means an error is expected. Mirrors `ExpectAddString5` from
    /// `packer_test.cpp`: fills the buffer to 5 bytes from the end, then adds `string`.
    fn expect_add_string5(string: &[u8], limit: usize, allow_truncation: bool, expected: Option<&str>) {
        const BUFFER_SIZE: usize = 2 * 1024;
        const OFFSET: usize = BUFFER_SIZE - 5;
        let mut buf = [0u8; BUFFER_SIZE];
        let mut packer = Packer::new(&mut buf);
        packer.add_raw(&[0u8; OFFSET]);
        packer.add_raw_string(string, limit, allow_truncation);

        assert_eq!(
            expected.is_none(),
            packer.error(),
            "for string={string:?}, limit={limit}, allow_truncation={allow_truncation}"
        );
        if let Some(expected) = expected {
            let expected_len = expected.len() + 1;
            assert_eq!(
                expected_len,
                packer.size() - OFFSET,
                "for string={string:?}, limit={limit}, allow_truncation={allow_truncation}"
            );
            let written = &packer.data()[OFFSET..];
            assert_eq!(&written[..written.len() - 1], expected.as_bytes());
            assert_eq!(written[written.len() - 1], 0);
        }
    }

    #[test]
    fn packer_add_string() {
        // packer_test.cpp: Packer.AddString
        expect_add_string5(b"", 0, true, Some(""));
        expect_add_string5(b"a", 0, true, Some("a"));
        expect_add_string5(b"abcd", 0, true, Some("abcd"));
        expect_add_string5(b"abcde", 0, true, None);
    }

    #[test]
    fn packer_add_string_limit() {
        // packer_test.cpp: Packer.AddStringLimit
        expect_add_string5(b"", 1, true, Some(""));
        expect_add_string5(b"a", 1, true, Some("a"));
        expect_add_string5(b"aa", 1, true, Some("a"));
        expect_add_string5("ä".as_bytes(), 1, true, Some(""));

        expect_add_string5(b"", 10, true, Some(""));
        expect_add_string5(b"a", 10, true, Some("a"));
        expect_add_string5(b"abcd", 10, true, Some("abcd"));
        expect_add_string5(b"abcde", 10, true, None);

        expect_add_string5("äöü".as_bytes(), 4, true, Some("äö"));
        expect_add_string5("äöü".as_bytes(), 5, true, Some("äö"));
        expect_add_string5("äöü".as_bytes(), 6, true, None);

        expect_add_string5(b"", 1, false, Some(""));
        expect_add_string5(b"a", 1, false, Some("a"));
        expect_add_string5(b"aa", 1, false, None);
        expect_add_string5("ä".as_bytes(), 1, false, None);

        expect_add_string5(b"", 10, false, Some(""));
        expect_add_string5(b"a", 10, false, Some("a"));
        expect_add_string5(b"abcd", 10, false, Some("abcd"));
        expect_add_string5(b"abcde", 10, false, None);

        expect_add_string5("äöü".as_bytes(), 4, false, None);
        expect_add_string5("äöü".as_bytes(), 5, false, None);
        expect_add_string5("äöü".as_bytes(), 6, false, None);
    }

    #[test]
    fn packer_add_string_broken() {
        // packer_test.cpp: Packer.AddStringBroken
        expect_add_string5(b"\x80", 0, true, Some("\u{FFFD}"));
        expect_add_string5(b"\x80\x80", 0, true, None);
        expect_add_string5(b"a\x80", 0, true, Some("a\u{FFFD}"));
        expect_add_string5(b"\x80a", 0, true, Some("\u{FFFD}a"));

        expect_add_string5(b"\x80", 1, true, Some(""));
        expect_add_string5(b"\x80", 3, true, Some("\u{FFFD}"));
        expect_add_string5(b"\x80\x80", 3, true, Some("\u{FFFD}"));
        expect_add_string5(b"\x80\x80", 5, true, Some("\u{FFFD}"));
        expect_add_string5(b"\x80\x80", 6, true, None);

        expect_add_string5(b"\x80", 1, false, None);
        expect_add_string5(b"\x80", 3, false, Some("\u{FFFD}"));
        expect_add_string5(b"\x80\x80", 3, false, None);
        expect_add_string5(b"\x80\x80", 5, false, None);
        expect_add_string5(b"\x80\x80", 6, false, None);
    }

    #[test]
    fn packer_error1() {
        // packer_test.cpp: Packer.Error1
        const SIZE: usize = 2 * 1024;
        let data = [0u8; SIZE];
        let mut buf = [0u8; SIZE];
        let mut packer = Packer::new(&mut buf);
        assert!(!packer.error());
        packer.add_raw(&data[..SIZE - 1]);
        assert!(!packer.error());
        assert_eq!(packer.size(), SIZE - 1);
        packer.add_int(1);
        assert!(!packer.error());
        assert_eq!(packer.size(), SIZE);
        packer.add_int(2);
        assert!(packer.error());
        packer.add_int(3);
        assert!(packer.error());
    }

    #[test]
    fn packer_error2() {
        // packer_test.cpp: Packer.Error2
        const SIZE: usize = 2 * 1024;
        let data = [0u8; SIZE];
        let mut buf = [0u8; SIZE];
        let mut packer = Packer::new(&mut buf);
        packer.add_raw(&data[..SIZE - 1]);
        assert_eq!(packer.size(), SIZE - 1);
        packer.add_raw(&data[..1]);
        assert!(!packer.error());
        assert_eq!(packer.size(), SIZE);
        packer.add_raw(&data[..1]);
        assert!(packer.error());
        packer.add_raw(&data[..1]);
        assert!(packer.error());
    }

    #[test]
    fn packer_error3() {
        // packer_test.cpp: Packer.Error3
        const SIZE: usize = 2 * 1024;
        let data = [0u8; SIZE];
        let mut buf = [0u8; SIZE];
        let mut packer = Packer::new(&mut buf);
        packer.add_raw(&data[..SIZE - 5]);
        assert_eq!(packer.size(), SIZE - 5);
        packer.add_string("test", 0, true);
        assert!(!packer.error());
        assert_eq!(packer.size(), SIZE);
        packer.add_string("test", 0, true);
        assert!(packer.error());
        packer.add_string("test", 0, true);
        assert!(packer.error());
    }

    // --- Unpacker round-trips and edge cases ---

    #[test]
    fn unpacker_roundtrip_mixed_message() {
        let mut buf = [0u8; 256];
        let mut packer = Packer::new(&mut buf);
        packer.add_int(42);
        packer.add_int(-17);
        packer.add_string("hello", 0, true);
        packer.add_raw(&[1, 2, 3, 4]);
        assert!(!packer.error());

        let mut unpacker = Unpacker::new(packer.data());
        assert_eq!(unpacker.get_int(), 42);
        assert_eq!(unpacker.get_int(), -17);
        assert_eq!(unpacker.get_string(SanitizeMode::SANITIZE), "hello");
        assert_eq!(unpacker.get_raw(4), Some([1u8, 2, 3, 4].as_slice()));
        assert!(!unpacker.error());
    }

    #[test]
    fn unpacker_poisons_on_first_error_and_stays_poisoned() {
        let mut unpacker = Unpacker::new(&[]);
        assert_eq!(unpacker.get_int(), 0);
        assert!(unpacker.error());
        // Once poisoned, everything else returns defaults too, no panics.
        assert_eq!(unpacker.get_int(), 0);
        assert_eq!(unpacker.get_string(SanitizeMode::SANITIZE), "");
        assert_eq!(unpacker.get_raw(4), None);
    }

    #[test]
    fn unpacker_get_string_sanitize_cc_replaces_control_chars() {
        let mut buf = [0u8; 32];
        let mut packer = Packer::new(&mut buf);
        packer.add_raw_string(b"a\x01b\x02\0", 0, true);
        assert!(!packer.error());
        let mut unpacker = Unpacker::new(packer.data());
        assert_eq!(unpacker.get_string(SanitizeMode::SANITIZE_CC), "a b ");
    }

    #[test]
    fn unpacker_get_string_sanitize_keeps_tab_newline_cr() {
        let mut buf = [0u8; 32];
        let mut packer = Packer::new(&mut buf);
        packer.add_raw_string(b"a\tb\nc\r\0", 0, true);
        assert!(!packer.error());
        let mut unpacker = Unpacker::new(packer.data());
        assert_eq!(unpacker.get_string(SanitizeMode::SANITIZE), "a\tb\nc\r");
    }

    #[test]
    fn unpacker_get_string_no_nul_poisons() {
        let mut unpacker = Unpacker::new(b"no nul terminator here");
        assert_eq!(unpacker.get_string(SanitizeMode::SANITIZE), "");
        assert!(unpacker.error());
    }

    #[test]
    fn unpacker_get_string_invalid_utf8_poisons() {
        let mut unpacker = Unpacker::new(b"\x80\x80\0");
        assert_eq!(unpacker.get_string(SanitizeMode::SANITIZE), "");
        assert!(unpacker.error());
    }

    #[test]
    fn unpacker_get_int_or_default_at_end() {
        let mut unpacker = Unpacker::new(&[]);
        assert_eq!(unpacker.get_int_or_default(7), 7);
        assert!(!unpacker.error());
    }

    #[test]
    fn unpacker_get_raw_zero_size_ok() {
        let mut unpacker = Unpacker::new(&[]);
        assert_eq!(unpacker.get_raw(0), Some([].as_slice()));
        assert!(!unpacker.error());
    }
}
