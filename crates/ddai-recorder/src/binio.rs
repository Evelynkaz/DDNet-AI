//! Minimal little-endian binary primitives for rec v1 (task 8.4a, `docs/formats.md` §16) —
//! matches that document's general convention for this project's own binary formats (raw LE
//! ints, length-prefixed UTF-8 strings, no padding, no compression at this layer — compression is
//! [`crate::writer`]/[`crate::reader`]'s job, one layer up), not a general-purpose serialization
//! framework: rec v1's shape is simple and fixed enough that pulling one in (`serde`+`bincode`/
//! `postcard`) would only add a dependency without buying anything this crate's own tests don't
//! already cover directly. [`Reader`] never panics on truncated/hostile input — every read
//! returns a [`DecodeError`] instead of indexing out of bounds or allocating an unbounded amount
//! from an attacker-controlled length prefix.

/// A conservative sanity cap on any single length-prefixed field (a string, or a raw byte blob)
/// — well above anything this format's own writer ever produces (the longest string rec v1 writes
/// is a chat message or map name, a handful of KiB at most), but far below "attempt to allocate
/// gigabytes because a corrupt length prefix said so".
const MAX_FIELD_LEN: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error("unexpected end of data: need {need} more byte(s), have {have}")]
    Eof { need: usize, have: usize },
    #[error("string field was not valid UTF-8")]
    InvalidUtf8,
    #[error("length-prefixed field claims {len} bytes, over the {cap} sanity cap")]
    LengthTooLarge { len: u64, cap: u64 },
}

/// A plain `Vec<u8>` accumulator — every `push_*` is infallible (writing never fails; only
/// reading a possibly-corrupt file can).
#[derive(Debug, Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer { buf: Vec::new() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn push_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn push_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_bytes32(&mut self, v: &[u8; 32]) {
        self.buf.extend_from_slice(v);
    }

    pub fn push_raw(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// `u32` byte-length prefix + raw UTF-8 bytes (`docs/formats.md`'s general string
    /// convention).
    pub fn push_string(&mut self, s: &str) {
        self.push_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

/// A cursor over a borrowed byte slice — every accessor advances past what it reads and returns
/// `Err(DecodeError::Eof)` rather than panicking if the slice runs out.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::Eof {
                need: n,
                have: self.remaining(),
            });
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    pub fn read_u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("exactly 4 bytes")))
    }

    pub fn read_i32(&mut self) -> Result<i32, DecodeError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().expect("exactly 4 bytes")))
    }

    pub fn read_u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("exactly 8 bytes")))
    }

    pub fn read_bytes32(&mut self) -> Result<[u8; 32], DecodeError> {
        Ok(self.take(32)?.try_into().expect("exactly 32 bytes"))
    }

    pub fn read_raw(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        self.take(n)
    }

    pub fn read_string(&mut self) -> Result<String, DecodeError> {
        let len = u64::from(self.read_u32()?);
        if len > MAX_FIELD_LEN {
            return Err(DecodeError::LengthTooLarge {
                len,
                cap: MAX_FIELD_LEN,
            });
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| DecodeError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_primitive() {
        let mut w = Writer::new();
        w.push_u8(0xAB);
        w.push_u32(0x1234_5678);
        w.push_i32(-42);
        w.push_u64(0x0102_0304_0506_0708);
        w.push_bytes32(&[7u8; 32]);
        w.push_raw(&[1, 2, 3]);
        w.push_string("héllo wörld");
        let bytes = w.into_bytes();

        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_u8().unwrap(), 0xAB);
        assert_eq!(r.read_u32().unwrap(), 0x1234_5678);
        assert_eq!(r.read_i32().unwrap(), -42);
        assert_eq!(r.read_u64().unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(r.read_bytes32().unwrap(), [7u8; 32]);
        assert_eq!(r.read_raw(3).unwrap(), &[1, 2, 3]);
        assert_eq!(r.read_string().unwrap(), "héllo wörld");
        assert!(r.is_empty());
    }

    #[test]
    fn negative_and_boundary_i32_round_trip() {
        for v in [i32::MIN, i32::MAX, 0, -1, 1] {
            let mut w = Writer::new();
            w.push_i32(v);
            let bytes = w.into_bytes();
            let mut r = Reader::new(&bytes);
            assert_eq!(r.read_i32().unwrap(), v);
        }
    }

    #[test]
    fn truncated_data_is_an_eof_error_not_a_panic() {
        let mut r = Reader::new(&[1, 2, 3]);
        let err = r.read_u32().unwrap_err();
        assert_eq!(err, DecodeError::Eof { need: 4, have: 3 });
    }

    #[test]
    fn empty_data_every_reader_errors_cleanly() {
        let mut r = Reader::new(&[]);
        assert!(r.read_u8().is_err());
        let mut r = Reader::new(&[]);
        assert!(r.read_u32().is_err());
        let mut r = Reader::new(&[]);
        assert!(r.read_bytes32().is_err());
        let mut r = Reader::new(&[]);
        assert!(r.read_string().is_err());
    }

    #[test]
    fn invalid_utf8_string_is_an_error_not_a_panic() {
        let mut w = Writer::new();
        w.push_u32(3);
        w.push_raw(&[0xFF, 0xFE, 0xFD]);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_string().unwrap_err(), DecodeError::InvalidUtf8);
    }

    #[test]
    fn absurd_string_length_prefix_is_rejected_not_a_huge_allocation() {
        let mut w = Writer::new();
        w.push_u32(u32::MAX);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        let err = r.read_string().unwrap_err();
        assert!(matches!(err, DecodeError::LengthTooLarge { .. }));
    }

    #[test]
    fn fuzz_like_random_bytes_never_panic_any_reader_method() {
        // Not a proptest dependency for one small property — a fixed, deterministic sweep over
        // short byte strings exercises the same "never panics" property directly.
        for len in 0..40 {
            let data: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();
            let mut r = Reader::new(&data);
            let _ = r.read_u8();
            let mut r = Reader::new(&data);
            let _ = r.read_u32();
            let mut r = Reader::new(&data);
            let _ = r.read_i32();
            let mut r = Reader::new(&data);
            let _ = r.read_u64();
            let mut r = Reader::new(&data);
            let _ = r.read_bytes32();
            let mut r = Reader::new(&data);
            let _ = r.read_string();
        }
    }
}
