//! Tiny little-endian binary reader/writer used by the rawmap, scenario and trace formats.
//!
//! Hand-written on purpose (see `docs/formats.md` and the crate root docs): all three formats
//! are simple enough (fixed-width fields, no compression) that a dependency would add more risk
//! (a subtly different encoding than what the C++ oracle writes) than it removes.

use std::fmt;

/// An error reading one of this crate's binary formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// The file is shorter than the format requires at this point.
    UnexpectedEof { context: &'static str },
    /// The file didn't start with the format's magic bytes.
    BadMagic { expected: &'static [u8], actual: Vec<u8> },
    /// The file declares a format version this crate doesn't know how to read.
    UnsupportedVersion { format: &'static str, version: u32 },
    /// A length-prefixed byte string wasn't valid UTF-8.
    InvalidUtf8 { context: &'static str },
    /// A value was outside the range the format allows (e.g. an unknown enum tag, an id/index
    /// out of bounds, or a duplicate where one wasn't allowed).
    InvalidValue { context: &'static str, value: i64 },
    /// The file has extra trailing bytes after the last field the format defines.
    TrailingBytes { extra: usize },
    /// A length-prefixed JSON blob (e.g. a trace's metadata) failed to parse as JSON at all
    /// (distinct from [`FormatError::InvalidUtf8`], which is about the raw bytes not being
    /// UTF-8; this is about well-formed UTF-8 that isn't valid/expected JSON).
    InvalidJson { context: &'static str },
    /// A trace's declared `input_schema`/`state_schema` doesn't match what this crate's reader
    /// actually decodes the body with — the file was written by a different (or differently
    /// configured) version of the format than this code understands.
    SchemaMismatch { context: &'static str },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::UnexpectedEof { context } => write!(f, "unexpected end of file while reading {context}"),
            FormatError::BadMagic { expected, actual } => {
                write!(f, "bad magic: expected {expected:?}, got {actual:?}")
            }
            FormatError::UnsupportedVersion { format, version } => {
                write!(f, "unsupported {format} version {version}")
            }
            FormatError::InvalidUtf8 { context } => write!(f, "invalid UTF-8 in {context}"),
            FormatError::InvalidValue { context, value } => write!(f, "invalid value {value} for {context}"),
            FormatError::TrailingBytes { extra } => write!(f, "{extra} trailing byte(s) after the end of the format"),
            FormatError::InvalidJson { context } => write!(f, "invalid JSON in {context}"),
            FormatError::SchemaMismatch { context } => write!(f, "schema mismatch in {context}"),
        }
    }
}

impl std::error::Error for FormatError {}

/// A cursor over an in-memory byte buffer with little-endian primitive reads.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn take(&mut self, n: usize, context: &'static str) -> Result<&'a [u8], FormatError> {
        if self.pos + n > self.buf.len() {
            return Err(FormatError::UnexpectedEof { context });
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn bytes(&mut self, n: usize, context: &'static str) -> Result<&'a [u8], FormatError> {
        self.take(n, context)
    }

    pub fn expect_magic(&mut self, magic: &'static [u8]) -> Result<(), FormatError> {
        let actual = self.take(magic.len(), "magic")?;
        if actual != magic {
            return Err(FormatError::BadMagic {
                expected: magic,
                actual: actual.to_vec(),
            });
        }
        Ok(())
    }

    pub fn u8(&mut self, context: &'static str) -> Result<u8, FormatError> {
        Ok(self.take(1, context)?[0])
    }

    pub fn u16(&mut self, context: &'static str) -> Result<u16, FormatError> {
        let b = self.take(2, context)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn i16(&mut self, context: &'static str) -> Result<i16, FormatError> {
        Ok(self.u16(context)? as i16)
    }

    pub fn u32(&mut self, context: &'static str) -> Result<u32, FormatError> {
        let b = self.take(4, context)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i32(&mut self, context: &'static str) -> Result<i32, FormatError> {
        Ok(self.u32(context)? as i32)
    }

    pub fn f32(&mut self, context: &'static str) -> Result<f32, FormatError> {
        Ok(f32::from_bits(self.u32(context)?))
    }

    pub fn f64(&mut self, context: &'static str) -> Result<f64, FormatError> {
        let b = self.take(8, context)?;
        Ok(f64::from_bits(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])))
    }

    /// Reads a `u16`-length-prefixed UTF-8 string.
    pub fn string16(&mut self, context: &'static str) -> Result<String, FormatError> {
        let len = self.u16(context)? as usize;
        let bytes = self.take(len, context)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| FormatError::InvalidUtf8 { context })
    }

    /// Reads a `u32`-length-prefixed UTF-8 string.
    pub fn string32(&mut self, context: &'static str) -> Result<String, FormatError> {
        let len = self.u32(context)? as usize;
        let bytes = self.take(len, context)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| FormatError::InvalidUtf8 { context })
    }

    pub fn array32<const N: usize>(&mut self, context: &'static str) -> Result<[u8; N], FormatError> {
        let b = self.take(N, context)?;
        let mut out = [0u8; N];
        out.copy_from_slice(b);
        Ok(out)
    }

    /// Number of bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Errors if the file has trailing bytes past what the format defined.
    pub fn expect_eof(&self) -> Result<(), FormatError> {
        if self.remaining() != 0 {
            return Err(FormatError::TrailingBytes {
                extra: self.remaining(),
            });
        }
        Ok(())
    }
}

/// An append-only little-endian byte buffer builder.
#[derive(Default)]
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

    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(b);
        self
    }

    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.buf.push(v);
        self
    }

    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn i16(&mut self, v: i16) -> &mut Self {
        self.u16(v as u16)
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.u32(v as u32)
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.u32(v.to_bits())
    }

    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
        self
    }

    pub fn string16(&mut self, s: &str) -> &mut Self {
        let bytes = s.as_bytes();
        assert!(
            bytes.len() <= u16::MAX as usize,
            "string too long for a u16 length prefix"
        );
        self.u16(bytes.len() as u16);
        self.bytes(bytes)
    }

    pub fn string32(&mut self, s: &str) -> &mut Self {
        let bytes = s.as_bytes();
        assert!(
            bytes.len() <= u32::MAX as usize,
            "string too long for a u32 length prefix"
        );
        self.u32(bytes.len() as u32);
        self.bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_all_primitives() {
        let mut w = Writer::new();
        w.u8(7)
            .u16(1234)
            .i16(-5)
            .u32(0xdead_beef)
            .i32(-42)
            .f32(1.5)
            .f64(-2.25)
            .string16("hi")
            .string32("world");
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u8("a").unwrap(), 7);
        assert_eq!(r.u16("b").unwrap(), 1234);
        assert_eq!(r.i16("c").unwrap(), -5);
        assert_eq!(r.u32("d").unwrap(), 0xdead_beef);
        assert_eq!(r.i32("e").unwrap(), -42);
        assert_eq!(r.f32("f").unwrap(), 1.5);
        assert_eq!(r.f64("g").unwrap(), -2.25);
        assert_eq!(r.string16("h").unwrap(), "hi");
        assert_eq!(r.string32("i").unwrap(), "world");
        r.expect_eof().unwrap();
    }

    #[test]
    fn f32_round_trips_bit_exact_including_nan_payload() {
        let v = f32::from_bits(0x7fc0_1234);
        let mut w = Writer::new();
        w.f32(v);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.f32("v").unwrap().to_bits(), v.to_bits());
    }

    #[test]
    fn unexpected_eof_is_reported() {
        let bytes = [1u8, 2];
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u32("x"), Err(FormatError::UnexpectedEof { context: "x" }));
    }

    #[test]
    fn bad_magic_is_reported() {
        let bytes = *b"XXXX";
        let mut r = Reader::new(&bytes);
        assert_eq!(
            r.expect_magic(b"YYYY"),
            Err(FormatError::BadMagic {
                expected: b"YYYY",
                actual: b"XXXX".to_vec()
            })
        );
    }

    #[test]
    fn trailing_bytes_are_reported() {
        let bytes = [1u8, 2, 3];
        let r = Reader::new(&bytes);
        assert_eq!(r.expect_eof(), Err(FormatError::TrailingBytes { extra: 3 }));
    }
}
