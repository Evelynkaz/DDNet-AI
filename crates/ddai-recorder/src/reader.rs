//! The exact inverse of [`crate::writer::RecordingWriter`] — see that module's docs for the file
//! layout. [`RecordingReader::next_frame`] streams frames back out one at a time, decompressing
//! (and sha256-checking) one chunk at a time rather than loading the whole file into memory —
//! this matters for a Swarfey session that could run for hours.

use crate::binio::{self, DecodeError};
use crate::format::{FormatError, Frame, Header};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// A sanity cap on the header's own claimed length — this project's own writer never produces a
/// header anywhere near this size (a handful of short strings); guards a corrupt/hostile leading
/// `u32` from driving an oversized allocation before a single byte of the header is even parsed.
const MAX_HEADER_LEN: usize = 1024 * 1024;
/// Same idea for one chunk's *uncompressed* length — `crate::writer`'s own flush thresholds keep
/// a real chunk far under this; a corrupt length prefix must not drive a huge allocation either.
const MAX_CHUNK_UNCOMPRESSED_LEN: usize = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: io::Error,
    },
    #[error("truncated rec v1 file: {0}")]
    Truncated(String),
    #[error("header length {0} exceeds the sanity cap ({MAX_HEADER_LEN})")]
    HeaderTooLarge(usize),
    #[error("chunk uncompressed length {0} exceeds the sanity cap ({MAX_CHUNK_UNCOMPRESSED_LEN})")]
    ChunkTooLarge(usize),
    #[error("chunk failed sha256 verification (file corrupt or truncated)")]
    ChunkHashMismatch,
    #[error("zstd decompression failed: {0}")]
    Zstd(io::Error),
    #[error("decompressed chunk length {actual} does not match the header's claimed {expected}")]
    ChunkLengthMismatch { expected: usize, actual: usize },
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Decode(#[from] DecodeError),
}

/// Reads exactly `buf.len()` bytes, but distinguishes a *clean* EOF (nothing at all left —
/// `Ok(false)`) from a *truncated* one (some, but not all, of the requested bytes were
/// available — an error): the difference between "the file ends here, as expected" and "the file
/// is corrupt/truncated mid-record".
fn read_exact_or_clean_eof(r: &mut impl Read, buf: &mut [u8]) -> io::Result<bool> {
    let mut total = 0;
    while total < buf.len() {
        match r.read(&mut buf[total..])? {
            0 if total == 0 => return Ok(false),
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated rec v1 file (mid-record)",
                ));
            }
            n => total += n,
        }
    }
    Ok(true)
}

#[derive(Debug)]
pub struct RecordingReader {
    path: std::path::PathBuf,
    file: BufReader<File>,
    header: Header,
    /// The current chunk's decompressed bytes plus a cursor into them — `None` once exhausted
    /// (the next [`RecordingReader::next_frame`] call reads a fresh chunk header).
    current_chunk: Option<(Vec<u8>, usize)>,
}

impl RecordingReader {
    pub fn open(path: &Path) -> Result<Self, ReaderError> {
        let io_err = |source: io::Error| ReaderError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = BufReader::new(File::open(path).map_err(io_err)?);

        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)
            .map_err(|_| ReaderError::Truncated("file ends before a complete header length prefix".to_string()))?;
        let header_len = u32::from_le_bytes(len_buf) as usize;
        if header_len > MAX_HEADER_LEN {
            return Err(ReaderError::HeaderTooLarge(header_len));
        }
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|_| ReaderError::Truncated("file ends before a complete header".to_string()))?;
        let mut r = binio::Reader::new(&header_bytes);
        let header = Header::decode(&mut r)?;

        Ok(RecordingReader {
            path: path.to_path_buf(),
            file,
            header,
            current_chunk: None,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Returns the next [`Frame`] in the recording, in exactly the order
    /// [`crate::writer::RecordingWriter::write_frame`] was called, or `Ok(None)` at a clean
    /// end-of-file (every chunk fully consumed, nothing truncated).
    pub fn next_frame(&mut self) -> Result<Option<Frame>, ReaderError> {
        loop {
            if let Some((buf, pos)) = &mut self.current_chunk {
                if *pos < buf.len() {
                    let mut r = binio::Reader::new(&buf[*pos..]);
                    let tag = r.read_u8()?;
                    let len = r.read_u32()? as usize;
                    let body = r.read_raw(len)?;
                    let frame = Frame::decode_body(tag, body)?;
                    *pos += 1 + 4 + len;
                    return Ok(Some(frame));
                }
                self.current_chunk = None;
            }

            match self.load_next_chunk()? {
                Some(bytes) => self.current_chunk = Some((bytes, 0)),
                None => return Ok(None),
            }
        }
    }

    /// Loads and verifies one chunk (length prefixes, sha256, zstd), or `Ok(None)` at a clean EOF
    /// before the next chunk even starts.
    fn load_next_chunk(&mut self) -> Result<Option<Vec<u8>>, ReaderError> {
        let io_err = |source: io::Error| ReaderError::Io {
            path: self.path.clone(),
            source,
        };

        let mut ulen_buf = [0u8; 4];
        if !read_exact_or_clean_eof(&mut self.file, &mut ulen_buf).map_err(io_err)? {
            return Ok(None);
        }
        let uncompressed_len = u32::from_le_bytes(ulen_buf) as usize;
        if uncompressed_len > MAX_CHUNK_UNCOMPRESSED_LEN {
            return Err(ReaderError::ChunkTooLarge(uncompressed_len));
        }

        let mut clen_buf = [0u8; 4];
        self.file
            .read_exact(&mut clen_buf)
            .map_err(|_| ReaderError::Truncated("chunk header cut off after uncompressed length".to_string()))?;
        let compressed_len = u32::from_le_bytes(clen_buf) as usize;
        // Review round 1, finding F6: bound the *compressed* length too, independently of
        // `uncompressed_len` — a hostile file's `compressed_len` field is an arbitrary claim, not
        // derived from anything else in the file, so without this check a corrupt/malicious
        // 4-byte value could by itself demand an allocation up to ~4 GiB just to `read_exact` the
        // (nonexistent) bytes into, before decompression is even attempted.
        if compressed_len > MAX_CHUNK_UNCOMPRESSED_LEN {
            return Err(ReaderError::ChunkTooLarge(compressed_len));
        }

        let mut expected_sha256 = [0u8; 32];
        self.file
            .read_exact(&mut expected_sha256)
            .map_err(|_| ReaderError::Truncated("chunk header cut off before its sha256".to_string()))?;

        let mut compressed = vec![0u8; compressed_len];
        self.file
            .read_exact(&mut compressed)
            .map_err(|_| ReaderError::Truncated("chunk body shorter than its own claimed length".to_string()))?;

        // Review round 1, finding F6 (a "zstd bomb"): `zstd::stream::decode_all` has no bound on
        // how much it allocates while decompressing — a tiny, deliberately crafted compressed
        // blob claiming a small `uncompressed_len` in *our* chunk header (which this reader has
        // not verified against the zstd frame's own content at this point) could still contain a
        // zstd frame that actually decompresses to gigabytes, long before the length-mismatch
        // check below ever runs. `zstd::bulk::decompress(data, capacity)` decompresses into a
        // buffer capped at exactly `capacity` bytes, erroring instead of growing past it — using
        // the already-capped `uncompressed_len` (at most `MAX_CHUNK_UNCOMPRESSED_LEN`) as that cap
        // means this can now never allocate more than this reader already considered acceptable.
        let uncompressed = zstd::bulk::decompress(&compressed, uncompressed_len).map_err(ReaderError::Zstd)?;
        if uncompressed.len() != uncompressed_len {
            return Err(ReaderError::ChunkLengthMismatch {
                expected: uncompressed_len,
                actual: uncompressed.len(),
            });
        }

        let mut hasher = Sha256::new();
        hasher.update(&uncompressed);
        let actual_sha256: [u8; 32] = hasher.finalize().into();
        if actual_sha256 != expected_sha256 {
            return Err(ReaderError::ChunkHashMismatch);
        }

        Ok(Some(uncompressed))
    }
}

/// Recomputes `path`'s whole-file sha256 and checks it against `<path>.sha256`
/// ([`crate::writer::sidecar_path`]) — task acceptance criterion 2's whole-file integrity check,
/// independent of (and a superset of) every individual chunk's own sha256 (it also covers the
/// header and the exact byte count, which no chunk hash alone verifies).
pub fn verify_whole_file_sha256(path: &Path) -> Result<bool, ReaderError> {
    let io_err = |source: io::Error| ReaderError::Io {
        path: path.to_path_buf(),
        source,
    };
    let sidecar_text = std::fs::read_to_string(crate::writer::sidecar_path(path)).map_err(io_err)?;
    let expected_hex = sidecar_text
        .split_whitespace()
        .next()
        .ok_or_else(|| ReaderError::Truncated("sidecar file is empty".to_string()))?;

    let mut file = File::open(path).map_err(io_err)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(io_err)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual: [u8; 32] = hasher.finalize().into();
    let actual_hex: String = actual.iter().map(|b| format!("{b:02x}")).collect();
    Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::CharacterRecord;
    use crate::writer::RecordingWriter;
    use ddai_net::generated::objects;

    fn test_header() -> Header {
        Header {
            server_address: "127.0.0.1:8303".to_string(),
            map_name: "Copy Love Box".to_string(),
            map_sha256: [2u8; 32],
            client_version: "DDNet 20.1".to_string(),
            start_time_unix_ms: 1_800_000_000_000,
            observer_nick: "Muha".to_string(),
        }
    }

    fn minimal_snapshot(tick: i32) -> Frame {
        Frame::Snapshot {
            tick,
            characters: vec![CharacterRecord {
                id: 0,
                character: objects::Character {
                    tick,
                    x: 0,
                    y: 0,
                    vel_x: 0,
                    vel_y: 0,
                    angle: 0,
                    direction: 0,
                    jumped: 0,
                    hooked_player: -1,
                    hook_state: 0,
                    hook_tick: 0,
                    hook_x: 0,
                    hook_y: 0,
                    hook_dx: 0,
                    hook_dy: 0,
                    player_flags: 1,
                    health: 10,
                    armor: 0,
                    ammo_count: -1,
                    weapon: 1,
                    emote: 0,
                    attack_tick: 0,
                },
                ddnet: None,
            }],
            players: vec![],
        }
    }

    fn write_test_recording(path: &Path, frame_count: i32) {
        let mut w = RecordingWriter::create(path, &test_header()).unwrap();
        for tick in 0..frame_count {
            w.write_frame(&minimal_snapshot(tick)).unwrap();
        }
        w.finish().unwrap();
    }

    /// Review round 1, finding F6: a chunk whose `uncompressed_len` header field claims a small
    /// size, but whose actual zstd payload decompresses to something far larger (a "zstd bomb"),
    /// must be rejected with a bounded allocation — not silently accepted, and not by letting
    /// decompression run unbounded before any check happens. Builds a real, highly-compressible
    /// 8 MiB payload (compresses to well under 1 KiB), then hand-assembles a rec v1 file whose
    /// chunk header lies about the uncompressed length being only 100 bytes.
    #[test]
    fn a_chunk_claiming_a_small_uncompressed_length_but_decompressing_much_larger_is_rejected() {
        let big_payload = vec![0xABu8; 8 * 1024 * 1024];
        let mut compressed = Vec::new();
        {
            use std::io::Write as _;
            let mut encoder = zstd::stream::Encoder::new(&mut compressed, 3).unwrap();
            encoder.write_all(&big_payload).unwrap();
            encoder.finish().unwrap();
        }
        assert!(compressed.len() < 4096, "test payload should compress extremely well");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bomb.rec");
        let mut header_bytes_writer = binio::Writer::new();
        test_header().encode(&mut header_bytes_writer);
        let header_bytes = header_bytes_writer.into_bytes();

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        // The lie: claims only 100 uncompressed bytes, but `compressed` really decompresses to
        // 8 MiB — real sha256 of the *real* (8 MiB) decompressed bytes, so the sha256 check alone
        // would not have caught this; only the length-mismatch/capacity check does.
        let mut hasher = Sha256::new();
        hasher.update(&big_payload);
        let real_sha256: [u8; 32] = hasher.finalize().into();
        file_bytes.extend_from_slice(&100u32.to_le_bytes()); // uncompressed_len (the lie)
        file_bytes.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        file_bytes.extend_from_slice(&real_sha256);
        file_bytes.extend_from_slice(&compressed);
        std::fs::write(&path, &file_bytes).unwrap();

        let mut r = RecordingReader::open(&path).unwrap();
        let err = r.next_frame().unwrap_err();
        // Must be a clean, bounded error — `ChunkLengthMismatch` (decompressed to more than the
        // 100-byte cap allowed) — never a multi-second hang or a multi-gigabyte allocation.
        assert!(
            matches!(err, ReaderError::ChunkLengthMismatch { .. } | ReaderError::Zstd(_)),
            "expected a clean rejection, got {err:?}"
        );
    }

    /// Review round 1, finding F6 (the other half): a chunk header's `compressed_len` field is an
    /// independent, unverified claim too — a hostile value there must be rejected before this
    /// reader ever allocates a buffer sized by it.
    #[test]
    fn a_chunk_claiming_an_absurd_compressed_length_is_rejected_without_allocating_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bomb2.rec");
        let mut header_bytes_writer = binio::Writer::new();
        test_header().encode(&mut header_bytes_writer);
        let header_bytes = header_bytes_writer.into_bytes();

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&100u32.to_le_bytes()); // uncompressed_len
        file_bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // compressed_len: the lie
        file_bytes.extend_from_slice(&[0u8; 32]); // sha256 (irrelevant, never reached)
        std::fs::write(&path, &file_bytes).unwrap();

        let mut r = RecordingReader::open(&path).unwrap();
        let err = r.next_frame().unwrap_err();
        assert!(
            matches!(err, ReaderError::ChunkTooLarge(_)),
            "expected ChunkTooLarge, got {err:?}"
        );
    }

    #[test]
    fn header_round_trips_through_a_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        write_test_recording(&path, 3);
        let r = RecordingReader::open(&path).unwrap();
        assert_eq!(r.header(), &test_header());
    }

    #[test]
    fn whole_file_sha256_sidecar_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        write_test_recording(&path, 50);
        assert!(verify_whole_file_sha256(&path).unwrap());
    }

    #[test]
    fn corrupting_a_byte_in_the_body_fails_whole_file_verification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        write_test_recording(&path, 50);
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(!verify_whole_file_sha256(&path).unwrap());
    }

    #[test]
    fn corrupting_a_chunk_byte_fails_next_frame_with_a_hash_mismatch_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        write_test_recording(&path, 5);
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip a byte well past the header+chunk-header (into the compressed payload).
        let flip_at = bytes.len() - 5;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let mut r = RecordingReader::open(&path).unwrap();
        // Either the zstd frame itself now fails to decode, or it decodes but the sha256 no
        // longer matches — both are legitimate outcomes of flipping a compressed-payload byte;
        // what must never happen is a panic or a silently-wrong frame.
        let result = r.next_frame();
        assert!(result.is_err(), "corrupted chunk bytes must surface as an error");
    }

    #[test]
    fn empty_recording_yields_no_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        write_test_recording(&path, 0);
        let mut r = RecordingReader::open(&path).unwrap();
        assert!(r.next_frame().unwrap().is_none());
        // Calling again after a clean EOF must stay `None`, not error or panic.
        assert!(r.next_frame().unwrap().is_none());
    }

    #[test]
    fn truncated_file_after_the_header_is_an_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        write_test_recording(&path, 5);
        let bytes = std::fs::read(&path).unwrap();
        // Cut off mid-chunk-header (after the header, before a full chunk header is present).
        let header_only_len = {
            let mut r = binio::Reader::new(&bytes[4..]);
            let _ = Header::decode(&mut r);
            bytes.len() - r.remaining()
        };
        let cut = header_only_len + 6; // header + part of the next chunk's length prefixes
        std::fs::write(&path, &bytes[..cut]).unwrap();

        let mut r = RecordingReader::open(&path).unwrap();
        assert!(r.next_frame().is_err(), "a truncated chunk header must be an error");
    }

    #[test]
    fn nonexistent_file_is_an_io_error_not_a_panic() {
        let err = RecordingReader::open(Path::new("/nonexistent/path/does-not-exist.rec")).unwrap_err();
        assert!(matches!(err, ReaderError::Io { .. }));
    }

    #[test]
    fn truncated_before_any_header_length_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        std::fs::write(&path, [1u8, 2]).unwrap();
        let err = RecordingReader::open(&path).unwrap_err();
        assert!(matches!(err, ReaderError::Truncated(_)));
    }
}
