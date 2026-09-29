//! The on-disk half of rec v1 (task 8.4a acceptance criterion 2): [`RecordingWriter`] streams
//! [`crate::format::Frame`]s straight to a file, buffering only a bounded amount in memory before
//! flushing a zstd-compressed chunk — safe for a recording that runs for hours.
//!
//! File layout (documented in full, in Russian, in `docs/formats.md` §16):
//!
//! ```text
//! header_len: u32 (LE)
//! header_bytes: [header_len]                    — Header::encode's output, uncompressed
//! repeated until EOF:
//!   chunk_uncompressed_len: u32 (LE)
//!   chunk_compressed_len:   u32 (LE)
//!   chunk_sha256:           [u8; 32]             — sha256 of the UNCOMPRESSED chunk bytes
//!   chunk_compressed_bytes: [chunk_compressed_len] — zstd frame (own content checksum enabled)
//! ```
//!
//! A chunk's uncompressed bytes are a back-to-back sequence of `(tag: u8, len: u32, body:
//! [len])` — [`crate::format::Frame::tag`]/[`crate::format::Frame::encode_body`] — a frame never
//! spans a chunk boundary, so [`crate::reader::RecordingReader`] can always decode a whole chunk
//! at once. Alongside the per-chunk sha256, [`RecordingWriter::finish`] also writes a
//! `sha256sum`-format sidecar (`<path>.sha256`) for the *whole* file — task acceptance criterion
//! 2's "sha256 per file", read as covering both the per-chunk integrity check (isolates which
//! chunk is corrupt) and a per-file check (catches truncation/corruption anywhere, including the
//! header) — see `docs/formats.md` §16 for the reasoning spelled out in full.

use crate::format::{Frame, Header};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Flush a chunk once buffered uncompressed frame bytes reach this — bounds memory for an
/// arbitrarily long recording while still giving zstd a large enough window to compress well
/// (recordings are extremely repetitive: most fields are unchanged tick to tick).
const CHUNK_FLUSH_BYTES: usize = 1024 * 1024;
/// ... or once this many frames have been buffered, whichever comes first — bounds the *time*
/// between a frame being written and it actually reaching disk, independent of frame size.
const CHUNK_FLUSH_FRAMES: usize = 4096;
const ZSTD_LEVEL: i32 = 3;

#[derive(Debug, thiserror::Error)]
pub enum WriterError {
    #[error("I/O error writing {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("zstd compression failed: {0}")]
    Zstd(io::Error),
}

/// Summary [`RecordingWriter::finish`] returns — enough for a caller (`ddnet-ai record`) to log a
/// short "recording complete" line without re-opening the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FinishSummary {
    pub frames_written: u64,
    pub bytes_written: u64,
    pub whole_file_sha256: [u8; 32],
}

pub struct RecordingWriter {
    path: PathBuf,
    file: BufWriter<File>,
    whole_file_hasher: Sha256,
    bytes_written: u64,
    frames_written: u64,
    pending: Vec<u8>,
    pending_frames: usize,
}

impl RecordingWriter {
    /// Creates `path` (truncating if it already exists — callers choose a fresh, timestamped path
    /// per session, same as every other `.../logs/<...>` writer in this workspace) and writes the
    /// header immediately, before any frame — `write_frame` and `finish` are the only other calls
    /// a caller needs.
    pub fn create(path: &Path, header: &Header) -> Result<Self, WriterError> {
        let file = File::create(path).map_err(|source| WriterError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut w = RecordingWriter {
            path: path.to_path_buf(),
            file: BufWriter::new(file),
            whole_file_hasher: Sha256::new(),
            bytes_written: 0,
            frames_written: 0,
            pending: Vec::new(),
            pending_frames: 0,
        };
        let mut hw = crate::binio::Writer::new();
        header.encode(&mut hw);
        let header_bytes = hw.into_bytes();
        w.write_raw(&(header_bytes.len() as u32).to_le_bytes())?;
        w.write_raw(&header_bytes)?;
        Ok(w)
    }

    fn write_raw(&mut self, bytes: &[u8]) -> Result<(), WriterError> {
        self.file.write_all(bytes).map_err(|source| WriterError::Io {
            path: self.path.clone(),
            source,
        })?;
        self.whole_file_hasher.update(bytes);
        self.bytes_written += bytes.len() as u64;
        Ok(())
    }

    /// Buffers `frame`, flushing a chunk to disk once [`CHUNK_FLUSH_BYTES`]/[`CHUNK_FLUSH_FRAMES`]
    /// is reached. A single, arbitrarily large frame is still accepted (not split — see the
    /// module docs on why a frame never spans a chunk) at the cost of one chunk being larger than
    /// the usual target; this only matters for a snapshot with an implausible number of
    /// characters, which `ddai_net::snapshot::MAX_ITEMS` (a hard cap the decoder itself enforces,
    /// well before this crate's writer ever sees a snapshot) already makes unreachable in
    /// practice.
    pub fn write_frame(&mut self, frame: &Frame) -> Result<(), WriterError> {
        let body = frame.encode_body();
        self.pending.push(frame.tag());
        self.pending.extend_from_slice(&(body.len() as u32).to_le_bytes());
        self.pending.extend_from_slice(&body);
        self.pending_frames += 1;
        self.frames_written += 1;
        if self.pending.len() >= CHUNK_FLUSH_BYTES || self.pending_frames >= CHUNK_FLUSH_FRAMES {
            self.flush_chunk()?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> Result<(), WriterError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let uncompressed = std::mem::take(&mut self.pending);
        self.pending_frames = 0;

        let mut chunk_hasher = Sha256::new();
        chunk_hasher.update(&uncompressed);
        let chunk_sha256: [u8; 32] = chunk_hasher.finalize().into();

        let mut compressed = Vec::new();
        {
            let mut encoder = zstd::stream::Encoder::new(&mut compressed, ZSTD_LEVEL).map_err(WriterError::Zstd)?;
            encoder.include_checksum(true).map_err(WriterError::Zstd)?;
            encoder.write_all(&uncompressed).map_err(WriterError::Zstd)?;
            encoder.finish().map_err(WriterError::Zstd)?;
        }

        self.write_raw(&(uncompressed.len() as u32).to_le_bytes())?;
        self.write_raw(&(compressed.len() as u32).to_le_bytes())?;
        self.write_raw(&chunk_sha256)?;
        self.write_raw(&compressed)?;
        Ok(())
    }

    /// Flushes any still-buffered frames, then writes the whole-file sha256 sidecar
    /// (`<path>.sha256`, `sha256sum -c`-compatible: `"<hex>  <basename>\n"`) and returns a
    /// summary. Consumes `self`: a `RecordingWriter` is meant to be finished exactly once.
    pub fn finish(mut self) -> Result<FinishSummary, WriterError> {
        self.flush_chunk()?;
        self.file.flush().map_err(|source| WriterError::Io {
            path: self.path.clone(),
            source,
        })?;
        let whole_file_sha256: [u8; 32] = self.whole_file_hasher.finalize().into();

        let sidecar_path = sidecar_path(&self.path);
        let basename = self.path.file_name().unwrap_or_default().to_string_lossy();
        let hex: String = whole_file_sha256.iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(&sidecar_path, format!("{hex}  {basename}\n")).map_err(|source| WriterError::Io {
            path: sidecar_path,
            source,
        })?;

        Ok(FinishSummary {
            frames_written: self.frames_written,
            bytes_written: self.bytes_written,
            whole_file_sha256,
        })
    }
}

/// `<path>.sha256` — the whole-file sidecar's path, also used by [`crate::reader`] to verify it.
pub fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".sha256");
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{CharacterRecord, PlayerRecord, RecordedGameMessage};
    use ddai_net::generated::objects;

    fn test_header() -> Header {
        Header {
            server_address: "127.0.0.1:8303".to_string(),
            map_name: "Copy Love Box".to_string(),
            map_sha256: [1u8; 32],
            client_version: "DDNet 20.1".to_string(),
            start_time_unix_ms: 1_800_000_000_000,
            observer_nick: "Muha".to_string(),
        }
    }

    fn snapshot_frame(tick: i32) -> Frame {
        Frame::Snapshot {
            tick,
            characters: vec![CharacterRecord {
                id: 0,
                character: objects::Character {
                    tick,
                    x: tick * 32,
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
            players: vec![PlayerRecord {
                id: 0,
                info: objects::PlayerInfo {
                    local: 0,
                    client_id: 0,
                    team: 0,
                    score: 0,
                    latency: 0,
                },
                client_info: None,
                ddnet: None,
            }],
        }
    }

    #[test]
    fn writes_header_and_sidecar_and_finish_reports_consistent_stats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        let mut w = RecordingWriter::create(&path, &test_header()).unwrap();
        for tick in 0..10 {
            w.write_frame(&snapshot_frame(tick)).unwrap();
        }
        let summary = w.finish().unwrap();
        assert_eq!(summary.frames_written, 10);

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, summary.bytes_written);

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual: [u8; 32] = hasher.finalize().into();
        assert_eq!(actual, summary.whole_file_sha256);

        let sidecar = std::fs::read_to_string(sidecar_path(&path)).unwrap();
        let hex: String = summary.whole_file_sha256.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(sidecar, format!("{hex}  test.rec\n"));
    }

    #[test]
    fn a_chunk_flush_boundary_does_not_lose_or_duplicate_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        let mut w = RecordingWriter::create(&path, &test_header()).unwrap();
        // Comfortably past CHUNK_FLUSH_FRAMES, forcing at least 3 chunk flushes.
        let total = CHUNK_FLUSH_FRAMES * 3 + 7;
        for tick in 0..total {
            w.write_frame(&snapshot_frame(tick as i32)).unwrap();
        }
        let summary = w.finish().unwrap();
        assert_eq!(summary.frames_written, total as u64);

        let mut r = crate::reader::RecordingReader::open(&path).unwrap();
        let mut ticks = Vec::new();
        while let Some(frame) = r.next_frame().unwrap() {
            if let Frame::Snapshot { tick, .. } = frame {
                ticks.push(tick);
            }
        }
        let expected: Vec<i32> = (0..total as i32).collect();
        assert_eq!(ticks, expected);
    }

    #[test]
    fn game_event_frames_interleave_with_snapshots_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rec");
        let mut w = RecordingWriter::create(&path, &test_header()).unwrap();
        w.write_frame(&snapshot_frame(0)).unwrap();
        w.write_frame(&Frame::GameEvent {
            tick_hint: 0,
            message: RecordedGameMessage::Chat {
                team: 0,
                client_id: 1,
                message: "gg".to_string(),
            },
        })
        .unwrap();
        w.write_frame(&snapshot_frame(1)).unwrap();
        w.finish().unwrap();

        let mut r = crate::reader::RecordingReader::open(&path).unwrap();
        let f0 = r.next_frame().unwrap().unwrap();
        assert!(matches!(f0, Frame::Snapshot { tick: 0, .. }));
        let f1 = r.next_frame().unwrap().unwrap();
        assert!(matches!(
            f1,
            Frame::GameEvent {
                message: RecordedGameMessage::Chat { .. },
                ..
            }
        ));
        let f2 = r.next_frame().unwrap().unwrap();
        assert!(matches!(f2, Frame::Snapshot { tick: 1, .. }));
        assert!(r.next_frame().unwrap().is_none());
    }
}
