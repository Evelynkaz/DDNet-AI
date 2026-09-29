//! `ddai-recorder`: the observer recorder (task 8.4a) — record human block games as a spectator,
//! in our own compact versioned format (rec v1, `docs/formats.md` §16), and reconstruct
//! trajectories/estimated inputs from a recording offline.
//!
//! - [`format`]: rec v1's in-memory shape and per-record encode/decode (`Header`, `Frame`).
//! - [`binio`]: the tiny little-endian primitives [`format`] is built on.
//! - [`writer`]: [`writer::RecordingWriter`] — chunked, zstd-compressed, sha256-checked file
//!   writer; the live recorder's on-disk half.
//! - [`reader`]: [`reader::RecordingReader`] — the exact inverse, plus whole-file integrity
//!   verification.
//! - [`anonymize`]: task acceptance criterion 2's `--anonymize` export — stable per-recording ids
//!   in place of nicknames.
//! - [`reconstruct`]: task acceptance criterion 3's offline trajectory/input reconstruction.
//!
//! Driving a live [`ddai_client::Client`] as an observer (`ddnet-ai record`) and the `rec
//! inspect`/`rec reconstruct` CLI both live in the `ddnet-ai` crate (`record_cmd`/`rec_cmd`),
//! exactly like `ddai-client`'s own `Session`/`Client` are driven by `ddnet-ai play`'s
//! `play_cmd` — this crate stays a pure format/offline-analysis library with no live networking
//! or CLI parsing of its own.

pub mod anonymize;
pub mod binio;
pub mod format;
pub mod reader;
pub mod reconstruct;
pub mod writer;
