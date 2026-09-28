//! `ddai-demo`: a safe, exact reader for DDNet's client-side `.demo` format (versions 4-6, plus 3
//! as a free consequence of the same general algorithm — see [`header`]).
//!
//! A `.demo` file is a fixed header, an optional timeline-markers block, an optional SHA256
//! extension, the embedded map's raw bytes, and then a stream of Huffman + variable-int
//! compressed chunks (tick markers, full snapshots, snapshot deltas, game messages) — see
//! `docs/formats.md` for the annotated byte layout and [`header`]/[`reader`] for the DDNet 20.1
//! `engine/shared/demo.{h,cpp}` citations each parsing step mirrors.
//!
//! ```no_run
//! let bytes = std::fs::read("game.demo").unwrap();
//! let demo = ddai_demo::Demo::parse(&bytes).unwrap();
//! println!("map: {} ({} bytes)", demo.map.name, demo.map.size);
//! for tick in demo.ticks() {
//!     let tick = tick.unwrap();
//!     if let Some(snap) = &tick.snapshot {
//!         let view = ddai_net::view::View::new(snap);
//!         println!("tick {}: {} characters", tick.tick, view.characters().len());
//!     }
//! }
//! ```
//!
//! No `unsafe`. Every parsing function returns a `Result` instead of panicking on malformed or
//! adversarial input (see `tests/robustness.rs`/`tests/hostile.rs`/`tests/fuzz_real_bytes.rs` for
//! the fuzz-style/regression proof) and every buffer this crate allocates is bounded: the whole
//! file by [`header::MAX_DEMO_FILE_SIZE`] (checked by the *caller* against the file's size before
//! even reading it — see `ddnet-ai demo`'s `read_file`), one chunk's decompressed bytes by the
//! fixed [`ddai_net::snapshot::MAX_SIZE`] (matching DDNet's own fixed-size demo-playback buffers),
//! and one tick's accumulated messages by [`reader::MAX_MESSAGES_PER_TICK`] — the last two are
//! this crate's own defensive caps DDNet's own reader doesn't have (review round 1 finding F2).

pub mod error;
pub mod header;
pub mod rawsnapshot;
pub mod reader;
#[cfg(any(test, feature = "test-util"))]
pub mod testutil;

pub use error::{HeaderError, TickIterError};
pub use header::{DemoHeader, MapInfo};
pub use reader::{Tick, TickIter};

/// A parsed `.demo` file: the header/map prelude, eagerly validated, plus [`Demo::ticks`] to
/// lazily walk the tick/chunk stream that follows the embedded map. Borrows the input buffer for
/// its whole lifetime (`'a`) — nothing here copies the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Demo<'a> {
    data: &'a [u8],
    pub header: DemoHeader,
    /// Ticks the recorder marked as timeline bookmarks (see `header::Prelude`'s docs for the one
    /// way this deliberately diverges from `CDemoPlayer::Load`).
    pub timeline_markers: Vec<i32>,
    pub map: MapInfo,
    map_offset: usize,
}

impl<'a> Demo<'a> {
    /// Parses `data`'s header/timeline-markers/map prelude. Does not touch the tick/chunk stream
    /// that follows the map — call [`Demo::ticks`] for that, which is where a hostile file's
    /// per-chunk errors actually surface (matches the fact that real `CDemoPlayer::Load` only
    /// does the same up-front work: header, timeline markers, `ScanFile` for keyframe positions —
    /// see [`header::Prelude`]'s docs for why this crate skips `ScanFile` specifically).
    pub fn parse(data: &'a [u8]) -> Result<Self, HeaderError> {
        let prelude = header::parse_prelude(data)?;
        Ok(Demo {
            data,
            header: prelude.header,
            timeline_markers: prelude.timeline_markers,
            map: prelude.map,
            map_offset: prelude.map_offset,
        })
    }

    /// The embedded map's raw bytes (`data[map_offset..map_offset + map_size]`) — hand these to
    /// `ddai_map::load_map` to decode the map itself.
    pub fn map_bytes(&self) -> &'a [u8] {
        let size = self.header.map_size as usize;
        &self.data[self.map_offset..self.map_offset + size]
    }

    /// A fresh iterator over the tick/chunk stream that follows the embedded map — see
    /// [`reader::TickIter`]. Cheap to call repeatedly (no shared mutable state with any other
    /// iterator this method returns).
    pub fn ticks(&self) -> TickIter<'a> {
        let start = self.map_offset + self.header.map_size as usize;
        TickIter::new(self.data, start, self.header.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{build_prelude_bytes, build_synthetic_demo};

    #[test]
    fn parses_synthetic_demo_end_to_end() {
        let bytes = build_synthetic_demo(6);
        let demo = Demo::parse(&bytes).expect("synthetic demo parses");
        assert_eq!(demo.header.version, 6);
        assert!(demo.map_bytes().is_empty());

        let ticks: Vec<Tick> = demo.ticks().map(|t| t.expect("no decode errors")).collect();
        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].tick, 10);
        let snap0 = ticks[0].snapshot.as_ref().expect("tick 10 has a snapshot");
        assert_eq!(snap0.items.len(), 1);
        assert_eq!(snap0.items[0].key, 1 << 16);
        assert_eq!(snap0.items[0].data, vec![7]);
        assert!(ticks[0].messages.is_empty());

        assert_eq!(ticks[1].tick, 11);
        // The delta had no changes, so tick 11's snapshot is identical to tick 10's.
        assert_eq!(ticks[1].snapshot.as_ref(), Some(snap0));
        assert_eq!(ticks[1].messages.len(), 1);
    }

    #[test]
    fn ticks_can_be_iterated_more_than_once() {
        let bytes = build_synthetic_demo(6);
        let demo = Demo::parse(&bytes).unwrap();
        let first: Vec<i32> = demo.ticks().map(|t| t.unwrap().tick).collect();
        let second: Vec<i32> = demo.ticks().map(|t| t.unwrap().tick).collect();
        assert_eq!(first, second);
        assert_eq!(first, vec![10, 11]);
    }

    #[test]
    fn empty_tick_stream_is_reported_as_empty_demo() {
        let bytes = build_prelude_bytes(6, 0);
        let demo = Demo::parse(&bytes).unwrap();
        let mut ticks = demo.ticks();
        assert_eq!(ticks.next(), Some(Err(TickIterError::EmptyDemo)));
        assert_eq!(ticks.next(), None, "iterator stays finished after the first error");
    }
}
