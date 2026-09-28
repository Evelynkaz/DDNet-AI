//! The tick/chunk stream: parses the file bytes that follow the embedded map (see
//! [`crate::header::Prelude::map_offset`]) into a per-tick sequence of decoded snapshots and game
//! messages. Ported from `CDemoPlayer::ReadChunkHeader`/`DoTick` (`engine/shared/demo.cpp:534-822`,
//! DDNet 20.1 `c9d208138f85755521f16a0096b6fe036c5c8698`), which carries the original Teeworlds
//! zlib-style notice:
//!
//!   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//!   If you are missing that file, acquire a complete release at teeworlds.com.
//!
//! This is an *altered* source version: rewritten in safe Rust as a plain iterator instead of a
//! stateful player object (no `Play`/`Pause`/`SeekTick`/real-time clock — this crate only ever
//! reads a file start to finish), driven the same way real code drives it once (`Play()` then a
//! single `Update(/*RealTime=*/false)`, which loops `DoTick` until `Pause()` — see
//! `CDemoEditor::Slice`, `demo.cpp:1491-1499`, the model [`Demo::ticks`] follows) — every
//! chunk-decode step below cites the exact lines it mirrors. Compression is
//! [`ddai_net::huffman::Huffman`]/[`ddai_net::packer::unpack_ints`] (task 2.2a); snapshot deltas
//! are [`ddai_net::delta::unpack_delta`] (task 2.2b); full (`CHUNKTYPE_SNAPSHOT`) snapshots are
//! [`crate::rawsnapshot::parse_raw_snapshot`] (this crate's own code — see that module's docs for
//! why); game messages are [`ddai_net::message::decode`].

use crate::error::TickIterError;
use crate::header::VERSION_TICK_COMPRESSION;
use crate::rawsnapshot::parse_raw_snapshot;
use ddai_net::delta::{self, StaticSizes};
use ddai_net::huffman::Huffman;
use ddai_net::message::{self, Msg, Registry};
use ddai_net::snapshot::Snapshot;
use std::sync::Arc;

// `CHUNKTYPEFLAG_TICKMARKER`/`CHUNKTICKFLAG_TICK_COMPRESSED`/`CHUNKMASK_TICK`/
// `CHUNKMASK_TICK_LEGACY`/`CHUNKMASK_TYPE`/`CHUNKMASK_SIZE` (`demo.cpp:240-254`).
// `CHUNKTICKFLAG_KEYFRAME` (0x40) is deliberately not modeled: it only feeds
// `CDemoPlayer::ScanFile`'s keyframe list for efficient random-access seeking
// (`demo.cpp:652-657`), which this crate never does — see `header::Prelude`'s docs.
const TICKMARKER_FLAG: u8 = 0x80;
const TICK_COMPRESSED_FLAG: u8 = 0x20;
const TICK_MASK: u8 = 0x1F;
const TICK_MASK_LEGACY: u8 = 0x3F;
const TYPE_MASK: u8 = 0x60;
const SIZE_MASK: u8 = 0x1F;

// `CHUNKTYPE_SNAPSHOT`/`CHUNKTYPE_MESSAGE`/`CHUNKTYPE_DELTA` (`demo.cpp:251-253`). `0` is unused
// by the format (never written by `CDemoRecorder::Write`) — a chunk with this numeric type is
// simply ignored, matching real `DoTick`'s `else` chain falling through with no matching branch.
const CHUNKTYPE_SNAPSHOT: u8 = 1;
const CHUNKTYPE_MESSAGE: u8 = 2;
const CHUNKTYPE_DELTA: u8 = 3;

// `MIN_TICK`/`MAX_TICK` (`engine/shared/protocol.h:94-95`).
const MIN_TICK: i32 = 0;
const MAX_TICK: i32 = 0x6FFF_FFFF;

/// `CSnapshot::MAX_SIZE` (`snapshot.h:50`) — the fixed size of the real
/// `m_aDecompressedSnapshotData`/`m_aChunkData` buffers (`demo.h:130,135`) every chunk's
/// Huffman-decompressed bytes, and then its variable-int-decompressed ints, must fit in.
const MAX_CHUNK_BYTES: usize = ddai_net::snapshot::MAX_SIZE;
/// [`MAX_CHUNK_BYTES`] in `i32` units — the bound on a chunk's fully decompressed int count
/// (`CVariableInt::Decompress`'s `DstSize` check, `compression.cpp:44-45`, against
/// `sizeof(m_aChunkData)`).
const MAX_CHUNK_INTS: usize = MAX_CHUNK_BYTES / 4;

/// Defensive cap on the number of message chunks accumulated for a single [`Tick`] — review
/// round 1 finding F2: a hostile file can pack a chunk header down to one byte (tick-marker flag
/// clear, type `MESSAGE`, size 0 — e.g. byte `0x40`), so without this cap the number of [`Msg`]
/// entries [`TickIter::step`] accumulates between two tick markers is bounded only by the whole
/// file's size, not by anything resembling real DDNet traffic: an ~8 MiB file of such chunks
/// requested a ~4.4 GB allocation (`Msg` is 528 bytes) before this cap existed. `MAX_CLIENTS` is
/// 64 (`protocol.h`); this is chosen far more generously (128x) so no real demo can plausibly
/// hit it — exceeding it is reported as [`TickIterError::TooManyMessages`], the same
/// "deliver-partial-then-error" way any other fatal decode error is (see [`TickIter::step`]).
pub const MAX_MESSAGES_PER_TICK: usize = 8192;

/// One tick's worth of decoded demo content, yielded by [`TickIter`] — mirrors one `DoTick` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tick {
    pub tick: i32,
    /// The tick's fully decoded snapshot: either freshly decoded from a `CHUNKTYPE_SNAPSHOT`/
    /// `CHUNKTYPE_DELTA` chunk this tick, or (`demo.cpp:801-807`) the last known snapshot
    /// replayed because this tick recorded none of its own (a normal, common case — DDNet only
    /// writes a new snapshot chunk when the *server* sent one that tick). `None` only before the
    /// very first snapshot chunk has ever appeared (`m_LastSnapshotDataSize == -1`,
    /// `demo.cpp:803`) — practically unreachable for any real demo, whose first recorded tick
    /// always carries one (`CDemoRecorder::RecordSnapshot`'s first call always takes the
    /// full-keyframe branch, `demo.cpp:339-350`).
    ///
    /// `Arc`, not an owned [`Snapshot`] (review round 1 finding F2): most real ticks replay the
    /// previous tick's snapshot unchanged (`demo.cpp:801-807` — the server simply had nothing new
    /// to send that tick), so a file with many ticks and few real snapshot/delta chunks would
    /// otherwise deep-clone a potentially large [`Snapshot`] once per tick for no reason; cloning
    /// an `Arc` is an O(1) refcount bump regardless of the snapshot's size.
    pub snapshot: Option<Arc<Snapshot>>,
    /// Every game message chunk recorded for this tick, in file order — capped at
    /// [`MAX_MESSAGES_PER_TICK`] entries.
    pub messages: Vec<Msg>,
}

/// Iterates a `.demo` file's tick/chunk stream — see the module docs and [`TickIter::step`]'s
/// docs for exactly when a `Some(Err(_))` (or the final, non-error `None`) is yielded. Constructed
/// via [`crate::Demo::ticks`].
pub struct TickIter<'a> {
    data: &'a [u8],
    pos: usize,
    version: u8,

    huffman: Huffman,
    static_sizes: StaticSizes,
    registry: Registry,
    // Reused scratch buffers, mirroring the real fixed-size
    // `m_aCompressedSnapshotData`/`m_aDecompressedSnapshotData`/`m_aChunkData` (`demo.h:130,131,135`)
    // — bounded, allocated once, never grown past [`MAX_CHUNK_BYTES`].
    decompressed: Vec<u8>,

    // `CDemoPlayer::CPlaybackInfo` fields this crate actually needs (`demo.h:78-100`): the rest
    // (timing/speed/pause/live-demo bookkeeping) only matters for real-time playback, which this
    // crate never does.
    current_tick: i32,
    previous_tick: i32,
    next_tick: i32,
    last_snapshot: Option<Arc<Snapshot>>,
    finished: bool,
    /// An error already decided (by [`TickIter::stop_with_error`]) but not yet surfaced, because
    /// the tick it interrupted had real content that must be yielded first (review round 1
    /// finding F5) — taken and returned on the *next* call to [`TickIter::step`].
    pending_error: Option<TickIterError>,
}

impl<'a> TickIter<'a> {
    pub(crate) fn new(data: &'a [u8], start_pos: usize, version: u8) -> Self {
        TickIter {
            data,
            pos: start_pos,
            version,
            huffman: Huffman::new(),
            static_sizes: StaticSizes::ddnet_06(),
            registry: Registry::new(),
            decompressed: vec![0u8; MAX_CHUNK_BYTES],
            current_tick: -1,
            previous_tick: -1,
            next_tick: -1,
            last_snapshot: None,
            finished: false,
            pending_error: None,
        }
    }

    fn read_u8(&mut self) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    /// `CDemoPlayer::ReadChunkHeader` (`demo.cpp:534-597`). `tick` is `*pTick` — the running tick
    /// cursor, seeded by the caller from `current_tick` and updated in place by every tick-marker
    /// chunk encountered (including ones this call itself doesn't stop at — see `next`).
    ///
    /// Distinguishes [`ChunkHeader::Truncated`] (the file simply ran out of bytes mid-header —
    /// `demo.cpp:566,583,590`) from [`ChunkHeader::BadTickMarker`] (a *malformed value*: a
    /// delta-encoded marker with no prior absolute tick, or a decoded tick outside
    /// `MIN_TICK..MAX_TICK`) — review round 1 finding F5: these were both folded into one
    /// `TickIterError::BadTickMarker` before, mislabeling a truncated extended-size byte.
    fn read_chunk_header(&mut self, tick: &mut i32) -> ChunkHeader {
        let Some(chunk_byte) = self.read_u8() else {
            return ChunkHeader::Eof;
        };

        if chunk_byte & TICKMARKER_FLAG != 0 {
            let tickdelta_legacy = chunk_byte & TICK_MASK_LEGACY;
            let new_tick = if self.version < VERSION_TICK_COMPRESSION && tickdelta_legacy != 0 {
                if *tick < 0 {
                    return ChunkHeader::BadTickMarker;
                }
                tick.wrapping_add(i32::from(tickdelta_legacy))
            } else if chunk_byte & TICK_COMPRESSED_FLAG != 0 {
                if *tick < 0 {
                    return ChunkHeader::BadTickMarker;
                }
                tick.wrapping_add(i32::from(chunk_byte & TICK_MASK))
            } else {
                let Some(bytes) = self.read_bytes(4) else {
                    return ChunkHeader::Truncated;
                };
                i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            if !(MIN_TICK..MAX_TICK).contains(&new_tick) {
                return ChunkHeader::BadTickMarker;
            }
            *tick = new_tick;
            return ChunkHeader::TickMarker;
        }

        let ty = (chunk_byte & TYPE_MASK) >> 5;
        let mut size = usize::from(chunk_byte & SIZE_MASK);
        if size == 30 {
            let Some(b) = self.read_u8() else {
                return ChunkHeader::Truncated;
            };
            size = usize::from(b);
        } else if size == 31 {
            let Some(b) = self.read_bytes(2) else {
                return ChunkHeader::Truncated;
            };
            size = (usize::from(b[1]) << 8) | usize::from(b[0]);
        }
        ChunkHeader::Data { ty, size }
    }

    /// Reads `size` raw bytes, then Huffman- and variable-int-decompresses them into `ints`,
    /// exactly mirroring `demo.cpp:713-736` (`m_aCompressedSnapshotData` -> `CNetBase::Decompress`
    /// -> `m_aDecompressedSnapshotData` -> `CVariableInt::Decompress` -> `m_aChunkData`), both
    /// stages bounded by [`MAX_CHUNK_BYTES`]/[`MAX_CHUNK_INTS`] the same way the real fixed-size
    /// buffers bound them. `size == 0` yields an empty `ints` (the `if(ChunkSize) {...}` guard,
    /// `demo.cpp:715`, skips both decompression calls entirely).
    fn read_chunk_payload(&mut self, size: usize, ints: &mut Vec<i32>) -> Result<(), TickIterError> {
        ints.clear();
        if size == 0 {
            return Ok(());
        }
        let Some(raw) = self.read_bytes(size) else {
            return Err(TickIterError::TruncatedChunk);
        };
        let decompressed_len = self
            .huffman
            .decompress(raw, &mut self.decompressed)
            .ok_or(TickIterError::HuffmanDecompressFailed)?;
        ddai_net::packer::unpack_ints(&self.decompressed[..decompressed_len], ints)
            .ok_or(TickIterError::IntpackDecompressFailed)?;
        if ints.len() > MAX_CHUNK_INTS {
            return Err(TickIterError::IntpackDecompressFailed);
        }
        Ok(())
    }
}

enum ChunkHeader {
    Eof,
    /// A malformed tick-marker *value* (see `read_chunk_header`'s docs).
    BadTickMarker,
    /// The file ran out of bytes mid-header (see `read_chunk_header`'s docs).
    Truncated,
    TickMarker,
    Data {
        ty: u8,
        size: usize,
    },
}

impl<'a> Iterator for TickIter<'a> {
    type Item = Result<Tick, TickIterError>;

    /// One [`Tick`] — see [`TickIter::step`] for the actual `DoTick` port. This wrapper folds in
    /// `CDemoPlayer::Play`'s own driving loop (`demo.cpp:1000-1011`:
    /// `while(m_Info.m_PreviousTick == -1) DoTick();`): the very first `DoTick`-equivalent pass
    /// over any file always has `current_tick == -1` (nothing has been "current" yet — the
    /// tick/chunk stream's first bytes are always its first tick marker, so that pass reads
    /// exactly that marker and nothing else, discovering the first real tick without ever
    /// producing content) and is never observable through `Play()` to any real caller (the demo
    /// player's actual listener is always driven by `Play()`, never by a bare `DoTick()` — see
    /// `CDemoEditor::Slice`, `demo.cpp:1489-1499`, the only real caller in the codebase). This
    /// loop reproduces that: it discards a `tick == -1` result and steps again, which — because
    /// [`TickIter::step`] only ever produces `tick == -1` on the very first call to `next` on a
    /// freshly constructed iterator (every later tick marker is validated `>= MIN_TICK` (0) by
    /// `step`'s `read_chunk_header`, `demo.cpp:570`) — runs at most once per [`TickIter`].
    ///
    /// This can discard real content: if a `.demo` file's tick/chunk stream begins with a
    /// snapshot/message chunk *before* its very first tick marker (structurally allowed by the
    /// format, though no known real recorder ever produces it — `CDemoRecorder::RecordSnapshot`
    /// always writes a tick marker synchronously before any snapshot data,
    /// `CDemoRecorder::Write`, `demo.cpp:335-366`), that content is attributed to `tick == -1`
    /// and silently dropped here — exactly as it would be unobservable to any real caller of
    /// `CDemoPlayer` too (`Play()` never returns until past this point, so nothing outside the
    /// player ever sees a `CurrentTick == -1` callback either).
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.step() {
                Some(Ok(tick)) if tick.tick == -1 => continue,
                other => return other,
            }
        }
    }
}

impl<'a> TickIter<'a> {
    /// `CDemoPlayer::DoTick` (`demo.cpp:676-822`). See [`TickIterError`]'s docs for the
    /// deliver-partial-then-error contract every fatal-error path below follows via
    /// [`TickIter::stop_with_error`], and the [`ChunkHeader::Eof`] arm's own doc comment for why
    /// EOF is handled separately from that (it is not itself an error).
    fn step(&mut self) -> Option<Result<Tick, TickIterError>> {
        if self.finished {
            return None;
        }
        // A previous call already decided this iterator must end in error, but had to yield a
        // partial `Tick` first (review round 1 finding F5) — that content was already returned by
        // the call that set this; this call's only job is to surface the error itself.
        if let Some(error) = self.pending_error.take() {
            self.finished = true;
            return Some(Err(error));
        }

        // `demo.cpp:679-681`.
        self.previous_tick = self.current_tick;
        self.current_tick = self.next_tick;
        let mut chunk_tick = self.current_tick;

        let mut got_snapshot = false;
        let mut tick_snapshot: Option<Arc<Snapshot>> = None;
        let mut messages = Vec::new();
        let mut ints: Vec<i32> = Vec::new();

        loop {
            let header = self.read_chunk_header(&mut chunk_tick);
            match header {
                ChunkHeader::Eof => {
                    self.finished = true;
                    // `demo.cpp:690-706`. Real DDNet delivers listener callbacks (snapshot/
                    // message) synchronously as each chunk is decoded, not deferred to a
                    // tick-marker/EOF boundary — so whatever this call already decoded before
                    // hitting EOF (`tick_snapshot`/`messages`) was already "delivered" and must
                    // still be surfaced here, exactly as if a tick marker had ended it cleanly;
                    // a real EOF is not itself reported as an error (matches `Stop()` with no
                    // error message, `demo.cpp:698`) — the iterator just ends after this `Tick`.
                    // Only when this iterator has *never* established a real current tick at all
                    // (`current_tick == -1` — no tick marker has ever been read yet; review round
                    // 1 finding F5 caught this checking `previous_tick` instead, which is also
                    // `-1` for the *second* `step` call, silently discarding that real tick's
                    // content too) is this the "no tick data at all" case.
                    return if self.current_tick == -1 {
                        Some(Err(TickIterError::EmptyDemo))
                    } else {
                        Some(Ok(Tick {
                            tick: self.current_tick,
                            snapshot: tick_snapshot,
                            messages,
                        }))
                    };
                }
                ChunkHeader::BadTickMarker => {
                    return self.stop_with_error(TickIterError::BadTickMarker, tick_snapshot, messages);
                }
                ChunkHeader::Truncated => {
                    return self.stop_with_error(TickIterError::TruncatedChunk, tick_snapshot, messages);
                }
                ChunkHeader::TickMarker => {
                    // Replay check (`demo.cpp:801-807`) applies to tick-marker chunks too — see
                    // the `Data` arm's identical check for why.
                    if !got_snapshot && let Some(last) = &self.last_snapshot {
                        tick_snapshot = Some(Arc::clone(last));
                    }
                    self.next_tick = chunk_tick;
                    break;
                }
                ChunkHeader::Data { ty, size } => {
                    if let Err(e) = self.read_chunk_payload(size, &mut ints) {
                        return self.stop_with_error(e, tick_snapshot, messages);
                    }

                    if ty == CHUNKTYPE_DELTA {
                        // `demo.cpp:738-776`.
                        let Some(base) = &self.last_snapshot else {
                            return self.stop_with_error(
                                TickIterError::DeltaBeforeFullSnapshot,
                                tick_snapshot,
                                messages,
                            );
                        };
                        // A delta payload is always at least its 3-int header
                        // (`CSnapshotDelta::CData`, `snapshot.h:91-97`); an empty chunk
                        // (`size == 0`, never produced by a real recorder — see
                        // `read_chunk_payload`'s docs) is reported the same way any other
                        // malformed delta is: logged and skipped, not fatal (`demo.cpp:749-757`
                        // never calls `Stop` on a bad delta, only on missing base data — an
                        // `Err` here is deliberately swallowed, nothing else to update).
                        if let Ok(snap) = delta::unpack_delta(base, &ints, &self.static_sizes) {
                            let snap = Arc::new(snap);
                            self.last_snapshot = Some(Arc::clone(&snap));
                            tick_snapshot = Some(snap);
                            got_snapshot = true;
                        }
                    } else if ty == CHUNKTYPE_SNAPSHOT {
                        // `demo.cpp:777-799`; an `Err` here is logged-and-continue too, same as
                        // an invalid delta above.
                        if let Ok(snap) = parse_raw_snapshot(&ints) {
                            let snap = Arc::new(snap);
                            self.last_snapshot = Some(Arc::clone(&snap));
                            tick_snapshot = Some(snap);
                            got_snapshot = true;
                        }
                    } else {
                        // `demo.cpp:800-820`: replay the last snapshot if this tick hasn't
                        // produced one of its own yet, then handle the remaining chunk types
                        // (message; anything else, including the unused numeric type 0, is
                        // silently ignored — its bytes were already consumed above).
                        if !got_snapshot && let Some(last) = &self.last_snapshot {
                            tick_snapshot = Some(Arc::clone(last));
                            got_snapshot = true;
                        }
                        if ty == CHUNKTYPE_MESSAGE {
                            // Review round 1 finding F2: bound the number of messages a single
                            // tick can accumulate — see [`MAX_MESSAGES_PER_TICK`]'s docs.
                            if messages.len() >= MAX_MESSAGES_PER_TICK {
                                return self.stop_with_error(
                                    TickIterError::TooManyMessages(MAX_MESSAGES_PER_TICK),
                                    tick_snapshot,
                                    messages,
                                );
                            }
                            // Demo message chunks are intpack-compressed like snapshots/deltas
                            // (`CDemoRecorder::Write` applies the same pipeline to every chunk
                            // type, `demo.cpp:281-333`, unlike the live network path where only
                            // snapshot/delta payloads are intpacked — see this crate's README).
                            // Reconstitute the original message bytes from `ints` the same way
                            // `CVariableInt::Decompress` originally produced them: each `i32` is
                            // 4 bytes of the message buffer in the host's native order, which for
                            // every DDNet target (x86/ARM, all little-endian) is little-endian.
                            let mut bytes = Vec::with_capacity(ints.len() * 4);
                            for v in &ints {
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                            let (msg, _answer) = message::decode(&bytes, &self.registry);
                            messages.push(msg);
                        }
                        // ty == 0 (unused) or any other value: nothing further to do.
                    }
                }
            }
        }

        Some(Ok(Tick {
            tick: self.current_tick,
            snapshot: tick_snapshot,
            messages,
        }))
    }

    /// Shared tail for every fatal decode error within [`TickIter::step`]'s inner loop — see
    /// [`TickIterError`]'s docs for the contract this implements: if a real current tick was
    /// already established (`current_tick != -1`), whatever this call already decoded for it
    /// (`tick_snapshot`/`messages`) was already synchronously "delivered" to a real listener
    /// before the error, in real DDNet, so it is yielded now as a normal `Tick` and `error` is
    /// stashed for the *next* call to `step`; otherwise (still within this iterator's one-time
    /// priming pass, `next`'s doc comment) there is no earlier `Tick` to protect and `error` is
    /// reported immediately.
    fn stop_with_error(
        &mut self,
        error: TickIterError,
        tick_snapshot: Option<Arc<Snapshot>>,
        messages: Vec<Msg>,
    ) -> Option<Result<Tick, TickIterError>> {
        if self.current_tick == -1 {
            self.finished = true;
            return Some(Err(error));
        }
        self.pending_error = Some(error);
        Some(Ok(Tick {
            tick: self.current_tick,
            snapshot: tick_snapshot,
            messages,
        }))
    }
}
