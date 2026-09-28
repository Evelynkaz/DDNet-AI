//! Error types for [`crate::header`] and [`crate::reader`] — every parsing function returns a
//! [`DemoError`] instead of panicking, matching this crate's "never panic on hostile input"
//! contract (task 8.4b acceptance criterion 1).

/// Why reading a `.demo` file's header/prelude (magic, version, netversion/map name/type strings,
/// timeline markers, optional SHA256 extension) failed — mirrors the failure points of
/// `CDemoPlayer::GetDemoInfo`/`Load` (`engine/shared/demo.cpp:1315-1410,842-913`, DDNet 20.1
/// `c9d208138f85755521f16a0096b6fe036c5c8698`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HeaderError {
    /// The file is larger than [`crate::header::MAX_DEMO_FILE_SIZE`] — this crate's own defensive
    /// cap (task 8.4b acceptance criterion 1's "file-size ... limits"; not a DDNet-side check).
    #[error("file is larger than the {0}-byte limit this reader accepts")]
    FileTooLarge(usize),
    /// Fewer than `sizeof(CDemoHeader)` (176) bytes in the file.
    #[error("file shorter than the demo header (176 bytes)")]
    TooShortForHeader,
    /// `m_aMarker` is not `"TWDEMO\0"` (`CDemoHeader::Valid`, `demo.cpp:39-47`).
    #[error("bad magic: not a TWDEMO file")]
    BadMagic,
    /// One of `m_aNetversion`/`m_aMapName`/`m_aType`/`m_aTimestamp` has no NUL terminator within
    /// its fixed-size field, or its bytes up to the first NUL are not valid UTF-8
    /// (`CDemoHeader::Valid`, `demo.cpp:42-46`: `mem_has_null` + `str_utf8_check` on each field).
    #[error("header string field {field} is not a NUL-terminated valid UTF-8 C string")]
    BadHeaderString { field: &'static str },
    /// `m_Version < gs_OldVersion` (3) — `demo.cpp:1341-1348`.
    #[error("demo version {0} is not supported (must be >= 3)")]
    UnsupportedVersion(u8),
    /// File ends before the `CTimelineMarkers` block (`version > 3` reads it unconditionally,
    /// `demo.cpp:1349-1359`).
    #[error("file shorter than the timeline markers block")]
    TooShortForTimelineMarkers,
    /// File ends before the 16-byte SHA256-extension UUID (`version >= 6`, `demo.cpp:1362-1397`).
    #[error("file shorter than the SHA256 extension marker")]
    TooShortForSha256Marker,
    /// The SHA256 extension UUID matched, but the file ends before the 32-byte digest that should
    /// follow it (`demo.cpp:1368-1377`).
    #[error("file shorter than the SHA256 digest")]
    TooShortForSha256Digest,
    /// `m_aMapSize` claims more bytes than remain in the file after the prelude.
    #[error("declared map size ({declared}) exceeds the remaining file size ({remaining})")]
    MapSizeExceedsFile { declared: u32, remaining: u64 },
    /// `m_aNetversion` starts with `"0.7"` — a Teeworlds 0.7 ("sixup") demo (`m_Sixup =
    /// str_startswith(m_aNetversion, "0.7")`, `demo.cpp:873`). Out of scope (the task spec's
    /// goal is 0.6+DDNet only — see docs/formats.md §18) and actively wrong to decode with this
    /// reader: every chunk/snapshot/message decision this crate makes (`StaticSizes::ddnet_06()`,
    /// `ddai_net::message`'s numbered-id tables) assumes the 0.6 wire format, which sixup's
    /// `CSnapshotDeltaSixup`/renumbered ids do not share (review round 1 finding F10) — rejected
    /// outright here rather than silently producing wrong data.
    #[error("Teeworlds 0.7 (\"sixup\") demos are not supported by this reader")]
    UnsupportedSixup,
}

/// Why decoding the tick/chunk stream (after the header+map prelude) failed — mirrors
/// `CDemoPlayer::ReadChunkHeader`/`DoTick` (`demo.cpp:534-822`).
///
/// Once a [`TickIterError`] other than via the very first call is yielded, the iterator that
/// produced it always yields `None` afterward (matches `CDemoPlayer::Stop` making `IsPlaying()`
/// false for good). Real DDNet delivers listener callbacks (snapshot/message) synchronously as
/// each chunk is decoded, not deferred to a tick-marker boundary — so when a fatal error occurs
/// partway through a tick that already had real content decoded, that content was already
/// "delivered" before the error and is not discarded here either: the iterator yields the partial
/// `Tick` first (review round 1 finding F5) and this error on the *next* call, except when nothing
/// has ever been established yet (this iterator's very first call, before even one tick marker
/// has been read), in which case there is no earlier `Tick` to protect and the error is immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TickIterError {
    /// `ReadChunkHeader` hit EOF before this iterator ever established a real current tick — the
    /// file has no tick data at all (`demo.cpp:692-695`, `"Empty demo"`). Note this is *not* an
    /// error yielded after a partial `Tick` (see the enum docs) — a real EOF after real ticks were
    /// already read is not reported as an error at all, matching `CDemoPlayer::Stop()` with no
    /// error message (`demo.cpp:698`): the iterator just ends (`None`) after its last real `Tick`.
    #[error("demo has no tick data (empty demo)")]
    EmptyDemo,
    /// A tick-marker's delta/absolute tick value was malformed: a delta-encoded marker appeared
    /// before any absolute tick was read, or the decoded tick fell outside `MIN_TICK..MAX_TICK`
    /// (`demo.cpp:552-571`). Distinct from [`TickIterError::TruncatedChunk`] (review round 1
    /// finding F5): this is a *malformed value*, not a short read.
    #[error("malformed or out-of-range tick marker")]
    BadTickMarker,
    /// The file ended in the middle of a chunk header (a tick marker's 4-byte absolute tick, or a
    /// chunk's 1/2-byte extended size field) or a chunk's declared data (`demo.cpp:566,583,590,717`).
    #[error("truncated chunk header or chunk data")]
    TruncatedChunk,
    /// Huffman decompression of a chunk's compressed bytes failed, or produced more than
    /// `CSnapshot::MAX_SIZE` (65536) bytes (`demo.cpp:723-728`, bounded by the real
    /// `m_aDecompressedSnapshotData[CSnapshot::MAX_SIZE]` buffer).
    #[error("Huffman decompression of chunk data failed")]
    HuffmanDecompressFailed,
    /// Variable-int decompression of a chunk's Huffman-decompressed bytes failed, or produced
    /// more than 16384 ints (`demo.cpp:730-735`, bounded by `m_aChunkData[CSnapshot::MAX_SIZE]`).
    #[error("variable-int decompression of chunk data failed")]
    IntpackDecompressFailed,
    /// A `CHUNKTYPE_DELTA` chunk appeared before any full `CHUNKTYPE_SNAPSHOT` chunk
    /// (`demo.cpp:740-744`, `"Delta snapshot before any full snapshot"`).
    #[error("delta chunk before any full snapshot")]
    DeltaBeforeFullSnapshot,
    /// More than [`crate::reader::MAX_MESSAGES_PER_TICK`] message chunks were read for a single
    /// tick — this crate's own defensive cap (task 8.4b acceptance criterion 1's "chunk limits";
    /// not a DDNet-side check — real `CDemoPlayer` has no such limit and will grow its listener's
    /// own storage without bound). Found by review round 1 (finding F2): a hostile file packing
    /// one-byte-header, zero-size `MESSAGE` chunks between two tick markers has no other bound on
    /// how many `Msg` entries a single [`crate::reader::Tick`] accumulates.
    #[error("more than {0} message chunks in a single tick")]
    TooManyMessages(usize),
}
