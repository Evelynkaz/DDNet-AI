// Ported from DDNet `src/engine/shared/network.{h,cpp}` (pinned rev c9d208138f85755521f16a0096b6fe036c5c8698,
// "20.1"), which carries the original Teeworlds zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same header/flag/validity layout,
// so it stays byte-for-byte compatible with the wire format every DDNet 0.6 client/server speaks.
// See docs/formats.md for the byte layout.
//
// Scope note: DDNet 20.1's `network.{h,cpp}` is shared between the 0.6+DDNet protocol (what we
// implement) and 0.7/"sixup" (out of scope, see the task spec's goal). Everything sixup-specific
// (the `NET_PACKETFLAG_UNUSED`/token-in-header path, 6-bit chunk split, `protocol7::*`) is
// intentionally left out here.
//
//! Connection-oriented packet and chunk framing: header layout, validation, and Huffman
//! (de)compression, exactly as DDNet 20.1 speaks it over UDP for 0.6+DDNet. See
//! `docs/formats.md` §"Протокол 0.6+DDNet: низкий уровень" for the annotated byte layouts.
//!
//! This module is deliberately unaware of the DDNet security token — appending/stripping/
//! verifying the trailing 4-byte token happens one layer up, in [`crate::conn`], exactly where
//! DDNet's own `CNetConnection::Feed`/`Flush` do it (`network_conn.cpp`), because whether a token
//! is expected at all depends on connection *state*, not on anything visible in the packet bytes
//! themselves.

use crate::huffman::Huffman;

/// Maximum size of a single UDP datagram DDNet will send or accept (`NET_MAX_PACKETSIZE`).
pub const MAX_PACKET_SIZE: usize = 1400;
/// `NET_MAX_CONNLESS_PAYLOAD`.
pub const MAX_CONNLESS_PAYLOAD: usize = MAX_PACKET_SIZE - 6;
/// Maximum size of a single chunk's payload — 10 bits in the chunk header (`NET_MAX_CHUNK_SIZE`).
pub const MAX_CHUNK_SIZE: usize = 1023;
/// Chunk header size when it carries a sequence number (vital chunk).
pub const MAX_CHUNK_HEADER_SIZE: usize = 3;
/// Connection-oriented packet header size, before any trailing security token.
pub const PACKET_HEADER_SIZE: usize = 3;
/// Bytes of "extra data" on an extended connectionless packet (`NET_CONNLESS_EXTRA_SIZE`).
pub const CONNLESS_EXTRA_SIZE: usize = 4;
/// 10-bit vital sequence number space (`NET_MAX_SEQUENCE`).
pub const MAX_SEQUENCE: u16 = 1 << 10;
/// Maximum number of chunks in one packet — one byte in the header (`NET_MAX_PACKET_CHUNKS`).
pub const MAX_PACKET_CHUNKS: u8 = 0xFF;
/// Size of the trailing DDNet security token appended to (almost) every packet once negotiated.
pub const SECURITY_TOKEN_SIZE: usize = 4;
/// Maximum chunk-data payload a single packet can carry (`network.h`'s
/// `CNetPacketConstruct::m_aChunkData`): the packet size budget minus the 3-byte header.
pub const MAX_CHUNK_DATA_SIZE: usize = MAX_PACKET_SIZE - PACKET_HEADER_SIZE;

/// The `TKEN` magic DDNet appends/expects during the security-token handshake
/// (`SECURITY_TOKEN_MAGIC`, `network.cpp:19`).
pub const SECURITY_TOKEN_MAGIC: [u8; 4] = *b"TKEN";

/// Connection-oriented packet header flag bits (`NET_PACKETFLAG_*`, `network.h:85-92`; the
/// 0.6+DDNet-relevant subset only — `UNUSED`/`TOKEN` are sixup/vanilla-0.6.5 leftovers we never
/// set or expect).
pub mod packet_flags {
    /// Packet contains a single control message (`NET_CTRLMSG_*`) instead of chunks.
    pub const CONTROL: u8 = 1 << 2;
    /// Connectionless packet (server browser query/response, not a connection chunk carrier).
    pub const CONNLESS: u8 = 1 << 3;
    /// Ask the peer to resend every unacked vital chunk.
    pub const RESEND: u8 = 1 << 4;
    /// Chunk data is Huffman-compressed.
    pub const COMPRESSION: u8 = 1 << 5;
    /// The subset [`IsValidConnectionOrientedPacket`][super::is_valid_flags] allows.
    pub const VALID_MASK: u8 = CONTROL | RESEND | COMPRESSION;
}

/// Chunk header flag bits (`NET_CHUNKFLAG_*`, `network.h:94-95`).
pub mod chunk_flags {
    /// Delivered reliably: given a sequence number, retransmitted until acked.
    pub const VITAL: u8 = 1;
    /// Marks a chunk as a retransmission (set by the sender when resending, never by a fresh
    /// send); purely informational, DDNet does not treat it specially on receipt.
    pub const RESEND: u8 = 2;
}

/// A decoded chunk header, `network.h:156-165` / `network.cpp:443-469`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    pub flags: u8,
    pub size: u16,
    /// Only meaningful (and only present on the wire) when `flags & VITAL != 0`.
    pub sequence: u16,
}

impl ChunkHeader {
    /// Packs this header into `dst`, returning the number of bytes written (2, or 3 if
    /// [`chunk_flags::VITAL`] is set), or `None` if `dst` is too small.
    ///
    /// `split` matches DDNet's default of 4 (the 0.6 chunk-size split; sixup's split of 6 is out
    /// of scope, see the module docs) — exposed as a parameter only so the byte layout stays
    /// visibly parameterised the same way `CNetChunkHeader::Pack` is.
    pub fn pack(&self, dst: &mut [u8], split: u32) -> Option<usize> {
        let vital = self.flags & chunk_flags::VITAL != 0;
        let needed = if vital { 3 } else { 2 };
        if dst.len() < needed {
            return None;
        }
        dst[0] = ((self.flags & 3) << 6) | (((self.size >> split) & 0x3f) as u8);
        dst[1] = (self.size & ((1u16 << split) - 1)) as u8;
        if vital {
            dst[1] |= ((self.sequence >> 2) & !((1u16 << split) - 1)) as u8;
            dst[2] = (self.sequence & 0xff) as u8;
        }
        Some(needed)
    }

    /// Packs with the default (0.6) split of 4, matching `CNetChunkHeader::Pack(pData)`.
    pub fn pack_default(&self, dst: &mut [u8]) -> Option<usize> {
        self.pack(dst, 4)
    }

    /// Unpacks a chunk header from the start of `src`.
    ///
    /// Returns the header and the number of bytes consumed, or `None` if `src` is shorter than
    /// the *minimum* 2-byte header — this mirrors the low-level `CNetChunkHeader::Unpack`, which
    /// trusts the caller already checked there is room; [`ChunkIter`] is the bounds-safe
    /// wrapper that also checks room for a vital chunk's 3rd byte and for `size` bytes of
    /// payload, matching `CPacketChunkUnpacker::UnpackNextChunk` (`network.cpp:57-127`).
    pub fn unpack(src: &[u8], split: u32) -> Option<(ChunkHeader, usize)> {
        if src.len() < 2 {
            return None;
        }
        let flags = (src[0] >> 6) & 3;
        let size = (u16::from(src[0] & 0x3f) << split) | u16::from(src[1] & ((1u16 << split) - 1) as u8);
        if flags & chunk_flags::VITAL != 0 {
            if src.len() < 3 {
                return None;
            }
            let sequence = ((u16::from(src[1]) & !((1u16 << split) - 1)) << 2) | u16::from(src[2]);
            Some((ChunkHeader { flags, size, sequence }, 3))
        } else {
            Some((
                ChunkHeader {
                    flags,
                    size,
                    sequence: 0,
                },
                2,
            ))
        }
    }

    /// Unpacks with the default (0.6) split of 4, matching `CNetChunkHeader::Unpack(pData)`.
    pub fn unpack_default(src: &[u8]) -> Option<(ChunkHeader, usize)> {
        Self::unpack(src, 4)
    }
}

/// A single chunk extracted from a packet's chunk data by a [`ChunkIter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk<'a> {
    pub vital: bool,
    /// 10-bit vital sequence number; `0` (meaningless) for non-vital chunks.
    pub sequence: u16,
    pub data: &'a [u8],
}

/// Whether `flags` is a combination [`is_valid_connection_oriented_packet`] would accept for the
/// *packet* header (`CNetBase::IsValidConnectionOrientedPacket`, `network.cpp:134-155`) — flags
/// outside [`packet_flags::VALID_MASK`] are always rejected.
pub fn is_valid_flags(flags: u8) -> bool {
    flags & !packet_flags::VALID_MASK == 0
}

/// Validates packet-level structure (flags, chunk count) against DDNet's exact rules
/// (`CNetBase::IsValidConnectionOrientedPacket`, `network.cpp:134-155`): flags outside
/// [`packet_flags::VALID_MASK`] are rejected outright; a control packet must carry zero chunks,
/// at least one byte of payload, and must not be compressed; otherwise at least one chunk is
/// required unless [`packet_flags::RESEND`] is set (a bare resend request may carry zero
/// chunks), and never more than [`MAX_PACKET_CHUNKS`].
pub fn is_valid_connection_oriented_packet(flags: u8, num_chunks: u8, data_len: usize) -> bool {
    if !is_valid_flags(flags) {
        return false;
    }
    if flags & packet_flags::CONTROL != 0 {
        return num_chunks == 0 && data_len > 0 && flags & packet_flags::COMPRESSION == 0;
    }
    let min_chunks = u8::from(flags & packet_flags::RESEND == 0);
    num_chunks >= min_chunks
}

/// A parsed connection-oriented packet: header fields plus the (already decompressed, if it was
/// compressed) chunk-data payload. Does *not* include or strip a trailing security token — see
/// the module docs.
#[derive(Debug, Clone)]
pub struct Packet {
    pub flags: u8,
    pub ack: u16,
    pub num_chunks: u8,
    pub data: Vec<u8>,
}

/// Reason [`unpack_packet`] rejected a datagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UnpackPacketError {
    #[error("packet shorter than the connection-oriented header")]
    TooShort,
    #[error("packet larger than NET_MAX_PACKETSIZE")]
    TooLarge,
    #[error("invalid packet flags/chunk-count combination")]
    InvalidStructure,
    #[error("packet is compressed but decompression was not allowed for it")]
    DecompressionNotAllowed,
    #[error("Huffman decompression failed or produced more data than fits a packet")]
    DecompressionFailed,
    #[error("this is a connectionless packet, not a connection-oriented one")]
    Connless,
}

/// Parses a connection-oriented (non-connectionless) UDP datagram: header, flags/structure
/// validation, and Huffman decompression if the [`packet_flags::COMPRESSION`] bit is set.
///
/// `allow_decompression`, when `false`, rejects a compressed packet outright instead of paying
/// for decompression — mirrors `CNetBase::UnpackPacket`'s `AllowDecompression` parameter, DDNet's
/// defence against spending CPU decompressing packets from addresses that have not earned a
/// connection slot yet (`network_server.cpp`'s pre-connection decompression budget).
///
/// This function only handles the *packet* layer: it does not know about, strip, or verify a
/// trailing security token (see the module docs) — that is [`crate::conn::Connection`]'s job,
/// since whether a token is even expected depends on connection state.
pub fn unpack_packet(
    datagram: &[u8],
    huffman: &Huffman,
    allow_decompression: bool,
) -> Result<Packet, UnpackPacketError> {
    if datagram.len() < PACKET_HEADER_SIZE {
        return Err(UnpackPacketError::TooShort);
    }
    if datagram.len() > MAX_PACKET_SIZE {
        return Err(UnpackPacketError::TooLarge);
    }
    let flags = datagram[0] >> 2;
    if flags & packet_flags::CONNLESS != 0 {
        return Err(UnpackPacketError::Connless);
    }

    let ack = (u16::from(datagram[0] & 0x3) << 8) | u16::from(datagram[1]);
    let num_chunks = datagram[2];
    let payload = &datagram[PACKET_HEADER_SIZE..];

    if !is_valid_connection_oriented_packet(flags, num_chunks, payload.len()) {
        return Err(UnpackPacketError::InvalidStructure);
    }

    let data = if flags & packet_flags::COMPRESSION != 0 {
        if !allow_decompression {
            return Err(UnpackPacketError::DecompressionNotAllowed);
        }
        huffman
            .decompress_vec(payload, MAX_CHUNK_DATA_SIZE)
            .ok_or(UnpackPacketError::DecompressionFailed)?
    } else {
        payload.to_vec()
    };

    Ok(Packet {
        flags,
        ack,
        num_chunks,
        data,
    })
}

/// Builds the wire bytes for a connection-oriented packet: appends the trailing DDNet security
/// token (if `security_token` is `Some`) to `chunk_data`, Huffman-compresses the result if that
/// is both successful and smaller, sets [`packet_flags::COMPRESSION`] accordingly, and writes the
/// 3-byte header — mirrors `CNetBase::SendPacket` (`network.cpp:193-270`). `flags` must not
/// already include [`packet_flags::COMPRESSION`], matching the `dbg_assert` in the C++
/// reference.
///
/// The token is appended unconditionally whenever `security_token` is `Some` — including while
/// it is still [`crate::conn::SecurityToken::Unknown`]'s wire value, which is how the very first
/// `CONNECT` gets its `ff ff ff ff` tail (`network_conn.cpp:365-377`'s comment: "if SecurityToken
/// is NET_SECURITY_TOKEN_UNKNOWN we will still append it hoping to negotiate it"). Passing `None`
/// (rather than a sentinel value) is how a connection with a negotiated-`Unsupported` token skips
/// the append entirely, matching the `NET_SECURITY_TOKEN_UNSUPPORTED` check in the same C++
/// function — this is the one piece of the handshake that lives in this module rather than
/// [`crate::conn`], since it has to happen before compression, deep inside packet construction.
///
/// Returns `None` if the result would exceed [`MAX_PACKET_SIZE`].
pub fn build_packet(
    flags: u8,
    ack: u16,
    num_chunks: u8,
    chunk_data: &[u8],
    security_token: Option<u32>,
    huffman: &Huffman,
) -> Option<Vec<u8>> {
    debug_assert!(flags & packet_flags::COMPRESSION == 0);
    let header_size = PACKET_HEADER_SIZE;

    let mut data = chunk_data.to_vec();
    if let Some(token) = security_token {
        data.extend_from_slice(&token.to_be_bytes());
    }

    let (final_flags, payload): (u8, &[u8]) = if flags & packet_flags::CONTROL == 0 {
        // `compress_vec` always succeeds size-wise (grows its own scratch buffer), so "successful
        // and smaller" is just a length comparison here, matching `network.cpp:232-244`.
        let compressed = huffman.compress_vec(&data);
        if !compressed.is_empty() && compressed.len() < data.len() {
            return build_packet_with_payload(
                flags | packet_flags::COMPRESSION,
                ack,
                num_chunks,
                &compressed,
                header_size,
            );
        }
        (flags, &data)
    } else {
        (flags, &data)
    };

    build_packet_with_payload(final_flags, ack, num_chunks, payload, header_size)
}

fn build_packet_with_payload(
    flags: u8,
    ack: u16,
    num_chunks: u8,
    payload: &[u8],
    header_size: usize,
) -> Option<Vec<u8>> {
    if header_size + payload.len() > MAX_PACKET_SIZE {
        return None;
    }
    let mut out = Vec::with_capacity(header_size + payload.len());
    out.push(((flags << 2) & 0xfc) | ((ack >> 8) & 0x3) as u8);
    out.push((ack & 0xff) as u8);
    out.push(num_chunks);
    out.extend_from_slice(payload);
    Some(out)
}

/// Iterates the chunks packed into a [`Packet`]'s `data`, mirroring `CPacketChunkUnpacker`
/// (`network.cpp:43-132`) — minus its sequence-number/ack bookkeeping, which lives in
/// [`crate::conn`] since it needs connection state (`m_Ack`, `SignalResend`) this module does not
/// have. This iterator only frames chunks; it never fails on well-formed input and simply stops
/// (yields no more items) as soon as the remaining bytes cannot hold another full chunk header +
/// payload, exactly like `UnpackNextChunk` returning `false`.
pub struct ChunkIter<'a> {
    remaining: &'a [u8],
    chunks_left: u8,
}

impl<'a> ChunkIter<'a> {
    pub fn new(packet: &'a Packet) -> Self {
        ChunkIter {
            remaining: &packet.data,
            chunks_left: packet.num_chunks,
        }
    }
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Chunk<'a>> {
        if self.chunks_left == 0 {
            return None;
        }
        let (header, header_len) = ChunkHeader::unpack_default(self.remaining)?;
        let body_start = header_len;
        let body_end = body_start.checked_add(header.size as usize)?;
        if body_end > self.remaining.len() {
            self.chunks_left = 0;
            return None;
        }
        let data = &self.remaining[body_start..body_end];
        self.remaining = &self.remaining[body_end..];
        self.chunks_left -= 1;
        Some(Chunk {
            vital: header.flags & chunk_flags::VITAL != 0,
            sequence: header.sequence,
            data,
        })
    }
}

/// Packs one chunk (header + payload) onto the end of `dst`'s already-written prefix
/// (`dst[..*pos]`), advancing `*pos`. Returns `false` (leaving `dst`/`pos` untouched) if it would
/// not fit — mirrors the chunk-packing half of `CNetConnection::QueueChunkEx`
/// (`network_conn.cpp:143-195`), minus the resend-buffer bookkeeping (connection-state, lives in
/// [`crate::conn`]).
pub fn pack_chunk(dst: &mut [u8], pos: &mut usize, flags: u8, sequence: u16, data: &[u8]) -> bool {
    let header = ChunkHeader {
        flags,
        size: data.len() as u16,
        sequence,
    };
    let mut header_buf = [0u8; MAX_CHUNK_HEADER_SIZE];
    let Some(header_len) = header.pack_default(&mut header_buf) else {
        return false;
    };
    if *pos + header_len + data.len() > dst.len() {
        return false;
    }
    dst[*pos..*pos + header_len].copy_from_slice(&header_buf[..header_len]);
    *pos += header_len;
    dst[*pos..*pos + data.len()].copy_from_slice(data);
    *pos += data.len();
    true
}

/// Like [`pack_chunk`], but appends to a growable `Vec<u8>` capped at `capacity` bytes instead of
/// a fixed slice — what [`crate::conn::Connection`] uses to build up a packet-in-progress, since
/// it does not know ahead of time how many chunks will be queued before the next flush.
pub fn pack_chunk_into(dst: &mut Vec<u8>, capacity: usize, flags: u8, sequence: u16, data: &[u8]) -> bool {
    let header = ChunkHeader {
        flags,
        size: data.len() as u16,
        sequence,
    };
    let mut header_buf = [0u8; MAX_CHUNK_HEADER_SIZE];
    let Some(header_len) = header.pack_default(&mut header_buf) else {
        return false;
    };
    if dst.len() + header_len + data.len() > capacity {
        return false;
    }
    dst.extend_from_slice(&header_buf[..header_len]);
    dst.extend_from_slice(data);
    true
}

/// The `Bottom..=Ack` backroom window DDNet uses to recognise a duplicate/already-acked vital
/// chunk versus one that is genuinely out of order and needs a resend request
/// (`CNetBase::IsSeqInBackroom`, `network.cpp:471-488`). Correctly handles wraparound of the
/// 10-bit sequence space (`MAX_SEQUENCE`).
pub fn is_seq_in_backroom(seq: u16, ack: u16) -> bool {
    let bottom = i32::from(ack) - i32::from(MAX_SEQUENCE) / 2;
    if bottom < 0 {
        seq as i32 <= ack as i32 || seq as i32 >= bottom + i32::from(MAX_SEQUENCE)
    } else {
        seq as i32 <= ack as i32 && seq as i32 >= bottom
    }
}

/// A connectionless packet's parsed form: server-browser-style queries/responses that carry no
/// connection state at all (`network.cpp:157-191, 313-340`).
#[derive(Debug, Clone)]
pub struct ConnlessPacket {
    /// `Some` for the "extended" `b"xe" + 4 bytes` variant (used by the server browser), `None`
    /// for the plain `0xFF * 6` variant.
    pub extra_data: Option<[u8; CONNLESS_EXTRA_SIZE]>,
    pub data: Vec<u8>,
}

const CONNLESS_EXTENDED_MAGIC: [u8; 2] = *b"xe";

/// Parses a connectionless datagram (caller has already determined `flags & CONNLESS != 0`; see
/// [`unpack_packet`]'s sibling check). Mirrors the connectionless branch of
/// `CNetBase::UnpackPacket` (`network.cpp:313-340`), the non-sixup path only.
pub fn unpack_connless_packet(datagram: &[u8]) -> Result<ConnlessPacket, UnpackPacketError> {
    const OFFSET: usize = 6;
    if datagram.len() < OFFSET {
        return Err(UnpackPacketError::TooShort);
    }
    if datagram.len() > MAX_PACKET_SIZE {
        return Err(UnpackPacketError::TooLarge);
    }
    let extra_data = if datagram[0..2] == CONNLESS_EXTENDED_MAGIC {
        let mut extra = [0u8; CONNLESS_EXTRA_SIZE];
        extra.copy_from_slice(&datagram[2..6]);
        Some(extra)
    } else {
        None
    };
    Ok(ConnlessPacket {
        extra_data,
        data: datagram[OFFSET..].to_vec(),
    })
}

/// Builds the wire bytes for a connectionless packet (`CNetBase::SendPacketConnless`,
/// `network.cpp:159-177`). `payload.len()` must be `<= MAX_CONNLESS_PAYLOAD - `(2 bytes less if
/// `extra_data` is `Some`, matching the C++ `dbg_assert`); returns `None` if it does not fit.
pub fn build_connless_packet(extra_data: Option<[u8; CONNLESS_EXTRA_SIZE]>, payload: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(6 + payload.len());
    match extra_data {
        Some(extra) => {
            out.extend_from_slice(&CONNLESS_EXTENDED_MAGIC);
            out.extend_from_slice(&extra);
        }
        None => out.extend_from_slice(&[0xFFu8; 6]),
    }
    if out.len() + payload.len() > MAX_PACKET_SIZE {
        return None;
    }
    out.extend_from_slice(payload);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ChunkHeader, ported from `src/test/chunk_header_test.cpp` (20.1) ---

    fn assert_header(header: &ChunkHeader, expected: &[u8]) {
        let mut buf = [0u8; 8];
        let n = header.pack_default(&mut buf).unwrap();
        assert_eq!(n, expected.len());
        assert_eq!(&buf[..n], expected);

        let (unpacked, consumed) = ChunkHeader::unpack_default(&buf[..n]).unwrap();
        assert_eq!(consumed, n);
        assert_eq!(unpacked, *header);
    }

    #[test]
    fn chunk_header_seq255() {
        // chunk_header_test.cpp: ChunkHeader.Seq255
        assert_header(
            &ChunkHeader {
                flags: chunk_flags::VITAL,
                size: 0,
                sequence: 255,
            },
            &[0x40, 0x30, 0xff],
        );
    }

    #[test]
    fn chunk_header_seq126() {
        // chunk_header_test.cpp: ChunkHeader.Seq126
        assert_header(
            &ChunkHeader {
                flags: chunk_flags::VITAL,
                size: 0,
                sequence: 126,
            },
            &[0x40, 0x10, 0x7e],
        );
    }

    #[test]
    fn chunk_header_seq63() {
        // chunk_header_test.cpp: ChunkHeader.Seq63
        assert_header(
            &ChunkHeader {
                flags: chunk_flags::VITAL,
                size: 0,
                sequence: 63,
            },
            &[0x40, 0x00, 0x3f],
        );
    }

    #[test]
    fn chunk_header_seq64() {
        // chunk_header_test.cpp: ChunkHeader.Seq64
        assert_header(
            &ChunkHeader {
                flags: chunk_flags::VITAL,
                size: 0,
                sequence: 64,
            },
            &[0x40, 0x10, 0x40],
        );
    }

    #[test]
    fn chunk_header_seq5() {
        // chunk_header_test.cpp: ChunkHeader.Seq5
        assert_header(
            &ChunkHeader {
                flags: chunk_flags::VITAL,
                size: 0,
                sequence: 5,
            },
            &[0x40, 0x00, 0x05],
        );
    }

    #[test]
    fn chunk_header_nonvital_roundtrip() {
        let header = ChunkHeader {
            flags: 0,
            size: 777,
            sequence: 0,
        };
        let mut buf = [0u8; 8];
        let n = header.pack_default(&mut buf).unwrap();
        assert_eq!(n, 2);
        let (unpacked, consumed) = ChunkHeader::unpack_default(&buf[..n]).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(unpacked.size, 777);
        assert_eq!(unpacked.flags, 0);
    }

    // --- Network packet tests, ported from `src/test/network_test.cpp` (20.1) ---

    fn header_byte0(flags: u8, ack: u16) -> u8 {
        ((flags << 2) & 0xfc) | ((ack >> 8) & 0x3) as u8
    }

    #[test]
    fn unpack_maximum_uncompressed_packet() {
        // network_test.cpp: Network.UnpackMaximumUncompressedPacket
        let huffman = Huffman::new();
        let mut datagram = vec![0u8; PACKET_HEADER_SIZE + MAX_CHUNK_DATA_SIZE];
        datagram[0] = header_byte0(0, 0);
        datagram[1] = 0;
        datagram[2] = 1; // one chunk
        let packet = unpack_packet(&datagram, &huffman, true).unwrap();
        assert_eq!(packet.data.len(), MAX_CHUNK_DATA_SIZE);

        let mut datagram2 = vec![0u8; MAX_PACKET_SIZE];
        datagram2[2] = 1;
        assert!(unpack_packet(&datagram2, &huffman, true).is_ok());
    }

    #[test]
    fn unpack_oversized_uncompressed_packet() {
        // network_test.cpp: Network.UnpackOversizedUncompressedPacket
        let huffman = Huffman::new();
        let mut datagram = vec![0u8; PACKET_HEADER_SIZE + MAX_CHUNK_DATA_SIZE + 1];
        datagram[2] = 1;
        assert_eq!(
            unpack_packet(&datagram, &huffman, true).unwrap_err(),
            UnpackPacketError::TooLarge
        );
    }

    fn compressed_datagram(huffman: &Huffman, payload_size: usize) -> Vec<u8> {
        let payload = vec![0u8; payload_size];
        let compressed = huffman.compress_vec(&payload);
        let mut datagram = vec![0u8; PACKET_HEADER_SIZE];
        datagram[0] = header_byte0(packet_flags::COMPRESSION, 0);
        datagram[1] = 0;
        datagram[2] = 1;
        datagram.extend_from_slice(&compressed);
        datagram
    }

    #[test]
    fn unpack_compressed_packet() {
        // network_test.cpp: Network.UnpackCompressedPacket
        let huffman = Huffman::new();
        let datagram = compressed_datagram(&huffman, 64);
        let packet = unpack_packet(&datagram, &huffman, true).unwrap();
        assert_eq!(packet.data.len(), 64);
    }

    #[test]
    fn unpack_compressed_packet_without_decompression() {
        // network_test.cpp: Network.UnpackCompressedPacketWithoutDecompression
        let huffman = Huffman::new();
        let datagram = compressed_datagram(&huffman, 64);
        assert_eq!(
            unpack_packet(&datagram, &huffman, false).unwrap_err(),
            UnpackPacketError::DecompressionNotAllowed
        );
    }

    #[test]
    fn unpack_oversized_compressed_packet() {
        // network_test.cpp: Network.UnpackOversizedCompressedPacket
        let huffman = Huffman::new();
        let datagram = compressed_datagram(&huffman, MAX_CHUNK_DATA_SIZE + 1);
        assert_eq!(
            unpack_packet(&datagram, &huffman, true).unwrap_err(),
            UnpackPacketError::DecompressionFailed
        );
    }

    #[test]
    fn unpack_uncompressed_packet_without_decompression() {
        // network_test.cpp: Network.UnpackUncompressedPacketWithoutDecompression
        let huffman = Huffman::new();
        let mut datagram = vec![0u8; MAX_PACKET_SIZE];
        datagram[2] = 1;
        assert!(unpack_packet(&datagram, &huffman, false).is_ok());
    }

    #[test]
    fn unpack_chunks() {
        // network_test.cpp: Network.UnpackChunks — three non-vital chunks of different sizes,
        // each filled with its own byte.
        let sizes = [1usize, 100, 7];
        let mut data = Vec::new();
        let mut pos = 0usize;
        let mut buf = vec![0u8; 1200];
        for (i, &size) in sizes.iter().enumerate() {
            let chunk_data = vec![i as u8; size];
            assert!(pack_chunk(&mut buf, &mut pos, 0, 0, &chunk_data));
        }
        data.extend_from_slice(&buf[..pos]);

        let packet = Packet {
            flags: 0,
            ack: 0,
            num_chunks: sizes.len() as u8,
            data,
        };
        let mut iter = ChunkIter::new(&packet);
        for (i, &size) in sizes.iter().enumerate() {
            let chunk = iter.next().unwrap();
            assert_eq!(chunk.data.len(), size);
            assert!(chunk.data.iter().all(|&b| b == i as u8));
        }
        assert!(iter.next().is_none());
    }

    #[test]
    fn build_and_unpack_roundtrip_uncompressible() {
        let huffman = Huffman::new();
        // High-entropy-ish data that Huffman won't shrink (all distinct rare bytes) still must
        // round-trip: `build_packet` falls back to uncompressed.
        let mut chunk_data = Vec::new();
        let mut pos = 0usize;
        let mut buf = vec![0u8; 64];
        assert!(pack_chunk(&mut buf, &mut pos, chunk_flags::VITAL, 1, b"hello"));
        chunk_data.extend_from_slice(&buf[..pos]);

        let datagram = build_packet(0, 7, 1, &chunk_data, None, &huffman).unwrap();
        let packet = unpack_packet(&datagram, &huffman, true).unwrap();
        assert_eq!(packet.ack, 7);
        assert_eq!(packet.num_chunks, 1);
        assert_eq!(packet.data, chunk_data);

        let chunk = ChunkIter::new(&packet).next().unwrap();
        assert!(chunk.vital);
        assert_eq!(chunk.sequence, 1);
        assert_eq!(chunk.data, b"hello");
    }

    #[test]
    fn build_packet_prefers_compression_when_smaller() {
        let huffman = Huffman::new();
        let chunk_data = vec![0u8; 512]; // all zero: symbol 0 has the shortest code, compresses well
        let datagram = build_packet(0, 0, 1, &chunk_data, None, &huffman).unwrap();
        assert!(datagram.len() < PACKET_HEADER_SIZE + chunk_data.len());
        let flags = datagram[0] >> 2;
        assert_ne!(flags & packet_flags::COMPRESSION, 0);
        let packet = unpack_packet(&datagram, &huffman, true).unwrap();
        assert_eq!(packet.data, chunk_data);
    }

    #[test]
    fn control_packet_never_compressed() {
        let huffman = Huffman::new();
        let payload = vec![0u8; 512]; // would otherwise compress
        let datagram = build_packet(packet_flags::CONTROL, 0, 0, &payload, None, &huffman).unwrap();
        let flags = datagram[0] >> 2;
        assert_eq!(flags & packet_flags::COMPRESSION, 0);
    }

    #[test]
    fn build_packet_appends_security_token_before_compression() {
        let huffman = Huffman::new();
        let datagram = build_packet(
            packet_flags::CONTROL,
            0,
            0,
            &[1u8, b'T', b'K', b'E', b'N'],
            Some(0xffff_ffff),
            &huffman,
        )
        .unwrap();
        let packet = unpack_packet(&datagram, &huffman, true).unwrap();
        // Control packets are never compressed, so the trailing token is visible verbatim.
        assert_eq!(packet.data, [1u8, b'T', b'K', b'E', b'N', 0xff, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn build_packet_without_security_token_appends_nothing() {
        let huffman = Huffman::new();
        let datagram = build_packet(packet_flags::CONTROL, 0, 0, &[0u8], None, &huffman).unwrap();
        let packet = unpack_packet(&datagram, &huffman, true).unwrap();
        assert_eq!(packet.data, [0u8]);
    }

    #[test]
    fn is_valid_connection_oriented_packet_rules() {
        assert!(is_valid_connection_oriented_packet(packet_flags::CONTROL, 0, 1));
        assert!(!is_valid_connection_oriented_packet(packet_flags::CONTROL, 1, 1));
        assert!(!is_valid_connection_oriented_packet(packet_flags::CONTROL, 0, 0));
        assert!(!is_valid_connection_oriented_packet(
            packet_flags::CONTROL | packet_flags::COMPRESSION,
            0,
            1
        ));
        assert!(is_valid_connection_oriented_packet(0, 1, 10));
        assert!(!is_valid_connection_oriented_packet(0, 0, 10));
        assert!(is_valid_connection_oriented_packet(packet_flags::RESEND, 0, 0));
        assert!(!is_valid_connection_oriented_packet(0x80, 0, 1));
        assert!(!is_valid_connection_oriented_packet(0, 0, 0));
    }

    #[test]
    fn is_seq_in_backroom_matches_reference_semantics() {
        // Straightforward, no-wraparound case.
        assert!(is_seq_in_backroom(5, 10));
        assert!(!is_seq_in_backroom(11, 10));
        // At ack exactly.
        assert!(is_seq_in_backroom(10, 10));
        // Wraparound: ack near zero, backroom window wraps to the top of the sequence space.
        assert!(is_seq_in_backroom(MAX_SEQUENCE - 1, 5));
        assert!(!is_seq_in_backroom(MAX_SEQUENCE / 2, 5));
    }

    #[test]
    fn connless_roundtrip_plain() {
        let datagram = build_connless_packet(None, b"gie3token").unwrap();
        let parsed = unpack_connless_packet(&datagram).unwrap();
        assert!(parsed.extra_data.is_none());
        assert_eq!(parsed.data, b"gie3token");
    }

    #[test]
    fn connless_roundtrip_extended() {
        let datagram = build_connless_packet(Some([1, 2, 3, 4]), b"payload").unwrap();
        let parsed = unpack_connless_packet(&datagram).unwrap();
        assert_eq!(parsed.extra_data, Some([1, 2, 3, 4]));
        assert_eq!(parsed.data, b"payload");
    }

    #[test]
    fn unpack_packet_rejects_connless_flag() {
        let huffman = Huffman::new();
        let mut datagram = vec![0u8; PACKET_HEADER_SIZE + 4];
        datagram[0] = header_byte0(packet_flags::CONNLESS, 0);
        assert_eq!(
            unpack_packet(&datagram, &huffman, true).unwrap_err(),
            UnpackPacketError::Connless
        );
    }

    #[test]
    fn unpack_packet_too_short() {
        let huffman = Huffman::new();
        assert_eq!(
            unpack_packet(&[0, 0], &huffman, true).unwrap_err(),
            UnpackPacketError::TooShort
        );
    }
}
