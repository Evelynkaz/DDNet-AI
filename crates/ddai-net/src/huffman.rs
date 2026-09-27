// Ported from DDNet `src/engine/shared/huffman.{h,cpp}` (pinned rev c9d208138f85755521f16a0096b6fe036c5c8698,
// "20.1"), which carries the original Teeworlds zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same algorithm, same fixed
// frequency table and node layout, so it stays byte-for-byte compatible with the wire format
// every DDNet 0.6 client/server speaks. See docs/formats.md for the byte layout.
//
//! DDNet's fixed Huffman code for compressing packet payloads.
//!
//! The tree is built once (in [`Huffman::new`]) from a fixed frequency table (256 byte values
//! plus one EOF symbol) using the exact same greedy merge DDNet's C++ implementation uses, so the
//! resulting bit patterns match the C++ encoder/decoder exactly. `compress`/`decompress` never
//! panic and never loop unboundedly on malformed input: both bound their work by the size of the
//! output buffer they are given (`decompress` in particular is exercised with fuzzed/mutated
//! input in `tests/robustness.rs` — the old TypeScript bot hung on malformed Huffman streams).

use std::fmt;

/// End-of-stream symbol: one past the 256 possible byte values.
const EOF_SYMBOL: u16 = 256;
/// 256 byte values + EOF.
const NUM_SYMBOLS: usize = EOF_SYMBOL as usize + 1;
/// A full binary tree with `NUM_SYMBOLS` leaves has `NUM_SYMBOLS - 1` internal nodes.
const NUM_NODES: usize = NUM_SYMBOLS * 2 - 1;

const LUT_BITS: u32 = 10;
const LUT_SIZE: usize = 1 << LUT_BITS;
const LUT_MASK: usize = LUT_SIZE - 1;

/// Sentinel "no child" marker, matching the C++ `0xffff` sentinel.
const NO_CHILD: u16 = 0xffff;

/// DDNet's fixed Huffman frequency table for the 256 byte values, `huffman.cpp:11-24` (20.1).
/// Frequency 0 is the (fictitious) EOF symbol slot's placeholder; the real EOF frequency (1) is
/// appended separately in [`Huffman::new`], exactly mirroring `ms_aFreqTable[HUFFMAN_EOF_SYMBOL]`
/// in the C++ table (last entry, value `1`).
#[rustfmt::skip]
const FREQ_TABLE: [u32; NUM_SYMBOLS] = [
    1 << 30, 4545, 2657, 431, 1950, 919, 444, 482, 2244, 617, 838, 542, 715, 1814, 304, 240, 754, 212, 647, 186,
    283, 131, 146, 166, 543, 164, 167, 136, 179, 859, 363, 113, 157, 154, 204, 108, 137, 180, 202, 176,
    872, 404, 168, 134, 151, 111, 113, 109, 120, 126, 129, 100, 41, 20, 16, 22, 18, 18, 17, 19,
    16, 37, 13, 21, 362, 166, 99, 78, 95, 88, 81, 70, 83, 284, 91, 187, 77, 68, 52, 68,
    59, 66, 61, 638, 71, 157, 50, 46, 69, 43, 11, 24, 13, 19, 10, 12, 12, 20, 14, 9,
    20, 20, 10, 10, 15, 15, 12, 12, 7, 19, 15, 14, 13, 18, 35, 19, 17, 14, 8, 5,
    15, 17, 9, 15, 14, 18, 8, 10, 2173, 134, 157, 68, 188, 60, 170, 60, 194, 62, 175, 71,
    148, 67, 167, 78, 211, 67, 156, 69, 1674, 90, 174, 53, 147, 89, 181, 51, 174, 63, 163, 80,
    167, 94, 128, 122, 223, 153, 218, 77, 200, 110, 190, 73, 174, 69, 145, 66, 277, 143, 141, 60,
    136, 53, 180, 57, 142, 57, 158, 61, 166, 112, 152, 92, 26, 22, 21, 28, 20, 26, 30, 21,
    32, 27, 20, 17, 23, 21, 30, 22, 22, 21, 27, 25, 17, 27, 23, 18, 39, 26, 15, 21,
    12, 18, 18, 27, 20, 18, 15, 19, 11, 17, 33, 12, 18, 15, 19, 18, 16, 26, 17, 18,
    9, 10, 25, 22, 22, 17, 20, 16, 6, 16, 15, 20, 14, 18, 24, 335, 1,
];

#[derive(Clone, Copy)]
struct Node {
    /// Bit pattern for this symbol, LSB first (only meaningful on leaves, `num_bits > 0`).
    bits: u32,
    /// Number of bits in `bits`; `0` marks an internal node.
    num_bits: u32,
    /// Child node indices (`[0]`, `[1]`); `NO_CHILD` for leaves.
    leaves: [u16; 2],
    /// The byte value this leaf decodes to (`EOF_SYMBOL` for the EOF leaf); unused on internal
    /// nodes.
    symbol: u16,
}

const EMPTY_NODE: Node = Node {
    bits: 0,
    num_bits: 0,
    leaves: [NO_CHILD, NO_CHILD],
    symbol: 0,
};

/// A fixed Huffman code, built once and reused for every packet.
///
/// Cheap to construct (`Huffman::new()` runs the same fixed-table build DDNet's `CHuffman::Init`
/// does) and `Send + Sync`, so callers typically build one and share it (e.g. behind an `Arc` or
/// in a `static` via `std::sync::OnceLock`).
pub struct Huffman {
    nodes: [Node; NUM_NODES],
    /// Index of the tree root in `nodes`.
    start_node: u16,
    /// Direct lookup table indexed by the low `LUT_BITS` bits of the input: either the decoded
    /// leaf (if it is reached within `LUT_BITS` bits) or the internal node reached after
    /// consuming `LUT_BITS` bits (tree walk continues bit-by-bit from there).
    decode_lut: [u16; LUT_SIZE],
}

impl fmt::Debug for Huffman {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Huffman").finish_non_exhaustive()
    }
}

impl Default for Huffman {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper node used only while building the tree: a (frequency, node-index) pair sorted by
/// frequency, mirroring `CHuffmanConstructNode` in `huffman.cpp`.
#[derive(Clone, Copy)]
struct BuildNode {
    node_id: u16,
    frequency: u64,
}

impl Huffman {
    /// Builds the fixed DDNet Huffman code. Deterministic: always produces the exact same tree.
    pub fn new() -> Self {
        let mut nodes = [EMPTY_NODE; NUM_NODES];
        for (i, node) in nodes.iter_mut().enumerate().take(NUM_SYMBOLS) {
            // Sentinel "not yet assigned a code" for leaves — deliberately *not* 0, which is
            // reserved to mark internal (merged) nodes below. `SetBitsRecursive`/`set_bits_recursive`
            // relies on `num_bits != 0` to tell "this is a symbol, give it its real depth" apart
            // from "this is an internal node, leave it alone", exactly like the C++
            // `m_NumBits = 0xFFFFFFFF` sentinel in `ConstructTree` (`huffman.cpp:61`).
            node.num_bits = u32::MAX;
            node.symbol = i as u16;
            node.leaves = [NO_CHILD, NO_CHILD];
        }

        let mut left: Vec<BuildNode> = (0..NUM_SYMBOLS)
            .map(|i| BuildNode {
                node_id: i as u16,
                frequency: u64::from(FREQ_TABLE[i]),
            })
            .collect();

        let mut num_nodes = NUM_SYMBOLS;
        while left.len() > 1 {
            // Stable sort descending by frequency, exactly like `std::stable_sort` in the C++
            // reference: ties keep their relative order, which affects the resulting bit
            // patterns and must match bit-for-bit.
            left.sort_by_key(|node| std::cmp::Reverse(node.frequency));

            let last = left.len() - 1;
            let a = left[last];
            let b = left[last - 1];
            nodes[num_nodes] = Node {
                bits: 0,
                num_bits: 0,
                leaves: [a.node_id, b.node_id],
                symbol: 0,
            };
            left[last - 1] = BuildNode {
                node_id: num_nodes as u16,
                frequency: a.frequency + b.frequency,
            };
            left.pop();
            num_nodes += 1;
        }

        let start_node = (num_nodes - 1) as u16;
        set_bits_recursive(&mut nodes, start_node, 0, 0);

        let mut huffman = Huffman {
            nodes,
            start_node,
            decode_lut: [0; LUT_SIZE],
        };
        huffman.build_decode_lut();
        huffman
    }

    fn build_decode_lut(&mut self) {
        for i in 0..LUT_SIZE {
            let mut bits = i as u32;
            let mut node_idx = self.start_node;
            for _ in 0..LUT_BITS {
                node_idx = self.nodes[node_idx as usize].leaves[(bits & 1) as usize];
                bits >>= 1;
                if self.nodes[node_idx as usize].num_bits != 0 {
                    break;
                }
            }
            self.decode_lut[i] = node_idx;
        }
    }

    /// Compresses `input`, appending the trailing EOF symbol, into `output`.
    ///
    /// Returns the number of bytes written, or [`None`] if `output` is not large enough — the
    /// caller is expected to fall back to sending the data uncompressed, exactly as DDNet does
    /// (`network.cpp:232-244`: use the compressed form only if it is both successful and
    /// smaller).
    pub fn compress(&self, input: &[u8], output: &mut [u8]) -> Option<usize> {
        let mut bits: u64 = 0;
        let mut bit_count: u32 = 0;
        let mut out_pos = 0usize;

        macro_rules! load_symbol {
            ($sym:expr) => {{
                let node = &self.nodes[$sym as usize];
                bits |= u64::from(node.bits) << bit_count;
                bit_count += node.num_bits;
            }};
        }

        // Writes out every full byte currently buffered in `bits`/`bit_count`. Safe, bounds-checked
        // byte-at-a-time (unlike the C++ reference's 8-byte-headroom fast path) — this layer's
        // hot path is the physics/game logic above it, not this framing, and there is no
        // `unsafe` budget here per the task constraints.
        macro_rules! flush_bytes {
            () => {{
                while bit_count >= 8 {
                    if out_pos == output.len() {
                        return None;
                    }
                    output[out_pos] = (bits & 0xff) as u8;
                    out_pos += 1;
                    bits >>= 8;
                    bit_count -= 8;
                }
            }};
        }

        for &byte in input {
            load_symbol!(byte as u16);
            flush_bytes!();
        }
        load_symbol!(EOF_SYMBOL);
        flush_bytes!();

        if bit_count != 0 {
            if out_pos == output.len() {
                return None;
            }
            output[out_pos] = (bits & 0xff) as u8;
            out_pos += 1;
        }

        Some(out_pos)
    }

    /// Compresses into a freshly allocated `Vec<u8>`.
    pub fn compress_vec(&self, input: &[u8]) -> Vec<u8> {
        // Worst case: every byte needs its longest code (bounded by the tree depth), plus EOF.
        // `input.len() * 4 + 8` is comfortably above anything the fixed table can produce.
        let mut out = vec![0u8; input.len() * 4 + 8];
        let n = self
            .compress(input, &mut out)
            .expect("scratch buffer sized generously above");
        out.truncate(n);
        out
    }

    /// Decompresses `input` into `output`, stopping at the EOF symbol.
    ///
    /// Bounded by construction: each iteration of the outer loop either consumes at least one
    /// bit of input or returns, and decoding a symbol never revisits already-consumed bits, so
    /// malformed input can make this return `None` but can never loop forever or read/write past
    /// the given slices — no `unsafe`, no panics, regardless of how `input` was mutated.
    ///
    /// Returns the number of bytes written, or `None` if `input` is not a valid compression
    /// (ran out of bits without hitting EOF, hit an impossible LUT entry, or the decoded data
    /// does not fit in `output`).
    pub fn decompress(&self, input: &[u8], output: &mut [u8]) -> Option<usize> {
        let mut src = input.iter();
        let mut bits: u32 = 0;
        let mut bit_count: u32 = 0;
        let mut out_pos = 0usize;

        let eof_idx = EOF_SYMBOL;

        loop {
            // {A} try to load a node from bits we already have, before topping up (mirrors the
            // C++ ordering, which hides the LUT-index latency behind the fill loop).
            let early_node = (bit_count >= LUT_BITS).then(|| self.decode_lut[(bits as usize) & LUT_MASK]);

            // {B} fill with new bits.
            while bit_count < 24 {
                match src.next() {
                    Some(&byte) => {
                        bits |= u32::from(byte) << bit_count;
                        bit_count += 8;
                    }
                    None => break,
                }
            }

            // {C} load the node now if {A} could not.
            let mut node_idx = early_node.unwrap_or(self.decode_lut[(bits as usize) & LUT_MASK]);

            // {D} check whether the LUT already resolved a full symbol.
            let node = &self.nodes[node_idx as usize];
            if node.num_bits != 0 {
                if bit_count < node.num_bits {
                    return None;
                }
                bits = shr_u32(bits, node.num_bits);
                bit_count -= node.num_bits;
            } else {
                if bit_count < LUT_BITS {
                    return None;
                }
                bits = shr_u32(bits, LUT_BITS);
                bit_count -= LUT_BITS;

                // Walk the tree bit by bit beyond what the LUT covered. `bit_count` is allowed
                // to wrap past zero here exactly like the C++ `unsigned` reference: child indices
                // strictly decrease every step (the tree is built bottom-up), and every original
                // byte/EOF symbol has a non-zero frequency and thus `num_bits != 0`, so this loop
                // always reaches a leaf within at most `NUM_NODES` steps regardless of what
                // `bit_count` underflows to — it never spins on the `bit_count == 0` guard alone.
                loop {
                    node_idx = self.nodes[node_idx as usize].leaves[(bits & 1) as usize];
                    bit_count = bit_count.wrapping_sub(1);
                    bits >>= 1;
                    if self.nodes[node_idx as usize].num_bits != 0 {
                        break;
                    }
                    if bit_count == 0 {
                        return None;
                    }
                }
            }

            if node_idx == eof_idx {
                break;
            }

            if out_pos == output.len() {
                return None;
            }
            output[out_pos] = self.nodes[node_idx as usize].symbol as u8;
            out_pos += 1;
        }

        Some(out_pos)
    }

    /// Decompresses into a freshly allocated `Vec<u8>`, bounded by `max_output_size` (see the
    /// module-level docs: never allocate unboundedly for hostile/malformed input).
    pub fn decompress_vec(&self, input: &[u8], max_output_size: usize) -> Option<Vec<u8>> {
        let mut out = vec![0u8; max_output_size];
        let n = self.decompress(input, &mut out)?;
        out.truncate(n);
        Some(out)
    }
}

/// `u32 >> n` where `n` may be exactly 32 (which is UB/panics for the built-in `>>` in debug
/// builds); only ever called with `n <= 32` here.
fn shr_u32(value: u32, n: u32) -> u32 {
    if n >= 32 { 0 } else { value >> n }
}

fn set_bits_recursive(nodes: &mut [Node; NUM_NODES], node_idx: u16, bits: u32, depth: u32) {
    let (leaf1, leaf0, has_bits) = {
        let node = &nodes[node_idx as usize];
        (node.leaves[1], node.leaves[0], node.num_bits != 0)
    };
    if leaf1 != NO_CHILD {
        set_bits_recursive(nodes, leaf1, bits | (1 << depth), depth + 1);
    }
    if leaf0 != NO_CHILD {
        set_bits_recursive(nodes, leaf0, bits, depth + 1);
    }
    if has_bits {
        let node = &mut nodes[node_idx as usize];
        node.bits = bits;
        node.num_bits = depth;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_eq(a: &[u8], b: &[u8]) -> bool {
        a == b
    }

    // Ported from `src/test/huffman_test.cpp` (20.1), cited per-test below.

    #[test]
    fn compression_input_size_zero() {
        // huffman_test.cpp: Huffman.CompressionInputSizeZero
        let huffman = Huffman::new();
        let mut compressed = [0u8; 2048];
        let n = huffman.compress(&[], &mut compressed).unwrap();
        let expected = [0x8A, 0x1B];
        assert_eq!(n, expected.len());
        assert!(mem_eq(&compressed[..n], &expected));

        let mut decompressed = [0u8; 2048];
        let n2 = huffman.decompress(&compressed[..n], &mut decompressed).unwrap();
        assert_eq!(n2, 0);
    }

    #[test]
    fn compression_should_not_change_data() {
        // huffman_test.cpp: Huffman.CompressionShouldNotChangeData
        let huffman = Huffman::new();
        for input_mod in 0u32..=0xFFFF {
            let mut input = [0u8; 64];
            input[0] = (input_mod & 0xFF) as u8;
            input[1] = ((input_mod >> 8) & 0xFF) as u8;

            let mut compressed = [0u8; 2048];
            let n = huffman.compress(&input, &mut compressed).unwrap();
            let max_size = if input_mod <= 0xFF { 12 } else { 14 };
            assert!(n >= 10);
            assert!(n <= max_size);

            let mut decompressed = [0u8; 2048];
            let n2 = huffman.decompress(&compressed[..n], &mut decompressed).unwrap();
            assert_eq!(n2, input.len());
            assert!(mem_eq(&input, &decompressed[..n2]));
        }
    }

    #[test]
    fn compression_compatible() {
        // huffman_test.cpp: Huffman.CompressionCompatible
        let huffman = Huffman::new();
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate().take(8) {
            *b = i as u8;
        }
        let mut compressed = [0u8; 2048];
        let n = huffman.compress(&input, &mut compressed).unwrap();
        let expected = [
            0x51, 0x58, 0x78, 0x76, 0x1B, 0xB7, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 0xc5, 0x0D,
        ];
        assert_eq!(n, expected.len(), "compression is not bit-compatible with DDNet 20.1");
        assert!(mem_eq(&compressed[..n], &expected));
    }

    #[test]
    fn compression_no_trailing_null() {
        // huffman_test.cpp: Huffman.CompressionNoTrailingNull
        let huffman = Huffman::new();
        let mut input = [0u8; 64];
        input[0] = 0x15;
        let mut compressed = [0u8; 2048];
        let n = huffman.compress(&input, &mut compressed).unwrap();
        let expected = [0xBE, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x15, 0x37];
        assert_eq!(n, expected.len());
        assert!(mem_eq(&compressed[..n], &expected));

        let mut decompressed = [0u8; 2048];
        let n2 = huffman.decompress(&compressed[..n], &mut decompressed).unwrap();
        assert_eq!(n2, input.len());
        assert!(mem_eq(&input, &decompressed[..n2]));
    }

    #[test]
    fn compression_truncated() {
        // huffman_test.cpp: Huffman.CompressionTruncated
        let huffman = Huffman::new();
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate().take(8) {
            *b = i as u8;
        }
        let mut compressed = [0u8; 2048];
        for compressed_size in 1..=14usize {
            assert_eq!(
                huffman.compress(&input, &mut compressed[..compressed_size]),
                None,
                "compression expected to fail with size {compressed_size}"
            );
        }
        for compressed_size in 15..=20usize {
            assert_eq!(
                huffman.compress(&input, &mut compressed[..compressed_size]),
                Some(15),
                "compression expected to succeed with size {compressed_size}"
            );
        }
    }

    #[test]
    fn decompression_table_lookup_integer_overflow() {
        // huffman_test.cpp: Huffman.DecompressionTableLookupIntegerOverflow
        // ("Test data found by fuzzing" in the original.)
        let huffman = Huffman::new();
        let mut out = [0u8; 2048];
        assert_eq!(huffman.decompress(&[0x1A], &mut out), None);
        assert_eq!(huffman.decompress(&[0x62, 0x91, 0x62, 0xA9], &mut out), None);
        assert_eq!(huffman.decompress(&[0x4C, 0x04, 0xFE, 0x00, 0x68], &mut out), None);
    }

    #[test]
    fn decompress_never_writes_past_output_on_malformed_input() {
        // Regression guard for the "old TS bot hung on this" note in the task spec: feed a
        // buffer of all-0xFF (a common way to get stuck walking the tree) with a deliberately
        // small output buffer and make sure we bail out rather than loop or overflow.
        let huffman = Huffman::new();
        let input = [0xFFu8; 4096];
        let mut out = [0u8; 4];
        // Whatever the result, it must return promptly (this test itself is the timeout guard
        // via the test harness) and never claim to have written more than fits.
        if let Some(n) = huffman.decompress(&input, &mut out) {
            assert!(n <= out.len());
        }
    }

    #[test]
    fn roundtrip_all_single_bytes() {
        let huffman = Huffman::new();
        for b in 0u16..256 {
            let input = [b as u8; 3];
            let compressed = huffman.compress_vec(&input);
            let decompressed = huffman.decompress_vec(&compressed, 4096).unwrap();
            assert_eq!(decompressed, input);
        }
    }

    #[test]
    fn empty_output_buffer_zero_length_input_still_needs_eof_byte() {
        let huffman = Huffman::new();
        // Even compressing zero bytes needs room for the EOF symbol.
        let mut out = [0u8; 0];
        assert_eq!(huffman.compress(&[], &mut out), None);
    }
}
