//! Differential tests against `libtw2` (dev-dependency only, decision D-029 — never linked into
//! non-test builds; see `Cargo.toml`): our Huffman compressor/decompressor and varint packer must
//! agree with `libtw2-huffman`/`libtw2-packer` on every input, since both implement the exact
//! same DDNet wire format. Task acceptance criterion 2: "Huffman compress/decompress and varint
//! packing agree with libtw2 on >= 100k random inputs (proptest), incl. edge cases (empty, max
//! size, all bytes)."

use ddai_net::huffman::Huffman;
use ddai_net::packer::{self, MAX_BYTES_PACKED};
use proptest::prelude::*;

struct NoWarn;
impl<T> libtw2_warn::Warn<T> for NoWarn {
    fn warn(&mut self, _warning: T) {}
}

fn libtw2_pack(value: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(MAX_BYTES_PACKED);
    libtw2_packer::with_packer(&mut out, |mut p| {
        p.write_int(value).expect("MAX_BYTES_PACKED is always enough room");
    });
    out
}

fn libtw2_unpack(bytes: &[u8]) -> i32 {
    let mut unpacker = libtw2_packer::Unpacker::new(bytes);
    unpacker
        .read_int(&mut NoWarn)
        .expect("caller guarantees a complete varint")
}

/// A generous scratch capacity: worst case for Huffman is a handful of bits per byte, well under
/// 4x the input length plus a few bytes for the EOF symbol.
fn scratch_capacity(input_len: usize) -> usize {
    input_len * 4 + 16
}

fn huffman_compress_ours(huffman: &Huffman, input: &[u8]) -> Vec<u8> {
    huffman.compress_vec(input)
}

fn huffman_compress_libtw2(input: &[u8]) -> Vec<u8> {
    libtw2_huffman::compress(input)
}

// --- Randomised differential tests (proptest) ------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// Our `pack_int` produces byte-identical output to `libtw2-packer`'s `write_int` for every
    /// possible `i32`.
    #[test]
    fn varint_pack_matches_libtw2(value in any::<i32>()) {
        let mut ours = [0u8; MAX_BYTES_PACKED];
        let n = packer::pack_int(&mut ours, value).expect("MAX_BYTES_PACKED is always enough room");
        let theirs = libtw2_pack(value);
        prop_assert_eq!(&ours[..n], theirs.as_slice());
    }

    /// Our `unpack_int` agrees with `libtw2-packer`'s `read_int` when fed the *same* (our own)
    /// encoding, and round-trips back to the original value.
    #[test]
    fn varint_unpack_matches_libtw2_on_our_encoding(value in any::<i32>()) {
        let mut buf = [0u8; MAX_BYTES_PACKED];
        let n = packer::pack_int(&mut buf, value).unwrap();
        let (ours, consumed) = packer::unpack_int(&buf[..n]).unwrap();
        prop_assert_eq!(consumed, n);
        prop_assert_eq!(ours, value);
        prop_assert_eq!(libtw2_unpack(&buf[..n]), value);
    }

    /// And the other way around: decoding *libtw2's* encoding with our `unpack_int` also
    /// round-trips.
    #[test]
    fn varint_unpack_matches_libtw2_on_their_encoding(value in any::<i32>()) {
        let theirs = libtw2_pack(value);
        let (ours, consumed) = packer::unpack_int(&theirs).unwrap();
        prop_assert_eq!(consumed, theirs.len());
        prop_assert_eq!(ours, value);
    }
}

proptest! {
    // Huffman inputs are bytes, not one i32 each, so this loop is inherently more expensive per
    // case than the varint ones above; 100k cases of up to ~1400 bytes each is still the
    // acceptance criterion's floor and comfortably fast (see the build report for the measured
    // wall time).
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// Our Huffman compressor produces byte-identical output to `libtw2-huffman` for random
    /// inputs up to a full DDNet packet's worth of chunk data.
    #[test]
    fn huffman_compress_matches_libtw2(input in prop::collection::vec(any::<u8>(), 0..1400)) {
        let huffman = Huffman::new();
        let ours = huffman_compress_ours(&huffman, &input);
        let theirs = huffman_compress_libtw2(&input);
        prop_assert_eq!(ours, theirs);
    }

    /// Decompressing our own compressed output with `libtw2-huffman` recovers the original
    /// bytes, and vice versa — proves wire compatibility, not just "both compress the same".
    #[test]
    fn huffman_roundtrip_cross_implementation(input in prop::collection::vec(any::<u8>(), 0..1400)) {
        let huffman = Huffman::new();
        let ours_compressed = huffman_compress_ours(&huffman, &input);
        let theirs_decompressed = libtw2_huffman::decompress(&ours_compressed).expect("valid compression");
        prop_assert_eq!(&theirs_decompressed, &input);

        let theirs_compressed = huffman_compress_libtw2(&input);
        let mut ours_decompressed = vec![0u8; scratch_capacity(input.len())];
        let n = huffman
            .decompress(&theirs_compressed, &mut ours_decompressed)
            .expect("valid compression");
        prop_assert_eq!(&ours_decompressed[..n], input.as_slice());
    }
}

// --- Edge cases (explicit, not randomised) ---------------------------------------------------

#[test]
fn huffman_edge_case_empty_input() {
    let huffman = Huffman::new();
    let input: &[u8] = &[];
    assert_eq!(huffman_compress_ours(&huffman, input), huffman_compress_libtw2(input));
}

#[test]
fn huffman_edge_case_max_packet_sized_input() {
    // `NET_MAX_PACKETSIZE - NET_PACKETHEADERSIZE`: the largest chunk-data payload a single DDNet
    // packet can carry.
    let huffman = Huffman::new();
    let input = vec![0xAAu8; 1397];
    assert_eq!(huffman_compress_ours(&huffman, &input), huffman_compress_libtw2(&input));
}

#[test]
fn huffman_edge_case_all_byte_values_present() {
    let huffman = Huffman::new();
    let input: Vec<u8> = (0u16..256).map(|b| b as u8).collect();
    assert_eq!(huffman_compress_ours(&huffman, &input), huffman_compress_libtw2(&input));
    // Repeated several times, in different relative frequencies, to also exercise every symbol's
    // code path through the tree more than once.
    let input2: Vec<u8> = (0u16..256)
        .flat_map(|b| std::iter::repeat_n(b as u8, ((b % 7) + 1) as usize))
        .collect();
    assert_eq!(
        huffman_compress_ours(&huffman, &input2),
        huffman_compress_libtw2(&input2)
    );
}

#[test]
fn huffman_edge_case_all_bytes_individually() {
    // Every single-byte input, matching `CompressionShouldNotChangeData` in spirit but checked
    // against libtw2 instead of a fixed size bound.
    let huffman = Huffman::new();
    for b in 0u16..256 {
        let input = [b as u8];
        assert_eq!(
            huffman_compress_ours(&huffman, &input),
            huffman_compress_libtw2(&input),
            "byte {b}"
        );
    }
}

#[test]
fn varint_edge_cases_match_libtw2() {
    let cases = [
        0,
        1,
        -1,
        63,
        64,
        -64,
        -65,
        i32::MAX,
        i32::MIN,
        i32::MAX - 1,
        i32::MIN + 1,
        1 << 6,
        1 << 13,
        1 << 20,
        1 << 27,
        -(1 << 6),
        -(1 << 13),
        -(1 << 20),
        -(1 << 27),
    ];
    for &value in &cases {
        let mut ours = [0u8; MAX_BYTES_PACKED];
        let n = packer::pack_int(&mut ours, value).unwrap();
        let theirs = libtw2_pack(value);
        assert_eq!(&ours[..n], theirs.as_slice(), "for value {value}");
    }
}
