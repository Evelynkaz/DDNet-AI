//! Robustness / fuzz-style tests: task acceptance criterion 5 — feed >= 10^6 random/mutated
//! datagrams into packet decode, Huffman decompress, the `Unpacker`, and `Connection::feed`,
//! and confirm none of them ever panics. "Bounded memory/time (each call O(input))" is a
//! structural property of the implementation (every loop is bounded by the size of its input or
//! output buffer — see the doc comments in `src/huffman.rs`, `src/packet.rs`, `src/packer.rs`,
//! `src/conn.rs`), which this test demonstrates empirically by simply completing in bounded time
//! for a million-plus varied/adversarial inputs rather than hanging.
//!
//! `#[test]` failing here means a panic (proptest reports the panicking input directly), so
//! "the run finished" already proves the "no panics" half of the acceptance criterion for every
//! case it generated; the case counts below sum to over 1,000,000.

use ddai_net::conn::{Config, Connection};
use ddai_net::huffman::Huffman;
use ddai_net::packer::{SanitizeMode, Unpacker};
use ddai_net::{control, packet};
use proptest::prelude::*;
use std::time::Duration;

/// Mutates `base` at up to `num_mutations` random byte positions (flips to an arbitrary byte),
/// simulating bit-rot / an adversarial peer tampering with an otherwise well-formed datagram —
/// the "mutated" half of "random/mutated datagrams".
fn mutate(mut base: Vec<u8>, positions: &[(usize, u8)]) -> Vec<u8> {
    for &(pos, byte) in positions {
        if !base.is_empty() {
            let idx = pos % base.len();
            base[idx] = byte;
        }
    }
    base
}

fn mutation_strategy(max_len: usize) -> impl Strategy<Value = Vec<(usize, u8)>> {
    prop::collection::vec((0..max_len.max(1), any::<u8>()), 0..8)
}

/// A real, well-formed `CONNECT` datagram, `TKEN`-magic and all — used as a mutation seed.
fn sample_connect_datagram(huffman: &Huffman) -> Vec<u8> {
    let payload = control::encode(&control::connect_payload());
    packet::build_packet(
        packet::packet_flags::CONTROL,
        0,
        0,
        &payload,
        Some(control::TOKEN_UNKNOWN),
        huffman,
    )
    .unwrap()
}

/// A real, well-formed data-carrying datagram (one vital chunk) — another mutation seed.
fn sample_data_datagram(huffman: &Huffman) -> Vec<u8> {
    let mut chunk_data = Vec::new();
    let mut pos = 0usize;
    let mut buf = vec![0u8; 64];
    packet::pack_chunk(
        &mut buf,
        &mut pos,
        packet::chunk_flags::VITAL,
        1,
        b"CLIENTVER-ish payload here",
    );
    chunk_data.extend_from_slice(&buf[..pos]);
    packet::build_packet(0, 3, 1, &chunk_data, Some(0x1234_5678), huffman).unwrap()
}

/// Like [`sample_data_datagram`], but with a caller-chosen token — used to build a datagram that
/// actually matches a real [`Connection`]'s negotiated token (see [`establish_online_client`]),
/// so mutating it has a real chance of still passing the trailing-token check and reaching the
/// chunk-iteration/ack-bookkeeping code inside `Connection::feed`'s `State::Online` path — F5: an
/// earlier version of this test used an arbitrary, non-matching token, so the client here was
/// always still `Connecting` (never online) and that whole code path went untested.
fn sample_data_datagram_with_token(huffman: &Huffman, ack: u16, sequence: u16, token: u32) -> Vec<u8> {
    let mut chunk_data = Vec::new();
    let mut pos = 0usize;
    let mut buf = vec![0u8; 64];
    packet::pack_chunk(
        &mut buf,
        &mut pos,
        packet::chunk_flags::VITAL,
        sequence,
        b"CLIENTVER-ish payload here",
    );
    chunk_data.extend_from_slice(&buf[..pos]);
    packet::build_packet(0, ack, 1, &chunk_data, Some(token), huffman).unwrap()
}

/// Drives a client-role [`Connection`] all the way to [`ddai_net::conn::State::Online`] against a
/// self-synthesized (not a second `Connection`) `CONNECTACCEPT`, and returns it along with the
/// token it negotiated — so fuzz cases can build datagrams a real online connection would
/// plausibly receive, instead of only ever exercising the pre-handshake code paths.
fn establish_online_client(huffman: &Huffman, now: Duration) -> (Connection, u32) {
    const TOKEN: u32 = 0xC0FF_EE42;
    let mut client = Connection::new(Config::default());
    client.connect(now, huffman);
    let _ = client.flush(huffman, now); // drain the CONNECT, irrelevant here

    let connect_accept_payload = control::encode(&control::connect_accept_payload());
    let mut with_token = connect_accept_payload;
    with_token.extend_from_slice(&TOKEN.to_be_bytes());
    let connect_accept = packet::build_packet(packet::packet_flags::CONTROL, 0, 0, &with_token, None, huffman).unwrap();

    let events = client.feed(&connect_accept, huffman, now);
    assert_eq!(events, vec![ddai_net::conn::Event::Connected]);
    assert!(client.is_online());
    let _ = client.flush(huffman, now); // drain the queued ACCEPT, irrelevant here
    (client, TOKEN)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300_000))]

    /// `packet::unpack_packet` never panics on arbitrary bytes.
    #[test]
    fn packet_decode_never_panics_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..1500)) {
        let huffman = Huffman::new();
        let _ = packet::unpack_packet(&bytes, &huffman, true);
        let _ = packet::unpack_connless_packet(&bytes);
    }

    /// Same, but mutating a real `CONNECT` or data datagram instead of pure noise — more likely
    /// to pass the early structural checks and actually exercise the compressed-payload/chunk
    /// paths deeper in.
    #[test]
    fn packet_decode_never_panics_on_mutated_real_datagrams(
        use_connect in any::<bool>(),
        mutations in mutation_strategy(1400),
    ) {
        let huffman = Huffman::new();
        let base = if use_connect { sample_connect_datagram(&huffman) } else { sample_data_datagram(&huffman) };
        let mutated = mutate(base, &mutations);
        let _ = packet::unpack_packet(&mutated, &huffman, true);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300_000))]

    /// `Huffman::decompress` never panics on arbitrary bytes, regardless of output buffer size —
    /// this is the specific function the task spec calls out ("the old TS bot hung on this").
    #[test]
    fn huffman_decompress_never_panics_on_random_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..1500),
        out_size in 0usize..2048,
    ) {
        let huffman = Huffman::new();
        let mut out = vec![0u8; out_size];
        if out_size == 0 {
            // `decompress` documents `OutputSize > 0` is not required on our side (unlike the
            // C++ `dbg_assert`) — a zero-size buffer must just fail cleanly, never panic.
            let _ = huffman.decompress(&bytes, &mut out);
        } else {
            let _ = huffman.decompress(&bytes, &mut out);
        }
    }

    /// All-`0xFF` and other adversarial-looking fixed patterns, at every length up to a full
    /// packet — the exact shape of input that found real bugs in the C++ reference's fuzz corpus
    /// (`DecompressionTableLookupIntegerOverflow`, ported in `src/huffman.rs`).
    #[test]
    fn huffman_decompress_never_panics_on_repeating_byte_patterns(byte in any::<u8>(), len in 0usize..1500) {
        let huffman = Huffman::new();
        let input = vec![byte; len];
        let mut out = vec![0u8; 4096];
        let _ = huffman.decompress(&input, &mut out);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200_000))]

    /// `Unpacker` never panics for any sequence of reads over arbitrary bytes — exercises every
    /// combination of int/string/raw reads, including after the unpacker has already poisoned
    /// itself (which must keep returning defaults, never panic or read out of bounds).
    #[test]
    fn unpacker_never_panics_on_arbitrary_reads(
        bytes in prop::collection::vec(any::<u8>(), 0..300),
        ops in prop::collection::vec(0u8..5, 0..40),
        raw_sizes in prop::collection::vec(0usize..40, 0..10),
    ) {
        let mut unpacker = Unpacker::new(&bytes);
        let mut raw_iter = raw_sizes.into_iter().cycle();
        for op in ops {
            match op {
                0 => { let _ = unpacker.get_int(); }
                1 => { let _ = unpacker.get_string(SanitizeMode::SANITIZE); }
                2 => { let _ = unpacker.get_string(SanitizeMode::SANITIZE_CC); }
                3 => { let _ = unpacker.get_raw(raw_iter.next().unwrap_or(0)); }
                _ => { let _ = unpacker.get_int_or_default(-1); }
            }
        }
        let _ = unpacker.get_rest();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200_000))]

    /// `Connection::feed` never panics on arbitrary bytes, in any of the four states it can be
    /// fed in (offline connections short-circuit inside `feed` itself, but we drive the state
    /// machine there too, for completeness) — including genuinely `Online`, with a *matching*
    /// negotiated token, so the random bytes have a real chance of passing the trailing-token
    /// check and reaching the chunk-iteration/ack-bookkeeping code (F5: an earlier version of
    /// this test only ever reached `Offline`/`Connecting`/`Pending`, since role `2`'s `accept()`
    /// puts the connection in `Pending` — never `Online` — without a matching `ACCEPT` fed back).
    #[test]
    fn connection_feed_never_panics_on_random_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..1500),
        role in 0u8..4,
    ) {
        let huffman = Huffman::new();
        let now = Duration::from_secs(1);
        let mut conn = match role {
            0 => Connection::new(Config::default()), // stays Offline
            1 => {
                let mut c = Connection::new(Config::default());
                c.connect(now, &huffman);
                c
            }
            2 => {
                let mut c = Connection::new(Config::default());
                c.accept(0xdead_beef, now, &huffman);
                c
            }
            _ => establish_online_client(&huffman, now).0,
        };
        let _ = conn.feed(&bytes, &huffman, now);
        // Also drive one more feed/flush cycle so a state transition triggered by the first
        // `feed` (e.g. Connecting -> Online) gets exercised by the *next* payload too.
        let _ = conn.flush(&huffman, now);
        let _ = conn.feed(&bytes, &huffman, now + Duration::from_millis(1));
    }

    /// Mutated real handshake/data datagrams fed into a genuinely `Online` connection (matching
    /// negotiated token) — more likely to reach the chunk-iteration and ack-bookkeeping code
    /// paths than pure noise, and (F5) actually exercises them now: a prior version built the
    /// seed datagram with an arbitrary token that never matched the client's real (still
    /// `Unknown`, since the client was never actually taken online) token, so every mutated
    /// datagram was rejected before the interesting code ever ran.
    #[test]
    fn connection_feed_never_panics_on_mutated_real_datagrams(mutations in mutation_strategy(1400)) {
        let huffman = Huffman::new();
        let now = Duration::from_secs(1);
        let (mut client, token) = establish_online_client(&huffman, now);

        let base = sample_data_datagram_with_token(&huffman, 0, 1, token);
        let mutated = mutate(base, &mutations);
        let _ = client.feed(&mutated, &huffman, now);
    }
}
