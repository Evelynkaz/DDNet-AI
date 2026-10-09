//! Probe harness: every function of this crate against the C library's, reached through `std`, which on
//! `x86_64-unknown-linux-gnu` is glibc (the reference, D-004/D-127). Linux only; the platform-independent
//! counterpart (golden hashes recorded from glibc, checked everywhere) is `tests/golden.rs`.
//!
//! * CI subset (plain `cargo test`): the special-value lists plus `LIBM_PROBES` (default 100 000)
//!   generated probes per function.
//! * Full run: `cargo test --release -p ddai-libm --test glibc_probe -- --ignored` runs 10^7 generated
//!   probes per function (`LIBM_PROBES` overrides) and, with `-- --ignored exhaustive`, every one of the
//!   2^32 `f32` inputs of the one-argument functions (`LIBM_STRIDE=n` samples every n-th).
//!
//! A probe passes when the results are bit-identical (NaN payload and sign included).
#![cfg(all(target_os = "linux", target_env = "gnu"))]

mod common;

use common::{for_each_input, ours};
use std::hint::black_box;

fn probes(default: u64) -> u64 {
    std::env::var("LIBM_PROBES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The C library's result for `name` (through `std`), operands as raw bits.
fn glibc(name: &str, args: &[u64]) -> u64 {
    let f = |i: usize| black_box(f32::from_bits(args[i] as u32));
    let d = |i: usize| black_box(f64::from_bits(args[i]));
    match name {
        "sinf" => u64::from(f(0).sin().to_bits()),
        "cosf" => u64::from(f(0).cos().to_bits()),
        "atanf" => u64::from(f(0).atan().to_bits()),
        "atan2f" => u64::from(f(0).atan2(f(1)).to_bits()),
        "hypotf" => u64::from(f(0).hypot(f(1)).to_bits()),
        "powf" => u64::from(f(0).powf(f(1)).to_bits()),
        "log" => d(0).ln().to_bits(),
        "atan2" => d(0).atan2(d(1)).to_bits(),
        "pow" => d(0).powf(d(1)).to_bits(),
        "hypot" => d(0).hypot(d(1)).to_bits(),
        other => panic!("unknown function {other}"),
    }
}

/// Runs `name`'s probe sequence with `n` generated probes and fails on any mismatch.
fn probe(name: &'static str, n: u64) {
    let (mut probes, mut bad) = (0u64, 0u64);
    let mut first = Vec::new();
    for_each_input(name, n, &mut |args| {
        probes += 1;
        let (a, b) = (ours(name, args), glibc(name, args));
        if a != b {
            bad += 1;
            if first.len() < 8 {
                first.push(format!("{name}({args:#x?}): ours {a:#x} glibc {b:#x}"));
            }
        }
    });
    eprintln!("{name}: {probes} probes, {bad} mismatches");
    for f in &first {
        eprintln!("  {f}");
    }
    assert_eq!(bad, 0, "{name}: {bad} mismatches of {probes} probes");
}

macro_rules! probe_tests {
    ($($ci:ident, $full:ident, $name:literal;)*) => {$(
        #[test]
        fn $ci() {
            probe($name, probes(100_000));
        }

        #[test]
        #[ignore = "10^7 probes per function (release build recommended)"]
        fn $full() {
            probe($name, probes(10_000_000));
        }
    )*};
}

probe_tests! {
    sinf_matches_glibc_ci_subset, sinf_matches_glibc_full, "sinf";
    cosf_matches_glibc_ci_subset, cosf_matches_glibc_full, "cosf";
    atanf_matches_glibc_ci_subset, atanf_matches_glibc_full, "atanf";
    atan2f_matches_glibc_ci_subset, atan2f_matches_glibc_full, "atan2f";
    powf_matches_glibc_ci_subset, powf_matches_glibc_full, "powf";
    log_matches_glibc_ci_subset, log_matches_glibc_full, "log";
    atan2_matches_glibc_ci_subset, atan2_matches_glibc_full, "atan2";
    pow_matches_glibc_ci_subset, pow_matches_glibc_full, "pow";
    hypotf_matches_glibc_ci_subset, hypotf_matches_glibc_full, "hypotf";
    hypot_matches_glibc_ci_subset, hypot_matches_glibc_full, "hypot";
}

/// Every one of the 2^32 `f32` inputs (3 threads); `LIBM_STRIDE=n` samples every n-th input.
fn exhaustive(name: &'static str) {
    let threads = 3u64;
    let stride: u64 = std::env::var("LIBM_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            std::thread::spawn(move || {
                let (mut probes, mut bad) = (0u64, 0u64);
                let mut first = Vec::new();
                let mut i = t;
                while i < (1u64 << 32) {
                    let args = [i];
                    probes += 1;
                    let (a, b) = (ours(name, &args), glibc(name, &args));
                    if a != b {
                        bad += 1;
                        if first.len() < 8 {
                            first.push(format!("{name}({i:#010x}): ours {a:#x} glibc {b:#x}"));
                        }
                    }
                    i += threads * stride;
                }
                (probes, bad, first)
            })
        })
        .collect();
    let (mut probes, mut bad) = (0, 0);
    for h in handles {
        let (p, b, first) = h.join().unwrap();
        probes += p;
        bad += b;
        for f in first {
            eprintln!("  {f}");
        }
    }
    eprintln!("{name}: {probes} probes (exhaustive over the f32 inputs), {bad} mismatches");
    assert_eq!(bad, 0, "{name}: {bad} mismatches");
}

#[test]
#[ignore = "exhaustive: all 2^32 f32 inputs"]
fn exhaustive_sinf() {
    exhaustive("sinf");
}

#[test]
#[ignore = "exhaustive: all 2^32 f32 inputs"]
fn exhaustive_cosf() {
    exhaustive("cosf");
}

#[test]
#[ignore = "exhaustive: all 2^32 f32 inputs"]
fn exhaustive_atanf() {
    exhaustive("atanf");
}

/// `std::atan2(int, int)` in DDNet's `CCharacterCore::Tick`: every aim vector in a window of +-`LIBM_AIM`
/// (default 4096; the full request of 8192 takes a few minutes) is checked against glibc.
#[test]
#[ignore = "every integer aim vector in a window"]
fn atan2_every_integer_aim_vector() {
    let w: i32 = std::env::var("LIBM_AIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let mut bad = 0u64;
    let mut first = Vec::new();
    for ty in -w..=w {
        for tx in -w..=w {
            let (y, x) = (f64::from(ty), f64::from(tx));
            let (a, b) = (ddai_libm::atan2(y, x), black_box(y).atan2(black_box(x)));
            if a.to_bits() != b.to_bits() {
                bad += 1;
                if first.len() < 8 {
                    first.push(format!(
                        "atan2({ty}, {tx}): ours {:#x} glibc {:#x}",
                        a.to_bits(),
                        b.to_bits()
                    ));
                }
            }
        }
    }
    eprintln!(
        "atan2: every integer aim vector in +-{w}: {bad} mismatches of {}",
        (2 * u64::from(w.unsigned_abs()) + 1).pow(2)
    );
    for f in &first {
        eprintln!("  {f}");
    }
    assert_eq!(bad, 0);
}
