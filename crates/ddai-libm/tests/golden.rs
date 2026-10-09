//! Golden hashes: the results glibc 2.39 (x86-64 FMA variant) gives for a fixed, platform-independent
//! sequence of probes, recorded on Linux in `tests/golden/hashes.txt` and checked here on EVERY platform.
//! This is the test that proves the point of the crate: Linux and Windows compute the same bits.
//!
//! `regenerate_golden_hashes` (Linux only, `--ignored`) writes the file from glibc itself.

mod common;

use common::{GOLDEN_BLOCK, GOLDEN_PROBES, NAMES, fnv, for_each_input, ours};

const GOLDEN: &str = include_str!("golden/hashes.txt");

/// FNV-1a hashes of `results`, one per block of `GOLDEN_BLOCK` consecutive results (the last block may
/// be shorter).
fn block_hashes(results: &[u64]) -> Vec<u64> {
    results
        .chunks(GOLDEN_BLOCK)
        .map(|c| c.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| fnv(h, v)))
        .collect()
}

fn parse_golden() -> std::collections::BTreeMap<String, (usize, Vec<u64>)> {
    let mut out = std::collections::BTreeMap::new();
    for line in GOLDEN.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let name = it.next().unwrap().to_string();
        let count: usize = it.next().unwrap().parse().unwrap();
        let hashes = it.map(|h| u64::from_str_radix(h, 16).unwrap()).collect();
        out.insert(name, (count, hashes));
    }
    out
}

fn our_results(name: &str) -> Vec<u64> {
    let mut results = Vec::new();
    for_each_input(name, GOLDEN_PROBES, &mut |args| results.push(ours(name, args)));
    results
}

#[test]
fn results_match_the_hashes_recorded_from_glibc() {
    let golden = parse_golden();
    for &name in NAMES {
        let (count, want) = golden.get(name).unwrap_or_else(|| panic!("{name}: no golden entry"));
        let results = our_results(name);
        assert_eq!(results.len(), *count, "{name}: the probe sequence changed length");
        let got = block_hashes(&results);
        assert_eq!(got.len(), want.len(), "{name}: block count");
        let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        if !bad.is_empty() {
            let first = bad[0];
            let mut shown = String::new();
            let mut idx = 0usize;
            for_each_input(name, GOLDEN_PROBES, &mut |args| {
                if idx / GOLDEN_BLOCK == first && idx % GOLDEN_BLOCK < 3 {
                    shown += &format!("\n  probe {idx}: args {args:#x?} -> {:#x}", results[idx]);
                }
                idx += 1;
            });
            panic!(
                "{name}: {} of {} blocks differ from the glibc golden hashes (first: block {first}, probes {}..){shown}",
                bad.len(),
                got.len(),
                first * GOLDEN_BLOCK
            );
        }
    }
}

/// Writes `tests/golden/hashes.txt` from glibc through `std` (Linux), after checking that our results equal
/// glibc's bit for bit.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
#[ignore = "regenerates tests/golden/hashes.txt from glibc"]
fn regenerate_golden_hashes() {
    use std::fmt::Write;
    use std::hint::black_box;
    let mut text = String::from(
        "# Golden hashes of glibc 2.39 (x86-64 FMA variant) results, see tests/golden.rs.\n\
         # name  results  FNV-1a hash per block of 4096 results\n",
    );
    for &name in NAMES {
        let mut glibc = Vec::new();
        for_each_input(name, GOLDEN_PROBES, &mut |args| {
            let f = |i: usize| black_box(f32::from_bits(args[i] as u32));
            let d = |i: usize| black_box(f64::from_bits(args[i]));
            glibc.push(match name {
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
            });
        });
        assert_eq!(
            glibc,
            our_results(name),
            "{name}: ours differs from glibc, refusing to record"
        );
        write!(text, "{name} {}", glibc.len()).unwrap();
        for h in block_hashes(&glibc) {
            write!(text, " {h:016x}").unwrap();
        }
        text.push('\n');
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/hashes.txt");
    std::fs::write(&path, text).unwrap();
    eprintln!("wrote {}", path.display());
}
