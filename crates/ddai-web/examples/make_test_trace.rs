//! Test-only helper (task 5.2a): writes a synthetic trace-b v2 fixture against a real `.map`
//! file's actual sha256, so Playwright (`tools/e2e/live-map.spec.ts`) can exercise the live map
//! view — including its phone-performance measurement, which needs 8+ characters — against the
//! real, largest local map (`BlmapChill`, 1244×667 tiles). The real Oracle B corpus never
//! generates more than 4 characters on a single real-map scenario (checked directly against the
//! corpus on disk), so this is how the task's "8+ characters" acceptance criterion is met: a real
//! map's real geometry, with a synthetic character count/tick count chosen for the measurement,
//! built with the exact same trace-b writer (`ddai_web::live::replay::testutil`) this crate's own
//! unit tests use — not a special-cased or hand-faked payload.
//!
//! `cargo run -p ddai-web --features test-util --example make_test_trace -- \
//!     --map <path/to/BlmapChill.map> --out <out.trb> [--characters 8] [--ticks 3000]`

use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let map_path = arg_value(&args, "--map").expect("--map <path> is required");
    let out_path = PathBuf::from(arg_value(&args, "--out").expect("--out <path> is required"));
    let characters: u32 = arg_value(&args, "--characters")
        .map(|s| s.parse().expect("--characters must be a u32"))
        .unwrap_or(8);
    let ticks: u32 = arg_value(&args, "--ticks")
        .map(|s| s.parse().expect("--ticks must be a u32"))
        .unwrap_or(3000);

    let map_bytes = std::fs::read(&map_path).unwrap_or_else(|e| panic!("reading {map_path}: {e}"));
    let mut hasher = Sha256::new();
    hasher.update(&map_bytes);
    let sha256: [u8; 32] = hasher.finalize().into();

    ddai_web::live::replay::testutil::write_trace_b_fixture(
        &out_path,
        characters,
        ticks,
        "real-map",
        sha256,
        Some(&map_path),
    );
    println!(
        "wrote {} ({} characters, {} ticks, map sha256 {})",
        out_path.display(),
        characters,
        ticks,
        sha256.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
}
