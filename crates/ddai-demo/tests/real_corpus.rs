//! `#[ignore]` tests over the real demo corpus — task 8.4b acceptance criterion 5: real demo
//! files never touch this repository (`.gitignore`'s `*.demo` rule), so these tests read from
//! `~/aiddnet/data/demos/` on disk and are skipped by default (`cargo test --workspace`, which
//! CI/the pre-commit checklist run, never needs this directory to exist).
//!
//! Run explicitly with a corpus present:
//!
//! ```bash
//! cargo test -p ddai-demo --release --test real_corpus -- --ignored --nocapture
//! ```
//!
//! Byte-exactness against DDNet 20.1's own reader (task acceptance criterion 2) is checked
//! separately, over the same corpus, by `tools/ddnet-oracle/parity_check_demo.sh` (a real C++
//! process, not something a `cargo test` can drive) — this file instead checks the same
//! "never panic on a real file" and "produces sane, non-empty output" properties `dataset`
//! consumers (task 8.4a) will actually rely on.

use ddai_demo::Demo;
use std::path::{Path, PathBuf};

fn chillerdragon_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/data/demos/chillerdragon/block-06")
}

fn public_samples_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/data/demos/public-samples")
}

fn collect_demo_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("demo")) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Parses every demo in `dir`, fully drains its tick stream, and returns
/// `(files_ok, files_failed, total_ticks, total_snapshots, total_messages)`. Panics (a test
/// failure) only on an actual Rust panic inside `ddai-demo` itself — a per-file parse/decode
/// `Err` is counted, not a hard failure, since this corpus includes files spanning demo versions
/// 4/5/6 and this function's job is "never panics", not "every byte is a valid demo".
fn drain_all(files: &[PathBuf]) -> (usize, usize, u64, u64, u64) {
    let mut ok = 0;
    let mut failed = 0;
    let mut total_ticks = 0u64;
    let mut total_snapshots = 0u64;
    let mut total_messages = 0u64;

    for path in files {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let demo = match Demo::parse(&bytes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}: header parse failed: {e}", path.display());
                failed += 1;
                continue;
            }
        };
        let mut file_ticks = 0u64;
        let mut decode_error = None;
        for tick in demo.ticks() {
            match tick {
                Ok(t) => {
                    file_ticks += 1;
                    if t.snapshot.is_some() {
                        total_snapshots += 1;
                    }
                    total_messages += t.messages.len() as u64;
                }
                Err(e) => {
                    decode_error = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = decode_error {
            eprintln!("{}: stopped after {file_ticks} ticks: {e}", path.display());
        }
        total_ticks += file_ticks;
        if file_ticks > 0 {
            ok += 1;
        } else {
            failed += 1;
        }
    }
    (ok, failed, total_ticks, total_snapshots, total_messages)
}

#[test]
#[ignore = "needs ~/aiddnet/data/demos/chillerdragon/block-06 on disk (D-040, local only)"]
fn chillerdragon_archive_never_panics_and_decodes() {
    let files = collect_demo_files(&chillerdragon_dir());
    assert!(
        !files.is_empty(),
        "expected the 228-file ChillerDragon corpus to be present"
    );
    println!("chillerdragon_archive: {} files", files.len());

    let (ok, failed, ticks, snapshots, messages) = drain_all(&files);
    println!(
        "chillerdragon_archive: {ok} ok, {failed} failed to decode any tick, {ticks} ticks total, \
         {snapshots} snapshots, {messages} messages"
    );

    // Every file in this 228-file archive decodes fully except two that are corrupt in the
    // archive itself (confirmed independently: DDNet 20.1's own reader,
    // `tools/ddnet-oracle/demo2json`, rejects both the exact same way — see docs/formats.md
    // §18.7) — `BlmapChill sick solo at end.demo` (a malformed tick-marker stream right after a
    // 0-byte declared map — a 0-byte map is not itself unusual, 30 other files in this archive
    // have one and decode fine) and `ChillBlock5 hdoffhoklnoff 3edges slowmo.demo` (its header's
    // `map_size` field is simply wrong — bigger than the embedded datafile actually is; not a
    // physically truncated file, see docs/formats.md §18.7's review round 1 finding F6 fix — real
    // DDNet trusts that same wrong field the same way `ddai-demo` does, so both reject it
    // identically). A THIRD failure here would be a real regression, so this asserts the count
    // exactly rather than just `<= 2`.
    assert_eq!(
        failed, 2,
        "expected exactly the 2 known-corrupt files to fail — see stderr above"
    );
    assert!(
        ticks > 300_000,
        "expected well over 300k ticks across the whole archive, got {ticks}"
    );
    assert!(snapshots > 0);
    assert!(messages > 0);
}

#[test]
#[ignore = "needs ~/aiddnet/data/demos/public-samples on disk"]
fn public_samples_never_panic_and_decode() {
    let files = collect_demo_files(&public_samples_dir());
    assert!(!files.is_empty(), "expected public-samples demos to be present");
    println!("public_samples: {} files", files.len());

    let (ok, failed, ticks, snapshots, _messages) = drain_all(&files);
    println!("public_samples: {ok} ok, {failed} failed, {ticks} ticks total, {snapshots} snapshots");
    assert!(ok > 0);
}
