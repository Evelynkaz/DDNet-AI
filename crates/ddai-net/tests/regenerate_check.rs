//! Task 2.2b acceptance criterion 1 / review round 1 finding F5: "re-running the generator
//! reproduces the committed files byte-for-byte ... a Rust test can't run it in CI since it needs
//! the DDNet tree — instead add a `#[ignore]` test or script that does."
//!
//! `#[ignore]`d by default (needs a real DDNet 20.1 source tree checked out locally, which is
//! never committed to this repository — see `tools/ddnet-protocol-gen/README.md`). Run explicitly
//! with:
//!
//! ```text
//! cargo test -p ddai-net --test regenerate_check -- --ignored
//! ```
//!
//! optionally overriding the DDNet tree path with the `DDNET_SRC` environment variable (defaults
//! to `~/aiddnet/build/ddnet-20.1/src`, the pinned commit's location documented in
//! `tools/ddnet-protocol-gen/README.md` and this crate's task spec).

use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "needs a real (uncommitted) DDNet 20.1 source tree — see the module docs"]
fn regenerating_reproduces_the_committed_generated_files_byte_for_byte() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/ddai-net is always two levels under the repository root");
    let generator = repo_root.join("tools/ddnet-protocol-gen/generate.py");
    let committed_dir = crate_dir.join("src/generated");

    let ddnet_src = std::env::var("DDNET_SRC").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME must be set to find the default DDNet tree path");
        format!("{home}/aiddnet/build/ddnet-20.1/src")
    });
    assert!(
        PathBuf::from(&ddnet_src).join("datasrc/network.py").is_file(),
        "DDNet source tree not found at {ddnet_src} (set DDNET_SRC to override) — this test needs \
         a real, uncommitted DDNet 20.1 checkout, see the module docs"
    );

    let out_dir = std::env::temp_dir().join(format!("ddai-net-regenerate-check-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).unwrap();

    let status = Command::new("python3")
        .arg(&generator)
        .arg(&ddnet_src)
        .arg("--out")
        .arg(&out_dir)
        .status()
        .expect("failed to run tools/ddnet-protocol-gen/generate.py — is python3 on PATH?");
    assert!(status.success(), "generate.py exited with {status}");

    let mut mismatches = Vec::new();
    for name in ["mod.rs", "enums.rs", "objects.rs", "messages.rs"] {
        let generated =
            std::fs::read_to_string(out_dir.join(name)).unwrap_or_else(|e| panic!("reading regenerated {name}: {e}"));
        let committed = std::fs::read_to_string(committed_dir.join(name))
            .unwrap_or_else(|e| panic!("reading committed {name}: {e}"));
        if generated != committed {
            mismatches.push(name);
        }
    }

    let _ = std::fs::remove_dir_all(&out_dir);

    assert!(
        mismatches.is_empty(),
        "regenerating produced different bytes than the committed files for: {mismatches:?} — \
         either the DDNet tree at {ddnet_src} is not pinned to the expected commit, or the \
         generator/committed files have drifted apart"
    );
}
