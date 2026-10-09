//! Task 8.8: every fly checkpoint written before the `Gm` neuron model (format v3, v4 and v5, in the local run directories) loads as a
//! `Rate` fly, and saving it again gives the same checkpoint (and, in the file's own layout, the same payload bytes): the layouts are unchanged, so an older
//! binary still reads what this one writes for a rate fly. Skipped (with a note) for files that are not on this machine.

use ddai_fly::bundle::{NeuronModel, load_bundle, peek_version, read_zstd_bytes, save_bundle};

fn runs() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var("HOME").ok()?).join("aiddnet/data/runs");
    p.exists().then_some(p)
}

#[test]
fn old_checkpoints_load_as_rate_flies_and_round_trip() {
    let Some(runs) = runs() else {
        eprintln!("note: no run directory, skipping");
        return;
    };
    let files = [
        "E-022/bundles/s2-upgraded.bundle",
        "E-029/bundles/bc-legacy.bundle",
        "E-029/bundles/bc-intent.bundle",
        "E-029/bundles/s2-upgraded-intent.bundle",
        "E-031/bundles/s2-pooled.bundle",
        "E-031/bundles/r2-mlp32.bundle",
        "E-031/bundles/s2-mlp-enc-32.bundle",
    ];
    let dir = tempfile::tempdir().unwrap();
    let (mut checked, mut same_layout) = (0, 0);
    let mut versions = std::collections::BTreeSet::new();
    for rel in files {
        let path = runs.join(rel);
        if !path.exists() {
            eprintln!("note: {} not found, skipping", path.display());
            continue;
        }
        let original = std::fs::read(&path).unwrap();
        let bytes = read_zstd_bytes(&path).unwrap();
        let version = peek_version(&path, &bytes).unwrap();
        assert!(version <= 5, "{rel}: version {version}");
        versions.insert(version);
        let b = load_bundle(&path).unwrap();
        assert!(matches!(b.neuron_model, NeuronModel::Rate), "{rel}");
        let again = dir.path().join("again.bundle");
        save_bundle(&again, &b).unwrap();
        // The postcard payload is what a reader decodes (the zstd stream around it depends on the compressor's version, not on this code).
        let new_bytes = read_zstd_bytes(&again).unwrap();
        let new_version = peek_version(&again, &new_bytes).unwrap();
        // Writing the same checkpoint again gives the same checkpoint, and, where the writer's layout is the file's own, the same payload. (The
        // writer picks the oldest layout that plays the fly correctly, 8.6 review F2, so a legacy fly an earlier writer left as v4 comes back as v3.)
        assert_eq!(
            load_bundle(&again).unwrap(),
            b,
            "{rel}: the re-written checkpoint differs"
        );
        if new_version == version {
            assert!(
                new_bytes == bytes,
                "{rel}: the re-written payload differs from the original"
            );
            if std::fs::read(&again).unwrap() != original {
                eprintln!("note: {rel}: payload identical, compressed stream differs (the zstd version of the writer)");
            }
            same_layout += 1;
        } else {
            eprintln!("note: {rel}: written as v{new_version} (the file is v{version})");
        }
        checked += 1;
    }
    eprintln!(
        "{checked} old checkpoints (format versions {versions:?}) round-trip; {same_layout} written back with the very same payload"
    );
    assert!(
        checked == 0 || same_layout > 0,
        "no checkpoint was written back in its own layout"
    );
}
