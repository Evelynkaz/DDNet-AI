//! Task 8.5a, the identity proof of the opponent-state channels: the E-008 selected fly (two-view, `hook_view = MaskedForHookHead`)
//! plays 24 arena games (`configs/arena/e022-identity.toml`: 8 on each of clb-left, pit and chillblock5-ruler) and the decision
//! hashes of both players, the results and the end ticks are **exactly** those recorded with the binary built before the encoder
//! learned the channels (`tests/golden/opp_identity.json`, made with `ddnet-ai arena run` at commit ca53644):
//!
//! * with the bundle as it is (an old bundle must load and play as before), and
//! * with the bundle upgraded by `upgrade_with_opponent_state` using the real `configs/fly/S-opponent-state.toml` (zero weights for
//!   the new channels, so the opponent's frozen state, freeze time, velocity and hook reach the network and change nothing).
//!
//! It needs the local data (the bundle, the S graph and the maps), so it is `#[ignore]`d:
//! `cargo test -p ddai-train --release --test opp_identity -- --ignored --nocapture`
//! (`E008_BUNDLE` overrides the bundle path).

use std::path::{Path, PathBuf};

use ddai_env::config::{PlayerSpec, RunConfig};
use ddai_env::models::{ModelBrains, player_from_arg};
use ddai_env::run::{load_arenas, run_condition};
use ddai_fly::bundle::{load_bundle, save_bundle, sha256_hex_of_file, upgrade_with_opponent_state};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn home(rel: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(rel)
}

/// Plays the identity set with `bundle` in the focal seat and returns `condition -> [[hash0, hash1, result, end tick]]`.
fn play(bundle: &Path, flyg: &Path, threads: usize) -> serde_json::Value {
    let text = std::fs::read_to_string(root().join("configs/arena/e022-identity.toml")).unwrap();
    let cfg = RunConfig::parse(&text).unwrap();
    let arenas = load_arenas(&cfg, &root().join("configs/arenas"), &home("aiddnet/data/maps")).unwrap();
    let models = std::sync::Arc::new(ModelBrains::new(Some(flyg.to_path_buf())));
    let factory = models.factory();
    let mut out = serde_json::Map::new();
    for cond in &cfg.condition {
        let mut cond = cond.clone();
        let mut focal: PlayerSpec = player_from_arg(&format!("fly:{}", bundle.display()));
        focal.count = 1;
        cond.players[0] = focal;
        let run = run_condition(
            &cfg,
            &cond,
            &arenas[&cond.arena],
            cfg.games_for(&cond),
            &factory,
            threads,
        )
        .unwrap();
        let rows: Vec<serde_json::Value> = run
            .games
            .iter()
            .map(|g| {
                serde_json::json!([
                    g.players[0].hash,
                    g.players[1].hash,
                    format!("{:?}", g.result),
                    g.end_tick
                ])
            })
            .collect();
        out.insert(cond.name.clone(), serde_json::Value::Array(rows));
    }
    serde_json::Value::Object(out)
}

#[test]
#[ignore = "needs the local E-008 bundle, the S graph and the maps"]
fn the_e008_fly_plays_the_same_24_games_before_and_after_the_opponent_channels() {
    let bundle = std::env::var_os("E008_BUNDLE").map_or_else(
        || home("aiddnet/data/runs/E-008/e008-p2-fly-d2-maskhook-s2/checkpoints/selected.bundle"),
        PathBuf::from,
    );
    let b = load_bundle(&bundle).unwrap();
    let flyg = PathBuf::from(&b.flyg_path_hint);
    assert_eq!(sha256_hex_of_file(&flyg).unwrap(), b.flyg_sha256);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("golden/opp_identity.json")).expect("golden hashes");

    // 1. The old bundle with the new code.
    assert_eq!(play(&bundle, &flyg, 3), golden, "the old bundle changed its play");

    // 2. The upgraded bundle: the channels exist, their weights are zero.
    let section = std::fs::read_to_string(root().join("configs/fly/S-opponent-state.toml")).unwrap();
    let up = upgrade_with_opponent_state(&b, ddai_flyg::load(&flyg).unwrap(), &section).unwrap();
    assert!(up.encoder_params.g.len() > b.encoder_params.g.len());
    assert!(
        up.encoder_params.g[b.encoder_params.g.len()..]
            .iter()
            .all(|&x| x == 0.0)
    );
    assert!(
        up.encoder_params.c[b.encoder_params.c.len()..]
            .iter()
            .all(|&x| x == 0.0)
    );
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("upgraded.bundle");
    save_bundle(&p, &up).unwrap();
    // One thread this time: the hashes do not depend on it either.
    assert_eq!(
        play(&p, &flyg, 1),
        golden,
        "the upgraded bundle with zero weights changed the play"
    );

    // 3. The check can fail: with the "frozen" and "freeze time left" gains non-zero the same games are played differently.
    let mut loud = up.clone();
    for g in &mut loud.encoder_params.g[b.encoder_params.g.len()..] {
        *g = 1.0;
    }
    let p2 = dir.path().join("loud.bundle");
    save_bundle(&p2, &loud).unwrap();
    assert_ne!(
        play(&p2, &flyg, 3),
        golden,
        "non-zero weights for the new channels must change the play"
    );
}
