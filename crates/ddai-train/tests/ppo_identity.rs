//! Task 8.5b, the identity proof of the policy's deterministic mode: the E-008 selected fly (two-view, upgraded with the opponent channels, zero
//! weights) plays the 24 arena games of `configs/arena/e022-identity.toml` through the PPO actor in **arg-max** mode (`PpoActor`, the
//! stochastic-policy code path of `ddai_fly::policy`) and the decision hashes of both players, the results and the end ticks are
//! **exactly** those recorded before any of this existed (`tests/golden/opp_identity.json`, the golden of 8.5a).
//!
//! It needs the local data (the bundle, the S graph and the maps), so it is `#[ignore]`d:
//! `cargo test -p ddai-train --release --test ppo_identity -- --ignored --nocapture` (`E008_BUNDLE` overrides the bundle path).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ddai_brain::Brain;
use ddai_env::config::{PlayerSpec, RunConfig};
use ddai_env::models::{ModelBrains, player_from_arg};
use ddai_env::run::{load_arenas, run_condition};
use ddai_fly::bundle::{FlyBrainTemplate, load_bundle, save_bundle, sha256_hex_of_file, upgrade_with_opponent_state};
use ddai_train::ppo::actor::{ActMode, PpoActor, WindowGrid};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn home(rel: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(rel)
}

fn play(bundle: &Path, flyg: &Path, threads: usize) -> serde_json::Value {
    let text = std::fs::read_to_string(root().join("configs/arena/e022-identity.toml")).unwrap();
    let cfg = RunConfig::parse(&text).unwrap();
    let arenas = load_arenas(&cfg, &root().join("configs/arenas"), &home("aiddnet/data/maps")).unwrap();
    let models = Arc::new(ModelBrains::new(Some(flyg.to_path_buf())));
    let inner = models.factory();
    let template = FlyBrainTemplate::load(bundle, Some(flyg)).unwrap();
    let cache: Mutex<HashMap<(), ()>> = Mutex::new(HashMap::new());
    let _ = &cache;
    // The focal fly comes from the PPO actor (arg-max); everything else from the usual factory.
    let factory = move |spec: &PlayerSpec| -> Result<Box<dyn Brain>, ddai_env::EnvError> {
        if spec.brain.starts_with("fly:") {
            let sink = Arc::new(Mutex::new(Vec::new()));
            let grid = WindowGrid {
                chunk: 32,
                burn_in: 8,
                decide_every: 2,
            };
            Ok(Box::new(PpoActor::new(
                &template,
                ActMode::Argmax,
                8.0,
                ddai_fly::policy::Temperatures::uniform(0.3),
                0,
                grid,
                sink,
            )))
        } else {
            inner(spec)
        }
    };
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
fn the_ppo_actor_in_argmax_mode_plays_the_same_24_games_as_the_recorded_binary() {
    let bundle = std::env::var_os("E008_BUNDLE").map_or_else(
        || home("aiddnet/data/runs/E-008/e008-p2-fly-d2-maskhook-s2/checkpoints/selected.bundle"),
        PathBuf::from,
    );
    let b = load_bundle(&bundle).unwrap();
    let flyg = PathBuf::from(&b.flyg_path_hint);
    assert_eq!(sha256_hex_of_file(&flyg).unwrap(), b.flyg_sha256);
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("golden/opp_identity.json")).expect("golden hashes");
    // The old bundle, played by the actor.
    assert_eq!(
        play(&bundle, &flyg, 3),
        golden,
        "the actor in arg-max mode changed the play of the old bundle"
    );
    // The upgraded bundle (zero weights for the opponent channels).
    let section = std::fs::read_to_string(root().join("configs/fly/S-opponent-state.toml")).unwrap();
    let up = upgrade_with_opponent_state(&b, ddai_flyg::load(&flyg).unwrap(), &section).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("upgraded.bundle");
    save_bundle(&p, &up).unwrap();
    assert_eq!(
        play(&p, &flyg, 1),
        golden,
        "the actor in arg-max mode changed the play of the upgraded bundle"
    );
}
