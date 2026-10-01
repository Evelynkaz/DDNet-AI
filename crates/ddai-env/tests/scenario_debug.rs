//! Diagnostic: plays one scenario's trials with the hybrid brain (default config on the work clock,
//! 4 ms, adaptive on) and prints, for every FAILED trial, the brain's decisions (chosen candidate, safety
//! flags, candidates evaluated). Heavy output, so `#[ignore]`:
//!
//! ```text
//! DDAI_SCEN=T13 DDAI_TRIALS=100 cargo test -p ddai-env --release --test scenario_debug -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_env::scenario::*;
use ddai_planner::brains::ClockKind;
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer};

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

fn map_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join("aiddnet/data/maps"))
        .unwrap_or_default()
}

/// The hybrid brain, logging every decision.
struct Tap {
    inner: Box<HybridBrain>,
    log: Arc<Mutex<Vec<String>>>,
}

impl Brain for Tap {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }
    fn decide(&mut self, obs: &Observation) -> Action {
        self.inner.decide(obs)
    }
    fn decide_in(&mut self, obs: &Observation, world: Option<&WorldView<'_>>) -> Action {
        let a = self.inner.decide_in(obs, world);
        if let Some(t) = self.inner.last_decision() {
            let o = &obs.self_state;
            self.log.lock().unwrap().push(format!(
                "tick {:3} me ({:6.0},{:6.0}) v ({:5.1},{:5.1}) jumps {} -> {:22} unsafe {} ext {} shielded {} incomplete {} plan_ok {} cands {:2} pruned {} danger [{}] belief {:.2} act d{} j{} h{} f{}",
                obs.tick,
                o.pos.x,
                o.pos.y,
                o.vel.x,
                o.vel.y,
                obs.self_state.jumps_left,
                t.chosen.map_or("none", |c| c.label()),
                u8::from(t.unsafe_choice),
                u8::from(t.extended),
                u8::from(t.shielded),
                u8::from(t.shield_incomplete),
                u8::from(t.shield_plan_ok),
                t.evaluated.iter().sum::<u32>(),
                t.pruned,
                t.danger.reasons(),
                t.react_belief,
                a.direction,
                u8::from(a.jump),
                u8::from(a.hook),
                u8::from(a.fire)
            ));
            if obs.tick <= 6 {
                for d in t.dump.iter().take(8) {
                    self.log.lock().unwrap().push(format!(
                        "      cand {:26} cheap {:7.3} combos {:?} step0 {}",
                        d.0, d.1, d.2, d.3
                    ));
                }
            }
        }
        a
    }
    fn name(&self) -> &str {
        "tap"
    }
}

#[test]
#[ignore = "diagnostic; set DDAI_SCEN"]
fn failed_trials_with_decisions() {
    let Ok(scen) = std::env::var("DDAI_SCEN") else { return };
    let trials: u32 = std::env::var("DDAI_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let dir = std::env::var("DDAI_SCEN_DIR").unwrap_or_else(|_| "configs/scenarios".to_string());
    let def = ScenarioDef::load_dir(&repo(&dir))
        .unwrap()
        .into_iter()
        .find(|d| d.id == scen)
        .expect("scenario");
    let world = load_world(&def, &map_dir()).unwrap();
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            let (mut failed, mut shown) = (0, 0);
            for trial in 0..trials {
                let log = Arc::new(Mutex::new(Vec::new()));
                let mut cfg = HybridConfig {
                    work_clock_us_per_tick: Some(2.2),
                    proposals: 0,
                    debug_dump: std::env::var("DDAI_DUMP").is_ok(),
                    ..HybridConfig::default()
                };
                // Options under test, e.g. DDAI_AIR=0.75 DDAI_FIXED=1.
                cfg.warm_fire_only = std::env::var("DDAI_FIRE_ONLY").is_ok();
                if let Some(v) = std::env::var("DDAI_WARM").ok().and_then(|v| v.parse().ok()) {
                    cfg.warm_bonus = v;
                }
                if let Some(v) = std::env::var("DDAI_AIR").ok().and_then(|v| v.parse().ok()) {
                    cfg.planner.jumpless_air_cost = v;
                }
                let brain = Tap {
                    inner: Box::new(HybridBrain::new(cfg, ClockKind::Wall, Box::new(NoProposer)).unwrap()),
                    log: Arc::clone(&log),
                };
                let out = run_trial(&def, &world, Box::new(brain), 0, 1, trial, true).unwrap();
                if !out.success {
                    failed += 1;
                    if shown < 3 {
                        shown += 1;
                        println!("--- {scen} trial {trial} FAILED");
                        for l in log.lock().unwrap().iter().take(40) {
                            println!("{l}");
                        }
                    }
                }
            }
            println!("{scen}: {failed} of {trials} trials failed");
        })
        .unwrap()
        .join()
        .unwrap();
}
