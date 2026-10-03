//! What one fly proposal costs, in the work clock's unit (task 3.7a, D-080): tee-tick equivalents.
//!
//! The arena's work clock counts simulated physics (`WORK_US_PER_TEE_TICK` = 1.25 us per tee-tick of a whole
//! decision); a proposer that runs a network instead of simulating physics (the fly) must be charged for its time
//! in the same unit or the work clock would let it in for free. The method:
//!
//! 1. play the same scenes (Copy Love Box, slot 0 the hybrid on the work clock, the others scripted) once without
//!    a proposer and measure the **wall microseconds per tee-tick of a whole decision** (`c`, as E-010 does);
//! 2. play them with the fly as proposer and time every `propose` call on the wall clock (`f`);
//! 3. `units = f / c`: how many tee-ticks of search one proposal costs *on this machine right now*. Both numbers
//!    come from the same process, interleaved, so the load of a shared VM scales them alike and cancels.
//!
//! The result (median over the repetitions of the median over decisions) is the constant behind
//! `ddai_fly::proposer::FLY_PROPOSAL_TEE_TICKS_PER_GSYNSTEP` (the cost is proportional to synapses x substeps).
//!
//! ```text
//! cargo test -p ddai-fly --release --test fly_proposal_cost -- --ignored --nocapture
//! ```

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use ddai_brain::{Brain, CharacterObservation, Observation, ResetContext, WorldView};
use ddai_fly::proposer::{FlyProposer, untrained_fly_brain};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::World;
use ddai_planner::brains::{ClockKind, ScriptedBrain, input_from_action};
use ddai_planner::hybrid::{
    HybridBrain, HybridConfig, HybridMode, NoProposer, ProposeCtx, Proposer, WORK_US_PER_TEE_TICK,
};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::planner::PlanStep;
use ddai_planner::types::{PlayerInput, empty_input};
use ddai_planner::vmath::Vec2;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

fn clb() -> Option<Arc<MapData>> {
    let dir = home().join("aiddnet/data/maps/copy-love-box");
    let f = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "map"))?;
    let m = ddai_map::load_map(&std::fs::read(f).ok()?).ok()?;
    Some(Arc::new(m.data))
}

fn char_obs(w: &World<f32>, id: i32) -> CharacterObservation {
    let core = w.cores.get(id as u8).expect("tee");
    let ch = w.characters[id as usize].as_ref().expect("character");
    let mut c = CharacterObservation::at_rest(id);
    c.pos = core.pos;
    c.vel = core.vel;
    c.hook_state = core.hook_state;
    c.hooked_player = core.hooked_player();
    c.is_frozen = ch.freeze_time > 0;
    c.freeze_ticks_remaining = ch.freeze_time;
    c.direction = core.direction;
    c
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A proposer that times the wall microseconds of the wrapped one's `propose` calls.
struct Timed {
    inner: Box<dyn Proposer>,
    us: Rc<RefCell<Vec<f64>>>,
}

impl Proposer for Timed {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }
    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        let t0 = Instant::now();
        self.inner.propose(ctx, out);
        self.us.borrow_mut().push(t0.elapsed().as_secs_f64() * 1e6);
    }
    fn costs_time(&self) -> bool {
        self.inner.costs_time()
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// One decision of slot 0: its wall microseconds and the telemetry.
struct Row {
    wall_us: f64,
    tel: ddai_planner::hybrid::DecisionTelemetry,
}

impl Row {
    /// Tee-ticks the work clock charged for the decision (physics ticks x tees, the proposer's units, rays).
    fn tee_ticks(&self) -> f64 {
        (self.tel.work.total_ticks() * u64::from(self.tel.sim_tees.max(1))
            + self.tel.work.proposal_units
            + 2 * self.tel.work.rays) as f64
    }
}

/// `(wall microseconds of the whole decision, tee-ticks of it)` per decision of slot 0, `decisions` of them.
fn play(map: &Arc<MapData>, proposer: Box<dyn Proposer>, tees: usize, decisions: usize) -> Vec<(f64, f64)> {
    // Whatever the proposer costs, the *search* must run at its usual size: this measures the cost per tee-tick of
    // the search and the shield, so the proposal must not shrink the budget here.
    let rows = play_with(map, proposer, tees, decisions, |c| {
        c.proposal_in_cap = false;
        c.adaptive.enabled = false;
    });
    rows.iter().map(|r| (r.wall_us, r.tee_ticks())).collect()
}

fn play_with(
    map: &Arc<MapData>,
    proposer: Box<dyn Proposer>,
    tees: usize,
    decisions: usize,
    tweak: impl FnOnce(&mut HybridConfig),
) -> Vec<Row> {
    let ids: Vec<i32> = (0..tees as i32).collect();
    let mut cfg = HybridConfig {
        mode: HybridMode::Deadline { budget_ms: 4.0 },
        work_clock_us_per_tick: Some(WORK_US_PER_TEE_TICK),
        ..HybridConfig::default()
    };
    tweak(&mut cfg);
    let mut hybrid = HybridBrain::new(cfg, ClockKind::Wall, proposer).expect("config");
    let mut rows: Vec<Row> = Vec::new();
    let mut scene = 0u64;
    while rows.len() < decisions {
        scene += 1;
        let mut pw = PhysicsWorld::new(map.clone(), scene);
        for i in 0..tees {
            pw.add_tee(
                i as i32,
                Vec2 {
                    x: (103.5 - 2.4 * i as f64 - if i > 0 { 1.5 } else { 0.0 }) * 32.0,
                    y: 84.5 * 32.0,
                },
            );
        }
        hybrid.reset(&ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: scene,
        });
        let mut scripted: Vec<ScriptedBrain> = (1..tees).map(|_| ScriptedBrain::new()).collect();
        for (k, b) in scripted.iter_mut().enumerate() {
            b.reset(&ResetContext {
                map: map.clone(),
                self_id: (k + 1) as i32,
                seed: scene * 10 + k as u64,
            });
        }
        let mut last: Vec<PlayerInput> = ids.iter().map(|_| empty_input()).collect();
        for _ in 0..60 {
            if rows.len() >= decisions {
                break;
            }
            let world = pw.inner().clone();
            let out = |id: i32| pw.get_tee(id).is_none_or(|t| t.frozen || !t.alive);
            if out(0) || (1..tees as i32).all(out) {
                break;
            }
            for (slot, &id) in ids.iter().enumerate() {
                if world.cores.get(id as u8).is_none() {
                    continue;
                }
                let target = if slot == 0 { 1 } else { 0 };
                let obs = Observation {
                    map: map.clone(),
                    tick: world.tick,
                    self_state: char_obs(&world, id),
                    others: ids
                        .iter()
                        .filter(|&&i| i != id && world.cores.get(i as u8).is_some())
                        .map(|&i| char_obs(&world, i))
                        .collect(),
                    target_id: Some(target),
                    tuning: TuningParams::default(),
                };
                let view = WorldView {
                    world: &world,
                    self_id: id,
                    lag_ticks: 0,
                    in_flight: &[],
                };
                let action = if slot == 0 {
                    let t0 = Instant::now();
                    let a = hybrid.decide_in(&obs, Some(&view));
                    let wall = t0.elapsed().as_secs_f64() * 1e6;
                    let t = hybrid.last_decision().expect("telemetry").clone();
                    rows.push(Row { wall_us: wall, tel: t });
                    a
                } else {
                    scripted[slot - 1].decide_in(&obs, Some(&view))
                };
                last[slot] = input_from_action(&action, &last[slot]);
            }
            for _ in 0..2 {
                for (slot, &id) in ids.iter().enumerate() {
                    pw.set_input(id, last[slot]);
                }
                pw.step();
            }
        }
    }
    rows
}

#[test]
#[ignore = "heavy; needs the Copy Love Box map and the compiled S graph"]
fn fly_proposal_cost_in_tee_ticks() {
    let (Some(map), flyg) = (clb(), home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg")) else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    if !flyg.exists() {
        eprintln!("no compiled S graph; skipping");
        return;
    }
    let cfg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let reps: usize = std::env::var("DDAI_CAL_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let brain = untrained_fly_brain(&flyg, &cfg_path, 1).expect("untrained fly");
    let model = brain.model();
    let nnz = *model.flyg().edges.row_start.last().expect("row_start") as f64;
    let substeps = f64::from(model.config().substeps_per_decision);
    println!(
        "S graph: {} neurons, {nnz} synapse rows, {substeps} substeps per decision; work clock {WORK_US_PER_TEE_TICK} us per tee-tick",
        model.num_neurons()
    );
    drop(brain);
    println!(
        "\n| tees | rep | us per tee-tick, no proposer: whole run / p10 of decisions | fly propose us: min / p10 / p50 / p99 | units p50/whole, p10/p10 |\n|---|---|---|---|---|"
    );
    let mut units_p50 = Vec::new();
    let mut units_lo = Vec::new();
    for tees in [2usize, 4] {
        for rep in 0..reps {
            let bare = play(&map, Box::new(NoProposer), tees, n);
            let c = bare.iter().map(|r| r.0).sum::<f64>() / bare.iter().map(|r| r.1).sum::<f64>();
            let mut per: Vec<f64> = bare.iter().map(|r| r.0 / r.1.max(1.0)).collect();
            let c10 = quantile(&mut per, 0.1);
            let us = Rc::new(RefCell::new(Vec::new()));
            let wrapped = Timed {
                inner: Box::new(FlyProposer::new(
                    untrained_fly_brain(&flyg, &cfg_path, 1).expect("fly"),
                    1,
                )),
                us: Rc::clone(&us),
            };
            let _ = play(&map, Box::new(wrapped), tees, n);
            let mut v = us.borrow().clone();
            let (mn, p10, p50, p99) = (
                quantile(&mut v, 0.0),
                quantile(&mut v, 0.1),
                quantile(&mut v, 0.5),
                quantile(&mut v, 0.99),
            );
            println!(
                "| {tees} | {rep} | {c:.3} / {c10:.3} | {mn:.0} / {p10:.0} / {p50:.0} / {p99:.0} | {:.0}, {:.0} |",
                p50 / c,
                p10 / c10
            );
            units_p50.push(p50 / c);
            units_lo.push(p10 / c10);
        }
    }
    let (u, lo) = (median(&mut units_p50), median(&mut units_lo));
    println!(
        "\nmedian units: {u:.0} (p50 fly / whole-run rate), {lo:.0} (p10 fly / p10 rate: the least load-affected pair); {u:.0} tee-ticks = {:.2} ms at {WORK_US_PER_TEE_TICK} us; per synapse-step {:.4e} tee-ticks ({:.0} per mega-synapse-step)",
        u * WORK_US_PER_TEE_TICK / 1000.0,
        u / (nnz * substeps),
        u / (nnz * substeps) * 1e6
    );
}

/// The price list: the S graph's proposal costs 500 tee-ticks (the calibration above), a `FlyProposer` reports
/// it, and a proposer that costs time says so (the search charges the work clock and the decision cap with it).
#[test]
fn the_s_graph_proposal_is_priced_at_500_tee_ticks() {
    let flyg = home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    if !flyg.exists() {
        eprintln!("no compiled S graph; skipping");
        return;
    }
    let cfg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    let brain = untrained_fly_brain(&flyg, &cfg_path, 1).expect("untrained fly");
    assert_eq!(ddai_fly::proposer::proposal_tee_ticks(&brain), 500);
    let p = FlyProposer::new(brain, 1);
    assert_eq!(p.work_units(), 500);
    assert!(p.costs_time());
}

/// The A/B of task 3.7a, point 1: does counting the proposal inside the decision cap bring `hybrid:fly` back under
/// 5 ms of work? Work clock (1.25 us per tee-tick; the fly charged 500 tee-ticks), CLB, slot 0 the hybrid against
/// scripted tees; three arms: no proposer, the fly with its time outside the cap (3.5-3.6) and inside (3.7a);
/// the extension of D-042 off (the plain decision) and on. Work = tee-ticks charged x rate: it does not depend on the
/// machine's load, the wall columns do.
///
/// ```text
/// DDAI_SPEED_DECISIONS=1000 cargo test -p ddai-fly --release --test fly_proposal_cost -- --ignored --nocapture fly_budget_ab
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map and the compiled S graph"]
fn fly_budget_ab() {
    let (Some(map), flyg) = (clb(), home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg")) else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    if !flyg.exists() {
        eprintln!("no compiled S graph; skipping");
        return;
    }
    let cfg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    let n: usize = std::env::var("DDAI_SPEED_DECISIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let fly = || -> Box<dyn Proposer> {
        Box::new(FlyProposer::new(
            untrained_fly_brain(&flyg, &cfg_path, 1).expect("fly"),
            1,
        ))
    };
    println!(
        "| arm | tees | extension | work ms p50 / p90 / p99 / max | proposal ms p50 | search budget ms p50 / min | candidates p50 | extended % | decision wall ms p50 / p99 |\n|---|---|---|---|---|---|---|---|---|"
    );
    for tees in [2usize, 4] {
        for extension in [false, true] {
            for (label, in_cap, with_fly) in [
                ("hybrid, no proposer", true, false),
                ("hybrid:fly, proposal outside the cap (3.6)", false, true),
                ("hybrid:fly, proposal inside the cap (3.7a)", true, true),
            ] {
                let proposer = if with_fly { fly() } else { Box::new(NoProposer) };
                let rows = play_with(&map, proposer, tees, n, |c| {
                    c.proposal_in_cap = in_cap;
                    c.adaptive.enabled = extension;
                });
                let mut work: Vec<f64> = rows
                    .iter()
                    .map(|r| r.tee_ticks() * WORK_US_PER_TEE_TICK / 1000.0)
                    .collect();
                let mut prop: Vec<f64> = rows.iter().map(|r| r.tel.proposal_ms).collect();
                let mut budget: Vec<f64> = rows.iter().map(|r| r.tel.budget_ms).collect();
                let mut cands: Vec<f64> = rows
                    .iter()
                    .map(|r| f64::from(r.tel.evaluated.iter().sum::<u32>()))
                    .collect();
                let mut wall: Vec<f64> = rows.iter().map(|r| r.wall_us / 1000.0).collect();
                let ext = 100.0 * rows.iter().filter(|r| r.tel.extended).count() as f64 / rows.len() as f64;
                println!(
                    "| {label} | {tees} | {} | {:.2} / {:.2} / {:.2} / {:.2} | {:.2} | {:.2} / {:.2} | {:.0} | {ext:.1} | {:.2} / {:.2} |",
                    if extension { "on" } else { "off" },
                    quantile(&mut work, 0.5),
                    quantile(&mut work, 0.9),
                    quantile(&mut work, 0.99),
                    quantile(&mut work, 1.0),
                    quantile(&mut prop, 0.5),
                    quantile(&mut budget, 0.5),
                    quantile(&mut budget, 0.0),
                    quantile(&mut cands, 0.5),
                    quantile(&mut wall, 0.5),
                    quantile(&mut wall, 0.99),
                );
            }
        }
    }
}

/// Task 3.7a, point 2 (in process, without a server): candidates per decision and the decision's wall time with 1, 2
/// and 4 search threads on the **wall clock**, the fly as proposer (inside the cap) or none. The machine is shared and
/// its load swings by a factor of two within minutes, so the arms are **interleaved**: `DDAI_ROUNDS` rounds, each
/// playing `DDAI_SPEED_DECISIONS` decisions per arm in turn, and the numbers are pooled per arm over the rounds. The
/// first 15 decisions of every call (a cold engine) are dropped. The load average is printed at every round.
///
/// ```text
/// DDAI_ROUNDS=8 DDAI_SPEED_DECISIONS=200 cargo test -p ddai-fly --release --test fly_proposal_cost -- --ignored --nocapture fly_wall_threads
/// ```
#[test]
#[ignore = "heavy; needs the Copy Love Box map and the compiled S graph"]
fn fly_wall_threads() {
    let (Some(map), flyg) = (clb(), home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg")) else {
        eprintln!("no Copy Love Box map; skipping");
        return;
    };
    if !flyg.exists() {
        eprintln!("no compiled S graph; skipping");
        return;
    }
    let loadavg = || {
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let cfg_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let n = env_usize("DDAI_SPEED_DECISIONS", 200);
    let rounds = env_usize("DDAI_ROUNDS", 8);
    let arms: Vec<(bool, usize, usize)> = [false, true]
        .into_iter()
        .flat_map(|f| {
            [2usize, 4]
                .into_iter()
                .flat_map(move |t| [1usize, 2, 4].into_iter().map(move |w| (f, t, w)))
        })
        .collect();
    let mut pooled: Vec<Vec<Row>> = arms.iter().map(|_| Vec::new()).collect();
    for round in 0..rounds {
        println!("round {round}: load average at the start: {}", loadavg());
        for (i, &(with_fly, tees, workers)) in arms.iter().enumerate() {
            let proposer: Box<dyn Proposer> = if with_fly {
                Box::new(FlyProposer::new(
                    untrained_fly_brain(&flyg, &cfg_path, 1).expect("fly"),
                    1,
                ))
            } else {
                Box::new(NoProposer)
            };
            let rows = play_with(&map, proposer, tees, n + 15, |c| {
                c.work_clock_us_per_tick = None;
                c.workers = workers;
            });
            pooled[i].extend(rows.into_iter().skip(15));
        }
    }
    println!("load average at the end: {}", loadavg());
    println!(
        "\n| proposer | tees | threads | decisions | candidates p50 / p90 / mean | decision wall ms p50 / p90 / p99 / max | proposal ms p50 / p99 | search budget ms p50 | search ms p50 / p99 | extended % |\n|---|---|---|---|---|---|---|---|---|---|"
    );
    for (&(with_fly, tees, workers), rows) in arms.iter().zip(&pooled) {
        let col = |f: &dyn Fn(&Row) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
        let mut cands = col(&|r| f64::from(r.tel.evaluated.iter().sum::<u32>()));
        let mean = cands.iter().sum::<f64>() / cands.len() as f64;
        let mut wall = col(&|r| r.wall_us / 1000.0);
        let mut prop = col(&|r| r.tel.proposal_ms);
        let mut budget = col(&|r| r.tel.budget_ms);
        let mut search = col(&|r| r.tel.search_ms);
        let ext = 100.0 * rows.iter().filter(|r| r.tel.extended).count() as f64 / rows.len() as f64;
        println!(
            "| {} | {tees} | {workers} | {} | {:.0} / {:.0} / {mean:.1} | {:.2} / {:.2} / {:.2} / {:.2} | {:.2} / {:.2} | {:.2} | {:.2} / {:.2} | {ext:.1} |",
            if with_fly { "fly" } else { "none" },
            rows.len(),
            quantile(&mut cands, 0.5),
            quantile(&mut cands, 0.9),
            quantile(&mut wall, 0.5),
            quantile(&mut wall, 0.9),
            quantile(&mut wall, 0.99),
            quantile(&mut wall, 1.0),
            quantile(&mut prop, 0.5),
            quantile(&mut prop, 0.99),
            quantile(&mut budget, 0.5),
            quantile(&mut search, 0.5),
            quantile(&mut search, 0.99),
        );
    }
}
