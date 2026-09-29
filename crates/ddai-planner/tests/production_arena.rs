//! Acceptance criterion 5 (D-041 production mode): decision-time and quality-vs-budget
//! measurements on `ddai_physics::World<f32>` (Copy Love Box). Not a correctness test (no
//! assertion compares against TS) -- prints numbers to stderr under `--nocapture`, the way
//! `docs/research/orig-run.md`'s own phase-0 harness does. `#[ignore]`d because it needs the
//! local, uncommitted Copy Love Box `.map` file (`CLAUDE.md`: map data is never committed) and
//! takes tens of seconds to minutes depending on how many games/decisions are requested via env
//! vars (see each test's doc comment).

use ddai_jsmath::Rng;
use ddai_planner::clock::WallClock;
use ddai_planner::config::PlannerConfig;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::planner::Planner;
use ddai_planner::scripted::scripted_action;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;
use std::sync::Arc;
use std::time::Instant;

const CLB_PATH: &str = "/home/ubuntu/aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map";

fn load_clb() -> Arc<ddai_physics::map::MapData> {
    let bytes = std::fs::read(CLB_PATH)
        .unwrap_or_else(|e| panic!("reading {CLB_PATH}: {e} (local map data, not committed -- see CLAUDE.md)"));
    let loaded = ddai_map::load_map(&bytes).unwrap_or_else(|e| panic!("parsing {CLB_PATH}: {e:?}"));
    Arc::new(loaded.data)
}

/// Standable tiles: free space with solid ground directly beneath -- generic over any
/// [`PlanCollision`], the same predicate `tools/ts-trace/gen-planner-dump.mjs` uses. Review round
/// 2, F15: `hall` restricts to the E-000 left-hall arena (WB boxes L1 `{79..104,67..79}` union L2
/// `{78..104,79..87}`) so a quality/timing run is directly comparable to E-000's own numbers and
/// to `phase_breakdown.rs`'s `hall=true` condition -- the same ranges, so the same tiles.
fn find_standable(col: &impl PlanCollision, hall: bool) -> Vec<Vec2> {
    let tile_center = |t: i32| f64::from(t) * 32.0 + 16.0;
    let mut out = Vec::new();
    for ty in 0..col.height() - 1 {
        for tx in 0..col.width() {
            if hall
                && !(((79..=104).contains(&tx) && (67..=79).contains(&ty))
                    || ((78..=104).contains(&tx) && (79..=87).contains(&ty)))
            {
                continue;
            }
            let px = tile_center(tx);
            let py = tile_center(ty);
            if col.is_solid(px, py) || col.is_freeze(px, py) || col.is_death(px, py) {
                continue;
            }
            if !col.is_solid(tile_center(tx), py + 32.0) {
                continue;
            }
            out.push(Vec2 { x: px, y: py });
        }
    }
    out
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = (((sorted_ms.len() - 1) as f64) * p).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

/// 95% Wilson score interval for a binomial proportion (`wins` out of `n`).
fn wilson_ci(wins: f64, n: f64) -> (f64, f64) {
    if n == 0.0 {
        return (0.0, 1.0);
    }
    let z = 1.959_964; // 95%
    let phat = wins / n;
    let denom = 1.0 + z * z / n;
    let center = phat + z * z / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) / n) + z * z / (4.0 * n * n)).sqrt();
    (
        ((center - margin) / denom).max(0.0),
        ((center + margin) / denom).min(1.0),
    )
}

fn fixed_iteration_cfg() -> PlannerConfig {
    // The "TS default" search (docs/research/orig-run.md §2.1): normal preset, budgetMs:0 (fixed
    // iterations, deterministic) -- population 20 x iterations 2 plus the opening book. Always
    // driven through `Planner::decide` (TS's own fixed-iteration path), never `decide_production`.
    ddai_planner::config::preset_normal()
}

/// Review round 1, F1: `Planner::decide` (TS's own `budgetMs`/`hardMs`, `decide_once`'s coarse
/// `overCap()` gate on just the opening-book/hookSeeds/throw-seed phase) is a *different*
/// mechanism from the new [`Planner::decide_production`] iterative-deepening search this task
/// adds -- `PlannerConfig.budget_ms`/`hard_ms` are meaningless to `decide_production`, which takes
/// its deadline as a plain parameter instead. Every "budgeted" condition below uses
/// `fixed_iteration_cfg()` (population/iterations/steps exactly like TS's live default) as the
/// *config* and drives it through `decide_production` with an explicit millisecond budget; only
/// the "fixed-iteration" condition itself calls plain `decide()`.
#[allow(clippy::too_many_arguments)]
fn decide_with(
    planner: &mut Planner<PhysicsWorld>,
    world: &mut PhysicsWorld,
    self_id: i32,
    enemy_id: i32,
    prev: ddai_planner::types::PlayerInput,
    enemy_input: ddai_planner::types::PlayerInput,
    clock: &WallClock,
    budget_ms: Option<f64>,
) -> ddai_planner::types::PlayerInput {
    match budget_ms {
        Some(ms) => planner.decide_production(world, self_id, enemy_id, prev, enemy_input, clock, ms),
        None => planner.decide(world, self_id, enemy_id, prev, enemy_input),
    }
}

/// Builds a fresh `PhysicsWorld` on Copy Love Box with `self`/`enemy` at two standable spots
/// `min_tiles`..`max_tiles` apart (tile Chebyshev-ish distance via plain Euclidean here, matching
/// `tools/ts-trace/lib.mjs`'s `arenaFromMap`), plus `extra_bystanders` more tees at other
/// standable spots (for the "6 tees" decision-timing variant -- acceptance criterion 5).
/// Picks two standable spots 3..12 tiles apart -- review round 2, F15: matches E-000's own spawn
/// distance range exactly (Copy Love Box is 387x250 tiles, so two uniformly-random standable
/// spots are very often far outside hook/hazard range of each other, producing long, uninteresting
/// games that mostly time out; this keeps spawns close enough that the pair actually has to
/// fight).
fn pick_close_pair(stand: &[Vec2], rng: &mut Rng) -> (Vec2, Vec2) {
    let pick = |rng: &mut Rng| stand[(rng.next_float() * stand.len() as f64) as usize];
    for _ in 0..10_000 {
        let a = pick(rng);
        let b = pick(rng);
        let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt() / 32.0;
        if (3.0..=12.0).contains(&d) {
            return (a, b);
        }
    }
    (pick(rng), pick(rng))
}

fn build_world(
    map: &Arc<ddai_physics::map::MapData>,
    stand: &[Vec2],
    rng: &mut Rng,
    extra_bystanders: usize,
    seed: u64,
) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(map.clone(), seed);
    let pick = |rng: &mut Rng| stand[(rng.next_float() * stand.len() as f64) as usize];
    let (a, b) = pick_close_pair(stand, rng);
    world.add_tee(0, a);
    world.add_tee(1, b);
    for id in 2..2 + extra_bystanders as i32 {
        world.add_tee(id, pick(rng));
        world.set_held_input(id, empty_input());
    }
    world
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    SelfWins,
    EnemyWins,
    Draw,
    Timeout,
}

// 16s of game time at 50 tick/s -- shorter than orig-run.md §3's 30s ceiling, traded off against
// this task's wall-clock budget on a machine shared with other agents (see BUILD REPORT); still
// well above the ~2-17s most real games resolved in during that phase-0 measurement.
const MAX_TICKS: i32 = 800;

/// Plays one 1v1 game: `self` (id 0) driven by a fresh `Planner<PhysicsWorld>` with `self_cfg`,
/// `enemy` (id 1) driven either by `scriptedAction` or by another `Planner` (`enemy_cfg`).
/// Decisions happen every tick (simplification vs. the live bot's 2-tick snapshot cadence --
/// acceptable here since this measures relative quality between budgets, not absolute play
/// strength against a human).
#[allow(clippy::too_many_arguments)]
fn play_game(
    map: &Arc<ddai_physics::map::MapData>,
    stand: &[Vec2],
    self_cfg: PlannerConfig,
    self_budget: Option<f64>,
    enemy_cfg: Option<PlannerConfig>,
    enemy_budget: Option<f64>,
    seed: u64,
    clock: &WallClock,
) -> Outcome {
    let mut spawn_rng = Rng::new((seed.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF) as u32);
    let mut world = build_world(map, stand, &mut spawn_rng, 0, seed);

    let mut self_planner: Planner<PhysicsWorld> = Planner::new(self_cfg);
    self_planner.reset();
    let mut enemy_planner: Option<Planner<PhysicsWorld>> = enemy_cfg.map(|c| {
        let mut p: Planner<PhysicsWorld> = Planner::new(c);
        p.reset();
        p
    });
    let mut script_rng = Rng::new((seed.wrapping_mul(7919).wrapping_add(17) & 0xFFFF_FFFF) as u32);

    let mut self_prev = empty_input();
    let mut enemy_prev = empty_input();
    let mut self_was_out = false;
    let mut enemy_was_out = false;

    for _tick in 0..MAX_TICKS {
        let self_tee = world.get_tee(0);
        let enemy_tee = world.get_tee(1);
        let (Some(st), Some(et)) = (self_tee, enemy_tee) else {
            break;
        };
        let self_out = st.frozen || !st.alive;
        let enemy_out = et.frozen || !et.alive;
        if self_out && !self_was_out && enemy_out && !enemy_was_out {
            return Outcome::Draw;
        }
        if self_out && !self_was_out {
            return Outcome::EnemyWins;
        }
        if enemy_out && !enemy_was_out {
            return Outcome::SelfWins;
        }
        self_was_out = self_out;
        enemy_was_out = enemy_out;

        let enemy_input = match &mut enemy_planner {
            Some(p) => decide_with(p, &mut world, 1, 0, enemy_prev, self_prev, clock, enemy_budget),
            None => scripted_action(&world, 1, 0, &enemy_prev, &mut script_rng),
        };
        let self_input = decide_with(
            &mut self_planner,
            &mut world,
            0,
            1,
            self_prev,
            enemy_input,
            clock,
            self_budget,
        );

        world.set_input(0, self_input);
        world.set_input(1, enemy_input);
        world.step();
        self_prev = self_input;
        enemy_prev = enemy_input;
    }
    Outcome::Timeout
}

fn games_env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Review round 1, F6: "measure thread CPU time (or stall-filtered wall) alongside wall". No
/// `unsafe` is allowed in this workspace (`Cargo.toml`'s `[workspace.lints.rust] unsafe_code =
/// "deny"`, and the task's own constraints), and per-thread CPU time has no safe stable-`std` API
/// -- so this measures the *stall-filtered wall* alternative the finding explicitly allows: a
/// tight busy-loop of `Instant::now()` reads, with no work between them, so every gap above a
/// noise floor is a scheduler/VM-steal stall, not real compute. Reported alongside every
/// wall-clock percentile below so the reader can judge how much of a p99 tail is this machine
/// stalling versus the search actually taking that long.
struct StallBaseline {
    rate_over_0_5ms: f64,
    rate_over_2ms: f64,
    max_ms: f64,
}
fn measure_stall_baseline(duration_ms: u64) -> StallBaseline {
    let start = Instant::now();
    let mut last = start;
    let mut gaps_ms = Vec::with_capacity(1_000_000);
    while start.elapsed().as_millis() < u128::from(duration_ms) {
        let now = Instant::now();
        gaps_ms.push(now.duration_since(last).as_secs_f64() * 1000.0);
        last = now;
    }
    let n = gaps_ms.len().max(1) as f64;
    let over = |t: f64| gaps_ms.iter().filter(|&&g| g > t).count() as f64 / n;
    StallBaseline {
        rate_over_0_5ms: over(0.5),
        rate_over_2ms: over(2.0),
        max_ms: gaps_ms.iter().copied().fold(0.0, f64::max),
    }
}

/// Review round 1, F6's "clean numbers" methodology: per-operation costs measured in isolation
/// (many iterations, reporting the *median* -- robust to the occasional VM stall
/// [`measure_stall_baseline`] shows this machine has), so `candidates x per-candidate-cost` gives
/// an independent cross-check against the end-to-end wall-clock numbers in [`timing_report`].
fn median_ms(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&xs, 0.5)
}

#[test]
#[ignore = "needs the local Copy Love Box map and takes a while -- see this file's doc comment"]
fn component_cost_report() {
    let stall = measure_stall_baseline(2000);
    eprintln!(
        "baseline machine stall rate (2s busy-loop, no work): {:.4}% of gaps > 0.5ms, {:.4}% > 2ms, max={:.3}ms",
        stall.rate_over_0_5ms * 100.0,
        stall.rate_over_2ms * 100.0,
        stall.max_ms
    );

    let map = load_clb();
    let world2 = PhysicsWorld::new(map.clone(), 1);
    let stand = find_standable(world2.collision(), false);
    let mut spawn_rng = Rng::new(7);
    let mut world = build_world(&map, &stand, &mut spawn_rng, 0, 7);
    world.set_input(0, empty_input());
    world.set_input(1, empty_input());

    let n = 20_000;
    let mut save_ms = Vec::with_capacity(n);
    let mut save_into_ms = Vec::with_capacity(n);
    let mut restore_ms = Vec::with_capacity(n);
    let mut step_ms = Vec::with_capacity(n);
    // The pre-loop value only exists to give `save_state_into`/`restore_state` something to act
    // on before the loop's own timed `save_state()` call overwrites it on iteration 1 -- never
    // itself read, by design.
    #[allow(unused_assignments)]
    let mut saved = world.save_state();
    for _ in 0..n {
        let t0 = Instant::now();
        saved = world.save_state();
        save_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t1 = Instant::now();
        world.save_state_into(&mut saved);
        save_into_ms.push(t1.elapsed().as_secs_f64() * 1000.0);

        let t2 = Instant::now();
        world.restore_state(&saved);
        restore_ms.push(t2.elapsed().as_secs_f64() * 1000.0);

        let t3 = Instant::now();
        world.step();
        step_ms.push(t3.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!(
        "per-op median (n={n}): save_state={:.4}ms save_state_into={:.4}ms restore_state={:.4}ms step={:.4}ms",
        median_ms(save_ms),
        median_ms(save_into_ms),
        median_ms(restore_ms),
        median_ms(step_ms),
    );
}

#[test]
#[ignore = "needs the local Copy Love Box map and takes a while -- see this file's doc comment"]
fn timing_report() {
    let map = load_clb();
    let world2 = PhysicsWorld::new(map.clone(), 1);
    // Whole-map spawns (not hall-restricted, unlike `quality_vs_budget_report` below -- F15 only
    // asked for the *quality* check to be E-000-comparable; this table's own published numbers
    // already are whole-map and stay that way here).
    let stand = find_standable(world2.collision(), false);
    assert!(stand.len() > 10, "expected plenty of standable tiles on Copy Love Box");

    let clock = WallClock::new();
    let n_decisions = games_env("DDAI_TIMING_DECISIONS", 300);
    for &tee_count in &[2usize, 6usize] {
        for (label, budget_ms) in [
            ("fixed-iteration (decide(), budgetMs=0)", None),
            ("decide_production 1ms", Some(1.0)),
            ("decide_production 2ms", Some(2.0)),
            ("decide_production 4ms", Some(4.0)),
            ("decide_production 8ms", Some(8.0)),
        ] {
            let mut spawn_rng = Rng::new(42);
            let mut world = build_world(&map, &stand, &mut spawn_rng, tee_count - 2, 1);
            let mut planner: Planner<PhysicsWorld> = Planner::new(fixed_iteration_cfg());
            planner.reset();
            let mut prev = empty_input();
            let mut script_rng = Rng::new(99);
            let mut enemy_prev = empty_input();

            let mut times_ms = Vec::with_capacity(n_decisions);
            let mut candidates = Vec::with_capacity(n_decisions);
            for _ in 0..n_decisions {
                let enemy_input = scripted_action(&world, 1, 0, &enemy_prev, &mut script_rng);
                let t0 = Instant::now();
                let out = decide_with(&mut planner, &mut world, 0, 1, prev, enemy_input, &clock, budget_ms);
                times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                candidates.push(planner.last_info.candidates);
                world.set_input(0, out);
                world.set_input(1, enemy_input);
                world.step();
                prev = out;
                enemy_prev = enemy_input;
                if world.get_tee(0).is_none_or(|t| !t.alive) || world.get_tee(1).is_none_or(|t| !t.alive) {
                    let mut r = Rng::new(43);
                    world = build_world(&map, &stand, &mut r, tee_count - 2, 2);
                }
            }
            times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
            let mean_candidates: f64 = candidates.iter().map(|&c| f64::from(c)).sum::<f64>() / candidates.len() as f64;
            eprintln!(
                "tees={tee_count} {label}: mean={:.3}ms p50={:.3}ms p90={:.3}ms p99={:.3}ms max={:.3}ms mean_candidates={mean_candidates:.1} (n={})",
                mean,
                percentile(&times_ms, 0.5),
                percentile(&times_ms, 0.9),
                percentile(&times_ms, 0.99),
                times_ms.last().copied().unwrap_or(0.0),
                times_ms.len(),
            );
        }
    }
}

/// W:L:D:timeout tally for one condition (review round 1, F9: "report W:L:D:timeout separately",
/// not folded into a single win-rate) plus the Wilson 95% CI on the win rate (draws counted as
/// half a win, `wins_weighted`, matching the CI's own convention -- `wins`/`losses`/`draws`/
/// `timeouts` themselves are exact game counts, never weighted).
struct Tally {
    wins: u32,
    losses: u32,
    draws: u32,
    timeouts: u32,
}
impl Tally {
    fn new() -> Self {
        Tally {
            wins: 0,
            losses: 0,
            draws: 0,
            timeouts: 0,
        }
    }
    fn record(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::SelfWins => self.wins += 1,
            Outcome::EnemyWins => self.losses += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::Timeout => self.timeouts += 1,
        }
    }
    fn n(&self) -> u32 {
        self.wins + self.losses + self.draws + self.timeouts
    }
    fn report(&self, label: &str) {
        let n = f64::from(self.n());
        let wins_weighted = f64::from(self.wins) + 0.5 * f64::from(self.draws);
        let (lo, hi) = wilson_ci(wins_weighted, n);
        eprintln!(
            "{label}: W={} L={} D={} timeout={} (n={}) win-rate={:.1}% (95% CI [{:.1}%, {:.1}%])",
            self.wins,
            self.losses,
            self.draws,
            self.timeouts,
            self.n(),
            100.0 * wins_weighted / n,
            100.0 * lo,
            100.0 * hi,
        );
    }
}

#[test]
#[ignore = "needs the local Copy Love Box map and takes a while -- see this file's doc comment"]
fn quality_vs_budget_report() {
    let map = load_clb();
    let world2 = PhysicsWorld::new(map.clone(), 1);
    // Review round 2, F15: E-000 left-hall arena (WB boxes L1 {79..104,67..79} union L2
    // {78..104,79..87}), spawn distance 3-12 tiles -- comparable to E-000's own numbers, unlike
    // round 1's whole-map spawn pairs.
    let stand = find_standable(world2.collision(), true);
    let clock = WallClock::new();

    let games_per_condition = games_env("DDAI_ARENA_GAMES", 60);
    eprintln!(
        "quality_vs_budget_report: {games_per_condition} games/condition (acceptance criterion 5 asks for >= 200; see BUILD REPORT for the actual n used and why)"
    );

    let skip_fixed = std::env::var("DDAI_ARENA_SKIP_FIXED").is_ok_and(|v| v == "1");
    for (label, self_budget) in [
        ("fixed-iteration", None),
        ("1ms", Some(1.0)),
        ("2ms", Some(2.0)),
        ("4ms", Some(4.0)),
        ("8ms", Some(8.0)),
    ] {
        if skip_fixed && self_budget.is_none() {
            eprintln!(
                "{label}: skipped (DDAI_ARENA_SKIP_FIXED=1 -- see BUILD REPORT for this condition's numbers from a separate run)"
            );
            continue;
        }
        let mut vs_scripted = Tally::new();
        for g in 0..games_per_condition {
            let outcome = play_game(
                &map,
                &stand,
                fixed_iteration_cfg(),
                self_budget,
                None,
                None,
                1000 + g as u64,
                &clock,
            );
            vs_scripted.record(outcome);
        }
        vs_scripted.report(&format!("{label} vs scripted"));

        if std::env::var("DDAI_ARENA_SKIP_SELF").is_ok_and(|v| v == "1") {
            continue;
        }
        let mut vs_fixed = Tally::new();
        for g in 0..games_per_condition {
            let outcome = play_game(
                &map,
                &stand,
                fixed_iteration_cfg(),
                self_budget,
                Some(fixed_iteration_cfg()),
                None,
                2000 + g as u64,
                &clock,
            );
            vs_fixed.record(outcome);
        }
        vs_fixed.report(&format!("{label} vs fixed-iteration self"));
    }
}
