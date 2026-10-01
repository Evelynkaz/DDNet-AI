//! Candidate scoring for the hybrid search: an [`Engine`] evaluates batches of plans on worker
//! worlds, either on the calling thread alone or on a **persistent worker pool** (task 3.5
//! criterion 4).
//!
//! **What a worker is.** A [`Worker`] owns a private planning world (`PhysicsWorld`) and a private
//! `Planner` whose only job is `evaluate_impl`. Before scoring, a worker copies the decision state
//! out of the shared [`Ctx`] with `restore_state` (`World::restore_from`/`clone_from`: it reuses
//! its buffers, no allocation once warm) and reloads only when the decision changes.
//!
//! **Why the result cannot depend on the thread count.** A candidate's score is a pure function of
//! the shared context and the candidate: every worker starts from the same snapshot, the opponent
//! RNG is re-seeded per rollout, and the one piece of planner state that could leak between
//! candidates (the `noThaw` escape cache) is made per-evaluation (`deterministic_thaw`). Workers
//! pull jobs from an atomic counter, but each result lands in its job's own slot and the caller
//! merges by candidate index, so in fixed-work mode the outcome is bit-identical for any number of
//! threads. In deadline mode the *set* of finished candidates depends on timing (that is the
//! point of a deadline); each finished score is still exact.
//!
//! **Synchronisation.** One mutex-guarded epoch counter and two condvars start and finish a batch;
//! the context and the job list are read-locked while workers run and written only between
//! batches. The caller works too (it is worker 0), so `workers = N` means N threads scoring.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;

use crate::clock::{Clock, WallClock};
use crate::fields::HazardField;
use crate::hybrid::config::HybridConfig;
use crate::hybrid::threat::ThreatSet;
use crate::physics_adapter::{PhysicsSavedState, PhysicsWorld};
use crate::plan_world::PlanWorld;
use crate::planner::{PlanStep, Planner};
use crate::shield::Bounded;
use crate::types::PlayerInput;
use crate::vmath::Vec2;

/// Everything a worker needs about the current decision. Written by the deciding thread between
/// batches.
pub struct Ctx {
    /// Bumped on every write: workers reload their world when it changes.
    pub generation: u64,
    /// The decision state (lag already rolled in, every tee's held input set).
    /// Boxed: a `World<f32>` snapshot is ~100 KB and worker/rayon threads have small stacks.
    pub saved: Box<PhysicsSavedState>,
    pub self_id: i32,
    pub victim_id: i32,
    pub prev: PlayerInput,
    pub victim_input: PlayerInput,
    pub opp_seed: u32,
    pub field: Arc<HazardField>,
    pub unfreeze: Arc<HazardField>,
    pub frozen_bystanders: Vec<Vec2>,
    pub frozen_bystander_vels: Vec<Vec2>,
    /// Tees the rope and the hammer must spare (the live bot's friends, ignored, AFK; task 3.5b):
    /// positions and velocities, empty in the arena.
    pub spares: Vec<Vec2>,
    pub spare_vels: Vec<Vec2>,
    /// An intermediate point to head for (the live bot's navigation), `None` = the target itself.
    pub travel_goal: Option<Vec2>,
    /// Threat tees besides the victim and the inputs they hold (`None` = the 1v1 model).
    pub threats: Option<ThreatSet>,
    /// Multiplier of the self-freeze weight: `1` normally, the planner's `escapeBias` during the
    /// extension (plans that get us frozen are penalised harder).
    pub self_freeze_bias: f64,
}

/// One rollout to run: plan `plan` of the batch under model combination `combo` (bit 0: the
/// victim reacts, bit `i + 1`: threat `i` reacts).
#[derive(Debug, Clone, Copy)]
pub struct Job {
    pub plan: usize,
    pub combo: u32,
    /// Steps of the plan to roll out (`0` = all of them). A short horizon is the early-pruning
    /// pre-score (task 3.5b): the same rollout, cut after the first steps.
    pub horizon: u32,
}

/// A batch: `plans` flat with `stride` steps each, and the jobs over them.
#[derive(Debug, Default)]
pub struct Batch {
    pub stride: usize,
    pub steps: Vec<PlanStep>,
    pub jobs: Vec<Job>,
}

impl Batch {
    pub fn clear(&mut self, stride: usize) {
        self.stride = stride;
        self.steps.clear();
        self.jobs.clear();
    }

    /// Adds a plan (exactly `stride` steps) and returns its index.
    pub fn push_plan(&mut self, plan: &[PlanStep]) -> usize {
        debug_assert_eq!(plan.len(), self.stride);
        self.steps.extend_from_slice(plan);
        self.steps.len() / self.stride - 1
    }

    pub fn plan(&self, i: usize) -> &[PlanStep] {
        &self.steps[i * self.stride..(i + 1) * self.stride]
    }

    pub fn push_job(&mut self, plan: usize, combo: u32) {
        self.jobs.push(Job {
            plan,
            combo,
            horizon: 0,
        });
    }

    /// A job over the first `horizon` steps of the plan only.
    pub fn push_job_short(&mut self, plan: usize, combo: u32, horizon: u32) {
        self.jobs.push(Job { plan, combo, horizon });
    }
}

/// What one finished rollout reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalResult {
    pub score: f64,
    /// Ticks the tee was frozen or dead / the victim was frozen or dead in the rollout.
    pub self_out: i32,
    pub enemy_out: i32,
    /// The victim ended frozen and resting in a hazard (or dead).
    pub enemy_sealed: bool,
}

/// The outcome of one job: a result unless the deadline cut the rollout, and the physics ticks
/// simulated either way (work counter).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalOut {
    pub res: Option<EvalResult>,
    pub ticks: u32,
}

const NO_OUT: EvalOut = EvalOut { res: None, ticks: 0 };

/// A private planner + world pair that scores candidates.
pub struct Worker {
    // Boxed: the planner is ~300 KB and the world ~100 KB, and threads have small stacks.
    planner: Box<Planner<PhysicsWorld>>,
    world: Box<PhysicsWorld>,
    loaded: u64,
    base_bias: f64,
    /// Snapshot buffer of the shield's escape phase (`plan_escape`).
    escape_slot: Box<Option<PhysicsSavedState>>,
}

impl Worker {
    fn new(cfg: &HybridConfig, world: PhysicsWorld) -> Worker {
        let mut planner = Box::new(Planner::new(cfg.planner));
        planner.deterministic_thaw = true;
        planner.track_rollout = true;
        Worker {
            base_bias: cfg.planner.self_freeze_bias,
            planner,
            world: Box::new(world),
            loaded: 0,
            escape_slot: Box::new(None),
        }
    }

    fn load(&mut self, ctx: &Ctx) {
        self.world.restore_state(&ctx.saved);
        match &mut self.planner.saved {
            Some(s) => s.assign_from(&ctx.saved),
            None => self.planner.saved = Some((*ctx.saved).clone()),
        }
        self.planner.opp_seed = ctx.opp_seed;
        let (bp, bv) = self.planner.frozen_bystanders_mut();
        bp.clone_from(&ctx.frozen_bystanders);
        bv.clone_from(&ctx.frozen_bystander_vels);
        let (sp, sv) = self.planner.spares_mut();
        sp.clone_from(&ctx.spares);
        sv.clone_from(&ctx.spare_vels);
        self.planner.set_travel_goal(ctx.travel_goal);
        self.planner.threats.clone_from(&ctx.threats);
        self.planner.cfg_mut().self_freeze_bias = self.base_bias * ctx.self_freeze_bias;
        self.loaded = ctx.generation;
    }

    /// Forgets the loaded snapshot: the next rollout reloads the world and the planner state. After a
    /// panic inside a rollout the world is left mid-rollout, and `load` would otherwise only run
    /// when the generation changes (review round 1, N3).
    fn invalidate(&mut self) {
        self.loaded = u64::MAX;
    }

    /// The hybrid shield's first escape (task 3.5b): rolls `plan` out exactly from the decision state
    /// under the model combination `combo` and asks whether we are safe: never frozen or dead on the
    /// way, and -- from where the plan ends -- some escape (the ordinary ones, then `extras`) settles.
    /// Returns the answer and the physics ticks it simulated beyond the rollout's own (none counted
    /// twice: the caller adds `ticks` to the work meter). The worker's world is restored afterwards.
    #[allow(clippy::too_many_arguments)]
    fn plan_escape(
        &mut self,
        ctx: &Ctx,
        plan: &[PlanStep],
        combo: u32,
        chosen: &PlayerInput,
        others: &HashMap<i32, PlayerInput>,
        deadline: Option<(&dyn Clock, f64)>,
        extras: &[PlayerInput],
    ) -> (Bounded<bool>, u32) {
        if self.loaded != ctx.generation {
            self.load(ctx);
        }
        self.planner.react_this_pass = combo & 1 == 1;
        if let Some(t) = &mut self.planner.threats {
            t.react_mask = combo >> 1;
        }
        let before = self.planner.eval_ticks;
        self.planner.keep_final = true;
        let res = self.planner.evaluate_impl(
            &mut self.world,
            ctx.self_id,
            ctx.victim_id,
            ctx.prev,
            plan,
            ctx.victim_input,
            &ctx.field,
            &ctx.unfreeze,
            None,
        );
        self.planner.keep_final = false;
        self.planner.react_this_pass = false;
        let ticks = (self.planner.eval_ticks - before) as u32;
        let verdict = if res.is_none() || self.planner.rollout_self_out > 0 {
            Bounded::Done(false)
        } else {
            // From where the plan ends the tee first keeps doing what the plan's last step did (a
            // hook held, a swing), and only then tries the standard escapes.
            let lead = [self.planner.last_input];
            crate::shield::escape_tail(
                &mut *self.world,
                ctx.self_id,
                chosen,
                others,
                deadline,
                &mut self.escape_slot,
                &lead,
                extras,
            )
        };
        self.world.restore_state(&ctx.saved);
        (verdict, ticks)
    }

    fn eval(&mut self, ctx: &Ctx, plan: &[PlanStep], combo: u32, deadline: Option<(&dyn Clock, f64)>) -> EvalOut {
        if self.loaded != ctx.generation {
            self.load(ctx);
        }
        self.planner.react_this_pass = combo & 1 == 1;
        if let Some(t) = &mut self.planner.threats {
            t.react_mask = combo >> 1;
        }
        let before = self.planner.eval_ticks;
        let res = self.planner.evaluate_impl(
            &mut self.world,
            ctx.self_id,
            ctx.victim_id,
            ctx.prev,
            plan,
            ctx.victim_input,
            &ctx.field,
            &ctx.unfreeze,
            deadline,
        );
        self.planner.react_this_pass = false;
        let ticks = (self.planner.eval_ticks - before) as u32;
        EvalOut {
            res: res.map(|score| EvalResult {
                score,
                self_out: self.planner.rollout_self_out,
                enemy_out: self.planner.rollout_enemy_out,
                enemy_sealed: self.planner.rollout_enemy_sealed,
            }),
            ticks,
        }
    }
}

struct Sync {
    epoch: u64,
    finished: usize,
    shutdown: bool,
}

struct Shared {
    ctx: RwLock<Box<Ctx>>,
    batch: RwLock<Batch>,
    results: Mutex<Vec<EvalOut>>,
    next: AtomicUsize,
    /// Deadline as `f64` bits on the pool's clock; `INFINITY` = none.
    deadline_bits: AtomicU64,
    sync: Mutex<Sync>,
    go: Condvar,
    done: Condvar,
    clock: Arc<WallClock>,
    helpers: usize,
    /// The work clock's meter (single-worker deadline runs): every finished rollout advances it at
    /// once, so the deadline can cut a batch between its rollouts, not only after the batch.
    meter: Mutex<Option<Arc<crate::hybrid::work::WorkMeter>>>,
}

impl Shared {
    fn deadline(&self) -> f64 {
        f64::from_bits(self.deadline_bits.load(Ordering::Relaxed))
    }

    /// Pulls jobs until none are left; every claimed job's slot is written exactly once.
    fn run(&self, worker: &mut Worker, clock: &dyn Clock) {
        let ctx = self.ctx.read().expect("ctx lock");
        let batch = self.batch.read().expect("batch lock");
        let deadline_ms = self.deadline();
        let deadline = deadline_ms.is_finite().then_some((clock, deadline_ms));
        let meter = self.meter.lock().expect("meter lock").clone();
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(job) = batch.jobs.get(i) else { break };
            // Do not start a rollout the deadline has already passed.
            let out = match deadline {
                Some((c, d)) if c.now_ms() >= d => NO_OUT,
                _ => {
                    let plan = batch.plan(job.plan);
                    let plan = if job.horizon > 0 && (job.horizon as usize) < plan.len() {
                        &plan[..job.horizon as usize]
                    } else {
                        plan
                    };
                    worker.eval(&ctx, plan, job.combo, deadline)
                }
            };
            if let Some(m) = &meter {
                m.add(u64::from(out.ticks));
            }
            self.results.lock().expect("results lock")[i] = out;
        }
    }
}

/// Scores batches of plans (see the module docs).
pub struct Engine {
    shared: Arc<Shared>,
    worker0: Worker,
    threads: Vec<JoinHandle<()>>,
    epoch: u64,
}

impl Engine {
    /// `template`: a world on the map (workers clone it once). `workers >= 1`; with `1` no thread
    /// is spawned.
    pub fn new(cfg: &HybridConfig, template: &PhysicsWorld, ctx: Box<Ctx>, clock: Arc<WallClock>) -> Engine {
        let workers = cfg.workers.max(1);
        let shared = Arc::new(Shared {
            ctx: RwLock::new(ctx),
            batch: RwLock::new(Batch::default()),
            results: Mutex::new(Vec::new()),
            next: AtomicUsize::new(0),
            deadline_bits: AtomicU64::new(f64::INFINITY.to_bits()),
            sync: Mutex::new(Sync {
                epoch: 0,
                finished: 0,
                shutdown: false,
            }),
            go: Condvar::new(),
            done: Condvar::new(),
            clock,
            helpers: workers - 1,
            meter: Mutex::new(None),
        });
        let worker0 = Worker::new(cfg, clone_world(template));
        let mut threads = Vec::new();
        for i in 1..workers {
            let sh = Arc::clone(&shared);
            let mut w = Worker::new(cfg, clone_world(template));
            threads.push(
                std::thread::Builder::new()
                    .name(format!("hybrid-eval-{i}"))
                    .spawn(move || helper_loop(&sh, &mut w))
                    .expect("spawn hybrid worker"),
            );
        }
        Engine {
            shared,
            worker0,
            threads,
            epoch: 0,
        }
    }

    /// Makes every finished rollout advance the work clock's meter (see `Shared::meter`).
    pub fn set_meter(&mut self, meter: Option<Arc<crate::hybrid::work::WorkMeter>>) {
        *self.shared.meter.lock().expect("meter lock") = meter;
    }

    /// See [`Worker::plan_escape`]; runs on the deciding thread's own worker.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_escape(
        &mut self,
        plan: &[PlanStep],
        combo: u32,
        chosen: &PlayerInput,
        others: &HashMap<i32, PlayerInput>,
        deadline: Option<(&dyn Clock, f64)>,
        extras: &[PlayerInput],
    ) -> (Bounded<bool>, u32) {
        let ctx = self.shared.ctx.read().expect("ctx lock");
        let out = self
            .worker0
            .plan_escape(&ctx, plan, combo, chosen, others, deadline, extras);
        if let Some(m) = self.shared.meter.lock().expect("meter lock").as_ref() {
            m.add(u64::from(out.1));
        }
        out
    }

    pub fn workers(&self) -> usize {
        self.threads.len() + 1
    }

    /// Runs `f` on the shared context (between batches) and bumps its generation.
    pub fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Ctx) -> R) -> R {
        let mut ctx = self.shared.ctx.write().expect("ctx lock");
        let r = f(&mut ctx);
        ctx.generation += 1;
        r
    }

    /// Scores every job of `batch` into `out` (index-aligned with `batch.jobs`). `clock` and
    /// `deadline_ms` bound the work: a rollout that has not finished by the deadline comes back
    /// with `res == None`. `clock` must be the pool's wall clock when there are helper threads.
    pub fn evaluate(&mut self, batch: &mut Batch, clock: &dyn Clock, deadline_ms: Option<f64>, out: &mut Vec<EvalOut>) {
        let n = batch.jobs.len();
        {
            let mut res = self.shared.results.lock().expect("results lock");
            res.clear();
            res.resize(n, NO_OUT);
        }
        std::mem::swap(&mut *self.shared.batch.write().expect("batch lock"), batch);
        self.shared.next.store(0, Ordering::Relaxed);
        self.shared
            .deadline_bits
            .store(deadline_ms.unwrap_or(f64::INFINITY).to_bits(), Ordering::Relaxed);
        if self.shared.helpers > 0 {
            let mut g = self.shared.sync.lock().expect("sync lock");
            self.epoch += 1;
            g.epoch = self.epoch;
            g.finished = 0;
            self.shared.go.notify_all();
        }
        self.shared.run(&mut self.worker0, clock);
        if self.shared.helpers > 0 {
            let mut g = self.shared.sync.lock().expect("sync lock");
            while g.finished < self.shared.helpers {
                g = self.shared.done.wait(g).expect("done wait");
            }
        }
        std::mem::swap(&mut *self.shared.batch.write().expect("batch lock"), batch);
        let res = self.shared.results.lock().expect("results lock");
        out.clear();
        out.extend_from_slice(&res);
    }

    pub fn clock(&self) -> &Arc<WallClock> {
        &self.shared.clock
    }
}

fn clone_world(template: &PhysicsWorld) -> PhysicsWorld {
    PhysicsWorld::from_world(template.inner().clone(), template.map().clone())
}

fn helper_loop(shared: &Shared, worker: &mut Worker) {
    let mut seen = 0u64;
    loop {
        {
            let mut g = shared.sync.lock().expect("sync lock");
            while g.epoch == seen && !g.shutdown {
                g = shared.go.wait(g).expect("go wait");
            }
            if g.shutdown {
                return;
            }
            seen = g.epoch;
        }
        // A panic in a rollout must not leave the deciding thread waiting forever: the jobs this
        // worker had claimed keep their default (cut) result and the batch still completes.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shared.run(worker, &*shared.clock))).is_err() {
            worker.invalidate();
        }
        let mut g = shared.sync.lock().expect("sync lock");
        g.finished += 1;
        if g.finished == shared.helpers {
            shared.done.notify_one();
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(mut g) = self.shared.sync.lock() {
            g.shutdown = true;
        }
        self.shared.go.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use ddai_physics::map::{MapData, TILE_SOLID, Tile};
    use std::time::Instant;

    fn hall() -> Arc<MapData> {
        let (w, h) = (60usize, 30usize);
        let mut game = vec![Tile::default(); w * h];
        for y in 0..h {
            for x in 0..w {
                if y >= 20 || x == 0 || x == w - 1 || y == 0 {
                    game[y * w + x] = Tile {
                        index: TILE_SOLID,
                        ..Tile::default()
                    };
                }
            }
        }
        Arc::new(MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        })
    }

    /// Where a rollout's time goes: the same 27 ticks of the same tees, raw physics against
    /// `evaluate_impl` (physics + scoring + hook logic). Run with
    /// `cargo test -p ddai-planner --release --lib rollout_cost_split -- --ignored --nocapture`.
    #[test]
    #[ignore = "micro-benchmark"]
    fn rollout_cost_split() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                for tees in [2usize, 4, 6] {
                    let mut w = PhysicsWorld::new(hall(), 1);
                    for i in 0..tees {
                        w.add_tee(
                            i as i32,
                            Vec2 {
                                x: (20.0 + 2.5 * i as f64) * 32.0,
                                y: 19.0 * 32.0 - 20.0,
                            },
                        );
                    }
                    let mut pl = Box::new(Planner::<PhysicsWorld>::new(crate::hybrid::config::hybrid_planner_preset()));
                    pl.track_rollout = true;
                    pl.deterministic_thaw = true;
                    let (field, unfreeze) = pl.hazard_fields(w.collision());
                    pl.saved = Some(w.save_state());
                    let saved = w.save_state();
                    let plan: Vec<PlanStep> = (0..9)
                        .map(|s| PlanStep {
                            dir: if s % 3 == 0 { 1 } else { 0 },
                            jump: i32::from(s == 2),
                            hook: 0,
                            fire: 0,
                            aim: 0.0,
                        })
                        .collect();
                    let n = 3000;
                    let t0 = Instant::now();
                    for _ in 0..n {
                        let _ = pl.evaluate_impl(
                            &mut w,
                            0,
                            1,
                            crate::types::empty_input(),
                            &plan,
                            crate::types::empty_input(),
                            &field,
                            &unfreeze,
                            None,
                        );
                    }
                    let full = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
                    let mut ev = Vec::new();
                    let t0 = Instant::now();
                    for _ in 0..n {
                        for k in 0..27 {
                            let mut i = crate::types::empty_input();
                            i.direction = i32::from(k % 9 < 3);
                            w.set_input(0, i);
                            w.step_into(&mut ev);
                        }
                        w.restore_state(&saved);
                    }
                    let phys = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n);
                    println!(
                        "{tees} tees: rollout {full:.1} us ({:.2} us/tick), raw physics {phys:.1} us ({:.2} us/tick) -> physics share {:.0}%",
                        full / 27.0,
                        phys / 27.0,
                        100.0 * phys / full
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
