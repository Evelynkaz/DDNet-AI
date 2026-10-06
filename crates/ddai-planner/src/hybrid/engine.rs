//! Candidate scoring for the hybrid search: an [`Engine`] evaluates batches of plans on worker
//! worlds, either on the calling thread alone or on a **persistent worker pool** (task 3.5
//! criterion 4; reworked in task 3.7a for the live bot).
//!
//! **What a worker is.** A [`Worker`] owns a private planning world (`PhysicsWorld`) and a private
//! `Planner` whose only job is `evaluate_impl`. Before scoring, a worker copies the decision state
//! out of the shared [`Ctx`] with `restore_state` (`World::restore_from`/`clone_from`: it reuses
//! its buffers, no allocation once warm) and reloads only when the decision changes. It also keeps
//! its own copy of the small per-decision values (ids, inputs, the hazard-field `Arc`s), so a
//! rollout holds **no lock**.
//!
//! **Why the result cannot depend on the thread count.** A candidate's score is a pure function of
//! the shared context and the candidate: every worker starts from the same snapshot, the opponent
//! RNG is re-seeded per rollout, and the one piece of planner state that could leak between
//! candidates (the `noThaw` escape cache) is made per-evaluation (`deterministic_thaw`). Workers
//! claim jobs in index order, but each result lands in its job's own slot and the caller merges by
//! candidate index, so in fixed-work mode the outcome is bit-identical for any number of threads.
//!
//! **Three ways to run a batch** (see [`Engine::evaluate`]):
//! * *one thread* (`workers = 1`, or a batch of one job): the caller scores the jobs in order;
//! * *the pool* (wall clock): the jobs are claimed from a shared counter by the helpers and the
//!   caller; a rollout cut by the deadline comes back as `None`. A job a helper has held for more than
//!   `STEAL_AFTER` (a stalled host thread) is **recomputed by the caller** (the score is a pure
//!   function, so the duplicate is identical), so a stall costs neither idle time nor a candidate;
//!   only past the deadline (plus a short grace) is a straggler left behind, its job cut and its late
//!   result thrown away (batches are numbered). The deciding thread holds no lock while it waits and
//!   a helper holds none while it loads or scores; what remains are the critical sections of the
//!   pool mutex and of the context mutex (a few microseconds of bookkeeping, the latter just an `Arc`
//!   clone), where a preempted helper could delay the caller for as long as the host keeps it off a core;
//! * *speculation* (work clock, task 3.7a): the work clock is a counter of finished work that must be
//!   advanced one rollout at a time, in the order a single thread would, or the search would depend on
//!   the thread count. So [`Engine::evaluate`] stays serial on the work clock, and the helpers only
//!   [`Engine::prefetch`] the rollouts the serial search is *about to* ask for into a cache keyed by
//!   the exact job (decision generation, plan bits, model combination, horizon). A lookup that
//!   misses just computes the job itself, so a wrong guess costs time, never a different answer:
//!   the work-clock decision is bit-identical for any number of workers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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

/// A parked helper first spins this long for the next batch (the chunks of a decision follow each
/// other within microseconds; waking a parked thread costs tens of them).
const SPIN: Duration = Duration::from_micros(120);

/// The caller recomputes a job a helper has held for this long (a rollout takes ~100 us; a helper that has not
/// finished in three times that is stalled).
const STEAL_AFTER: Duration = Duration::from_micros(300);

/// A batch with a deadline: the caller waits this long past it for helpers still unwinding a cut
/// rollout (a rollout checks the deadline every plan step, ~10 us), then leaves them behind (ms).
const DEADLINE_GRACE_MS: f64 = 0.15;

/// Stack of a helper thread: a rollout keeps world-sized values on it, and the 2 MiB default is close to what one needs.
const HELPER_STACK_BYTES: usize = 16 << 20;

/// `announce` value that tells spinning helpers to quit.
const SHUTDOWN: u64 = u64::MAX;

/// Everything a worker needs about the current decision. Written by the deciding thread between
/// batches.
#[derive(Clone)]
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
    /// Task 3.7b: the victim's predicted input per plan step (`HybridConfig::mirror_samples`); empty = it holds `victim_input`.
    pub victim_plan: Vec<PlayerInput>,
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

    /// The steps a job rolls out: its plan, cut to the horizon.
    fn job_steps(&self, job: &Job) -> &[PlanStep] {
        let plan = self.plan(job.plan);
        if job.horizon > 0 && (job.horizon as usize) < plan.len() {
            &plan[..job.horizon as usize]
        } else {
            plan
        }
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
    /// Task 3.9: extra work the tick counter does not see, in tee-ticks (the v2 hook gate's projections of a moving victim, one each).
    pub units: u32,
}

const NO_OUT: EvalOut = EvalOut {
    res: None,
    ticks: 0,
    units: 0,
};

/// A private planner + world pair that scores candidates.
pub struct Worker {
    // Boxed: the planner is ~300 KB and the world ~100 KB, and threads have small stacks.
    planner: Box<Planner<PhysicsWorld>>,
    world: Box<PhysicsWorld>,
    loaded: u64,
    base_bias: f64,
    /// Snapshot buffer of the shield's escape phase (`plan_escape`).
    escape_slot: Box<Option<PhysicsSavedState>>,
    /// The loaded decision's small values, copied out of the [`Ctx`] so a rollout needs no lock.
    self_id: i32,
    victim_id: i32,
    prev: PlayerInput,
    victim_input: PlayerInput,
    field: Arc<HazardField>,
    unfreeze: Arc<HazardField>,
}

impl Worker {
    fn new(cfg: &HybridConfig, world: PhysicsWorld, ctx: &Ctx) -> Worker {
        let mut planner = Box::new(Planner::new(cfg.planner));
        planner.deterministic_thaw = true;
        planner.track_rollout = true;
        planner.launch_memo = Some(Box::default());
        Worker {
            base_bias: cfg.planner.self_freeze_bias,
            planner,
            world: Box::new(world),
            loaded: 0,
            escape_slot: Box::new(None),
            self_id: ctx.self_id,
            victim_id: ctx.victim_id,
            prev: ctx.prev,
            victim_input: ctx.victim_input,
            field: Arc::clone(&ctx.field),
            unfreeze: Arc::clone(&ctx.unfreeze),
        }
    }

    fn load(&mut self, ctx: &Ctx) {
        // A new decision (and possibly a new map): nothing memoised before may be reused.
        if let Some(m) = &mut self.planner.launch_memo {
            m.new_epoch();
        }
        self.world.restore_state(&ctx.saved);
        // Task 3.9: `rope_ceiling_cost` needs the map's ceiling field (cached by the map; nothing when the cost is off).
        self.planner.prepare_ceiling(self.world.collision());
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
        self.planner.set_predicted(&ctx.victim_plan);
        self.planner.cfg_mut().self_freeze_bias = self.base_bias * ctx.self_freeze_bias;
        self.self_id = ctx.self_id;
        self.victim_id = ctx.victim_id;
        self.prev = ctx.prev;
        self.victim_input = ctx.victim_input;
        self.field = Arc::clone(&ctx.field);
        self.unfreeze = Arc::clone(&ctx.unfreeze);
        self.loaded = ctx.generation;
    }

    /// Loads the decision if the worker has not seen this generation of it.
    fn ensure_loaded(&mut self, ctx: &Ctx) {
        if self.loaded != ctx.generation {
            self.load(ctx);
        }
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
        self.ensure_loaded(ctx);
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

    /// One rollout of `plan` (already cut to its horizon) on the loaded decision.
    fn eval(&mut self, plan: &[PlanStep], combo: u32, deadline: Option<(&dyn Clock, f64)>) -> EvalOut {
        self.planner.react_this_pass = combo & 1 == 1;
        if let Some(t) = &mut self.planner.threats {
            t.react_mask = combo >> 1;
        }
        let before = self.planner.eval_ticks;
        let projections = self.planner.intercept_count();
        let res = self.planner.evaluate_impl(
            &mut self.world,
            self.self_id,
            self.victim_id,
            self.prev,
            plan,
            self.victim_input,
            &self.field,
            &self.unfreeze,
            deadline,
        );
        self.planner.react_this_pass = false;
        let ticks = (self.planner.eval_ticks - before) as u32;
        let units = (self.planner.intercept_count() - projections) as u32;
        EvalOut {
            res: res.map(|score| EvalResult {
                score,
                self_out: self.planner.rollout_self_out,
                enemy_out: self.planner.rollout_enemy_out,
                enemy_sealed: self.planner.rollout_enemy_sealed,
            }),
            ticks,
            units,
        }
    }
}

/// One job's slot in the pool's current batch.
#[derive(Clone, Copy)]
struct Slot {
    done: bool,
    out: EvalOut,
}

/// The pool's shared state: the numbered current batch and the claim/finish bookkeeping.
struct Pool {
    /// Numbers the batches; a helper's late result for an older batch is thrown away.
    epoch: u64,
    /// Jobs of the current batch may be claimed.
    open: bool,
    /// Deadline of the batch on the pool's clock; `INFINITY` = none.
    deadline_ms: f64,
    batch: Batch,
    n: usize,
    next: usize,
    finished: usize,
    slots: Vec<Slot>,
    shutdown: bool,
}

/// A job a thread has taken out of the pool, its plan copied into the thread's own buffer.
struct Claimed {
    idx: usize,
    combo: u32,
    deadline_ms: f64,
}

struct Shared {
    /// The current decision, published as an `Arc` under a tiny mutex: a helper clones the `Arc` (a few ns under the
    /// lock) and loads its world from it **without holding any lock**, so a helper the host preempts mid-load cannot
    /// block the deciding thread's next `with_ctx` (it writes a fresh copy instead, see [`Engine::with_ctx`]).
    ctx: Mutex<Arc<Ctx>>,
    pool: Mutex<Pool>,
    /// Helpers park here until a new batch (or shutdown).
    go: Condvar,
    /// The caller waits here for the batch to finish.
    done: Condvar,
    /// The newest batch number, for helpers that are still spinning; `SHUTDOWN` = quit.
    announce: AtomicU64,
    clock: Arc<WallClock>,
    helpers: usize,
    /// Test hooks: a helper sleeps this long before each job (a stalled host thread), and the next
    /// helper job panics.
    #[cfg(test)]
    test_stall_us: AtomicU64,
    /// How many jobs a helper has started stalling on.
    #[cfg(test)]
    test_stalled: AtomicU64,
    #[cfg(test)]
    test_panic_next: std::sync::atomic::AtomicBool,
}

impl Shared {
    /// Publishes `batch` as the pool's current batch (swapping it in) and returns its number.
    fn open_batch(&self, batch: &mut Batch, deadline_ms: Option<f64>) -> u64 {
        let n = batch.jobs.len();
        let mut p = self.pool.lock().expect("pool lock");
        p.epoch += 1;
        std::mem::swap(&mut p.batch, batch);
        p.n = n;
        p.next = 0;
        p.finished = 0;
        p.open = true;
        p.deadline_ms = deadline_ms.unwrap_or(f64::INFINITY);
        p.slots.clear();
        p.slots.resize(
            n,
            Slot {
                done: false,
                out: NO_OUT,
            },
        );
        p.epoch
    }

    /// Takes the next unclaimed job of batch `epoch`, copying its plan into `plan`.
    fn claim(&self, epoch: u64, plan: &mut Vec<PlanStep>) -> Option<Claimed> {
        let mut p = self.pool.lock().expect("pool lock");
        if p.epoch != epoch || !p.open || p.next >= p.n {
            return None;
        }
        let idx = p.next;
        p.next += 1;
        let job = p.batch.jobs[idx];
        plan.clear();
        plan.extend_from_slice(p.batch.job_steps(&job));
        Some(Claimed {
            idx,
            combo: job.combo,
            deadline_ms: p.deadline_ms,
        })
    }

    /// The caller found no job left to claim and a helper has held one too long: takes the lowest
    /// unfinished job to compute it a second time (a batch without a deadline must complete).
    fn steal(&self, epoch: u64, plan: &mut Vec<PlanStep>) -> Option<Claimed> {
        let p = self.pool.lock().expect("pool lock");
        if p.epoch != epoch || !p.open {
            return None;
        }
        let idx = (0..p.n).find(|&i| !p.slots[i].done)?;
        let job = p.batch.jobs[idx];
        plan.clear();
        plan.extend_from_slice(p.batch.job_steps(&job));
        Some(Claimed {
            idx,
            combo: job.combo,
            deadline_ms: p.deadline_ms,
        })
    }

    /// Files a finished job, unless its batch is over or another thread has filed it first.
    fn complete(&self, epoch: u64, idx: usize, out: EvalOut) {
        let mut p = self.pool.lock().expect("pool lock");
        if p.epoch != epoch || !p.open || p.slots[idx].done {
            return;
        }
        p.slots[idx] = Slot { done: true, out };
        p.finished += 1;
        if p.finished == p.n {
            self.done.notify_all();
        }
    }

    /// The current decision (a cheap clone of the published `Arc`).
    fn ctx(&self) -> Arc<Ctx> {
        Arc::clone(&self.ctx.lock().expect("ctx lock"))
    }

    /// Scores a claimed job on `worker`. The decision is (re)loaded from a clone of the published `Arc`; neither the
    /// load nor the rollout holds a lock.
    fn score(&self, worker: &mut Worker, plan: &[PlanStep], c: &Claimed, clock: &dyn Clock) -> EvalOut {
        {
            let ctx = self.ctx();
            worker.ensure_loaded(&ctx);
        }
        let deadline = c.deadline_ms.is_finite().then_some((clock, c.deadline_ms));
        // Do not start a rollout the deadline has already passed.
        if let Some((clk, d)) = deadline
            && clk.now_ms() >= d
        {
            return NO_OUT;
        }
        worker.eval(plan, c.combo, deadline)
    }

    /// Claims and scores jobs of batch `epoch` until none are left.
    fn work(&self, worker: &mut Worker, epoch: u64, plan: &mut Vec<PlanStep>, clock: &dyn Clock, guarded: bool) {
        while let Some(c) = self.claim(epoch, plan) {
            let out = if guarded {
                // A panic in a rollout must not leave the deciding thread waiting: the job keeps its
                // default (cut) result and the batch still completes.
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    #[cfg(test)]
                    self.test_hooks();
                    self.score(worker, plan, &c, clock)
                })) {
                    Ok(o) => o,
                    Err(_) => {
                        worker.invalidate();
                        NO_OUT
                    }
                }
            } else {
                self.score(worker, plan, &c, clock)
            };
            self.complete(epoch, c.idx, out);
        }
    }

    #[cfg(test)]
    fn test_hooks(&self) {
        let us = self.test_stall_us.load(Ordering::Relaxed);
        if us > 0 {
            self.test_stalled.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(Duration::from_micros(us));
        }
        assert!(
            !self.test_panic_next.swap(false, Ordering::Relaxed),
            "injected helper panic"
        );
    }

    /// A parked helper's wait for a batch newer than `seen`: spins briefly, then sleeps.
    /// `None` = shut down.
    fn wait_for_batch(&self, seen: u64) -> Option<u64> {
        let t0 = Instant::now();
        while t0.elapsed() < SPIN {
            let a = self.announce.load(Ordering::Acquire);
            if a == SHUTDOWN {
                return None;
            }
            if a != seen {
                return Some(a);
            }
            std::hint::spin_loop();
        }
        let mut p = self.pool.lock().expect("pool lock");
        loop {
            if p.shutdown {
                return None;
            }
            if p.epoch != seen {
                return Some(p.epoch);
            }
            p = self.go.wait(p).expect("go wait");
        }
    }
}

/// The rollouts the helpers have already computed for the serial work-clock search.
#[derive(Default)]
struct Prefetched {
    /// The decision generation the results belong to.
    generation: u64,
    batch: Batch,
    outs: Vec<EvalOut>,
    valid: bool,
}

fn same_steps(a: &[PlanStep], b: &[PlanStep]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.dir == y.dir
                && x.jump == y.jump
                && x.hook == y.hook
                && x.fire == y.fire
                && x.aim.to_bits() == y.aim.to_bits()
        })
}

impl Prefetched {
    /// The finished rollout of exactly this job in this decision, if there is one.
    fn find(&self, generation: u64, plan: &[PlanStep], combo: u32, horizon: u32) -> Option<EvalOut> {
        if !self.valid || self.generation != generation {
            return None;
        }
        self.batch
            .jobs
            .iter()
            .zip(&self.outs)
            .find(|(j, o)| {
                j.combo == combo && j.horizon == horizon && o.res.is_some() && same_steps(self.batch.plan(j.plan), plan)
            })
            .map(|(_, o)| *o)
    }
}

/// Scores batches of plans (see the module docs).
pub struct Engine {
    shared: Arc<Shared>,
    worker0: Worker,
    threads: Vec<JoinHandle<()>>,
    /// The deciding thread's buffer for the plan of the job it is scoring.
    plan_buf: Vec<PlanStep>,
    /// The work clock's meter: every finished rollout advances it at once, in job order, so the
    /// deadline can cut a batch between its rollouts, not only after the batch. Its presence makes
    /// the engine serial-and-speculative (see the module docs).
    meter: Option<Arc<crate::hybrid::work::WorkMeter>>,
    cache: Prefetched,
    /// Rollouts the helpers computed ahead / the serial search then took from the cache (all time).
    spec_computed: u64,
    spec_hits: u64,
}

impl Engine {
    /// `template`: a world on the map (workers clone it once). `workers >= 1`; with `1` no thread
    /// is spawned.
    pub fn new(cfg: &HybridConfig, template: &PhysicsWorld, ctx: Box<Ctx>, clock: Arc<WallClock>) -> Engine {
        let workers = cfg.workers.max(1);
        let worker0 = Worker::new(cfg, clone_world(template), &ctx);
        let helper_workers: Vec<Worker> = (1..workers)
            .map(|_| Worker::new(cfg, clone_world(template), &ctx))
            .collect();
        let shared = Arc::new(Shared {
            ctx: Mutex::new(Arc::from(ctx)),
            pool: Mutex::new(Pool {
                epoch: 0,
                open: false,
                deadline_ms: f64::INFINITY,
                batch: Batch::default(),
                n: 0,
                next: 0,
                finished: 0,
                slots: Vec::new(),
                shutdown: false,
            }),
            go: Condvar::new(),
            done: Condvar::new(),
            announce: AtomicU64::new(0),
            clock,
            helpers: workers - 1,
            #[cfg(test)]
            test_stall_us: AtomicU64::new(0),
            #[cfg(test)]
            test_stalled: AtomicU64::new(0),
            #[cfg(test)]
            test_panic_next: std::sync::atomic::AtomicBool::new(false),
        });
        let mut threads = Vec::new();
        for (i, mut w) in helper_workers.into_iter().enumerate() {
            let sh = Arc::clone(&shared);
            threads.push(
                std::thread::Builder::new()
                    .name(format!("hybrid-eval-{}", i + 1))
                    // A rollout works on a world-sized snapshot; the default 2 MiB leaves little margin (3.7b review F1).
                    .stack_size(HELPER_STACK_BYTES)
                    .spawn(move || helper_loop(&sh, &mut w))
                    .expect("spawn hybrid worker"),
            );
        }
        Engine {
            shared,
            worker0,
            threads,
            plan_buf: Vec::with_capacity(32),
            meter: None,
            cache: Prefetched::default(),
            spec_computed: 0,
            spec_hits: 0,
        }
    }

    /// Puts the engine on the work clock (`Some`) or back on wall time (`None`). On the work clock
    /// every finished rollout advances the meter and the helpers only speculate ([`Engine::prefetch`]).
    pub fn set_meter(&mut self, meter: Option<Arc<crate::hybrid::work::WorkMeter>>) {
        self.meter = meter;
        self.cache.valid = false;
    }

    /// Whether the engine is on the work clock with helper threads: the caller may
    /// [`prefetch`](Engine::prefetch) the rollouts it is about to ask for.
    pub fn speculating(&self) -> bool {
        self.meter.is_some() && self.shared.helpers > 0
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
        let ctx = self.shared.ctx();
        let out = self
            .worker0
            .plan_escape(&ctx, plan, combo, chosen, others, deadline, extras);
        if let Some(m) = &self.meter {
            m.add(u64::from(out.1));
        }
        out
    }

    pub fn workers(&self) -> usize {
        self.threads.len() + 1
    }

    /// Runs `f` on the shared context (between batches) and bumps its generation. In place when no helper holds a
    /// clone of the published context (always, in steady state: a helper drops its clone as soon as it has loaded);
    /// if a helper that the host has preempted is still reading it, `f` runs on a fresh copy that is published
    /// instead, so the deciding thread never waits for that helper (one allocation, only then).
    pub fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Ctx) -> R) -> R {
        let mut slot = self.shared.ctx.lock().expect("ctx lock");
        if let Some(ctx) = Arc::get_mut(&mut slot) {
            let r = f(ctx);
            ctx.generation += 1;
            return r;
        }
        let mut fresh = Ctx::clone(&slot);
        let r = f(&mut fresh);
        fresh.generation += 1;
        *slot = Arc::new(fresh);
        r
    }

    /// Task 3.7b: the context of the decision just made (kept by the loss diagnosis to score its plans later).
    pub fn debug_ctx(&self) -> Arc<Ctx> {
        self.shared.ctx()
    }

    /// Task 3.7b: scores `plans` in the decision `ctx` with the opponent's inputs given per plan step (`predicted`)
    /// instead of the model combinations: what the evaluator would have said had it known them. The worker is left
    /// loaded with that older decision, which the next decision's generation replaces.
    pub fn debug_eval_predicted(
        &mut self,
        ctx: &Ctx,
        plans: &[&[PlanStep]],
        predicted: &[PlayerInput],
    ) -> Vec<Option<EvalResult>> {
        self.worker0.load(ctx);
        self.worker0.planner.set_predicted(predicted);
        let out = plans.iter().map(|p| self.worker0.eval(p, 0, None).res).collect();
        self.worker0.planner.set_predicted(&[]);
        self.worker0.invalidate();
        out
    }

    /// Scores every job of `batch` into `out` (index-aligned with `batch.jobs`). `clock` and
    /// `deadline_ms` bound the work: a rollout that has not finished by the deadline comes back
    /// with `res == None`. `clock` must be the pool's wall clock when there are helper threads and
    /// the engine is not on the work clock.
    pub fn evaluate(&mut self, batch: &mut Batch, clock: &dyn Clock, deadline_ms: Option<f64>, out: &mut Vec<EvalOut>) {
        out.clear();
        if batch.jobs.is_empty() {
            return;
        }
        if self.shared.helpers == 0 || self.meter.is_some() || batch.jobs.len() == 1 {
            self.run_serial(batch, clock, deadline_ms, out);
        } else {
            self.run_pool(batch, clock, deadline_ms, out);
        }
    }

    /// The deciding thread scores the jobs one after the other. On the work clock the meter advances
    /// after each (a finished rollout from the prefetch cache counts exactly like a computed one).
    fn run_serial(&mut self, batch: &Batch, clock: &dyn Clock, deadline_ms: Option<f64>, out: &mut Vec<EvalOut>) {
        let ctx = self.shared.ctx();
        for job in &batch.jobs {
            let o = match deadline_ms {
                // Do not start a rollout the deadline has already passed.
                Some(d) if clock.now_ms() >= d => NO_OUT,
                _ => {
                    let plan = batch.job_steps(job);
                    let cached = if self.cache.valid {
                        self.cache
                            .find(ctx.generation, batch.plan(job.plan), job.combo, job.horizon)
                    } else {
                        None
                    };
                    self.spec_hits += u64::from(cached.is_some());
                    cached.unwrap_or_else(|| {
                        self.worker0.ensure_loaded(&ctx);
                        self.worker0.eval(plan, job.combo, deadline_ms.map(|d| (clock, d)))
                    })
                }
            };
            if let Some(m) = &self.meter {
                m.add(u64::from(o.ticks));
                m.add_units(u64::from(o.units));
            }
            out.push(o);
        }
    }

    /// The helpers and the caller share the jobs (see the module docs).
    fn run_pool(&mut self, batch: &mut Batch, clock: &dyn Clock, deadline_ms: Option<f64>, out: &mut Vec<EvalOut>) {
        let shared = Arc::clone(&self.shared);
        let n = batch.jobs.len();
        out.clear();
        let epoch = shared.open_batch(batch, deadline_ms);
        shared.announce.store(epoch, Ordering::Release);
        shared.go.notify_all();
        shared.work(&mut self.worker0, epoch, &mut self.plan_buf, clock, false);
        // Everything is claimed; wait for the helpers' last rollouts. A job a helper has held for longer than
        // `STEAL_AFTER` is recomputed by the caller (the score is a pure function, so the duplicate is identical),
        // with or without a deadline: a helper the host has stalled must neither leave the caller idle nor cost the
        // batch a candidate while there is still time. Only when the deadline has passed are the stragglers left
        // behind, their jobs cut.
        let mut p = shared.pool.lock().expect("pool lock");
        while p.finished < n {
            let remain_ms = deadline_ms.map(|d| d + DEADLINE_GRACE_MS - clock.now_ms());
            if remain_ms.is_some_and(|r| r.is_nan() || r <= 0.0) {
                break;
            }
            let wait = remain_ms.map_or(STEAL_AFTER, |r| Duration::from_secs_f64(r / 1000.0).min(STEAL_AFTER));
            let (g, timeout) = shared.done.wait_timeout(p, wait).expect("done wait");
            p = g;
            if timeout.timed_out() && p.finished < n && deadline_ms.is_none_or(|d| clock.now_ms() < d) {
                drop(p);
                if let Some(c) = shared.steal(epoch, &mut self.plan_buf) {
                    let o = shared.score(&mut self.worker0, &self.plan_buf, &c, clock);
                    shared.complete(epoch, c.idx, o);
                }
                p = shared.pool.lock().expect("pool lock");
            }
        }
        out.extend(p.slots.iter().map(|s| if s.done { s.out } else { NO_OUT }));
        p.open = false;
        std::mem::swap(&mut p.batch, batch);
    }

    /// Work clock only: computes `batch` on all workers without advancing the meter and keeps the
    /// results for [`Engine::evaluate`] to pick up (replacing an earlier prefetch). The caller
    /// passes the jobs the serial search will ask for next; the search's answer does not depend on
    /// whether they are right. A no-op unless [`Engine::speculating`].
    pub fn prefetch(&mut self, batch: &mut Batch, clock: &dyn Clock) {
        if !self.speculating() || batch.jobs.len() < 2 {
            return;
        }
        let generation = self.shared.ctx().generation;
        let mut outs = std::mem::take(&mut self.cache.outs);
        self.run_pool(batch, clock, None, &mut outs);
        self.cache.outs = outs;
        std::mem::swap(&mut self.cache.batch, batch);
        self.cache.generation = generation;
        self.cache.valid = true;
        self.spec_computed += self.cache.batch.jobs.len() as u64;
    }

    /// `(rollouts prefetched, rollouts the search took from the cache)` since the engine was built:
    /// the difference is speculation that was never asked for (work clock only).
    pub fn spec_stats(&self) -> (u64, u64) {
        (self.spec_computed, self.spec_hits)
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
    let mut plan: Vec<PlanStep> = Vec::with_capacity(32);
    while let Some(epoch) = shared.wait_for_batch(seen) {
        seen = epoch;
        shared.work(worker, epoch, &mut plan, &*shared.clock, true);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(mut p) = self.shared.pool.lock() {
            p.shutdown = true;
            p.open = false;
        }
        self.shared.announce.store(SHUTDOWN, Ordering::Release);
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

    pub(super) fn hall() -> Arc<MapData> {
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

/// The pool's guarantees (task 3.7a): the same answers as one thread, no waiting on a stalled
/// helper, a late result never lands in a newer batch, a panicking job costs one job, and scoring
/// on a helper thread allocates nothing.
#[cfg(test)]
mod pool_tests {
    use super::bench::hall;
    use super::*;
    use crate::fields::{hazard_field, unfreeze_field};
    use crate::hybrid::work::WorkMeter;

    fn scene() -> PhysicsWorld {
        let mut w = PhysicsWorld::new(hall(), 1);
        for i in 0..4 {
            w.add_tee(
                i,
                Vec2 {
                    x: (20.0 + 2.5 * f64::from(i)) * 32.0,
                    y: 19.0 * 32.0 - 20.0,
                },
            );
        }
        w
    }

    fn ctx_of(pw: &PhysicsWorld) -> Box<Ctx> {
        Box::new(Ctx {
            generation: 1,
            saved: Box::new(pw.save_state()),
            self_id: 0,
            victim_id: 1,
            prev: crate::types::empty_input(),
            victim_input: crate::types::empty_input(),
            victim_plan: vec![],
            opp_seed: 7,
            field: Arc::new(hazard_field(pw.collision())),
            unfreeze: Arc::new(unfreeze_field(pw.collision())),
            frozen_bystanders: vec![],
            frozen_bystander_vels: vec![],
            spares: vec![],
            spare_vels: vec![],
            travel_goal: None,
            threats: Some(ThreatSet {
                ids: vec![2, 3],
                inputs: vec![crate::types::empty_input(), crate::types::empty_input()],
                react_mask: 0,
                weight: 1.0,
                hook_targets: true,
            }),
            self_freeze_bias: 1.0,
        })
    }

    fn engine(workers: usize) -> Engine {
        let pw = scene();
        let cfg = HybridConfig {
            workers,
            ..HybridConfig::fixed()
        };
        Engine::new(&cfg, &pw, ctx_of(&pw), Arc::new(WallClock::new()))
    }

    /// `n` distinct plans, each under a model combination and some with a short horizon.
    fn fill(b: &mut Batch, n: usize, salt: usize) {
        b.clear(9);
        for k in 0..n {
            let j = k + salt;
            let plan: Vec<PlanStep> = (0..9)
                .map(|s| PlanStep {
                    dir: ((j + s) % 3) as i32 - 1,
                    jump: i32::from(s == 1 && j.is_multiple_of(4)),
                    hook: i32::from(j.is_multiple_of(2) && s < 6),
                    fire: i32::from(s > 5 && j.is_multiple_of(5)),
                    aim: 0.07 * (j as f64 - 10.0),
                })
                .collect();
            let pi = b.push_plan(&plan);
            if k % 7 == 3 {
                b.push_job_short(pi, (j % 8) as u32, 3);
            } else {
                b.push_job(pi, (j % 8) as u32);
            }
        }
    }

    fn reference(n: usize, salt: usize) -> Vec<EvalOut> {
        let mut e = engine(1);
        let (mut b, mut out) = (Batch::default(), Vec::new());
        fill(&mut b, n, salt);
        e.evaluate(&mut b, &WallClock::new(), None, &mut out);
        assert!(out.iter().all(|o| o.res.is_some()));
        out
    }

    #[test]
    fn the_pool_scores_exactly_like_one_thread() {
        let want = reference(40, 0);
        for workers in [2usize, 3, 4] {
            let mut e = engine(workers);
            let (mut b, mut out) = (Batch::default(), Vec::new());
            // Several batches in a row: the helpers' workers are reused and must stay exact.
            for round in 0..4 {
                fill(&mut b, 40, 0);
                e.evaluate(&mut b, &WallClock::new(), None, &mut out);
                assert_eq!(out, want, "workers = {workers}, round {round}");
            }
        }
    }

    #[test]
    fn a_stalled_helper_cannot_stretch_a_deadline_batch_and_its_late_result_is_thrown_away() {
        let mut e = engine(2);
        let (mut b, mut out) = (Batch::default(), Vec::new());
        let clock = Arc::clone(e.clock());
        // Warm both workers up first (a cold first rollout is slow on a busy machine).
        fill(&mut b, 12, 0);
        e.evaluate(&mut b, &*clock, None, &mut out);
        e.shared.test_stall_us.store(600_000, Ordering::Relaxed);
        // Enough jobs that the helper certainly joins in before the caller has done them all.
        fill(&mut b, 300, 0);
        let t0 = Instant::now();
        let deadline = clock.now_ms() + 100.0;
        e.evaluate(&mut b, &*clock, Some(deadline), &mut out);
        let took = t0.elapsed();
        assert!(
            took < Duration::from_millis(350),
            "the caller waited for the stalled helper: {took:?}"
        );
        assert!(
            e.shared.test_stalled.load(Ordering::Relaxed) >= 1,
            "the helper never took a job, so nothing was tested"
        );
        assert!(
            out.iter().any(|o| o.res.is_some()),
            "the caller scored what it could while the helper slept"
        );
        // The very next batch starts while the helper still sleeps on its old job; when it wakes its
        // result belongs to a finished batch and must vanish.
        e.shared.test_stall_us.store(0, Ordering::Relaxed);
        let want = reference(12, 5);
        fill(&mut b, 12, 5);
        e.evaluate(&mut b, &*clock, None, &mut out);
        assert_eq!(out, want);
        std::thread::sleep(Duration::from_millis(700));
        fill(&mut b, 12, 5);
        e.evaluate(&mut b, &*clock, None, &mut out);
        assert_eq!(
            out, want,
            "a late result from the stalled helper leaked into a newer batch"
        );
    }

    #[test]
    fn a_result_filed_for_an_older_batch_is_ignored_and_an_older_batch_cannot_claim() {
        let e = engine(1);
        let shared = &e.shared;
        let mut b = Batch::default();
        let mut plan = Vec::new();
        fill(&mut b, 4, 0);
        let old = shared.open_batch(&mut b, None);
        let held = shared.claim(old, &mut plan).expect("a job to hold");
        fill(&mut b, 4, 9);
        let new = shared.open_batch(&mut b, None);
        assert!(new > old);
        let bogus = EvalOut {
            res: Some(EvalResult {
                score: 1.0,
                self_out: 0,
                enemy_out: 0,
                enemy_sealed: false,
            }),
            ticks: 9,
            units: 0,
        };
        shared.complete(old, held.idx, bogus);
        {
            let p = shared.pool.lock().expect("pool");
            assert!(
                !p.slots[held.idx].done && p.finished == 0,
                "the late result landed in the new batch"
            );
        }
        assert!(
            shared.claim(old, &mut plan).is_none(),
            "a helper that is a batch behind must not claim"
        );
        assert!(shared.claim(new, &mut plan).is_some());
        // A closed batch takes no results either.
        shared.pool.lock().expect("pool").open = false;
        shared.complete(new, 0, bogus);
        assert!(!shared.pool.lock().expect("pool").slots[0].done);
    }

    #[test]
    fn a_stalled_helper_does_not_cost_a_batch_a_candidate_while_the_deadline_is_far() {
        // Review 3.7a F1: the caller used to sit idle until the deadline and the job the helper held came back cut,
        // which made stage 1 of the search stop early. Now the caller recomputes it.
        let want = reference(60, 4);
        let mut e = engine(2);
        let (mut b, mut out) = (Batch::default(), Vec::new());
        let clock = Arc::clone(e.clock());
        fill(&mut b, 60, 4);
        e.evaluate(&mut b, &*clock, None, &mut out); // warm both workers up
        e.shared.test_stall_us.store(800_000, Ordering::Relaxed);
        fill(&mut b, 60, 4);
        let t0 = Instant::now();
        let deadline = clock.now_ms() + 5_000.0;
        e.evaluate(&mut b, &*clock, Some(deadline), &mut out);
        let took = t0.elapsed();
        assert!(
            e.shared.test_stalled.load(Ordering::Relaxed) >= 1,
            "the helper never took a job, so nothing was tested"
        );
        assert!(
            out.iter().all(|o| o.res.is_some()),
            "{} of {} jobs were cut although the deadline was 5 s away",
            out.iter().filter(|o| o.res.is_none()).count(),
            out.len()
        );
        assert_eq!(out, want, "the recomputed job differs from the healthy answer");
        assert!(
            took < Duration::from_millis(600),
            "the batch waited for the stalled helper: {took:?}"
        );
        e.shared.test_stall_us.store(0, Ordering::Relaxed);
    }

    #[test]
    fn a_new_decision_never_waits_for_a_helper_that_is_still_reading_the_old_one() {
        // Review 3.7a F2: with_ctx must not block behind a preempted helper; it publishes a fresh copy instead.
        let mut e = engine(2);
        let held = e.shared.ctx(); // a helper in the middle of its load
        let t0 = Instant::now();
        let gen_before = held.generation;
        e.with_ctx(|c| c.victim_id = 3);
        assert!(t0.elapsed() < Duration::from_millis(50));
        let now = e.shared.ctx();
        assert_eq!((now.victim_id, now.generation), (3, gen_before + 1));
        assert_eq!(
            (held.victim_id, held.generation),
            (1, gen_before),
            "the reader's copy is untouched"
        );
        drop(held);
        drop(now);
        // Steady state: nobody holds a clone, the context is rewritten in place (no allocation, same generation counter).
        e.with_ctx(|c| c.victim_id = 1);
        assert_eq!(e.shared.ctx().generation, gen_before + 2);
    }

    #[test]
    fn a_batch_without_a_deadline_completes_by_recomputing_what_a_stalled_helper_holds() {
        let want = reference(10, 2);
        let mut e = engine(2);
        e.shared.test_stall_us.store(400_000, Ordering::Relaxed);
        let (mut b, mut out) = (Batch::default(), Vec::new());
        fill(&mut b, 10, 2);
        let t0 = Instant::now();
        e.evaluate(&mut b, &WallClock::new(), None, &mut out);
        assert!(t0.elapsed() < Duration::from_millis(200), "{:?}", t0.elapsed());
        assert_eq!(
            out, want,
            "the recomputed job is the same as the one a healthy helper would give"
        );
        e.shared.test_stall_us.store(0, Ordering::Relaxed);
        // Drop waits for the sleeping helper; that is fine.
    }

    #[test]
    fn a_panicking_helper_job_is_cut_and_the_pool_carries_on_exactly() {
        let want = reference(30, 1);
        let mut e = engine(2);
        let (mut b, mut out) = (Batch::default(), Vec::new());
        let mut seen_cut = false;
        for _ in 0..20 {
            e.shared.test_panic_next.store(true, Ordering::Relaxed);
            fill(&mut b, 30, 1);
            e.evaluate(&mut b, &WallClock::new(), None, &mut out);
            let cut = out.iter().filter(|o| o.res.is_none()).count();
            assert!(cut <= 1, "one panic costs one job, not {cut}");
            if cut == 1 {
                seen_cut = true;
                break;
            }
        }
        assert!(seen_cut, "the injected panic never hit a helper job");
        e.shared.test_panic_next.store(false, Ordering::Relaxed);
        fill(&mut b, 30, 1);
        e.evaluate(&mut b, &WallClock::new(), None, &mut out);
        assert_eq!(out, want, "the helper whose job panicked must reload its world");
    }

    #[test]
    fn the_work_clock_prefetch_is_exact_counted_once_and_dropped_when_the_decision_changes() {
        let want = reference(20, 3);
        let mut e = engine(3);
        let meter = WorkMeter::new();
        meter.set_scale(4);
        e.set_meter(Some(Arc::clone(&meter)));
        assert!(e.speculating());
        let (mut b, mut out) = (Batch::default(), Vec::new());
        let clock = WallClock::new();
        fill(&mut b, 20, 3);
        e.prefetch(&mut b, &clock);
        assert_eq!(meter.ticks(), 0, "speculation must not advance the work clock");
        fill(&mut b, 20, 3);
        e.evaluate(&mut b, &clock, None, &mut out);
        assert_eq!(out, want);
        assert_eq!(e.spec_stats(), (20, 20));
        let ticks: u64 = want.iter().map(|o| u64::from(o.ticks)).sum();
        assert_eq!(
            meter.ticks(),
            ticks * 4,
            "the meter counts each rollout once, as the caller consumes it"
        );
        // A new decision invalidates the cache: the same jobs are computed again, not served.
        e.with_ctx(|_| ());
        fill(&mut b, 20, 3);
        e.evaluate(&mut b, &clock, None, &mut out);
        assert_eq!(out, want);
        assert_eq!(e.spec_stats(), (20, 20), "no stale hit after the generation changed");
        // Without helpers (or a meter) there is nothing to speculate with.
        assert!(!engine(1).speculating());
        assert!(!engine(3).speculating());
    }

    #[test]
    fn scoring_on_a_helper_thread_allocates_nothing_once_warm() {
        let e = engine(1);
        let pw = scene();
        let cfg = HybridConfig::fixed();
        let ctx_guard = e.shared.ctx();
        let mut worker = Worker::new(&cfg, clone_world(&pw), &ctx_guard);
        drop(ctx_guard);
        let shared = Arc::clone(&e.shared);
        let info = std::thread::scope(|s| {
            s.spawn(|| {
                let mut plan = Vec::with_capacity(32);
                let mut batch = Batch::default();
                let mut round = |worker: &mut Worker| {
                    fill(&mut batch, 24, 0);
                    let epoch = shared.open_batch(&mut batch, None);
                    shared.work(worker, epoch, &mut plan, &*shared.clock, true);
                    let p = shared.pool.lock().expect("pool");
                    assert_eq!(p.finished, 24);
                    assert!(p.slots.iter().all(|s| s.done && s.out.res.is_some()));
                };
                for _ in 0..3 {
                    round(&mut worker);
                }
                // `fill` itself builds plan vectors; count only the pool's own path.
                let mut mine = Batch::default();
                fill(&mut mine, 24, 0);
                let mut spare = Batch::default();
                fill(&mut spare, 24, 0);
                allocation_counter::measure(|| {
                    for _ in 0..10 {
                        std::mem::swap(&mut mine, &mut spare);
                        let epoch = shared.open_batch(&mut mine, None);
                        shared.work(&mut worker, epoch, &mut plan, &*shared.clock, true);
                        // The pool keeps the batch until the next open; hand its buffers back.
                        let mut p = shared.pool.lock().expect("pool");
                        p.open = false;
                        std::mem::swap(&mut p.batch, &mut mine);
                    }
                })
            })
            .join()
            .expect("scoring thread")
        });
        assert_eq!(info.count_total, 0, "helper-thread scoring allocated: {info:?}");
    }
}
