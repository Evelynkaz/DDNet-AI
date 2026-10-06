//! The hybrid decision (task 3.5, D-041/D-042/D-048): a pool of candidates from five sources is
//! scored by exact rollouts on `World<f32>`; the best few are re-scored under every combination of
//! the opponents' modelled responses; the plan with the best robust value wins; the planner's
//! shield has the last word.
//!
//! ```text
//! decide
//!  |- setup      threats (1vN), danger flags, hazard fields, decision snapshot
//!  |- pool       warm plan -> proposals -> book -> throws -> techniques (order depends on danger)
//!  |- stage 1    every candidate under the cheap model (all opponents hold)   [budget * (1 - f)]
//!  |- CEM        population samples refitted to the elites (planner's own)
//!  |- stage 2    top-M re-scored under every hold/react combination            [budget * f]
//!  |- extension  only if danger is flagged AND the choice still ends with us out (D-042)
//!  |- shield     escapeExists / saferInput with all opponents' modelled inputs
//! ```
//!
//! **Order of the pool.** Calm: warm, proposals, book, offensive techniques, throws, a few
//! defensive techniques. When the danger flags fire (two or more opponents in the radius, freeze
//! near us, or we are hooked) the defensive techniques come right after the warm plan; when the
//! warm plan itself ends with us frozen under some modelled response (the probe) they come before
//! anything else and get the whole first chunk.
//!
//! **Determinism.** In [`HybridMode::Fixed`] no clock is read, candidates are generated in a fixed
//! order by one thread and scored by the [`Engine`] (a pure function per candidate), results are
//! merged by candidate index: the decision does not depend on the thread count.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ddai_brain::Observation;
use ddai_jsmath::{self as js, Rng};
use ddai_physics::map::MapData;

use crate::clock::{Clock, WallClock};
use crate::fields::{EDGE_GAP_PX, HazardField, freeze_gap_px, hazard_nearness, hazard_tiles, wrap_angle};
use crate::hybrid::anchors::AnchorCache;
use crate::hybrid::config::{HybridConfig, HybridMode, RobustMode};
use crate::hybrid::engine::{Batch, Ctx, Engine, EvalOut, EvalResult};
use crate::hybrid::proposer::{ProposeCtx, Proposer};
use crate::hybrid::techniques::{Generated, Tech, TechCaps, TechCtx, generate};
use crate::hybrid::threat::{Danger, MAX_THREATS, ReactBelief, ThreatSet, robust_value_weighted, threat_radius};
use crate::hybrid::{ABS_AIM, is_abs_aim, resolve_aim};
use crate::physics_adapter::{PhysicsSavedState, PhysicsWorld};
use crate::plan_world::PlanWorld;
use crate::planner::{PlanStep, Planner};
use crate::scripted::scripted_action;
use crate::throw_lines::{
    ThrowSituation, frozen_throw_lines, frozen_throw_worth_trying, throw_lines, throw_worth_trying,
};
use crate::types::{HOOK_FLYING, HOOK_GRABBED, PlayerInput, TeeState};
use crate::vmath::vdistance;

/// Model combinations a plan is scored under, at most (every subset of one or two relevant
/// opponents: 1, 2 or 4; more opponents skip the re-scoring by default).
pub const MAX_COMBOS: usize = 4;

/// Frozen non-victim tees closer than this are passed to the planner as frozen bystanders.
const BYSTANDER_PX: f64 = crate::brains::BYSTANDER_PX;

/// Which world the engine's workers currently score in (two-world search, task 3.5b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lens {
    /// The decision's local world with its threats (the only world of the single-world search).
    Full,
    /// Us and the victim only: cheap to roll out, so many candidates fit; ranks the pool.
    Reduced,
}

/// The opponent model's own planner (task 3.7b): the live `preset_normal` search as the victim would run it, in the victim's seat.
struct MirrorState {
    planner: Box<Planner<PhysicsWorld>>,
    /// The plan it chose last time (the victim's own warm start), in relative aims.
    warm: Option<Vec<PlanStep>>,
    /// Scratch for the plans of one prediction.
    plans: Vec<Vec<PlanStep>>,
    /// The decision snapshot with only us and the victim in it, when something else (a frozen body, a spared tee) is in the decision world.
    saved_pair: Option<PhysicsSavedState>,
}

impl MirrorState {
    fn new(steps_cfg: crate::config::PlannerConfig) -> MirrorState {
        let mut planner = Box::new(Planner::new(steps_cfg));
        planner.deterministic_thaw = true;
        MirrorState {
            planner,
            warm: None,
            plans: Vec::new(),
            saved_pair: None,
        }
    }
}

/// The opponent model is skipped for a victim that has kept its direction neutral and its hook in for this many decisions in a row.
const PASSIVE_DECISIONS: u32 = 6;

/// The opponent model's search is cut after this long at most (ms, task 3.7b review F2); the cap can shorten it.
const MIRROR_MAX_MS: f64 = 2.0;

/// The search never gets less than this under the decision cap (ms).
const MIN_SEARCH_MS: f64 = 1.0;

/// On the work clock the helpers speculate this many rollouts per worker ahead of the serial search.
const SPEC_PER_WORKER: usize = 2;

/// A tee within this many px of a spared position is that spared tee.
const SPARE_MATCH_PX: f64 = 8.0;

/// A free tee this close is within hammer reach (hammer range plus the tee radii and a step), for the threat order.
const HAMMER_THREAT_PX: f64 = 90.0;

/// A flying hook counts as aimed at us when its line passes within this many px of our centre (the hook box is generous).
const HOOK_AIM_SLACK_PX: f64 = 40.0;

/// Ticks of travel at the current speed the shield skip leaves between us and the nearest hazard, on top of `shield_skip_tiles`.
const SKIP_LOOKAHEAD_TICKS: f64 = 27.0;

/// The search budget while we are frozen (ms); no adaptive extension either.
const FROZEN_SEARCH_MS: f64 = 1.0;

/// Where a candidate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Warm,
    Proposal,
    Book,
    Throw,
    Cem,
    Tech(Tech),
}

/// Number of [`Source`] kinds (techniques count as one kind in the per-source tables).
pub const SOURCE_KINDS: usize = 6;

impl Source {
    pub fn kind(self) -> usize {
        match self {
            Source::Warm => 0,
            Source::Proposal => 1,
            Source::Book => 2,
            Source::Throw => 3,
            Source::Cem => 4,
            Source::Tech(_) => 5,
        }
    }

    pub fn kind_name(kind: usize) -> &'static str {
        ["warm", "proposal", "book", "throw", "cem", "technique"][kind]
    }

    /// The label telemetry shows for the chosen plan ("T14 panic hook", "book", ...).
    pub fn label(self) -> &'static str {
        match self {
            Source::Tech(t) => t.name(),
            other => Source::kind_name(other.kind()),
        }
    }
}

/// Physics ticks simulated per phase, summed over all workers (D-045: unlike a clock this does not
/// count host stalls, so it is the measure the 5 ms goal is proven with).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorkCounters {
    /// Rolling the planning world forward by the input lag.
    pub lag: u64,
    /// The proposer's own simulation (the scripted proposer; a fly is measured in wall time).
    pub proposal: u64,
    /// The opponent model's search from the victim's seat (`HybridConfig::mirror_samples`).
    pub mirror: u64,
    /// The nominal cost of a proposer that does not simulate physics (the fly), in tee-tick
    /// equivalents (task 3.7a): charged to the work clock, not part of [`WorkCounters::total_ticks`].
    pub proposal_units: u64,
    pub stage1: u64,
    pub stage2: u64,
    /// The adaptive extension.
    pub extension: u64,
    /// The shield's `escapeExists`/`saferInput` rollouts.
    pub shield: u64,
    /// Hook-anchor ray casts.
    pub rays: u64,
    /// Task 3.9: work the tick counter does not see, in tee-ticks, already charged to the work clock: the v2 hook gate's projections of a
    /// moving victim (one tee-tick each, measured 0.9 us), in the search's rollouts and the opponent model's. Zero with the v2 switches off.
    pub units: u64,
    pub rollouts_stage1: u32,
    pub rollouts_stage2: u32,
    pub rollouts_extension: u32,
}

impl WorkCounters {
    /// Physics ticks of the whole decision.
    pub fn total_ticks(&self) -> u64 {
        self.lag + self.proposal + self.mirror + self.stage1 + self.stage2 + self.extension + self.shield
    }

    pub fn add(&mut self, o: &WorkCounters) {
        self.lag += o.lag;
        self.proposal += o.proposal;
        self.mirror += o.mirror;
        self.proposal_units += o.proposal_units;
        self.stage1 += o.stage1;
        self.stage2 += o.stage2;
        self.extension += o.extension;
        self.shield += o.shield;
        self.rays += o.rays;
        self.units += o.units;
        self.rollouts_stage1 += o.rollouts_stage1;
        self.rollouts_stage2 += o.rollouts_stage2;
        self.rollouts_extension += o.rollouts_extension;
    }
}

/// Everything worth showing about one decision (the web page, the clips, the arena summary).
#[derive(Debug, Clone, Default)]
pub struct DecisionTelemetry {
    pub work: WorkCounters,
    /// Candidates generated / scored (stage 1 or later) per [`Source::kind`].
    pub generated: [u32; SOURCE_KINDS],
    pub evaluated: [u32; SOURCE_KINDS],
    /// The source of the chosen plan (its label), and the technique if it came from one.
    pub chosen: Option<Source>,
    pub chosen_plan: Vec<PlanStep>,
    pub threat_ids: Vec<i32>,
    pub victim_id: i32,
    pub danger: Danger,
    /// Model combinations the top candidates were scored under (1 = no robust stage).
    pub combos: u32,
    pub budget_ms: f64,
    /// Wall time of the search (deadline mode only; 0 in fixed mode, which reads no clock).
    pub search_ms: f64,
    pub proposal_ms: f64,
    /// Wall time spent inside candidate scoring (`Engine::evaluate`), deadline mode only; the rest
    /// of `search_ms` is pool generation and bookkeeping.
    pub rollout_ms: f64,
    pub shield_ms: f64,
    /// The extension ran (danger flagged and the choice still ended with us out).
    pub extended: bool,
    pub shielded: bool,
    pub shield_incomplete: bool,
    /// The shield ran (not frozen, shield on), and whether the remainder of the chosen plan was its
    /// first escape (task 3.5b).
    pub shield_ran: bool,
    pub shield_plan_ok: bool,
    /// The shield was skipped: no hazard within reach of our path (`shield_skip_tiles`).
    pub shield_skipped: bool,
    /// Tees in the decision world after dropping the non-local ones, and how many were dropped.
    pub sim_tees: u32,
    pub dropped_tees: u32,
    /// Candidates early pruning skipped (pre-scored, no full rollout).
    pub pruned: u32,
    /// The search ran out of time before its pool was exhausted.
    pub out_of_time: bool,
    /// Work clock with helper threads: rollouts the helpers computed ahead, and how many of them the
    /// serial search then used. Not part of the decision (the JSON leaves them out: it must be equal
    /// for any thread count).
    pub spec_prefetched: u32,
    pub spec_used: u32,
    pub best_score: f64,
    pub robust_value: f64,
    /// Mean estimated probability that the modelled opponents react (see `ReactBelief`).
    pub react_belief: f64,
    /// The chosen plan ends with us out under some modelled response.
    pub unsafe_choice: bool,
    /// `HybridConfig::debug_dump`: the best candidates by cheap score: `(label, cheap score,
    /// per-combination scores, first step)`.
    pub dump: Vec<(String, f64, Vec<f64>, String)>,
    /// `HybridConfig::debug_pool` (task 3.7b): the whole pool with its scores, the indices re-scored under every
    /// combination (`top`), the index of the pick, and what the robust choice weighed (`weights` by combination,
    /// `lambda`). Empty/`None` unless the flag is on.
    pub pool: Vec<PoolRec>,
    /// `HybridConfig::mirror`: the victim's predicted input for the first plan step (`None` = the hold model was used).
    pub mirror_first: Option<PlayerInput>,
    /// Time the opponent model took (ms on the decision's clock; 0 when it did not run) and whether its deadline cut it short.
    pub mirror_ms: f64,
    pub mirror_cut: bool,
    /// Task 3.9 fire counters of the opt-in switches (so an inert one is visible): polish variants and wall-throw candidates put in
    /// the pool (they are counted under `generated` as `cem` / `throw`, too). JSON: `generated.polish` / `generated.wall`, only when non-zero.
    pub polished: u32,
    pub wall_cands: u32,
    pub pick: Option<usize>,
    pub top: Vec<usize>,
    pub weights: Vec<f64>,
    pub lambda: f64,
}

/// One candidate of a decision's pool, for the loss diagnosis (`HybridConfig::debug_pool`).
#[derive(Debug, Clone)]
pub struct PoolRec {
    pub src: Source,
    pub plan: Vec<PlanStep>,
    /// The score that ranked the pool (stage 1), `None` if the deadline cut it before its rollout.
    pub cheap: Option<f64>,
    /// Full-model scores by combination (`None` = not re-scored) and the ticks we were out in each.
    pub scores: [Option<f64>; MAX_COMBOS],
    pub self_out: [Option<i32>; MAX_COMBOS],
}

/// A plan scored after the fact by a decision's own evaluator (`HybridSearch::debug_score`).
#[derive(Debug, Clone)]
pub struct DebugScore {
    /// Score under every model combination of the decision, and the robust value (what `choose` ranks by,
    /// without the warm bonus).
    pub combos: Vec<f64>,
    pub robust: f64,
    pub self_out: i32,
}

/// What a plan did in the true world (`HybridSearch::debug_truth`): ticks (from the plan's start) at which we
/// and the victim first were out (frozen or dead), `-1` = never within the plan; how close to a freeze we came.
#[derive(Debug, Clone, Copy, Default)]
pub struct TruthOutcome {
    pub me_out_tick: i32,
    pub enemy_out_tick: i32,
    /// The last tick at which we hooked the victim or hit it with the hammer, `-1` = never.
    pub touch_tick: i32,
    pub min_gap_px: f64,
    pub end_gap_enemy_px: f64,
    pub end_dist_px: f64,
}

impl DecisionTelemetry {
    pub fn to_json(&self) -> String {
        let src = |a: &[u32; SOURCE_KINDS]| {
            (0..SOURCE_KINDS)
                .map(|k| format!("\"{}\":{}", Source::kind_name(k), a[k]))
                .collect::<Vec<_>>()
                .join(",")
        };
        let ids = self
            .threat_ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let plan = self
            .chosen_plan
            .iter()
            .map(|s| format!("[{},{},{},{}]", s.dir, s.jump, s.hook, s.fire))
            .collect::<Vec<_>>()
            .join(",");
        let w = &self.work;
        let dump = if self.dump.is_empty() {
            String::new()
        } else {
            let items = self
                .dump
                .iter()
                .map(|(l, s, all, first)| {
                    format!(
                        "{{\"src\":\"{l}\",\"score\":{:.3},\"combos\":[{}],\"step0\":\"{first}\"}}",
                        finite(*s),
                        all.iter()
                            .map(|v| format!("{:.3}", finite(*v)))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!(",\"dump\":[{items}]")
        };
        // The optional counters of the work object, present only when they are non-zero (so a hybrid without the v2 switches prints what it always did).
        let mut generated_tail = String::new();
        if self.polished > 0 {
            generated_tail.push_str(&format!(",\"polish\":{}", self.polished));
        }
        if self.wall_cands > 0 {
            generated_tail.push_str(&format!(",\"wall\":{}", self.wall_cands));
        }
        let mut work_tail = String::new();
        if w.mirror > 0 {
            work_tail.push_str(&format!(",\"mirror\":{}", w.mirror));
        }
        if w.units > 0 {
            work_tail.push_str(&format!(",\"units\":{}", w.units));
        }
        format!(
            "{{\"chosen\":\"{}\",\"plan\":[{}],\"victim\":{},\"threats\":[{}],\"danger\":\"{}\",\"combos\":{},\
\"generated\":{{{}{}}},\"evaluated\":{{{}}},\"budget_ms\":{},\"search_ms\":{:.3},\"proposal_ms\":{:.3},\"mirror_ms\":{:.3},\"rollout_ms\":{:.3},\"shield_ms\":{:.3},\
\"extended\":{},\"shielded\":{},\"shield_incomplete\":{},\"shield_plan_ok\":{},\"sim_tees\":{},\"dropped_tees\":{},\"pruned\":{},\"out_of_time\":{},\"unsafe\":{},\"best_score\":{:.4},\"robust\":{:.4},\"react_belief\":{:.3},\
\"work\":{{\"ticks\":{},\"lag\":{},\"proposal\":{},\"proposal_units\":{},\"stage1\":{},\"stage2\":{},\"extension\":{},\"shield\":{},\"rays\":{}{}}}{}}}",
            self.chosen.map_or("none", Source::label),
            plan,
            self.victim_id,
            ids,
            self.danger.reasons(),
            self.combos,
            src(&self.generated),
            generated_tail,
            src(&self.evaluated),
            self.budget_ms,
            self.search_ms,
            self.proposal_ms,
            self.mirror_ms,
            self.rollout_ms,
            self.shield_ms,
            self.extended,
            self.shielded,
            self.shield_incomplete,
            self.shield_plan_ok,
            self.sim_tees,
            self.dropped_tees,
            self.pruned,
            self.out_of_time,
            self.unsafe_choice,
            finite(self.best_score),
            finite(self.robust_value),
            finite(self.react_belief),
            w.total_ticks(),
            w.lag,
            w.proposal,
            w.proposal_units,
            w.stage1,
            w.stage2,
            w.extension,
            w.shield,
            w.rays,
            work_tail,
            if self.mirror_cut {
                format!("{dump},\"mirror_cut\":true")
            } else {
                dump
            },
        )
    }
}

fn finite(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

/// One candidate and its scores by model combination.
#[derive(Clone)]
struct Cand {
    plan: Vec<PlanStep>,
    src: Source,
    res: [Option<EvalResult>; MAX_COMBOS],
    /// The early-pruning pre-score (first steps of the plan, cheap model); `None` = not pre-scored.
    pre: Option<f64>,
    /// Two-world search (task 3.5b): the cheap score in the *reduced* world (us and the victim only),
    /// which ranks the pool; `res` then holds the full-world scores (threats simulated) of the few
    /// candidates that earned a re-scoring.
    red: Option<EvalResult>,
}

impl Cand {
    fn new(plan: Vec<PlanStep>, src: Source) -> Cand {
        Cand {
            plan,
            src,
            res: [None; MAX_COMBOS],
            pre: None,
            red: None,
        }
    }

    /// Whether early pruning may skip this candidate's full rollout: book plans, throw lines and CEM
    /// samples. Techniques (whose payoff may lie beyond a short horizon), the warm plan and the
    /// proposals always get a full rollout.
    fn prunable(&self) -> bool {
        matches!(self.src, Source::Book | Source::Throw | Source::Cem)
    }

    /// The score that ranks the pool: the reduced-world one when there is one (two-world search), else the
    /// stage-1 score under "everybody holds".
    fn cheap(&self) -> Option<f64> {
        self.red.or(self.res[0]).map(|r| r.score)
    }

    /// The full-world score under "everybody holds" (what the robust choice starts from).
    fn full0(&self) -> Option<f64> {
        self.res[0].map(|r| r.score)
    }

    fn complete(&self, combos: usize) -> bool {
        self.res[..combos].iter().all(Option::is_some)
    }

    fn worst_self_out(&self, combos: usize) -> i32 {
        self.res[..combos]
            .iter()
            .flatten()
            .map(|r| r.self_out)
            .max()
            .unwrap_or(0)
    }
}

/// A plan signature for de-duplication: the discrete moves and the aim rounded to 1/50 rad.
fn signature(plan: &[PlanStep]) -> Vec<i32> {
    plan.iter()
        .flat_map(|s| [s.dir, s.jump, s.hook, s.fire, (s.aim * 50.0).round() as i32])
        .collect()
}

/// What the brain hands the search for one decision.
pub struct DecisionInput<'a> {
    pub obs: &'a Observation,
    pub self_id: i32,
    pub victim_id: i32,
    /// The input this client sent last (the plan's starting point).
    pub prev: PlayerInput,
    pub lag_ticks: u32,
    /// Physics ticks the brain simulated to roll the planning world forward (work counter).
    pub roll_ticks: u64,
}

/// The decision procedure and everything it keeps between decisions.
pub struct HybridSearch {
    cfg: HybridConfig,
    // Boxed: the planner is ~300 KB and a world/snapshot ~100 KB; threads have small stacks.
    planner: Box<Planner<PhysicsWorld>>,
    /// The decision planning world: the brain syncs it, rolls it forward and hands it over.
    world: Box<PhysicsWorld>,
    saved: Box<PhysicsSavedState>,
    engine: Engine,
    proposer: Box<dyn Proposer>,
    anchors: AnchorCache,
    map: Arc<MapData>,
    /// The work clock's counter, when the brain runs on one.
    meter: Option<Arc<crate::hybrid::work::WorkMeter>>,
    /// The decision's model combinations as reaction masks (see `decide`) and their probabilities.
    masks: Vec<u32>,
    mask_weights: Vec<f64>,
    /// Online per-opponent estimate that it reacts rather than holds, and what was predicted for
    /// each modelled opponent at the previous decision (hold input, scripted reply).
    beliefs: HashMap<i32, ReactBelief>,
    prev_predictions: Vec<(i32, PlayerInput, PlayerInput)>,
    /// Whether the mode reads a clock (deadline) and the wall time spent scoring this decision.
    timed: bool,
    rollout_ms: f64,
    /// The shield's two snapshot buffers, kept across decisions.
    shield_bufs: Box<crate::shield::ShieldBuffers<PhysicsWorld>>,
    batch: Batch,
    outs: Vec<EvalOut>,
    last_prop_ticks: u64,
    /// Task 3.7b diagnostics (`debug_score`): the last full decision left the engine's context ready, and the worst-case
    /// weight it chose with.
    diag_ready: bool,
    last_lambda: f64,
    /// Task 3.7b (`HybridConfig::mirror`): the planner that plays the victim's seat and what it last chose.
    mirror: Option<Box<MirrorState>>,
    /// Task 3.9: the decision's priced extra work so far (`WorkCounters::units`).
    dec_units: u64,
    /// The victim's predicted inputs of the current decision, one per plan step (empty = the victim holds its input).
    mirror_inputs: Vec<PlayerInput>,
    /// The victim that has kept its direction neutral and its hook in for `.1` decisions in a row (an idle or camping opponent,
    /// for which "it keeps its input" is the right model and the opponent model has nothing to add).
    passive: (i32, u32),
    /// What the last decision's proposer cost against the cap (ms): the opponent model's deadline leaves room for it.
    last_proposal_ms: f64,
    /// Two-world search: the decision snapshot without the threats, the threat set to switch back to, and
    /// the world the engine currently scores in.
    saved_red: Box<PhysicsSavedState>,
    lens_threats: Option<ThreatSet>,
    lens: Lens,
    /// What the live bot knows (task 3.5b): tees to spare with their velocities, and where to head.
    spares: Vec<crate::vmath::Vec2>,
    spare_vels: Vec<crate::vmath::Vec2>,
    spare_ids: Vec<i32>,
    travel_goal: Option<crate::vmath::Vec2>,
    /// Distance to the hazards the physics sees, front layer included, for the shield skip; per map (collision identity).
    skip_field: Option<(u64, Arc<HazardField>)>,
}

impl HybridSearch {
    pub fn new(
        cfg: HybridConfig,
        proposer: Box<dyn Proposer>,
        world: PhysicsWorld,
        clock: Arc<WallClock>,
    ) -> HybridSearch {
        cfg.validate().expect("valid hybrid config");
        let planner = Box::new(Planner::new(cfg.planner));
        let saved = Box::new(world.save_state());
        let saved_red = saved.clone();
        let map = world.map().clone();
        let ctx = Box::new(Ctx {
            generation: 0,
            saved: saved.clone(),
            self_id: 0,
            victim_id: 1,
            prev: crate::types::empty_input(),
            victim_input: crate::types::empty_input(),
            victim_plan: Vec::new(),
            opp_seed: 1,
            field: Arc::new(HazardField {
                width: 0,
                height: 0,
                dist: Vec::new(),
            }),
            unfreeze: Arc::new(HazardField {
                width: 0,
                height: 0,
                dist: Vec::new(),
            }),
            frozen_bystanders: Vec::new(),
            frozen_bystander_vels: Vec::new(),
            spares: Vec::new(),
            spare_vels: Vec::new(),
            travel_goal: None,
            threats: None,
            self_freeze_bias: 1.0,
        });
        let engine = Engine::new(&cfg, &world, ctx, clock);
        HybridSearch {
            cfg,
            planner,
            world: Box::new(world),
            saved,
            engine,
            proposer,
            anchors: AnchorCache::new(),
            map,
            meter: None,
            masks: Vec::new(),
            mask_weights: Vec::new(),
            beliefs: HashMap::new(),
            prev_predictions: Vec::new(),
            timed: false,
            rollout_ms: 0.0,
            shield_bufs: Box::default(),
            batch: Batch::default(),
            outs: Vec::new(),
            last_prop_ticks: 0,
            diag_ready: false,
            last_lambda: 0.0,
            mirror: None,
            dec_units: 0,
            mirror_inputs: Vec::new(),
            passive: (-1, 0),
            last_proposal_ms: 0.0,
            saved_red,
            lens_threats: None,
            lens: Lens::Full,
            spares: Vec::new(),
            spare_vels: Vec::new(),
            spare_ids: Vec::new(),
            travel_goal: None,
            skip_field: None,
        }
    }

    /// The live bot's spared tees (friends, ignored, AFK: the rope and the hammer must not catch them,
    /// and they are no threat) and the point to head for; they hold until replaced (`set_live`
    /// with empty slices and `None` clears them). Positions and velocities in pixels (per tick).
    pub fn set_live(
        &mut self,
        spares: &[crate::vmath::Vec2],
        spare_vels: &[crate::vmath::Vec2],
        spare_ids: &[i32],
        travel_goal: Option<crate::vmath::Vec2>,
    ) {
        self.spares.clear();
        self.spares.extend_from_slice(spares);
        self.spare_vels.clear();
        self.spare_vels.extend_from_slice(spare_vels);
        self.spare_ids.clear();
        self.spare_ids.extend_from_slice(spare_ids);
        self.travel_goal = travel_goal;
    }

    /// Whether `t` is one of the spared tees (by client id when the live bot names them, else by where
    /// they are).
    fn is_spared(&self, t: &TeeState) -> bool {
        self.spare_ids.contains(&t.id) || self.spares.iter().any(|p| vdistance(*p, t.pos) < SPARE_MATCH_PX)
    }

    /// Lets the search report its work to a work clock.
    pub fn set_meter(&mut self, meter: Option<Arc<crate::hybrid::work::WorkMeter>>) {
        self.engine.set_meter(meter.clone());
        self.meter = meter;
    }

    pub fn config(&self) -> &HybridConfig {
        &self.cfg
    }

    pub fn world_mut(&mut self) -> &mut PhysicsWorld {
        &mut self.world
    }

    /// Gives the proposer back (the brain rebuilds the search when the map changes).
    pub fn into_proposer(mut self) -> Box<dyn Proposer> {
        std::mem::replace(&mut self.proposer, Box::new(crate::hybrid::proposer::NoProposer))
    }

    pub fn proposer_name(&self) -> &str {
        self.proposer.name()
    }

    /// What the work clock charges for one `propose` call of this search's proposer (tee-tick equivalents).
    pub fn proposer_work_units(&self) -> u64 {
        self.proposer.work_units()
    }

    /// The proposer, for the read-only visualisation stream (task 7.4).
    pub fn proposer(&self) -> &dyn Proposer {
        &*self.proposer
    }

    pub fn proposer_mut(&mut self) -> &mut dyn Proposer {
        &mut *self.proposer
    }

    pub fn workers(&self) -> usize {
        self.engine.workers()
    }

    /// New episode: forgets the warm plan and hidden state, reseeds every random source.
    pub fn reset(&mut self, ctx: &ddai_brain::ResetContext) {
        self.planner.set_search_seed(ctx.seed as u32);
        self.planner.warm = None;
        self.mirror = None;
        self.passive = (-1, 0);
        self.beliefs.clear();
        self.prev_predictions.clear();
        self.proposer.reset(ctx);
        self.last_prop_ticks = self.proposer.work_ticks();
    }

    /// Runs `jobs` (candidate index, combo) as one batch and files the results into `cands`.
    /// Returns whether some rollout was cut by the deadline. Ticks are added to `ticks`.
    fn run_jobs(
        &mut self,
        clock: &dyn Clock,
        cands: &mut [Cand],
        jobs: &[(usize, u32)],
        deadline_ms: Option<f64>,
        ticks: &mut u64,
        rollouts: &mut u32,
    ) -> bool {
        if jobs.is_empty() {
            return false;
        }
        let n = self.cfg.planner.steps as usize;
        self.batch.clear(n);
        let mut last: Option<(usize, usize)> = None;
        for &(ci, combo) in jobs {
            let pi = match last {
                Some((lc, lp)) if lc == ci => lp,
                _ => {
                    let p = self.batch.push_plan(&cands[ci].plan);
                    last = Some((ci, p));
                    p
                }
            };
            self.batch.push_job(pi, self.masks[combo as usize]);
        }
        let t0 = self.timed.then(|| clock.now_ms());
        self.engine
            .evaluate(&mut self.batch, clock, deadline_ms, &mut self.outs);
        if let Some(t0) = t0 {
            self.rollout_ms += clock.now_ms() - t0;
        }
        let mut cut = false;
        for (k, &(ci, combo)) in jobs.iter().enumerate() {
            let o = self.outs[k];
            *ticks += u64::from(o.ticks);
            self.dec_units += u64::from(o.units);
            match o.res {
                Some(r) => {
                    if self.lens == Lens::Reduced {
                        cands[ci].red = Some(r);
                    } else {
                        cands[ci].res[combo as usize] = Some(r);
                    }
                    *rollouts += 1;
                }
                None => cut = true,
            }
        }
        // The meter was advanced by the engine after each rollout.
        cut
    }

    /// Work clock with helper threads (task 3.7a): the helpers score, in parallel and without touching the
    /// clock, the rollouts the serial search below is about to ask for; the search then finds them in the
    /// engine's cache and charges the work clock one rollout at a time, in its own order. `fill` pushes the
    /// plans and jobs (it gets the model-combination masks). A wrong guess only wastes a helper's time.
    fn prefetch(&mut self, clock: &dyn Clock, fill: impl FnOnce(&mut Batch, &[u32])) {
        if !self.engine.speculating() {
            return;
        }
        self.batch.clear(self.cfg.planner.steps as usize);
        fill(&mut self.batch, &self.masks);
        self.engine.prefetch(&mut self.batch, clock);
    }

    /// How many rollouts ahead the work clock speculates.
    fn spec_window(&self) -> usize {
        SPEC_PER_WORKER * self.engine.workers()
    }

    /// Prefetches the (candidate, combination) rollouts of `jobs`, each over `horizon` steps (`0` = all).
    fn prefetch_jobs(&mut self, clock: &dyn Clock, cands: &[Cand], jobs: &[(usize, u32)], horizon: u32) {
        self.prefetch(clock, |b, masks| {
            let mut last: Option<(usize, usize)> = None;
            for &(ci, combo) in jobs {
                let pi = match last {
                    Some((lc, lp)) if lc == ci => lp,
                    _ => {
                        let p = b.push_plan(&cands[ci].plan);
                        last = Some((ci, p));
                        p
                    }
                };
                if horizon > 0 {
                    b.push_job_short(pi, masks[combo as usize], horizon);
                } else {
                    b.push_job(pi, masks[combo as usize]);
                }
            }
        });
    }

    /// Task 3.7b (`HybridConfig::mirror`): what would the victim do? A small `preset_normal` search from the victim's
    /// seat against what we keep doing -- its book seeds, its last plan one step on, a few CEM samples -- with the planner's flip
    /// hysteresis; the inputs of its best plan, one per plan step, land in `self.mirror_inputs`.
    ///
    /// On the wall clock the search has a deadline (`deadline_ms`, a reading of `clock`): rollouts after it are not started, one that
    /// is running is cut, and the best plan found so far is used -- or none (the rollouts then keep "the victim holds its input").
    /// Returns the physics ticks simulated (already charged to the work meter) and whether the deadline cut it short.
    #[allow(clippy::too_many_arguments)]
    fn mirror_predict(
        &mut self,
        clock: &dyn Clock,
        deadline_ms: Option<f64>,
        self_id: i32,
        victim_id: i32,
        me: &TeeState,
        victim: &TeeState,
        field: &crate::fields::HazardField,
        unfreeze: &crate::fields::HazardField,
    ) -> (u64, bool) {
        let samples = self.cfg.mirror_samples;
        let m = self.mirror.get_or_insert_with(|| {
            // The live `preset_normal` search (task 3.9: or the planner `mirror_planner` names), on the step layout of the hybrid's own
            // plans (`predicted[s]` is indexed by step).
            let own = self.cfg.planner;
            let mut state = MirrorState::new(crate::config::PlannerConfig {
                steps: own.steps,
                plan_step: own.plan_step,
                front_steps: own.front_steps,
                front_step: own.front_step,
                ..self.cfg.mirror_planner.unwrap_or_else(crate::config::preset_normal)
            });
            state.planner.prepare_ceiling(self.world.collision());
            Box::new(state)
        });
        let MirrorState {
            planner,
            warm,
            plans,
            saved_pair,
        } = &mut **m;
        // The model plays the two tees alone: whatever else is in the decision world (a frozen body, a spared tee) is taken out for
        // its rollouts and put back afterwards.
        let others: Vec<i32> = self
            .world
            .all_tees()
            .iter()
            .map(|t| t.id)
            .filter(|&id| id != self_id && id != victim_id)
            .collect();
        for id in &others {
            self.world.remove_tee(*id);
        }
        let snapshot: &PhysicsSavedState = if others.is_empty() {
            &self.saved
        } else {
            match saved_pair {
                Some(s) => self.world.save_state_into(s),
                None => *saved_pair = Some(self.world.save_state()),
            }
            saved_pair.as_ref().expect("just saved")
        };
        match &mut planner.saved {
            Some(saved) => saved.assign_from(snapshot),
            None => planner.saved = Some(snapshot.clone()),
        }
        planner.opp_seed = self.planner.opp_seed;
        planner.rng = Rng::new(self.planner.opp_seed ^ 0x9e37_79b9);
        let hold_us = crate::brains::enemy_input_from_tee(me);
        let victim_prev = crate::brains::enemy_input_from_tee(victim);
        let aim_v = js::atan2(me.pos.y - victim.pos.y, me.pos.x - victim.pos.x);
        let before = planner.eval_ticks;
        plans.clear();
        plans.extend(planner.seed_plans(&*self.world, victim_id, self_id, field, aim_v));
        planner.warm.clone_from(warm);
        planner.warm_shift_steps = 1;
        if let Some(w) = warm.as_ref() {
            plans.push(
                w[1..]
                    .iter()
                    .copied()
                    .chain(std::iter::once(*w.last().unwrap()))
                    .collect(),
            );
        }
        let dist = planner.build_dist(0.0);
        for _ in 0..samples {
            plans.push(planner.sample_plan(&dist));
        }
        let mut scores: Vec<f64> = Vec::with_capacity(plans.len());
        let mut charged = before;
        let mut priced = planner.intercept_count();
        let mut cut = false;
        for plan in plans.iter() {
            if !scores.is_empty() && deadline_ms.is_some_and(|d| clock.now_ms() >= d) {
                cut = true;
                break;
            }
            let r = planner.evaluate_impl(
                &mut *self.world,
                victim_id,
                self_id,
                victim_prev,
                plan,
                hold_us,
                field,
                unfreeze,
                deadline_ms.map(|d| (clock, d)),
            );
            if let Some(m) = &self.meter {
                m.add_units(2 * (planner.eval_ticks - charged));
            }
            charged = planner.eval_ticks;
            let projections = planner.intercept_count();
            self.dec_units += projections - priced;
            if let Some(m) = &self.meter {
                m.add_units(projections - priced);
            }
            priced = projections;
            match r {
                Some(score) => scores.push(score),
                None => {
                    cut = true;
                    break;
                }
            }
        }
        let mut best: Option<usize> = None;
        let mut best_stay: Option<usize> = None;
        let (mut best_score, mut stay_score) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, &score) in scores.iter().enumerate() {
            let plan = &plans[i];
            if plan[0].dir == victim_prev.direction && score > stay_score {
                stay_score = score;
                best_stay = Some(i);
            }
            if score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        let Some(mut bi) = best else {
            self.mirror_inputs.clear();
            if !others.is_empty() {
                self.world.restore_state(&self.saved);
            }
            return (planner.eval_ticks - before, cut);
        };
        let flip_margin = planner.config().flip_margin;
        if let Some(si) = best_stay
            && flip_margin > 0.0
            && plans[bi][0].dir != victim_prev.direction
            && best_score - stay_score < flip_margin
        {
            bi = si;
        }
        // The best plan once more, recording the inputs it sends (not once the deadline has passed: then the victim holds its input).
        if deadline_ms.is_some_and(|d| clock.now_ms() >= d) {
            cut = true;
            self.mirror_inputs.clear();
        } else {
            let mut rec = std::mem::take(&mut self.mirror_inputs);
            rec.clear();
            planner.record_inputs = Some(rec);
            let _ = planner.evaluate_impl(
                &mut *self.world,
                victim_id,
                self_id,
                victim_prev,
                &plans[bi],
                hold_us,
                field,
                unfreeze,
                None,
            );
            self.mirror_inputs = planner.record_inputs.take().unwrap_or_default();
            *warm = Some(plans[bi].clone());
            if let Some(m) = &self.meter {
                m.add_units(2 * (planner.eval_ticks - charged));
            }
            let projections = planner.intercept_count();
            self.dec_units += projections - priced;
            if let Some(m) = &self.meter {
                m.add_units(projections - priced);
            }
        }
        if !others.is_empty() {
            self.world.restore_state(&self.saved);
        }
        (planner.eval_ticks - before, cut)
    }

    /// Task 3.7b: scores `plans` with the evaluator of the decision just made -- the same snapshot, threats and model
    /// combinations -- without touching anything the next decision reads (the work meter moves, which only the
    /// relative budget of the next decision would notice, and it starts from a fresh reading). `None` when the last
    /// decision did not get as far as a pick. Diagnostics only: allocates.
    pub fn debug_score(&mut self, clock: &dyn Clock, plans: &[Vec<PlanStep>]) -> Option<Vec<DebugScore>> {
        if !self.diag_ready || self.lens != Lens::Full {
            return None;
        }
        let ncombos = self.masks.len();
        let mut cands: Vec<Cand> = plans
            .iter()
            .map(|p| Cand::new(sanitize(p.clone(), self.cfg.planner.steps as usize), Source::Warm))
            .collect();
        let jobs: Vec<(usize, u32)> = (0..cands.len())
            .flat_map(|ci| (0..ncombos as u32).map(move |co| (ci, co)))
            .collect();
        let (mut ticks, mut rolls) = (0u64, 0u32);
        let was_timed = self.timed;
        self.timed = false;
        let meter_before = self.meter.as_ref().map(|m| m.ticks());
        let cut = self.run_jobs(clock, &mut cands, &jobs, None, &mut ticks, &mut rolls);
        if let (Some(m), Some(t)) = (&self.meter, meter_before) {
            m.rewind_to(t);
        }
        self.timed = was_timed;
        debug_assert!(!cut);
        Some(
            cands
                .iter()
                .map(|c| {
                    let combos = scores_of(c, ncombos);
                    let robust = if combos.len() == ncombos && ncombos > 1 {
                        robust_value_weighted(&combos, &self.mask_weights[..ncombos], self.last_lambda)
                    } else {
                        combos.first().copied().unwrap_or(f64::NAN)
                    };
                    DebugScore {
                        combos,
                        robust,
                        self_out: c.worst_self_out(ncombos),
                    }
                })
                .collect(),
        )
    }

    /// Task 3.7b: the context of the last decision, if it got as far as a pick (`debug_oracle` scores plans in it later).
    pub fn debug_ctx(&self) -> Option<Arc<Ctx>> {
        self.diag_ready.then(|| self.engine.debug_ctx())
    }

    /// Task 3.7b: scores `plans` in a kept decision context with the opponent's actual inputs per plan step.
    pub fn debug_oracle(&mut self, ctx: &Ctx, plans: &[&[PlanStep]], predicted: &[PlayerInput]) -> Vec<Option<f64>> {
        let n = self.cfg.planner.steps as usize;
        let sane: Vec<Vec<PlanStep>> = plans.iter().map(|p| sanitize(p.to_vec(), n)).collect();
        let refs: Vec<&[PlanStep]> = sane.iter().map(Vec::as_slice).collect();
        self.engine
            .debug_eval_predicted(ctx, &refs, predicted)
            .into_iter()
            .map(|r| r.map(|e| e.score))
            .collect()
    }

    /// Task 3.7b: rolls `plan` out from the planning world as it stands (the caller synced it to the true world and
    /// rolled it through the input lag) against the opponent's *recorded* inputs, `opp(t)` for the plan's tick `t`,
    /// and reports what happened. The world is put back. Diagnostics only: allocates.
    pub fn debug_truth(
        &mut self,
        self_id: i32,
        victim_id: i32,
        prev: PlayerInput,
        plan: &[PlanStep],
        opp: &dyn Fn(usize) -> PlayerInput,
        mut inputs_out: Option<&mut Vec<PlayerInput>>,
    ) -> TruthOutcome {
        let saved = self.world.save_state();
        let track_aim = self.cfg.planner.track_aim;
        let mut out = TruthOutcome {
            me_out_tick: -1,
            enemy_out_tick: -1,
            touch_tick: -1,
            min_gap_px: f64::INFINITY,
            end_gap_enemy_px: f64::INFINITY,
            end_dist_px: f64::INFINITY,
        };
        let mut events = Vec::new();
        let mut input = prev;
        let mut t = 0usize;
        for (s, st) in plan.iter().enumerate() {
            let me_now = self.world.get_tee(self_id);
            let en_now = self.world.get_tee(victim_id);
            let enemy_dist = match (me_now, en_now) {
                (Some(m), Some(e)) => vdistance(m.pos, e.pos),
                _ => 0.0,
            };
            let mut aim = st.aim;
            if is_abs_aim(aim) {
                aim -= ABS_AIM;
            } else if track_aim && let (Some(m), Some(e)) = (me_now, en_now) {
                aim += crate::trig::atan2(e.pos.y - m.pos.y, e.pos.x - m.pos.x);
            }
            self.planner.swing_target_frozen = en_now.is_some_and(|e| e.frozen);
            self.planner.swing_rope_on = me_now.is_some_and(|m| m.hooked_player == victim_id);
            self.planner.swing_target = en_now;
            let (hook_ok, aim, snapped) = self.planner.gate_and_snap(
                &*self.world,
                self_id,
                victim_id,
                me_now.as_ref(),
                en_now.as_ref(),
                *st,
                input,
                aim,
            );
            input = self.planner.step_to_input_snapped(
                &*self.world,
                *st,
                input,
                enemy_dist,
                hook_ok,
                me_now.map(|m| m.pos),
                en_now.map(|e| e.pos),
                en_now.map(|e| e.vel),
                aim,
                snapped,
            );
            if let Some(v) = inputs_out.as_deref_mut() {
                v.push(input);
            }
            for _ in 0..self.planner.step_ticks[s] {
                self.world.set_input(self_id, input);
                self.world.set_input(victim_id, opp(t));
                self.world.step_into(&mut events);
                t += 1;
                let tick = t as i32;
                for e in &events {
                    if let crate::types::WorldEvent::HammerHit { from, to } = e
                        && *from == self_id
                        && *to == victim_id
                    {
                        out.touch_tick = tick;
                    }
                }
                if let Some(m) = self.world.get_tee(self_id) {
                    if m.hooked_player == victim_id {
                        out.touch_tick = tick;
                    }
                    if (m.frozen || !m.alive) && out.me_out_tick < 0 {
                        out.me_out_tick = tick;
                    }
                    let gap = if m.frozen || !m.alive {
                        0.0
                    } else {
                        freeze_gap_px(self.world.collision(), m.pos.x, m.pos.y)
                    };
                    out.min_gap_px = out.min_gap_px.min(gap);
                }
                if let Some(e) = self.world.get_tee(victim_id)
                    && (e.frozen || !e.alive)
                    && out.enemy_out_tick < 0
                {
                    out.enemy_out_tick = tick;
                }
            }
        }
        if let (Some(m), Some(e)) = (self.world.get_tee(self_id), self.world.get_tee(victim_id)) {
            out.end_dist_px = vdistance(m.pos, e.pos);
            out.end_gap_enemy_px = if e.frozen || !e.alive {
                0.0
            } else {
                freeze_gap_px(self.world.collision(), e.pos.x, e.pos.y)
            };
        }
        self.world.restore_state(&saved);
        out
    }

    /// Points the engine's workers at the reduced world (us and the victim) or the full one (the local
    /// tees with the threats). A no-op unless the decision runs the two-world search.
    fn set_lens(&mut self, lens: Lens) {
        if self.lens == lens || self.lens_threats.is_none() {
            return;
        }
        let (saved, saved_red, threats) = (&*self.saved, &*self.saved_red, &self.lens_threats);
        self.engine.with_ctx(|c| match lens {
            Lens::Reduced => {
                c.saved.assign_from(saved_red);
                c.threats = None;
            }
            Lens::Full => {
                c.saved.assign_from(saved);
                c.threats.clone_from(threats);
            }
        });
        self.lens = lens;
    }

    /// Pre-scores `idx` of `cands` over the first `horizon` steps under the cheap model (early
    /// pruning). Results land in `Cand::pre`; ticks are added to `ticks`. A pre-score cut by the
    /// deadline stays `None`.
    fn prescreen(
        &mut self,
        clock: &dyn Clock,
        cands: &mut [Cand],
        idx: &[usize],
        horizon: u32,
        deadline_ms: Option<f64>,
        ticks: &mut u64,
    ) {
        if idx.is_empty() {
            return;
        }
        let n = self.cfg.planner.steps as usize;
        self.batch.clear(n);
        for &ci in idx {
            let pi = self.batch.push_plan(&cands[ci].plan);
            self.batch.push_job_short(pi, self.masks[0], horizon);
        }
        let t0 = self.timed.then(|| clock.now_ms());
        self.engine
            .evaluate(&mut self.batch, clock, deadline_ms, &mut self.outs);
        if let Some(t0) = t0 {
            self.rollout_ms += clock.now_ms() - t0;
        }
        for (k, &ci) in idx.iter().enumerate() {
            *ticks += u64::from(self.outs[k].ticks);
            self.dec_units += u64::from(self.outs[k].units);
            cands[ci].pre = self.outs[k].res.map(|r| r.score);
        }
    }

    /// One decision. `clock` is only read in deadline mode.
    #[allow(clippy::too_many_lines)]
    pub fn decide(&mut self, clock: &dyn Clock, inp: &DecisionInput<'_>) -> (PlayerInput, DecisionTelemetry) {
        let timed = matches!(self.cfg.mode, HybridMode::Deadline { .. });
        self.timed = timed;
        self.rollout_ms = 0.0;
        self.dec_units = 0;
        self.diag_ready = false;
        let now = |c: &dyn Clock| if timed { c.now_ms() } else { 0.0 };
        let mut tel = DecisionTelemetry {
            victim_id: inp.victim_id,
            combos: 1,
            ..DecisionTelemetry::default()
        };
        tel.work.lag = inp.roll_ticks;
        let spec0 = self.engine.spec_stats();
        if let Some(m) = &self.meter {
            m.set_scale(self.world.all_tees().len());
            m.add(inp.roll_ticks);
        }
        tel.budget_ms = match self.cfg.mode {
            HybridMode::Deadline { budget_ms } => budget_ms,
            HybridMode::Fixed => 0.0,
        };
        let (self_id, victim_id, prev) = (inp.self_id, inp.victim_id, inp.prev);
        let (Some(me), Some(victim)) = (self.world.get_tee(self_id), self.world.get_tee(victim_id)) else {
            return (prev, tel);
        };
        if !me.alive || !victim.alive {
            return (prev, tel);
        }
        self.planner.opp_seed = js::opp_seed_next(self.planner.opp_seed);
        let cfg = self.cfg.clone();
        let n = cfg.planner.steps as usize;
        let track_aim = cfg.planner.track_aim;
        let aim_at = js::atan2(victim.pos.y - me.pos.y, victim.pos.x - me.pos.x);
        let aim_base = if track_aim { aim_at } else { 0.0 };
        let (field, unfreeze) = self.planner.hazard_fields(self.world.collision());

        // ---- who is around --------------------------------------------------------------------
        let all = self.world.all_tees();
        let radius = cfg
            .threat_radius_px
            .unwrap_or_else(|| threat_radius(cfg.decision_ticks, inp.lag_ticks));
        let dist_me = |t: &TeeState| vdistance(t.pos, me.pos);
        let mut threats: Vec<TeeState> = if cfg.threat_model {
            all.iter()
                .filter(|t| {
                    t.id != self_id
                        && t.id != victim_id
                        && t.alive
                        && !t.frozen
                        && dist_me(t) <= radius
                        && !self.is_spared(t)
                })
                .copied()
                .collect()
        } else {
            Vec::new()
        };
        // Most dangerous first (review F2): a hook flying at us, then hammer reach, then whoever closes on us
        // fastest; distance only breaks ties. The local-tee cap and the reply models keep the front of this list.
        threats.sort_by_key(|t| threat_rank(t, &me));
        threats.truncate(MAX_THREATS);
        // Everybody whose hook holds us stays in the simulation whatever the cap says (their pull is in every rollout).
        let hookers: Vec<i32> = all
            .iter()
            .filter(|t| t.id != self_id && t.alive && t.hooked_player == self_id && t.hook_state == HOOK_GRABBED)
            .map(|t| t.id)
            .collect();
        let hooked_by = all
            .iter()
            .find(|t| {
                t.id != self_id && t.alive && !t.frozen && t.hooked_player == self_id && t.hook_state == HOOK_GRABBED
            })
            .copied();
        let mut frozen_by: Vec<&TeeState> = all
            .iter()
            .filter(|t| {
                t.id != self_id
                    && t.id != victim_id
                    && t.alive
                    && t.frozen
                    && dist_me(t) <= BYSTANDER_PX
                    && !self.is_spared(t)
            })
            .collect();
        frozen_by.sort_by(|a, b| dist_me(a).total_cmp(&dist_me(b)).then(a.id.cmp(&b.id)));
        let victim_in_radius = !victim.frozen && dist_me(&victim) <= radius;
        let threats_in_radius = threats.len() as u32;
        // ---- local tees (task 3.5b, F2): the rollouts step only the tees that can matter -------
        // A crowd of 12 tees must not cost 12 tees per physics tick: what is simulated is us, the
        // victim, whoever hooks us, the free tees in the threat radius (nearest first) and the
        // nearest frozen body, at most `max_sim_tees` of them. The rest are removed from the decision world
        // (and so from the snapshot every rollout restores); the caller's world is never touched.
        // Spared tees (friends, AFK) the caller left in the world stay in the rollouts as bodies (tee-tee
        // collision; the bot has already chosen them: contact range and the lane ahead, review F4), at most three,
        // outside the cap: they are obstacles, not opponents.
        let mut spared_bodies: Vec<&TeeState> = all
            .iter()
            .filter(|t| t.id != self_id && t.id != victim_id && t.alive && self.is_spared(t))
            .collect();
        spared_bodies.sort_by(|a, b| dist_me(a).total_cmp(&dist_me(b)).then(a.id.cmp(&b.id)));
        let spared_bodies: Vec<i32> = spared_bodies.iter().take(3).map(|t| t.id).collect();
        let (local_ids, dropped) = local_tees(
            cfg.max_sim_tees,
            self_id,
            victim_id,
            hookers
                .iter()
                .copied()
                .chain((me.hooked_player >= 0).then_some(me.hooked_player)),
            &spared_bodies,
            &threats,
            &frozen_by,
            &all,
        );
        threats.retain(|t| local_ids.contains(&t.id));
        frozen_by.retain(|t| local_ids.contains(&t.id));
        for id in &dropped {
            self.world.remove_tee(*id);
        }
        tel.sim_tees = self.world.all_tees().len() as u32;
        tel.dropped_tees = dropped.len() as u32;
        if let Some(m) = &self.meter {
            m.set_scale(tel.sim_tees as usize);
        }
        let mut danger = Danger {
            opponents_in_radius: u32::from(victim_in_radius) + threats_in_radius,
            near_freeze: !me.frozen && freeze_gap_px(self.world.collision(), me.pos.x, me.pos.y) < EDGE_GAP_PX,
            hooked_by: hooked_by.map(|t| t.id),
            probe_self_out: false,
        };
        tel.threat_ids = threats.iter().map(|t| t.id).collect();
        let victim_input = crate::brains::enemy_input_from_tee(&victim);
        let threat_set = (!threats.is_empty()).then(|| ThreatSet {
            ids: threats.iter().map(|t| t.id).collect(),
            inputs: threats.iter().map(crate::brains::enemy_input_from_tee).collect(),
            react_mask: 0,
            weight: cfg.threat_weight,
            hook_targets: cfg.hook_threats,
        });
        self.planner.threats.clone_from(&threat_set);
        self.planner.set_frozen_bystanders(
            frozen_by.iter().map(|t| t.pos).collect(),
            frozen_by.iter().map(|t| t.vel).collect(),
        );
        let (ps, pv) = self.planner.spares_mut();
        ps.clone_from(&self.spares);
        pv.clone_from(&self.spare_vels);
        self.planner.set_travel_goal(self.travel_goal);

        // What each modelled opponent did since the last decision, against what "hold" and "react"
        // predicted for it then (see `ReactBelief`).
        for (id, hold, react) in std::mem::take(&mut self.prev_predictions) {
            if let Some(t) = all.iter().find(|t| t.id == id && t.alive && !t.frozen) {
                let observed = crate::brains::enemy_input_from_tee(t);
                self.beliefs.entry(id).or_default().observe(&observed, &hold, &react);
            }
        }

        // Model combinations. Bit 0 = the victim reacts, bit `i + 1` = threat `i` reacts; only the
        // opponents that can act on us matter (a frozen or far victim's reaction changes nothing, so
        // it would only double the stage-2 cost). `masks[0] = 0` is the cheap model (everybody
        // holds), the last mask is "everybody reacts"; a job names its combination by index.
        let mut relevant: Vec<u32> = Vec::with_capacity(1 + threats.len());
        if victim_in_radius {
            relevant.push(0);
        }
        relevant.extend((0..threats.len() as u32).map(|i| i + 1));
        self.masks.clear();
        // `max_combos = 1` means exactly one combination (the cheap model), whatever the opponents.
        if cfg.robust.enabled && cfg.robust.max_combos >= 2 && relevant.len() <= cfg.robust.max_relevant {
            let bit = |j: usize| 1u32 << relevant[j];
            let all: u32 = (0..relevant.len()).map(bit).fold(0, |a, b| a | b);
            if relevant.len() <= 2 && cfg.robust.max_combos >= (1 << relevant.len()) {
                // Every subset: 1, 2 or 4 combinations.
                for subset in 0u32..(1u32 << relevant.len()) {
                    let mask = (0..relevant.len())
                        .filter(|j| (subset >> j) & 1 == 1)
                        .fold(0, |a, j| a | bit(j));
                    self.masks.push(mask);
                }
            } else if cfg.robust.max_combos <= 2 || relevant.len() < 3 {
                // The two extremes only: everybody holds / everybody reacts.
                self.masks.extend([0, all]);
            } else {
                // Three or more relevant opponents would be 8+ combinations per plan, more than a
                // 4 ms budget can pay for: the two extremes (everybody holds / everybody reacts)
                // plus each of the two most important opponents reacting alone.
                self.masks.extend([0, all, bit(0), bit(1)]);
            }
        } else if cfg.robust.enabled && cfg.robust.max_combos >= 2 && cfg.robust.crowd_stage && !relevant.is_empty() {
            // More opponents than the full stage affords: the two extremes only.
            let all: u32 = relevant.iter().fold(0, |a, &b| a | (1u32 << b));
            self.masks.extend([0, all]);
        } else {
            self.masks.push(0);
        }
        let crowd = cfg.robust.crowd_stage && relevant.len() > cfg.robust.max_relevant;
        let stage2_m = if crowd {
            cfg.robust.crowd_top_m
        } else {
            cfg.robust.top_m
        };
        let ncombos = self.masks.len();
        tel.combos = ncombos as u32;
        // Probability of each combination from the per-opponent beliefs, and how much the worst
        // case counts: it fades when the opponents look like they hold (an idle victim must not
        // make us play scared).
        let belief_of = |bit: u32| -> f64 {
            let id = if bit == 0 {
                victim_id
            } else {
                threats[bit as usize - 1].id
            };
            self.beliefs.get(&id).copied().unwrap_or_default().p
        };
        self.mask_weights.clear();
        for &mask in &self.masks {
            let p = relevant.iter().fold(1.0, |acc, &bit| {
                let w = belief_of(bit);
                acc * if (mask >> bit) & 1 == 1 { w } else { 1.0 - w }
            });
            self.mask_weights.push(p);
        }
        let mean_belief = if relevant.is_empty() {
            0.5
        } else {
            relevant.iter().map(|&b| belief_of(b)).sum::<f64>() / relevant.len() as f64
        };
        tel.react_belief = mean_belief;
        let lambda_eff = if cfg.robust.belief_lambda {
            cfg.robust.lambda * (2.0 * mean_belief).min(1.0)
        } else {
            cfg.robust.lambda
        };

        // ---- decision snapshot for the workers -----------------------------------------------
        self.world.save_state_into(&mut self.saved);
        let opp_seed = self.planner.opp_seed;
        {
            let (saved, ts) = (&*self.saved, threat_set.clone());
            let (f, u) = (Arc::clone(&field), Arc::clone(&unfreeze));
            let (bp, bv) = (
                frozen_by.iter().map(|t| t.pos).collect::<Vec<_>>(),
                frozen_by.iter().map(|t| t.vel).collect::<Vec<_>>(),
            );
            let (spares, spare_vels, goal) = (&self.spares, &self.spare_vels, self.travel_goal);
            self.engine.with_ctx(|c| {
                c.spares.clone_from(spares);
                c.spare_vels.clone_from(spare_vels);
                c.travel_goal = goal;
                c.saved.assign_from(saved);
                c.self_id = self_id;
                c.victim_id = victim_id;
                c.prev = prev;
                c.victim_input = victim_input;
                c.victim_plan.clear();
                c.opp_seed = opp_seed;
                c.field = f;
                c.unfreeze = u;
                c.frozen_bystanders = bp;
                c.frozen_bystander_vels = bv;
                c.threats = ts;
                c.self_freeze_bias = 1.0;
            });
        }

        // Two-world search: a second snapshot of the same decision without the threats. The pool is
        // scored in it (cheap: two tees, many candidates) and only the best few are re-scored in the
        // full world with the threats and their modelled replies.
        let two = cfg.two_world && threat_set.is_some();
        self.lens = Lens::Full;
        self.lens_threats = None;
        if two {
            for t in &threats {
                self.world.remove_tee(t.id);
            }
            self.world.save_state_into(&mut self.saved_red);
            self.world.restore_state(&self.saved);
            self.lens_threats = threat_set.clone();
        }

        // ---- candidate pool ------------------------------------------------------------------
        let mut seen: HashSet<Vec<i32>> = HashSet::new();
        let mut push = |cands: &mut Vec<Cand>, tel: &mut DecisionTelemetry, plan: Vec<PlanStep>, src: Source| {
            if plan.len() == n && seen.insert(signature(&plan)) {
                tel.generated[src.kind()] += 1;
                cands.push(Cand::new(plan, src));
            }
        };
        let mut warm_c: Vec<Cand> = Vec::new();
        let have_warm = if let Some(warm) = self.planner.warm.clone()
            && warm.len() == n
        {
            let shifted: Vec<PlanStep> = warm[1..]
                .iter()
                .copied()
                .chain(std::iter::once(*warm.last().unwrap()))
                .collect();
            push(&mut warm_c, &mut tel, shifted, Source::Warm);
            true
        } else {
            false
        };

        // ---- the opponent model (task 3.7b): the victim's predicted plan replaces "it holds its input" ----
        // A duel only: with other free opponents in the radius the decision is a defence against several and the model's share of the
        // budget (and its two-tee view of a many-tee fight) costs more than it gives (E-017 section 4: 1vN and crowds lost 4 points with it).
        // Not for a victim that has done nothing for a while (neutral direction, no hook out: an idle or camping opponent): "it keeps
        // its input" is right for it, and a model that expects an attack makes us play away from a victim that never comes.
        self.passive = if self.passive.0 == victim_id && victim.direction == 0 && victim.hook_state <= 0 {
            (victim_id, self.passive.1.saturating_add(1))
        } else {
            (victim_id, u32::from(victim.direction == 0 && victim.hook_state <= 0))
        };
        self.mirror_inputs.clear();
        // The clock is read only when the model runs: a step clock (tests) advances on every read.
        let (mut t_mirror, mut mirror_ms) = (0.0, 0.0);
        if cfg.mirror
            && victim_in_radius
            && threats_in_radius == 0
            && !victim.frozen
            && !me.frozen
            && self.passive.1 < PASSIVE_DECISIONS
        {
            t_mirror = now(clock);
            // Under a cap on a real (or injected) clock the model is time-limited: what the shield, the search's minimum and the last
            // proposal leave of the cap, at most `MIRROR_MAX_MS`. Never on the work clock: the arena's games stay a pure function of the
            // state (its decisions of 2 tees take 1.35 ms of work on average, but some run over 2 ms).
            let reserve = cfg.shield_reserve_ms_per_tee * js::max(1.0, f64::from(tel.sim_tees));
            let deadline = if timed && self.meter.is_none() {
                cfg.decision_cap_ms.map(|cap| {
                    t_mirror
                        + js::min(
                            MIRROR_MAX_MS,
                            js::max(cap - reserve - MIN_SEARCH_MS - self.last_proposal_ms, 0.0),
                        )
                })
            } else {
                None
            };
            // The model's world is the two tees alone: its ticks cost two tee-ticks each (charged to the meter inside), whatever the
            // decision world's tee count (a frozen body or a spared tee may be in it); `work.mirror` counts them in ticks of the
            // decision world.
            let (ticks, cut) =
                self.mirror_predict(clock, deadline, self_id, victim_id, &me, &victim, &field, &unfreeze);
            tel.work.mirror = (2 * ticks).div_ceil(u64::from(tel.sim_tees.max(1)));
            tel.mirror_cut = cut;
            if let Some(first) = self.mirror_inputs.first() {
                tel.mirror_first = Some(*first);
                let plan_inputs = &self.mirror_inputs;
                self.engine.with_ctx(|c| c.victim_plan.clone_from(plan_inputs));
            }
            mirror_ms = now(clock) - t_mirror;
            tel.mirror_ms = mirror_ms;
        }

        let t_prop = now(clock);
        let mut props: Vec<Vec<PlanStep>> = Vec::new();
        if cfg.proposals > 0 {
            let pctx = ProposeCtx {
                obs: inp.obs,
                world: &self.world,
                saved: &self.saved,
                map: &self.map,
                self_id,
                victim_id,
                steps: n,
                step_ticks: &self.planner.step_ticks,
                aim_at,
                track_aim,
                prev,
                k: cfg.proposals,
            };
            self.proposer.propose(&pctx, &mut props);
        }
        let pt = self.proposer.work_ticks();
        tel.work.proposal = pt - self.last_prop_ticks;
        self.last_prop_ticks = pt;
        // A proposer's time counts against the decision cap (task 3.7a). On the work clock its cost is
        // charged to the meter first: the simulated ticks of a scripted proposer, and the nominal
        // tee-tick cost of one that does not simulate physics (the fly, `Proposer::work_units`).
        let proposer_costs = self.proposer.costs_time();
        // Charged (and reported) only where something was spent: on the work clock, in a decision that asked the
        // proposer at all. On the wall clock the time is measured, not priced.
        if proposer_costs && self.meter.is_some() && cfg.proposals > 0 {
            tel.work.proposal_units = self.proposer.work_units();
        }
        if let Some(m) = &self.meter {
            m.add(tel.work.proposal);
            m.add_units(tel.work.proposal_units);
        }
        tel.proposal_ms = now(clock) - t_prop;
        // The search budget (D-042: 4 ms) starts here, after the proposals, and covers the candidate
        // generation below. What the proposals took (`proposal_ms`) is taken off the decision cap
        // (`HybridConfig::proposal_in_cap`), so proposals + search + shield fit the cap together.
        let t_search = now(clock);
        let proposal_counts = cfg.proposal_in_cap && proposer_costs;
        let counted_proposal_ms = if proposal_counts { tel.proposal_ms } else { 0.0 };
        self.last_proposal_ms = counted_proposal_ms;
        // Where the decision's time began as far as the extension of D-042 is concerned.
        let t_decision = if tel.work.mirror > 0 {
            t_mirror
        } else if proposal_counts {
            t_prop
        } else {
            t_search
        };
        let mut prop_c: Vec<Cand> = Vec::new();
        for p in props.into_iter().take(cfg.proposals) {
            push(&mut prop_c, &mut tel, sanitize(p, n), Source::Proposal);
        }

        let mut book_c: Vec<Cand> = Vec::new();
        for p in self
            .planner
            .seed_plans(&*self.world, self_id, victim_id, &field, aim_at)
        {
            push(&mut book_c, &mut tel, p, Source::Book);
        }

        let mut throw_c: Vec<Cand> = Vec::new();
        let mut wall_c: Vec<Cand> = Vec::new();
        {
            let at = if track_aim { 0.0 } else { aim_at };
            let sit = ThrowSituation {
                separation: vdistance(me.pos, victim.pos),
                enemy_hazard_nearness: hazard_nearness(&field, victim.pos.x, victim.pos.y),
                me_frozen: me.frozen,
                enemy_frozen: victim.frozen,
                enemy_alive: victim.alive,
            };
            let frozen_case =
                cfg.planner.frozen_throw > 0 && frozen_throw_worth_trying(&sit) && victim.freeze_ticks_left >= 30;
            let lines = if frozen_case {
                frozen_throw_lines(cfg.planner.steps, at)
            } else if cfg.planner.freeze_throw > 0 && throw_worth_trying(&sit) {
                throw_lines(cfg.planner.steps, at)
            } else {
                Vec::new()
            };
            for p in lines.into_iter().take(cfg.throw_cap) {
                push(&mut throw_c, &mut tel, p, Source::Throw);
            }
            // Task 3.9: the wayblock guard's wall swings / air chains toward a wall beside us, for a frozen victim.
            if cfg.wall_throws && frozen_case {
                let wall_dir = wall_side(self.world.collision(), me.pos);
                for p in self
                    .planner
                    .wall_throw_lines(&*self.world, &me, at, wall_dir, cfg.planner.air_chain)
                {
                    push(&mut wall_c, &mut tel, p, Source::Throw);
                }
                tel.wall_cands = wall_c.len() as u32;
            }
        }

        let mut generated = Generated::default();
        let rays_before = self.anchors.rays;
        if cfg.techniques {
            let anchors = self.anchors.select(self.world.collision(), me.pos, cfg.anchors);
            let me_strong = strength(self.world.inner(), self_id, victim_id);
            let tc = TechCtx {
                col: self.world.collision(),
                field: &field,
                me: &me,
                victim: &victim,
                threats: &threats,
                hooked_by: hooked_by.as_ref(),
                steps: n,
                aim_at,
                me_strong,
            };
            generated = generate(
                &tc,
                &anchors,
                &TechCaps {
                    generic_escape: danger.flagged(),
                    frozen_offence: cfg.finish_families,
                    ..TechCaps::default()
                },
            );
        }
        tel.work.rays = self.anchors.rays - rays_before;
        if let Some(m) = &self.meter {
            m.add_units(2 * tel.work.rays);
        }
        let mut def_c: Vec<Cand> = Vec::new();
        for tp in &generated.defence {
            push(&mut def_c, &mut tel, tp.plan.clone(), Source::Tech(tp.tech));
        }
        let mut off_c: Vec<Cand> = Vec::new();
        for tp in &generated.offence {
            push(&mut off_c, &mut tel, tp.plan.clone(), Source::Tech(tp.tech));
        }

        // ---- the probe: does the warm plan (or standing still) end with us out? ---------------
        let sim_tees = js::max(1.0, f64::from(tel.sim_tees));
        let shield_reserve = cfg.shield_reserve_ms_per_tee * sim_tees;
        let budget = match cfg.mode {
            // The decision cap (search + shield) shortens the search when the shield's reserve
            // would not fit under it (many tees); never below `MIN_SEARCH_MS`.
            HybridMode::Deadline { budget_ms } => {
                let budget_ms = match cfg.decision_cap_ms {
                    Some(cap) => js::min(
                        budget_ms,
                        js::max(cap - shield_reserve - counted_proposal_ms - mirror_ms, MIN_SEARCH_MS),
                    ),
                    None => budget_ms,
                };
                // Frozen, the game ignores our movement inputs: only a cheap search keeps the warm
                // plan alive for the moment we thaw (review round 1, F4).
                if me.frozen {
                    js::min(budget_ms, FROZEN_SEARCH_MS)
                } else {
                    budget_ms
                }
            }
            HybridMode::Fixed => 0.0,
        };
        tel.budget_ms = budget;
        let two_world = cfg.two_world && threat_set.is_some();
        let static_stage1_end = if ncombos > 1 || two_world {
            t_search + budget * (1.0 - cfg.stage2_fraction)
        } else {
            t_search + budget
        };
        let search_end = t_search + budget;
        // Stage 2 usually needs less than its share (3-4 plans x one extra combination): with
        // `stage2_dynamic` stage 1 keeps going until what stage 2 will cost, estimated from the average
        // rollout so far (+20%), is all that is left of the budget. Never earlier than the static share.
        let stage2_jobs = if two_world {
            (stage2_m + 3) * ncombos
        } else if ncombos > 1 {
            (stage2_m + 1) * (ncombos - 1)
        } else {
            0
        };
        // A full-world rollout costs more than a reduced one: scale the estimate by the tees' ratio.
        let full_ratio = if two_world {
            f64::from(tel.sim_tees) / (f64::from(tel.sim_tees) - threats.len() as f64).max(1.0)
        } else {
            1.0
        };
        let dyn_stage2 = cfg.stage2_dynamic && stage2_jobs > 0 && relevant.len() >= cfg.stage2_dynamic_min_relevant;
        let stage1_end_at = |now_ms: f64, rollouts: u32| -> f64 {
            if !dyn_stage2 || rollouts < 3 {
                return static_stage1_end;
            }
            let avg = (now_ms - t_search) / f64::from(rollouts);
            let need = stage2_jobs as f64 * avg * 1.2 * full_ratio;
            js::min(search_end, js::max(static_stage1_end, search_end - need))
        };
        let mut stage1_end = static_stage1_end;
        // Wall clock with a pool: the pool takes a few candidates at a time. The work clock always goes one
        // candidate at a time (the helpers only speculate), which is what makes it independent of the
        // thread count.
        let chunk = if !timed {
            usize::MAX
        } else if self.engine.workers() > 1 && self.meter.is_none() {
            2 * self.engine.workers()
        } else {
            1
        };
        let mut cands: Vec<Cand> = Vec::new();
        let mut have_any = false;
        let mut out_of_time = false;
        let prune_on = cfg.prune.enabled && !me.frozen;
        let prune_steps = cfg.prune.steps as u32;
        let mut pre_hist: Vec<f64> = Vec::new();
        let mut ticks1 = 0u64;
        let mut roll1 = 0u32;

        let probe_plan = if have_warm {
            None
        } else {
            Some(vec![
                PlanStep {
                    dir: 0,
                    jump: 0,
                    hook: 0,
                    fire: 0,
                    aim: 0.0,
                };
                n
            ])
        };
        {
            // The probe = stage-1 scoring of the warm plan under "all hold" plus the all-react
            // combination; both are reused by the later stages.
            let mut probe = std::mem::take(&mut warm_c);
            if probe.is_empty()
                && let Some(p) = probe_plan
            {
                probe.push(Cand::new(p, Source::Warm));
                tel.generated[Source::Warm.kind()] += 1;
            }
            let base = cands.len();
            cands.extend(probe);
            if cands.len() > base {
                let mut jobs: Vec<(usize, u32)> = vec![(base, 0)];
                if ncombos > 1 {
                    jobs.push((base, (ncombos - 1) as u32));
                }
                self.prefetch_jobs(clock, &cands, &jobs, 0);
                let cut = self.run_jobs(clock, &mut cands, &jobs, None, &mut ticks1, &mut roll1);
                debug_assert!(!cut);
                have_any = true;
                if cands[base].res[..ncombos].iter().flatten().any(|r| r.self_out > 0) {
                    danger.probe_self_out = true;
                }
            }
        }
        tel.danger = danger;
        if two {
            // The pool is scored in the reduced world from here on; the warm plan joins it.
            self.set_lens(Lens::Reduced);
            if !cands.is_empty() {
                let _ = self.run_jobs(clock, &mut cands, &[(0, 0)], None, &mut ticks1, &mut roll1);
            }
        }

        // ---- stage 1: order the pool by danger, score under the cheap model --------------------
        // The situational defensive techniques (T9, T10, T12, T13, T14, T15, T17, ...) exist only
        // when their geometric trigger fired, so there are few of them and they go first: a tee
        // falling over freeze without a jump must try the wall hook before anything else, however
        // small the budget. The generic wall/ceiling escape (danger without a cause) waits behind
        // the book and the attack unless the probe says we are doomed or we are hooked; putting
        // it ahead every time starved the attack (danger flags alone are true in most decisions
        // of a fight).
        let urgent = danger.escape_first() || danger.hooked_by.is_some();
        let (def_first, def_rest): (Vec<Cand>, Vec<Cand>) = def_c
            .into_iter()
            .partition(|c| urgent || !matches!(c.src, Source::Tech(Tech::AnchorEscape)));
        let mut order: Vec<Cand> = Vec::new();
        order.extend(def_first);
        order.extend(prop_c);
        order.extend(book_c);
        order.extend(off_c);
        order.extend(throw_c);
        order.extend(wall_c);
        order.extend(def_rest);
        let first_pool = cands.len();
        cands.extend(order);
        let mut next = first_pool;
        let mut spec_end = next;
        while next < cands.len() {
            if timed {
                stage1_end = stage1_end_at(now(clock), roll1);
            }
            if timed && have_any && now(clock) >= stage1_end {
                out_of_time = true;
                break;
            }
            if next >= spec_end && self.engine.speculating() {
                // Work clock: the helpers score the next few pool candidates (and their pruning pre-scores) now.
                spec_end = (next + self.spec_window()).min(cands.len());
                let (lo, hi) = (next, spec_end);
                let prescreen_steps = if prune_on { prune_steps } else { 0 };
                self.prefetch(clock, |b, masks| {
                    for c in &cands[lo..hi] {
                        let pi = b.push_plan(&c.plan);
                        b.push_job(pi, masks[0]);
                        if prescreen_steps > 0 && c.prunable() {
                            b.push_job_short(pi, masks[0], prescreen_steps);
                        }
                    }
                });
            }
            let end = next.saturating_add(chunk).min(cands.len());
            let dl = (timed && have_any).then_some(stage1_end);
            let mut full: Vec<usize> = (next..end).collect();
            if prune_on {
                let idx: Vec<usize> = full.iter().copied().filter(|&i| cands[i].prunable()).collect();
                self.prescreen(clock, &mut cands, &idx, prune_steps, dl, &mut ticks1);
                for &i in &idx {
                    if let Some(p) = cands[i].pre
                        && !pre_keep(&mut pre_hist, p, &cfg.prune)
                    {
                        full.retain(|&x| x != i);
                        tel.pruned += 1;
                    }
                }
            }
            let jobs: Vec<(usize, u32)> = full.iter().map(|&i| (i, 0)).collect();
            let cut = self.run_jobs(clock, &mut cands, &jobs, dl, &mut ticks1, &mut roll1);
            have_any |= cands[next..end].iter().any(|c| c.cheap().is_some());
            next = end;
            if cut {
                out_of_time = true;
                break;
            }
        }

        // ---- CEM ------------------------------------------------------------------------------
        let mut dist = self.planner.build_dist(aim_base);
        let seed_scored = |cands: &[Cand]| -> Vec<(Vec<PlanStep>, f64)> {
            cands
                .iter()
                .filter(|c| matches!(c.src, Source::Book | Source::Proposal | Source::Throw))
                .filter_map(|c| c.cheap().map(|s| (c.plan.clone(), s)))
                .collect()
        };
        let mut scored = seed_scored(&cands);
        let cem_iteration = |this: &mut HybridSearch,
                             it: i32,
                             cands: &mut Vec<Cand>,
                             scored: &mut Vec<(Vec<PlanStep>, f64)>,
                             dist: &mut Vec<crate::planner::StepDist>,
                             tel: &mut DecisionTelemetry,
                             clock: &dyn Clock,
                             dl: f64,
                             ticks: &mut u64,
                             rolls: &mut u32,
                             seen: &mut HashSet<Vec<i32>>,
                             pre_hist: &mut Vec<f64>|
         -> bool {
            if it > 0 {
                scored.clear();
            }
            let pop = cfg.planner.population.max(0) as usize;
            let mut done = 0usize;
            let mut cut_any = false;
            let mut spec_done = 0usize;
            while done < pop {
                if timed && now(clock) >= dl {
                    cut_any = true;
                    break;
                }
                if done >= spec_done && this.engine.speculating() {
                    // Work clock: the samples to come are already determined by the RNG state; the helpers
                    // score the next few of them (a copy of the RNG draws them, the real one is untouched).
                    let want = this.spec_window().min(pop - done);
                    let preview = this.planner.preview_plans(dist, want);
                    let prescreen_steps = if prune_on { prune_steps } else { 0 };
                    this.prefetch(clock, |b, masks| {
                        for plan in &preview {
                            let pi = b.push_plan(plan);
                            b.push_job(pi, masks[0]);
                            if prescreen_steps > 0 {
                                b.push_job_short(pi, masks[0], prescreen_steps);
                            }
                        }
                    });
                    spec_done = done + want;
                }
                let k = chunk.min(pop - done);
                let base = cands.len();
                for _ in 0..k {
                    let plan = this.planner.sample_plan(dist);
                    tel.generated[Source::Cem.kind()] += 1;
                    seen.insert(signature(&plan));
                    cands.push(Cand::new(plan, Source::Cem));
                }
                let dlo = timed.then_some(dl);
                let mut full: Vec<usize> = (base..base + k).collect();
                if prune_on {
                    let idx = full.clone();
                    this.prescreen(clock, cands, &idx, prune_steps, dlo, ticks);
                    for &i in &idx {
                        if let Some(p) = cands[i].pre
                            && !pre_keep(pre_hist, p, &cfg.prune)
                        {
                            full.retain(|&x| x != i);
                            tel.pruned += 1;
                        }
                    }
                }
                let jobs: Vec<(usize, u32)> = full.iter().map(|&i| (i, 0)).collect();
                let cut = this.run_jobs(clock, cands, &jobs, dlo, ticks, rolls);
                for c in &cands[base..] {
                    if let Some(s) = c.cheap() {
                        scored.push((c.plan.clone(), s));
                    }
                }
                done += k;
                if cut {
                    cut_any = true;
                    break;
                }
            }
            scored.sort_by(|a, b| score_desc(a.1, b.1));
            let elite_n = cfg.planner.elite.max(0) as usize;
            let elites: Vec<Vec<PlanStep>> = scored.iter().take(elite_n).map(|(p, _)| p.clone()).collect();
            this.planner.refit(dist, &elites);
            cut_any
        };
        if !(timed && (out_of_time || now(clock) >= stage1_end)) {
            for it in 0..cfg.planner.iterations {
                if timed {
                    stage1_end = stage1_end_at(now(clock), roll1);
                }
                if timed && now(clock) >= stage1_end {
                    out_of_time = true;
                    break;
                }
                let cut = cem_iteration(
                    self,
                    it,
                    &mut cands,
                    &mut scored,
                    &mut dist,
                    &mut tel,
                    clock,
                    stage1_end,
                    &mut ticks1,
                    &mut roll1,
                    &mut seen,
                    &mut pre_hist,
                );
                if cut {
                    out_of_time = true;
                    break;
                }
            }
        } else {
            out_of_time = true;
        }

        // ---- polish (task 3.9, `polishRope`): hold-the-hook variants of the best plan so far ----------
        // Not bounded by the end of stage 1: CEM keeps going until it, so on a deadline the polish (three rollouts at most) would
        // never run (found by the screening of E-020: not one game differed). It takes its few rollouts from stage 2's share and
        // stops at the end of the search budget.
        if cfg.polish && !me.frozen && !(timed && now(clock) >= search_end) {
            let mut best_i: Option<usize> = None;
            for (i, c) in cands.iter().enumerate() {
                if let Some(score) = c.cheap()
                    && best_i.is_none_or(|b| score > cands[b].cheap().unwrap_or(f64::NEG_INFINITY))
                {
                    best_i = Some(i);
                }
            }
            let mut variants: Vec<Vec<PlanStep>> = Vec::new();
            if let Some(bi) = best_i
                && self
                    .planner
                    .polish_variants(&*self.world, self_id, victim_id, prev, &cands[bi].plan, &mut variants)
            {
                let base = cands.len();
                for plan in variants {
                    if seen.insert(signature(&plan)) {
                        tel.generated[Source::Cem.kind()] += 1;
                        tel.polished += 1;
                        cands.push(Cand::new(plan, Source::Cem));
                    }
                }
                let jobs: Vec<(usize, u32)> = (base..cands.len()).map(|i| (i, 0)).collect();
                let cut = self.run_jobs(
                    clock,
                    &mut cands,
                    &jobs,
                    timed.then_some(search_end),
                    &mut ticks1,
                    &mut roll1,
                );
                out_of_time |= cut;
            }
        }
        tel.work.stage1 = ticks1;
        tel.work.rollouts_stage1 = roll1;

        // ---- stage 2: robust re-scoring of the best few ---------------------------------------
        let mut ticks2 = 0u64;
        let mut roll2 = 0u32;
        let mut top = self.top_indices(&cands, stage2_m, prev.direction);
        if two_world {
            self.set_lens(Lens::Full);
            self.add_protected(&cands, &mut top, danger.flagged());
        }
        // The combinations to score in the full world: all of them in the two-world search (the reduced
        // score is not one of them), else everything but the stage-1 "everybody holds".
        let first_combo = u32::from(!two_world);
        if ncombos > 1 || two_world {
            let mut jobs: Vec<(usize, u32)> = Vec::new();
            for &ci in &top {
                for combo in first_combo..ncombos as u32 {
                    if cands[ci].res[combo as usize].is_none() {
                        jobs.push((ci, combo));
                    }
                }
            }
            let dl = timed.then_some(search_end);
            self.prefetch_jobs(clock, &cands, &jobs, 0);
            let cut = self.run_jobs(clock, &mut cands, &jobs, dl, &mut ticks2, &mut roll2);
            out_of_time |= cut;
        }
        tel.work.stage2 = ticks2;
        tel.work.rollouts_stage2 = roll2;

        // ---- choose ---------------------------------------------------------------------------
        let lambda = lambda_eff;
        let mut pick = choose(
            &cands,
            &top,
            ncombos,
            &self.mask_weights,
            lambda,
            cfg.robust.mode,
            prev.direction,
            cfg.planner.flip_margin,
            cfg.warm_bonus,
            cfg.warm_fire_only,
        );

        // ---- adaptive extension (D-042) -------------------------------------------------------
        let unsafe_now =
            |cands: &[Cand], pick: Option<usize>| pick.is_some_and(|i| cands[i].worst_self_out(ncombos.max(1)) > 0);
        if timed && cfg.adaptive.enabled && !me.frozen && danger.flagged() && unsafe_now(&cands, pick) {
            tel.extended = true;
            let ext_end = t_decision + cfg.adaptive.max_total_ms - shield_reserve;
            let mut ticks_e = 0u64;
            let mut roll_e = 0u32;
            // First whatever the pool still holds unscored (defensive techniques come first in
            // a dangerous pool), then more CEM.
            let pending: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].cheap().is_none()).collect();
            if two_world {
                self.set_lens(Lens::Reduced);
            }
            let mut spec_end = 0usize;
            for (gi, group) in pending.chunks(chunk.max(1)).enumerate() {
                if now(clock) >= ext_end {
                    break;
                }
                let jobs: Vec<(usize, u32)> = group.iter().map(|&i| (i, 0)).collect();
                if gi >= spec_end && self.engine.speculating() {
                    let ahead: Vec<(usize, u32)> = pending
                        .iter()
                        .skip(gi * chunk.max(1))
                        .take(self.spec_window())
                        .map(|&i| (i, 0))
                        .collect();
                    self.prefetch_jobs(clock, &cands, &ahead, 0);
                    spec_end = gi + ahead.len().div_ceil(chunk.max(1));
                }
                self.run_jobs(clock, &mut cands, &jobs, Some(ext_end), &mut ticks_e, &mut roll_e);
            }
            let mut rounds = 0;
            loop {
                top = self.top_indices(&cands, stage2_m, prev.direction);
                if two_world {
                    self.set_lens(Lens::Full);
                    self.add_protected(&cands, &mut top, true);
                }
                if ncombos > 1 || two_world {
                    let mut jobs: Vec<(usize, u32)> = Vec::new();
                    for &ci in &top {
                        for combo in first_combo..ncombos as u32 {
                            if cands[ci].res[combo as usize].is_none() {
                                jobs.push((ci, combo));
                            }
                        }
                    }
                    self.prefetch_jobs(clock, &cands, &jobs, 0);
                    self.run_jobs(clock, &mut cands, &jobs, Some(ext_end), &mut ticks_e, &mut roll_e);
                }
                pick = choose(
                    &cands,
                    &top,
                    ncombos,
                    &self.mask_weights,
                    lambda,
                    cfg.robust.mode,
                    prev.direction,
                    cfg.planner.flip_margin,
                    cfg.warm_bonus,
                    cfg.warm_fire_only,
                );
                if !unsafe_now(&cands, pick) || now(clock) >= ext_end || rounds >= 3 {
                    break;
                }
                rounds += 1;
                if two_world {
                    self.set_lens(Lens::Reduced);
                }
                let cut = cem_iteration(
                    self,
                    1,
                    &mut cands,
                    &mut scored,
                    &mut dist,
                    &mut tel,
                    clock,
                    ext_end,
                    &mut ticks_e,
                    &mut roll_e,
                    &mut seen,
                    &mut pre_hist,
                );
                if cut {
                    top = self.top_indices(&cands, stage2_m, prev.direction);
                    if two_world {
                        self.set_lens(Lens::Full);
                        self.add_protected(&cands, &mut top, true);
                    }
                    if ncombos > 1 || two_world {
                        let mut jobs: Vec<(usize, u32)> = Vec::new();
                        for &ci in &top {
                            for combo in first_combo..ncombos as u32 {
                                if cands[ci].res[combo as usize].is_none() {
                                    jobs.push((ci, combo));
                                }
                            }
                        }
                        self.prefetch_jobs(clock, &cands, &jobs, 0);
                        self.run_jobs(clock, &mut cands, &jobs, Some(ext_end), &mut ticks_e, &mut roll_e);
                    }
                    pick = choose(
                        &cands,
                        &top,
                        ncombos,
                        &self.mask_weights,
                        lambda,
                        cfg.robust.mode,
                        prev.direction,
                        cfg.planner.flip_margin,
                        cfg.warm_bonus,
                        cfg.warm_fire_only,
                    );
                    break;
                }
            }
            tel.work.extension = ticks_e;
            tel.work.rollouts_extension = roll_e;
        }
        // Everything from here (the shield) runs in the full world.
        self.set_lens(Lens::Full);
        for c in &cands {
            if c.cheap().is_some() {
                tel.evaluated[c.src.kind()] += 1;
            }
        }
        if cfg.debug_dump {
            let mut idx: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].cheap().is_some()).collect();
            idx.sort_by(|&a, &b| score_desc(cands[a].cheap().unwrap(), cands[b].cheap().unwrap()).then(a.cmp(&b)));
            for &i in idx.iter().take(14) {
                let c = &cands[i];
                let s0 = c.plan[0];
                tel.dump.push((
                    c.src.label().to_string(),
                    c.cheap().unwrap_or(0.0),
                    c.res[..ncombos].iter().flatten().map(|r| r.score).collect(),
                    format!(
                        "d{} j{} h{} f{} a{:.2}",
                        s0.dir,
                        s0.jump,
                        s0.hook,
                        s0.fire,
                        if is_abs_aim(s0.aim) { s0.aim - ABS_AIM } else { s0.aim }
                    ),
                ));
            }
        }
        tel.out_of_time = out_of_time;
        if cfg.debug_pool {
            tel.pool = cands
                .iter()
                .map(|c| PoolRec {
                    src: c.src,
                    plan: c.plan.clone(),
                    cheap: c.cheap(),
                    scores: c.res.map(|r| r.map(|e| e.score)),
                    self_out: c.res.map(|r| r.map(|e| e.self_out)),
                })
                .collect();
            tel.pick = pick;
            tel.top.clone_from(&top);
            tel.weights.clone_from(&self.mask_weights);
            tel.lambda = lambda;
        }
        self.last_lambda = lambda;
        self.diag_ready = pick.is_some();
        let (computed1, hits1) = self.engine.spec_stats();
        tel.spec_prefetched = (computed1 - spec0.0) as u32;
        tel.spec_used = (hits1 - spec0.1) as u32;
        tel.search_ms = now(clock) - t_search;
        tel.rollout_ms = self.rollout_ms;

        // ---- the chosen plan -> input ---------------------------------------------------------
        let Some(pick) = pick else {
            tel.work.units = self.dec_units;
            return (prev, tel);
        };
        let chosen_cand = cands[pick].clone();
        let best = chosen_cand.plan.clone();
        tel.chosen = Some(chosen_cand.src);
        tel.chosen_plan.clone_from(&best);
        tel.best_score = chosen_cand.full0().or_else(|| chosen_cand.cheap()).unwrap_or(0.0);
        tel.unsafe_choice = chosen_cand.worst_self_out(ncombos.max(1)) > 0;
        tel.robust_value = if chosen_cand.complete(ncombos) {
            robust_value_weighted(&scores_of(&chosen_cand, ncombos), &self.mask_weights[..ncombos], lambda)
        } else {
            tel.best_score
        };

        let rest = cfg.planner.rest_aim && best[0].hook == 0 && best[0].fire == 0;
        let aim0 = if rest {
            aim_at
        } else {
            resolve_aim(best[0].aim, aim_base)
        };
        // The same hook gate (and, with `hook_snap_aim`, aim snap) the rollouts applied to this step (task 3.9).
        let (hook_ok, aim0, snapped) = self.planner.gate_and_snap(
            &*self.world,
            self_id,
            victim_id,
            Some(&me),
            Some(&victim),
            best[0],
            prev,
            aim0,
        );
        self.planner.swing_target_frozen = victim.frozen;
        self.planner.swing_rope_on = me.hooked_player == victim_id;
        self.planner.swing_target = Some(victim);
        let mut chosen = self.planner.step_to_input_snapped(
            &*self.world,
            best[0],
            prev,
            vdistance(me.pos, victim.pos),
            hook_ok,
            Some(me.pos),
            Some(victim.pos),
            Some(victim.vel),
            aim0,
            snapped,
        );

        // ---- shield ---------------------------------------------------------------------------
        let t_shield = now(clock);
        // Far from every hazard the physics sees (front layer too, review F5) *and* too slow to get there within
        // the plan: a fast tee 14 tiles from a freeze is not safe (it covers `speed` px every tick).
        let far_from_hazard = cfg.shield_skip_tiles > 0 && hooked_by.is_none() && {
            let col = self.world.collision();
            let id = crate::plan_world::PlanCollision::identity(col);
            if self.skip_field.as_ref().is_none_or(|(i, _)| *i != id) {
                self.skip_field = Some((id, Arc::new(crate::fields::hazard_field_full(col))));
            }
            let sf = &self.skip_field.as_ref().expect("just built").1;
            let reach_tiles = me.vel.x.hypot(me.vel.y) * SKIP_LOOKAHEAD_TICKS / 32.0;
            f64::from(hazard_tiles(sf, me.pos.x, me.pos.y)) >= f64::from(cfg.shield_skip_tiles) + reach_tiles
        };
        tel.shield_skipped = cfg.planner.shield && !me.frozen && far_from_hazard;
        if cfg.planner.shield && !me.frozen && !far_from_hazard {
            let hold = self.planner.shield_hold(1);
            let mut others: HashMap<i32, PlayerInput> = HashMap::new();
            others.insert(victim_id, victim_input);
            if let Some(ts) = &threat_set {
                for (i, id) in ts.ids.iter().enumerate() {
                    others.insert(*id, ts.inputs[i]);
                }
            }
            // The worst modelled response to the chosen plan is what the shield must survive.
            let mut worst_mask = 0u32;
            if ncombos > 1 && chosen_cand.complete(ncombos) {
                let worst = (0..ncombos)
                    .min_by(|&a, &b| {
                        let (sa, sb) = (
                            chosen_cand.res[a].map_or(f64::INFINITY, |r| r.score),
                            chosen_cand.res[b].map_or(f64::INFINITY, |r| r.score),
                        );
                        sa.total_cmp(&sb)
                    })
                    .map_or(0, |w| self.masks[w]);
                worst_mask = worst;
                let mut rng = Rng::new(self.planner.opp_seed);
                if worst & 1 == 1 {
                    let inp = scripted_action(&*self.world, victim_id, self_id, &victim_input, &mut rng);
                    others.insert(victim_id, inp);
                }
                if let Some(ts) = &threat_set {
                    for (i, id) in ts.ids.iter().enumerate() {
                        if (worst >> (i + 1)) & 1 == 1 {
                            let inp = scripted_action(&*self.world, *id, self_id, &ts.inputs[i], &mut rng);
                            others.insert(*id, inp);
                        }
                    }
                }
            }
            // Hook escapes at anchors (walls, ceilings): the ordinary escapes only walk and jump.
            let extras: Vec<PlayerInput> = if cfg.shield_hook_anchors > 0 {
                let rays0 = self.anchors.rays;
                let anchors = self.anchors.select(self.world.collision(), me.pos, cfg.anchors);
                if let Some(m) = &self.meter {
                    m.add_units(2 * (self.anchors.rays - rays0));
                }
                crate::hybrid::anchors::hook_escapes(&anchors, me.pos, cfg.shield_hook_anchors)
            } else {
                Vec::new()
            };
            let was_on = crate::prof::is_enabled();
            crate::prof::enable();
            let (s0, _) = crate::prof::counters();
            if let Some(m) = &self.meter {
                m.begin_shield();
            }
            let engine = &mut self.engine;
            let use_plan = cfg.shield_plan_escape;
            let plan_ok = std::cell::Cell::new(false);
            if timed {
                let call_start = t_decision;
                let reserve_deadline = now(clock) + shield_reserve;
                let dl: Option<(&dyn Clock, f64)> = Some((clock, reserve_deadline));
                let status = {
                    let mut first = || {
                        let v = engine.plan_escape(&best, worst_mask, &chosen, &others, dl, &extras).0;
                        plan_ok.set(v == crate::shield::Bounded::Done(true));
                        v
                    };
                    let mut opts = crate::shield::ShieldOpts {
                        first: if use_plan { Some(&mut first) } else { None },
                        extras: &extras,
                    };
                    crate::shield::escape_exists_ext(
                        &mut *self.world,
                        self_id,
                        &chosen,
                        hold,
                        &others,
                        dl,
                        &mut self.shield_bufs,
                        &mut opts,
                    )
                };
                if !matches!(status, crate::shield::Bounded::Done(true)) {
                    // Not `Done(true)`: either no escape exists (`Done(false)`) or the check ran out of
                    // time (`TimedOut`). An input with no escape is the slowest to disprove, so a
                    // timeout can be unconfirmed danger (`shield_timeout_danger`); the search for a
                    // safer input then gets the extension budget. Otherwise a timeout extends only when
                    // the search itself judged the plan unsafe (3.5 review round 1, F1: a substitute
                    // the shield finds late can throw away a verified plan).
                    let danger = matches!(status, crate::shield::Bounded::Done(false))
                        || tel.unsafe_choice
                        || cfg.shield_timeout_danger;
                    let safer_deadline = if danger && cfg.adaptive.enabled {
                        js::max(reserve_deadline, call_start + cfg.adaptive.max_total_ms)
                    } else {
                        reserve_deadline
                    };
                    match crate::shield::safer_input_ext(
                        &mut *self.world,
                        self_id,
                        &chosen,
                        hold,
                        &others,
                        Some(&prev),
                        Some((clock, safer_deadline)),
                        &mut self.shield_bufs,
                        &extras,
                    ) {
                        crate::shield::Bounded::Done(Some(safer)) => {
                            chosen = safer;
                            tel.shielded = true;
                        }
                        crate::shield::Bounded::Done(None) => {}
                        crate::shield::Bounded::TimedOut => tel.shield_incomplete = true,
                    }
                }
            } else {
                // Fixed-work mode: no deadline, so the answers are complete (never `TimedOut`).
                let has_escape = {
                    let mut first = || {
                        let v = engine.plan_escape(&best, worst_mask, &chosen, &others, None, &extras).0;
                        plan_ok.set(v == crate::shield::Bounded::Done(true));
                        v
                    };
                    let mut opts = crate::shield::ShieldOpts {
                        first: if use_plan { Some(&mut first) } else { None },
                        extras: &extras,
                    };
                    matches!(
                        crate::shield::escape_exists_ext(
                            &mut *self.world,
                            self_id,
                            &chosen,
                            hold,
                            &others,
                            None,
                            &mut self.shield_bufs,
                            &mut opts,
                        ),
                        crate::shield::Bounded::Done(true)
                    )
                };
                if !has_escape
                    && let crate::shield::Bounded::Done(Some(safer)) = crate::shield::safer_input_ext(
                        &mut *self.world,
                        self_id,
                        &chosen,
                        hold,
                        &others,
                        Some(&prev),
                        None,
                        &mut self.shield_bufs,
                        &extras,
                    )
                {
                    chosen = safer;
                    tel.shielded = true;
                }
            }
            tel.shield_ran = true;
            tel.shield_plan_ok = plan_ok.get();
            let (s1, _) = crate::prof::counters();
            tel.work.shield = s1 - s0;
            if let Some(m) = &self.meter {
                m.end_shield();
            }
            if !was_on {
                crate::prof::disable();
            }
        }
        tel.shield_ms = now(clock) - t_shield;
        tel.work.units = self.dec_units;

        // Predictions for the next decision's belief update.
        self.prev_predictions.clear();
        let mut modelled_ids: Vec<i32> = Vec::with_capacity(1 + threats.len());
        if victim_in_radius {
            modelled_ids.push(victim_id);
        }
        modelled_ids.extend(threats.iter().map(|t| t.id));
        for id in modelled_ids {
            if let Some(t) = all.iter().find(|t| t.id == id) {
                let hold = crate::brains::enemy_input_from_tee(t);
                let react = scripted_action(&*self.world, id, self_id, &hold, &mut Rng::new(7));
                self.prev_predictions.push((id, hold, react));
            }
        }

        // ---- bookkeeping ----------------------------------------------------------------------
        self.planner.warm = Some(normalize(&best, aim_base));
        let chosen = self.planner.maybe_release(&*self.world, self_id, chosen);
        self.planner.committed = Some(chosen);
        (chosen, tel)
    }

    /// Two-world search: besides the best by reduced score, a decision with danger flagged also
    /// re-scores up to two defensive techniques (best reduced score first): in the reduced world the
    /// threats that make them worth it do not exist, so they would never rank.
    fn add_protected(&self, cands: &[Cand], top: &mut Vec<usize>, danger: bool) {
        if !danger {
            return;
        }
        let mut def: Vec<usize> = (0..cands.len())
            .filter(|&i| {
                !top.contains(&i)
                    && cands[i].cheap().is_some()
                    && matches!(cands[i].src, Source::Tech(t) if t.is_defensive())
            })
            .collect();
        def.sort_by(|&a, &b| score_desc(cands[a].cheap().unwrap(), cands[b].cheap().unwrap()).then(a.cmp(&b)));
        top.extend(def.into_iter().take(2));
    }

    /// The `m` best candidates by cheap score (descending, ties by pool order), plus the best one
    /// that keeps the current direction if it is not among them (for the flip hysteresis).
    fn top_indices(&self, cands: &[Cand], m: usize, prev_dir: i32) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].cheap().is_some()).collect();
        idx.sort_by(|&a, &b| score_desc(cands[a].cheap().unwrap(), cands[b].cheap().unwrap()).then(a.cmp(&b)));
        let mut top: Vec<usize> = idx.iter().copied().take(m.max(1)).collect();
        if let Some(&stay) = idx.iter().find(|&&i| cands[i].plan[0].dir == prev_dir)
            && !top.contains(&stay)
        {
            top.push(stay);
        }
        top
    }
}

/// Task 3.9 (`HybridConfig::wall_throws`): the side (`-1` left, `1` right) of the nearer solid wall within
/// [`crate::hybrid::config::WALL_REACH_TILES`] tiles of `pos` at its height; `0` when there is none or both sides are equally far.
fn wall_side(col: &impl crate::plan_world::PlanCollision, pos: crate::vmath::Vec2) -> i32 {
    let reach = f64::from(crate::hybrid::config::WALL_REACH_TILES) * 32.0;
    let first = |dir: f64| {
        let mut d = 16.0;
        while d <= reach {
            if col.is_solid(pos.x + dir * d, pos.y) {
                return Some(d);
            }
            d += 16.0;
        }
        None
    };
    match (first(-1.0), first(1.0)) {
        (Some(l), Some(r)) if l < r => -1,
        (Some(l), Some(r)) if r < l => 1,
        (Some(_), None) => -1,
        (None, Some(_)) => 1,
        _ => 0,
    }
}

/// How dangerous a free tee is to us right now, for ordering (smaller = more dangerous): class 0 = its hook is in
/// flight toward us, 1 = within hammer reach, 2 = the rest; then whole px/tick of closing speed (faster first);
/// distance (in 4 px steps) only breaks ties, then the id.
fn threat_rank(t: &TeeState, me: &TeeState) -> (u8, i32, i32, i32) {
    let rel = crate::vmath::Vec2 {
        x: t.pos.x - me.pos.x,
        y: t.pos.y - me.pos.y,
    };
    let d = rel.x.hypot(rel.y).max(1e-6);
    let closing = -(rel.x * (t.vel.x - me.vel.x) + rel.y * (t.vel.y - me.vel.y)) / d;
    let class = if hook_flying_at(t, me) {
        0
    } else if d <= HAMMER_THREAT_PX {
        1
    } else {
        2
    };
    (class, -(closing.max(0.0).floor() as i32), (d / 4.0) as i32, t.id)
}

/// Is `t`'s hook in flight on a line that passes within `HOOK_AIM_SLACK_PX` of us (and still ahead of the hook head)?
fn hook_flying_at(t: &TeeState, me: &TeeState) -> bool {
    if t.hook_state != HOOK_FLYING {
        return false;
    }
    let (dx, dy) = (me.pos.x - t.hook_pos.x, me.pos.y - t.hook_pos.y);
    let len = t.hook_dir.x.hypot(t.hook_dir.y);
    if len < 1e-9 {
        return false;
    }
    let (ux, uy) = (t.hook_dir.x / len, t.hook_dir.y / len);
    let along = dx * ux + dy * uy;
    let across = (dx * uy - dy * ux).abs();
    along > -HOOK_AIM_SLACK_PX && across <= HOOK_AIM_SLACK_PX
}

/// Which tees the rollouts keep (task 3.5b, F2): us, the victim, whoever hooks us (or we hook), then the free threats
/// (inside the threat radius, nearest first: `threats` is sorted) and the nearest frozen body within
/// reach, up to `max_tees` in all (`0` = keep everybody, the 3.5 behaviour). A tee that is none of
/// these cannot act on us within the two decisions the threat radius is built for, and becomes a
/// threat itself the moment it comes inside it, so it is not simulated: an idle crowd at the far wall
/// costs nothing. Returns the ids kept and the ids to drop.
#[allow(clippy::too_many_arguments)]
fn local_tees(
    max_tees: usize,
    self_id: i32,
    victim_id: i32,
    must_keep: impl IntoIterator<Item = i32>,
    bodies: &[i32],
    threats: &[TeeState],
    frozen_by: &[&TeeState],
    all: &[TeeState],
) -> (Vec<i32>, Vec<i32>) {
    if max_tees == 0 {
        return (all.iter().map(|t| t.id).collect(), Vec::new());
    }
    let mut keep: Vec<i32> = vec![self_id, victim_id];
    for id in must_keep {
        if !keep.contains(&id) {
            keep.push(id);
        }
    }
    let mut ordered: Vec<i32> = threats.iter().map(|t| t.id).collect();
    ordered.extend(frozen_by.iter().take(1).map(|t| t.id));
    for id in ordered {
        if keep.len() >= max_tees {
            break;
        }
        if !keep.contains(&id) {
            keep.push(id);
        }
    }
    // Spared tees in contact range stay as bodies, outside the cap.
    for &id in bodies {
        if !keep.contains(&id) {
            keep.push(id);
        }
    }
    let dropped = all.iter().map(|t| t.id).filter(|id| !keep.contains(id)).collect();
    (keep, dropped)
}

/// Early pruning's gate: does a candidate with pre-score `p` go on to a full rollout? The first
/// `warmup` do (no distribution yet); after that those in the top `keep` share of the pre-scores seen
/// so far. `hist` collects every pre-score of the decision.
fn pre_keep(hist: &mut Vec<f64>, p: f64, cfg: &crate::hybrid::config::PruneConfig) -> bool {
    let go = if hist.len() < cfg.warmup {
        true
    } else {
        let mut sorted = hist.clone();
        sorted.sort_by(f64::total_cmp);
        let cut = sorted[((1.0 - cfg.keep) * sorted.len() as f64).floor() as usize];
        p >= cut
    };
    hist.push(p);
    go
}

fn scores_of(c: &Cand, combos: usize) -> Vec<f64> {
    c.res[..combos].iter().flatten().map(|r| r.score).collect()
}

/// Picks the winner among `top`: the best robust value among candidates with all combinations
/// scored; with the flip hysteresis of `decide_once` (keep the current direction unless a flip is
/// clearly better). Falls back to the best cheap score if no candidate is complete.
#[allow(clippy::too_many_arguments)]
fn choose(
    cands: &[Cand],
    top: &[usize],
    combos: usize,
    weights: &[f64],
    lambda: f64,
    mode: RobustMode,
    prev_dir: i32,
    flip_margin: f64,
    warm_bonus: f64,
    warm_fire_only: bool,
) -> Option<usize> {
    // (safe, value): with `SafeFirst` a plan no modelled reply freezes us in outranks every plan
    // some reply does, and safe plans compete on the cheap score; the rest (and `Mix`) on the mix.
    let value = |i: usize| -> Option<(bool, f64)> {
        let c = &cands[i];
        if !c.complete(combos) {
            return None;
        }
        let mix = if combos > 1 {
            robust_value_weighted(&scores_of(c, combos), &weights[..combos], lambda)
        } else {
            c.full0()?
        };
        // Commitment: the warm plan (the previous decision's plan, one step on) gets a small bonus, so that
        // a freshly generated plan that waits one step before it acts cannot beat the plan that is already
        // acting, decision after decision (T18: a hammer swing postponed for ever).
        // Review F7: a plan that some modelled reply freezes us after does not earn the commitment bonus.
        let bonus =
            if c.src == Source::Warm && (!warm_fire_only || c.plan[0].fire != 0) && c.worst_self_out(combos) == 0 {
                warm_bonus
            } else {
                0.0
            };
        let mix = mix + bonus;
        if mode == RobustMode::SafeFirst && combos > 1 {
            let safe = c.worst_self_out(combos) == 0;
            Some((safe, if safe { c.full0()? + bonus } else { mix }))
        } else {
            Some((false, mix))
        }
    };
    let better = |a: (bool, f64), b: (bool, f64)| a.0 && !b.0 || (a.0 == b.0 && a.1 > b.1);
    let mut best: Option<(usize, (bool, f64))> = None;
    let mut best_stay: Option<(usize, (bool, f64))> = None;
    for &i in top {
        let Some(v) = value(i) else { continue };
        if best.is_none_or(|(_, bv)| better(v, bv)) {
            best = Some((i, v));
        }
        if cands[i].plan[0].dir == prev_dir && best_stay.is_none_or(|(_, sv)| better(v, sv)) {
            best_stay = Some((i, v));
        }
    }
    let (mut bi, bv) = match best {
        Some(b) => b,
        None => {
            // Nothing complete: the best cheap score anywhere.
            let mut cheap: Option<(usize, f64)> = None;
            for (i, c) in cands.iter().enumerate() {
                if let Some(s) = c.cheap()
                    && cheap.is_none_or(|(_, bs)| s > bs)
                {
                    cheap = Some((i, s));
                }
            }
            return cheap.map(|(i, _)| i);
        }
    };
    if flip_margin > 0.0
        && let Some((si, sv)) = best_stay
        && cands[bi].plan[0].dir != prev_dir
        && bv.0 == sv.0
        && bv.1 - sv.1 < flip_margin
    {
        bi = si;
    }
    Some(bi)
}

/// `b - a` with NaN as equal (the planner's descending sort).
fn score_desc(a: f64, b: f64) -> std::cmp::Ordering {
    let d = b - a;
    if d < 0.0 {
        std::cmp::Ordering::Less
    } else if d > 0.0 {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Pads/truncates a proposal to `n` steps and replaces non-finite aims (a proposal is untrusted).
fn sanitize(mut plan: Vec<PlanStep>, n: usize) -> Vec<PlanStep> {
    if plan.is_empty() {
        return plan;
    }
    while plan.len() < n {
        plan.push(*plan.last().unwrap());
    }
    plan.truncate(n);
    for s in &mut plan {
        s.dir = s.dir.clamp(-1, 1);
        s.jump = i32::from(s.jump != 0);
        s.hook = i32::from(s.hook != 0);
        s.fire = i32::from(s.fire != 0);
        if !s.aim.is_finite() {
            s.aim = 0.0;
        }
    }
    plan
}

/// A plan as the next decision's warm start: technique plans' absolute aims become aims relative to
/// the victim direction (the encoding CEM samples around and `refit` averages).
fn normalize(plan: &[PlanStep], aim_base: f64) -> Vec<PlanStep> {
    plan.iter()
        .map(|s| {
            if is_abs_aim(s.aim) {
                PlanStep {
                    aim: wrap_angle(s.aim - ABS_AIM - aim_base),
                    ..*s
                }
            } else {
                *s
            }
        })
        .collect()
}

/// `Some(true)` when `me` holds the strong side of a hook duel with `other` (the tee that is
/// processed *later* in the tick, i.e. spawned earlier), `None` when either is not in the world.
fn strength(world: &ddai_physics::world::World<f32>, me: i32, other: i32) -> Option<bool> {
    let pos = |id: i32| world.entity_order.iter().position(|&x| i32::from(x) == id);
    Some(pos(me)? > pos(other)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(dir: i32, scores: &[f64]) -> Cand {
        let mut c = Cand::new(
            vec![PlanStep {
                dir,
                jump: 0,
                hook: 0,
                fire: 0,
                aim: 0.0,
            }],
            Source::Cem,
        );
        for (i, &s) in scores.iter().enumerate() {
            c.res[i] = Some(EvalResult {
                score: s,
                self_out: 0,
                enemy_out: 0,
                enemy_sealed: false,
            });
        }
        c
    }

    #[test]
    fn choose_prefers_the_best_worst_case_not_the_best_cheap_score() {
        // A wins the cheap model but is refuted by the reacting opponent; B is steady.
        let cands = vec![cand(1, &[5.0, -6.0]), cand(1, &[3.0, 2.5])];
        let pick = choose(&cands, &[0, 1], 2, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.0, 0.0, false);
        assert_eq!(pick, Some(1));
        // Pure mean would still prefer B (2.75 vs -0.5); pure cheap prefers A.
        let pick = choose(&cands, &[0, 1], 1, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.0, 0.0, false);
        assert_eq!(pick, Some(0));
    }

    #[test]
    fn safe_first_keeps_the_best_cheap_plan_among_those_no_reply_freezes_us_in() {
        // A has the best cheap score but one reply freezes us; B and C are safe, B is cheaper-better.
        let mut a = cand(1, &[9.0, 8.0]);
        a.res[1].as_mut().unwrap().self_out = 5;
        let b = cand(1, &[4.0, -3.0]);
        let c = cand(1, &[2.0, 1.9]);
        let cands = vec![a, b, c];
        assert_eq!(
            choose(
                &cands,
                &[0, 1, 2],
                2,
                &[1.0; 4],
                0.5,
                RobustMode::SafeFirst,
                1,
                0.0,
                0.0,
                false
            ),
            Some(1)
        );
        // The mix would take C (mean 1.95, worst 1.9) over B (mean 0.5, worst -3.0).
        assert_eq!(
            choose(
                &cands,
                &[0, 1, 2],
                2,
                &[1.0; 4],
                0.5,
                RobustMode::Mix,
                1,
                0.0,
                0.0,
                false
            ),
            Some(0)
        );
        // Nobody safe: fall back to the mix.
        let mut d = cand(1, &[9.0, 8.0]);
        d.res[1].as_mut().unwrap().self_out = 5;
        let mut e = cand(1, &[3.0, 2.0]);
        e.res[0].as_mut().unwrap().self_out = 1;
        let cands = vec![d, e];
        assert_eq!(
            choose(
                &cands,
                &[0, 1],
                2,
                &[1.0; 4],
                0.5,
                RobustMode::SafeFirst,
                1,
                0.0,
                0.0,
                false
            ),
            Some(0)
        );
    }

    #[test]
    fn incomplete_candidates_are_not_eligible_for_the_robust_choice() {
        let cands = vec![cand(1, &[9.0]), cand(1, &[3.0, 2.0])];
        assert_eq!(
            choose(&cands, &[0, 1], 2, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.0, 0.0, false),
            Some(1)
        );
        // Nothing complete: fall back to the best cheap score.
        let cands = vec![cand(1, &[9.0]), cand(1, &[3.0])];
        assert_eq!(
            choose(&cands, &[0, 1], 2, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.0, 0.0, false),
            Some(0)
        );
    }

    #[test]
    fn flip_hysteresis_keeps_the_direction_unless_the_flip_is_clearly_better() {
        let cands = vec![cand(-1, &[2.0]), cand(1, &[1.7])];
        // prev direction +1: the flip to -1 wins by 0.3 < 0.6, so the stay plan is kept.
        assert_eq!(
            choose(&cands, &[0, 1], 1, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.6, 0.0, false),
            Some(1)
        );
        let cands = vec![cand(-1, &[2.5]), cand(1, &[1.7])];
        assert_eq!(
            choose(&cands, &[0, 1], 1, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.6, 0.0, false),
            Some(0)
        );
    }

    #[test]
    fn ties_go_to_the_first_candidate() {
        let cands = vec![cand(1, &[1.0]), cand(1, &[1.0]), cand(1, &[1.0])];
        assert_eq!(
            choose(
                &cands,
                &[0, 1, 2],
                1,
                &[1.0; 4],
                0.5,
                RobustMode::Mix,
                1,
                0.0,
                0.0,
                false
            ),
            Some(0)
        );
    }

    #[test]
    fn the_commitment_bonus_goes_only_to_a_warm_plan_that_no_reply_freezes() {
        // The warm plan A scores 0.2 below B under both replies. With the bonus (0.3) it wins, unless a modelled
        // reply freezes us after it (review F7): then it has to stand on its own score.
        let mk = |warm_self_out: i32| {
            let mut a = cand(1, &[1.0, 1.0]);
            a.src = Source::Warm;
            a.res[1].as_mut().unwrap().self_out = warm_self_out;
            vec![a, cand(1, &[1.2, 1.2])]
        };
        let pick = |cands: &[Cand]| choose(cands, &[0, 1], 2, &[1.0; 4], 0.5, RobustMode::Mix, 1, 0.0, 0.3, false);
        assert_eq!(pick(&mk(0)), Some(0), "a safe warm plan keeps its bonus");
        // A reply freezes us after the warm plan: no bonus, so its 1.0 loses to B's 1.2 (with the bonus it would be 1.3).
        let unsafe_cands = mk(1);
        assert_eq!(pick(&unsafe_cands), Some(1));
    }

    #[test]
    fn warm_plans_lose_their_absolute_aims() {
        let plan = vec![
            PlanStep {
                dir: 1,
                jump: 0,
                hook: 1,
                fire: 0,
                aim: crate::hybrid::abs_aim(-1.0),
            },
            PlanStep {
                dir: 1,
                jump: 0,
                hook: 0,
                fire: 0,
                aim: 0.3,
            },
        ];
        let w = normalize(&plan, 0.25);
        assert!((w[0].aim - -1.25).abs() < 1e-12);
        assert_eq!(w[1].aim, 0.3);
        assert!(w.iter().all(|s| !is_abs_aim(s.aim)));
    }

    #[test]
    fn proposals_are_sanitized() {
        let bad = vec![PlanStep {
            dir: 5,
            jump: 7,
            hook: -1,
            fire: 2,
            aim: f64::NAN,
        }];
        let p = sanitize(bad, 3);
        assert_eq!(p.len(), 3);
        assert!(
            p.iter()
                .all(|s| s.dir == 1 && s.jump == 1 && s.hook == 1 && s.fire == 1 && s.aim == 0.0)
        );
    }

    #[test]
    fn signatures_ignore_sub_pixel_aim_noise_only() {
        let a = [PlanStep {
            dir: 1,
            jump: 0,
            hook: 0,
            fire: 0,
            aim: 0.5,
        }];
        let b = [PlanStep { aim: 0.5001, ..a[0] }];
        let c = [PlanStep { aim: 0.6, ..a[0] }];
        assert_eq!(signature(&a), signature(&b));
        assert_ne!(signature(&a), signature(&c));
    }

    #[test]
    fn source_labels_name_the_technique() {
        assert_eq!(Source::Tech(Tech::T14).label(), "T14 panic hook");
        assert_eq!(Source::Book.label(), "book");
        assert_eq!(Source::Tech(Tech::T9).kind(), 5);
    }

    #[test]
    fn telemetry_json_is_valid_json() {
        let mut t = DecisionTelemetry {
            chosen: Some(Source::Tech(Tech::T14)),
            chosen_plan: vec![PlanStep {
                dir: 1,
                jump: 0,
                hook: 1,
                fire: 0,
                aim: 0.0,
            }],
            threat_ids: vec![2, 3],
            budget_ms: 4.0,
            best_score: f64::NAN,
            ..DecisionTelemetry::default()
        };
        t.generated[5] = 3;
        let j = t.to_json();
        assert!(j.contains("\"chosen\":\"T14 panic hook\""), "{j}");
        assert!(j.contains("\"threats\":[2,3]"), "{j}");
        assert!(
            j.contains("\"best_score\":0.0000"),
            "NaN must not leak into the JSON: {j}"
        );
        assert!(!j.contains("NaN") && !j.contains("inf"));
    }
}
