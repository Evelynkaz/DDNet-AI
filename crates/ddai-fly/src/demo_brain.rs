//! Task 7.3, acceptance criterion 9's learning demo: no physics world needed — synthetic
//! observations (a random spawn on a real block map, an opponent at a random relative position/
//! velocity) + a scripted teacher (direction towards the opponent, hook when close with clear
//! line of sight, jump when a wall is ahead), training encoder + fly + decoder together via
//! [`crate::brain_train::brain_train_step`] and [`crate::optim::guarded_adam_step`] (fly `a`/`b`/
//! `theta`) / [`crate::flat_adam`] (encoder `g`/`c`, decoder `W`/`b`).
//!
//! **Scope** (documented, not silently narrowed): the scripted teacher FLY.md/the task spec
//! describe only labels `direction`/`jump`/`hook` — `fire`/`aim` have no defined teacher signal
//! for this synthetic scenario (no weapon/aim target exists in it), so this demo trains and
//! reports metrics for exactly those three heads, not all seven decoder outputs.
//!
//! **The MLP control's parameter count** (acceptance criterion 9: "a same-size MLP baseline"):
//! sized so its *total* parameter count is as close as achievable to the fly pipeline's own
//! (encoder `g`/`c` + connectome `a`/`b`/`theta` + decoder `W`/`b`) — but the two architectures'
//! parameter *sharing* is fundamentally different (the MLP is dense per raw input feature; the
//! connectome's parameters are shared per type-pair, not per input feature), so exact parity is
//! not achievable without either crippling the MLP's single hidden layer to a handful of units or
//! inflating it past the fly's own count. This module computes the MLP's hidden size from the
//! fly pipeline's actual parameter count and reports **both** counts alongside the metrics below,
//! rather than asserting silent parity — see the crate README's "Демо обучения" section and the
//! task's build report for the honest numbers this produced.
//!
//! ## Honest evaluation (review round 1, F2, CONFIRMED — acceptance criterion 9)
//! An earlier revision reported plain accuracy against a uniformly-random synthetic distribution
//! and a hardcoded "chance" baseline (`1/3`, `1/2`) that silently assumed every class was balanced
//! — on the real teacher, `jump` is a rare event (a wall directly ahead within a short lookahead)
//! and a model that always predicts "no jump" scored a misleadingly high plain accuracy. Fixed by:
//! - [`HeadMetrics`]/[`BinaryMetrics`]/[`DirectionMetrics`]: every reported head carries its own
//!   **label prevalence**, the **majority-class baseline accuracy** actually computed from that
//!   prevalence (not a hardcoded constant), **per-class recall**, **balanced accuracy** (the mean
//!   of per-class recall — the metric that actually penalizes a constant predictor), **AUROC**
//!   (binary heads) and a **Wilson 95% CI** on plain accuracy (so a difference of a few points on
//!   `held_out_samples` reflects sampling noise rather than a real effect).
//! - [`sample_scenario_stratified`]: rejection-samples the *jump* label towards
//!   [`BrainDemoConfig::jump_target_positive_rate`] (both training batches and the held-out set),
//!   so gradient signal and evaluation statistics for the rarest head aren't dominated by one
//!   class. `direction`/`hook` are left at whatever rate the stratified draw's *other* fields
//!   happen to produce (not independently stratified) — see that function's doc comment.
//! - [`evaluate_fly`]/[`evaluate_mlp`] are free functions (not a closure trapped inside
//!   [`run_brain_demo`]), so the same evaluation a caller sees printed from one map/seed can be
//!   re-run against other maps/seeds — see `tests/brain_demo_generalization.rs` for the held-out
//!   maps/multi-seed protocol this enables.

use ddai_brain::{CharacterObservation, Observation};
use serde::Serialize;
use std::sync::Arc;

use crate::backward::BackwardIndex;
use crate::brain_train::{BrainSequence, DecisionTargets, brain_train_step};
use crate::decoder::{
    DecoderModel, DecoderParams, DecoderScratch, DecoderTargets, DnCalibration, decoder_forward_into,
};
use crate::encoder::{EncoderModel, EncoderParams, RayGridFeatures, compute_proprioception_values};
use crate::flat_adam::{FlatAdamConfig, FlatAdamState, clip_grad_norm_multi};
use crate::model::FlyModel;
use crate::optim::{GuardedAdamConfig, GuardedAdamState, ParamGradients, clip_grad_norm};
use crate::rng::SplitMix64;
use crate::state::FlyState;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BrainDemoConfig {
    pub batch_size: usize,
    pub t_decisions: usize,
    pub steps: usize,
    pub grad_clip_norm: f32,
    pub lr_fly: f32,
    pub lr_encoder: f32,
    pub lr_decoder: f32,
    pub seed: u64,
    /// DDNet's real hook length is a few hundred px (`tuning.hook_length()`, default 380); the
    /// teacher treats the opponent as "hookable" within this range plus clear line of sight.
    pub hook_range_px: f32,
    /// How far ahead (px) the teacher looks for a wall before deciding to jump.
    pub jump_lookahead_px: f32,
    /// How far from the tee (px) an opponent can spawn.
    pub opponent_spawn_radius_px: f32,
    /// Review round 1, F2: [`sample_scenario_stratified`] rejection-samples towards this fraction
    /// of `jump == true` (the task's own guidance: "stratify to 30-50% positive").
    pub jump_target_positive_rate: f32,
    /// Cap on [`sample_scenario_stratified`]'s rejection loop, so a map/config combination where
    /// the target rate is unreachable (e.g. a fully open map has *no* wall-ahead scenarios at all)
    /// degrades to "whatever the natural rate is" instead of hanging.
    pub jump_stratify_max_attempts: usize,
    /// Size of the held-out evaluation set [`run_brain_demo`] draws once (seed-offset from
    /// training, see that function) and reuses for before/after-fly and after-MLP metrics.
    pub held_out_samples: usize,
}

impl Default for BrainDemoConfig {
    fn default() -> Self {
        BrainDemoConfig {
            batch_size: 16,
            t_decisions: 4,
            steps: 200,
            grad_clip_norm: 1.0,
            lr_fly: 2e-4,
            lr_encoder: 2e-2,
            lr_decoder: 2e-2,
            seed: 42,
            hook_range_px: 380.0,
            jump_lookahead_px: 40.0,
            opponent_spawn_radius_px: 500.0,
            jump_target_positive_rate: 0.4,
            jump_stratify_max_attempts: 300,
            held_out_samples: 200,
        }
    }
}

// --- Synthetic scenario generation --------------------------------------------------------------

fn tile_is_solid(map: &ddai_physics::map::MapData, x: f32, y: f32) -> bool {
    use crate::encoder::round_to_int_f32;
    use ddai_physics::map::{TILE_NOHOOK, TILE_SOLID};
    if map.width == 0 || map.height == 0 {
        return false;
    }
    // Round-then-divide (review round 1, F18 — same fix as `crate::encoder::tile_class`, not the
    // truncating-divide an earlier revision used here): this function samples arbitrary
    // continuous positions (a character's actual position, a line-of-sight midpoint), where the
    // rounding rule actually changes which tile gets picked for about half of every tile's width.
    let nx = ((round_to_int_f32(x) / 32) as i64).clamp(0, map.width as i64 - 1) as usize;
    let ny = ((round_to_int_f32(y) / 32) as i64).clamp(0, map.height as i64 - 1) as usize;
    let idx = ny * map.width as usize + nx;
    let index = map.game.get(idx).map(|t| t.index).unwrap_or(0);
    index == TILE_SOLID || index == TILE_NOHOOK
}

fn line_of_sight_clear(map: &ddai_physics::map::MapData, from: (f32, f32), to: (f32, f32)) -> bool {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let steps = (dist / 16.0).ceil().max(1.0) as usize;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        if tile_is_solid(map, from.0 + dx * t, from.1 + dy * t) {
            return false;
        }
    }
    true
}

fn sample_open_position(map: &ddai_physics::map::MapData, rng: &mut SplitMix64) -> (f32, f32) {
    for _ in 0..500 {
        let tx = (rng.next_u64() % map.width.max(1) as u64) as f32;
        let ty = (rng.next_u64() % map.height.max(1) as u64) as f32;
        let (x, y) = (tx * 32.0 + 16.0, ty * 32.0 + 16.0);
        if !tile_is_solid(map, x, y) {
            return (x, y);
        }
    }
    // Fallback: map center, whatever it is (a degenerate all-solid map is a config error the
    // caller should notice from the accuracy numbers, not a panic here).
    (map.width as f32 * 16.0, map.height as f32 * 16.0)
}

/// One synthetic sample: self spawns on open ground somewhere on the map, an opponent spawns at a
/// random offset (also open ground, best-effort) with a random velocity. `pub` (review round 2,
/// F20): the **natural**, unstratified distribution this draws from is itself a metric worth
/// reporting alongside [`sample_scenario_stratified`]'s -- exposed so a caller (`tests/brain_demo_
/// generalization.rs`) can draw from it directly instead of re-implementing an equivalent
/// generator that risks drifting from this one.
pub fn sample_scenario(
    map: Arc<ddai_physics::map::MapData>,
    cfg: &BrainDemoConfig,
    rng: &mut SplitMix64,
) -> Observation {
    let self_pos = sample_open_position(&map, rng);
    let mut opp_pos;
    loop {
        let angle = rng.next_f32_unit() * std::f32::consts::TAU;
        let radius = rng.next_f32_unit() * cfg.opponent_spawn_radius_px;
        opp_pos = (self_pos.0 + angle.cos() * radius, self_pos.1 + angle.sin() * radius);
        if opp_pos.0 >= 0.0
            && opp_pos.1 >= 0.0
            && opp_pos.0 < map.width as f32 * 32.0
            && opp_pos.1 < map.height as f32 * 32.0
            && !tile_is_solid(&map, opp_pos.0, opp_pos.1)
        {
            break;
        }
    }
    // Velocities in px/tick (review round 1, F4 — matches `CharacterObservation::vel`'s unit):
    // `tuning.ground_control_speed()`'s default is `10.0`; the ranges below (±20 horizontal,
    // ±15 vertical) cover ordinary running plus jump/fall/hook-boosted speeds without the
    // absurd (~10x too fast) magnitudes an earlier revision used under a mistaken px/s
    // assumption.
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(self_pos.0, self_pos.1);
    me.vel = ddai_physics::vmath::Vec2::new(rng.next_f32_unit() * 40.0 - 20.0, rng.next_f32_unit() * 30.0 - 15.0);
    me.grounded = rng.next_f32_unit() < 0.6;
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(opp_pos.0, opp_pos.1);
    opp.vel = ddai_physics::vmath::Vec2::new(rng.next_f32_unit() * 40.0 - 20.0, rng.next_f32_unit() * 30.0 - 15.0);

    Observation {
        map,
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    }
}

/// The scripted teacher (task spec, acceptance criterion 9): direction towards the opponent, hook
/// when close with clear line of sight, jump when a wall is ahead. `direction`: `0` = left, `1` =
/// stop, `2` = right (matching `DecoderConfig::direction_actions`' default order).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeacherLabels {
    pub direction: u8,
    pub jump: bool,
    pub hook: bool,
}

pub fn scripted_teacher(obs: &Observation, cfg: &BrainDemoConfig) -> TeacherLabels {
    // `others[0]` (not `target_or_nearest()`) is deliberate here, not the `others[0]` pattern
    // review round 1's F13 flagged in `crate::encoder` -- `sample_scenario` always constructs
    // `obs.others` with exactly one entry (this demo's teacher only ever has one opponent to
    // react to), so there is no ordering ambiguity to get wrong.
    let opp = &obs.others[0];
    let dx = opp.pos.x - obs.self_state.pos.x;
    let dy = opp.pos.y - obs.self_state.pos.y;
    let dist = (dx * dx + dy * dy).sqrt();

    let direction = if dx.abs() < 24.0 {
        1
    } else if dx < 0.0 {
        0
    } else {
        2
    };
    let hook = dist < cfg.hook_range_px
        && line_of_sight_clear(
            &obs.map,
            (obs.self_state.pos.x, obs.self_state.pos.y),
            (opp.pos.x, opp.pos.y),
        );
    let ahead_x = obs.self_state.pos.x
        + if direction == 0 {
            -cfg.jump_lookahead_px
        } else {
            cfg.jump_lookahead_px
        };
    let jump = tile_is_solid(&obs.map, ahead_x, obs.self_state.pos.y);

    TeacherLabels { direction, jump, hook }
}

/// Review round 1, F2 (CONFIRMED): on a real block map, a uniformly-random `sample_scenario` draw
/// makes `jump == true` a rare event (a wall directly ahead within `jump_lookahead_px`) — one
/// measured real-map rate was ~1.5% positive, so a constant "never jump" predictor scored a
/// misleadingly high plain accuracy and there was essentially no gradient signal for the positive
/// class. Rejection-samples towards [`BrainDemoConfig::jump_target_positive_rate`] instead: each
/// call first picks which class it *wants* (a coin flip at that rate), then keeps re-drawing whole
/// scenarios until `scripted_teacher`'s `jump` label matches, up to
/// [`BrainDemoConfig::jump_stratify_max_attempts`] tries (falling back to the last draw, whatever
/// its label, rather than looping forever or panicking — some maps/configs may not have any wall-
/// ahead scenario at all, or may be *all* wall-ahead depending on `jump_lookahead_px`).
///
/// **Not independently stratified**: `direction`/`hook` are whatever the accepted draw's own
/// geometry happens to produce, not separately targeted — the review's own measurements found
/// `direction` (~1/3 per class) and `hook` (~66% positive) both reasonably balanced already on the
/// real map tested, unlike `jump`; if a future map/config combination turns out not to be, the
/// same technique generalizes (stratify on whichever field needs it).
pub fn sample_scenario_stratified(
    map: Arc<ddai_physics::map::MapData>,
    cfg: &BrainDemoConfig,
    rng: &mut SplitMix64,
) -> (Observation, TeacherLabels) {
    let want_positive = rng.next_f32_unit() < cfg.jump_target_positive_rate;
    for _ in 0..cfg.jump_stratify_max_attempts {
        let obs = sample_scenario(Arc::clone(&map), cfg, rng);
        let labels = scripted_teacher(&obs, cfg);
        if labels.jump == want_positive {
            return (obs, labels);
        }
    }
    let obs = sample_scenario(map, cfg, rng);
    let labels = scripted_teacher(&obs, cfg);
    (obs, labels)
}

fn flat_features(features: &RayGridFeatures, an: &crate::encoder::ProprioceptionValues) -> Vec<f32> {
    use crate::encoder::Channel;
    let mut out = Vec::with_capacity(features.num_directions() * features.num_bins() * 7 + 9);
    for &ch in &[
        Channel::OpponentPosition,
        Channel::OpponentApproach,
        Channel::OpponentHook,
        Channel::OtherPlayers,
        Channel::Walls,
        Channel::FreezeDeathTiles,
        Channel::NoHookTiles,
    ] {
        out.extend_from_slice(features.spatial(ch));
    }
    out.push(features.scalar(Channel::SelfVelocityX));
    out.push(features.scalar(Channel::SelfVelocityY));
    out.push(features.scalar(Channel::SelfVelocityFlow));
    out.extend_from_slice(&[
        an.grounded,
        an.airborne,
        an.own_hook,
        an.jumps_left,
        an.freeze_timer,
        an.speed,
    ]);
    out
}

// --- MLP control (hand-written forward/backward, no ML framework) -------------------------------

/// One hidden layer (tanh), three small linear heads: 3-way direction, 1 jump logit, 1 hook logit
/// — see the module doc comment for how `hidden` is sized.
#[derive(Debug, Clone, Serialize)]
pub struct MlpParams {
    pub w1: Vec<f32>,    // hidden x input
    pub b1: Vec<f32>,    // hidden
    pub w_dir: Vec<f32>, // 3 x hidden
    pub b_dir: [f32; 3],
    pub w_jump: Vec<f32>, // hidden
    pub b_jump: f32,
    pub w_hook: Vec<f32>,
    pub b_hook: f32,
    pub input_dim: usize,
    pub hidden: usize,
}

impl MlpParams {
    pub fn num_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w_dir.len() + 3 + self.w_hook.len() + 1 + self.w_jump.len() + 1
    }

    fn init(input_dim: usize, hidden: usize, rng: &mut SplitMix64) -> Self {
        let scale = (1.0 / input_dim as f32).sqrt();
        let mut next = || (rng.next_f32_unit() * 2.0 - 1.0) * scale;
        MlpParams {
            w1: (0..hidden * input_dim).map(|_| next()).collect(),
            b1: vec![0.0; hidden],
            w_dir: (0..3 * hidden).map(|_| next()).collect(),
            b_dir: [0.0; 3],
            w_jump: (0..hidden).map(|_| next()).collect(),
            b_jump: 0.0,
            w_hook: (0..hidden).map(|_| next()).collect(),
            b_hook: 0.0,
            input_dim,
            hidden,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct MlpGradients {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w_dir: Vec<f32>,
    b_dir: [f32; 3],
    w_jump: Vec<f32>,
    b_jump: f32,
    w_hook: Vec<f32>,
    b_hook: f32,
}

impl MlpGradients {
    fn zeros(p: &MlpParams) -> Self {
        MlpGradients {
            w1: vec![0.0; p.w1.len()],
            b1: vec![0.0; p.b1.len()],
            w_dir: vec![0.0; p.w_dir.len()],
            b_dir: [0.0; 3],
            w_jump: vec![0.0; p.w_jump.len()],
            b_jump: 0.0,
            w_hook: vec![0.0; p.w_hook.len()],
            b_hook: 0.0,
        }
    }
}

fn mlp_forward_hidden(p: &MlpParams, x: &[f32]) -> Vec<f32> {
    (0..p.hidden)
        .map(|h| {
            let row = &p.w1[h * p.input_dim..(h + 1) * p.input_dim];
            (p.b1[h] + row.iter().zip(x).map(|(&w, &xi)| w * xi).sum::<f32>()).tanh()
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct MlpPrediction {
    pub direction_probs: [f32; 3],
    pub jump_prob: f32,
    pub hook_prob: f32,
}

pub fn mlp_forward(p: &MlpParams, x: &[f32]) -> MlpPrediction {
    let hidden = mlp_forward_hidden(p, x);
    let mut logits = [0.0f32; 3];
    for (c, l) in logits.iter_mut().enumerate() {
        let row = &p.w_dir[c * p.hidden..(c + 1) * p.hidden];
        *l = p.b_dir[c] + row.iter().zip(&hidden).map(|(&w, &h)| w * h).sum::<f32>();
    }
    let max = logits[0].max(logits[1]).max(logits[2]);
    let exps = [
        (logits[0] - max).exp(),
        (logits[1] - max).exp(),
        (logits[2] - max).exp(),
    ];
    let sum = exps[0] + exps[1] + exps[2];
    let direction_probs = [exps[0] / sum, exps[1] / sum, exps[2] / sum];

    let jump_logit = p.b_jump + p.w_jump.iter().zip(&hidden).map(|(&w, &h)| w * h).sum::<f32>();
    let hook_logit = p.b_hook + p.w_hook.iter().zip(&hidden).map(|(&w, &h)| w * h).sum::<f32>();
    MlpPrediction {
        direction_probs,
        jump_prob: crate::activation::sigmoid(jump_logit),
        hook_prob: crate::activation::sigmoid(hook_logit),
    }
}

fn mlp_loss_and_grad(p: &MlpParams, x: &[f32], labels: TeacherLabels) -> (f32, MlpGradients) {
    let hidden = mlp_forward_hidden(p, x);
    let mut grads = MlpGradients::zeros(p);
    let mut loss = 0.0f32;
    let mut grad_hidden = vec![0.0f32; p.hidden];

    // direction (3-way CE)
    let mut logits = [0.0f32; 3];
    for (c, l) in logits.iter_mut().enumerate() {
        let row = &p.w_dir[c * p.hidden..(c + 1) * p.hidden];
        *l = p.b_dir[c] + row.iter().zip(&hidden).map(|(&w, &h)| w * h).sum::<f32>();
    }
    let max = logits[0].max(logits[1]).max(logits[2]);
    let exps = [
        (logits[0] - max).exp(),
        (logits[1] - max).exp(),
        (logits[2] - max).exp(),
    ];
    let sum = exps[0] + exps[1] + exps[2];
    let probs = [exps[0] / sum, exps[1] / sum, exps[2] / sum];
    loss += -(probs[labels.direction as usize].max(1e-12)).ln();
    for (c, &pc) in probs.iter().enumerate() {
        let d_logit = pc - f32::from(c == labels.direction as usize);
        grads.b_dir[c] += d_logit;
        for h in 0..p.hidden {
            grads.w_dir[c * p.hidden + h] += d_logit * hidden[h];
            grad_hidden[h] += d_logit * p.w_dir[c * p.hidden + h];
        }
    }

    // jump / hook (BCE)
    let jump_logit = p.b_jump + p.w_jump.iter().zip(&hidden).map(|(&w, &h)| w * h).sum::<f32>();
    let jump_p = crate::activation::sigmoid(jump_logit);
    let y = f32::from(labels.jump);
    loss += -(y * jump_p.max(1e-12).ln() + (1.0 - y) * (1.0 - jump_p).max(1e-12).ln());
    let d_jump = jump_p - y;
    grads.b_jump += d_jump;
    for h in 0..p.hidden {
        grads.w_jump[h] += d_jump * hidden[h];
        grad_hidden[h] += d_jump * p.w_jump[h];
    }

    let hook_logit = p.b_hook + p.w_hook.iter().zip(&hidden).map(|(&w, &h)| w * h).sum::<f32>();
    let hook_p = crate::activation::sigmoid(hook_logit);
    let y = f32::from(labels.hook);
    loss += -(y * hook_p.max(1e-12).ln() + (1.0 - y) * (1.0 - hook_p).max(1e-12).ln());
    let d_hook = hook_p - y;
    grads.b_hook += d_hook;
    for h in 0..p.hidden {
        grads.w_hook[h] += d_hook * hidden[h];
        grad_hidden[h] += d_hook * p.w_hook[h];
    }

    // Backprop through tanh into w1/b1.
    for h in 0..p.hidden {
        let d_pre = grad_hidden[h] * (1.0 - hidden[h] * hidden[h]);
        grads.b1[h] += d_pre;
        let row = &mut grads.w1[h * p.input_dim..(h + 1) * p.input_dim];
        for (g, &xi) in row.iter_mut().zip(x) {
            *g += d_pre * xi;
        }
    }

    (loss, grads)
}

// --- Evaluation metrics (review round 1, F2) ------------------------------------------------------

/// Wilson score interval (95%, `z = 1.959964`) for a proportion `k/n` — a plain `p +- 1.96*sqrt(p(1-
/// p)/n)` Wald interval can extend below 0 or above 1 and is a poor approximation for the small
/// `held_out_samples` this demo actually uses; Wilson's doesn't have either problem. `n == 0`
/// returns the vacuous `(0.0, 1.0)`.
pub fn wilson_ci95(k: usize, n: usize) -> (f32, f32) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959964f32;
    let n_f = n as f32;
    let p = k as f32 / n_f;
    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let center = p + z2 / (2.0 * n_f);
    let margin = z * (p * (1.0 - p) / n_f + z2 / (4.0 * n_f * n_f)).sqrt();
    (
        ((center - margin) / denom).clamp(0.0, 1.0),
        ((center + margin) / denom).clamp(0.0, 1.0),
    )
}

/// AUROC via the Mann-Whitney U statistic (rank-sum of the positive class's scores over ties-
/// averaged ranks), equivalent to "probability a random positive scores higher than a random
/// negative" — the standard rank-based estimator, exact (no thresholding/binning). `NaN` if either
/// class is empty (undefined, same convention `crate::probe`'s own AUROC-adjacent code — if any —
/// would use; the caller decides how to display a `NaN`, this function doesn't invent a fallback
/// number).
pub fn auroc_binary(scores: &[f32], labels: &[bool]) -> f32 {
    assert_eq!(scores.len(), labels.len());
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).expect("scores must not be NaN"));
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        // 1-based average rank over the tied block [i, j].
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &k in &order[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let rank_sum_pos: f64 = (0..scores.len()).filter(|&k| labels[k]).map(|k| ranks[k]).sum();
    let u = rank_sum_pos - (n_pos as f64 * (n_pos as f64 + 1.0) / 2.0);
    (u / (n_pos as f64 * n_neg as f64)) as f32
}

/// Every metric [`evaluate_fly`]/[`evaluate_mlp`] report for one binary head (`jump`/`hook`):
/// prevalence and the majority-class baseline actually computed from it (review round 1, F2 — not
/// a hardcoded `0.5`), plain accuracy plus its Wilson 95% CI, per-class recall, balanced accuracy
/// (mean of the two recalls — the metric a constant predictor cannot inflate), and AUROC.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BinaryMetrics {
    pub n: usize,
    pub prevalence: f32,
    pub majority_baseline_accuracy: f32,
    pub accuracy: f32,
    pub accuracy_wilson_ci95: (f32, f32),
    pub recall_positive: f32,
    pub recall_negative: f32,
    pub balanced_accuracy: f32,
    pub auroc: f32,
}

fn binary_metrics(probs: &[f32], labels: &[bool]) -> BinaryMetrics {
    let n = labels.len();
    let n_pos = labels.iter().filter(|&&l| l).count();
    let prevalence = n_pos as f32 / n.max(1) as f32;
    let majority_baseline_accuracy = prevalence.max(1.0 - prevalence);
    let (mut tp, mut tn, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for (&p, &y) in probs.iter().zip(labels) {
        match (p >= 0.5, y) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let recall_positive = if tp + fn_ > 0 {
        tp as f32 / (tp + fn_) as f32
    } else {
        f32::NAN
    };
    let recall_negative = if tn + fp > 0 {
        tn as f32 / (tn + fp) as f32
    } else {
        f32::NAN
    };
    let balanced_accuracy = match (recall_positive.is_finite(), recall_negative.is_finite()) {
        (true, true) => 0.5 * (recall_positive + recall_negative),
        (true, false) => recall_positive,
        (false, true) => recall_negative,
        (false, false) => f32::NAN,
    };
    BinaryMetrics {
        n,
        prevalence,
        majority_baseline_accuracy,
        accuracy: (tp + tn) as f32 / n.max(1) as f32,
        accuracy_wilson_ci95: wilson_ci95(tp + tn, n),
        recall_positive,
        recall_negative,
        balanced_accuracy,
        auroc: auroc_binary(probs, labels),
    }
}

/// The 3-way analogue of [`BinaryMetrics`] for `direction`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DirectionMetrics {
    pub n: usize,
    pub class_prevalence: [f32; 3],
    pub majority_baseline_accuracy: f32,
    pub accuracy: f32,
    pub accuracy_wilson_ci95: (f32, f32),
    pub per_class_recall: [f32; 3],
    pub balanced_accuracy: f32,
}

fn direction_metrics(pred_probs: &[[f32; 3]], labels: &[u8]) -> DirectionMetrics {
    let n = labels.len();
    let mut class_count = [0usize; 3];
    for &l in labels {
        class_count[l as usize] += 1;
    }
    let class_prevalence: [f32; 3] = std::array::from_fn(|c| class_count[c] as f32 / n.max(1) as f32);
    let majority_baseline_accuracy = class_prevalence.iter().copied().fold(0.0f32, f32::max);
    let mut correct = 0usize;
    let mut per_class_correct = [0usize; 3];
    for (probs, &y) in pred_probs.iter().zip(labels) {
        let pred = (0..3)
            .max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap())
            .unwrap() as u8;
        if pred == y {
            correct += 1;
            per_class_correct[y as usize] += 1;
        }
    }
    let per_class_recall: [f32; 3] = std::array::from_fn(|c| {
        if class_count[c] > 0 {
            per_class_correct[c] as f32 / class_count[c] as f32
        } else {
            f32::NAN
        }
    });
    let finite_recalls: Vec<f32> = per_class_recall.iter().copied().filter(|r| r.is_finite()).collect();
    let balanced_accuracy = if finite_recalls.is_empty() {
        f32::NAN
    } else {
        finite_recalls.iter().sum::<f32>() / finite_recalls.len() as f32
    };
    DirectionMetrics {
        n,
        class_prevalence,
        majority_baseline_accuracy,
        accuracy: correct as f32 / n.max(1) as f32,
        accuracy_wilson_ci95: wilson_ci95(correct, n),
        per_class_recall,
        balanced_accuracy,
    }
}

/// Every head this demo has a teacher signal for (see the module doc comment's "Scope" note).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct HeadMetrics {
    pub direction: DirectionMetrics,
    pub jump: BinaryMetrics,
    pub hook: BinaryMetrics,
}

/// Runs the fly pipeline (encoder -> `t_decisions` fly decisions from `v_init` -> decoder) over
/// `held_out` and reports [`HeadMetrics`] against `held_out`'s own `scripted_teacher` labels.
/// Public (review round 1, F2) so a caller can re-run this against maps/seeds
/// [`run_brain_demo`] never saw during training — see `tests/brain_demo_generalization.rs`.
///
/// Review round 1, F15 (CONFIRMED): reads each decision's `DecisionOutput::dn_rates` (the same
/// `0.5 * (r_before_last + r_last)` average `crate::brain_train::brain_train_step`/
/// `crate::brain::FlyBrain::decide` both use), not `state.v()` alone after the loop (which is
/// `r_last` only, silently a different, untrained-for readout).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_fly(
    model: &FlyModel,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &DecoderParams,
    calib: &DnCalibration,
    v_init: &[f32],
    t_decisions: usize,
    held_out: &[Observation],
    held_out_labels: &[TeacherLabels],
) -> HeadMetrics {
    let mut input_buf = vec![0.0f32; encoder.num_inputs()];
    let mut features = RayGridFeatures::new(encoder.ray_grid_config());
    let mut scratch = DecoderScratch::new(decoder);
    let mut dn_rates = vec![0.0f32; model.num_outputs()];

    let mut dir_probs = Vec::with_capacity(held_out.len());
    let mut jump_probs = Vec::with_capacity(held_out.len());
    let mut hook_probs = Vec::with_capacity(held_out.len());
    for obs in held_out {
        let mut state = FlyState::new(model);
        state.set_v(model, v_init);
        for _ in 0..t_decisions {
            let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
            features.compute(obs, encoder.ray_grid_config());
            encoder.forward(&features, &an, encoder_params, &mut input_buf);
            let out = state.step_decision(model, &input_buf);
            dn_rates.copy_from_slice(out.dn_rates);
        }
        let decoded = decoder_forward_into(decoder, &dn_rates, calib, decoder_params, &mut scratch, false);
        dir_probs.push(decoded.direction_probs);
        jump_probs.push(decoded.jump_prob);
        hook_probs.push(decoded.hook_prob);
    }

    let dir_labels: Vec<u8> = held_out_labels.iter().map(|l| l.direction).collect();
    let jump_labels: Vec<bool> = held_out_labels.iter().map(|l| l.jump).collect();
    let hook_labels: Vec<bool> = held_out_labels.iter().map(|l| l.hook).collect();
    HeadMetrics {
        direction: direction_metrics(&dir_probs, &dir_labels),
        jump: binary_metrics(&jump_probs, &jump_labels),
        hook: binary_metrics(&hook_probs, &hook_labels),
    }
}

/// The MLP control's analogue of [`evaluate_fly`] — same held-out set, same [`HeadMetrics`] shape,
/// so the two are directly comparable (review round 1, F2).
pub fn evaluate_mlp(
    mlp: &MlpParams,
    ray_grid_config: &crate::encoder::RayGridConfig,
    held_out: &[Observation],
    held_out_labels: &[TeacherLabels],
) -> HeadMetrics {
    let mut dir_probs = Vec::with_capacity(held_out.len());
    let mut jump_probs = Vec::with_capacity(held_out.len());
    let mut hook_probs = Vec::with_capacity(held_out.len());
    for obs in held_out {
        let an = compute_proprioception_values(&obs.self_state, ray_grid_config);
        let mut features = RayGridFeatures::new(ray_grid_config);
        features.compute(obs, ray_grid_config);
        let x = flat_features(&features, &an);
        let pred = mlp_forward(mlp, &x);
        dir_probs.push(pred.direction_probs);
        jump_probs.push(pred.jump_prob);
        hook_probs.push(pred.hook_prob);
    }
    let dir_labels: Vec<u8> = held_out_labels.iter().map(|l| l.direction).collect();
    let jump_labels: Vec<bool> = held_out_labels.iter().map(|l| l.jump).collect();
    let hook_labels: Vec<bool> = held_out_labels.iter().map(|l| l.hook).collect();
    HeadMetrics {
        direction: direction_metrics(&dir_probs, &dir_labels),
        jump: binary_metrics(&jump_probs, &jump_labels),
        hook: binary_metrics(&hook_probs, &hook_labels),
    }
}

// --- Training ---------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct StepMetric {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub fly_applied: bool,
    pub fly_lr_scale: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DemoReport {
    pub fly_num_params: usize,
    pub mlp_num_params: usize,
    pub held_out_n: usize,
    pub metrics_before_fly: HeadMetrics,
    pub metrics_after_fly: HeadMetrics,
    pub metrics_after_mlp: HeadMetrics,
    /// Review round 1, F12: how many of the guarded encoder/decoder Adam steps (out of
    /// `steps * 13` — 2 encoder + 11 decoder param groups per training step) were skipped because
    /// a gradient or the resulting params weren't finite. Nonzero here on an otherwise-healthy run
    /// is worth investigating (a symptom of too-high a learning rate or a genuine upstream bug),
    /// but a skip itself never corrupts training (see `crate::flat_adam::FlatAdamState::
    /// step_guarded`'s doc comment) — reported for transparency, not as a pass/fail signal.
    pub guarded_steps_skipped: usize,
    pub metrics: Vec<StepMetric>,
    /// The trained MLP control's own parameters (review round 1, F2) — exposed so a caller can
    /// re-run [`evaluate_mlp`] against maps/seeds this call never saw, the same way it can re-run
    /// [`evaluate_fly`] against the (already-mutated-in-place) `encoder_params`/`decoder_params`
    /// this function was given; see `tests/brain_demo_generalization.rs`.
    pub mlp: MlpParams,
}

/// Every encoder/decoder Adam state, one [`FlatAdamState`] per flat parameter group, all stepped
/// through [`FlatAdamState::step_guarded`] (review round 1, F12 — an earlier revision used the
/// plain, unguarded [`FlatAdamState::step`] here, so one non-finite gradient anywhere in this
/// group would have permanently corrupted that state's Adam moments for the rest of the run).
struct EncoderDecoderAdam {
    encoder_g: FlatAdamState,
    encoder_c: FlatAdamState,
    direction_lr_w: FlatAdamState,
    direction_lr_b: FlatAdamState,
    direction_stop_w: FlatAdamState,
    direction_stop_b: FlatAdamState,
    jump_w: FlatAdamState,
    jump_b: FlatAdamState,
    hook_w: FlatAdamState,
    hook_b: FlatAdamState,
    fire_w: FlatAdamState,
    fire_b: FlatAdamState,
    aim_pair_theta: FlatAdamState,
    aim_unpaired_theta: FlatAdamState,
}

impl EncoderDecoderAdam {
    fn new(encoder_params: &EncoderParams, decoder_params: &DecoderParams) -> Self {
        EncoderDecoderAdam {
            encoder_g: FlatAdamState::new(encoder_params.g.len()),
            encoder_c: FlatAdamState::new(encoder_params.c.len()),
            direction_lr_w: FlatAdamState::new(decoder_params.direction_lr_w.len()),
            direction_lr_b: FlatAdamState::new(1),
            direction_stop_w: FlatAdamState::new(decoder_params.direction_stop_w.len()),
            direction_stop_b: FlatAdamState::new(1),
            jump_w: FlatAdamState::new(decoder_params.jump_w.len()),
            jump_b: FlatAdamState::new(1),
            hook_w: FlatAdamState::new(decoder_params.hook_w.len()),
            hook_b: FlatAdamState::new(1),
            fire_w: FlatAdamState::new(decoder_params.fire_w.len()),
            fire_b: FlatAdamState::new(1),
            aim_pair_theta: FlatAdamState::new(decoder_params.aim_pair_theta.len()),
            aim_unpaired_theta: FlatAdamState::new(decoder_params.aim_unpaired_theta.len()),
        }
    }

    /// Steps every group, returns how many of the 13 (2 encoder + 11 decoder) were skipped.
    fn step(
        &mut self,
        encoder_params: &mut EncoderParams,
        encoder_grad: &crate::encoder::EncoderGradients,
        encoder_cfg: &FlatAdamConfig,
        decoder_params: &mut DecoderParams,
        decoder_grad: &crate::decoder::DecoderGradients,
        decoder_cfg: &FlatAdamConfig,
    ) -> usize {
        let mut skipped = 0usize;
        macro_rules! step_vec {
            ($state:ident, $params:expr, $grad:expr, $cfg:expr) => {
                if !self.$state.step_guarded($params, $grad, $cfg) {
                    skipped += 1;
                }
            };
        }
        step_vec!(encoder_g, &mut encoder_params.g, &encoder_grad.g, encoder_cfg);
        step_vec!(encoder_c, &mut encoder_params.c, &encoder_grad.c, encoder_cfg);
        step_vec!(
            direction_lr_w,
            &mut decoder_params.direction_lr_w,
            &decoder_grad.direction_lr_w,
            decoder_cfg
        );
        step_vec!(
            direction_lr_b,
            std::slice::from_mut(&mut decoder_params.direction_lr_b),
            &[decoder_grad.direction_lr_b],
            decoder_cfg
        );
        step_vec!(
            direction_stop_w,
            &mut decoder_params.direction_stop_w,
            &decoder_grad.direction_stop_w,
            decoder_cfg
        );
        step_vec!(
            direction_stop_b,
            std::slice::from_mut(&mut decoder_params.direction_stop_b),
            &[decoder_grad.direction_stop_b],
            decoder_cfg
        );
        step_vec!(jump_w, &mut decoder_params.jump_w, &decoder_grad.jump_w, decoder_cfg);
        step_vec!(
            jump_b,
            std::slice::from_mut(&mut decoder_params.jump_b),
            &[decoder_grad.jump_b],
            decoder_cfg
        );
        step_vec!(hook_w, &mut decoder_params.hook_w, &decoder_grad.hook_w, decoder_cfg);
        step_vec!(
            hook_b,
            std::slice::from_mut(&mut decoder_params.hook_b),
            &[decoder_grad.hook_b],
            decoder_cfg
        );
        step_vec!(fire_w, &mut decoder_params.fire_w, &decoder_grad.fire_w, decoder_cfg);
        step_vec!(
            fire_b,
            std::slice::from_mut(&mut decoder_params.fire_b),
            &[decoder_grad.fire_b],
            decoder_cfg
        );
        step_vec!(
            aim_pair_theta,
            &mut decoder_params.aim_pair_theta,
            &decoder_grad.aim_pair_theta,
            decoder_cfg
        );
        step_vec!(
            aim_unpaired_theta,
            &mut decoder_params.aim_unpaired_theta,
            &decoder_grad.aim_unpaired_theta,
            decoder_cfg
        );
        skipped
    }
}

/// Runs the whole demo: builds a warmed rest state, then `config.steps` batches of
/// `config.batch_size` synthetic samples each, training encoder + fly + decoder with
/// `guarded_adam_step` (fly) / guarded `crate::flat_adam` (encoder, decoder) — and, alongside, an
/// MLP control of comparable size on the same raw features. Returns held-out [`HeadMetrics`]
/// before/after training for both (see [`evaluate_fly`]/[`evaluate_mlp`]), plus the per-step loss/
/// grad-norm curve.
#[allow(clippy::too_many_arguments)]
pub fn run_brain_demo(
    model: &mut FlyModel,
    index: &BackwardIndex,
    encoder: &EncoderModel,
    encoder_params: &mut EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &mut DecoderParams,
    calib: &DnCalibration,
    map: Arc<ddai_physics::map::MapData>,
    config: &BrainDemoConfig,
) -> DemoReport {
    let mut rng = SplitMix64::new(config.seed);

    let mut warm = FlyState::new(model);
    let warm_report = warm.warm_up(model);
    if !warm_report.converged {
        eprintln!("warning: warm-up did not converge within the default cap; proceeding with its best-effort state");
    }
    let v_init = warm.v().to_vec();

    // --- Held-out evaluation set (fixed, seed offset so it never overlaps training draws) ---
    let mut eval_rng = SplitMix64::new(config.seed.wrapping_add(0x5EED));
    let mut held_out = Vec::with_capacity(config.held_out_samples);
    let mut held_out_labels = Vec::with_capacity(config.held_out_samples);
    for _ in 0..config.held_out_samples {
        let (obs, labels) = sample_scenario_stratified(Arc::clone(&map), config, &mut eval_rng);
        held_out.push(obs);
        held_out_labels.push(labels);
    }

    let metrics_before_fly = evaluate_fly(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        &v_init,
        config.t_decisions,
        &held_out,
        &held_out_labels,
    );

    // --- Adam state for every trainable group ---
    let mut fly_adam = GuardedAdamState::new(model.params());
    let guarded_cfg = GuardedAdamConfig {
        adam: crate::optim::AdamConfig {
            lr_a: config.lr_fly,
            lr_b: config.lr_fly,
            lr_theta: config.lr_fly,
            ..crate::optim::AdamConfig::default()
        },
        ..GuardedAdamConfig::default()
    };
    let mut ed_adam = EncoderDecoderAdam::new(encoder_params, decoder_params);
    let encoder_adam_cfg = FlatAdamConfig {
        lr: config.lr_encoder,
        ..FlatAdamConfig::default()
    };
    let decoder_adam_cfg = FlatAdamConfig {
        lr: config.lr_decoder,
        ..FlatAdamConfig::default()
    };
    let mut guarded_steps_skipped = 0usize;

    // The world model is never trained in this demo (`world_model: None` on every decision's
    // targets below), so one fixed head/params pair suffices — built once, not per sample.
    let world_model =
        crate::world_model::WorldModelHead::new(model, &crate::world_model::WorldModelConfig::default()).unwrap();
    let world_model_params = world_model.init_default_params();

    let mut metrics = Vec::with_capacity(config.steps);
    for step in 0..config.steps {
        let mut batch_loss = 0.0f32;
        let mut fly_grad = ParamGradients::zeros_like(model.params());
        let mut encoder_grad = crate::encoder::EncoderGradients::zeros(encoder.num_params());
        let mut decoder_grad = decoder.zeros_gradients();

        for _ in 0..config.batch_size {
            let (obs, labels) = sample_scenario_stratified(Arc::clone(&map), config, &mut rng);
            let decoder_targets = DecoderTargets {
                direction: Some(labels.direction),
                jump: Some(labels.jump),
                hook: Some(labels.hook),
                fire: None,
                aim: None,
            };
            let mut targets = vec![DecisionTargets::default(); config.t_decisions];
            targets[config.t_decisions - 1] = DecisionTargets {
                decoder: decoder_targets,
                world_model: None,
            };
            // Static scenario repeated across the window (no physics stepping between
            // decisions in this synthetic demo — see the module doc comment).
            let observations = vec![obs; config.t_decisions];
            let seq = BrainSequence {
                v_init: v_init.clone(),
                observations,
                targets,
            };
            let (loss, grads, _final_v) = brain_train_step(
                model,
                index,
                encoder,
                encoder_params,
                decoder,
                decoder_params,
                calib,
                &world_model,
                &world_model_params,
                &seq,
            );
            batch_loss += loss;
            fly_grad.add_assign(&grads.fly);
            add_encoder_gradients(&mut encoder_grad, &grads.encoder);
            add_decoder_gradients_local(&mut decoder_grad, &grads.decoder);
        }
        let inv_batch = 1.0 / config.batch_size as f32;
        fly_grad.scale(inv_batch);
        for g in encoder_grad.g.iter_mut().chain(&mut encoder_grad.c) {
            *g *= inv_batch;
        }
        scale_decoder_gradients(&mut decoder_grad, inv_batch);
        batch_loss *= inv_batch;

        let grad_norm = clip_grad_norm(&mut fly_grad, config.grad_clip_norm);
        clip_grad_norm_multi(
            &mut [
                &mut encoder_grad.g,
                &mut encoder_grad.c,
                &mut decoder_grad.direction_lr_w,
                std::slice::from_mut(&mut decoder_grad.direction_lr_b),
                &mut decoder_grad.direction_stop_w,
                std::slice::from_mut(&mut decoder_grad.direction_stop_b),
                &mut decoder_grad.jump_w,
                std::slice::from_mut(&mut decoder_grad.jump_b),
                &mut decoder_grad.hook_w,
                std::slice::from_mut(&mut decoder_grad.hook_b),
                &mut decoder_grad.fire_w,
                std::slice::from_mut(&mut decoder_grad.fire_b),
                &mut decoder_grad.aim_pair_theta,
                &mut decoder_grad.aim_unpaired_theta,
            ],
            config.grad_clip_norm,
        );

        // `guarded_adam_step` mutates its `params` argument in place — since `model`'s own
        // `FlyParams` live behind `set_params` (which must run to recompute derived weights/
        // bias/decay, never mutated directly), the candidate is built on a scratch clone here and
        // only committed to `model` itself once the guard has actually accepted it.
        let mut fly_params_candidate = model.params().clone();
        let outcome = crate::optim::guarded_adam_step(
            &mut fly_params_candidate,
            &fly_grad,
            &mut fly_adam,
            &guarded_cfg,
            |candidate| {
                let mut probe_model = model.clone();
                if probe_model.set_params(candidate.clone()).is_err() {
                    return false;
                }
                // Reuses the already-converged `warm` state (not a fresh re-convergence search —
                // cheap, and the guard only needs "does one step from a sane state stay finite", not
                // "did this candidate's own resting point converge") against the candidate's weights.
                let mut probe_state = warm.clone();
                let out = probe_state.step_decision(&probe_model, &vec![0.0; probe_model.num_inputs()]);
                out.dn_rates.iter().all(|x| x.is_finite())
            },
        );
        let (fly_applied, fly_lr_scale) = match outcome {
            crate::optim::GuardedStepOutcome::Applied => {
                model
                    .set_params(fly_params_candidate)
                    .expect("a validated candidate must still apply");
                (true, fly_adam.lr_scale)
            }
            crate::optim::GuardedStepOutcome::SkippedNonFiniteGradient => (false, fly_adam.lr_scale),
            crate::optim::GuardedStepOutcome::RolledBack { lr_scale_after } => (false, lr_scale_after),
        };

        guarded_steps_skipped += ed_adam.step(
            encoder_params,
            &encoder_grad,
            &encoder_adam_cfg,
            decoder_params,
            &decoder_grad,
            &decoder_adam_cfg,
        );

        metrics.push(StepMetric {
            step,
            loss: batch_loss,
            grad_norm,
            fly_applied,
            fly_lr_scale,
        });
    }

    let metrics_after_fly = evaluate_fly(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        &v_init,
        config.t_decisions,
        &held_out,
        &held_out_labels,
    );

    // --- MLP control ---
    let input_dim = flat_features(
        &RayGridFeatures::new(encoder.ray_grid_config()),
        &compute_proprioception_values(&held_out[0].self_state, encoder.ray_grid_config()),
    )
    .len();
    let fly_num_params = model.params().a.len()
        + model.params().b.len()
        + model.params().theta.len()
        + encoder.num_params() * 2
        + decoder_params.direction_lr_w.len()
        + 1 // direction_lr_b
        + decoder_params.direction_stop_w.len()
        + 1 // direction_stop_b
        + decoder_params.jump_w.len()
        + 1 // jump_b
        + decoder_params.hook_w.len()
        + 1; // hook_b
    let target_mlp_hidden = (fly_num_params as f32 / (input_dim as f32 + 5.0)).round().max(1.0) as usize;
    let mut mlp = MlpParams::init(input_dim, target_mlp_hidden, &mut rng);
    let mlp_num_params = mlp.num_params();
    let mut mlp_w1 = FlatAdamState::new(mlp.w1.len());
    let mut mlp_b1 = FlatAdamState::new(mlp.b1.len());
    let mut mlp_wdir = FlatAdamState::new(mlp.w_dir.len());
    let mut mlp_bdir = FlatAdamState::new(3);
    let mut mlp_wjump = FlatAdamState::new(mlp.w_jump.len());
    let mut mlp_bjump = FlatAdamState::new(1);
    let mut mlp_whook = FlatAdamState::new(mlp.w_hook.len());
    let mut mlp_bhook = FlatAdamState::new(1);
    let mlp_adam_cfg = FlatAdamConfig::default();

    for _ in 0..config.steps {
        let mut acc = MlpGradients::zeros(&mlp);
        for _ in 0..config.batch_size {
            let (obs, labels) = sample_scenario_stratified(Arc::clone(&map), config, &mut rng);
            let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
            let mut features = RayGridFeatures::new(encoder.ray_grid_config());
            features.compute(&obs, encoder.ray_grid_config());
            let x = flat_features(&features, &an);
            let (_loss, g) = mlp_loss_and_grad(&mlp, &x, labels);
            for (a, b) in acc.w1.iter_mut().zip(&g.w1) {
                *a += b;
            }
            for (a, b) in acc.b1.iter_mut().zip(&g.b1) {
                *a += b;
            }
            for (a, b) in acc.w_dir.iter_mut().zip(&g.w_dir) {
                *a += b;
            }
            for i in 0..3 {
                acc.b_dir[i] += g.b_dir[i];
            }
            for (a, b) in acc.w_jump.iter_mut().zip(&g.w_jump) {
                *a += b;
            }
            acc.b_jump += g.b_jump;
            for (a, b) in acc.w_hook.iter_mut().zip(&g.w_hook) {
                *a += b;
            }
            acc.b_hook += g.b_hook;
        }
        let inv = 1.0 / config.batch_size as f32;
        for x in acc
            .w1
            .iter_mut()
            .chain(&mut acc.b1)
            .chain(&mut acc.w_dir)
            .chain(&mut acc.w_jump)
            .chain(&mut acc.w_hook)
        {
            *x *= inv;
        }
        acc.b_jump *= inv;
        acc.b_hook *= inv;
        for x in &mut acc.b_dir {
            *x *= inv;
        }
        mlp_w1.step(&mut mlp.w1, &acc.w1, &mlp_adam_cfg);
        mlp_b1.step(&mut mlp.b1, &acc.b1, &mlp_adam_cfg);
        mlp_wdir.step(&mut mlp.w_dir, &acc.w_dir, &mlp_adam_cfg);
        mlp_bdir.step(&mut mlp.b_dir, &acc.b_dir, &mlp_adam_cfg);
        mlp_wjump.step(&mut mlp.w_jump, &acc.w_jump, &mlp_adam_cfg);
        mlp_bjump.step(std::slice::from_mut(&mut mlp.b_jump), &[acc.b_jump], &mlp_adam_cfg);
        mlp_whook.step(&mut mlp.w_hook, &acc.w_hook, &mlp_adam_cfg);
        mlp_bhook.step(std::slice::from_mut(&mut mlp.b_hook), &[acc.b_hook], &mlp_adam_cfg);
    }

    let metrics_after_mlp = evaluate_mlp(&mlp, encoder.ray_grid_config(), &held_out, &held_out_labels);

    DemoReport {
        fly_num_params,
        mlp_num_params,
        held_out_n: held_out.len(),
        metrics_before_fly,
        metrics_after_fly,
        metrics_after_mlp,
        guarded_steps_skipped,
        metrics,
        mlp,
    }
}

fn add_encoder_gradients(acc: &mut crate::encoder::EncoderGradients, g: &crate::encoder::EncoderGradients) {
    for (a, b) in acc.g.iter_mut().zip(&g.g) {
        *a += b;
    }
    for (a, b) in acc.c.iter_mut().zip(&g.c) {
        *a += b;
    }
}

fn add_decoder_gradients_local(acc: &mut crate::decoder::DecoderGradients, g: &crate::decoder::DecoderGradients) {
    for (a, b) in acc.direction_lr_w.iter_mut().zip(&g.direction_lr_w) {
        *a += b;
    }
    acc.direction_lr_b += g.direction_lr_b;
    for (a, b) in acc.direction_stop_w.iter_mut().zip(&g.direction_stop_w) {
        *a += b;
    }
    acc.direction_stop_b += g.direction_stop_b;
    for (a, b) in acc.jump_w.iter_mut().zip(&g.jump_w) {
        *a += b;
    }
    acc.jump_b += g.jump_b;
    for (a, b) in acc.hook_w.iter_mut().zip(&g.hook_w) {
        *a += b;
    }
    acc.hook_b += g.hook_b;
    for (a, b) in acc.fire_w.iter_mut().zip(&g.fire_w) {
        *a += b;
    }
    acc.fire_b += g.fire_b;
    for (a, b) in acc.aim_pair_theta.iter_mut().zip(&g.aim_pair_theta) {
        *a += b;
    }
    for (a, b) in acc.aim_unpaired_theta.iter_mut().zip(&g.aim_unpaired_theta) {
        *a += b;
    }
}

fn scale_decoder_gradients(g: &mut crate::decoder::DecoderGradients, scale: f32) {
    for x in g
        .direction_lr_w
        .iter_mut()
        .chain(&mut g.direction_stop_w)
        .chain(&mut g.jump_w)
        .chain(&mut g.hook_w)
        .chain(&mut g.fire_w)
        .chain(&mut g.aim_pair_theta)
        .chain(&mut g.aim_unpaired_theta)
    {
        *x *= scale;
    }
    g.direction_lr_b *= scale;
    g.direction_stop_b *= scale;
    g.jump_b *= scale;
    g.hook_b *= scale;
    g.fire_b *= scale;
}

#[cfg(test)]
mod tests;
