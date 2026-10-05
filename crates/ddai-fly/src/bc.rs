//! Behaviour-cloning loss on the action heads (task 8.2), written in **logit space** so the fly's
//! tied decoder and the MLP/GRU controls (`ddai-controls`) share one definition of the loss and of
//! what a "head" is: 3-way direction (softmax), jump/hook/fire (sigmoid), aim (a 2-vector `(c, s)`
//! whose angle `atan2(s, c)` is scored by the von Mises NLL, exactly the decoder's
//! `-kappa * cos(target - mu)`).
//!
//! What this adds over [`crate::decoder::decoder_loss_and_grad`] (which is left untouched):
//! * **soft targets**: the planner teacher's elite-set frequencies (`ddai-planner`'s
//!   `EliteFirstStep`) mixed with the hard label by [`LossConfig::soft_mix`];
//! * **per-head weights and per-head masks** (human data has no soft target; aim only matters when
//!   a hook or shot leaves the tee, so its loss is masked otherwise);
//! * **positive-class weights** for the rare binary heads;
//! * a **per-step weight** (technique upsampling, burn-in steps at `0`).

use serde::{Deserialize, Serialize};

use crate::activation::sigmoid;

/// The five action heads' pre-activation outputs for one decision.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadLogits {
    /// `[left, stop, right]` logits.
    pub dir: [f32; 3],
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
    /// The aim population vector `(C, S)`; the predicted ring angle is `atan2(S, C)`.
    pub aim_c: f32,
    pub aim_s: f32,
}

impl HeadLogits {
    pub fn dir_probs(&self) -> [f32; 3] {
        softmax3(self.dir)
    }
    pub fn jump_prob(&self) -> f32 {
        sigmoid(self.jump)
    }
    pub fn hook_prob(&self) -> f32 {
        sigmoid(self.hook)
    }
    pub fn fire_prob(&self) -> f32 {
        sigmoid(self.fire)
    }
    pub fn aim_angle(&self) -> f32 {
        self.aim_s.atan2(self.aim_c)
    }
}

/// Decision thresholds of the three binary heads under argmax selection: the key is pressed when
/// `p >= threshold`. `0.5` is the plain argmax of a head; a head trained with a positive-class weight
/// presses far more often than the teacher at `0.5` (E-005, review F5), so a trained model carries
/// thresholds calibrated on validation data (`ddai-train`: the rate-matched threshold). A model
/// without calibrated thresholds (every bundle written before format v2) uses the default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HeadThresholds {
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
}

impl Default for HeadThresholds {
    fn default() -> Self {
        HeadThresholds {
            jump: 0.5,
            hook: 0.5,
            fire: 0.5,
        }
    }
}

impl HeadThresholds {
    /// Every threshold must be a probability strictly inside `(0, 1)`.
    pub fn validate(&self) -> Result<(), String> {
        for (name, t) in [("jump", self.jump), ("hook", self.hook), ("fire", self.fire)] {
            if !(t.is_finite() && t > 0.0 && t < 1.0) {
                return Err(format!("{name} threshold {t} is not inside (0, 1)"));
            }
        }
        Ok(())
    }
}

impl HeadLogits {
    pub fn jump_on(&self, th: &HeadThresholds) -> bool {
        self.jump_prob() >= th.jump
    }
    pub fn hook_on(&self, th: &HeadThresholds) -> bool {
        self.hook_prob() >= th.hook
    }
    pub fn fire_on(&self, th: &HeadThresholds) -> bool {
        self.fire_prob() >= th.fire
    }
}

/// How a trained model sees the **own hook state** (the `own_hook` proprioception input) in the hook head.
///
/// A hook head that has the input learns "hook <=> my hook is already out" and never starts or releases a
/// hook (E-005 review F2). `MaskedForHookHead` trains and plays the model in two views: the direction, jump,
/// fire and aim heads read the full observation, the hook head reads the same observation with the own
/// hook state hidden (it then has to decide from the opponent, the rays and the tiles). It costs two passes
/// of the network per decision (two recurrent states for a stateful model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HookView {
    #[default]
    Shared,
    MaskedForHookHead,
}

/// `obs` with the own hook state hidden (reported as idle): what the hook head sees under
/// [`HookView::MaskedForHookHead`]. Nothing but the `own_hook` proprioception input reads that field.
pub fn mask_own_hook(obs: &ddai_brain::Observation) -> ddai_brain::Observation {
    let mut o = obs.clone();
    o.self_state.hook_state = ddai_brain::HOOK_IDLE;
    o
}

/// The logits of the two views combined: the hook head from `masked`, everything else from `full`.
pub fn combine_hook_view(full: &HeadLogits, masked: &HeadLogits) -> HeadLogits {
    HeadLogits {
        hook: masked.hook,
        ..*full
    }
}

/// `softmax` over three logits.
pub fn softmax3(l: [f32; 3]) -> [f32; 3] {
    let m = l[0].max(l[1]).max(l[2]);
    let e = [(l[0] - m).exp(), (l[1] - m).exp(), (l[2] - m).exp()];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

/// The teacher's soft target for one decision: the elite set's first-step statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftTargets {
    /// `[left, stop, right]` fractions, summing to one.
    pub dir: [f32; 3],
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
}

/// Which heads carry a loss on a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadMask {
    pub dir: bool,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    pub aim: bool,
}

impl HeadMask {
    pub const ALL: HeadMask = HeadMask {
        dir: true,
        jump: true,
        hook: true,
        fire: true,
        aim: true,
    };
    pub const NONE: HeadMask = HeadMask {
        dir: false,
        jump: false,
        hook: false,
        fire: false,
        aim: false,
    };
}

/// What one decision is trained towards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepTargets {
    /// `0` left, `1` stop, `2` right.
    pub dir: u8,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    /// Ring-convention angle (radians) of the aim vector; only scored when `mask.aim`.
    pub aim: f32,
    pub soft: Option<SoftTargets>,
    pub mask: HeadMask,
    /// Multiplies this decision's whole loss and gradient. `0` = no loss (burn-in).
    pub weight: f32,
    /// Multiplies only the hook head's loss on this decision (`1` = nothing). Used to emphasise the
    /// *start* and *release* decisions (label differs from the own hook being out), which a head that
    /// copies the own-hook input gets wrong (E-005 review F2).
    pub hook_scale: f32,
}

impl StepTargets {
    /// A decision with no loss (a burn-in step: the network runs, nothing is scored).
    pub fn burn_in() -> Self {
        StepTargets {
            dir: 1,
            jump: false,
            hook: false,
            fire: false,
            aim: 0.0,
            soft: None,
            mask: HeadMask::NONE,
            weight: 0.0,
            hook_scale: 1.0,
        }
    }
}

/// Head weights and shaping of the loss; identical for the fly, the MLP and the GRU.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LossConfig {
    pub w_dir: f32,
    pub w_jump: f32,
    pub w_hook: f32,
    pub w_fire: f32,
    pub w_aim: f32,
    /// `0` = hard labels only, `1` = soft targets only (where a decision has one; a decision
    /// without a soft target always uses its hard label).
    pub soft_mix: f32,
    /// Positive-class weights of the binary heads (jump, hook, fire).
    pub pos_weight: [f32; 3],
    /// Von Mises concentration of the aim loss (the decoder's `aim_kappa`).
    pub aim_kappa: f32,
    /// Smoothing applied to soft targets so a unanimous elite set is not a probability-one target.
    pub soft_smoothing: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        LossConfig {
            w_dir: 1.0,
            w_jump: 1.0,
            w_hook: 1.0,
            w_fire: 1.0,
            w_aim: 1.0,
            soft_mix: 0.5,
            pos_weight: [1.0, 1.0, 1.0],
            aim_kappa: 4.0,
            soft_smoothing: 0.02,
        }
    }
}

/// Per-head loss values of one decision, already multiplied by the head weight and the step
/// weight (so `total` is what the optimiser minimises).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StepLoss {
    pub total: f32,
    pub dir: f32,
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
    pub aim: f32,
}

impl StepLoss {
    pub fn add(&mut self, o: &StepLoss) {
        self.total += o.total;
        self.dir += o.dir;
        self.jump += o.jump;
        self.hook += o.hook;
        self.fire += o.fire;
        self.aim += o.aim;
    }
    pub fn scale(&mut self, k: f32) {
        self.total *= k;
        self.dir *= k;
        self.jump *= k;
        self.hook *= k;
        self.fire *= k;
        self.aim *= k;
    }
}

/// The von Mises NLL `-kappa * cos(target - mu)`, `mu = atan2(s, c)`, and its gradient with
/// respect to `(c, s)`; a zero gradient at the degenerate `(0, 0)` (same convention as the
/// decoder's own aim loss).
pub fn aim_nll_and_grad(c: f32, s: f32, target: f32, kappa: f32) -> (f32, f32, f32) {
    let r2 = c * c + s * s;
    if r2 < 1e-12 {
        return (kappa, 0.0, 0.0);
    }
    let mu = s.atan2(c);
    let loss = -kappa * (target - mu).cos();
    let d_mu = -kappa * (target - mu).sin();
    (loss, d_mu * (-s / r2), d_mu * (c / r2))
}

/// Mixes a hard 0/1 label with a soft probability, smoothing the soft part.
fn mixed_binary_target(hard: bool, soft: Option<f32>, mix: f32, smoothing: f32) -> f32 {
    let y = f32::from(hard);
    match soft {
        Some(q) => {
            let q = q * (1.0 - smoothing) + 0.5 * smoothing;
            (1.0 - mix) * y + mix * q
        }
        None => y,
    }
}

/// Weighted BCE with a soft target `t` and positive weight `w`; returns `(loss, d loss / d logit)`.
fn weighted_bce(logit: f32, t: f32, pos_weight: f32) -> (f32, f32) {
    let p = sigmoid(logit);
    let loss = -(pos_weight * t * p.max(1e-12).ln() + (1.0 - t) * (1.0 - p).max(1e-12).ln());
    (loss, p * (pos_weight * t + 1.0 - t) - pos_weight * t)
}

/// The loss of one decision and its gradient with respect to the head logits.
///
/// A decision with `weight == 0` or an all-false mask returns zeros. Everything the caller needs
/// to log is in [`StepLoss`]; the gradient is already scaled by the head weights and `weight`.
pub fn head_loss_and_grad(logits: &HeadLogits, t: &StepTargets, cfg: &LossConfig) -> (StepLoss, HeadLogits) {
    let mut loss = StepLoss::default();
    let mut d = HeadLogits::default();
    if t.weight == 0.0 {
        return (loss, d);
    }
    let mix = cfg.soft_mix.clamp(0.0, 1.0);

    if t.mask.dir {
        let p = logits.dir_probs();
        let mut target = [0.0f32; 3];
        target[t.dir.min(2) as usize] = 1.0;
        if let Some(s) = &t.soft {
            for (i, tv) in target.iter_mut().enumerate() {
                let q = s.dir[i] * (1.0 - cfg.soft_smoothing) + cfg.soft_smoothing / 3.0;
                *tv = (1.0 - mix) * *tv + mix * q;
            }
        }
        let ce: f32 = target.iter().zip(&p).map(|(&tv, &pv)| -tv * pv.max(1e-12).ln()).sum();
        let w = cfg.w_dir * t.weight;
        loss.dir = w * ce;
        for i in 0..3 {
            d.dir[i] = w * (p[i] - target[i]);
        }
    }
    let soft = t.soft.as_ref();
    if t.mask.jump {
        let tv = mixed_binary_target(t.jump, soft.map(|s| s.jump), mix, cfg.soft_smoothing);
        let (l, g) = weighted_bce(logits.jump, tv, cfg.pos_weight[0]);
        let w = cfg.w_jump * t.weight;
        loss.jump = w * l;
        d.jump = w * g;
    }
    if t.mask.hook {
        let tv = mixed_binary_target(t.hook, soft.map(|s| s.hook), mix, cfg.soft_smoothing);
        let (l, g) = weighted_bce(logits.hook, tv, cfg.pos_weight[1]);
        let w = cfg.w_hook * t.weight * t.hook_scale;
        loss.hook = w * l;
        d.hook = w * g;
    }
    if t.mask.fire {
        let tv = mixed_binary_target(t.fire, soft.map(|s| s.fire), mix, cfg.soft_smoothing);
        let (l, g) = weighted_bce(logits.fire, tv, cfg.pos_weight[2]);
        let w = cfg.w_fire * t.weight;
        loss.fire = w * l;
        d.fire = w * g;
    }
    if t.mask.aim {
        let (l, dc, ds) = aim_nll_and_grad(logits.aim_c, logits.aim_s, t.aim, cfg.aim_kappa);
        let w = cfg.w_aim * t.weight;
        loss.aim = w * l;
        d.aim_c = w * dc;
        d.aim_s = w * ds;
    }
    loss.total = loss.dir + loss.jump + loss.hook + loss.fire + loss.aim;
    (loss, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets() -> StepTargets {
        StepTargets {
            dir: 2,
            jump: true,
            hook: false,
            fire: true,
            aim: 0.7,
            soft: Some(SoftTargets {
                dir: [0.1, 0.2, 0.7],
                jump: 0.9,
                hook: 0.1,
                fire: 0.6,
            }),
            mask: HeadMask::ALL,
            weight: 1.7,
            hook_scale: 1.0,
        }
    }

    fn logits() -> HeadLogits {
        HeadLogits {
            dir: [0.3, -0.2, 0.5],
            jump: -0.4,
            hook: 0.8,
            fire: 0.1,
            aim_c: 0.6,
            aim_s: -0.9,
        }
    }

    fn cfg() -> LossConfig {
        LossConfig {
            w_dir: 1.0,
            w_jump: 0.7,
            w_hook: 1.3,
            w_fire: 0.9,
            w_aim: 0.5,
            soft_mix: 0.4,
            pos_weight: [2.0, 3.0, 1.5],
            aim_kappa: 4.0,
            soft_smoothing: 0.02,
        }
    }

    /// Central finite differences on the loss against the analytic gradient, in f64 inputs' f32.
    #[test]
    fn gradient_matches_finite_differences() {
        let (t, c) = (targets(), cfg());
        let base = logits();
        let (_, g) = head_loss_and_grad(&base, &t, &c);
        let eps = 1e-3f32;
        let probe = |edit: &dyn Fn(&mut HeadLogits, f32)| -> f32 {
            let (mut a, mut b) = (base, base);
            edit(&mut a, eps);
            edit(&mut b, -eps);
            (head_loss_and_grad(&a, &t, &c).0.total - head_loss_and_grad(&b, &t, &c).0.total) / (2.0 * eps)
        };
        let cases: [(&str, f32, f32); 8] = [
            ("dir0", g.dir[0], probe(&|l, e| l.dir[0] += e)),
            ("dir1", g.dir[1], probe(&|l, e| l.dir[1] += e)),
            ("dir2", g.dir[2], probe(&|l, e| l.dir[2] += e)),
            ("jump", g.jump, probe(&|l, e| l.jump += e)),
            ("hook", g.hook, probe(&|l, e| l.hook += e)),
            ("fire", g.fire, probe(&|l, e| l.fire += e)),
            ("aim_c", g.aim_c, probe(&|l, e| l.aim_c += e)),
            ("aim_s", g.aim_s, probe(&|l, e| l.aim_s += e)),
        ];
        for (name, analytic, numeric) in cases {
            assert!(
                (analytic - numeric).abs() < 2e-3 * (1.0 + numeric.abs()),
                "{name}: analytic {analytic} vs numeric {numeric}"
            );
        }
    }

    #[test]
    fn hook_scale_multiplies_only_the_hook_head() {
        let c = cfg();
        let base = targets();
        let mut scaled = base;
        scaled.hook_scale = 3.0;
        let (l1, g1) = head_loss_and_grad(&logits(), &base, &c);
        let (l3, g3) = head_loss_and_grad(&logits(), &scaled, &c);
        assert!((l3.hook - 3.0 * l1.hook).abs() < 1e-6 && (g3.hook - 3.0 * g1.hook).abs() < 1e-6);
        assert_eq!((l3.dir, l3.jump, l3.fire, l3.aim), (l1.dir, l1.jump, l1.fire, l1.aim));
        assert_eq!(
            (g3.dir, g3.jump, g3.fire, g3.aim_c),
            (g1.dir, g1.jump, g1.fire, g1.aim_c)
        );
        assert!((l3.total - (l1.total + 2.0 * l1.hook)).abs() < 1e-5);
    }

    #[test]
    fn the_masked_view_hides_only_the_own_hook_and_combines_only_the_hook_head() {
        use ddai_brain::{CharacterObservation, HOOK_GRABBED, HOOK_IDLE, Observation};
        let map = std::sync::Arc::new(ddai_physics::map::MapData {
            width: 2,
            height: 2,
            game: vec![Default::default(); 4],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        });
        let mut me = CharacterObservation::at_rest(0);
        me.hook_state = HOOK_GRABBED;
        let mut opp = CharacterObservation::at_rest(1);
        opp.hook_state = HOOK_GRABBED;
        let obs = Observation {
            map,
            tick: 7,
            self_state: me,
            others: vec![opp],
            target_id: Some(1),
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        let m = mask_own_hook(&obs);
        assert_eq!(m.self_state.hook_state, HOOK_IDLE);
        assert_eq!(
            m.others[0].hook_state, HOOK_GRABBED,
            "the opponent's hook stays visible"
        );
        assert_eq!((m.tick, m.target_id), (7, Some(1)));
        assert_eq!(obs.self_state.hook_state, HOOK_GRABBED, "the original is untouched");
        let (a, b) = (
            logits(),
            HeadLogits {
                hook: -3.0,
                jump: 9.0,
                ..logits()
            },
        );
        let c = combine_hook_view(&a, &b);
        assert_eq!((c.hook, c.jump, c.dir, c.fire), (-3.0, a.jump, a.dir, a.fire));
    }

    #[test]
    fn zero_weight_and_masked_heads_contribute_nothing() {
        let c = cfg();
        let mut t = targets();
        t.weight = 0.0;
        let (l, g) = head_loss_and_grad(&logits(), &t, &c);
        assert_eq!(l, StepLoss::default());
        assert_eq!(g, HeadLogits::default());

        let mut t = targets();
        t.mask = HeadMask {
            hook: true,
            ..HeadMask::NONE
        };
        let (l, g) = head_loss_and_grad(&logits(), &t, &c);
        assert!(l.hook > 0.0 && l.dir == 0.0 && l.jump == 0.0 && l.fire == 0.0 && l.aim == 0.0);
        assert_eq!(
            (g.dir, g.jump, g.fire, g.aim_c, g.aim_s),
            ([0.0; 3], 0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(l.total, l.hook);
    }

    #[test]
    fn hard_only_matches_plain_cross_entropy_and_weight_scales_linearly() {
        let mut t = targets();
        t.soft = None;
        t.mask = HeadMask {
            dir: true,
            ..HeadMask::NONE
        };
        let c = LossConfig {
            w_dir: 1.0,
            ..LossConfig::default()
        };
        t.weight = 1.0;
        let (l1, _) = head_loss_and_grad(&logits(), &t, &c);
        let p = logits().dir_probs();
        assert!((l1.dir - (-p[2].ln())).abs() < 1e-6);
        t.weight = 2.5;
        let (l2, _) = head_loss_and_grad(&logits(), &t, &c);
        assert!((l2.dir - 2.5 * l1.dir).abs() < 1e-5);
    }

    #[test]
    fn soft_targets_pull_towards_the_elite_frequencies() {
        // With soft_mix = 1 and a soft target of 0.9 for jump, the optimal logit is ln(9).
        let mut t = targets();
        t.mask = HeadMask {
            jump: true,
            ..HeadMask::NONE
        };
        t.weight = 1.0;
        let c = LossConfig {
            soft_mix: 1.0,
            soft_smoothing: 0.0,
            pos_weight: [1.0; 3],
            ..LossConfig::default()
        };
        let l = HeadLogits {
            jump: 9.0f32.ln(),
            ..HeadLogits::default()
        };
        let (_, g) = head_loss_and_grad(&l, &t, &c);
        assert!(g.jump.abs() < 1e-5, "gradient at the soft optimum: {}", g.jump);
    }
}
