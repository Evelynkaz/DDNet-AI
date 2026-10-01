//! Per-head metrics against held-out labels (task 8.2, acceptance criterion 3): what the
//! learning curves and the E-005 tables report.
//!
//! Every head is compared with its own **majority baseline** and reported with a class-balanced
//! accuracy and, for the binary heads, the AUROC - a raw accuracy on a skewed head is not
//! evidence of learning (the 7.3 demo's hook head looked fine on a stratified set and was at
//! chance on the natural one). The aim head is scored only where the aim matters (a hook or shot
//! is intended) by the angular error of `atan2(S, C)` against the label.

use ddai_brain::{HOOK_FLYING, HOOK_GRABBED};
use ddai_fly::bc::{HeadLogits, HeadThresholds};
use serde::Serialize;

use crate::seq::Window;

/// Metrics of one binary head.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BinaryMetrics {
    pub n: u64,
    pub positives: u64,
    /// Fraction of positive labels.
    pub prevalence: f64,
    /// Fraction of decisions the head presses at the threshold in use (compare with `prevalence`:
    /// a head that presses twice as often as the teacher is miscalibrated, whatever its AUROC).
    pub pred_rate: f64,
    /// The decision threshold the point metrics (`accuracy`, `precision`, ...) were computed at.
    pub threshold: f64,
    pub accuracy: f64,
    /// Accuracy of always predicting the majority class.
    pub majority_baseline: f64,
    /// Mean of the per-class recalls (`0.5` for any constant predictor).
    pub balanced_accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    /// Area under the ROC curve (`0.5` = chance); `NaN`-free: `0.5` when one class is missing.
    pub auroc: f64,
}

/// Metrics of the three-way direction head.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DirMetrics {
    pub n: u64,
    /// Label counts `[left, stop, right]`.
    pub counts: [u64; 3],
    pub accuracy: f64,
    pub majority_baseline: f64,
    pub balanced_accuracy: f64,
    /// `confusion[label][predicted]`.
    pub confusion: [[u64; 3]; 3],
    /// Fraction of decisions whose label is among the two most likely directions: what matters when
    /// the head proposes candidates to a search (D-041) rather than acting alone.
    pub top2_accuracy: f64,
}

/// Metrics of the aim head, over the decisions where the aim matters.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AimMetrics {
    pub n: u64,
    pub median_error_deg: f64,
    pub mean_error_deg: f64,
    pub within_15deg: f64,
    pub within_45deg: f64,
}

/// The hook head on the decisions where the own hook is in one state (`not_out`: no hook in the air
/// or attached; `out`: flying or grabbed). The hook head's AUROC pools both and rewards the shortcut
/// "hook <=> my hook is already out" (own hook state is an input); this split shows whether the
/// head can *start* a hook and *let one go* (review F2 of E-005).
#[derive(Debug, Clone, Default, Serialize)]
pub struct HookStateMetrics {
    pub n: u64,
    /// Fraction of decisions where the teacher presses the hook.
    pub label_hook_rate: f64,
    /// The same for the model, at the threshold in use.
    pub pred_hook_rate: f64,
    /// `P(pred hook | label hook)`.
    pub recall: f64,
    /// `P(pred no hook | label no hook)`.
    pub specificity: f64,
    pub accuracy: f64,
    pub auroc: f64,
}

/// [`HookStateMetrics`] for both own-hook states, with the two events named.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HookByState {
    pub not_out: HookStateMetrics,
    pub out: HookStateMetrics,
    /// Start of a hook (own hook not out): how often the teacher presses / the model presses, and
    /// how often the model presses where the teacher does (`= not_out.recall`).
    pub start_label_rate: f64,
    pub start_pred_rate: f64,
    pub start_accuracy: f64,
    /// Release of a hook (own hook out): how often the teacher / the model lets go, and how often
    /// the model lets go where the teacher does (`= out.specificity`).
    pub release_label_rate: f64,
    pub release_pred_rate: f64,
    pub release_accuracy: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HeadReport {
    /// Scored decisions.
    pub steps: u64,
    pub dir: DirMetrics,
    pub jump: BinaryMetrics,
    pub hook: BinaryMetrics,
    pub fire: BinaryMetrics,
    pub aim: AimMetrics,
    pub hook_by_state: HookByState,
    /// Decisions where direction, jump and hook are all right at once (aim and fire not counted).
    pub joint_dir_jump_hook: f64,
    /// The same, with the direction counted as right when it is among the two most likely.
    pub joint_top2_dir_jump_hook: f64,
}

/// Collects predictions and labels, then computes a [`HeadReport`].
#[derive(Default)]
pub struct MetricsAccumulator {
    steps: u64,
    dir: Vec<(u8, u8, u8)>,
    joint: Vec<(bool, bool)>,
    jump: Vec<(f32, bool)>,
    hook: Vec<(f32, bool)>,
    fire: Vec<(f32, bool)>,
    aim_err: Vec<f32>,
    /// `(hook probability, label, own hook out)` of every scored hook decision.
    hook_state: Vec<(f32, bool, bool)>,
    thresholds: HeadThresholds,
}

fn wrap_pi(a: f32) -> f32 {
    a.sin().atan2(a.cos())
}

impl MetricsAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point metrics (accuracy, precision, recall, rates) at these decision thresholds instead of
    /// `0.5` (the model's own thresholds, so the report is about the policy that plays).
    pub fn with_thresholds(thresholds: HeadThresholds) -> Self {
        MetricsAccumulator {
            thresholds,
            ..Self::default()
        }
    }

    /// The thresholds that make each binary head press as often as its labels do on everything
    /// accumulated (rate matching; see [`rate_matched_threshold`]). **The hook head is matched on the
    /// decisions where the own hook is not out** (the decision to *start* a hook): pooled over both
    /// states, the rate is dominated by "hold while out", the model holds almost always, and matching
    /// the pooled rate pushes the threshold up and the start rate far *below* the teacher's (measured on
    /// the E-005 bundles: fly start rate 0.46 -> 0.20 against the teacher's 0.35; MLP-w 0.19 -> 0.08).
    pub fn rate_matched_thresholds(&self) -> HeadThresholds {
        let starts: Vec<(f32, bool)> = self.hook_state.iter().filter(|r| !r.2).map(|r| (r.0, r.1)).collect();
        HeadThresholds {
            jump: rate_matched_threshold(&self.jump),
            hook: rate_matched_threshold(&starts),
            fire: rate_matched_threshold(&self.fire),
        }
    }

    /// Adds every scored decision of `window` (its `targets` with a positive weight and the head
    /// masks) with the predictions `logits` (one per decision).
    pub fn add_window(&mut self, window: &Window, logits: &[HeadLogits]) {
        assert_eq!(window.targets.len(), logits.len());
        let th = self.thresholds;
        for (k, (t, l)) in window.targets.iter().zip(logits).enumerate() {
            if t.weight <= 0.0 {
                continue;
            }
            self.steps += 1;
            if t.mask.dir {
                let p = l.dir_probs();
                let mut order = [0usize, 1, 2];
                order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
                self.dir.push((t.dir, order[0] as u8, order[1] as u8));
                if t.mask.jump && t.mask.hook {
                    let rest = l.jump_on(&th) == t.jump && l.hook_on(&th) == t.hook;
                    self.joint.push((
                        rest && order[0] as u8 == t.dir,
                        rest && (order[0] as u8 == t.dir || order[1] as u8 == t.dir),
                    ));
                }
            }
            if t.mask.jump {
                self.jump.push((l.jump_prob(), t.jump));
            }
            if t.mask.hook {
                self.hook.push((l.hook_prob(), t.hook));
                let own = window.observations[k].self_state.hook_state;
                self.hook_state
                    .push((l.hook_prob(), t.hook, matches!(own, HOOK_FLYING | HOOK_GRABBED)));
            }
            if t.mask.fire {
                self.fire.push((l.fire_prob(), t.fire));
            }
            if t.mask.aim {
                self.aim_err.push(wrap_pi(l.aim_angle() - t.aim).abs());
            }
        }
    }

    /// Adds everything `other` collected.
    pub fn merge(&mut self, other: MetricsAccumulator) {
        self.steps += other.steps;
        self.dir.extend(other.dir);
        self.joint.extend(other.joint);
        self.jump.extend(other.jump);
        self.hook.extend(other.hook);
        self.fire.extend(other.fire);
        self.aim_err.extend(other.aim_err);
        self.hook_state.extend(other.hook_state);
    }

    pub fn finish(self) -> HeadReport {
        let th = self.thresholds;
        HeadReport {
            steps: self.steps,
            joint_dir_jump_hook: frac(self.joint.iter().filter(|j| j.0).count(), self.joint.len()),
            joint_top2_dir_jump_hook: frac(self.joint.iter().filter(|j| j.1).count(), self.joint.len()),
            dir: dir_metrics(&self.dir),
            jump: binary_metrics(&self.jump, th.jump),
            hook: binary_metrics(&self.hook, th.hook),
            fire: binary_metrics(&self.fire, th.fire),
            aim: aim_metrics(&self.aim_err),
            hook_by_state: hook_by_state(&self.hook_state, th.hook),
        }
    }
}

fn frac(k: usize, n: usize) -> f64 {
    if n == 0 { 0.0 } else { k as f64 / n as f64 }
}

fn dir_metrics(pairs: &[(u8, u8, u8)]) -> DirMetrics {
    let mut m = DirMetrics {
        n: pairs.len() as u64,
        ..DirMetrics::default()
    };
    let mut top2 = 0u64;
    for &(label, pred, second) in pairs {
        let (l, p) = (label.min(2) as usize, pred.min(2) as usize);
        m.counts[l] += 1;
        m.confusion[l][p] += 1;
        top2 += u64::from(label == pred || label == second);
    }
    m.top2_accuracy = frac(top2 as usize, pairs.len());
    if m.n > 0 {
        let correct: u64 = (0..3).map(|c| m.confusion[c][c]).sum();
        m.accuracy = correct as f64 / m.n as f64;
        m.majority_baseline = *m.counts.iter().max().unwrap_or(&0) as f64 / m.n as f64;
        let recalls: Vec<f64> = (0..3)
            .filter(|&c| m.counts[c] > 0)
            .map(|c| m.confusion[c][c] as f64 / m.counts[c] as f64)
            .collect();
        m.balanced_accuracy = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
    }
    m
}

/// AUROC by the rank-sum (Mann-Whitney) formula with ties given their average rank.
pub fn auroc(scores: &[(f32, bool)]) -> f64 {
    let pos = scores.iter().filter(|s| s.1).count() as f64;
    let neg = scores.len() as f64 - pos;
    if pos == 0.0 || neg == 0.0 {
        return 0.5;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].0.total_cmp(&scores[b].0));
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]].0 == scores[idx[i]].0 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            if scores[k].1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    (rank_sum_pos - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

/// The threshold at which a head presses as often as its labels do (the teacher's rate), by
/// ranking the scores: with `k` positive labels among `n`, the threshold is halfway between the
/// `k`-th and `(k+1)`-th highest probability. `0.5` when a class is missing (nothing to match); kept
/// inside `[0.01, 0.99]` so one tied or saturated score cannot switch a head off or on for good.
///
/// Why rate matching and not best-F1: the heads are trained with positive-class weights (3/1/6) so
/// that rare keys get gradient at all; that shifts the probabilities up and, at `0.5`, the models
/// jump 33-48% of the time against the teacher's 21% and fire 10-19% against 4-5% (E-005, review F5).
/// Matching the teacher's rate undoes the shift directly and does not trade precision for recall,
/// which best-F1 would (it favours over-pressing a rare key). The choice is a property of the
/// validation labels, so it is stable and needs no search.
pub fn rate_matched_threshold(scores: &[(f32, bool)]) -> f32 {
    let positives = scores.iter().filter(|s| s.1).count();
    if positives == 0 || positives == scores.len() {
        return 0.5;
    }
    let mut p: Vec<f32> = scores.iter().map(|s| s.0).collect();
    p.sort_by(|a, b| b.total_cmp(a));
    let t = 0.5 * (p[positives - 1] + p[positives]);
    t.clamp(0.01, 0.99)
}

fn hook_by_state(rows: &[(f32, bool, bool)], threshold: f32) -> HookByState {
    let part = |out: bool| {
        let scores: Vec<(f32, bool)> = rows.iter().filter(|r| r.2 == out).map(|r| (r.0, r.1)).collect();
        let m = binary_metrics(&scores, threshold);
        let neg = scores.len() as f64 - m.positives as f64;
        let tn = scores.iter().filter(|s| !s.1 && s.0 < threshold).count() as f64;
        HookStateMetrics {
            n: m.n,
            label_hook_rate: m.prevalence,
            pred_hook_rate: m.pred_rate,
            recall: m.recall,
            specificity: if neg > 0.0 { tn / neg } else { 0.0 },
            accuracy: m.accuracy,
            auroc: m.auroc,
        }
    };
    let (not_out, out) = (part(false), part(true));
    HookByState {
        start_label_rate: not_out.label_hook_rate,
        start_pred_rate: not_out.pred_hook_rate,
        start_accuracy: not_out.recall,
        release_label_rate: if out.n > 0 { 1.0 - out.label_hook_rate } else { 0.0 },
        release_pred_rate: if out.n > 0 { 1.0 - out.pred_hook_rate } else { 0.0 },
        release_accuracy: out.specificity,
        not_out,
        out,
    }
}

fn binary_metrics(scores: &[(f32, bool)], threshold: f32) -> BinaryMetrics {
    let n = scores.len() as u64;
    let mut m = BinaryMetrics {
        n,
        auroc: 0.5,
        threshold: f64::from(threshold),
        ..BinaryMetrics::default()
    };
    if n == 0 {
        return m;
    }
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for &(p, y) in scores {
        match (p >= threshold, y) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    m.positives = tp + fn_;
    m.prevalence = m.positives as f64 / n as f64;
    m.pred_rate = (tp + fp) as f64 / n as f64;
    m.accuracy = (tp + tn) as f64 / n as f64;
    m.majority_baseline = m.prevalence.max(1.0 - m.prevalence);
    let recall_pos = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        f64::NAN
    };
    let recall_neg = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        f64::NAN
    };
    m.balanced_accuracy = match (recall_pos.is_nan(), recall_neg.is_nan()) {
        (false, false) => 0.5 * (recall_pos + recall_neg),
        (false, true) => recall_pos,
        (true, false) => recall_neg,
        (true, true) => 0.5,
    };
    m.precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    m.recall = if recall_pos.is_nan() { 0.0 } else { recall_pos };
    m.auroc = auroc(scores);
    m
}

fn aim_metrics(errors: &[f32]) -> AimMetrics {
    if errors.is_empty() {
        return AimMetrics::default();
    }
    let mut e: Vec<f32> = errors.to_vec();
    e.sort_by(f32::total_cmp);
    let n = e.len();
    let deg = |x: f32| f64::from(x).to_degrees();
    AimMetrics {
        n: n as u64,
        median_error_deg: deg(e[n / 2]),
        mean_error_deg: deg(e.iter().sum::<f32>() / n as f32),
        within_15deg: e.iter().filter(|&&x| deg(x) <= 15.0).count() as f64 / n as f64,
        within_45deg: e.iter().filter(|&&x| deg(x) <= 45.0).count() as f64 / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_brain::{CharacterObservation, Observation};
    use ddai_fly::bc::{HeadMask, StepTargets};
    use std::sync::Arc;

    #[test]
    fn auroc_matches_hand_worked_cases() {
        // Perfectly separated.
        assert_eq!(auroc(&[(0.1, false), (0.2, false), (0.8, true), (0.9, true)]), 1.0);
        // Perfectly inverted.
        assert_eq!(auroc(&[(0.9, false), (0.8, false), (0.2, true), (0.1, true)]), 0.0);
        // All tied: chance.
        assert_eq!(auroc(&[(0.5, false), (0.5, true), (0.5, false), (0.5, true)]), 0.5);
        // One positive below one of two negatives: 3 of 4 pairs ordered correctly... here pairs
        // (pos 0.5 vs neg 0.2 ok, vs neg 0.7 wrong) = 0.5.
        assert_eq!(auroc(&[(0.2, false), (0.7, false), (0.5, true)]), 0.5);
        // A single class is chance by convention.
        assert_eq!(auroc(&[(0.3, true), (0.6, true)]), 0.5);
    }

    #[test]
    fn a_constant_predictor_scores_the_majority_baseline_and_balanced_accuracy_one_half() {
        let scores: Vec<(f32, bool)> = (0..100).map(|i| (0.1, i % 10 == 0)).collect();
        let m = binary_metrics(&scores, 0.5);
        assert!((m.prevalence - 0.1).abs() < 1e-9);
        assert!((m.accuracy - 0.9).abs() < 1e-9 && (m.majority_baseline - 0.9).abs() < 1e-9);
        assert!((m.balanced_accuracy - 0.5).abs() < 1e-9);
        assert_eq!((m.precision, m.recall), (0.0, 0.0));
        assert_eq!(m.auroc, 0.5);
    }

    #[test]
    fn direction_metrics_count_confusions() {
        let pairs = [
            (0, 0, 1),
            (0, 1, 0),
            (2, 2, 1),
            (2, 2, 1),
            (1, 1, 0),
            (1, 1, 0),
            (1, 1, 2),
            (1, 0, 1),
        ];
        let m = dir_metrics(&pairs);
        assert_eq!(m.n, 8);
        assert_eq!(m.counts, [2, 4, 2]);
        assert!(
            (m.top2_accuracy - 1.0).abs() < 1e-9,
            "every label is in the top two here"
        );
        assert_eq!(m.confusion[0], [1, 1, 0]);
        assert!((m.accuracy - 6.0 / 8.0).abs() < 1e-9);
        assert!((m.majority_baseline - 0.5).abs() < 1e-9);
        assert!((m.balanced_accuracy - (0.5 + 0.75 + 1.0) / 3.0).abs() < 1e-9);
    }

    #[test]
    fn aim_error_wraps_around_pi() {
        // A prediction of +3.0 rad for a label of -3.0 rad is 0.283 rad (16 deg) off, not 6.
        let m = aim_metrics(&[wrap_pi(3.0 - -3.0).abs()]);
        assert!((m.median_error_deg - (2.0 * std::f64::consts::PI - 6.0).to_degrees()).abs() < 1e-3);
        assert_eq!(m.within_45deg, 1.0);
        assert_eq!(m.within_15deg, 0.0);
    }

    #[test]
    fn the_accumulator_scores_only_weighted_masked_steps() {
        let map = Arc::new(ddai_physics::map::MapData {
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
        let obs = Observation {
            map,
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: vec![],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        let t = |weight: f32, mask: HeadMask| StepTargets {
            dir: 2,
            jump: true,
            hook: false,
            fire: false,
            aim: 0.0,
            soft: None,
            mask,
            weight,
        };
        let w = Window {
            observations: vec![obs.clone(), obs.clone(), obs],
            targets: vec![
                t(0.0, HeadMask::ALL),
                t(1.0, HeadMask::ALL),
                t(
                    1.0,
                    HeadMask {
                        aim: false,
                        hook: false,
                        ..HeadMask::ALL
                    },
                ),
            ],
            start: 0,
            mirrored: false,
        };
        let logits = HeadLogits {
            dir: [0.0, 0.0, 5.0],
            jump: 3.0,
            hook: -3.0,
            fire: -3.0,
            aim_c: 1.0,
            aim_s: 0.0,
        };
        let mut acc = MetricsAccumulator::new();
        acc.add_window(&w, &[logits; 3]);
        let r = acc.finish();
        assert_eq!(r.steps, 2);
        assert_eq!((r.dir.n, r.jump.n, r.hook.n, r.aim.n), (2, 2, 1, 1));
        assert_eq!(r.dir.accuracy, 1.0);
        assert_eq!(r.jump.accuracy, 1.0);
        assert!(r.aim.median_error_deg < 1e-3);
        // Only the first scored step has both jump and hook scored; all three are right there.
        assert_eq!((r.joint_dir_jump_hook, r.joint_top2_dir_jump_hook), (1.0, 1.0));
        assert_eq!(r.dir.top2_accuracy, 1.0);
    }

    #[test]
    fn the_rate_matched_threshold_makes_the_head_press_as_often_as_the_labels() {
        // 20% positives; the head is shifted up (negatives sit at 0.55-0.70, positives at 0.8-0.95),
        // so at 0.5 it presses on every decision.
        let mut scores = Vec::new();
        for i in 0..80 {
            scores.push((0.55 + 0.15 * (i as f32 / 80.0), false));
        }
        for i in 0..20 {
            scores.push((0.80 + 0.15 * (i as f32 / 20.0), true));
        }
        let at_half = binary_metrics(&scores, 0.5);
        assert!((at_half.pred_rate - 1.0).abs() < 1e-9, "presses everywhere at 0.5");
        let t = rate_matched_threshold(&scores);
        assert!(t > 0.70 && t < 0.80, "{t}");
        let m = binary_metrics(&scores, t);
        assert!(
            (m.pred_rate - m.prevalence).abs() < 1e-9,
            "{} vs {}",
            m.pred_rate,
            m.prevalence
        );
        assert!((m.threshold - f64::from(t)).abs() < 1e-9);
        assert_eq!((m.precision, m.recall), (1.0, 1.0));
        // Nothing to match without both classes.
        assert_eq!(rate_matched_threshold(&[(0.9, true), (0.8, true)]), 0.5);
        assert_eq!(rate_matched_threshold(&[(0.9, false)]), 0.5);
        assert_eq!(rate_matched_threshold(&[]), 0.5);
        // Clamped away from 0 and 1.
        let sat = [(1.0, true), (0.0, false), (0.0, false)];
        assert!((0.01..=0.99).contains(&rate_matched_threshold(&sat)));
    }

    #[test]
    fn the_hook_threshold_is_matched_on_start_decisions_not_on_the_pooled_rate() {
        // 10 start decisions (own hook not out): the teacher presses in 3, the head scores them 0.9 / 0.4
        // (so at 0.5 it presses in 3 only if the 0.9s are the right ones). 90 hold decisions (own hook
        // out): the teacher keeps pressing, the head says 0.95 - they dominate the pooled rate.
        let map = Arc::new(ddai_physics::map::MapData {
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
        let obs = |hook_state: i32| {
            let mut me = CharacterObservation::at_rest(0);
            me.hook_state = hook_state;
            Observation {
                map: map.clone(),
                tick: 0,
                self_state: me,
                others: vec![],
                target_id: None,
                tuning: ddai_physics::tuning::TuningParams::default(),
            }
        };
        let t = |hook: bool| StepTargets {
            dir: 1,
            jump: false,
            hook,
            fire: false,
            aim: 0.0,
            soft: None,
            mask: HeadMask::ALL,
            weight: 1.0,
        };
        let logit = |p: f32| HeadLogits {
            hook: (p / (1.0 - p)).ln(),
            jump: -5.0,
            fire: -5.0,
            ..HeadLogits::default()
        };
        let mut observations = Vec::new();
        let mut targets = Vec::new();
        let mut logits = Vec::new();
        for i in 0..10 {
            observations.push(obs(0));
            targets.push(t(i < 3));
            logits.push(logit(if i < 3 { 0.9 } else { 0.7 }));
        }
        for _ in 0..90 {
            observations.push(obs(HOOK_GRABBED));
            targets.push(t(true));
            logits.push(logit(0.95));
        }
        let w = Window {
            observations,
            targets,
            start: 0,
            mirrored: false,
        };
        let mut acc = MetricsAccumulator::new();
        acc.add_window(&w, &logits);
        let th = acc.rate_matched_thresholds();
        // Starts: 3 of 10 labelled, so the threshold sits between the 0.9s and the 0.7s. The pooled
        // rate (93 of 100 labelled) would have put it between 0.95 and 0.9 and silenced every start.
        assert!(th.hook > 0.7 && th.hook < 0.9, "{}", th.hook);
        let mut acc = MetricsAccumulator::with_thresholds(th);
        acc.add_window(&w, &logits);
        let r = acc.finish();
        assert!((r.hook_by_state.start_pred_rate - r.hook_by_state.start_label_rate).abs() < 1e-9);
        assert_eq!(r.hook_by_state.start_accuracy, 1.0);
    }

    #[test]
    fn hook_start_and_release_are_scored_separately_by_own_hook_state() {
        // Not out: the teacher starts a hook in 2 of 10; the model never does.
        // Out: the teacher lets go in 2 of 10; the model never does (always presses).
        let mut rows = Vec::new();
        for i in 0..10 {
            rows.push((0.1f32, i < 2, false));
        }
        for i in 0..10 {
            rows.push((0.9f32, i >= 2, true));
        }
        let h = hook_by_state(&rows, 0.5);
        assert_eq!((h.not_out.n, h.out.n), (10, 10));
        assert!((h.start_label_rate - 0.2).abs() < 1e-9 && h.start_pred_rate == 0.0);
        assert_eq!(h.start_accuracy, 0.0, "it never starts a hook");
        assert!((h.release_label_rate - 0.2).abs() < 1e-9 && h.release_pred_rate == 0.0);
        assert_eq!(h.release_accuracy, 0.0, "it never lets go");
        // Pooled, the same head looks excellent: the shortcut "hook <=> own hook out".
        let pooled: Vec<(f32, bool)> = rows.iter().map(|r| (r.0, r.1)).collect();
        assert!(binary_metrics(&pooled, 0.5).accuracy > 0.6);
        // A head that follows the teacher scores 1 on both.
        let good: Vec<(f32, bool, bool)> = rows
            .iter()
            .map(|&(_, y, out)| (if y { 0.9 } else { 0.1 }, y, out))
            .collect();
        let g = hook_by_state(&good, 0.5);
        assert_eq!((g.start_accuracy, g.release_accuracy), (1.0, 1.0));
        // A missing state reports zeros, not NaN.
        let none = hook_by_state(&[(0.9, true, true)], 0.5);
        assert_eq!((none.not_out.n, none.start_pred_rate), (0, 0.0));
    }

    #[test]
    fn the_accumulator_reads_the_own_hook_state_from_the_observation_and_uses_its_thresholds() {
        let map = Arc::new(ddai_physics::map::MapData {
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
        let obs = |hook_state: i32| {
            let mut me = CharacterObservation::at_rest(0);
            me.hook_state = hook_state;
            Observation {
                map: map.clone(),
                tick: 0,
                self_state: me,
                others: vec![],
                target_id: None,
                tuning: ddai_physics::tuning::TuningParams::default(),
            }
        };
        let t = |hook: bool| StepTargets {
            dir: 1,
            jump: false,
            hook,
            fire: false,
            aim: 0.0,
            soft: None,
            mask: HeadMask::ALL,
            weight: 1.0,
        };
        let w = Window {
            observations: vec![obs(0), obs(HOOK_FLYING), obs(HOOK_GRABBED), obs(0)],
            targets: vec![t(true), t(true), t(false), t(false)],
            start: 0,
            mirrored: false,
        };
        // Hook probability 0.6 everywhere (logit ln(1.5)).
        let l = HeadLogits {
            hook: 1.5f32.ln(),
            jump: -5.0,
            fire: -5.0,
            ..HeadLogits::default()
        };
        let mut acc = MetricsAccumulator::new();
        acc.add_window(&w, &[l; 4]);
        let r = acc.finish();
        assert_eq!((r.hook_by_state.not_out.n, r.hook_by_state.out.n), (2, 2));
        assert_eq!(r.hook.pred_rate, 1.0, "0.6 >= 0.5 presses");
        assert_eq!(r.hook_by_state.release_pred_rate, 0.0);
        // At a calibrated threshold of 0.7 the same head presses nowhere.
        let mut acc = MetricsAccumulator::with_thresholds(HeadThresholds {
            hook: 0.7,
            ..HeadThresholds::default()
        });
        acc.add_window(&w, &[l; 4]);
        let r = acc.finish();
        assert_eq!(r.hook.pred_rate, 0.0);
        assert_eq!(r.hook.threshold, 0.7f32 as f64);
        assert_eq!(r.hook_by_state.release_pred_rate, 1.0);
        assert_eq!(r.hook_by_state.start_accuracy, 0.0);
    }
}
