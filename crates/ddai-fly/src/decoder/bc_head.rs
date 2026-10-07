//! The tied decoder in logit space for behaviour cloning (task 8.2): [`decoder_logits`] runs every
//! head forward to a [`HeadLogits`], [`decoder_bc_loss_and_grad`] scores it with the shared
//! [`crate::bc`] loss and backpropagates `dL/dlogits` through the mirror-tied structure (pooled
//! per-type weights, `L`/`R`-tied direction, tied aim pairs) into the decoder parameters and into
//! `dL/d(dn_rates)`, the exact input shape [`crate::backward::backward`] takes.
//!
//! The forward formulas are those of [`super::decoder_forward_into`] and the backward step is
//! that of [`super::decoder_loss_and_grad`]; only the loss in between is different (soft targets,
//! head weights and masks, class weights). A test pins forward equality with the original.

use super::{
    DecoderGradients, DecoderModel, DecoderParams, DnCalibration, hook_logit, mean_of, pooled_backward, pooled_logit,
    population_vector,
};
use crate::bc::{HeadLogits, LossConfig, StepLoss, StepTargets, head_loss_and_grad};

/// The five heads' logits for calibrated DN z-scores `z`. `hook_latch` is the fly's own previous hook command: it selects the hazard of an
/// intent hook head (`HeadLogits::hook` is then the logit of the hook key given the latch, see [`super::hook_logit`]) and is ignored by a
/// legacy one.
pub fn decoder_logits(decoder: &DecoderModel, z: &[f32], params: &DecoderParams, hook_latch: bool) -> HeadLogits {
    let side_logit = |left: bool| {
        decoder
            .direction_lr
            .iter()
            .zip(&params.direction_lr_w)
            .map(|(g, &w)| w * mean_of(z, if left { &g.left } else { &g.right }))
            .sum::<f32>()
            + params.direction_lr_b
    };
    let stop = pooled_logit(
        &decoder.direction_stop,
        &params.direction_stop_w,
        params.direction_stop_b,
        z,
    );
    let (c, s) = population_vector(
        z,
        &decoder.aim_pairs,
        &params.aim_pair_theta,
        &decoder.aim_unpaired,
        &params.aim_unpaired_theta,
    );
    HeadLogits {
        dir: [side_logit(true), stop, side_logit(false)],
        jump: pooled_logit(&decoder.jump, &params.jump_w, params.jump_b, z),
        hook: hook_logit(decoder, params, z, hook_latch),
        fire: pooled_logit(&decoder.fire, &params.fire_w, params.fire_b, z),
        aim_c: c,
        aim_s: s,
    }
}

/// Loss, decoder gradients and `dL/d(dn_rates)` of one decision. `dn_rates` are the raw DN rates
/// (`DecisionOutput::dn_rates` order); the frozen calibration and its clip are applied here, and
/// the clip's zero subgradient is respected in `grad_dn`.
pub fn decoder_bc_loss_and_grad(
    decoder: &DecoderModel,
    dn_rates: &[f32],
    calib: &DnCalibration,
    params: &DecoderParams,
    targets: &StepTargets,
    cfg: &LossConfig,
) -> (StepLoss, DecoderGradients, Vec<f32>, HeadLogits) {
    assert_eq!(dn_rates.len(), decoder.num_outputs);
    let clip_at = decoder.config.z_clip;
    let z = calib.z(dn_rates, clip_at);
    let logits = decoder_logits(decoder, &z, params, targets.hook_latch);
    let (loss, d) = head_loss_and_grad(&logits, targets, cfg);
    if targets.weight == 0.0 {
        return (
            loss,
            decoder.zeros_gradients(),
            vec![0.0f32; decoder.num_outputs],
            logits,
        );
    }
    let (grads, grad_dn) =
        decoder_logits_backward(decoder, dn_rates, calib, params, &d, &targets.mask, targets.hook_latch);
    (loss, grads, grad_dn, logits)
}

/// The backward half of [`decoder_bc_loss_and_grad`], for any loss on the head logits (task 8.5b: the PPO objective): given
/// `d = dL/d(head logits)` of one decision, the decoder gradients and `dL/d(dn_rates)`. Heads whose `mask` flag is off are skipped
/// (their entries of `d` must then be zero, as `head_loss_and_grad` leaves them). `hook_latch` must be the one the logits were made with:
/// of an intent hook head only the hazard it selected gets a gradient (the release hazard's through `hook = -z_release`).
pub fn decoder_logits_backward(
    decoder: &DecoderModel,
    dn_rates: &[f32],
    calib: &DnCalibration,
    params: &DecoderParams,
    d: &HeadLogits,
    mask: &crate::bc::HeadMask,
    hook_latch: bool,
) -> (DecoderGradients, Vec<f32>) {
    assert_eq!(dn_rates.len(), decoder.num_outputs);
    let clip_at = decoder.config.z_clip;
    let z = calib.z(dn_rates, clip_at);
    let mut grads = decoder.zeros_gradients();
    let mut grad_dn = vec![0.0f32; decoder.num_outputs];
    let mut grad_z = vec![0.0f32; z.len()];

    // direction: [left, stop, right] logits; left/right share weights and bias.
    if mask.dir {
        let (d_left, d_stop, d_right) = (d.dir[0], d.dir[1], d.dir[2]);
        grads.direction_lr_b += d_left + d_right;
        for (g, (&w, gw)) in decoder
            .direction_lr
            .iter()
            .zip(params.direction_lr_w.iter().zip(grads.direction_lr_w.iter_mut()))
        {
            *gw += d_left * mean_of(&z, &g.left) + d_right * mean_of(&z, &g.right);
            if !g.left.is_empty() {
                let contrib = d_left * w / g.left.len() as f32;
                for &slot in &g.left {
                    grad_z[slot] += contrib;
                }
            }
            if !g.right.is_empty() {
                let contrib = d_right * w / g.right.len() as f32;
                for &slot in &g.right {
                    grad_z[slot] += contrib;
                }
            }
        }
        pooled_backward(
            &decoder.direction_stop,
            &params.direction_stop_w,
            d_stop,
            &z,
            &mut grads.direction_stop_w,
            &mut grads.direction_stop_b,
            &mut grad_z,
        );
    }
    if mask.jump {
        pooled_backward(
            &decoder.jump,
            &params.jump_w,
            d.jump,
            &z,
            &mut grads.jump_w,
            &mut grads.jump_b,
            &mut grad_z,
        );
    }
    if mask.hook {
        match &params.hook_release {
            Some(r) if hook_latch => pooled_backward(
                &decoder.hook,
                &r.w,
                -d.hook,
                &z,
                &mut grads.hook_release_w,
                &mut grads.hook_release_b,
                &mut grad_z,
            ),
            _ => pooled_backward(
                &decoder.hook,
                &params.hook_w,
                d.hook,
                &z,
                &mut grads.hook_w,
                &mut grads.hook_b,
                &mut grad_z,
            ),
        }
    }
    if mask.fire {
        pooled_backward(
            &decoder.fire,
            &params.fire_w,
            d.fire,
            &z,
            &mut grads.fire_w,
            &mut grads.fire_b,
            &mut grad_z,
        );
    }
    if mask.aim {
        let (d_c, d_s) = (d.aim_c, d.aim_s);
        for (pair, (&theta, gtheta)) in decoder
            .aim_pairs
            .iter()
            .zip(params.aim_pair_theta.iter().zip(grads.aim_pair_theta.iter_mut()))
        {
            let (zl, zr) = (z[pair.l_slot], z[pair.r_slot]);
            *gtheta += d_c * (-(zl - zr) * theta.sin()) + d_s * ((zl + zr) * theta.cos());
            grad_z[pair.l_slot] += d_c * theta.cos() + d_s * theta.sin();
            grad_z[pair.r_slot] += d_c * (-theta.cos()) + d_s * theta.sin();
        }
        for (&slot, (&theta, gtheta)) in decoder.aim_unpaired.iter().zip(
            params
                .aim_unpaired_theta
                .iter()
                .zip(grads.aim_unpaired_theta.iter_mut()),
        ) {
            let zj = z[slot];
            *gtheta += d_c * (-zj * theta.sin()) + d_s * (zj * theta.cos());
            grad_z[slot] += d_c * theta.cos() + d_s * theta.sin();
        }
    }

    for i in 0..decoder.num_outputs {
        if grad_z[i] == 0.0 {
            continue;
        }
        let raw = (dn_rates[i] - calib.mu[i]) / calib.sigma[i];
        if raw.abs() >= clip_at {
            continue; // clipped: subgradient 0 (same kink convention as the original).
        }
        grad_dn[i] = grad_z[i] / calib.sigma[i];
    }
    (grads, grad_dn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bc::{HeadMask, SoftTargets};
    use crate::decoder::{DecoderConfig, decoder_forward, decoder_loss_and_grad};
    use crate::decoder::{DecoderTargets, add_l1_penalty};
    use crate::model::FlyModel;

    /// A graph shaped like the real output layer: a tied `L`/`R` direction type, pooled stop/jump/
    /// hook/fire types with two members per side, and two tied aim pairs plus one unpaired member.
    fn fixture_flyg() -> ddai_flyg::Flyg {
        use crate::brain_fixtures::{FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
        use ddai_flyg::{NeuronRole, Side, Sign};
        let names = [
            "DN_LR", "DN_STOP", "DN_JUMP", "DN_HOOK", "DN_FIRE", "DN_AIM_A", "DN_AIM_B", "DN_AIM_U",
        ];
        let types: Vec<FxType> = names
            .iter()
            .map(|&name| FxType {
                name,
                sign: Sign::Excitatory,
            })
            .collect();
        let member = |type_index: u32, side: Side| FxNeuron {
            type_index,
            role: NeuronRole::Output,
            side,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        };
        let mut neurons = vec![member(0, Side::L), member(0, Side::R)];
        for ti in [1u32, 2, 3, 4] {
            neurons.push(member(ti, Side::L));
            neurons.push(member(ti, Side::R));
            neurons.push(member(ti, Side::L));
            neurons.push(member(ti, Side::R));
        }
        for ti in [5u32, 6] {
            neurons.push(member(ti, Side::L));
            neurons.push(member(ti, Side::R));
        }
        neurons.push(member(7, Side::M));
        let group = |action: &'static str, types: Vec<&'static str>, side: Option<Side>| FxOutputGroup {
            action,
            member_type_names: types,
            side_filter: side,
        };
        let groups = vec![
            group("direction_left", vec!["DN_LR"], Some(Side::L)),
            group("direction_right", vec!["DN_LR"], Some(Side::R)),
            group("direction_stop", vec!["DN_STOP"], None),
            group("jump", vec!["DN_JUMP"], None),
            group("hook", vec!["DN_HOOK"], None),
            group("fire", vec!["DN_FIRE"], None),
            group("aim", vec!["DN_AIM_A", "DN_AIM_B", "DN_AIM_U"], None),
        ];
        build_brain_flyg(&types, &neurons, &[], &[], &groups)
    }

    fn setup() -> (FlyModel, DecoderModel, DecoderParams, DnCalibration) {
        let flyg = fixture_flyg();
        let config = crate::config::FlyConfig::default();
        let fly_params = crate::params::FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, fly_params).unwrap();
        let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
        let mut params = decoder.init_default_params();
        // Non-trivial weights so every head has signal.
        let mut k = 0.13f32;
        for v in params
            .direction_lr_w
            .iter_mut()
            .chain(&mut params.direction_stop_w)
            .chain(&mut params.jump_w)
            .chain(&mut params.hook_w)
            .chain(&mut params.fire_w)
        {
            k = (k * 1.7 + 0.31).fract();
            *v = k - 0.5;
        }
        let n = model.num_outputs();
        let calib = DnCalibration {
            mu: (0..n).map(|i| 0.1 * i as f32).collect(),
            sigma: vec![1.5; n],
        };
        (model, decoder, params, calib)
    }

    fn rates(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed).sin() + 1.0) * 1.2).collect()
    }

    #[test]
    fn logits_agree_with_the_original_forward() {
        let (model, decoder, params, calib) = setup();
        let r = rates(model.num_outputs(), 0.4);
        let z = calib.z(&r, decoder.config().z_clip);
        let l = decoder_logits(&decoder, &z, &params, false);
        let orig = decoder_forward(&decoder, &r, &calib, &params, false);
        for (a, b) in l.dir_probs().iter().zip(&orig.direction_probs) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((l.jump_prob() - orig.jump_prob).abs() < 1e-6);
        assert!((l.hook_prob() - orig.hook_prob).abs() < 1e-6);
        assert!((l.fire_prob() - orig.fire_prob).abs() < 1e-6);
        assert!((l.aim_angle() - orig.aim_angle).abs() < 1e-5);
    }

    /// With hard labels, unit weights and no soft part the BC loss is the original decoder loss
    /// (direction CE + three BCE + von Mises), with the same gradients.
    #[test]
    fn hard_only_bc_loss_equals_the_original_decoder_loss_and_gradients() {
        let (model, decoder, params, calib) = setup();
        let r = rates(model.num_outputs(), 1.1);
        let orig_t = DecoderTargets {
            direction: Some(2),
            jump: Some(true),
            hook: Some(false),
            fire: Some(true),
            aim: Some(0.9),
        };
        let (l0, g0, dn0) = decoder_loss_and_grad(&decoder, &r, &calib, &params, &orig_t);
        let t = StepTargets {
            dir: 2,
            jump: true,
            hook: false,
            fire: true,
            aim: 0.9,
            soft: None,
            mask: HeadMask::ALL,
            weight: 1.0,
            hook_scale: 1.0,
            hook_latch: false,
        };
        let cfg = LossConfig {
            soft_mix: 0.0,
            aim_kappa: decoder.config().aim_kappa,
            ..LossConfig::default()
        };
        let (l1, g1, dn1, _) = decoder_bc_loss_and_grad(&decoder, &r, &calib, &params, &t, &cfg);
        assert!((l0 - l1.total).abs() < 1e-5, "{l0} vs {}", l1.total);
        let _ = add_l1_penalty;
        assert_eq!(g0.direction_lr_w.len(), g1.direction_lr_w.len());
        for (a, b) in g0
            .direction_lr_w
            .iter()
            .chain(&g0.jump_w)
            .chain(&g0.hook_w)
            .chain(&g0.fire_w)
            .chain(&g0.aim_pair_theta)
            .zip(
                g1.direction_lr_w
                    .iter()
                    .chain(&g1.jump_w)
                    .chain(&g1.hook_w)
                    .chain(&g1.fire_w)
                    .chain(&g1.aim_pair_theta),
            )
        {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        for (a, b) in dn0.iter().zip(&dn1) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// Finite differences of the whole (soft-mixed, weighted, masked) loss against the analytic
    /// `dL/d(dn_rates)` and the decoder gradients.
    #[test]
    fn gradients_match_finite_differences_with_soft_targets_and_masks() {
        let (model, decoder, params, calib) = setup();
        let r = rates(model.num_outputs(), 2.3);
        let t = StepTargets {
            dir: 0,
            jump: false,
            hook: true,
            fire: false,
            aim: -1.2,
            soft: Some(SoftTargets {
                dir: [0.5, 0.3, 0.2],
                jump: 0.2,
                hook: 0.8,
                fire: 0.1,
            }),
            mask: HeadMask {
                fire: false,
                ..HeadMask::ALL
            },
            weight: 2.0,
            hook_scale: 1.0,
            hook_latch: false,
        };
        let cfg = LossConfig {
            w_dir: 1.0,
            w_jump: 0.6,
            w_hook: 1.4,
            w_fire: 1.0,
            w_aim: 0.7,
            soft_mix: 0.5,
            pos_weight: [1.5, 2.0, 1.0],
            hazard_pos_weight: None,
            aim_kappa: 4.0,
            soft_smoothing: 0.02,
        };
        let (_, g, dn, _) = decoder_bc_loss_and_grad(&decoder, &r, &calib, &params, &t, &cfg);
        let loss_at = |r: &[f32], p: &DecoderParams| decoder_bc_loss_and_grad(&decoder, r, &calib, p, &t, &cfg).0.total;
        let eps = 1e-3f32;
        for i in (0..r.len()).step_by(7) {
            let (mut a, mut b) = (r.clone(), r.clone());
            a[i] += eps;
            b[i] -= eps;
            let numeric = (loss_at(&a, &params) - loss_at(&b, &params)) / (2.0 * eps);
            assert!(
                (dn[i] - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                "dn[{i}]: {} vs {numeric}",
                dn[i]
            );
        }
        for i in 0..params.hook_w.len() {
            let (mut a, mut b) = (params.clone(), params.clone());
            a.hook_w[i] += eps;
            b.hook_w[i] -= eps;
            let numeric = (loss_at(&r, &a) - loss_at(&r, &b)) / (2.0 * eps);
            assert!(
                (g.hook_w[i] - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                "hook_w[{i}]"
            );
        }
        for i in 0..params.aim_pair_theta.len() {
            let (mut a, mut b) = (params.clone(), params.clone());
            a.aim_pair_theta[i] += eps;
            b.aim_pair_theta[i] -= eps;
            let numeric = (loss_at(&r, &a) - loss_at(&r, &b)) / (2.0 * eps);
            assert!(
                (g.aim_pair_theta[i] - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                "aim theta[{i}]"
            );
        }
        // The masked fire head must have produced no gradient at all.
        assert!(g.fire_w.iter().all(|&x| x == 0.0) && g.fire_b == 0.0);
    }

    fn intent_setup() -> (FlyModel, DecoderModel, DecoderParams, DnCalibration) {
        let (model, decoder, params, calib) = setup();
        let mut params = params.with_intent_hook();
        // Move the release hazard off the mirror image so it is a head of its own.
        let r = params.hook_release.as_mut().unwrap();
        for (i, w) in r.w.iter_mut().enumerate() {
            *w += 0.3 - 0.2 * i as f32;
        }
        r.b += 0.15;
        (model, decoder, params, calib)
    }

    /// The intent upgrade is the legacy head in function: with the release hazard the mirror image of the press hazard, the probability of
    /// the hook key is the legacy head's whatever the latch is, **bit for bit**.
    #[test]
    fn the_intent_upgrade_of_a_legacy_head_gives_the_same_hook_logit_for_both_latches() {
        let (model, decoder, params, calib) = setup();
        let up = params.with_intent_hook();
        assert!(up.is_intent() && !params.is_intent());
        for seed in [0.2f32, 1.1, 2.9] {
            let r = rates(model.num_outputs(), seed);
            let z = calib.z(&r, decoder.config().z_clip);
            let legacy = decoder_logits(&decoder, &z, &params, false);
            for latch in [false, true] {
                let i = decoder_logits(&decoder, &z, &up, latch);
                assert_eq!(i.hook.to_bits(), legacy.hook.to_bits(), "latch {latch}");
                assert_eq!(
                    (i.dir, i.jump.to_bits(), i.fire.to_bits()),
                    (legacy.dir, legacy.jump.to_bits(), legacy.fire.to_bits())
                );
            }
            // A legacy head ignores the latch.
            assert_eq!(decoder_logits(&decoder, &z, &params, true), legacy);
        }
    }

    /// Gradients of the intent head by finite differences, for both latches: only the hazard the latch selects gets a gradient (the other's
    /// is exactly zero: the transition is structurally the one the latch allows), the decoder weights and `dL/d(dn_rates)` match.
    #[test]
    fn the_intent_head_gradients_match_finite_differences_and_only_the_selected_hazard_gets_one() {
        let (model, decoder, params, calib) = intent_setup();
        let r = rates(model.num_outputs(), 0.9);
        let cfg = LossConfig {
            hazard_pos_weight: Some([2.5, 4.0]),
            soft_mix: 0.3,
            ..LossConfig::default()
        };
        for latch in [false, true] {
            for hook in [false, true] {
                let t = StepTargets {
                    dir: 1,
                    jump: false,
                    hook,
                    fire: false,
                    aim: 0.2,
                    soft: Some(SoftTargets {
                        dir: [0.2, 0.5, 0.3],
                        jump: 0.2,
                        hook: 0.6,
                        fire: 0.1,
                    }),
                    mask: HeadMask::ALL,
                    weight: 1.3,
                    hook_scale: 1.0,
                    hook_latch: latch,
                };
                let (_, g, dn, logits) = decoder_bc_loss_and_grad(&decoder, &r, &calib, &params, &t, &cfg);
                // The hook logit given the latch: the press hazard, or minus the release hazard.
                let z = calib.z(&r, decoder.config().z_clip);
                assert_eq!(logits.hook, crate::decoder::hook_logit(&decoder, &params, &z, latch));
                let (sel_w, other_w, other_b, sel_b) = if latch {
                    (&g.hook_release_w, &g.hook_w, g.hook_b, g.hook_release_b)
                } else {
                    (&g.hook_w, &g.hook_release_w, g.hook_release_b, g.hook_b)
                };
                assert!(
                    sel_w.iter().any(|&x| x != 0.0) && sel_b != 0.0,
                    "the selected hazard learns"
                );
                assert!(
                    other_w.iter().all(|&x| x == 0.0) && other_b == 0.0,
                    "the other hazard gets nothing"
                );
                let loss_at =
                    |r: &[f32], p: &DecoderParams| decoder_bc_loss_and_grad(&decoder, r, &calib, p, &t, &cfg).0.total;
                let eps = 1e-3f32;
                for i in (0..r.len()).step_by(5) {
                    let (mut a, mut b) = (r.clone(), r.clone());
                    a[i] += eps;
                    b[i] -= eps;
                    let numeric = (loss_at(&a, &params) - loss_at(&b, &params)) / (2.0 * eps);
                    assert!(
                        (dn[i] - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                        "dn[{i}] latch {latch}: {} vs {numeric}",
                        dn[i]
                    );
                }
                for i in 0..params.hook_w.len() {
                    let (mut a, mut b) = (params.clone(), params.clone());
                    a.hook_w[i] += eps;
                    b.hook_w[i] -= eps;
                    let numeric = (loss_at(&r, &a) - loss_at(&r, &b)) / (2.0 * eps);
                    assert!(
                        (g.hook_w[i] - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                        "press w[{i}] latch {latch}"
                    );
                    let (mut a, mut b) = (params.clone(), params.clone());
                    a.hook_release.as_mut().unwrap().w[i] += eps;
                    b.hook_release.as_mut().unwrap().w[i] -= eps;
                    let numeric = (loss_at(&r, &a) - loss_at(&r, &b)) / (2.0 * eps);
                    assert!(
                        (g.hook_release_w[i] - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                        "release w[{i}] latch {latch}: {} vs {numeric}",
                        g.hook_release_w[i]
                    );
                }
                let (mut a, mut b) = (params.clone(), params.clone());
                a.hook_release.as_mut().unwrap().b += eps;
                b.hook_release.as_mut().unwrap().b -= eps;
                let numeric = (loss_at(&r, &a) - loss_at(&r, &b)) / (2.0 * eps);
                assert!(
                    (g.hook_release_b - numeric).abs() < 3e-3 * (1.0 + numeric.abs()),
                    "release b latch {latch}"
                );
            }
        }
    }

    #[test]
    fn a_release_hazard_of_the_wrong_shape_or_with_a_nan_is_refused() {
        let (_, decoder, params, _) = intent_setup();
        assert!(decoder.validate_params_shape(&params).is_ok());
        let mut bad = params.clone();
        bad.hook_release.as_mut().unwrap().w.push(0.0);
        assert!(decoder.validate_params_shape(&bad).is_err());
        let mut bad = params.clone();
        bad.hook_release.as_mut().unwrap().b = f32::NAN;
        assert!(decoder.validate_params_shape(&bad).is_err());
    }
}
