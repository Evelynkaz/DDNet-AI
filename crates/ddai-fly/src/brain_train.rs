//! End-to-end differentiability (task 7.3, acceptance criterion 6): one function that, for a
//! recorded sequence of [`ddai_brain::Observation`]s and per-decision targets, runs the whole
//! pipeline forward (encode -> `t_decisions` fly steps, recorded -> decode + world-model head),
//! computes every configured loss, and runs [`crate::backward::backward`] once to get gradients
//! covering **all four** trainable groups in one call: the encoder's `g`/`c`
//! ([`crate::encoder::EncoderGradients`]), the connectome's `a`/`b`/`theta`
//! ([`crate::optim::ParamGradients`], via 7.2's own `backward`), the decoder's linear
//! heads/preferred angles ([`crate::decoder::DecoderGradients`]), and the world-model head's
//! linear heads ([`crate::world_model::WorldModelGradients`]).
//!
//! Not a batch/training-loop API (contrast `crate::train::train_step`, which parallelizes many
//! independent sequences with `rayon`): this is deliberately the smallest unit "does the whole
//! chain actually differentiate correctly", the thing `tests/gradcheck_brain.rs` checks end to
//! end. `crate::demo_brain`'s learning demo (acceptance criterion 9) calls this once per sample
//! and reduces the results itself.
//!
//! ## Recovering `dn_rates` from the recorder
//! `crate::state::DecisionOutput::dn_rates` for decision `t` is `0.5 * (r_before_last + r_last)`
//! (task 7.1's own averaging rule — see `crate::backward`'s module doc comment). This function
//! re-derives exactly that from the recorded trajectory rather than re-running `step_decision`
//! (which would redo the whole recurrence): `r_last = f(V_last)` is exactly `state.v()` right
//! after the decision's `step_decision_recording` call (already computed once, reused both for
//! the decoder's `dn_rates` and the world-model readout — see [`WORLD_MODEL_READOUT_SUBSTEP`]'s
//! doc comment for why the two share a sample point); `r_before_last = f(V)` one flat substep
//! earlier (`recorder.v_at(last - 1)`, or the window's own `v_init` for decision `0` with
//! `substeps_per_decision == 1` — the same boundary case `crate::backward::backward`'s own module
//! doc comment documents).

use ddai_brain::Observation;

use crate::activation::activation;
use crate::backward::{BackwardIndex, BpttScratch, ExtraRateGrad, backward};
use crate::decoder::{
    DecoderGradients, DecoderModel, DecoderParams, DecoderTargets, DnCalibration, add_l1_penalty,
    decoder_loss_and_grad, l1_penalty_value,
};
use crate::encoder::{EncoderGradients, EncoderModel, EncoderParams, RayGridFeatures, compute_proprioception_values};
use crate::model::FlyModel;
use crate::optim::ParamGradients;
use crate::recorder::TrajectoryRecorder;
use crate::state::FlyState;
use crate::world_model::{
    HorizonTargets, WorldModelGradients, WorldModelHead, WorldModelParams, world_model_loss_and_grad,
};

/// The world-model readout is taken at the same point in the decision as `r_last` (the *last*
/// substep, before it's blended with `r_before_last` for `dn_rates`) — a single, documented
/// convention rather than a separate configurable point: "what the fly's central state looks like
/// right at decision time" is unambiguous only if it's tied to one specific substep, and the last
/// one is the one already being computed for the decoder's own averaging anyway.
pub const WORLD_MODEL_READOUT_SUBSTEP_FROM_END: usize = 0; // 0 = the decision's last substep.

/// One decision's targets: the decoder's (task spec criterion 4) and, optionally, the world
/// model's (criterion 5) — `None` for the latter skips that decision's world-model loss/gradient
/// entirely (a caller training only the action heads on most decisions and the world model on a
/// few, say, passes `None` for the rest).
#[derive(Debug, Clone, Default)]
pub struct DecisionTargets {
    pub decoder: DecoderTargets,
    pub world_model: Option<[HorizonTargets; 3]>,
}

/// A whole recorded episode/window: `observations.len() == targets.len() == t_decisions`.
#[derive(Debug, Clone)]
pub struct BrainSequence {
    pub v_init: Vec<f32>,
    pub observations: Vec<Observation>,
    pub targets: Vec<DecisionTargets>,
}

/// Every gradient [`brain_train_step`] produces, one field per trainable group.
#[derive(Debug, Clone)]
pub struct BrainGradients {
    pub encoder: EncoderGradients,
    pub fly: ParamGradients,
    pub decoder: DecoderGradients,
    pub world_model: WorldModelGradients,
}

/// Runs the whole pipeline forward, computes every configured loss, and backpropagates once.
/// Returns `(total_loss, gradients, final_v)` (`final_v`: the window's ending `V`, for a caller
/// chaining truncated-BPTT windows across an episode). Panics if
/// `seq.observations.len() != seq.targets.len()` or `seq.v_init.len() != model.num_neurons()` (a
/// caller-assembled-sequence mismatch, matching `crate::backward::backward`'s own convention).
#[allow(clippy::too_many_arguments)]
pub fn brain_train_step(
    model: &FlyModel,
    index: &BackwardIndex,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &DecoderParams,
    calib: &DnCalibration,
    world_model: &WorldModelHead,
    world_model_params: &WorldModelParams,
    seq: &BrainSequence,
) -> (f32, BrainGradients, Vec<f32>) {
    let t_decisions = seq.observations.len();
    assert_eq!(
        seq.targets.len(),
        t_decisions,
        "brain_train_step: observations/targets length mismatch"
    );
    assert_eq!(
        seq.v_init.len(),
        model.num_neurons(),
        "brain_train_step: v_init.len() must equal num_neurons"
    );

    let r_max = model.config().r_max;
    let substeps = model.config().substeps_per_decision as usize;
    let mut recorder = TrajectoryRecorder::new(model.num_neurons(), t_decisions * substeps);
    let mut state = FlyState::new(model);
    state.set_v(model, &seq.v_init);
    let mut input_buf = vec![0.0f32; encoder.num_inputs()];

    // Forward: encode + record every substep. Features/an_values/r_last are kept per decision
    // (not just used and discarded) since the encoder's own `backward` and the world-model
    // readout both need them again below.
    let mut features_per_decision = Vec::with_capacity(t_decisions);
    let mut an_values_per_decision = Vec::with_capacity(t_decisions);
    let mut r_last_per_decision: Vec<Vec<f32>> = Vec::with_capacity(t_decisions);
    for obs in &seq.observations {
        let an_values = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
        let mut features = RayGridFeatures::new(encoder.ray_grid_config());
        features.compute(obs, encoder.ray_grid_config());
        encoder.forward(&features, &an_values, encoder_params, &mut input_buf);

        state.step_decision_recording(model, &input_buf, &mut recorder);
        r_last_per_decision.push(state.v().iter().map(|&v| activation(v, r_max)).collect());
        features_per_decision.push(features);
        an_values_per_decision.push(an_values);
    }
    let final_v = state.v().to_vec();

    // Losses + grad_dn (decoder) + grad_r (world model), per decision.
    let mut total_loss = 0.0f32;
    let mut grad_dn_per_decision: Vec<Vec<f32>> = Vec::with_capacity(t_decisions);
    let mut decoder_grads = decoder.zeros_gradients();
    let mut world_model_grads = world_model.zeros_gradients();
    let mut extra_taps: Vec<(usize, usize, Vec<f32>)> = Vec::new();

    for (t, target) in seq.targets.iter().enumerate() {
        let last = t * substeps + substeps - 1;
        let r_before_last: Vec<f32> = if last >= 1 {
            recorder.v_at(last - 1).iter().map(|&v| activation(v, r_max)).collect()
        } else {
            seq.v_init.iter().map(|&v| activation(v, r_max)).collect()
        };
        let r_last = &r_last_per_decision[t];
        let dn_rates_full: Vec<f32> = r_last
            .iter()
            .zip(&r_before_last)
            .map(|(&a, &b)| 0.5 * (a + b))
            .collect();
        let dn_rates: Vec<f32> = model
            .output_neuron_indices()
            .iter()
            .map(|&i| dn_rates_full[i as usize])
            .collect();

        let (loss, d_grads, grad_dn) =
            decoder_loss_and_grad(decoder, &dn_rates, calib, decoder_params, &target.decoder);
        total_loss += loss;
        add_decoder_gradients(&mut decoder_grads, &d_grads);
        grad_dn_per_decision.push(grad_dn);

        if let Some(wm_targets) = &target.world_model {
            let (wm_loss, mut wm_grads, mut grad_r) =
                world_model_loss_and_grad(world_model, r_last, model.num_neurons(), world_model_params, wm_targets);
            // Review round 1, F5 (CONFIRMED): `world_model_loss_and_grad` itself is an unweighted
            // loss (same "mechanism here, policy at the call site" split the rest of this crate
            // uses) — the world-model head's own `loss_weight` (FLY.md §7: "0.1-0.3 of the main
            // loss") must be applied here, at the one place that actually combines it with the
            // decoder's loss, or a large-magnitude world-model MSE silently swamps the action
            // loss's gradient into the shared connectome/encoder parameters.
            let weight = world_model.loss_weight();
            total_loss += weight * wm_loss;
            scale_world_model_gradients(&mut wm_grads, weight);
            for g in &mut grad_r {
                *g *= weight;
            }
            add_world_model_gradients(&mut world_model_grads, &wm_grads);
            extra_taps.push((t, substeps - 1, grad_r));
        }
    }

    // Review round 1, F12 (CONFIRMED): `add_l1_penalty` had no callers anywhere in the crate, so
    // `DecoderConfig::l1_weight` (FLY.md §6's "L1 on `W_dec`") was silently dead. Applied once per
    // `brain_train_step` call (not once per decision in the loop above) -- it's a penalty on the
    // *parameters* themselves, not a per-decision quantity, so adding it `t_decisions` times would
    // scale the regularization strength with window length for no reason.
    total_loss += l1_penalty_value(decoder_params, decoder.config().l1_weight);
    add_l1_penalty(&mut decoder_grads, decoder_params, decoder.config().l1_weight);

    let grad_dn_refs: Vec<&[f32]> = grad_dn_per_decision.iter().map(Vec::as_slice).collect();
    let extra_refs: Vec<ExtraRateGrad<'_>> = extra_taps
        .iter()
        .map(|(decision, local_substep, grad)| ExtraRateGrad {
            decision: *decision,
            local_substep: *local_substep,
            grad,
        })
        .collect();
    let mut bptt_scratch = BpttScratch::new(model);
    let bptt = backward(
        model,
        index,
        &recorder,
        &seq.v_init,
        t_decisions,
        &grad_dn_refs,
        &extra_refs,
        false,
        &mut bptt_scratch,
    );

    let mut encoder_grads = crate::encoder::EncoderGradients::zeros(encoder.num_params());
    for t in 0..t_decisions {
        encoder.backward(
            &features_per_decision[t],
            &an_values_per_decision[t],
            &bptt.grad_inputs[t],
            &mut encoder_grads,
        );
    }

    let fly_grads = ParamGradients {
        a: bptt.grad_a,
        b: bptt.grad_b,
        theta: bptt.grad_theta,
    };

    (
        total_loss,
        BrainGradients {
            encoder: encoder_grads,
            fly: fly_grads,
            decoder: decoder_grads,
            world_model: world_model_grads,
        },
        final_v,
    )
}

fn add_decoder_gradients(acc: &mut DecoderGradients, g: &DecoderGradients) {
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

fn add_world_model_gradients(acc: &mut WorldModelGradients, g: &WorldModelGradients) {
    for k in 0..3 {
        for (a, b) in acc.horizons[k].w_reg.iter_mut().zip(&g.horizons[k].w_reg) {
            *a += b;
        }
        for (a, b) in acc.horizons[k].b_reg.iter_mut().zip(&g.horizons[k].b_reg) {
            *a += b;
        }
        for (a, b) in acc.horizons[k].w_bin.iter_mut().zip(&g.horizons[k].w_bin) {
            *a += b;
        }
        for (a, b) in acc.horizons[k].b_bin.iter_mut().zip(&g.horizons[k].b_bin) {
            *a += b;
        }
    }
}

fn scale_world_model_gradients(g: &mut WorldModelGradients, scale: f32) {
    for h in &mut g.horizons {
        for x in h
            .w_reg
            .iter_mut()
            .chain(&mut h.b_reg)
            .chain(&mut h.w_bin)
            .chain(&mut h.b_bin)
        {
            *x *= scale;
        }
    }
}

#[cfg(test)]
mod tests;
