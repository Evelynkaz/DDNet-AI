//! Behaviour-cloning forward/backward through the whole fly pipeline (task 8.2): encode → fly
//! (recorded) → tied decoder → the shared [`crate::bc`] loss → [`crate::backward::backward`] →
//! encoder gradients, for one *window* of decisions.
//!
//! Differs from [`crate::brain_train::brain_train_step`] (7.3's end-to-end differentiability
//! check, left untouched) in what a training run needs: soft targets, head weights/masks and a
//! per-step weight through [`StepTargets`]; the flyvis-style activity regulariser
//! ([`ActivityRegularizerConfig`]) tapped at each decision's last substep; reusable scratch
//! ([`BcWorkspace`]) so a rayon worker does not reallocate the multi-megabyte trajectory recorder
//! per window; and the per-decision head logits back out, for online metrics. The world-model
//! head is not trained here.
//!
//! Also [`brain_bc_forward`], the same pipeline without recording or gradients, for evaluation.

use ddai_brain::Observation;

use crate::backward::{BackwardIndex, BpttScratch, ExtraRateGrad, backward};
use crate::bc::{HeadLogits, LossConfig, StepLoss, StepTargets};
use crate::decoder::decoder_logits;
use crate::decoder::{DecoderGradients, DecoderModel, DecoderParams, DnCalibration, decoder_bc_loss_and_grad};
use crate::encoder::{EncoderGradients, EncoderModel, EncoderParams, RayGridFeatures, compute_proprioception_values};
use crate::model::FlyModel;
use crate::optim::{ActivityRegularizerConfig, ParamGradients, activity_regularizer_rate_grad};
use crate::recorder::TrajectoryRecorder;
use crate::state::FlyState;

/// A window of consecutive decisions: the starting membrane state, the observations, and what
/// each decision is trained towards.
#[derive(Debug, Clone)]
pub struct BcSequence {
    pub v_init: Vec<f32>,
    pub observations: Vec<Observation>,
    pub targets: Vec<StepTargets>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BcStepConfig {
    pub loss: LossConfig,
    pub activity: ActivityRegularizerConfig,
}

/// Everything one window produces. Losses and gradients are **sums over the window's decisions**
/// (each already multiplied by its step weight); the caller divides by the batch's total weight.
#[derive(Debug, Clone)]
pub struct BcStepOutput {
    pub loss: StepLoss,
    /// The activity regulariser's value, summed over the window's decisions.
    pub activity_loss: f32,
    /// Sum of the step weights that had a loss.
    pub weight_sum: f32,
    pub encoder: EncoderGradients,
    pub fly: ParamGradients,
    pub decoder: DecoderGradients,
    pub final_v: Vec<f32>,
    /// One entry per decision (also for burn-in decisions).
    pub logits: Vec<HeadLogits>,
}

/// Per-thread scratch reused across windows.
pub struct BcWorkspace {
    recorder: TrajectoryRecorder,
    bptt: BpttScratch,
    features: RayGridFeatures,
    input_buf: Vec<f32>,
    max_decisions: usize,
}

impl BcWorkspace {
    /// `max_decisions`: the longest window this workspace will be asked to run.
    pub fn new(model: &FlyModel, encoder: &EncoderModel, max_decisions: usize) -> Self {
        let substeps = model.config().substeps_per_decision as usize;
        BcWorkspace {
            recorder: TrajectoryRecorder::new(model.num_neurons(), max_decisions * substeps),
            bptt: BpttScratch::new(model),
            features: RayGridFeatures::new(encoder.ray_grid_config()),
            input_buf: vec![0.0; encoder.num_inputs()],
            max_decisions,
        }
    }
}

/// Forward + backward over one window. Panics on a length mismatch between `observations`,
/// `targets` and `v_init`, or a window longer than the workspace (caller-assembled inputs).
#[allow(clippy::too_many_arguments)]
pub fn brain_bc_step(
    model: &FlyModel,
    index: &BackwardIndex,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &DecoderParams,
    calib: &DnCalibration,
    seq: &BcSequence,
    cfg: &BcStepConfig,
    ws: &mut BcWorkspace,
) -> BcStepOutput {
    let t_decisions = seq.observations.len();
    assert_eq!(
        seq.targets.len(),
        t_decisions,
        "brain_bc_step: observations/targets mismatch"
    );
    assert_eq!(seq.v_init.len(), model.num_neurons(), "brain_bc_step: v_init length");
    assert!(
        t_decisions <= ws.max_decisions,
        "brain_bc_step: window longer than the workspace"
    );
    let substeps = model.config().substeps_per_decision as usize;

    ws.recorder.reset();
    let mut state = FlyState::new(model);
    state.set_v(model, &seq.v_init);

    // Forward, recording every substep; the features/proprioception of each decision are kept for
    // the encoder's backward pass.
    let mut features: Vec<RayGridFeatures> = Vec::with_capacity(t_decisions);
    let mut proprio = Vec::with_capacity(t_decisions);
    let mut dn_per_decision: Vec<Vec<f32>> = Vec::with_capacity(t_decisions);
    // The encoder's input vector of each decision, kept only for the control readout (task 8.7).
    let mut enc_per_decision: Vec<Vec<f32>> = Vec::with_capacity(if decoder.reads_encoder() { t_decisions } else { 0 });
    let mut type_means: Vec<Vec<f32>> = Vec::with_capacity(t_decisions);
    for obs in &seq.observations {
        let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
        ws.features.compute(obs, encoder.ray_grid_config());
        encoder.forward(&ws.features, &an, encoder_params, &mut ws.input_buf);
        let out = state.step_decision_recording(model, &ws.input_buf, &mut ws.recorder);
        dn_per_decision.push(out.dn_rates.to_vec());
        type_means.push(out.per_type_mean_rate.to_vec());
        if decoder.reads_encoder() {
            enc_per_decision.push(ws.input_buf.clone());
        }
        features.push(ws.features.clone());
        proprio.push(an);
    }
    let final_v = state.v().to_vec();

    // Losses, decoder gradients and dL/d(dn_rates) per decision.
    let mut loss = StepLoss::default();
    let mut weight_sum = 0.0f32;
    let mut decoder_grads = decoder.zeros_gradients();
    let mut grad_dn: Vec<Vec<f32>> = Vec::with_capacity(t_decisions);
    let mut logits = Vec::with_capacity(t_decisions);
    for (t, target) in seq.targets.iter().enumerate() {
        let enc_t: &[f32] = enc_per_decision.get(t).map_or(&[], Vec::as_slice);
        let (l, dg, gdn, lg) = decoder_bc_loss_and_grad(
            decoder,
            &dn_per_decision[t],
            enc_t,
            calib,
            decoder_params,
            target,
            &cfg.loss,
        );
        loss.add(&l);
        if target.weight > 0.0 {
            weight_sum += target.weight;
            add_decoder_gradients(&mut decoder_grads, &dg);
        }
        grad_dn.push(gdn);
        logits.push(lg);
    }

    // Activity regulariser: penalises a type's mean rate outside a band, tapped at the last
    // substep of every decision.
    let mut activity_loss = 0.0f32;
    let mut act_taps: Vec<(usize, Vec<f32>)> = Vec::new();
    if cfg.activity.weight > 0.0 {
        for (t, means) in type_means.iter().enumerate() {
            for &m in means {
                let below = (cfg.activity.low - m).max(0.0);
                let above = (m - cfg.activity.high).max(0.0);
                activity_loss += cfg.activity.weight * (below * below + above * above);
            }
            let g = activity_regularizer_rate_grad(model, means, &cfg.activity);
            if g.iter().any(|&x| x != 0.0) {
                act_taps.push((t, g));
            }
        }
    }

    let grad_dn_refs: Vec<&[f32]> = grad_dn.iter().map(Vec::as_slice).collect();
    let extra: Vec<ExtraRateGrad<'_>> = act_taps
        .iter()
        .map(|(decision, grad)| ExtraRateGrad {
            decision: *decision,
            local_substep: substeps - 1,
            grad,
        })
        .collect();
    let bptt = backward(
        model,
        index,
        &ws.recorder,
        &seq.v_init,
        t_decisions,
        &grad_dn_refs,
        &extra,
        false,
        &mut ws.bptt,
    );

    let mut encoder_grads = encoder.zero_grads();
    for t in 0..t_decisions {
        encoder.backward_with_params(
            &features[t],
            &proprio[t],
            encoder_params,
            &bptt.grad_inputs[t],
            &mut encoder_grads,
        );
    }

    BcStepOutput {
        loss,
        activity_loss,
        weight_sum,
        encoder: encoder_grads,
        fly: ParamGradients {
            a: bptt.grad_a,
            b: bptt.grad_b,
            theta: bptt.grad_theta,
        },
        decoder: decoder_grads,
        final_v,
        logits,
    }
}

/// The hook head's readout alone (task 8.7): the network is **frozen**, so the pipeline runs forward only (no recording, no backward pass)
/// and only the hook head is scored; the output's connectome and encoder gradients are zero. Same windows, targets, loss and per-step
/// weights as [`brain_bc_step`], so a readout trained here sees exactly the data of a full BC step.
#[allow(clippy::too_many_arguments)]
pub fn brain_readout_step(
    model: &FlyModel,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &DecoderParams,
    calib: &DnCalibration,
    seq: &BcSequence,
    cfg: &BcStepConfig,
    ws: &mut BcWorkspace,
) -> BcStepOutput {
    assert_eq!(
        seq.targets.len(),
        seq.observations.len(),
        "brain_readout_step: observations/targets mismatch"
    );
    assert_eq!(
        seq.v_init.len(),
        model.num_neurons(),
        "brain_readout_step: v_init length"
    );
    let mut state = FlyState::new(model);
    state.set_v(model, &seq.v_init);
    let mut loss = StepLoss::default();
    let mut weight_sum = 0.0f32;
    let mut decoder_grads = decoder.zeros_gradients();
    let mut logits = Vec::with_capacity(seq.observations.len());
    for (obs, target) in seq.observations.iter().zip(&seq.targets) {
        let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
        ws.features.compute(obs, encoder.ray_grid_config());
        encoder.forward(&ws.features, &an, encoder_params, &mut ws.input_buf);
        let out = state.step_decision(model, &ws.input_buf);
        let hook_only = StepTargets {
            mask: crate::bc::HeadMask {
                hook: target.mask.hook,
                ..crate::bc::HeadMask::NONE
            },
            ..*target
        };
        let (l, dg, _, lg) = decoder_bc_loss_and_grad(
            decoder,
            out.dn_rates,
            &ws.input_buf,
            calib,
            decoder_params,
            &hook_only,
            &cfg.loss,
        );
        loss.add(&l);
        if target.weight > 0.0 {
            weight_sum += target.weight;
            add_decoder_gradients(&mut decoder_grads, &dg);
        }
        logits.push(lg);
    }
    BcStepOutput {
        loss,
        activity_loss: 0.0,
        weight_sum,
        encoder: encoder.zero_grads(),
        fly: ParamGradients::zeros_like(model.params()),
        decoder: decoder_grads,
        final_v: state.v().to_vec(),
        logits,
    }
}

/// The same pipeline forward only, for evaluation: the head logits of every decision of `seq`,
/// starting from `v_init`. `hook_latches[t]` is the fly's own previous hook command at decision `t` (what an intent hook head selects its
/// hazard by; ignored by a legacy one); it has one entry per observation.
#[allow(clippy::too_many_arguments)]
pub fn brain_bc_forward(
    model: &FlyModel,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &DecoderParams,
    calib: &DnCalibration,
    v_init: &[f32],
    observations: &[Observation],
    hook_latches: &[bool],
) -> Vec<HeadLogits> {
    assert_eq!(
        hook_latches.len(),
        observations.len(),
        "brain_bc_forward: observations/latches mismatch"
    );
    let mut state = FlyState::new(model);
    state.set_v(model, v_init);
    let mut features = RayGridFeatures::new(encoder.ray_grid_config());
    let mut input_buf = vec![0.0f32; encoder.num_inputs()];
    let mut z = vec![0.0f32; decoder.num_outputs()];
    observations
        .iter()
        .zip(hook_latches)
        .map(|(obs, &latch)| {
            let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
            features.compute(obs, encoder.ray_grid_config());
            encoder.forward(&features, &an, encoder_params, &mut input_buf);
            let out = state.step_decision(model, &input_buf);
            calib.z_into(out.dn_rates, decoder.config().z_clip, &mut z);
            decoder_logits(decoder, &z, &input_buf, decoder_params, latch)
        })
        .collect()
}

/// `acc += g` for every decoder gradient field.
pub fn add_decoder_gradients(acc: &mut DecoderGradients, g: &DecoderGradients) {
    let add = |a: &mut [f32], b: &[f32]| {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
    };
    add(&mut acc.direction_lr_w, &g.direction_lr_w);
    acc.direction_lr_b += g.direction_lr_b;
    add(&mut acc.direction_stop_w, &g.direction_stop_w);
    acc.direction_stop_b += g.direction_stop_b;
    add(&mut acc.jump_w, &g.jump_w);
    acc.jump_b += g.jump_b;
    add(&mut acc.hook_w, &g.hook_w);
    acc.hook_b += g.hook_b;
    add(&mut acc.fire_w, &g.fire_w);
    acc.fire_b += g.fire_b;
    add(&mut acc.aim_pair_theta, &g.aim_pair_theta);
    add(&mut acc.aim_unpaired_theta, &g.aim_unpaired_theta);
    add(&mut acc.hook_release_w, &g.hook_release_w);
    acc.hook_release_b += g.hook_release_b;
    acc.hook_wide.add(&g.hook_wide);
}
