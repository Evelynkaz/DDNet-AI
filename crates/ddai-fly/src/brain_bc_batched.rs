//! The batched twin of [`crate::brain_bc::brain_bc_step`] (task 7.2b): encode → fly (recorded,
//! the whole mini-batch at once through [`crate::batched::BatchedEngine`]) → tied decoder → the
//! shared [`crate::bc`] loss → batched BPTT → encoder gradients, for a *batch* of windows.
//!
//! Per window it does exactly what `brain_bc_step` does (same loss, same activity regulariser
//! taps, same encoder backward); what changes is that the fly's forward/backward run once for
//! all windows (weights read once per substep for the whole batch, SIMD over the windows) instead
//! of once per window. The fly's parameter gradient therefore comes back already **summed over the
//! batch** ([`BcBatchOutput::fly`]); the encoder/decoder gradients stay per window, so a caller
//! can sum them in whatever fixed order it needs for reproducibility.

use ddai_brain::Observation;
use rayon::prelude::*;

use crate::batched::{BatchedEngine, BatchedForwardOptions, BatchedSeqGrad, BatchedSeqInput};
use crate::bc::{HeadLogits, StepLoss};
use crate::brain_bc::{BcSequence, BcStepConfig, add_decoder_gradients};
use crate::decoder::{DecoderGradients, DecoderModel, DecoderParams, DnCalibration, decoder_bc_loss_and_grad};
use crate::encoder::{EncoderGradients, EncoderModel, EncoderParams, RayGridFeatures, compute_proprioception_values};
use crate::model::FlyModel;
use crate::optim::{ParamGradients, activity_regularizer_rate_grad};
use crate::train::MemoryCapExceeded;

/// What one window contributed (the per-window half of [`crate::brain_bc::BcStepOutput`]).
#[derive(Debug, Clone)]
pub struct BcWindowOutput {
    pub loss: StepLoss,
    pub activity_loss: f32,
    /// Sum of the step weights that had a loss.
    pub weight_sum: f32,
    pub encoder: EncoderGradients,
    pub decoder: DecoderGradients,
    /// One entry per decision (also for burn-in decisions).
    pub logits: Vec<HeadLogits>,
}

/// Everything [`brain_bc_batched_step`] produces.
#[derive(Debug, Clone)]
pub struct BcBatchOutput {
    /// `dL/d{a, b, theta}` of the connectome, summed over all windows.
    pub fly: ParamGradients,
    /// One entry per window, in input order.
    pub windows: Vec<BcWindowOutput>,
}

/// One window's encoded inputs, kept for the encoder's backward pass.
struct Encoded {
    features: Vec<RayGridFeatures>,
    proprio: Vec<crate::encoder::ProprioceptionValues>,
    inputs: Vec<Vec<f32>>,
}

fn encode_window(encoder: &EncoderModel, params: &EncoderParams, observations: &[Observation]) -> Encoded {
    let cfg = encoder.ray_grid_config();
    let mut features = Vec::with_capacity(observations.len());
    let mut proprio = Vec::with_capacity(observations.len());
    let mut inputs = Vec::with_capacity(observations.len());
    let mut scratch = RayGridFeatures::new(cfg);
    for obs in observations {
        let an = compute_proprioception_values(&obs.self_state, cfg);
        scratch.compute(obs, cfg);
        let mut input = vec![0.0f32; encoder.num_inputs()];
        encoder.forward(&scratch, &an, params, &mut input);
        features.push(scratch.clone());
        proprio.push(an);
        inputs.push(input);
    }
    Encoded {
        features,
        proprio,
        inputs,
    }
}

/// The loss side of one window, between the forward and the backward pass.
struct Scored {
    loss: StepLoss,
    activity_loss: f32,
    weight_sum: f32,
    decoder: DecoderGradients,
    grad_dn: Vec<Vec<f32>>,
    logits: Vec<HeadLogits>,
    /// `(decision, last substep, dL/dr)` activity-regulariser taps.
    taps: Vec<(usize, usize, Vec<f32>)>,
}

/// Forward + backward over a batch of windows with the batched engine. Panics on a length mismatch
/// between a window's `observations`, `targets` and `v_init` (caller-assembled inputs, like
/// `brain_bc_step`). `memory_cap_bytes` caps the engine's working set (windows are chunked in time
/// exactly when the whole batch does not fit, see [`crate::batched`]); `Err` only if not even a
/// one-decision segment fits.
#[allow(clippy::too_many_arguments)]
pub fn brain_bc_batched_step(
    model: &FlyModel,
    engine: &mut BatchedEngine,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    decoder: &DecoderModel,
    decoder_params: &DecoderParams,
    calib: &DnCalibration,
    seqs: &[BcSequence],
    cfg: &BcStepConfig,
    memory_cap_bytes: Option<usize>,
) -> Result<BcBatchOutput, MemoryCapExceeded> {
    for seq in seqs {
        assert_eq!(
            seq.targets.len(),
            seq.observations.len(),
            "brain_bc_batched_step: observations/targets mismatch"
        );
        assert_eq!(
            seq.v_init.len(),
            model.num_neurons(),
            "brain_bc_batched_step: v_init length"
        );
    }
    if seqs.is_empty() {
        return Ok(BcBatchOutput {
            fly: ParamGradients::zeros_like(model.params()),
            windows: Vec::new(),
        });
    }
    let substeps = model.config().substeps_per_decision as usize;

    // Encode every window (independent, parallel).
    let encoded: Vec<Encoded> = seqs
        .par_iter()
        .map(|seq| encode_window(encoder, encoder_params, &seq.observations))
        .collect();

    // Forward, the whole batch at once.
    let inputs: Vec<BatchedSeqInput<'_>> = seqs
        .iter()
        .zip(&encoded)
        .map(|(seq, enc)| BatchedSeqInput {
            v_init: &seq.v_init,
            inputs: &enc.inputs,
        })
        .collect();
    let want_activity = cfg.activity.weight > 0.0;
    engine.forward(
        model,
        &inputs,
        &BatchedForwardOptions {
            memory_cap_bytes,
            segment_decisions: None,
            type_means: want_activity,
        },
    )?;

    // Losses, decoder gradients and dL/d(dn_rates) per window (independent, parallel; the engine
    // is only read).
    let engine_ref: &BatchedEngine = engine;
    let scored: Vec<Scored> = (0..seqs.len())
        .into_par_iter()
        .map(|b| {
            let seq = &seqs[b];
            let t_len = seq.observations.len();
            let mut loss = StepLoss::default();
            let mut weight_sum = 0.0f32;
            let mut decoder_grads = decoder.zeros_gradients();
            let mut grad_dn = Vec::with_capacity(t_len);
            let mut logits = Vec::with_capacity(t_len);
            let mut dn = vec![0.0f32; model.num_outputs()];
            let mut activity_loss = 0.0f32;
            let mut taps = Vec::new();
            let mut means = vec![0.0f32; model.num_types()];
            for (t, target) in seq.targets.iter().enumerate() {
                engine_ref.dn_rates(b, t, &mut dn);
                let (l, dg, gdn, lg) = decoder_bc_loss_and_grad(decoder, &dn, calib, decoder_params, target, &cfg.loss);
                loss.add(&l);
                if target.weight > 0.0 {
                    weight_sum += target.weight;
                    add_decoder_gradients(&mut decoder_grads, &dg);
                }
                grad_dn.push(gdn);
                logits.push(lg);
                if want_activity {
                    engine_ref.type_mean_rates(b, t, &mut means);
                    for &m in &means {
                        let below = (cfg.activity.low - m).max(0.0);
                        let above = (m - cfg.activity.high).max(0.0);
                        activity_loss += cfg.activity.weight * (below * below + above * above);
                    }
                    let g = activity_regularizer_rate_grad(model, &means, &cfg.activity);
                    if g.iter().any(|&x| x != 0.0) {
                        taps.push((t, substeps - 1, g));
                    }
                }
            }
            Scored {
                loss,
                activity_loss,
                weight_sum,
                decoder: decoder_grads,
                grad_dn,
                logits,
                taps,
            }
        })
        .collect();

    // Backward, the whole batch at once.
    let grads: Vec<BatchedSeqGrad<'_>> = scored
        .iter()
        .map(|s| BatchedSeqGrad {
            grad_dn: &s.grad_dn,
            extra_taps: &s.taps,
        })
        .collect();
    let bptt = engine.backward(model, &grads, false);

    // Encoder gradients per window (independent, parallel).
    let windows: Vec<BcWindowOutput> = scored
        .into_par_iter()
        .zip(encoded.par_iter())
        .zip(bptt.grad_inputs.par_iter())
        .map(|((s, enc), gin)| {
            let mut encoder_grads = encoder.zero_grads();
            for ((features, proprio), grad_input) in enc.features.iter().zip(&enc.proprio).zip(gin) {
                encoder.backward_with_params(features, proprio, encoder_params, grad_input, &mut encoder_grads);
            }
            BcWindowOutput {
                loss: s.loss,
                activity_loss: s.activity_loss,
                weight_sum: s.weight_sum,
                encoder: encoder_grads,
                decoder: s.decoder,
                logits: s.logits,
            }
        })
        .collect();

    Ok(BcBatchOutput {
        fly: bptt.grad,
        windows,
    })
}
