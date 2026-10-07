//! The batched forward/backward of the fly as a **policy** (task 8.5b): windows of observations in, head logits out, and any loss on
//! the head logits back through the decoder, the batched BPTT and the encoder to the parameters.
//!
//! It is [`crate::brain_bc_batched::brain_bc_batched_step`] with the loss taken out. A recurrent PPO learner needs the logits of
//! *all* decisions of a mini-batch before it can form its loss (the ratio, the entropy and the KL anchor of a decision are functions
//! of the logits of **both views** of a two-view fly, see below), so the work is two calls around the caller's own code:
//!
//! 1. [`policy_forward`]: encode every window, run the batch through the engine (recording), decode the head logits;
//! 2. the caller turns the logits into `dL/d(logits)` per decision;
//! 3. [`policy_backward`]: decoder backward, batched BPTT, encoder backward.
//!
//! **Two views.** A fly trained with `HookView::MaskedForHookHead` plays in two views (`two_view`): the full network decides everything but
//! the hook, a second pass of the same weights over the observation with the own hook state hidden decides the hook. Both views are lanes
//! of one engine batch (lane `b` the full view of window `b`, lane `n + b` its masked view): a gradient on the hook logit goes to the
//! masked lane, every other gradient to the full lane, and the parameter gradient is the sum, exactly like the BC loss of that mode
//! (`ddai-train::trainer::two_view_losses`). Windows are plain data; a window starts from the given membrane state(s) and may carry any
//! number of unscored burn-in decisions at the front (the caller leaves their `dL/d(logits)` zero).

use ddai_brain::Observation;
use rayon::prelude::*;

use crate::batched::{BatchedEngine, BatchedForwardOptions, BatchedSeqGrad, BatchedSeqInput};
use crate::bc::{HeadLogits, HeadMask, combine_hook_view, mask_own_hook};
use crate::brain_bc_batched::{Encoded, encode_window};
use crate::decoder::{
    DecoderGradients, DecoderModel, DecoderParams, DnCalibration, decoder_logits, decoder_logits_backward,
};
use crate::encoder::{EncoderGradients, EncoderModel, EncoderParams};
use crate::model::FlyModel;
use crate::optim::ParamGradients;
use crate::train::MemoryCapExceeded;

/// One window of consecutive decisions of one episode.
#[derive(Debug, Clone)]
pub struct PolicyWindow {
    /// The membrane state before the first decision, of the full view and (two views) of the masked view.
    pub v_init: Vec<f32>,
    pub v_init_masked: Vec<f32>,
    pub observations: Vec<Observation>,
}

/// What [`policy_forward`] leaves for [`policy_backward`] (and for the caller to read).
pub struct PolicyForward {
    two_view: bool,
    /// Per lane (full lanes first, then the masked ones).
    encoded: Vec<Encoded>,
    n_windows: usize,
    /// The logits the fly **plays** with, `[window][decision]`: the full view's, the hook head from the masked view when two views.
    pub logits: Vec<Vec<HeadLogits>>,
}

/// The parameter gradients of one [`policy_backward`].
pub struct PolicyBackward {
    /// `dL/d{a, b, theta}`, summed over the windows (and views).
    pub fly: ParamGradients,
    /// Per window, the two views summed.
    pub encoder: Vec<EncoderGradients>,
    pub decoder: Vec<DecoderGradients>,
}

/// The models and parameters a pass runs with.
#[derive(Clone, Copy)]
pub struct PolicyNet<'a> {
    pub model: &'a FlyModel,
    pub encoder: &'a EncoderModel,
    pub encoder_params: &'a EncoderParams,
    pub decoder: &'a DecoderModel,
    pub decoder_params: &'a DecoderParams,
    pub calib: &'a DnCalibration,
}

/// Encodes and runs `windows` through `engine` and decodes the head logits of every decision.
pub fn policy_forward(
    net: PolicyNet<'_>,
    engine: &mut BatchedEngine,
    windows: &[PolicyWindow],
    two_view: bool,
    memory_cap_bytes: Option<usize>,
) -> Result<PolicyForward, MemoryCapExceeded> {
    let n = windows.len();
    let n_neurons = net.model.num_neurons();
    // The PPO policy has no latch (task 8.6): `decoder_logits` is called with `false` below, which is only right for a legacy hook head.
    debug_assert!(
        !net.decoder_params.is_intent(),
        "policy_forward: an intent hook head needs the latch of every decision"
    );
    for w in windows {
        assert_eq!(w.v_init.len(), n_neurons, "policy_forward: v_init length");
        assert!(
            !two_view || w.v_init_masked.len() == n_neurons,
            "policy_forward: v_init_masked length"
        );
    }
    let encoded: Vec<Encoded> = (0..if two_view { 2 * n } else { n })
        .into_par_iter()
        .map(|lane| {
            let w = &windows[lane % n];
            if lane < n {
                encode_window(net.encoder, net.encoder_params, &w.observations)
            } else {
                let masked: Vec<Observation> = w.observations.iter().map(mask_own_hook).collect();
                encode_window(net.encoder, net.encoder_params, &masked)
            }
        })
        .collect();
    let inputs: Vec<BatchedSeqInput<'_>> = encoded
        .iter()
        .enumerate()
        .map(|(lane, enc)| BatchedSeqInput {
            v_init: if lane < n {
                &windows[lane].v_init
            } else {
                &windows[lane - n].v_init_masked
            },
            inputs: &enc.inputs,
        })
        .collect();
    if inputs.is_empty() {
        return Ok(PolicyForward {
            two_view,
            encoded,
            n_windows: 0,
            logits: Vec::new(),
        });
    }
    engine.forward(
        net.model,
        &inputs,
        &BatchedForwardOptions {
            memory_cap_bytes,
            segment_decisions: None,
            type_means: false,
            no_grad_decisions: 0,
        },
    )?;
    let engine_ref: &BatchedEngine = engine;
    let clip_at = net.decoder.config().z_clip;
    let lane_logits: Vec<Vec<HeadLogits>> = (0..inputs.len())
        .into_par_iter()
        .map(|lane| {
            let t_len = encoded[lane].inputs.len();
            let mut dn = vec![0.0f32; net.model.num_outputs()];
            let mut z = vec![0.0f32; net.model.num_outputs()];
            (0..t_len)
                .map(|t| {
                    engine_ref.dn_rates(lane, t, &mut dn);
                    net.calib.z_into(&dn, clip_at, &mut z);
                    decoder_logits(net.decoder, &z, net.decoder_params, false)
                })
                .collect()
        })
        .collect();
    let logits = (0..n)
        .map(|b| {
            if two_view {
                lane_logits[b]
                    .iter()
                    .zip(&lane_logits[n + b])
                    .map(|(f, m)| combine_hook_view(f, m))
                    .collect()
            } else {
                lane_logits[b].clone()
            }
        })
        .collect();
    Ok(PolicyForward {
        two_view,
        encoded,
        n_windows: n,
        logits,
    })
}

fn zero_logits(l: &HeadLogits) -> bool {
    *l == HeadLogits::default()
}

/// The backward pass for `d_logits[window][decision]` = `dL/d(played logits)` (the hook component is the masked view's under two views).
/// `engine` must still hold the recording of the [`policy_forward`] that produced `fwd`, with the parameters unchanged.
pub fn policy_backward(
    net: PolicyNet<'_>,
    engine: &mut BatchedEngine,
    fwd: &PolicyForward,
    d_logits: &[Vec<HeadLogits>],
) -> PolicyBackward {
    let n = fwd.n_windows;
    assert_eq!(d_logits.len(), n, "policy_backward: one gradient list per window");
    if n == 0 {
        return PolicyBackward {
            fly: ParamGradients::zeros_like(net.model.params()),
            encoder: Vec::new(),
            decoder: Vec::new(),
        };
    }
    let lanes = fwd.encoded.len();
    let hook_only = HeadMask {
        hook: true,
        ..HeadMask::NONE
    };
    let all_but_hook = HeadMask {
        hook: false,
        ..HeadMask::ALL
    };
    let engine_ref: &BatchedEngine = engine;
    // Per lane: the decoder gradients and dL/d(dn_rates) of every decision.
    let scored: Vec<(DecoderGradients, Vec<Vec<f32>>)> = (0..lanes)
        .into_par_iter()
        .map(|lane| {
            let b = lane % n;
            let t_len = fwd.encoded[lane].inputs.len();
            assert_eq!(d_logits[b].len(), t_len, "policy_backward: a gradient per decision");
            let (mask, masked_lane) = if !fwd.two_view {
                (HeadMask::ALL, false)
            } else if lane < n {
                (all_but_hook, false)
            } else {
                (hook_only, true)
            };
            let mut dn = vec![0.0f32; net.model.num_outputs()];
            let mut grads = net.decoder.zeros_gradients();
            let mut grad_dn = Vec::with_capacity(t_len);
            for (t, d) in d_logits[b].iter().enumerate() {
                let d = if fwd.two_view {
                    if masked_lane {
                        HeadLogits {
                            hook: d.hook,
                            ..HeadLogits::default()
                        }
                    } else {
                        HeadLogits { hook: 0.0, ..*d }
                    }
                } else {
                    *d
                };
                if zero_logits(&d) {
                    grad_dn.push(vec![0.0f32; net.model.num_outputs()]);
                    continue;
                }
                engine_ref.dn_rates(lane, t, &mut dn);
                let (g, gdn) =
                    decoder_logits_backward(net.decoder, &dn, net.calib, net.decoder_params, &d, &mask, false);
                crate::brain_bc::add_decoder_gradients(&mut grads, &g);
                grad_dn.push(gdn);
            }
            (grads, grad_dn)
        })
        .collect();
    let seq_grads: Vec<BatchedSeqGrad<'_>> = scored
        .iter()
        .map(|(_, gdn)| BatchedSeqGrad {
            grad_dn: gdn,
            extra_taps: &[],
        })
        .collect();
    let bptt = engine.backward(net.model, &seq_grads, false);
    let per_lane_enc: Vec<EncoderGradients> = fwd
        .encoded
        .par_iter()
        .zip(bptt.grad_inputs.par_iter())
        .map(|(enc, gin)| {
            let mut g = net.encoder.zero_grads();
            for ((features, proprio), grad_input) in enc.features.iter().zip(&enc.proprio).zip(gin) {
                net.encoder
                    .backward_with_params(features, proprio, net.encoder_params, grad_input, &mut g);
            }
            g
        })
        .collect();
    // Sum the views of a window in a fixed order (full, then masked).
    let mut encoder = Vec::with_capacity(n);
    let mut decoder = Vec::with_capacity(n);
    for b in 0..n {
        let mut e = per_lane_enc[b].clone();
        let mut d = scored[b].0.clone();
        if fwd.two_view {
            add_vec(&mut e.g, &per_lane_enc[n + b].g);
            add_vec(&mut e.c, &per_lane_enc[n + b].c);
            add_vec(&mut e.bin_gain, &per_lane_enc[n + b].bin_gain);
            crate::brain_bc::add_decoder_gradients(&mut d, &scored[n + b].0);
        }
        encoder.push(e);
        decoder.push(d);
    }
    PolicyBackward {
        fly: bptt.grad,
        encoder,
        decoder,
    }
}

fn add_vec(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}
