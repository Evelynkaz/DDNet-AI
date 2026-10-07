//! The wide hook readout (task 8.7, bundle format v5): what the hook head reads from the network.
//!
//! The fly's hook head used to read twelve numbers: the mean calibrated DN `z` of each of eleven hook type groups, through a linear map of
//! twelve parameters ([`crate::decoder`]). Task 8.6 measured that the information the planner's press decision depends on is intact in the
//! network's state (an MLP on the membrane state or on the DN `z` of four frames reaches AUROC 0.81 to 0.82) and is lost in this readout
//! (twelve parameters, optimally fitted: 0.67, after BC: 0.51 to 0.56). A [`HookReadout`] other than [`HookReadout::Pooled`] widens it:
//!
//! * [`HookReadout::LinearDn`]: a linear map of **all** calibrated DN `z` slots (FLY.md section 1 point 1: still a linear DN decoder);
//! * [`HookReadout::MlpDn`]: a one-hidden-layer MLP over all DN `z` slots (a **departure** from "narrow linear decoder", FLY.md section 1
//!   point 1: it is a labelled kind, not "the fly" until the owner decides).
//!
//! * [`HookReadout::EncoderMlp`]: **the control of FLY.md section 1 point 3, not a fly**: the same MLP on the encoder's input vector (what the
//!   connectome receives), with no connectome and no pooled head in the hook logit. It exists to say how much of a wide readout's gain is the
//!   network's and how much any MLP's; it is never a candidate for playing.
//!
//! A readout never reads the membrane state `V` or any non-DN neuron: `V` includes the input neurons, so such a readout would bypass the
//! connectome (FLY.md sections 1 and 6; the 8.6 review). The `V` probes of 8.6 stay an offline diagnostic.
//!
//! The wide part is a **residual** on the pooled head: `hook logit = pooled(z) + wide(x)`. Its output weights start at zero, so a fly upgraded
//! with a wide readout plays bit for bit like the original until training moves them (the identity discipline of 8.5a and 8.6), and
//! `LinearDn` is exactly a linear map of all 100 slots (the pooled head is a linear map of group means of the same slots).
//!
//! The input of the wide part is the calibrated DN `z` itself (already standardised per slot by the frozen, scene-fitted
//! [`crate::decoder::DnCalibration`], clipped to +-10). Nothing here sees the hook latch: the wide part is a function of the DN state only.
//!
//! Everything on the playing path is allocation-free: the caller supplies the scratch buffers ([`crate::decoder::DecoderScratch`]).

use serde::{Deserialize, Serialize};

use crate::decoder::DecoderError;
use crate::model::FlyModel;
use crate::rng::SplitMix64;

/// How the hook head reads the network (bundle format v5). `Pooled` is the head of every bundle written before 8.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HookReadout {
    #[default]
    Pooled,
    LinearDn,
    MlpDn {
        hidden: u32,
    },
    /// The control: an MLP on the encoder input, bypassing the connectome (see the module docs).
    EncoderMlp {
        hidden: u32,
    },
}

impl HookReadout {
    /// Whether the readout reads the encoder's input vector instead of the DN state (the control).
    pub fn reads_encoder(&self) -> bool {
        matches!(self, HookReadout::EncoderMlp { .. })
    }

    /// A short name for logs and tables (`pooled`, `linear-dn`, `mlp-dn-32`, `mlp-enc-32`).
    pub fn label(&self) -> String {
        match *self {
            HookReadout::Pooled => "pooled".into(),
            HookReadout::LinearDn => "linear-dn".into(),
            HookReadout::MlpDn { hidden } => format!("mlp-dn-{hidden}"),
            HookReadout::EncoderMlp { hidden } => format!("mlp-enc-{hidden}"),
        }
    }

    /// Parses [`HookReadout::label`]'s spelling.
    pub fn parse(s: &str) -> Result<Self, String> {
        let bad = || format!("unknown hook readout {s:?} (pooled, linear-dn, mlp-dn-<H>, mlp-enc-<H>)");
        match s {
            "pooled" => Ok(HookReadout::Pooled),
            "linear-dn" => Ok(HookReadout::LinearDn),
            _ => {
                let (enc, h) = match (s.strip_prefix("mlp-dn-"), s.strip_prefix("mlp-enc-")) {
                    (Some(h), _) => (false, h),
                    (_, Some(h)) => (true, h),
                    _ => return Err(bad()),
                };
                let hidden = h.parse::<u32>().ok().filter(|&h| h > 0).ok_or_else(bad)?;
                Ok(if enc {
                    HookReadout::EncoderMlp { hidden }
                } else {
                    HookReadout::MlpDn { hidden }
                })
            }
        }
    }
}

/// The wide part's parameters: for an MLP `w1` is `hidden x n_in` row-major, `b1` and `w2` have `hidden` entries; for `LinearDn` there is
/// no hidden layer: `w1`/`b1` are empty and `w2` has `n_in` entries (the weights on the DN `z`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookWide {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
}

/// `dL/d` of the learned weights, same shapes as [`HookWide`] (empty when the decoder has no wide readout).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HookWideGrads {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
}

impl HookWideGrads {
    /// `self += o`.
    pub fn add(&mut self, o: &HookWideGrads) {
        for (a, b) in self
            .w1
            .iter_mut()
            .chain(&mut self.b1)
            .chain(&mut self.w2)
            .zip(o.w1.iter().chain(&o.b1).chain(&o.w2))
        {
            *a += b;
        }
    }
}

/// The structure of a decoder's wide readout, resolved against a graph: which kind, which neurons it reads, the layer sizes.
#[derive(Debug, Clone)]
pub struct HookWideModel {
    kind: HookReadout,
    n_in: usize,
    hidden: usize,
}

impl HookWideModel {
    /// The structure of `kind` on `model`'s graph (it reads every DN slot); `None` for [`HookReadout::Pooled`].
    pub fn resolve(kind: HookReadout, model: &FlyModel) -> Result<Option<Self>, DecoderError> {
        let n_in = if kind.reads_encoder() {
            model.num_inputs()
        } else {
            model.num_outputs()
        };
        let hidden = match kind {
            HookReadout::Pooled => return Ok(None),
            HookReadout::LinearDn => 0,
            HookReadout::MlpDn { hidden } | HookReadout::EncoderMlp { hidden } => {
                if hidden == 0 {
                    return Err(DecoderError::InvalidConfig(
                        "a hook readout MLP needs hidden > 0".into(),
                    ));
                }
                hidden as usize
            }
        };
        if n_in == 0 {
            return Err(DecoderError::InvalidConfig("the hook readout reads no DN".into()));
        }
        Ok(Some(HookWideModel { kind, n_in, hidden }))
    }

    pub fn kind(&self) -> HookReadout {
        self.kind
    }
    /// Whether this is the encoder-input control.
    pub fn reads_encoder(&self) -> bool {
        self.kind.reads_encoder()
    }
    /// The vector the readout reads: the calibrated DN `z`, or (the control) the encoder's input vector `enc`.
    pub fn input<'a>(&self, z: &'a [f32], enc: &'a [f32]) -> &'a [f32] {
        if self.reads_encoder() {
            assert_eq!(
                enc.len(),
                self.n_in,
                "the control readout needs the encoder input ({} numbers)",
                self.n_in
            );
            enc
        } else {
            z
        }
    }
    /// Number of input features.
    pub fn n_in(&self) -> usize {
        self.n_in
    }
    /// Hidden units (`0` for the linear readout).
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    /// Lengths of `(w1, b1, w2)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        if self.hidden == 0 {
            (0, 0, self.n_in)
        } else {
            (self.hidden * self.n_in, self.hidden, self.hidden)
        }
    }
    /// Number of learned parameters.
    pub fn num_params(&self) -> usize {
        let (a, b, c) = self.shape();
        a + b + c
    }
    pub fn zero_grads(&self) -> HookWideGrads {
        let (a, b, c) = self.shape();
        HookWideGrads {
            w1: vec![0.0; a],
            b1: vec![0.0; b],
            w2: vec![0.0; c],
        }
    }

    /// Initial parameters: small Gaussian first layer (`N(0, 2 / n_in)`, deterministic in `seed`), and **zero output
    /// weights**, so the readout adds exactly nothing until trained.
    pub fn init_params(&self, seed: u64) -> HookWide {
        let (n1, n2, n3) = self.shape();
        let mut rng = SplitMix64::new(seed ^ 0x8007_0001);
        let scale = (2.0 / self.n_in as f32).sqrt();
        HookWide {
            w1: (0..n1).map(|_| rng.next_gaussian() * scale).collect(),
            b1: vec![0.0; n2],
            w2: vec![0.0; n3],
        }
    }

    /// Shapes and finiteness of saved parameters.
    pub fn validate(&self, p: &HookWide) -> Result<(), DecoderError> {
        let (n1, n2, n3) = self.shape();
        for (name, got, want) in [
            ("hook_wide.w1", p.w1.len(), n1),
            ("hook_wide.b1", p.b1.len(), n2),
            ("hook_wide.w2", p.w2.len(), n3),
        ] {
            if got != want {
                return Err(DecoderError::ParamShapeMismatch(format!(
                    "{name}.len() == {got}, expected {want}"
                )));
            }
        }
        if !p.w1.iter().chain(&p.b1).chain(&p.w2).all(|x| x.is_finite()) {
            return Err(DecoderError::NonFiniteParam("hook_wide has a non-finite value".into()));
        }
        Ok(())
    }

    /// The wide part's logit for the calibrated DN `z` (one entry per DN slot); `h` (`hidden` long, may be empty for the linear readout)
    /// receives the hidden activations.
    pub fn forward(&self, p: &HookWide, z: &[f32], h: &mut [f32]) -> f32 {
        let n = self.n_in;
        assert_eq!(z.len(), n, "the hook readout reads {n} features, got {}", z.len());
        assert_eq!(
            h.len(),
            self.hidden,
            "the hidden-activation scratch must be `hidden` long"
        );
        if self.hidden == 0 {
            return p.w2.iter().zip(z).map(|(&w, &zi)| w * zi).sum();
        }
        let mut out = 0.0f32;
        for (j, (hj, (&b, &w2))) in h.iter_mut().zip(p.b1.iter().zip(&p.w2)).enumerate() {
            let row = &p.w1[j * n..(j + 1) * n];
            let a = b + row.iter().zip(z).map(|(&w, &zi)| w * zi).sum::<f32>();
            let act = a.max(0.0);
            *hj = act;
            out += w2 * act;
        }
        out
    }

    /// [`HookWideModel::forward`] without the hidden activations: the same logit, **no scratch and no allocation** (the forward-only paths that
    /// have no scratch to hand, such as `FlyBrain::forward_logits`).
    pub fn logit(&self, p: &HookWide, z: &[f32]) -> f32 {
        let n = self.n_in;
        assert_eq!(z.len(), n, "the hook readout reads {n} features, got {}", z.len());
        if self.hidden == 0 {
            return p.w2.iter().zip(z).map(|(&w, &zi)| w * zi).sum();
        }
        let mut out = 0.0f32;
        for (j, (&b, &w2)) in p.b1.iter().zip(&p.w2).enumerate() {
            let row = &p.w1[j * n..(j + 1) * n];
            let a = b + row.iter().zip(z).map(|(&w, &zi)| w * zi).sum::<f32>();
            out += w2 * a.max(0.0);
        }
        out
    }

    /// Backward of [`HookWideModel::forward`]: adds `d_logit * dlogit/dparam` to `g` and `d_logit * dlogit/dz` to `grad_z`. `h` must be the
    /// activations the forward pass left for the same `z`. The ReLU's kink has subgradient `0` (the crate's convention).
    pub fn backward(
        &self,
        p: &HookWide,
        z: &[f32],
        h: &[f32],
        d_logit: f32,
        g: &mut HookWideGrads,
        grad_z: &mut [f32],
    ) {
        let n = self.n_in;
        assert_eq!(z.len(), n, "the hook readout reads {n} features, got {}", z.len());
        assert_eq!(grad_z.len(), n);
        assert_eq!(h.len(), self.hidden, "the hidden activations must be `hidden` long");
        if self.hidden == 0 {
            for i in 0..n {
                g.w2[i] += d_logit * z[i];
                grad_z[i] += d_logit * p.w2[i];
            }
            return;
        }
        for (j, (&hj, &w2)) in h.iter().zip(&p.w2).enumerate() {
            g.w2[j] += d_logit * hj;
            if hj <= 0.0 {
                continue;
            }
            let da = d_logit * w2;
            g.b1[j] += da;
            let row = &p.w1[j * n..(j + 1) * n];
            let grow = &mut g.w1[j * n..(j + 1) * n];
            for i in 0..n {
                grow[i] += da * z[i];
                grad_z[i] += da * row[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_of(kind: HookReadout, n_in: usize) -> HookWideModel {
        let hidden = match kind {
            HookReadout::MlpDn { hidden } | HookReadout::EncoderMlp { hidden } => hidden as usize,
            _ => 0,
        };
        HookWideModel { kind, n_in, hidden }
    }

    #[test]
    fn labels_round_trip() {
        for k in [
            HookReadout::Pooled,
            HookReadout::LinearDn,
            HookReadout::MlpDn { hidden: 32 },
        ] {
            assert_eq!(HookReadout::parse(&k.label()).unwrap(), k, "{}", k.label());
        }
        for bad in ["", "mlp-dn-0", "mlp-dn-x", "mlp-v-dn-32", "wide"] {
            assert!(HookReadout::parse(bad).is_err(), "{bad:?}");
        }
    }

    fn perturbed(m: &HookWideModel) -> HookWide {
        let mut p = m.init_params(3);
        let mut rng = SplitMix64::new(11);
        for w in p.w2.iter_mut().chain(&mut p.b1) {
            *w = rng.next_gaussian() * 0.5;
        }
        p
    }

    #[test]
    fn zero_output_weights_add_nothing_and_the_gradient_matches_finite_differences() {
        for kind in [HookReadout::LinearDn, HookReadout::MlpDn { hidden: 5 }] {
            let m = model_of(kind, 7);
            let x: Vec<f32> = (0..7).map(|i| (i as f32 * 0.83).sin() * 2.0).collect();
            let mut h = vec![0.0; m.hidden()];
            assert_eq!(m.forward(&m.init_params(1), &x, &mut h), 0.0, "{kind:?}");

            let p = perturbed(&m);
            let logit = |p: &HookWide, x: &[f32]| m.forward(p, x, &mut vec![0.0; m.hidden()]);
            let mut g = m.zero_grads();
            let mut gx = vec![0.0; 7];
            let _ = m.forward(&p, &x, &mut h);
            m.backward(&p, &x, &h, 1.7, &mut g, &mut gx);
            let eps = 1e-3f32;
            for i in 0..7 {
                let (mut a, mut b) = (x.clone(), x.clone());
                a[i] += eps;
                b[i] -= eps;
                let num = 1.7 * (logit(&p, &a) - logit(&p, &b)) / (2.0 * eps);
                assert!(
                    (gx[i] - num).abs() < 2e-3 * (1.0 + num.abs()),
                    "{kind:?} gx[{i}] {} vs {num}",
                    gx[i]
                );
            }
            let flat = |p: &HookWide| -> Vec<f32> { p.w1.iter().chain(&p.b1).chain(&p.w2).copied().collect() };
            let analytic: Vec<f32> = g.w1.iter().chain(&g.b1).chain(&g.w2).copied().collect();
            assert_eq!(analytic.len(), flat(&p).len());
            for (k, &a) in analytic.iter().enumerate() {
                let bump = |p: &HookWide, d: f32| {
                    let mut q = p.clone();
                    let (n1, n2) = (q.w1.len(), q.b1.len());
                    if k < n1 {
                        q.w1[k] += d;
                    } else if k < n1 + n2 {
                        q.b1[k - n1] += d;
                    } else {
                        q.w2[k - n1 - n2] += d;
                    }
                    q
                };
                let num = 1.7 * (logit(&bump(&p, eps), &x) - logit(&bump(&p, -eps), &x)) / (2.0 * eps);
                assert!(
                    (a - num).abs() < 2e-3 * (1.0 + num.abs()),
                    "{kind:?} param {k}: {a} vs {num}"
                );
            }
        }
    }

    #[test]
    fn logit_equals_forward_and_wrong_lengths_panic() {
        for kind in [HookReadout::LinearDn, HookReadout::MlpDn { hidden: 5 }] {
            let m = model_of(kind, 7);
            let p = perturbed(&m);
            let x: Vec<f32> = (0..7).map(|i| (i as f32 * 0.61).cos() * 3.0).collect();
            let mut h = vec![0.0; m.hidden()];
            assert_eq!(
                m.logit(&p, &x).to_bits(),
                m.forward(&p, &x, &mut h).to_bits(),
                "{kind:?}"
            );
            assert!(
                std::panic::catch_unwind(|| m.logit(&p, &x[..6])).is_err(),
                "a short input must not be silently truncated"
            );
        }
        let m = model_of(HookReadout::MlpDn { hidden: 5 }, 7);
        let p = perturbed(&m);
        assert!(
            std::panic::catch_unwind(|| m.forward(&p, &[0.0; 7], &mut [0.0; 4])).is_err(),
            "a short scratch must not truncate"
        );
    }

    #[test]
    fn validate_refuses_wrong_shapes_and_nan() {
        let m = model_of(HookReadout::MlpDn { hidden: 4 }, 6);
        let p = m.init_params(2);
        assert!(m.validate(&p).is_ok());
        let mut bad = p.clone();
        bad.w1.pop();
        assert!(m.validate(&bad).is_err());
        let mut bad = p.clone();
        bad.w2[0] = f32::NAN;
        assert!(m.validate(&bad).is_err());
    }
}
