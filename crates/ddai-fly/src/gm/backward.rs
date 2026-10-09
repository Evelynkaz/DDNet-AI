//! Hand-written truncated BPTT for the `Gm` neuron model: the exact transpose of
//! [`GmModel::step_once`] over a window recorded by [`GmRecorder`] (the layout and the equations
//! are in the module docs of [`crate::gm`]). The transposed CSR carries the message gradient back
//! (`dH[u] += sum_v W[v,u] dM[v]`); the per-neuron gradients of the shared `f_psi` are accumulated
//! straight into the parameter-shaped output. Verified against central finite differences of an
//! independent dense f64 implementation in `gm/tests.rs`.

use super::{GmModel, GmParams, GmRecorder, GmUpdate, softsign_grad_from_out};

/// Everything [`gm_backward`] produces.
#[derive(Debug, Clone)]
pub struct GmBackwardOut {
    pub grads: GmParams,
    /// `dL/d(input current)` per decision and afferent (the encoder's backward pass takes these).
    pub grad_inputs: Vec<Vec<f32>>,
}

/// Reusable buffers (sized once from the model; `gm_backward` allocates only its return value).
#[derive(Debug, Clone)]
pub struct GmScratch {
    dh: Vec<f32>,
    dhi: Vec<f32>,
    dm: Vec<f32>,
}

impl GmScratch {
    pub fn new(model: &GmModel) -> Self {
        let len = model.state_len();
        GmScratch {
            dh: vec![0.0; len],
            dhi: vec![0.0; len],
            dm: vec![0.0; len],
        }
    }
}

/// Backward over the `t_decisions` decisions in `rec`. `grad_dn_rates[t]` is `dL/d(dn_rates)` of
/// decision `t` (`num_outputs` long). Gradients do not flow into the window's initial state
/// (truncated BPTT, like the rate model's default).
pub fn gm_backward(
    model: &GmModel,
    rec: &GmRecorder,
    t_decisions: usize,
    grad_dn_rates: &[&[f32]],
    scratch: &mut GmScratch,
) -> GmBackwardOut {
    assert_eq!(
        rec.decisions(),
        t_decisions,
        "gm_backward: recorder holds a different number of decisions"
    );
    assert_eq!(
        grad_dn_rates.len(),
        t_decisions,
        "gm_backward: grad_dn_rates.len() must equal t_decisions"
    );
    for g in grad_dn_rates {
        assert_eq!(
            g.len(),
            model.num_outputs(),
            "gm_backward: grad_dn_rates[t].len() mismatch"
        );
    }
    let mut grads = GmParams::zeros(&model.shape);
    let mut grad_inputs = vec![vec![0.0f32; model.aff.len()]; t_decisions];
    scratch.dh.fill(0.0);
    let k_steps = model.config.steps as usize;
    let gated = model.config.update == GmUpdate::Gated;
    macro_rules! go {
        ($d:literal, $hd:literal, $o:literal) => {
            model.backward_impl::<$d, $hd, $o>(
                rec,
                t_decisions,
                k_steps,
                grad_dn_rates,
                scratch,
                &mut grads,
                &mut grad_inputs,
            )
        };
    }
    match (model.shape.d, model.shape.hd, gated) {
        (8, 8, false) => go!(8, 8, 8),
        (8, 8, true) => go!(8, 8, 16),
        (8, 16, false) => go!(8, 16, 8),
        (8, 16, true) => go!(8, 16, 16),
        (8, 32, false) => go!(8, 32, 8),
        (8, 32, true) => go!(8, 32, 16),
        (16, 8, false) => go!(16, 8, 16),
        (16, 8, true) => go!(16, 8, 32),
        (16, 16, false) => go!(16, 16, 16),
        (16, 16, true) => go!(16, 16, 32),
        (16, 32, false) => go!(16, 32, 16),
        (16, 32, true) => go!(16, 32, 32),
        _ => unreachable!("GmConfig::validate admits only d in {{8,16}}, hidden in {{8,16,32}}"),
    }
    GmBackwardOut { grads, grad_inputs }
}

impl GmModel {
    #[allow(clippy::too_many_arguments)]
    fn backward_impl<const D: usize, const HD: usize, const O: usize>(
        &self,
        rec: &GmRecorder,
        t_decisions: usize,
        k_steps: usize,
        grad_dn: &[&[f32]],
        scratch: &mut GmScratch,
        g: &mut GmParams,
        grad_inputs: &mut [Vec<f32>],
    ) {
        let gated = O != D;
        let p = &self.params;
        let n = self.n;
        for t in (0..t_decisions).rev() {
            // The readout of decision `t` taps the state after its last step.
            let dn_h = rec.decision_dn_h(t);
            for (k, &i) in self.dn.iter().enumerate() {
                let ds = grad_dn[t][k];
                if ds == 0.0 {
                    continue;
                }
                let slot = self.ro_slot[k] as usize;
                let i = i as usize;
                g.ro_b[slot] += ds;
                for d in 0..D {
                    g.ro_w[slot * D + d] += ds * dn_h[k * D + d];
                    scratch.dh[i * D + d] += ds * p.ro_w[slot * D + d];
                }
            }
            for s in (0..k_steps).rev() {
                let view = rec.step_views(t * k_steps + s);
                // --- the update network, neuron by neuron ---------------------------------
                for v in 0..n {
                    let dh: &[f32; D] = scratch.dh[v * D..(v + 1) * D].try_into().expect("dh row");
                    let a: &[f32; O] = view.a[v * O..(v + 1) * O].try_into().expect("a row");
                    let mut dout = [0.0f32; O];
                    let dhi_row: &mut [f32; D] = (&mut scratch.dhi[v * D..(v + 1) * D]).try_into().expect("dhi row");
                    if gated {
                        let hi: &[f32; D] = view.h_inj[v * D..(v + 1) * D].try_into().expect("h row");
                        for d in 0..D {
                            let za = a[d];
                            let ca = a[D + d];
                            let z = 0.5 + 0.5 * za;
                            dhi_row[d] = (1.0 - z) * dh[d];
                            dout[d] = dh[d] * (ca - hi[d]) * 0.5 * softsign_grad_from_out(za);
                            dout[D + d] = dh[d] * z * softsign_grad_from_out(ca);
                        }
                    } else {
                        for d in 0..D {
                            dout[d] = dh[d] * softsign_grad_from_out(a[d]);
                            dhi_row[d] = 0.0;
                        }
                    }
                    let h1: &[f32; HD] = view.h1[v * HD..(v + 1) * HD].try_into().expect("h1 row");
                    for o in 0..O {
                        g.b2[o] += dout[o];
                    }
                    let mut dpre1 = [0.0f32; HD];
                    for j in 0..HD {
                        let hj = h1[j];
                        let w2row: &[f32; O] = p.w2[j * O..(j + 1) * O].try_into().expect("w2 row");
                        let gw2: &mut [f32; O] = (&mut g.w2[j * O..(j + 1) * O]).try_into().expect("gw2 row");
                        let mut acc = 0.0f32;
                        for o in 0..O {
                            gw2[o] += hj * dout[o];
                            acc += w2row[o] * dout[o];
                        }
                        dpre1[j] = if hj > 0.0 { acc } else { 0.0 };
                    }
                    for j in 0..HD {
                        g.b1[j] += dpre1[j];
                    }
                    let key = self.key_of[v] as usize;
                    let eta: &[f32; D] = p.eta[key * D..(key + 1) * D].try_into().expect("eta row");
                    let m: &[f32; D] = view.m[v * D..(v + 1) * D].try_into().expect("m row");
                    let mut dm = [0.0f32; D];
                    let mut deta = [0.0f32; D];
                    for k in 0..D {
                        let w1m: &[f32; HD] = p.w1m[k * HD..(k + 1) * HD].try_into().expect("w1m row");
                        let w1e: &[f32; HD] = p.w1e[k * HD..(k + 1) * HD].try_into().expect("w1e row");
                        let gm: &mut [f32; HD] = (&mut g.w1m[k * HD..(k + 1) * HD]).try_into().expect("gw1m row");
                        let mut am = 0.0f32;
                        let mut ae = 0.0f32;
                        for j in 0..HD {
                            gm[j] += m[k] * dpre1[j];
                            am += w1m[j] * dpre1[j];
                            ae += w1e[j] * dpre1[j];
                        }
                        let ge: &mut [f32; HD] = (&mut g.w1e[k * HD..(k + 1) * HD]).try_into().expect("gw1e row");
                        for j in 0..HD {
                            ge[j] += eta[k] * dpre1[j];
                        }
                        dm[k] = am;
                        deta[k] = ae;
                    }
                    scratch.dm[v * D..(v + 1) * D].copy_from_slice(&dm);
                    for k in 0..D {
                        g.eta[key * D + k] += deta[k];
                    }
                }
                // --- the message operator, transposed: dH[u] += sum_v W[v,u] dM[v] -----------
                for u in 0..n {
                    let (a, b) = (self.t_row_start[u] as usize, self.t_row_start[u + 1] as usize);
                    let mut acc = [0.0f32; D];
                    for e in a..b {
                        let v = self.t_post[e] as usize;
                        let w = self.t_w[e];
                        let dmv: &[f32; D] = scratch.dm[v * D..(v + 1) * D].try_into().expect("dm row");
                        for d in 0..D {
                            acc[d] += w * dmv[d];
                        }
                    }
                    let dhi_row: &mut [f32; D] = (&mut scratch.dhi[u * D..(u + 1) * D]).try_into().expect("dhi row");
                    for d in 0..D {
                        dhi_row[d] += acc[d];
                    }
                }
                // --- the injection, backward (afferents); everyone else passes through -------
                scratch.dh.copy_from_slice(&scratch.dhi);
                let inputs = rec.decision_inputs(t);
                for (k, &i) in self.aff.iter().enumerate() {
                    let i = i as usize;
                    let slot = self.aff_slot[k] as usize;
                    let y: &[f32; D] = view.h_inj[i * D..(i + 1) * D].try_into().expect("h row");
                    let hp: &[f32; D] = view.h_pre[i * D..(i + 1) * D].try_into().expect("h row");
                    let dhi_row: &[f32; D] = scratch.dhi[i * D..(i + 1) * D].try_into().expect("dhi row");
                    let mut dpre = [0.0f32; D];
                    for d in 0..D {
                        dpre[d] = dhi_row[d] * softsign_grad_from_out(y[d]);
                    }
                    let e = inputs[k];
                    let mut de = 0.0f32;
                    for d in 0..D {
                        g.bg[d] += dpre[d];
                        g.inj[slot * D + d] += e * dpre[d];
                        de += p.inj[slot * D + d] * dpre[d];
                    }
                    grad_inputs[t][k] += de;
                    let dh_row: &mut [f32; D] = (&mut scratch.dh[i * D..(i + 1) * D]).try_into().expect("dh row");
                    for kk in 0..D {
                        let wrow: &[f32; D] = p.wh[kk * D..(kk + 1) * D].try_into().expect("wh row");
                        let gw: &mut [f32; D] = (&mut g.wh[kk * D..(kk + 1) * D]).try_into().expect("gwh row");
                        let mut acc = 0.0f32;
                        for d in 0..D {
                            gw[d] += hp[kk] * dpre[d];
                            acc += wrow[d] * dpre[d];
                        }
                        dh_row[kk] = acc;
                    }
                }
            }
        }
    }
}
