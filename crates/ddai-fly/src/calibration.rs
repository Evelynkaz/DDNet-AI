//! Data-driven DN calibration (task 8.2).
//!
//! [`crate::decoder::calibrate_from_rest`] measures each DN's mean and spread from a near-empty
//! scene, where every spread sits at the `min_sigma` floor (its own doc comment says so). With
//! `r_max = 10` and `min_sigma = 0.05`, a DN whose rate moves by 0.5 (5% of its range) is already
//! at the `|z| <= 10` clip, so on real scenes the heads read saturated, gradient-free `z`. This
//! module calibrates instead from the rates the DNs actually take while the fly *plays*: it runs
//! the encoder + fly over recorded windows of observations and takes the mean and standard
//! deviation over all scored decisions. The result is still a frozen per-slot `(mu, sigma)` — the
//! decoder contract is unchanged — but `z` now spans a useful range.
//!
//! **Mirror symmetry.** Recorded scenes are not mirror-symmetric, so measured `L`/`R` homologs
//! would differ slightly, which would break the decoder's exact action-level mirror guarantee
//! (the failure mode `calibrate_from_rest`'s F19 note documents). [`symmetrize_calibration`]
//! therefore averages `mu` and `sigma` over every `L`/`R` homolog pair (same type, paired by
//! ascending dense neuron index, the rule the aim head uses), which makes `z` mirror-equivariant
//! by construction.

use ddai_brain::Observation;
use ddai_flyg::Side;

use crate::decoder::{DecoderError, DnCalibration};
use crate::encoder::{EncoderModel, EncoderParams, RayGridFeatures, compute_proprioception_values};
use crate::model::FlyModel;
use crate::state::FlyState;

/// Averages `mu` and `sigma` across `L`/`R` homolog DN pairs (see the module doc comment).
pub fn symmetrize_calibration(model: &FlyModel, calib: &mut DnCalibration) {
    let flyg = model.flyg();
    // (type_index) -> (left slots, right slots), each sorted by ascending dense index (the output
    // slot order already is ascending dense index).
    let mut by_type: std::collections::BTreeMap<u32, (Vec<usize>, Vec<usize>)> = std::collections::BTreeMap::new();
    for (slot, &dense) in model.output_neuron_indices().iter().enumerate() {
        let n = &flyg.neurons[dense as usize];
        let e = by_type.entry(n.type_index).or_default();
        match n.side {
            Side::L => e.0.push(slot),
            Side::R => e.1.push(slot),
            _ => {}
        }
    }
    for (left, right) in by_type.values() {
        for (&l, &r) in left.iter().zip(right) {
            let mu = 0.5 * (calib.mu[l] + calib.mu[r]);
            let sigma = 0.5 * (calib.sigma[l] + calib.sigma[r]);
            calib.mu[l] = mu;
            calib.mu[r] = mu;
            calib.sigma[l] = sigma;
            calib.sigma[r] = sigma;
        }
    }
}

/// Calibrates from `windows` of observations: each window starts from `v_init` (the resting
/// state), the first `burn_in` decisions are run but not counted, and the DN rates of the rest
/// enter the statistics. The result is symmetrised and `sigma` floored at `min_sigma`.
pub fn calibrate_from_windows(
    model: &FlyModel,
    encoder: &EncoderModel,
    encoder_params: &EncoderParams,
    v_init: &[f32],
    windows: &[Vec<Observation>],
    burn_in: usize,
    min_sigma: f32,
) -> Result<DnCalibration, DecoderError> {
    let n_out = model.num_outputs();
    let mut sum = vec![0.0f64; n_out];
    let mut sum_sq = vec![0.0f64; n_out];
    let mut count = 0u64;
    let mut features = RayGridFeatures::new(encoder.ray_grid_config());
    let mut input_buf = vec![0.0f32; encoder.num_inputs()];
    for window in windows {
        let mut state = FlyState::new(model);
        state.set_v(model, v_init);
        for (t, obs) in window.iter().enumerate() {
            let an = compute_proprioception_values(&obs.self_state, encoder.ray_grid_config());
            features.compute(obs, encoder.ray_grid_config());
            encoder.forward(&features, &an, encoder_params, &mut input_buf);
            let out = state.step_decision(model, &input_buf);
            if t < burn_in {
                continue;
            }
            for (i, &r) in out.dn_rates.iter().enumerate() {
                let r = f64::from(r);
                sum[i] += r;
                sum_sq[i] += r * r;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(DecoderError::InvalidConfig(
            "calibrate_from_windows needs at least one scored decision".to_string(),
        ));
    }
    let n = count as f64;
    let mu: Vec<f32> = sum.iter().map(|&s| (s / n) as f32).collect();
    let sigma: Vec<f32> = sum
        .iter()
        .zip(&sum_sq)
        .map(|(&s, &q)| {
            let m = s / n;
            (((q / n) - m * m).max(0.0).sqrt() as f32).max(min_sigma)
        })
        .collect();
    let mut calib = DnCalibration { mu, sigma };
    symmetrize_calibration(model, &mut calib);
    Ok(calib)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlyConfig;
    use crate::params::FlyParams;

    /// Two `L`/`R` output pairs of one type, one of another, plus an unpaired `M` member.
    fn model_with_pairs() -> FlyModel {
        use crate::brain_fixtures::{FxNeuron, FxType, build_brain_flyg};
        use ddai_flyg::{NeuronRole, Sign};
        let types = vec![
            FxType {
                name: "A",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "B",
                sign: Sign::Excitatory,
            },
        ];
        let out = |type_index: u32, side: Side| FxNeuron {
            type_index,
            role: NeuronRole::Output,
            side,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        };
        let neurons = vec![
            out(0, Side::L),
            out(0, Side::R),
            out(0, Side::L),
            out(0, Side::R),
            out(1, Side::L),
            out(1, Side::R),
            out(1, Side::M),
        ];
        let flyg = build_brain_flyg(&types, &neurons, &[], &[], &[]);
        let params = FlyParams::init_default(&flyg, &FlyConfig::default(), 1);
        FlyModel::new(flyg, FlyConfig::default(), params).unwrap()
    }

    #[test]
    fn symmetrize_averages_homolog_pairs_and_leaves_unpaired_members_alone() {
        let model = model_with_pairs();
        let mut c = DnCalibration {
            mu: vec![1.0, 3.0, 5.0, 9.0, 2.0, 4.0, 7.0],
            sigma: vec![0.5, 1.5, 2.0, 4.0, 1.0, 3.0, 8.0],
        };
        symmetrize_calibration(&model, &mut c);
        assert_eq!(c.mu, vec![2.0, 2.0, 7.0, 7.0, 3.0, 3.0, 7.0]);
        assert_eq!(c.sigma, vec![1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 8.0]);
    }

    #[test]
    fn empty_input_is_an_error() {
        let model = model_with_pairs();
        let flyg_encoder = crate::encoder::EncoderModel::new(
            &model,
            crate::encoder::RayGridConfig::default(),
            &crate::encoder::ProprioceptionConfig::default(),
        )
        .unwrap();
        let p = EncoderParams::init_default(flyg_encoder.num_params());
        let v = vec![0.0; model.num_neurons()];
        assert!(calibrate_from_windows(&model, &flyg_encoder, &p, &v, &[], 0, 0.05).is_err());
    }
}
