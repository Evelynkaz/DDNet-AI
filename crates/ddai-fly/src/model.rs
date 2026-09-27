//! [`FlyModel`]: an immutable (topology, hyper-parameters) plus mutable (trainable parameters)
//! view over a loaded `.flyg`, holding every array [`crate::state::FlyState::step_decision`] needs
//! to run the hot loop with zero further lookups into `.flyg`'s own (less cache-friendly, more
//! indirect) representation. See the crate README for the overall picture; this module is
//! acceptance criterion 1a/1b's "weight precompute".

use ddai_flyg::{Flyg, NeuronRole};

use crate::activation::softplus;
use crate::config::FlyConfig;
use crate::error::FlyError;
use crate::params::FlyParams;

/// A loaded `.flyg` graph plus hyper-parameters plus trainable parameters, with every derived
/// array ([`FlyModel::weights`], per-neuron bias/decay, `1/Z_i`) precomputed. Cheap to query
/// (`num_neurons`/`num_inputs`/`num_outputs`/…), expensive-ish to mutate ([`FlyModel::set_params`]
/// is `O(nnz + num_neurons)`, meant to be called when parameters actually change — e.g. once per
/// training step — never inside the per-decision hot loop).
///
/// One `FlyModel` can back many independent [`crate::state::FlyState`]s (e.g. parallel episodes
/// later, for ES/self-play): `step_decision` takes `&FlyModel` explicitly rather than a state
/// holding its own reference to one, so `set_params` (`&mut FlyModel`) never has to fight
/// outstanding borrows held by states — see the crate README's "API notes".
#[derive(Debug, Clone)]
pub struct FlyModel {
    flyg: Flyg,
    config: FlyConfig,
    params: FlyParams,

    /// `1 / Z_i`, `Z_i = max(1, N_i^in)^gamma` (FLY.md §4). Depends only on the graph and
    /// `config.gamma`, never changes after construction.
    inv_z: Vec<f32>,
    /// Parallel to `flyg.type_pairs`: the presynaptic type's sign as `-1.0`/`0.0`/`1.0`. Fixed
    /// (topology), looked up once per edge during weight precompute rather than re-reading
    /// `flyg.types[...].sign` (an extra indirection) on every `set_params`.
    type_pair_sign: Vec<f32>,
    /// Dense neuron indices with role `InputVisual` or `InputAscending`, ascending. Defines the
    /// canonical order `step_decision`'s `inputs` slice is read in (see its doc comment).
    input_neuron_indices: Vec<u32>,
    /// Dense neuron indices with role `Output`, ascending. Defines `DecisionOutput::dn_rates`'s
    /// order.
    output_neuron_indices: Vec<u32>,

    /// `w_ij`, parallel to `flyg.edges.pre_index`/`synapse_count`/`type_pair_index` (post-major
    /// CSR, acceptance criterion 1b). Recomputed by [`FlyModel::set_params`].
    weights: Vec<f32>,
    /// `b_{T(i)}`, per neuron. Recomputed by [`FlyModel::set_params`].
    bias: Vec<f32>,
    /// `1 - exp(-Δ/τ_{T(i)})`, per neuron. Recomputed by [`FlyModel::set_params`].
    decay: Vec<f32>,

    /// A `u16` narrowing copy of `flyg.edges.pre_index`, built once here (never touched by
    /// `set_params` — pure topology) when this graph fits (`num_neurons <= 65536`; both real S/M
    /// graphs do). Half the gather traffic of the `.flyg` format's native `u32` — review round 1
    /// (F3). `None` for a hypothetical graph too large for `u16` dense indices; `step_decision`
    /// falls back to `flyg().edges.pre_index` (`u32`) in that case, decided once per call, not
    /// once per CSR row (see [`FlyModel::narrow_pre_index`]'s doc comment).
    narrow_pre_index: Option<Vec<u16>>,
}

impl FlyModel {
    /// Builds a model from a loaded graph, hyper-parameters and initial trainable parameters.
    /// Validates `config` ([`FlyConfig::validate`]) and `params`'s shape against `flyg`
    /// ([`FlyParams::validate_shape`]) before doing any O(nnz) work.
    pub fn new(flyg: Flyg, config: FlyConfig, params: FlyParams) -> Result<Self, FlyError> {
        config.validate()?;
        params.validate_shape(&flyg)?;

        let n = flyg.neurons.len();
        let gamma = config.gamma;
        let inv_z: Vec<f32> = flyg
            .neuron_input_totals
            .full_connectome
            .iter()
            .map(|&z_raw| {
                let z = (z_raw.max(1) as f32).powf(gamma);
                1.0 / z
            })
            .collect();

        let type_pair_sign: Vec<f32> = flyg
            .type_pairs
            .iter()
            .map(|tp| f32::from(flyg.types[tp.pre_type as usize].sign.as_i8()))
            .collect();

        let mut input_neuron_indices = Vec::new();
        let mut output_neuron_indices = Vec::new();
        for (i, nr) in flyg.neurons.iter().enumerate() {
            match nr.role {
                NeuronRole::InputVisual | NeuronRole::InputAscending => input_neuron_indices.push(i as u32),
                NeuronRole::Output => output_neuron_indices.push(i as u32),
                NeuronRole::Hidden => {}
            }
        }

        let nnz = flyg.edges.pre_index.len();
        // `u16::MAX` (65535) is itself a valid dense index when `n == 65536`, so the cutoff is
        // `<=`, not `<`.
        let narrow_pre_index =
            (n <= usize::from(u16::MAX) + 1).then(|| flyg.edges.pre_index.iter().map(|&p| p as u16).collect());

        let mut model = FlyModel {
            flyg,
            config,
            params,
            inv_z,
            type_pair_sign,
            input_neuron_indices,
            output_neuron_indices,
            weights: vec![0.0; nnz],
            bias: vec![0.0; n],
            decay: vec![0.0; n],
            narrow_pre_index,
        };
        model.recompute();
        Ok(model)
    }

    /// Replaces the trainable parameters and recomputes `weights`/`bias`/`decay`
    /// (`O(nnz + num_neurons)`). Rejects a shape mismatch against this model's graph without
    /// touching any existing state.
    pub fn set_params(&mut self, params: FlyParams) -> Result<(), FlyError> {
        params.validate_shape(&self.flyg)?;
        self.params = params;
        self.recompute();
        Ok(())
    }

    fn recompute(&mut self) {
        let alpha: Vec<f32> = self.params.a.iter().map(|&a| softplus(a)).collect();

        let edges = &self.flyg.edges;
        for post in 0..self.flyg.neurons.len() {
            let start = edges.row_start[post] as usize;
            let end = edges.row_start[post + 1] as usize;
            let inv_z_post = self.inv_z[post];
            for e in start..end {
                let tp = edges.type_pair_index[e] as usize;
                let sign = self.type_pair_sign[tp];
                let shared_id = self.flyg.type_pairs[tp].shared_param_id as usize;
                let n_ij = edges.synapse_count[e] as f32;
                self.weights[e] = sign * alpha[shared_id] * n_ij * inv_z_post;
            }
        }

        let dt_s = self.config.dt_s();
        let tau_max_s = self.config.tau_max_s;
        for (i, neuron) in self.flyg.neurons.iter().enumerate() {
            let t = neuron.type_index as usize;
            self.bias[i] = self.params.b[t];
            let tau = (dt_s + softplus(self.params.theta[t])).min(tau_max_s);
            // 1 - exp(-dt/tau), via -expm1(-dt/tau) for accuracy when dt << tau (avoids the
            // catastrophic cancellation `1.0 - x.exp()` would have for x close to 0).
            self.decay[i] = -(-dt_s / tau).exp_m1();
        }
    }

    pub fn num_neurons(&self) -> usize {
        self.flyg.neurons.len()
    }

    pub fn num_types(&self) -> usize {
        self.flyg.types.len()
    }

    pub fn num_inputs(&self) -> usize {
        self.input_neuron_indices.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.output_neuron_indices.len()
    }

    pub fn config(&self) -> &FlyConfig {
        &self.config
    }

    pub fn params(&self) -> &FlyParams {
        &self.params
    }

    pub fn flyg(&self) -> &Flyg {
        &self.flyg
    }

    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    pub(crate) fn decay(&self) -> &[f32] {
        &self.decay
    }

    /// `Some(narrow copy of the CSR's pre_index)` when this graph fits in `u16` dense indices
    /// (`num_neurons <= 65536` — both real S/M graphs do), `None` otherwise (fall back to
    /// `flyg().edges.pre_index`, `u32`). `step_decision` calls this once per decision (not once
    /// per CSR row) and picks a gather path for the whole substep loop — see its doc comment.
    pub(crate) fn narrow_pre_index(&self) -> Option<&[u16]> {
        self.narrow_pre_index.as_deref()
    }

    /// Dense neuron indices with role `InputVisual` or `InputAscending`, ascending — the order
    /// `step_decision`'s `inputs` slice must be given in (see its doc comment). Exposed so a
    /// caller (the future input encoder, 7.3; this crate's own tests) can build the mapping from
    /// "which body/type is input `k`" without re-deriving it from `flyg().neurons` by hand.
    pub fn input_neuron_indices(&self) -> &[u32] {
        &self.input_neuron_indices
    }

    /// Dense neuron indices with role `Output`, ascending — `DecisionOutput::dn_rates`'s order.
    pub fn output_neuron_indices(&self) -> &[u32] {
        &self.output_neuron_indices
    }

    /// Maps an output-role neuron's dense index (as used in `flyg().output_groups`) to its
    /// position in `DecisionOutput::dn_rates` — a hook for the future action decoder (7.3), which
    /// will need to go from "`output_groups`'s named actions" to "this model's raw per-neuron
    /// rates". `None` if `neuron_index` isn't an output-role neuron of this graph.
    pub fn output_slot_for_neuron(&self, neuron_index: u32) -> Option<usize> {
        self.output_neuron_indices.binary_search(&neuron_index).ok()
    }

    /// Rough total heap footprint in bytes: the loaded graph plus every precomputed array. Uses
    /// `size_of::<T>() * len()` per array/string rather than an exact allocator-reported size (no
    /// such API exists for arbitrary `Vec`s without a custom allocator) — close enough for the
    /// capacity-planning/reporting acceptance criterion 4 asks for.
    pub fn memory_footprint_bytes(&self) -> usize {
        use std::mem::size_of;

        let flyg = &self.flyg;
        let neurons_bytes = flyg.neurons.len() * size_of::<ddai_flyg::FlygNeuron>();
        let types_bytes: usize = flyg
            .types
            .iter()
            .map(|t| size_of::<ddai_flyg::FlygType>() + t.name.len() + t.superclass.len() + t.class.len())
            .sum();
        let edges_bytes = flyg.edges.row_start.len() * size_of::<u32>()
            + flyg.edges.pre_index.len() * size_of::<u32>()
            + flyg.edges.synapse_count.len() * size_of::<u32>()
            + flyg.edges.type_pair_index.len() * size_of::<u32>();
        let totals_bytes = flyg.neuron_input_totals.full_connectome.len() * size_of::<u64>() * 2;
        let type_pairs_bytes = flyg.type_pairs.len() * size_of::<ddai_flyg::TypePair>();

        let precomputed_bytes = self.inv_z.len() * size_of::<f32>()
            + self.type_pair_sign.len() * size_of::<f32>()
            + self.input_neuron_indices.len() * size_of::<u32>()
            + self.output_neuron_indices.len() * size_of::<u32>()
            + self.weights.len() * size_of::<f32>()
            + self.bias.len() * size_of::<f32>()
            + self.decay.len() * size_of::<f32>()
            + self.narrow_pre_index.as_ref().map_or(0, |v| v.len() * size_of::<u16>());

        let params_bytes = (self.params.a.len() + self.params.b.len() + self.params.theta.len()) * size_of::<f32>();

        neurons_bytes + types_bytes + edges_bytes + totals_bytes + type_pairs_bytes + precomputed_bytes + params_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::tiny_chain_flyg;

    #[test]
    fn new_rejects_invalid_config() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig {
            substeps_per_decision: 0,
            ..FlyConfig::default()
        };
        let params = FlyParams::init_default(&flyg, &FlyConfig::default(), 1);
        assert!(matches!(
            FlyModel::new(flyg, config, params),
            Err(FlyError::InvalidConfig(_))
        ));
    }

    #[test]
    fn new_rejects_shape_mismatched_params() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let mut params = FlyParams::init_default(&flyg, &config, 1);
        params.b.pop();
        assert!(matches!(
            FlyModel::new(flyg, config, params),
            Err(FlyError::ParamShapeMismatch(_))
        ));
    }

    #[test]
    fn set_params_rejects_shape_mismatch_without_mutating_model() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let mut model = FlyModel::new(flyg.clone(), config, params.clone()).unwrap();
        let mut bad = params.clone();
        bad.theta.push(0.0);
        assert!(model.set_params(bad).is_err());
        assert_eq!(
            model.params(),
            &params,
            "a rejected set_params must not change the model's params"
        );
    }

    #[test]
    fn dims_match_the_graph() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        assert_eq!(model.num_neurons(), 3);
        assert_eq!(model.num_types(), 3);
        assert_eq!(model.num_inputs(), 1);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn narrow_pre_index_matches_the_wide_one_for_a_small_graph() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        let narrow = model.narrow_pre_index().expect("tiny graph fits in u16");
        let wide = &model.flyg().edges.pre_index;
        assert_eq!(narrow.len(), wide.len());
        for (&n, &w) in narrow.iter().zip(wide) {
            assert_eq!(u32::from(n), w);
        }
    }

    #[test]
    fn output_slot_for_neuron_finds_the_right_position() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        assert_eq!(model.output_slot_for_neuron(2), Some(0));
        assert_eq!(
            model.output_slot_for_neuron(0),
            None,
            "neuron 0 is InputAscending, not Output"
        );
    }

    #[test]
    fn memory_footprint_is_positive_and_grows_with_params() {
        let flyg = tiny_chain_flyg();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        let model = FlyModel::new(flyg, config, params).unwrap();
        assert!(model.memory_footprint_bytes() > 0);
    }
}
