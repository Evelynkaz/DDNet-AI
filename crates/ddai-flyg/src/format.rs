//! The `.flyg` v1 data model. See `README.md` for the file-level picture and
//! `docs/formats.md` (in the main repo, Russian) for the on-disk format writeup.
//!
//! Every collection here that is iterated to produce deterministic output (by the builder in
//! `ddai-connectome`) is a plain `Vec`, never a `HashMap` — see that crate's `subgraph` module for
//! how ordering is made independent of hash-iteration order. This crate itself doesn't rebuild
//! anything from scratch (no `HashMap` here at all): it only defines the shape and validates it.

use serde::{Deserialize, Serialize};

/// Bumped whenever the *meaning* of a stored field changes in a way an old reader would
/// misinterpret (see `ddai_connectome::tables::TABLES_FORMAT_VERSION` for the same convention).
pub const FLYG_FORMAT_VERSION: u32 = 1;

/// One of the 6 neurotransmitter classes MaleCNS v1.0 actually predicts a *sign* from, plus
/// `Unclear`. Deliberately a *separate* enum from `ddai_connectome::tables::NtClass` — this crate
/// has no dependency on `ddai-connectome` (see the crate-level docs), and this enum only needs to
/// round-trip through the `.flyg` file, not parse raw source-data strings. Dopamine/serotonin/
/// octopamine are grouped as `Modulatory` here (they all get the same treatment: excluded from
/// fast transmission by default — see `Sign`); the source NT class actually observed is not lost,
/// it is just not distinguished further since nothing downstream of this file needs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NtClassUsed {
    Acetylcholine,
    Glutamate,
    Gaba,
    Histamine,
    Modulatory,
    /// No usable neurotransmitter evidence at all for this type (no `consensus_nt`, and either no
    /// per-neuron `predicted_nt` rows or all of them `unclear`).
    Unknown,
}

/// A fixed synapse sign, per FLY.md §4/§7.3 and D-013: `+1` excitatory (ACh), `-1` inhibitory
/// (GABA/Glu/histamine), `0` neutral (excluded from fast transmission — moduolatory NTs by
/// default, or an uncertain type unless the config asks to use the best guess anyway).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i8)]
pub enum Sign {
    Inhibitory = -1,
    Neutral = 0,
    Excitatory = 1,
}

impl Sign {
    pub fn as_i8(self) -> i8 {
        self as i8
    }
}

/// Where a neuron sits in the sensory → central → motor pipeline (FLY.md §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronRole {
    /// Visual projection neuron (VPN) seed — "eyes".
    InputVisual,
    /// Ascending neuron (AN) seed — "body sense".
    InputAscending,
    /// Selected via the path-flow search between inputs and outputs.
    Hidden,
    /// Descending neuron (DN) seed — motor output.
    Output,
}

/// Soma side, straight from the source `somaSide` column (`L`/`R`/`M`), plus `Unknown` for the
/// rare Traced body with no side annotation at all (kept explicit rather than silently coerced to
/// one side, since that would corrupt bilateral-pairing and receptive-field sign logic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Side {
    L,
    R,
    M,
    Unknown,
}

/// A receptive-field center in the game's ray-grid "ommatidia" convention (FLY.md §5): degrees,
/// azimuth 0 = straight ahead, negative = left eye's field, positive = right eye's; elevation 0 =
/// eye equator, positive = dorsal (up). See `ddai-connectome`'s `subgraph::rf` module for how this
/// is computed and the exact linear map used.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReceptiveField {
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    /// `true` when this neuron had no hex-coordinated presynaptic partner at all and the center
    /// came from the documented fallback (spread by bodyId order within its type/side) rather
    /// than real synaptic input — see FLY.md §5 and `ddai-connectome`'s `subgraph::rf` module.
    pub is_fallback: bool,
}

/// One row of the `neurons` table (dense index = its position = the CSR's neuron index).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlygNeuron {
    pub body_id: i64,
    /// Index into [`Flyg::types`].
    pub type_index: u32,
    pub role: NeuronRole,
    pub side: Side,
    /// The source `group` column (bilateral-homolog id), when the source data has one.
    pub group_id: Option<i64>,
    /// `Some` only for `role == InputVisual`; always `None` otherwise.
    pub rf: Option<ReceptiveField>,
}

/// One row of the `types` table (dense index = [`FlygNeuron::type_index`]). Only types that
/// actually have at least one neuron in this subgraph appear here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlygType {
    pub name: String,
    pub superclass: String,
    pub class: String,
    /// The NT class the sign below was actually derived from (see `ddai-connectome`'s
    /// `subgraph::signs` module for the full rule, including the `consensus_nt`/`predicted_nt`
    /// fallback).
    pub nt_class_used: NtClassUsed,
    pub sign: Sign,
    /// In \[0, 1\]. 1.0 for a type resolved directly from a non-`unclear` `consensus_nt` (treated
    /// as fully curated); otherwise, from the `predicted_nt` fallback, the winning NT class's
    /// confidence-weighted vote total divided by how many of the type's neurons voted at all
    /// (**not** divided by the total weighted vote sum — that would make a single voting
    /// neuron's share trivially 1.0 regardless of its own confidence; see `ddai-connectome`'s
    /// `subgraph::signs` module). `0.0` if the type has no NT evidence of any kind.
    pub nt_confidence: f32,
    /// `true` when `nt_confidence` is below the config's threshold (or the type has no NT
    /// evidence at all) — regardless of whether `sign` ended up `Neutral` or a best-guess value
    /// (see the config's `use_best_guess_when_uncertain`).
    pub uncertain: bool,
    /// Number of this subgraph's neurons that have this type (not the full connectome's count).
    pub neuron_count: u32,
}

/// CSR edge storage, indexed by **postsynaptic** neuron: row `i`'s slice is
/// `pre_index[row_start[i]..row_start[i+1]]` (and the parallel `synapse_count`/`type_pair_index`
/// slices), each entry one presynaptic partner of neuron `i` within this subgraph. Within a row,
/// `pre_index` is sorted ascending (part of "CSR well-formed" — see [`crate::validate`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FlygEdges {
    /// Length `neurons.len() + 1`. `row_start[0] == 0`, non-decreasing, `row_start[n] ==
    /// pre_index.len()`.
    pub row_start: Vec<u32>,
    /// Length = total edge count (nnz). Dense neuron index of the presynaptic partner.
    pub pre_index: Vec<u32>,
    /// Parallel to `pre_index`: `N_ij`, the synapse count for that edge, within this subgraph.
    pub synapse_count: Vec<u32>,
    /// Parallel to `pre_index`: index into [`Flyg::type_pairs`].
    pub type_pair_index: Vec<u32>,
}

impl FlygEdges {
    pub fn num_edges(&self) -> usize {
        self.pre_index.len()
    }

    /// The presynaptic partners of neuron `post` (as `(pre_index, synapse_count,
    /// type_pair_index)` triples), or an empty slice-backed iterator if `post` is out of range.
    pub fn row(&self, post: u32) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        let (start, end) = match (self.row_start.get(post as usize), self.row_start.get(post as usize + 1)) {
            (Some(&s), Some(&e)) => (s as usize, e as usize),
            _ => (0, 0),
        };
        self.pre_index[start..end]
            .iter()
            .zip(&self.synapse_count[start..end])
            .zip(&self.type_pair_index[start..end])
            .map(|((&pre, &n), &tp)| (pre, n, tp))
    }
}

/// One row of the `type_pairs` table — see FLY.md §4 for why pairs of types (not pairs of
/// neurons) are the unit of a learnable connection-strength parameter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TypePair {
    /// Index into [`Flyg::types`].
    pub pre_type: u32,
    /// Index into [`Flyg::types`].
    pub post_type: u32,
    /// Sum of `N_ij` over this subgraph's edges with this exact (pre_type, post_type) pair.
    pub total_synapses: u64,
    /// Which learnable parameter this pair's connection strength shares. Pairs at or above the
    /// config's weak-pair threshold each get their own unique id; pairs below it share one id per
    /// (pre type's sign, pre superclass, post superclass) — see `subgraph::type_pairs` for the
    /// exact assignment and `Flyg::summary.shared_param_count` for how many distinct ids exist.
    pub shared_param_id: u32,
}

/// Per-neuron total input synapse counts, parallel to [`Flyg::neurons`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NeuronInputTotals {
    /// `N_i^in`: total incoming synapses over the **full** MaleCNS connectome (not just this
    /// subgraph) — the normalizer `Z_i` in FLY.md §4 is derived from this (`γ` is not applied
    /// here; this is the raw count the model computes `Z_i` from at load time).
    pub full_connectome: Vec<u64>,
    /// Total incoming synapses within this subgraph only (sum of `synapse_count` over the
    /// neuron's CSR row) — provided for convenience/sanity-checking; always `<=` the
    /// corresponding `full_connectome` entry.
    pub in_subgraph: Vec<u64>,
}

/// Which game input channels feed a given visual-input type (FLY.md §5's mapping table), from the
/// selection config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputChannelMapping {
    /// Index into [`Flyg::types`]; that type's role must be `InputVisual` or `InputAscending`.
    pub type_index: u32,
    /// Game channel names (e.g. `"opponent_position"`), free-form strings from the config —
    /// deliberately not an enum, so the config can name channels the game side defines without a
    /// matching Rust enum variant needing to exist here too.
    pub channels: Vec<String>,
}

/// One descending neuron behind an action head, with its side (FLY.md §6).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutputMember {
    /// Dense index into [`Flyg::neurons`]; that neuron's role must be `Output`.
    pub neuron_index: u32,
    pub side: Side,
}

/// One action head (e.g. `"direction_left"`) and the DN neurons that drive it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputGroup {
    pub action: String,
    pub members: Vec<OutputMember>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RoleCounts {
    pub input_visual: u32,
    pub input_ascending: u32,
    pub hidden: u32,
    pub output: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SignCounts {
    pub excitatory: u32,
    pub inhibitory: u32,
    pub neutral: u32,
}

/// Summary statistics baked into the file at build time (acceptance criterion 1's "summary
/// statistics") — a convenience snapshot for `flyg-info`/reports, computed once by the builder
/// from the same data as the rest of the file. Not re-derived at load time, but every field here
/// is redundant with (computable from) the rest of the file, so [`crate::validate`] cross-checks
/// the counts instead of trusting them blindly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub neurons_by_role: RoleCounts,
    pub num_types: u32,
    pub num_edges: u32,
    pub num_type_pairs: u32,
    /// Number of *distinct* `shared_param_id` values across `type_pairs` (strong pairs each count
    /// individually; weak pairs sharing one id count once).
    pub shared_param_count: u32,
    pub sign_counts: SignCounts,
    pub uncertain_types: u32,
    /// Number of `InputVisual` neurons whose RF center came from the documented fallback (no
    /// hex-coordinated presynaptic partner at all) rather than real hex data.
    pub rf_fallback_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlygHeader {
    pub format_version: u32,
    /// sha256 (hex) of the `connectome.tables` file this subgraph was selected from.
    pub source_tables_sha256: String,
    /// sha256 (hex) of the selection config file's raw bytes.
    pub config_sha256: String,
    /// The generator's own version string (`ddai-connectome`'s crate version), for provenance —
    /// not format-checked, purely informational.
    pub generator_version: String,
}

/// The complete `.flyg` v1 file contents. See the module docs and `docs/formats.md` for the
/// on-disk layout (postcard + zstd, same convention as `ddai_connectome::tables`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Flyg {
    pub header: FlygHeader,
    pub neurons: Vec<FlygNeuron>,
    pub types: Vec<FlygType>,
    pub edges: FlygEdges,
    pub neuron_input_totals: NeuronInputTotals,
    pub type_pairs: Vec<TypePair>,
    pub input_channels: Vec<InputChannelMapping>,
    pub output_groups: Vec<OutputGroup>,
    pub summary: Summary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_row_returns_the_right_slice_per_post_neuron() {
        // post=0: no edges; post=1: one edge from pre=0; post=2: two edges from pre=0,1.
        let edges = FlygEdges {
            row_start: vec![0, 0, 1, 3],
            pre_index: vec![0, 0, 1],
            synapse_count: vec![5, 7, 9],
            type_pair_index: vec![0, 1, 1],
        };
        assert_eq!(edges.row(0).collect::<Vec<_>>(), vec![]);
        assert_eq!(edges.row(1).collect::<Vec<_>>(), vec![(0, 5, 0)]);
        assert_eq!(edges.row(2).collect::<Vec<_>>(), vec![(0, 7, 1), (1, 9, 1)]);
        assert_eq!(edges.num_edges(), 3);
    }

    #[test]
    fn edges_row_out_of_range_post_returns_empty_rather_than_panicking() {
        let edges = FlygEdges {
            row_start: vec![0, 0],
            pre_index: vec![],
            synapse_count: vec![],
            type_pair_index: vec![],
        };
        assert_eq!(edges.row(99).collect::<Vec<_>>(), vec![]);
    }

    #[test]
    fn sign_as_i8_matches_the_documented_values() {
        assert_eq!(Sign::Excitatory.as_i8(), 1);
        assert_eq!(Sign::Inhibitory.as_i8(), -1);
        assert_eq!(Sign::Neutral.as_i8(), 0);
    }
}
