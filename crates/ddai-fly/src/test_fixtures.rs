//! Tiny hand-built [`Flyg`] graphs for this crate's own correctness tests (`tests/correctness.rs`)
//! and unit tests. `#[doc(hidden)]` and not covered by semver — this exists so both the unit tests
//! in `src/` and the integration tests in `tests/` can share one builder instead of hand-rolling
//! CSR/summary bookkeeping in every test file. Every graph this module builds is run through
//! `ddai_flyg::validate` before being returned, so a bug in the builder itself fails loudly instead
//! of silently feeding the model malformed input.

use ddai_flyg::{
    Flyg, FlygEdges, FlygHeader, FlygNeuron, FlygType, NeuronInputTotals, NeuronRole, NtClassUsed, ReceptiveField,
    RoleCounts, Side, Sign, SignCounts, Summary, TypePair,
};

/// One entry of the type table a fixture graph is built from.
#[derive(Debug, Clone, Copy)]
pub struct FxType {
    pub name: &'static str,
    pub sign: Sign,
}

/// One neuron. `full_connectome_in` is `N_i^in` (FLY.md §4's `Z_i` input) — the fixture doesn't
/// derive it from the edges given (a real subgraph's neurons almost always have *more* full-graph
/// input than the subgraph captures), so tests set it explicitly to whatever makes the hand
/// computation clean (often exactly the subgraph's own in-degree, giving `Z_i` a known value).
#[derive(Debug, Clone, Copy)]
pub struct FxNeuron {
    pub type_index: u32,
    pub role: NeuronRole,
    pub side: Side,
    pub full_connectome_in: u64,
}

/// One edge, `pre -> post`, both dense neuron indices into the `neurons` slice passed to
/// [`build_flyg`].
#[derive(Debug, Clone, Copy)]
pub struct FxEdge {
    pub pre: u32,
    pub post: u32,
    pub synapse_count: u32,
}

/// Builds a valid [`Flyg`] from plain descriptions: assigns `body_id = ` dense index (already
/// ascending, matching the format's own invariant), gives every `(pre_type, post_type)` pair
/// actually used its own unique `shared_param_id` (no weak-pair merging — fixtures are small
/// enough that merging would only make tests harder to read, not exercise anything the real
/// builder's own tests, in `ddai-connectome`, don't already cover), and fills in `summary`
/// correctly. Panics (via the `validate` call at the end) if the description is internally
/// inconsistent — a bug in a test's fixture, not in production data, so a panic here is the right
/// failure mode.
pub fn build_flyg(types: &[FxType], neurons: &[FxNeuron], edges: &[FxEdge]) -> Flyg {
    let n = neurons.len();
    let t = types.len();

    let flyg_neurons: Vec<FlygNeuron> = neurons
        .iter()
        .enumerate()
        .map(|(i, nr)| FlygNeuron {
            body_id: i as i64,
            type_index: nr.type_index,
            role: nr.role,
            side: nr.side,
            group_id: None,
            rf: matches!(nr.role, NeuronRole::InputVisual).then_some(ReceptiveField {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                is_fallback: false,
            }),
        })
        .collect();

    let mut neuron_count = vec![0u32; t];
    for nr in neurons {
        neuron_count[nr.type_index as usize] += 1;
    }
    let flyg_types: Vec<FlygType> = types
        .iter()
        .zip(neuron_count)
        .map(|(ty, count)| FlygType {
            name: ty.name.to_string(),
            superclass: "test".to_string(),
            class: "test".to_string(),
            nt_class_used: match ty.sign {
                Sign::Excitatory => NtClassUsed::Acetylcholine,
                Sign::Inhibitory => NtClassUsed::Gaba,
                Sign::Neutral => NtClassUsed::Modulatory,
            },
            sign: ty.sign,
            nt_confidence: 1.0,
            uncertain: false,
            neuron_count: count,
        })
        .collect();

    // Sort edges by (post, pre) for a well-formed CSR (ascending pre_index within a row).
    let mut sorted_edges: Vec<FxEdge> = edges.to_vec();
    sorted_edges.sort_by_key(|e| (e.post, e.pre));

    // One shared_param_id per distinct (pre_type, post_type) pair, assigned in first-seen order.
    let mut pair_id: Vec<((u32, u32), u32)> = Vec::new();
    let mut pair_totals: Vec<u64> = Vec::new();
    let type_pair_index_of =
        |pre_type: u32, post_type: u32, pair_id: &mut Vec<((u32, u32), u32)>, pair_totals: &mut Vec<u64>| -> usize {
            if let Some(pos) = pair_id.iter().position(|&(k, _)| k == (pre_type, post_type)) {
                pair_totals[pos] += 0; // no-op, kept for symmetry with the insert branch below
                pos
            } else {
                pair_id.push(((pre_type, post_type), pair_id.len() as u32));
                pair_totals.push(0);
                pair_id.len() - 1
            }
        };

    let mut row_start = vec![0u32; n + 1];
    let mut pre_index = Vec::with_capacity(sorted_edges.len());
    let mut synapse_count = Vec::with_capacity(sorted_edges.len());
    let mut type_pair_index = Vec::with_capacity(sorted_edges.len());
    let mut in_subgraph = vec![0u64; n];

    for e in &sorted_edges {
        row_start[e.post as usize + 1] += 1;
    }
    for i in 0..n {
        row_start[i + 1] += row_start[i];
    }

    for e in &sorted_edges {
        let pre_type = neurons[e.pre as usize].type_index;
        let post_type = neurons[e.post as usize].type_index;
        let pos = type_pair_index_of(pre_type, post_type, &mut pair_id, &mut pair_totals);
        pair_totals[pos] += u64::from(e.synapse_count);
        pre_index.push(e.pre);
        synapse_count.push(e.synapse_count);
        type_pair_index.push(pos as u32);
        in_subgraph[e.post as usize] += u64::from(e.synapse_count);
    }

    let type_pairs: Vec<TypePair> = pair_id
        .iter()
        .zip(&pair_totals)
        .map(|(&((pre_type, post_type), id), &total)| TypePair {
            pre_type,
            post_type,
            total_synapses: total,
            shared_param_id: id,
        })
        .collect();

    let mut role_counts = RoleCounts::default();
    for nr in neurons {
        match nr.role {
            NeuronRole::InputVisual => role_counts.input_visual += 1,
            NeuronRole::InputAscending => role_counts.input_ascending += 1,
            NeuronRole::Hidden => role_counts.hidden += 1,
            NeuronRole::Output => role_counts.output += 1,
        }
    }
    let mut sign_counts = SignCounts::default();
    for ty in types {
        match ty.sign {
            Sign::Excitatory => sign_counts.excitatory += 1,
            Sign::Inhibitory => sign_counts.inhibitory += 1,
            Sign::Neutral => sign_counts.neutral += 1,
        }
    }

    let flyg = Flyg {
        header: FlygHeader {
            format_version: ddai_flyg::FLYG_FORMAT_VERSION,
            source_tables_sha256: "0".repeat(64),
            config_sha256: "0".repeat(64),
            generator_version: "ddai-fly test fixture".to_string(),
        },
        neurons: flyg_neurons,
        types: flyg_types,
        edges: FlygEdges {
            row_start,
            pre_index,
            synapse_count,
            type_pair_index,
        },
        neuron_input_totals: NeuronInputTotals {
            full_connectome: neurons.iter().map(|nr| nr.full_connectome_in).collect(),
            in_subgraph,
        },
        type_pairs,
        input_channels: Vec::new(),
        output_groups: Vec::new(),
        summary: Summary {
            neurons_by_role: role_counts,
            num_types: t as u32,
            num_edges: sorted_edges.len() as u32,
            num_type_pairs: pair_id.len() as u32,
            shared_param_count: pair_id.len() as u32,
            sign_counts,
            uncertain_types: 0,
            rf_fallback_count: 0,
        },
    };
    ddai_flyg::validate(&flyg).expect("test fixture must build a valid .flyg");
    flyg
}

/// Looks up the `shared_param_id` [`build_flyg`] assigned to a given `(pre_type, post_type)` pair,
/// so a test can set `FlyParams::a[that id]` to a specific value.
pub fn shared_param_id_for(flyg: &Flyg, pre_type: u32, post_type: u32) -> u32 {
    flyg.type_pairs
        .iter()
        .find(|tp| tp.pre_type == pre_type && tp.post_type == post_type)
        .unwrap_or_else(|| panic!("no type_pair ({pre_type}, {post_type}) in this fixture"))
        .shared_param_id
}

/// A minimal 3-neuron chain (`input -type0-> hidden -type1-> output`, one edge per hop, 3 distinct
/// types), used where a test just needs *some* valid, shape-checkable graph and doesn't care about
/// its numeric behaviour (e.g. `FlyParams` shape-validation tests).
pub fn tiny_chain_flyg() -> Flyg {
    let types = [
        FxType {
            name: "in_t",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "hidden_t",
            sign: Sign::Excitatory,
        },
        FxType {
            name: "out_t",
            sign: Sign::Excitatory,
        },
    ];
    let neurons = [
        FxNeuron {
            type_index: 0,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 10,
        },
        FxNeuron {
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 10,
        },
        FxNeuron {
            type_index: 2,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 10,
        },
    ];
    let edges = [
        FxEdge {
            pre: 0,
            post: 1,
            synapse_count: 5,
        },
        FxEdge {
            pre: 1,
            post: 2,
            synapse_count: 5,
        },
    ];
    build_flyg(&types, &neurons, &edges)
}
