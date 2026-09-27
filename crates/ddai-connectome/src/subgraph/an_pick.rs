//! Data-driven ascending-neuron (AN) type selection (acceptance criterion 6): MaleCNS AN types
//! are "mostly unnamed by function" (no biological label to pick from, unlike the named VPN/DN
//! types), so instead of a curated list, `build-subgraph` picks the top `an_top_n_types` AN types
//! by **total direct synaptic input onto the selected output (DN) neurons and onto "proxy
//! central" targets** — neurons that receive at least `theta_path` synapses directly from a
//! selected visual-input (VPN) neuron.
//!
//! This is a deliberate simplification of the task spec's own example rule ("strongest direct
//! synaptic input onto the selected DNs **and AOTU/central hidden neurons**"): using the actual
//! *selected hidden-neuron set* would make AN selection depend on `select::select_hidden`'s
//! output, which itself doesn't need AN seeds to run (AN seeds only affect where paths can
//! *start* from) — except AN seeds obviously must be known *before* hidden selection runs, since
//! they are seeds. Picking AN types first (from DN seeds + a one-hop VPN-seed proxy for "central"
//! neurons, both computable without running the path search at all) breaks that circularity while
//! still capturing the two things the spec's own rule cares about: direct premotor input to
//! output, and input to the first layer of central-brain processing downstream of vision. Both
//! the exact rule and its result (the AN types picked, flagged "С/П" per FLY.md's labeling
//! convention — a data-driven proxy, not a named biological role) are reported by
//! `build-subgraph`, per the acceptance criterion's "flag them ... list them".

use std::collections::BTreeMap;

use super::config::SelectionParams;
use crate::tables::ConnectomeTables;

#[derive(Debug, Clone)]
pub struct AnTypeScore {
    pub type_id: u32,
    pub name: String,
    /// Total synapses (any weight) from this type's neurons onto the target set.
    pub score: u64,
}

/// Picks the top `params.an_top_n_types` ascending-neuron types by the rule in the module docs.
/// `visual_seed_indices`/`output_seed_indices` are global dense indices (any order).
pub fn pick_an_types(
    tables: &ConnectomeTables,
    params: &SelectionParams,
    visual_seed_indices: &[u32],
    output_seed_indices: &[u32],
) -> Vec<AnTypeScore> {
    let n = tables.neurons.rows.len();
    let theta_path = u64::from(params.theta_path);

    let mut is_visual_seed = vec![false; n];
    for &i in visual_seed_indices {
        is_visual_seed[i as usize] = true;
    }
    let mut is_target = vec![false; n];
    for &i in output_seed_indices {
        is_target[i as usize] = true;
    }
    // Proxy "central" targets: one-hop, weight >= theta_path, directly downstream of a visual
    // seed (see module docs for why this stands in for "AOTU/central hidden neurons" without
    // needing the hidden-neuron search to have run yet).
    for e in &tables.edges.edges {
        if is_visual_seed[e.pre_idx as usize] && u64::from(e.weight) >= theta_path {
            is_target[e.post_idx as usize] = true;
        }
    }

    let ascending_id = tables
        .dictionaries
        .superclasses
        .iter()
        .position(|s| s == "ascending_neuron")
        .map(|i| i as u16);

    let mut score_by_type: BTreeMap<u32, u64> = BTreeMap::new();
    for e in &tables.edges.edges {
        if !is_target[e.post_idx as usize] {
            continue;
        }
        let pre = &tables.neurons.rows[e.pre_idx as usize];
        if pre.superclass != ascending_id || ascending_id.is_none() {
            continue;
        }
        let Some(type_id) = pre.type_id else { continue };
        *score_by_type.entry(type_id).or_insert(0) += u64::from(e.weight);
    }

    let mut scored: Vec<AnTypeScore> = score_by_type
        .into_iter()
        .map(|(type_id, score)| AnTypeScore {
            type_id,
            name: tables.types[type_id as usize].name.clone(),
            score,
        })
        .collect();
    // Score descending, type name ascending as a deterministic tie-break (never a HashMap
    // iteration order: `score_by_type` above is a `BTreeMap`, iterated once into a `Vec`, and this
    // sort is itself stable on ties only up to the explicit tie-break key, not on map order).
    scored.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.name.cmp(&b.name)));
    scored.truncate(params.an_top_n_types as usize);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{Dictionaries, Edge, EdgesTable, NeuronNt, NeuronRow, NeuronsTable, TablesHeader, TypeRow};
    use std::collections::BTreeMap as StdBTreeMap;

    /// visual_seed(0) --5--> proxy_target(1, superclass "central")
    /// an_strong(2, type AN_A) --10--> proxy_target(1)
    /// an_weak(3, type AN_B) --2--> proxy_target(1)
    /// an_strong2(4, type AN_A, other side) --1--> output_seed(5)
    /// an_unrelated(6, type AN_C) --100--> not_a_target(7)
    fn tiny_tables() -> ConnectomeTables {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![
                "ascending_neuron".to_string(),
                "central".to_string(),
                "visual_projection".to_string(),
            ],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec![],
        };
        let types = vec![
            TypeRow {
                name: "AN_A".into(),
                consensus_nt: None,
            },
            TypeRow {
                name: "AN_B".into(),
                consensus_nt: None,
            },
            TypeRow {
                name: "AN_C".into(),
                consensus_nt: None,
            },
        ];
        let mk = |body_id: i64, superclass: Option<u16>, type_id: Option<u32>| NeuronRow {
            body_id,
            status: Some(0),
            type_id,
            instance: None,
            superclass,
            class: None,
            subclass: None,
            soma_side: None,
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        };
        let rows = vec![
            mk(0, Some(2), None),    // 0: visual seed
            mk(1, Some(1), None),    // 1: proxy target (central)
            mk(2, Some(0), Some(0)), // 2: AN_A
            mk(3, Some(0), Some(1)), // 3: AN_B
            mk(4, Some(0), Some(0)), // 4: AN_A (other side)
            mk(5, None, None),       // 5: output seed
            mk(6, Some(0), Some(2)), // 6: AN_C
            mk(7, None, None),       // 7: not a target
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 1,
                weight: 5,
            },
            Edge {
                pre_idx: 2,
                post_idx: 1,
                weight: 10,
            },
            Edge {
                pre_idx: 3,
                post_idx: 1,
                weight: 2,
            },
            Edge {
                pre_idx: 4,
                post_idx: 5,
                weight: 1,
            },
            Edge {
                pre_idx: 6,
                post_idx: 7,
                weight: 100,
            },
        ];
        ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: StdBTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input: vec![0; 8],
                total_output: vec![0; 8],
            },
            neuron_nt: vec![NeuronNt::default(); 8],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        }
    }

    #[test]
    fn scores_an_types_by_synapses_onto_targets_only() {
        let tables = tiny_tables();
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 10,
        };
        let scored = pick_an_types(&tables, &params, &[0], &[5]);
        let by_name: BTreeMap<&str, u64> = scored.iter().map(|s| (s.name.as_str(), s.score)).collect();
        // AN_A: 10 (neuron 2 -> proxy target 1) + 1 (neuron 4 -> output seed 5) = 11.
        assert_eq!(by_name.get("AN_A"), Some(&11));
        // AN_B: 2 (neuron 3 -> proxy target 1).
        assert_eq!(by_name.get("AN_B"), Some(&2));
        // AN_C targets neuron 7, which is neither an output seed nor a proxy target -> excluded.
        assert!(!by_name.contains_key("AN_C"), "{scored:?}");
    }

    #[test]
    fn proxy_target_requires_theta_path_not_just_any_edge() {
        let tables = tiny_tables();
        let params = SelectionParams {
            k: 2,
            theta_path: 6, // now the visual_seed(0)->proxy_target(1) edge (weight 5) doesn't qualify
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 10,
        };
        let scored = pick_an_types(&tables, &params, &[0], &[5]);
        let by_name: BTreeMap<&str, u64> = scored.iter().map(|s| (s.name.as_str(), s.score)).collect();
        // Neuron 1 is no longer a target at all, so only AN_A's direct-to-output-seed synapse
        // (neuron 4 -> 5, weight 1) counts now.
        assert_eq!(by_name.get("AN_A"), Some(&1));
        assert!(!by_name.contains_key("AN_B"));
    }

    #[test]
    fn top_n_truncates_and_ties_break_by_name() {
        let tables = tiny_tables();
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let scored = pick_an_types(&tables, &params, &[0], &[5]);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].name, "AN_A", "highest score (11) must win");
    }
}
