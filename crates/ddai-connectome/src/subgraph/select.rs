//! Hidden-neuron selection (acceptance criterion 3): path-flow ranking between input and output
//! seeds, then bilateral completeness.
//!
//! **Path-flow score.** For hop budget `k`, define (over edges with `weight >= theta_path`):
//! `F[h][v]` = the best score of any path of **at most** `h` edges from an input seed to `v`
//! (`F[0][v] = 1.0` if `v` is an input seed, else `0.0`); `B[h][v]` = the same, backwards, to an
//! output seed. Both are relaxed hop by hop:
//! ```text
//! F[h][b] = max(F[h-1][b], max over edges a->b of F[h-1][a] * weight(a,b) / N_in^full(b))
//! B[h][a] = max(B[h-1][a], max over edges a->b of weight(a,b) / N_in^full(b) * B[h-1][b])
//! ```
//! Both are monotone non-decreasing in `h` (each carries the previous hop's value forward), so
//! for a candidate hidden neuron `v` the best score achievable within a total budget of `k` hops
//! is `max over h1 in 0..=k of F[h1][v] * B[k-h1][v]` — trying every split of the budget between
//! the "input → v" and "v → output" halves (see the module tests for why fixing the split at
//! `h1 + h2 == k` exactly, rather than also trying `< k`, already covers the true best case).
//!
//! **Bilateral completeness.** After keeping the top `max_hidden` candidates by score, every
//! selected neuron's homolog(s) are added too: same `group` id if it has one, else every other
//! neuron of the same type on the mirrored side (`L` <-> `R`; `M`/unknown sides have no natural
//! mirror and are left alone) — see the task spec's acceptance criterion 3(c). This can push the
//! final hidden count slightly above `max_hidden`; the actual final count is what gets reported.

use std::collections::{BTreeSet, VecDeque};

use ddai_flyg::Side;

use super::common::neuron_side;
use super::config::SelectionParams;
use crate::tables::ConnectomeTables;

#[derive(Debug, Clone, Default)]
pub struct SeedSets {
    /// Global dense indices, sorted ascending, deduplicated.
    pub visual_input: Vec<u32>,
    pub ascending_input: Vec<u32>,
    pub output: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct SelectionResult {
    pub hidden: Vec<u32>,
    /// How many of `hidden` came from the top-`max_hidden` ranked candidates (before completion).
    pub hidden_core_count: u32,
    /// How many extra neurons bilateral completeness added, split by which rule pulled them in.
    pub bilateral_added_via_group: u32,
    pub bilateral_added_via_type_fallback: u32,
}

fn mirror_side(side: Side) -> Option<Side> {
    match side {
        Side::L => Some(Side::R),
        Side::R => Some(Side::L),
        Side::M | Side::Unknown => None,
    }
}

/// Runs the forward/backward path-flow relaxation and returns `score[v]` for every neuron
/// (`0.0` for anything unreachable within `k` hops from both sides).
fn path_flow_scores(tables: &ConnectomeTables, params: &SelectionParams, seeds: &SeedSets) -> Vec<f64> {
    let n = tables.neurons.rows.len();
    let k = params.k as usize;
    let theta_path = u64::from(params.theta_path);

    let mut is_input = vec![false; n];
    for &i in seeds.visual_input.iter().chain(&seeds.ascending_input) {
        is_input[i as usize] = true;
    }
    let mut is_output = vec![false; n];
    for &i in &seeds.output {
        is_output[i as usize] = true;
    }

    let mut f_prev: Vec<f64> = (0..n).map(|i| if is_input[i] { 1.0 } else { 0.0 }).collect();
    let mut b_prev: Vec<f64> = (0..n).map(|i| if is_output[i] { 1.0 } else { 0.0 }).collect();
    // `f_at[h]`/`b_at[h]` for h in 0..=k, needed afterwards to try every hop-budget split.
    let mut f_at: Vec<Vec<f64>> = vec![f_prev.clone()];
    let mut b_at: Vec<Vec<f64>> = vec![b_prev.clone()];

    for _hop in 1..=k {
        let mut f_next = f_prev.clone();
        let mut b_next = b_prev.clone();
        for e in &tables.edges.edges {
            if u64::from(e.weight) < theta_path {
                continue;
            }
            let post = e.post_idx as usize;
            let pre = e.pre_idx as usize;
            let denom = tables.neurons.total_input[post].max(1) as f64;
            let factor = f64::from(e.weight) / denom;

            let cand_f = f_prev[pre] * factor;
            if cand_f > f_next[post] {
                f_next[post] = cand_f;
            }
            let cand_b = factor * b_prev[post];
            if cand_b > b_next[pre] {
                b_next[pre] = cand_b;
            }
        }
        f_at.push(f_next.clone());
        b_at.push(b_next.clone());
        f_prev = f_next;
        b_prev = b_next;
    }

    let mut score = vec![0.0f64; n];
    for v in 0..n {
        if is_input[v] || is_output[v] {
            continue;
        }
        let mut best = 0.0f64;
        for h1 in 0..=k {
            let cand = f_at[h1][v] * b_at[k - h1][v];
            if cand > best {
                best = cand;
            }
        }
        score[v] = best;
    }
    score
}

/// Neurons sharing both `group_id` **and** type with `idx` (excluding `idx` itself), or, if `idx`
/// has no `group`, every other Traced neuron of the same type on the mirrored side.
///
/// The group rule requires the same type as well as the same `group` id: `group` links bilateral
/// homologs, but nothing guarantees a `group` id is only ever shared within one type on the real
/// data, and a homolog is specifically "the same cell on the other side" — a different-typed
/// neuron that happens to share a `group` id is not that (see the round-1 review finding this
/// fixes: without this filter, the group rule pulled in unrelated-typed neurons).
fn homologs_of(tables: &ConnectomeTables, idx: u32, traced_id: Option<u16>) -> (Vec<u32>, bool) {
    let row = &tables.neurons.rows[idx as usize];
    if let Some(group_id) = row.group {
        let homologs: Vec<u32> = tables
            .neurons
            .rows
            .iter()
            .enumerate()
            .filter(|(i, r)| {
                *i as u32 != idx && r.group == Some(group_id) && r.status == traced_id && r.type_id == row.type_id
            })
            .map(|(i, _)| i as u32)
            .collect();
        return (homologs, false);
    }
    let Some(mirrored) = mirror_side(neuron_side(tables, idx)) else {
        return (Vec::new(), true);
    };
    let Some(type_id) = row.type_id else {
        return (Vec::new(), true);
    };
    let homologs: Vec<u32> = tables
        .neurons
        .rows
        .iter()
        .enumerate()
        .filter(|(i, r)| {
            *i as u32 != idx
                && r.type_id == Some(type_id)
                && r.status == traced_id
                && neuron_side(tables, *i as u32) == mirrored
        })
        .map(|(i, _)| i as u32)
        .collect();
    (homologs, true)
}

/// Selects the hidden-neuron set: ranks candidates by path-flow score, keeps the top
/// `params.max_hidden`, then enforces bilateral completeness. `seeds` must already contain every
/// instance (both sides) of the configured input/output types — this function only ever *adds*
/// neurons to (never removes from, never reclassifies) that seed set.
pub fn select_hidden(tables: &ConnectomeTables, params: &SelectionParams, seeds: &SeedSets) -> SelectionResult {
    let n = tables.neurons.rows.len();
    let traced_id = tables
        .dictionaries
        .statuses
        .iter()
        .position(|s| s == "Traced")
        .map(|i| i as u16);

    let score = path_flow_scores(tables, params, seeds);

    let mut is_seed = vec![false; n];
    for &i in seeds
        .visual_input
        .iter()
        .chain(&seeds.ascending_input)
        .chain(&seeds.output)
    {
        is_seed[i as usize] = true;
    }

    // A hidden neuron must have a `type` (the `.flyg` format shares signs/parameters per type,
    // so an untyped neuron could never get one) — untyped neurons are still traversed as
    // intermediate hops by `path_flow_scores` above, just never picked as a terminal hidden
    // neuron themselves.
    let mut candidates: Vec<u32> = (0..n as u32)
        .filter(|&v| {
            !is_seed[v as usize] && score[v as usize] > 0.0 && tables.neurons.rows[v as usize].type_id.is_some()
        })
        .collect();
    // Highest score first; ties broken by bodyId ascending for determinism (no HashMap iteration
    // order ever reaches this sort — `score` is a plain `Vec` indexed by dense idx).
    candidates.sort_by(|&a, &b| {
        score[b as usize]
            .partial_cmp(&score[a as usize])
            .expect("scores are finite: products of finite non-negative factors")
            .then_with(|| {
                tables.neurons.rows[a as usize]
                    .body_id
                    .cmp(&tables.neurons.rows[b as usize].body_id)
            })
    });
    candidates.truncate(params.max_hidden as usize);
    let hidden_core_count = candidates.len() as u32;

    // Bilateral completeness is a **closure**, not a single pass over the ranked core: a neuron
    // pulled in as someone else's homolog can have homologs of its own that the core never
    // touched directly (e.g. a group-less core candidate's mirror-side homolog, found via the
    // type fallback, may itself have a `group` linking it to a third neuron — see the round-1
    // review finding this fixes, and `bilateral_completion_closes_a_two_hop_chain` below). A
    // worklist keeps processing newly-added neurons until nothing new turns up.
    let mut selected: BTreeSet<u32> = candidates.iter().copied().collect();
    let mut added_via_group = 0u32;
    let mut added_via_fallback = 0u32;
    let mut worklist: VecDeque<u32> = candidates.iter().copied().collect();
    while let Some(idx) = worklist.pop_front() {
        let (homologs, used_fallback) = homologs_of(tables, idx, traced_id);
        for h in homologs {
            if is_seed[h as usize] {
                continue; // already selected via a different role — never reclassify it
            }
            if selected.insert(h) {
                if used_fallback {
                    added_via_fallback += 1;
                } else {
                    added_via_group += 1;
                }
                worklist.push_back(h); // propagate closure: h's own homologs must be visited too
            }
        }
    }

    SelectionResult {
        hidden: selected.into_iter().collect(),
        hidden_core_count,
        bilateral_added_via_group: added_via_group,
        bilateral_added_via_type_fallback: added_via_fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{Dictionaries, Edge, EdgesTable, NeuronNt, NeuronRow, NeuronsTable, TablesHeader, TypeRow};
    use std::collections::BTreeMap;

    /// input(0) --5--> hidden_strong(1) --5--> output(2)
    /// input(0) --1--> hidden_weak(3)   --1--> output(2)     (weight 1 < theta_path -> unreachable)
    /// hidden_strong(1)'s homolog(4) shares its `group` but is otherwise disconnected -> must
    /// still be pulled in by bilateral completeness even though it has score 0 on its own.
    /// unreachable(5) has no path to/from anything.
    fn tiny_tables() -> ConnectomeTables {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec!["L".to_string(), "R".to_string()],
        };
        let types = vec![TypeRow {
            name: "T".into(),
            consensus_nt: None,
        }];
        let mk = |body_id: i64, side: Option<u16>, group: Option<i64>| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(0),
            instance: None,
            superclass: None,
            class: None,
            subclass: None,
            soma_side: side,
            group,
            ol_hex1: None,
            ol_hex2: None,
        };
        let rows = vec![
            mk(0, Some(0), None),     // 0: input
            mk(1, Some(0), Some(99)), // 1: hidden_strong, group 99
            mk(2, Some(0), None),     // 2: output
            mk(3, Some(0), None),     // 3: hidden_weak (only reachable via weak edges)
            mk(4, Some(1), Some(99)), // 4: homolog of 1 via group, R side, otherwise isolated
            mk(5, Some(0), None),     // 5: unreachable
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 1,
                weight: 5,
            },
            Edge {
                pre_idx: 1,
                post_idx: 2,
                weight: 5,
            },
            Edge {
                pre_idx: 0,
                post_idx: 3,
                weight: 1,
            },
            Edge {
                pre_idx: 3,
                post_idx: 2,
                weight: 1,
            },
        ];
        let total_input = {
            let mut t = vec![0u64; 6];
            t[1] = 5;
            t[2] = 6; // 5 (from 1) + 1 (from 3)
            t[3] = 1;
            t
        };
        ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: BTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 6],
            },
            neuron_nt: vec![NeuronNt::default(); 6],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        }
    }

    fn seeds() -> SeedSets {
        SeedSets {
            visual_input: vec![0],
            ascending_input: vec![],
            output: vec![2],
        }
    }

    #[test]
    fn strong_path_neuron_is_selected_weak_path_neuron_is_not() {
        let tables = tiny_tables();
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(&tables, &params, &seeds());
        assert!(result.hidden.contains(&1), "hidden_strong must be selected: {result:?}");
        assert!(
            !result.hidden.contains(&3),
            "hidden_weak's only paths use weight-1 edges, below theta_path=3: {result:?}"
        );
        assert!(!result.hidden.contains(&5), "unreachable neuron must never be selected");
    }

    #[test]
    fn bilateral_completeness_pulls_in_a_disconnected_group_homolog() {
        let tables = tiny_tables();
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(&tables, &params, &seeds());
        assert!(
            result.hidden.contains(&4),
            "neuron 4 shares group 99 with selected neuron 1 and must be pulled in even though \
             it has zero path-flow score on its own: {result:?}"
        );
        assert_eq!(result.bilateral_added_via_group, 1);
        assert_eq!(result.bilateral_added_via_type_fallback, 0);
    }

    #[test]
    fn max_hidden_caps_the_core_but_completion_can_exceed_it() {
        let tables = tiny_tables();
        let params = SelectionParams {
            k: 2,
            theta_path: 1, // now both strong and weak paths qualify -> 2 candidates (1, 3)
            theta_edge: 1,
            max_hidden: 1, // cap the *core* at 1
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(&tables, &params, &seeds());
        assert_eq!(result.hidden_core_count, 1, "only the top-1 candidate makes the core");
        // Neuron 1 has the strictly higher score (weight 5 vs weight 1 on every hop), so it's
        // the one kept; its homolog (4) then pushes the final hidden count above max_hidden=1.
        assert!(result.hidden.contains(&1));
        assert!(result.hidden.contains(&4));
        assert_eq!(
            result.hidden.len(),
            2,
            "completion legitimately exceeds max_hidden: {result:?}"
        );
    }

    #[test]
    fn a_neuron_already_seeded_is_never_added_as_hidden_via_completeness() {
        // Give the output neuron (2) the same group as hidden_strong (1) — completeness must not
        // try to "select" neuron 2 as hidden, since it is already an output.
        let mut tables = tiny_tables();
        tables.neurons.rows[2].group = Some(99);
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(&tables, &params, &seeds());
        assert!(
            !result.hidden.contains(&2),
            "neuron 2 is a seeded output, not hidden: {result:?}"
        );
    }

    #[test]
    fn fallback_pairing_by_type_and_mirrored_side_when_group_is_absent() {
        // Two same-type neurons on opposite sides, neither with a `group`, one selected via a
        // strong path; its homolog-by-type-and-side must be pulled in via the fallback rule.
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec!["L".to_string(), "R".to_string()],
        };
        let types = vec![TypeRow {
            name: "T".into(),
            consensus_nt: None,
        }];
        let mk = |body_id: i64, side: Option<u16>| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(0),
            instance: None,
            superclass: None,
            class: None,
            subclass: None,
            soma_side: side,
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        };
        let rows = vec![
            mk(0, Some(0)), // input
            mk(1, Some(0)), // hidden, L side, no group
            mk(2, Some(0)), // output
            mk(3, Some(1)), // hidden's homolog by type+side, R side, no group, disconnected
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 1,
                weight: 5,
            },
            Edge {
                pre_idx: 1,
                post_idx: 2,
                weight: 5,
            },
        ];
        let mut total_input = vec![0u64; 4];
        total_input[1] = 5;
        total_input[2] = 5;
        let tables = ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: BTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 4],
            },
            neuron_nt: vec![NeuronNt::default(); 4],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        };
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(
            &tables,
            &params,
            &SeedSets {
                visual_input: vec![0],
                ascending_input: vec![],
                output: vec![2],
            },
        );
        assert!(result.hidden.contains(&3), "{result:?}");
        assert_eq!(result.bilateral_added_via_type_fallback, 1);
        assert_eq!(result.bilateral_added_via_group, 0);
    }

    /// Regression for the round-1 review finding: completion must be a **fixpoint**, not a single
    /// pass over the ranked core. Chain: core candidate A (type T, side L, no group) is selected
    /// by score; its only homolog is B (type T, side R, no group) via the *fallback* rule; B is
    /// not itself in the core (score 0) and was never ranked, so the old single-pass code added B
    /// but never asked what B's own homologs are — B in fact has a `group` linking it to C (type
    /// T, side M — a group-mate is one specific matched neuron, not required to be the mirror
    /// side), which must also end up selected.
    #[test]
    fn bilateral_completion_closes_a_two_hop_chain() {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec!["L".to_string(), "M".to_string(), "R".to_string()],
        };
        let types = vec![TypeRow {
            name: "T".into(),
            consensus_nt: None,
        }];
        let mk = |body_id: i64, side: Option<u16>, group: Option<i64>| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(0),
            instance: None,
            superclass: None,
            class: None,
            subclass: None,
            soma_side: side,
            group,
            ol_hex1: None,
            ol_hex2: None,
        };
        // 0: input, 1: output, 2: A (core, L, no group), 3: B (A's fallback homolog, R, group=9,
        // otherwise disconnected), 4: C (B's group-mate, M, group=9, otherwise disconnected).
        let rows = vec![
            mk(0, Some(0), None),
            mk(1, Some(0), None),
            mk(2, Some(0), None),
            mk(3, Some(2), Some(9)),
            mk(4, Some(1), Some(9)),
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 2,
                weight: 5,
            },
            Edge {
                pre_idx: 2,
                post_idx: 1,
                weight: 5,
            },
        ];
        let mut total_input = vec![0u64; 5];
        total_input[1] = 5;
        total_input[2] = 5;
        let tables = ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: BTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 5],
            },
            neuron_nt: vec![NeuronNt::default(); 5],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        };
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(
            &tables,
            &params,
            &SeedSets {
                visual_input: vec![0],
                ascending_input: vec![],
                output: vec![1],
            },
        );
        assert!(
            result.hidden.contains(&2),
            "A (the core candidate) must be selected: {result:?}"
        );
        assert!(
            result.hidden.contains(&3),
            "B must be pulled in as A's fallback homolog: {result:?}"
        );
        assert!(
            result.hidden.contains(&4),
            "C must be pulled in as B's group-mate — this is exactly the closure a single pass \
             over the original core misses, since B was never itself ranked/scored: {result:?}"
        );
        assert_eq!(
            result.bilateral_added_via_type_fallback, 1,
            "only B came in via the fallback rule"
        );
        assert_eq!(result.bilateral_added_via_group, 1, "only C came in via the group rule");
    }

    #[test]
    fn group_homolog_rule_requires_the_same_type_not_just_the_same_group_id() {
        // Regression for the round-1 review finding: a `group` id shared across two *different*
        // types must not make them homologs of each other.
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec!["L".to_string(), "R".to_string()],
        };
        let types = vec![
            TypeRow {
                name: "T".into(),
                consensus_nt: None,
            },
            TypeRow {
                name: "OtherType".into(),
                consensus_nt: None,
            },
        ];
        let mk = |body_id: i64, type_id: u32, side: Option<u16>, group: Option<i64>| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(type_id),
            instance: None,
            superclass: None,
            class: None,
            subclass: None,
            soma_side: side,
            group,
            ol_hex1: None,
            ol_hex2: None,
        };
        // 0: input, 1: output, 2: A (core, type T, group=9), 3: same group=9 but type OtherType
        // (must NOT be pulled in), 4: same group=9 AND same type T (must be pulled in).
        let rows = vec![
            mk(0, 0, Some(0), None),
            mk(1, 0, Some(0), None),
            mk(2, 0, Some(0), Some(9)),
            mk(3, 1, Some(1), Some(9)),
            mk(4, 0, Some(1), Some(9)),
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 2,
                weight: 5,
            },
            Edge {
                pre_idx: 2,
                post_idx: 1,
                weight: 5,
            },
        ];
        let mut total_input = vec![0u64; 5];
        total_input[1] = 5;
        total_input[2] = 5;
        let tables = ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: BTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 5],
            },
            neuron_nt: vec![NeuronNt::default(); 5],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        };
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(
            &tables,
            &params,
            &SeedSets {
                visual_input: vec![0],
                ascending_input: vec![],
                output: vec![1],
            },
        );
        assert!(result.hidden.contains(&2));
        assert!(
            !result.hidden.contains(&3),
            "different-typed group-mate must not be pulled in: {result:?}"
        );
        assert!(
            result.hidden.contains(&4),
            "same-typed group-mate must be pulled in: {result:?}"
        );
        assert_eq!(result.bilateral_added_via_group, 1);
    }

    /// F4 (review round 1): with a hop budget `k`, a neuron whose shortest input-to-output path
    /// through it has length exactly `k` must be selected, and one whose shortest path through it
    /// has length `k+1` (one hop beyond budget) must not — even on a graph that offers no shorter
    /// alternative. Two parallel paths from the same input to the same output: `0->1->4` (length
    /// 2, fits `k=2` exactly) and `0->2->3->4` (length 3, one hop over budget) — node 1 must be
    /// selected, nodes 2 and 3 must not, even though 2 and 3 are each "only" 1 or 2 hops from a
    /// seed individually (the *combined* input+output distance through them is what must be
    /// checked, not either distance alone).
    #[test]
    fn neuron_one_hop_beyond_k_is_excluded() {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec![],
        };
        let types = vec![TypeRow {
            name: "T".into(),
            consensus_nt: None,
        }];
        let mk = |body_id: i64| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(0),
            instance: None,
            superclass: None,
            class: None,
            subclass: None,
            soma_side: None,
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        };
        // 0 = input, 4 = output; 1 on the length-2 path, 2/3 on the length-3 path.
        let rows = vec![mk(0), mk(1), mk(2), mk(3), mk(4)];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 1,
                weight: 5,
            },
            Edge {
                pre_idx: 1,
                post_idx: 4,
                weight: 5,
            },
            Edge {
                pre_idx: 0,
                post_idx: 2,
                weight: 5,
            },
            Edge {
                pre_idx: 2,
                post_idx: 3,
                weight: 5,
            },
            Edge {
                pre_idx: 3,
                post_idx: 4,
                weight: 5,
            },
        ];
        let mut total_input = vec![0u64; 5];
        total_input[1] = 5;
        total_input[2] = 5;
        total_input[3] = 5;
        total_input[4] = 10; // node 4 receives from both 1->4 and 3->4
        let tables = ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: BTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 5],
            },
            neuron_nt: vec![NeuronNt::default(); 5],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        };
        let params = SelectionParams {
            k: 2,
            theta_path: 3,
            theta_edge: 1,
            max_hidden: 10,
            weak_pair_threshold: 20,
            an_top_n_types: 1,
        };
        let result = select_hidden(
            &tables,
            &params,
            &SeedSets {
                visual_input: vec![0],
                ascending_input: vec![],
                output: vec![4],
            },
        );
        assert!(
            result.hidden.contains(&1),
            "node 1 (length-2 path, fits k=2 exactly) must be selected: {result:?}"
        );
        assert!(
            !result.hidden.contains(&2) && !result.hidden.contains(&3),
            "nodes 2 and 3 (length-3 path, one hop over budget k=2) must be excluded: {result:?}"
        );
    }
}
