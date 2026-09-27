//! Structural validation for a [`Flyg`] value: every index in range, CSR well-formed, no NaN, no
//! autapses, and the baked-in [`Summary`] consistent with the rest of the file. Used both by
//! [`crate::io::load`] (so a corrupted or hand-edited file is rejected with a clear reason instead
//! of panicking or silently misbehaving downstream) and by the builder before it writes a file at
//! all.

use std::collections::HashMap;

use crate::error::ValidationError;
use crate::format::{Flyg, NeuronRole, RoleCounts, Sign, SignCounts};

fn err(msg: impl Into<String>) -> ValidationError {
    ValidationError(msg.into())
}

/// Validates `flyg`'s internal consistency and returns the *first* problem found. Validation
/// stops at the first error rather than collecting all of them: once one structural invariant is
/// broken (e.g. an out-of-range index), later checks that index into the same data can no longer
/// tell a real second problem apart from downstream noise caused by the first one — so a single,
/// concrete, reproducible failure is more useful here than a best-effort list.
pub fn validate(flyg: &Flyg) -> Result<(), ValidationError> {
    let n = flyg.neurons.len();
    let t = flyg.types.len();
    let p = flyg.type_pairs.len();

    if flyg.header.format_version != crate::format::FLYG_FORMAT_VERSION {
        return Err(err(format!(
            "header.format_version {} != expected {}",
            flyg.header.format_version,
            crate::format::FLYG_FORMAT_VERSION
        )));
    }

    // --- neurons -------------------------------------------------------------------------------
    let mut role_counts = RoleCounts::default();
    let mut type_neuron_count = vec![0u32; t];
    for (i, nrow) in flyg.neurons.iter().enumerate() {
        if nrow.type_index as usize >= t {
            return Err(err(format!(
                "neurons[{i}].type_index {} out of range (types.len() == {t})",
                nrow.type_index
            )));
        }
        type_neuron_count[nrow.type_index as usize] += 1;

        match nrow.role {
            NeuronRole::InputVisual => {
                role_counts.input_visual += 1;
                if nrow.rf.is_none() {
                    return Err(err(format!("neurons[{i}] is InputVisual but has no receptive field")));
                }
            }
            NeuronRole::InputAscending => role_counts.input_ascending += 1,
            NeuronRole::Hidden => role_counts.hidden += 1,
            NeuronRole::Output => role_counts.output += 1,
        }
        if !matches!(nrow.role, NeuronRole::InputVisual) && nrow.rf.is_some() {
            return Err(err(format!(
                "neurons[{i}] has role {:?} but carries a receptive field (only InputVisual should)",
                nrow.role
            )));
        }
        if let Some(rf) = nrow.rf
            && (!rf.azimuth_deg.is_finite() || !rf.elevation_deg.is_finite())
        {
            return Err(err(format!(
                "neurons[{i}].rf contains a non-finite value (NaN or ±inf): {rf:?}"
            )));
        }
    }

    // --- types -----------------------------------------------------------------------------------
    let mut sign_counts = SignCounts::default();
    let mut uncertain_types = 0u32;
    for (i, trow) in flyg.types.iter().enumerate() {
        if trow.nt_confidence.is_nan() {
            return Err(err(format!("types[{i}].nt_confidence is NaN")));
        }
        if !(0.0..=1.0).contains(&trow.nt_confidence) {
            return Err(err(format!(
                "types[{i}].nt_confidence {} outside [0, 1]",
                trow.nt_confidence
            )));
        }
        if trow.neuron_count != type_neuron_count[i] {
            return Err(err(format!(
                "types[{i}].neuron_count {} != actual neuron count {} in `neurons`",
                trow.neuron_count, type_neuron_count[i]
            )));
        }
        match trow.sign {
            Sign::Excitatory => sign_counts.excitatory += 1,
            Sign::Inhibitory => sign_counts.inhibitory += 1,
            Sign::Neutral => sign_counts.neutral += 1,
        }
        if trow.uncertain {
            uncertain_types += 1;
        }
    }

    // --- edges (CSR) -----------------------------------------------------------------------------
    let edges = &flyg.edges;
    if edges.row_start.len() != n + 1 {
        return Err(err(format!(
            "edges.row_start.len() {} != neurons.len()+1 {}",
            edges.row_start.len(),
            n + 1
        )));
    }
    if edges.row_start.first() != Some(&0) {
        return Err(err("edges.row_start[0] != 0"));
    }
    let nnz = edges.pre_index.len();
    if edges.synapse_count.len() != nnz || edges.type_pair_index.len() != nnz {
        return Err(err(format!(
            "edges' parallel arrays have mismatched lengths: pre_index={nnz}, synapse_count={}, type_pair_index={}",
            edges.synapse_count.len(),
            edges.type_pair_index.len()
        )));
    }
    if edges.row_start.last() != Some(&(nnz as u32)) {
        return Err(err(format!(
            "edges.row_start's last entry {:?} != pre_index.len() {nnz}",
            edges.row_start.last()
        )));
    }
    for w in edges.row_start.windows(2) {
        if w[0] > w[1] {
            return Err(err(format!(
                "edges.row_start is not non-decreasing: {} > {}",
                w[0], w[1]
            )));
        }
    }
    for post in 0..n {
        let start = edges.row_start[post] as usize;
        let end = edges.row_start[post + 1] as usize;
        let mut prev_pre: Option<u32> = None;
        for idx in start..end {
            let pre = edges.pre_index[idx];
            if pre as usize >= n {
                return Err(err(format!(
                    "edges row for post={post}: pre_index {pre} out of range (neurons.len() == {n})"
                )));
            }
            if pre as usize == post {
                return Err(err(format!("edges row for post={post}: autapse (pre == post) present")));
            }
            if let Some(prev) = prev_pre
                && prev >= pre
            {
                return Err(err(format!(
                    "edges row for post={post}: pre_index not strictly ascending ({prev} then {pre})"
                )));
            }
            prev_pre = Some(pre);
            if edges.synapse_count[idx] == 0 {
                return Err(err(format!("edges row for post={post}, pre={pre}: synapse_count is 0")));
            }
            let tp = edges.type_pair_index[idx];
            if tp as usize >= p {
                return Err(err(format!(
                    "edges row for post={post}, pre={pre}: type_pair_index {tp} out of range (type_pairs.len() == {p})"
                )));
            }
            // The referenced type_pair's (pre_type, post_type) must match this edge's actual
            // endpoints' types — otherwise `type_pair_index` could silently point at the wrong
            // pair's `total_synapses`/`shared_param_id` (review round 1, F5).
            let pair = &flyg.type_pairs[tp as usize];
            let pre_type = flyg.neurons[pre as usize].type_index;
            let post_type = flyg.neurons[post].type_index;
            if pair.pre_type != pre_type || pair.post_type != post_type {
                return Err(err(format!(
                    "edges row for post={post}, pre={pre}: type_pair_index {tp} is ({}, {}), but this edge's \
                     actual types are ({pre_type}, {post_type})",
                    pair.pre_type, pair.post_type
                )));
            }
        }
    }

    // --- per-neuron totals -----------------------------------------------------------------------
    if flyg.neuron_input_totals.full_connectome.len() != n || flyg.neuron_input_totals.in_subgraph.len() != n {
        return Err(err(format!(
            "neuron_input_totals arrays must have length {n} (full_connectome={}, in_subgraph={})",
            flyg.neuron_input_totals.full_connectome.len(),
            flyg.neuron_input_totals.in_subgraph.len()
        )));
    }
    for i in 0..n {
        if flyg.neuron_input_totals.in_subgraph[i] > flyg.neuron_input_totals.full_connectome[i] {
            return Err(err(format!(
                "neuron_input_totals[{i}]: in_subgraph ({}) > full_connectome ({})",
                flyg.neuron_input_totals.in_subgraph[i], flyg.neuron_input_totals.full_connectome[i]
            )));
        }
    }

    // --- type_pairs --------------------------------------------------------------------------
    for (i, row) in flyg.type_pairs.iter().enumerate() {
        if row.pre_type as usize >= t {
            return Err(err(format!("type_pairs[{i}].pre_type {} out of range", row.pre_type)));
        }
        if row.post_type as usize >= t {
            return Err(err(format!("type_pairs[{i}].post_type {} out of range", row.post_type)));
        }
        if row.total_synapses == 0 {
            return Err(err(format!("type_pairs[{i}].total_synapses is 0")));
        }
    }
    let mut distinct_shared_ids: HashMap<u32, ()> = HashMap::new();
    for row in &flyg.type_pairs {
        distinct_shared_ids.insert(row.shared_param_id, ());
    }

    // --- input_channels ------------------------------------------------------------------------
    for (i, ch) in flyg.input_channels.iter().enumerate() {
        if ch.type_index as usize >= t {
            return Err(err(format!(
                "input_channels[{i}].type_index {} out of range",
                ch.type_index
            )));
        }
        // Acceptance criterion: input channels are "per visual input type" specifically — an
        // ascending (proprioceptive) type is not a valid target for this mapping.
        let has_visual_input_neuron = flyg
            .neurons
            .iter()
            .any(|nr| nr.type_index == ch.type_index && matches!(nr.role, NeuronRole::InputVisual));
        if !has_visual_input_neuron {
            return Err(err(format!(
                "input_channels[{i}] references type_index {} which has no InputVisual neuron",
                ch.type_index
            )));
        }
    }

    // --- output_groups -------------------------------------------------------------------------
    for (gi, group) in flyg.output_groups.iter().enumerate() {
        for (mi, member) in group.members.iter().enumerate() {
            if member.neuron_index as usize >= n {
                return Err(err(format!(
                    "output_groups[{gi}].members[{mi}].neuron_index {} out of range",
                    member.neuron_index
                )));
            }
            if !matches!(flyg.neurons[member.neuron_index as usize].role, NeuronRole::Output) {
                return Err(err(format!(
                    "output_groups[{gi}].members[{mi}] points at neuron {} whose role is not Output",
                    member.neuron_index
                )));
            }
        }
    }

    // --- summary cross-check ---------------------------------------------------------------------
    let s = &flyg.summary;
    if s.neurons_by_role != role_counts {
        return Err(err(format!(
            "summary.neurons_by_role {:?} != actual {role_counts:?}",
            s.neurons_by_role
        )));
    }
    if s.num_types as usize != t {
        return Err(err(format!("summary.num_types {} != types.len() {t}", s.num_types)));
    }
    if s.num_edges as usize != nnz {
        return Err(err(format!(
            "summary.num_edges {} != actual edge count {nnz}",
            s.num_edges
        )));
    }
    if s.num_type_pairs as usize != p {
        return Err(err(format!(
            "summary.num_type_pairs {} != type_pairs.len() {p}",
            s.num_type_pairs
        )));
    }
    if s.shared_param_count as usize != distinct_shared_ids.len() {
        return Err(err(format!(
            "summary.shared_param_count {} != actual distinct shared_param_id count {}",
            s.shared_param_count,
            distinct_shared_ids.len()
        )));
    }
    if s.sign_counts != sign_counts {
        return Err(err(format!(
            "summary.sign_counts {:?} != actual {sign_counts:?}",
            s.sign_counts
        )));
    }
    if s.uncertain_types != uncertain_types {
        return Err(err(format!(
            "summary.uncertain_types {} != actual {uncertain_types}",
            s.uncertain_types
        )));
    }
    let actual_rf_fallback = flyg
        .neurons
        .iter()
        .filter(|nr| nr.rf.is_some_and(|rf| rf.is_fallback))
        .count() as u32;
    if s.rf_fallback_count != actual_rf_fallback {
        return Err(err(format!(
            "summary.rf_fallback_count {} != actual {actual_rf_fallback}",
            s.rf_fallback_count
        )));
    }

    Ok(())
}
