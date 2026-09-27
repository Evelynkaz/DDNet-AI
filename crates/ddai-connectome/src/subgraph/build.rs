//! Orchestrates the whole `build-subgraph` pipeline: resolve configured type names, seed
//! inputs/outputs, pick AN types, select hidden neurons, compute signs and receptive fields,
//! build the induced subgraph and type-pair table, and assemble a validated
//! [`ddai_flyg::Flyg`]. See this crate's README for the algorithm in prose.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ddai_flyg::{
    Flyg, FlygEdges, FlygHeader, FlygNeuron, FlygType, InputChannelMapping, NeuronInputTotals, NeuronRole, OutputGroup,
    OutputMember, RoleCounts, Side, Sign, SignCounts, Summary, TypePair,
};

use super::an_pick::{AnTypeScore, pick_an_types};
use super::common::neuron_side;
use super::config::{SideFilter, SubgraphConfig, load_config};
use super::rf::{RfStats, compute_receptive_fields};
use super::select::{SeedSets, select_hidden};
use super::signs::compute_type_signs;
use crate::hashing::hash_file;
use crate::tables::{ConnectomeTables, load_tables_file, resolve_tables_file_path};

/// Everything `report::render` needs beyond what's already in the built [`Flyg`] itself.
#[derive(Debug, Clone, Default)]
pub struct BuildReport {
    pub missing_visual_types: Vec<String>,
    pub missing_dn_types: Vec<String>,
    pub missing_input_channel_types: Vec<String>,
    pub missing_output_group_types: Vec<String>,
    pub an_candidates_considered: usize,
    pub an_types_picked: Vec<AnTypeScore>,
    pub rf_stats: RfStats,
    pub hidden_core_count: u32,
    pub bilateral_added_via_group: u32,
    pub bilateral_added_via_type_fallback: u32,
    pub uncertain_type_names: Vec<String>,
    pub hidden_dead_end_no_input: u32,
    pub hidden_dead_end_no_output: u32,
}

fn build_type_name_index(tables: &ConnectomeTables) -> BTreeMap<&str, u32> {
    tables
        .types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i as u32))
        .collect()
}

fn resolve_names(index: &BTreeMap<&str, u32>, names: &[String]) -> (Vec<u32>, Vec<String>) {
    let mut found = Vec::new();
    let mut missing = Vec::new();
    for name in names {
        match index.get(name.as_str()) {
            Some(&id) => found.push(id),
            None => missing.push(name.clone()),
        }
    }
    (found, missing)
}

/// Dense indices (ascending — same order as `bodyId`, per `tables.rs`'s own invariant) of every
/// Traced neuron whose type is in `type_ids`.
fn neurons_of_types(tables: &ConnectomeTables, type_ids: &BTreeSet<u32>, traced_id: Option<u16>) -> Vec<u32> {
    tables
        .neurons
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            r.status == traced_id && traced_id.is_some() && r.type_id.is_some_and(|t| type_ids.contains(&t))
        })
        .map(|(i, _)| i as u32)
        .collect()
}

fn name_or_empty(dict: &[String], id: Option<u16>) -> String {
    id.map(|i| dict[i as usize].clone()).unwrap_or_default()
}

/// Builds the `Flyg` value (no file I/O, no hashing — those are [`run_build_subgraph`]'s job, so
/// this function stays unit-testable against hand-built [`ConnectomeTables`] fixtures).
pub fn build_flyg(
    tables: &ConnectomeTables,
    config: &SubgraphConfig,
    source_tables_sha256: String,
    config_sha256: String,
) -> Result<(Flyg, BuildReport)> {
    let traced_id = tables
        .dictionaries
        .statuses
        .iter()
        .position(|s| s == "Traced")
        .map(|i| i as u16);
    let name_index = build_type_name_index(tables);

    // --- seeds -----------------------------------------------------------------------------
    let (visual_type_ids, missing_visual_types) = resolve_names(&name_index, &config.inputs.visual_types);
    let (dn_type_ids, missing_dn_types) = resolve_names(&name_index, &config.outputs.dn_types);
    if visual_type_ids.is_empty() {
        bail!(
            "none of the configured inputs.visual_types resolved against the tables (missing: {missing_visual_types:?})"
        );
    }
    if dn_type_ids.is_empty() {
        bail!("none of the configured outputs.dn_types resolved against the tables (missing: {missing_dn_types:?})");
    }

    let visual_set: BTreeSet<u32> = visual_type_ids.into_iter().collect();
    let dn_set: BTreeSet<u32> = dn_type_ids.into_iter().collect();
    let visual_input = neurons_of_types(tables, &visual_set, traced_id);
    let output = neurons_of_types(tables, &dn_set, traced_id);
    if visual_input.is_empty() {
        bail!("configured inputs.visual_types resolved but produced zero Traced neurons");
    }
    if output.is_empty() {
        bail!("configured outputs.dn_types resolved but produced zero Traced neurons");
    }

    let an_types_picked = pick_an_types(tables, &config.selection, &visual_input, &output);
    let an_set: BTreeSet<u32> = an_types_picked.iter().map(|s| s.type_id).collect();
    let ascending_input = neurons_of_types(tables, &an_set, traced_id);

    // --- hidden-neuron selection --------------------------------------------------------------
    let seeds = SeedSets {
        visual_input: visual_input.clone(),
        ascending_input: ascending_input.clone(),
        output: output.clone(),
    };
    let selection = select_hidden(tables, &config.selection, &seeds);

    // --- assemble the selected-neuron set (roles, local ordering) ------------------------------
    let mut role_by_global: BTreeMap<u32, NeuronRole> = BTreeMap::new();
    for &i in &visual_input {
        role_by_global.insert(i, NeuronRole::InputVisual);
    }
    for &i in &ascending_input {
        role_by_global.insert(i, NeuronRole::InputAscending);
    }
    for &i in &selection.hidden {
        role_by_global.insert(i, NeuronRole::Hidden);
    }
    for &i in &output {
        role_by_global.insert(i, NeuronRole::Output);
    }
    // `BTreeMap::keys()` is already sorted ascending, which — per `tables.rs`'s own invariant
    // that the dense neuron index is bodyId-sorted — is exactly the deterministic ordering the
    // `.flyg` neuron table wants.
    let all_selected: Vec<u32> = role_by_global.keys().copied().collect();
    let n_local = all_selected.len();
    let global_to_local: BTreeMap<u32, u32> = all_selected.iter().enumerate().map(|(l, &g)| (g, l as u32)).collect();

    for &g in &all_selected {
        if tables.neurons.rows[g as usize].type_id.is_none() {
            bail!(
                "internal error: selected neuron bodyId {} has no type (inputs/outputs are seeded by type \
                 name and hidden candidates are filtered to typed neurons — this should be unreachable)",
                tables.neurons.rows[g as usize].body_id
            );
        }
    }

    // --- receptive fields (visual inputs only) --------------------------------------------------
    let (rf_map, rf_stats) = compute_receptive_fields(tables, &visual_input, config.rf.min_hex_support);

    // --- types present in the subgraph, sorted by name for a deterministic, readable table -----
    let all_signs = compute_type_signs(tables, &config.nt);
    let mut local_type_ids: Vec<u32> = {
        let set: BTreeSet<u32> = all_selected
            .iter()
            .map(|&g| tables.neurons.rows[g as usize].type_id.unwrap())
            .collect();
        set.into_iter().collect()
    };
    local_type_ids.sort_by(|&a, &b| tables.types[a as usize].name.cmp(&tables.types[b as usize].name));
    let global_type_to_local: BTreeMap<u32, u32> =
        local_type_ids.iter().enumerate().map(|(l, &g)| (g, l as u32)).collect();

    let mut type_neuron_count: BTreeMap<u32, u32> = BTreeMap::new();
    let mut type_first_neuron: BTreeMap<u32, u32> = BTreeMap::new();
    for &g in &all_selected {
        let t = tables.neurons.rows[g as usize].type_id.unwrap();
        *type_neuron_count.entry(t).or_insert(0) += 1;
        type_first_neuron.entry(t).or_insert(g);
    }

    let mut flyg_types = Vec::with_capacity(local_type_ids.len());
    let mut uncertain_type_names = Vec::new();
    for &global_type_id in &local_type_ids {
        let trow = &tables.types[global_type_id as usize];
        let sign_info = all_signs[global_type_id as usize];
        let first_neuron = type_first_neuron[&global_type_id];
        let first_row = &tables.neurons.rows[first_neuron as usize];
        if sign_info.uncertain {
            uncertain_type_names.push(trow.name.clone());
        }
        flyg_types.push(FlygType {
            name: trow.name.clone(),
            superclass: name_or_empty(&tables.dictionaries.superclasses, first_row.superclass),
            class: name_or_empty(&tables.dictionaries.classes, first_row.class),
            nt_class_used: sign_info.nt_class_used,
            sign: sign_info.sign,
            nt_confidence: sign_info.confidence,
            uncertain: sign_info.uncertain,
            neuron_count: type_neuron_count[&global_type_id],
        });
    }

    // --- neurons table ---------------------------------------------------------------------------
    let mut flyg_neurons = Vec::with_capacity(n_local);
    for &g in &all_selected {
        let row = &tables.neurons.rows[g as usize];
        let role = role_by_global[&g];
        let local_type = global_type_to_local[&row.type_id.unwrap()];
        let side = neuron_side(tables, g);
        let rf = if matches!(role, NeuronRole::InputVisual) {
            rf_map.get(&g).copied()
        } else {
            None
        };
        flyg_neurons.push(FlygNeuron {
            body_id: row.body_id,
            type_index: local_type,
            role,
            side,
            group_id: row.group,
            rf,
        });
    }

    // --- induced-subgraph edges + type-pair totals ---------------------------------------------
    struct RawEdge {
        local_pre: u32,
        local_post: u32,
        weight: u32,
        global_pre_type: u32,
        global_post_type: u32,
    }
    let theta_edge = config.selection.theta_edge;
    let mut raw_edges: Vec<RawEdge> = Vec::new();
    let mut pair_totals: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for e in &tables.edges.edges {
        if e.weight < theta_edge {
            continue;
        }
        let (Some(&local_pre), Some(&local_post)) = (global_to_local.get(&e.pre_idx), global_to_local.get(&e.post_idx))
        else {
            continue;
        };
        let pre_type = tables.neurons.rows[e.pre_idx as usize].type_id.unwrap();
        let post_type = tables.neurons.rows[e.post_idx as usize].type_id.unwrap();
        *pair_totals.entry((pre_type, post_type)).or_insert(0) += u64::from(e.weight);
        raw_edges.push(RawEdge {
            local_pre,
            local_post,
            weight: e.weight,
            global_pre_type: pre_type,
            global_post_type: post_type,
        });
    }
    // The CSR fill below places each edge by *position*, not by looking up its own `local_post`
    // against `row_start` — so it requires `raw_edges` to already be grouped by `local_post`
    // ascending (and, within a row, `local_pre` ascending — `validate()` checks the latter).
    // `tables.edges.edges` (the real `connectome.tables`, built by `build_tables_from_raw`) is
    // already sorted by `(post_idx, pre_idx)`, and filtering/relabeling through a strictly
    // increasing map (`global_to_local`) preserves that order — but sorting explicitly here,
    // rather than relying on that as an unenforced input invariant, means this is correct even
    // for a `ConnectomeTables` some other caller (or a hand-built test fixture) didn't sort.
    raw_edges.sort_by_key(|re| (re.local_post, re.local_pre));

    let mut pair_rows: Vec<(u32, u32, u64)> = pair_totals
        .iter()
        .map(|(&(gpre, gpost), &total)| (global_type_to_local[&gpre], global_type_to_local[&gpost], total))
        .collect();
    pair_rows.sort_by_key(|&(lpre, lpost, _)| (lpre, lpost));

    let weak_threshold = config.selection.weak_pair_threshold;
    let mut shared_id_by_pair_idx: Vec<u32> = Vec::with_capacity(pair_rows.len());
    let mut weak_group_ids: BTreeMap<(i8, String, String), u32> = BTreeMap::new();
    let mut next_id = 0u32;
    for &(lpre, lpost, total) in &pair_rows {
        let id = if total >= weak_threshold {
            let id = next_id;
            next_id += 1;
            id
        } else {
            let pre_sign = flyg_types[lpre as usize].sign.as_i8();
            let pre_superclass = flyg_types[lpre as usize].superclass.clone();
            let post_superclass = flyg_types[lpost as usize].superclass.clone();
            *weak_group_ids
                .entry((pre_sign, pre_superclass, post_superclass))
                .or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
        };
        shared_id_by_pair_idx.push(id);
    }
    let type_pairs: Vec<TypePair> = pair_rows
        .iter()
        .zip(&shared_id_by_pair_idx)
        .map(|(&(lpre, lpost, total), &sid)| TypePair {
            pre_type: lpre,
            post_type: lpost,
            total_synapses: total,
            shared_param_id: sid,
        })
        .collect();
    let pair_index_lookup: BTreeMap<(u32, u32), u32> = pair_rows
        .iter()
        .enumerate()
        .map(|(i, &(lpre, lpost, _))| ((lpre, lpost), i as u32))
        .collect();

    let mut row_start = vec![0u32; n_local + 1];
    for re in &raw_edges {
        row_start[re.local_post as usize + 1] += 1;
    }
    for i in 0..n_local {
        row_start[i + 1] += row_start[i];
    }
    let nnz = raw_edges.len();
    let mut pre_index = vec![0u32; nnz];
    let mut synapse_count = vec![0u32; nnz];
    let mut type_pair_index = vec![0u32; nnz];
    for (i, re) in raw_edges.iter().enumerate() {
        pre_index[i] = re.local_pre;
        synapse_count[i] = re.weight;
        let lpre_type = global_type_to_local[&re.global_pre_type];
        let lpost_type = global_type_to_local[&re.global_post_type];
        type_pair_index[i] = pair_index_lookup[&(lpre_type, lpost_type)];
    }

    // --- per-neuron totals -----------------------------------------------------------------------
    let mut full_connectome = vec![0u64; n_local];
    for (local_idx, &global_idx) in all_selected.iter().enumerate() {
        full_connectome[local_idx] = tables.neurons.total_input[global_idx as usize];
    }
    let mut in_subgraph = vec![0u64; n_local];
    let mut out_degree_in_subgraph = vec![0u32; n_local];
    for re in &raw_edges {
        in_subgraph[re.local_post as usize] += u64::from(re.weight);
        out_degree_in_subgraph[re.local_pre as usize] += 1;
    }

    // --- dead ends (review round 1 residual): hidden neurons with no in-subgraph input or no
    // in-subgraph output edge at all — a hidden neuron the model would only ever see silence
    // from, or that never feeds anything else in this subgraph. Not an error (inputs/outputs are
    // whole-type seeds, so some hidden relay legitimately being a pure "sink"/"source" *within
    // the subgraph* while it has real partners outside it is expected), purely a reported metric.
    let mut hidden_dead_end_no_input = 0u32;
    let mut hidden_dead_end_no_output = 0u32;
    for (local_idx, &global_idx) in all_selected.iter().enumerate() {
        if role_by_global[&global_idx] != NeuronRole::Hidden {
            continue;
        }
        if in_subgraph[local_idx] == 0 {
            hidden_dead_end_no_input += 1;
        }
        if out_degree_in_subgraph[local_idx] == 0 {
            hidden_dead_end_no_output += 1;
        }
    }

    // --- input channels (visual types only — see ddai-flyg's validate.rs) ----------------------
    let mut input_channels = Vec::new();
    let mut missing_input_channel_types = Vec::new();
    for ic in &config.input_channels {
        let resolved = name_index
            .get(ic.type_name.as_str())
            .and_then(|g| global_type_to_local.get(g))
            .filter(|&&local_type_id| {
                flyg_neurons
                    .iter()
                    .any(|n| n.type_index == local_type_id && matches!(n.role, NeuronRole::InputVisual))
            });
        match resolved {
            Some(&local_type_id) => input_channels.push(InputChannelMapping {
                type_index: local_type_id,
                channels: ic.channels.clone(),
            }),
            None => missing_input_channel_types.push(ic.type_name.clone()),
        }
    }
    input_channels.sort_by_key(|c| c.type_index);

    // --- output groups -----------------------------------------------------------------------
    let mut output_groups = Vec::new();
    let mut missing_output_group_types = Vec::new();
    for og in &config.output_groups {
        let mut members = Vec::new();
        for ty in &og.types {
            let Some(&local_type_id) = name_index.get(ty.as_str()).and_then(|g| global_type_to_local.get(g)) else {
                missing_output_group_types.push(ty.clone());
                continue;
            };
            for (local_idx, n) in flyg_neurons.iter().enumerate() {
                if n.type_index != local_type_id || !matches!(n.role, NeuronRole::Output) {
                    continue;
                }
                let side_ok = match og.side {
                    SideFilter::Both => true,
                    SideFilter::L => matches!(n.side, Side::L),
                    SideFilter::R => matches!(n.side, Side::R),
                };
                if side_ok {
                    members.push(OutputMember {
                        neuron_index: local_idx as u32,
                        side: n.side,
                    });
                }
            }
        }
        members.sort_by_key(|m| m.neuron_index);
        output_groups.push(OutputGroup {
            action: og.action.clone(),
            members,
        });
    }

    // --- summary -------------------------------------------------------------------------------
    let mut role_counts = RoleCounts::default();
    for n in &flyg_neurons {
        match n.role {
            NeuronRole::InputVisual => role_counts.input_visual += 1,
            NeuronRole::InputAscending => role_counts.input_ascending += 1,
            NeuronRole::Hidden => role_counts.hidden += 1,
            NeuronRole::Output => role_counts.output += 1,
        }
    }
    let mut sign_counts = SignCounts::default();
    let mut uncertain_types = 0u32;
    for t in &flyg_types {
        match t.sign {
            Sign::Excitatory => sign_counts.excitatory += 1,
            Sign::Inhibitory => sign_counts.inhibitory += 1,
            Sign::Neutral => sign_counts.neutral += 1,
        }
        if t.uncertain {
            uncertain_types += 1;
        }
    }
    let distinct_shared: BTreeSet<u32> = type_pairs.iter().map(|p| p.shared_param_id).collect();
    let rf_fallback_count = flyg_neurons
        .iter()
        .filter(|n| n.rf.is_some_and(|rf| rf.is_fallback))
        .count() as u32;

    let summary = Summary {
        neurons_by_role: role_counts,
        num_types: flyg_types.len() as u32,
        num_edges: raw_edges.len() as u32,
        num_type_pairs: type_pairs.len() as u32,
        shared_param_count: distinct_shared.len() as u32,
        sign_counts,
        uncertain_types,
        rf_fallback_count,
    };

    let flyg = Flyg {
        header: FlygHeader {
            format_version: ddai_flyg::FLYG_FORMAT_VERSION,
            source_tables_sha256,
            config_sha256,
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
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
            full_connectome,
            in_subgraph,
        },
        type_pairs,
        input_channels,
        output_groups,
        summary,
    };
    ddai_flyg::validate(&flyg).context("the subgraph builder produced an invalid .flyg (this is a bug)")?;

    let report = BuildReport {
        missing_visual_types,
        missing_dn_types,
        missing_input_channel_types,
        missing_output_group_types,
        an_candidates_considered: an_types_picked.len(),
        an_types_picked,
        rf_stats,
        hidden_core_count: selection.hidden_core_count,
        bilateral_added_via_group: selection.bilateral_added_via_group,
        bilateral_added_via_type_fallback: selection.bilateral_added_via_type_fallback,
        uncertain_type_names,
        hidden_dead_end_no_input,
        hidden_dead_end_no_output,
    };
    Ok((flyg, report))
}

pub struct BuildSubgraphSummary {
    pub flyg_path: PathBuf,
    pub report_path: PathBuf,
    pub flyg_sha256: String,
    pub neurons_total: usize,
    pub role_counts: RoleCounts,
    pub num_edges: usize,
    pub wall_time: Duration,
    pub peak_rss_kb: Option<u64>,
}

/// Reads `/proc/self/status`'s `VmHWM` (peak resident set size) — Linux-specific, but this
/// project only ever runs on Linux (see `CLAUDE.md`'s hardware section). Returns `None` rather
/// than erroring if unavailable (e.g. a non-Linux CI runner, or a sandboxed environment where
/// `/proc` is hidden), since it's purely a reported metric, not something correctness depends on.
fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.trim().trim_end_matches(" kB").trim().parse().ok();
        }
    }
    None
}

/// Runs `build-subgraph` end to end: load tables + config, build and validate the `.flyg`, hash
/// everything for the header, save it, render and write the Markdown report.
pub fn run_build_subgraph(
    tables_arg: &Path,
    config_path: &Path,
    out_path: &Path,
    report_path: &Path,
) -> Result<BuildSubgraphSummary> {
    let started = Instant::now();
    let tables_file = resolve_tables_file_path(tables_arg);
    let tables =
        load_tables_file(&tables_file).with_context(|| format!("loading tables from {}", tables_file.display()))?;
    let config = load_config(config_path)?;

    let source_tables_sha256 = hash_file(&tables_file)
        .with_context(|| format!("hashing {}", tables_file.display()))?
        .sha256_hex;
    let config_sha256 = hash_file(config_path)
        .with_context(|| format!("hashing {}", config_path.display()))?
        .sha256_hex;

    let (flyg, report) = build_flyg(&tables, &config, source_tables_sha256, config_sha256)?;

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    ddai_flyg::save(&flyg, out_path).with_context(|| format!("writing {}", out_path.display()))?;
    let flyg_sha256 = hash_file(out_path)
        .with_context(|| format!("hashing {}", out_path.display()))?
        .sha256_hex;

    let wall_time = started.elapsed();
    let peak_rss_kb = peak_rss_kb();

    let markdown = super::report::render(&flyg, &report, &config, &flyg_sha256, wall_time, peak_rss_kb);
    if let Some(parent) = report_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(report_path, markdown).with_context(|| format!("writing {}", report_path.display()))?;

    Ok(BuildSubgraphSummary {
        flyg_path: out_path.to_path_buf(),
        report_path: report_path.to_path_buf(),
        flyg_sha256,
        neurons_total: flyg.neurons.len(),
        role_counts: flyg.summary.neurons_by_role,
        num_edges: flyg.edges.num_edges(),
        wall_time,
        peak_rss_kb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subgraph::config::{InputsConfig, NtSignParams, OutputsConfig, SelectionParams};
    use crate::tables::{
        Dictionaries, Edge, EdgesTable, NeuronNt, NeuronRow, NeuronsTable, NtClass, TablesHeader, TypeRow,
    };
    use std::collections::BTreeMap as StdBTreeMap;

    /// A small but realistic-shaped connectome: 2 visual input types (L/R each), 1 AN type (only
    /// findable by the data-driven rule, not named in config), 1 hidden type reachable within 2
    /// hops, 1 output DN type (L/R). Exercises the whole pipeline end to end.
    fn tiny_tables() -> ConnectomeTables {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![
                "ascending_neuron".to_string(),
                "central".to_string(),
                "descending_neuron".to_string(),
                "visual_projection".to_string(),
            ],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec!["L".to_string(), "R".to_string()],
        };
        let types = vec![
            TypeRow {
                name: "AN_X".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
            TypeRow {
                name: "Central1".into(),
                consensus_nt: Some(NtClass::Gaba),
            },
            TypeRow {
                name: "DN_OUT".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
            TypeRow {
                name: "VPN_IN".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
        ];
        let mk = |body_id: i64, superclass: u16, type_id: u32, side: u16| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(type_id),
            instance: None,
            superclass: Some(superclass),
            class: None,
            subclass: None,
            soma_side: Some(side),
            group: Some(body_id / 2), // pairs (0,1), (2,3), ... share a group
            ol_hex1: None,
            ol_hex2: None,
        };
        // 0,1: VPN_IN (L,R); 2,3: AN_X (L,R); 4,5: Central1 (L,R); 6,7: DN_OUT (L,R).
        let rows = vec![
            mk(0, 3, 3, 0),
            mk(1, 3, 3, 1),
            mk(2, 0, 0, 0),
            mk(3, 0, 0, 1),
            mk(4, 1, 1, 0),
            mk(5, 1, 1, 1),
            mk(6, 2, 2, 0),
            mk(7, 2, 2, 1),
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 4,
                weight: 10,
            }, // VPN_IN(L) -> Central1(L)
            Edge {
                pre_idx: 1,
                post_idx: 5,
                weight: 10,
            }, // VPN_IN(R) -> Central1(R)
            Edge {
                pre_idx: 2,
                post_idx: 4,
                weight: 8,
            }, // AN_X(L) -> Central1(L)  (so an_pick's proxy-target rule finds AN_X)
            Edge {
                pre_idx: 4,
                post_idx: 6,
                weight: 10,
            }, // Central1(L) -> DN_OUT(L)
            Edge {
                pre_idx: 5,
                post_idx: 7,
                weight: 10,
            }, // Central1(R) -> DN_OUT(R)
        ];
        let mut total_input = vec![0u64; 8];
        for e in &edges {
            total_input[e.post_idx as usize] += u64::from(e.weight);
        }
        ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: StdBTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 8],
            },
            neuron_nt: vec![NeuronNt::default(); 8],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        }
    }

    fn tiny_config() -> SubgraphConfig {
        SubgraphConfig {
            selection: SelectionParams {
                k: 2,
                theta_path: 3,
                theta_edge: 3,
                max_hidden: 10,
                weak_pair_threshold: 20,
                an_top_n_types: 1,
            },
            nt: NtSignParams::default(),
            rf: crate::subgraph::config::RfParams::default(),
            inputs: InputsConfig {
                visual_types: vec!["VPN_IN".to_string()],
            },
            outputs: OutputsConfig {
                dn_types: vec!["DN_OUT".to_string()],
            },
            input_channels: vec![],
            output_groups: vec![],
        }
    }

    #[test]
    fn end_to_end_build_produces_a_valid_flyg_with_expected_roles() {
        let tables = tiny_tables();
        let config = tiny_config();
        let (flyg, report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        ddai_flyg::validate(&flyg).unwrap();

        assert_eq!(flyg.summary.neurons_by_role.input_visual, 2);
        assert_eq!(flyg.summary.neurons_by_role.output, 2);
        assert_eq!(
            flyg.summary.neurons_by_role.input_ascending, 2,
            "AN_X must be picked by the data-driven rule"
        );
        assert_eq!(
            flyg.summary.neurons_by_role.hidden, 2,
            "Central1 must be reached within k=2 hops"
        );
        assert!(report.missing_visual_types.is_empty());
        assert!(report.missing_dn_types.is_empty());
        assert_eq!(report.an_types_picked.len(), 1);
        assert_eq!(report.an_types_picked[0].name, "AN_X");
    }

    #[test]
    fn missing_type_name_is_reported_not_a_hard_error_when_others_resolve() {
        let tables = tiny_tables();
        let mut config = tiny_config();
        config.inputs.visual_types.push("NoSuchType".to_string());
        let (_flyg, report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        assert_eq!(report.missing_visual_types, vec!["NoSuchType".to_string()]);
    }

    #[test]
    fn all_missing_visual_types_is_a_hard_error() {
        let tables = tiny_tables();
        let mut config = tiny_config();
        config.inputs.visual_types = vec!["NoSuchType".to_string()];
        let err = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap_err();
        assert!(format!("{err:#}").contains("visual_types"));
    }

    #[test]
    fn build_is_deterministic_across_repeated_runs() {
        let tables = tiny_tables();
        let config = tiny_config();
        let (flyg1, _) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        let (flyg2, _) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        assert_eq!(flyg1, flyg2);
    }

    #[test]
    fn output_group_config_produces_side_filtered_members() {
        let tables = tiny_tables();
        let mut config = tiny_config();
        config.output_groups.push(super::super::config::OutputGroupConfig {
            action: "direction_left".to_string(),
            types: vec!["DN_OUT".to_string()],
            side: SideFilter::L,
        });
        let (flyg, _report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        assert_eq!(flyg.output_groups.len(), 1);
        let group = &flyg.output_groups[0];
        assert_eq!(group.action, "direction_left");
        assert_eq!(group.members.len(), 1);
        assert_eq!(flyg.neurons[group.members[0].neuron_index as usize].side, Side::L);
    }

    #[test]
    fn input_channel_config_is_attached_to_the_right_local_type_index() {
        let tables = tiny_tables();
        let mut config = tiny_config();
        config.input_channels.push(super::super::config::InputChannelConfig {
            type_name: "VPN_IN".to_string(),
            channels: vec!["opponent_position".to_string()],
        });
        let (flyg, report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        assert!(report.missing_input_channel_types.is_empty());
        assert_eq!(flyg.input_channels.len(), 1);
        let type_idx = flyg.input_channels[0].type_index;
        assert_eq!(flyg.types[type_idx as usize].name, "VPN_IN");
    }

    /// Regression test: `tiny_tables()`'s edge list is **not** sorted by `(post_idx, pre_idx)`
    /// (its posts run 4, 5, 4, 6, 7 — real `connectome.tables` is always sorted this way, but
    /// nothing before this test enforced that a hand-built/synthetic `ConnectomeTables` must be).
    /// The CSR builder used to fill `pre_index`/`synapse_count`/`type_pair_index` by *position*
    /// assuming that sortedness, which silently scrambled rows for exactly this kind of input —
    /// `ddai_flyg::validate` didn't catch it, because a locally-still-ascending scrambled row is
    /// indistinguishable from a correct one without independent knowledge of each edge's true
    /// post neuron. This asserts every CSR/type-pair array's *exact* contents, hand-computed from
    /// `tiny_tables()`'s 5 edges and `tiny_config()`'s `theta_edge=3`/`weak_pair_threshold=20`.
    #[test]
    fn csr_and_type_pairs_are_exactly_correct_even_from_an_unsorted_edge_list() {
        let tables = tiny_tables();
        let config = tiny_config();
        let (flyg, _report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();

        // All 8 neurons participate (2 visual + 2 AN + 2 hidden + 2 output), local index == the
        // bodyId-sorted global index here (0..=7), so the edges below reference them directly.
        assert_eq!(flyg.neurons.len(), 8);
        // Local type order is alphabetical: AN_X=0, Central1=1, DN_OUT=2, VPN_IN=3 (coincides
        // with `tiny_tables()`'s own global type ids, since it happened to declare them in that
        // same order — verified explicitly here rather than assumed).
        let type_id = |name: &str| flyg.types.iter().position(|t| t.name == name).unwrap() as u32;
        assert_eq!(
            (
                type_id("AN_X"),
                type_id("Central1"),
                type_id("DN_OUT"),
                type_id("VPN_IN")
            ),
            (0, 1, 2, 3)
        );

        assert_eq!(flyg.edges.row_start, vec![0, 0, 0, 0, 0, 2, 3, 4, 5]);
        assert_eq!(
            flyg.edges.pre_index,
            vec![0, 2, 1, 4, 5],
            "post=4's row must be (pre=0, pre=2), sorted"
        );
        assert_eq!(flyg.edges.synapse_count, vec![10, 8, 10, 10, 10]);

        // Type pairs, sorted by (pre_type, post_type): (AN_X, Central1)=8 synapses (< threshold
        // 20 -> weak, own shared-param group), (Central1, DN_OUT)=20 (>= threshold -> own id),
        // (VPN_IN, Central1)=20 (>= threshold -> own id).
        let pairs: Vec<(u32, u32, u64)> = flyg
            .type_pairs
            .iter()
            .map(|p| (p.pre_type, p.post_type, p.total_synapses))
            .collect();
        assert_eq!(pairs, vec![(0, 1, 8), (1, 2, 20), (3, 1, 20)]);
        assert!(
            flyg.types[0].sign == Sign::Excitatory,
            "AN_X is ACh -> needed for the weak-group key below"
        );
        // The two >= threshold pairs each get their own id; the one weak pair gets a fresh id of
        // its own too (nothing else shares its (sign, pre_superclass, post_superclass) key here).
        let shared_ids: Vec<u32> = flyg.type_pairs.iter().map(|p| p.shared_param_id).collect();
        assert_eq!(shared_ids.len(), 3, "{shared_ids:?}");
        assert_eq!(
            shared_ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            3,
            "all 3 pairs get distinct ids here: {shared_ids:?}"
        );

        // type_pair_index, parallel to the sorted edge arrays above: post=4's two edges are both
        // the (VPN_IN, Central1)/(AN_X, Central1) pairs; post=6/7's are both (Central1, DN_OUT).
        let pair_idx_of = |pre_t: u32, post_t: u32| {
            flyg.type_pairs
                .iter()
                .position(|p| p.pre_type == pre_t && p.post_type == post_t)
                .unwrap() as u32
        };
        let expected_type_pair_index = vec![
            pair_idx_of(3, 1), // pre=0 (VPN_IN) -> post=4 (Central1)
            pair_idx_of(0, 1), // pre=2 (AN_X) -> post=4 (Central1)
            pair_idx_of(3, 1), // pre=1 (VPN_IN) -> post=5 (Central1)
            pair_idx_of(1, 2), // pre=4 (Central1) -> post=6 (DN_OUT)
            pair_idx_of(1, 2), // pre=5 (Central1) -> post=7 (DN_OUT)
        ];
        assert_eq!(flyg.edges.type_pair_index, expected_type_pair_index);
    }

    /// F4 (review round 1): `neuron_input_totals.full_connectome` and `.in_subgraph` must be able
    /// to differ — a neuron's full-connectome input total includes synapses the induced subgraph
    /// drops (here: an edge below `theta_edge`, from a neuron that isn't even selected).
    #[test]
    fn neuron_input_totals_full_and_in_subgraph_can_differ() {
        let mut tables = tiny_tables();
        let config = tiny_config();

        // Extra neuron (index 8), not of any seed/hidden-reachable type, wired into Central1(4)
        // with weight 1 — below theta_edge=3, so this edge never makes it into the subgraph, but
        // it's still part of Central1's full-connectome `total_input`.
        tables.neurons.rows.push(NeuronRow {
            body_id: 100,
            status: Some(0),
            type_id: Some(1), // reuse Central1's type; irrelevant to this test
            instance: None,
            superclass: Some(1),
            class: None,
            subclass: None,
            soma_side: Some(0),
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });
        tables.neurons.total_input.push(0);
        tables.neurons.total_output.push(0);
        tables.neuron_nt.push(NeuronNt::default());
        tables.neurons.total_input[4] += 1; // Central1's full-connectome total grows
        tables.edges.edges.push(Edge {
            pre_idx: 8,
            post_idx: 4,
            weight: 1,
        });
        tables.edges.edges.sort_by_key(|e| (e.post_idx, e.pre_idx));

        let (flyg, _report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        let central1_idx = flyg.neurons.iter().position(|n| n.body_id == 4).unwrap();
        let full = flyg.neuron_input_totals.full_connectome[central1_idx];
        let in_sub = flyg.neuron_input_totals.in_subgraph[central1_idx];
        assert_eq!(full, 19, "18 (original 10+8) + 1 (new sub-threshold edge)");
        assert_eq!(in_sub, 18, "the weight-1 edge from an unselected neuron must not count");
        assert!(in_sub < full, "full={full} in_sub={in_sub}");
    }

    /// F4 (review round 1): two *different* type pairs, both below `weak_pair_threshold`, that
    /// share the same (pre sign, pre superclass, post superclass) key must share one
    /// `shared_param_id` — not just any single weak pair getting its own id trivially.
    #[test]
    fn two_weak_pairs_with_the_same_key_share_one_shared_param_id() {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec!["ascending_neuron".to_string(), "central".to_string()],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec![],
        };
        let types = vec![
            TypeRow {
                name: "PreA".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
            TypeRow {
                name: "PreB".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
            TypeRow {
                name: "Post".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
        ];
        let mk = |body_id: i64, type_id: u32, superclass: u16| NeuronRow {
            body_id,
            status: Some(0),
            type_id: Some(type_id),
            instance: None,
            superclass: Some(superclass),
            class: None,
            subclass: None,
            soma_side: None,
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        };
        // 0: PreA (visual input seed), 1: PreB (visual input seed), 2: Post (output seed).
        // Both PreA->Post and PreB->Post are weak (weight 5 < threshold 20), and PreA/PreB share
        // (sign=Excitatory, superclass="ascending_neuron") with Post's superclass "central".
        let rows = vec![mk(0, 0, 0), mk(1, 1, 0), mk(2, 2, 1)];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 2,
                weight: 5,
            },
            Edge {
                pre_idx: 1,
                post_idx: 2,
                weight: 5,
            },
        ];
        let mut total_input = vec![0u64; 3];
        total_input[2] = 10;
        let tables = ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: StdBTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input,
                total_output: vec![0; 3],
            },
            neuron_nt: vec![NeuronNt::default(); 3],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        };
        let config = SubgraphConfig {
            selection: SelectionParams {
                k: 1,
                theta_path: 1,
                theta_edge: 1,
                max_hidden: 10,
                weak_pair_threshold: 20, // both pairs (total 5 each) are weak
                an_top_n_types: 0,
            },
            nt: NtSignParams::default(),
            rf: crate::subgraph::config::RfParams::default(),
            inputs: InputsConfig {
                visual_types: vec!["PreA".to_string(), "PreB".to_string()],
            },
            outputs: OutputsConfig {
                dn_types: vec!["Post".to_string()],
            },
            input_channels: vec![],
            output_groups: vec![],
        };
        let (flyg, _report) = build_flyg(&tables, &config, "t".repeat(64), "c".repeat(64)).unwrap();
        assert_eq!(
            flyg.type_pairs.len(),
            2,
            "(PreA,Post) and (PreB,Post) are distinct pairs: {:?}",
            flyg.type_pairs
        );
        let ids: Vec<u32> = flyg.type_pairs.iter().map(|p| p.shared_param_id).collect();
        assert_eq!(
            ids[0], ids[1],
            "both weak pairs share (sign, pre superclass, post superclass) -> same shared_param_id: {:?}",
            flyg.type_pairs
        );
        assert_eq!(flyg.summary.shared_param_count, 1, "one distinct id across both pairs");
    }
}
