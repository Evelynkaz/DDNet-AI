//! Receptive fields for visual (VPN) input neurons (acceptance criterion 5).
//!
//! **RF center.** For a visual-input neuron, the RF center is the synapse-weighted mean of
//! (`assignedOlHex1`, `assignedOlHex2`) over its presynaptic partners — over the **full** traced
//! connectome (every incoming synapse the neuron has at all, not just the subgraph's edges — see
//! `docs/research/fly-data.md` §1.4 for why: a VPN's retinotopic identity comes from its columnar
//! inputs regardless of which of its other partners end up selected). Only 15 real MaleCNS v1.0
//! types actually carry `assignedOlHex1/2` directly (L1–L5, Mi1/4/9, Tm1/2/4/9/20, C2/C3, T1 — the
//! true optic-lobe columnar layer); a great many of a VPN's *direct* presynaptic partners are one
//! layer downstream of those (T4/T5, TmY, Tm5Y, …) and carry no hex of their own at all. Review
//! round 1 (F2) found this left several visual-input types (HSE/HSN/H2, most of VS, much of LC9)
//! resting on 0–2 real synapses of hex support — not enough to trust as a real retinotopic
//! position. So a presynaptic partner without its own hex gets a **one-hop derived** hex instead:
//! the synapse-weighted mean hex of *its own* directly-hex-carrying presynaptic partners (not
//! recursive — exactly one hop past the direct layer), memoized per neuron since many VPNs share
//! the same upstream relays. A VPN's RF is then the synapse-weighted mean over this *extended*
//! partner set (direct-hex partners at their own hex, one-hop partners at their derived hex).
//!
//! **Hex → (azimuth, elevation) map.** This is an *engineering decision* (the task spec calls it
//! an "approximate linear map", not a biologically derived one): hex1/hex2 are per-optic-lobe
//! local column coordinates (same numeric range on both sides — mirror geometry, not an absolute
//! left/right sign), observed on the real data to range roughly 1..=36 / 1..=39. We use
//! `u = hex1 - hex2` as the nasal–temporal axis and `v = hex1 + hex2` as the dorsal–ventral axis
//! (elevation), min–max-normalize each over **every** hex-coordinated neuron in the whole
//! connectome (so S and M share the exact same calibration, independent of which subgraph is
//! being built). Elevation is scaled straight to ±[`ELEVATION_FOV_DEG`] (both eyes share the same
//! range, no side-dependent sign — [`normalize_signed`]). Azimuth instead first maps `u` to an
//! **unsigned** magnitude in `[0, `[`AZIMUTH_FOV_DEG`]`]` ("how far into this eye's periphery"),
//! then multiplies by the eye-side sign (`L` → −1, `R` → +1, per the spec — [`eye_sign`]): mapping
//! straight to a signed `[-fov, fov]` range and taking `abs()` afterwards would *not* give the
//! same thing — both ends of `u`'s range would collapse to the same `fov` magnitude, destroying
//! exactly the within-eye spread this is trying to preserve (see [`normalize_unsigned`]'s docs and
//! this module's tests). This does mean the *magnitude* of a neuron's azimuth is the same
//! function of hex coordinates on both eyes — only the sign differs — which is consistent with
//! FLY.md's "L/R share parameters, policy is mirror-symmetric" principle.
//!
//! **Minimum support / fallback.** A visual-input neuron whose *extended* hex support (direct +
//! one-hop-derived synapse weight) is below the configured `min_hex_support` (review round 1, F2
//! — default 10) gets a fallback center instead of trusting a thin real signal: spread evenly by
//! `bodyId` order across the azimuth range of its own (type, side) group's members that *do* clear
//! the threshold (or that side's full `[-fov, 0]`/`[0, fov]` half-range if none in the group
//! clear it), with elevation set to that group's mean elevation among the members that clear it
//! (or 0.0 if none do). Every fallback neuron is flagged (`ReceptiveField::is_fallback`).
//!
//! **Reporting.** [`RfStats::per_type`] carries hex-support and azimuth/elevation spread
//! statistics for **every** (type, side) group among the visual inputs, not just a hand-picked
//! few — acceptance criterion 5's LC10a/LC4/LPLC2 sanity check is just three rows of this table.

use std::collections::BTreeMap;

use ddai_flyg::{ReceptiveField, Side};

use super::common::neuron_side;
use crate::tables::ConnectomeTables;

/// Half-width of the mapped azimuth/elevation range, in degrees — see the module docs. Chosen,
/// not measured: a fly's compound eyes each see roughly a full forward hemisphere with substantial
/// binocular overlap, so ±90°/±60° is a reasonable, round-numbered approximation.
pub const AZIMUTH_FOV_DEG: f32 = 90.0;
pub const ELEVATION_FOV_DEG: f32 = 60.0;

/// Global calibration derived once from every hex-coordinated neuron in the whole connectome
/// (independent of any subgraph selection — see module docs).
#[derive(Debug, Clone, Copy)]
struct HexCalibration {
    diff_min: f64,
    diff_max: f64,
    sum_min: f64,
    sum_max: f64,
    /// Number of neurons in the whole connectome with both hex coordinates present (informational,
    /// printed in the report).
    hex_neuron_count: u64,
}

fn compute_hex_calibration(tables: &ConnectomeTables) -> HexCalibration {
    let mut diff_min = f64::INFINITY;
    let mut diff_max = f64::NEG_INFINITY;
    let mut sum_min = f64::INFINITY;
    let mut sum_max = f64::NEG_INFINITY;
    let mut count = 0u64;
    for row in &tables.neurons.rows {
        if let (Some(h1), Some(h2)) = (row.ol_hex1, row.ol_hex2) {
            let (h1, h2) = (f64::from(h1), f64::from(h2));
            let diff = h1 - h2;
            let sum = h1 + h2;
            diff_min = diff_min.min(diff);
            diff_max = diff_max.max(diff);
            sum_min = sum_min.min(sum);
            sum_max = sum_max.max(sum);
            count += 1;
        }
    }
    HexCalibration {
        diff_min,
        diff_max,
        sum_min,
        sum_max,
        hex_neuron_count: count,
    }
}

/// Affine map `[lo, hi] -> [-fov, fov]`, for elevation (which has no eye-side sign convention —
/// both eyes share the same dorsal/ventral range). Returns `0.0` if `lo == hi` (degenerate range
/// — cannot happen on the real data, since hex coordinates span dozens of values, but guarded
/// rather than dividing by zero for a synthetic/tiny test fixture).
fn normalize_signed(value: f64, lo: f64, hi: f64, fov: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    let t = (value - lo) / (hi - lo); // in [0, 1]
    ((t * 2.0 - 1.0) as f32) * fov
}

/// Affine map `[lo, hi] -> [0, fov]`, for the azimuth *magnitude* — how far into this eye's
/// periphery a neuron sits, before the eye-side sign is applied on top (see [`eye_sign`]).
/// Deliberately not "map to `[-fov, fov]` then take `abs()`": that would make magnitude
/// *non-monotonic* in `value` (both ends of the `[lo, hi]` range would map to the same `fov`
/// magnitude, collapsing exactly the spread this is trying to preserve — see the module tests).
fn normalize_unsigned(value: f64, lo: f64, hi: f64, fov: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    let t = (value - lo) / (hi - lo); // in [0, 1]
    (t as f32) * fov
}

fn eye_sign(side: Side) -> f32 {
    match side {
        Side::L => -1.0,
        Side::R => 1.0,
        Side::M | Side::Unknown => 0.0,
    }
}

fn side_label(side: Side) -> &'static str {
    match side {
        Side::L => "L",
        Side::R => "R",
        Side::M => "M",
        Side::Unknown => "?",
    }
}

/// `row_start[i]..row_start[i+1]` indexes `tables.edges.edges` for post-neuron `i`'s incoming
/// edges — cheap to build in one pass since `tables.edges.edges` is already sorted by
/// `(post_idx, pre_idx)` (see `tables.rs`'s module docs).
fn build_post_row_start(tables: &ConnectomeTables) -> Vec<u32> {
    let n = tables.neurons.rows.len();
    let mut row_start = vec![0u32; n + 1];
    for e in &tables.edges.edges {
        row_start[e.post_idx as usize + 1] += 1;
    }
    for i in 0..n {
        row_start[i + 1] += row_start[i];
    }
    row_start
}

/// The neuron's own hex coordinates, if MaleCNS v1.0 assigned it any directly.
fn direct_hex(tables: &ConnectomeTables, idx: u32) -> Option<(f64, f64)> {
    let row = &tables.neurons.rows[idx as usize];
    match (row.ol_hex1, row.ol_hex2) {
        (Some(h1), Some(h2)) => Some((f64::from(h1), f64::from(h2))),
        _ => None,
    }
}

/// One-hop-derived hex for a neuron with no direct hex of its own: the synapse-weighted mean hex
/// of *its own* directly-hex-carrying presynaptic partners (over the full connectome). Not
/// recursive — a partner more than one hop from the direct hex layer gets no derived hex either.
fn derive_one_hop_hex(tables: &ConnectomeTables, post_row_start: &[u32], idx: u32) -> Option<(f64, f64)> {
    let start = post_row_start[idx as usize] as usize;
    let end = post_row_start[idx as usize + 1] as usize;
    let mut sum_h1 = 0.0f64;
    let mut sum_h2 = 0.0f64;
    let mut sum_w = 0.0f64;
    for e in &tables.edges.edges[start..end] {
        if let Some((h1, h2)) = direct_hex(tables, e.pre_idx) {
            let w = f64::from(e.weight);
            sum_h1 += w * h1;
            sum_h2 += w * h2;
            sum_w += w;
        }
    }
    if sum_w > 0.0 {
        Some((sum_h1 / sum_w, sum_h2 / sum_w))
    } else {
        None
    }
}

/// Direct hex if `idx` has one, else its one-hop-derived hex (memoized in `cache`, since many
/// VPNs share the same upstream relay neurons).
fn effective_hex(
    tables: &ConnectomeTables,
    post_row_start: &[u32],
    cache: &mut BTreeMap<u32, Option<(f64, f64)>>,
    idx: u32,
) -> Option<(f64, f64)> {
    if let Some(h) = direct_hex(tables, idx) {
        return Some(h);
    }
    if let Some(&cached) = cache.get(&idx) {
        return cached;
    }
    let derived = derive_one_hop_hex(tables, post_row_start, idx);
    cache.insert(idx, derived);
    derived
}

/// Synapse-weighted mean *effective* hex coordinate (direct or one-hop-derived) of
/// `neuron_idx`'s presynaptic partners, over the full connectome, plus the total support weight
/// (`Some((h1_mean, h2_mean, support))`, `None` if no partner has any effective hex at all).
fn effective_hex_and_support(
    tables: &ConnectomeTables,
    post_row_start: &[u32],
    cache: &mut BTreeMap<u32, Option<(f64, f64)>>,
    neuron_idx: u32,
) -> Option<(f64, f64, f64)> {
    let start = post_row_start[neuron_idx as usize] as usize;
    let end = post_row_start[neuron_idx as usize + 1] as usize;
    let mut sum_h1 = 0.0f64;
    let mut sum_h2 = 0.0f64;
    let mut sum_w = 0.0f64;
    for e in &tables.edges.edges[start..end] {
        if let Some((h1, h2)) = effective_hex(tables, post_row_start, cache, e.pre_idx) {
            let w = f64::from(e.weight);
            sum_h1 += w * h1;
            sum_h2 += w * h2;
            sum_w += w;
        }
    }
    if sum_w > 0.0 {
        Some((sum_h1 / sum_w, sum_h2 / sum_w, sum_w))
    } else {
        None
    }
}

fn min_median_max_f64(values: &[f64]) -> (f64, f64, f64) {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

fn min_median_max_f32(values: &[f32]) -> (f32, f32, f32) {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

/// Hex-support and azimuth/elevation spread statistics for one (visual input type, side) group —
/// review round 1 (F2)'s "report per-type hex-support statistics ... and the RF azimuth/elevation
/// spread per type/side".
#[derive(Debug, Clone)]
pub struct PerTypeSideStats {
    pub type_name: String,
    pub side_label: String,
    pub count: usize,
    pub fallback_count: usize,
    pub hex_support_min: f64,
    pub hex_support_median: f64,
    pub hex_support_max: f64,
    /// How many of this group's members have support below `RfStats::min_hex_support` (i.e. use
    /// the documented fallback specifically for lack of support, not for lack of *any* signal).
    pub below_threshold_count: usize,
    pub azimuth_min_deg: f32,
    pub azimuth_median_deg: f32,
    pub azimuth_max_deg: f32,
    pub elevation_min_deg: f32,
    pub elevation_median_deg: f32,
    pub elevation_max_deg: f32,
}

#[derive(Debug, Clone, Default)]
pub struct RfStats {
    pub total_visual_inputs: usize,
    pub fallback_count: usize,
    pub hex_neuron_count_full_connectome: u64,
    pub min_hex_support: u64,
    /// One row per (type, side) group among the visual inputs, sorted by type name then side.
    pub per_type: Vec<PerTypeSideStats>,
}

/// Computes receptive fields for every neuron in `visual_input_indices` (dense indices into
/// `tables.neurons.rows`, any order — the result is returned as a map keyed by that same index, so
/// callers don't need to pass a matching order back in). `min_hex_support` is the configured
/// minimum synapse-weighted hex support (direct + one-hop-derived) below which the documented
/// fallback is used instead of a real-but-thin signal (review round 1, F2).
pub fn compute_receptive_fields(
    tables: &ConnectomeTables,
    visual_input_indices: &[u32],
    min_hex_support: u64,
) -> (BTreeMap<u32, ReceptiveField>, RfStats) {
    let calibration = compute_hex_calibration(tables);
    let post_row_start = build_post_row_start(tables);
    let mut derived_cache: BTreeMap<u32, Option<(f64, f64)>> = BTreeMap::new();

    // Pass 1: extended (direct + one-hop-derived) hex + support for every visual input.
    let mut support: BTreeMap<u32, f64> = BTreeMap::new();
    let mut raw_hex: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
    for &idx in visual_input_indices {
        match effective_hex_and_support(tables, &post_row_start, &mut derived_cache, idx) {
            Some((h1, h2, s)) => {
                support.insert(idx, s);
                raw_hex.insert(idx, (h1, h2));
            }
            None => {
                support.insert(idx, 0.0);
            }
        }
    }

    // Pass 2: real RF center where support clears the threshold; record which fell back.
    let min_support_f64 = min_hex_support as f64;
    let mut result: BTreeMap<u32, ReceptiveField> = BTreeMap::new();
    let mut needs_fallback: Vec<u32> = Vec::new();
    for &idx in visual_input_indices {
        let side = neuron_side(tables, idx);
        if support[&idx] >= min_support_f64 {
            let (h1, h2) = raw_hex[&idx];
            let azimuth = eye_sign(side)
                * normalize_unsigned(h1 - h2, calibration.diff_min, calibration.diff_max, AZIMUTH_FOV_DEG);
            let elevation = normalize_signed(h1 + h2, calibration.sum_min, calibration.sum_max, ELEVATION_FOV_DEG);
            result.insert(
                idx,
                ReceptiveField {
                    azimuth_deg: azimuth,
                    elevation_deg: elevation,
                    is_fallback: false,
                },
            );
        } else {
            needs_fallback.push(idx);
        }
    }

    // Pass 3: fallback, grouped by (type_id, side) — spread by bodyId order across the group's
    // own real (above-threshold) azimuth range (or the full FOV half-range if the group has none).
    let mut by_group: BTreeMap<(u32, Side), Vec<u32>> = BTreeMap::new();
    for &idx in &needs_fallback {
        let type_id = tables.neurons.rows[idx as usize].type_id.unwrap_or(u32::MAX);
        by_group
            .entry((type_id, neuron_side(tables, idx)))
            .or_default()
            .push(idx);
    }
    for ((_, side), mut members) in by_group {
        members.sort_by_key(|&idx| tables.neurons.rows[idx as usize].body_id);
        let type_id = tables.neurons.rows[members[0] as usize].type_id;
        let group_real_azimuths: Vec<f32> = visual_input_indices
            .iter()
            .filter(|&&other| {
                tables.neurons.rows[other as usize].type_id == type_id && neuron_side(tables, other) == side
            })
            .filter_map(|&other| result.get(&other))
            .map(|rf| rf.azimuth_deg)
            .collect();
        let group_real_elevation_mean: f32 = {
            let elevations: Vec<f32> = visual_input_indices
                .iter()
                .filter(|&&other| {
                    tables.neurons.rows[other as usize].type_id == type_id && neuron_side(tables, other) == side
                })
                .filter_map(|&other| result.get(&other))
                .map(|rf| rf.elevation_deg)
                .collect();
            if elevations.is_empty() {
                0.0
            } else {
                elevations.iter().sum::<f32>() / elevations.len() as f32
            }
        };
        let (lo, hi) = if group_real_azimuths.is_empty() {
            match side {
                Side::L => (-AZIMUTH_FOV_DEG, 0.0),
                Side::R => (0.0, AZIMUTH_FOV_DEG),
                Side::M | Side::Unknown => (-AZIMUTH_FOV_DEG, AZIMUTH_FOV_DEG),
            }
        } else {
            let lo = group_real_azimuths.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = group_real_azimuths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (lo, hi)
        };
        let m = members.len();
        for (rank, &idx) in members.iter().enumerate() {
            let azimuth = if m == 1 {
                (lo + hi) / 2.0
            } else {
                lo + (hi - lo) * (rank as f32) / ((m - 1) as f32)
            };
            result.insert(
                idx,
                ReceptiveField {
                    azimuth_deg: azimuth,
                    elevation_deg: group_real_elevation_mean,
                    is_fallback: true,
                },
            );
        }
    }

    // Per-(type, side) statistics — every group, not just a hand-picked few.
    let mut groups: BTreeMap<(String, String), Vec<u32>> = BTreeMap::new();
    for &idx in visual_input_indices {
        let tn = type_name_of(tables, idx).unwrap_or("<no type>").to_string();
        let sl = side_label(neuron_side(tables, idx)).to_string();
        groups.entry((tn, sl)).or_default().push(idx);
    }
    let mut per_type = Vec::with_capacity(groups.len());
    for ((type_name, side_label), members) in groups {
        let supports: Vec<f64> = members.iter().map(|&idx| support[&idx]).collect();
        let azimuths: Vec<f32> = members.iter().map(|&idx| result[&idx].azimuth_deg).collect();
        let elevations: Vec<f32> = members.iter().map(|&idx| result[&idx].elevation_deg).collect();
        let (hex_support_min, hex_support_median, hex_support_max) = min_median_max_f64(&supports);
        let (azimuth_min_deg, azimuth_median_deg, azimuth_max_deg) = min_median_max_f32(&azimuths);
        let (elevation_min_deg, elevation_median_deg, elevation_max_deg) = min_median_max_f32(&elevations);
        per_type.push(PerTypeSideStats {
            type_name,
            side_label,
            count: members.len(),
            fallback_count: members.iter().filter(|&&idx| result[&idx].is_fallback).count(),
            hex_support_min,
            hex_support_median,
            hex_support_max,
            below_threshold_count: supports.iter().filter(|&&s| s < min_support_f64).count(),
            azimuth_min_deg,
            azimuth_median_deg,
            azimuth_max_deg,
            elevation_min_deg,
            elevation_median_deg,
            elevation_max_deg,
        });
    }

    let stats = RfStats {
        total_visual_inputs: visual_input_indices.len(),
        fallback_count: needs_fallback.len(),
        hex_neuron_count_full_connectome: calibration.hex_neuron_count,
        min_hex_support,
        per_type,
    };

    (result, stats)
}

fn type_name_of(tables: &ConnectomeTables, idx: u32) -> Option<&str> {
    tables.neurons.rows[idx as usize]
        .type_id
        .map(|t| tables.types[t as usize].name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{Dictionaries, Edge, EdgesTable, NeuronNt, NeuronRow, NeuronsTable, TablesHeader, TypeRow};
    use std::collections::BTreeMap as StdBTreeMap;

    /// A tiny connectome: 4 hex-coordinated "columnar" presynaptic neurons at distinct hex
    /// positions, and 3 "VPN"-type postsynaptic neurons (2 on side L wired to different columnar
    /// neurons, 1 on side R with no incoming edge at all -> must fall back), plus a same-side (L)
    /// VPN with no hex-input to test intra-group fallback spread against a real sibling.
    fn tiny_tables() -> ConnectomeTables {
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec!["ol_intrinsic".to_string(), "visual_projection".to_string()],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec!["L".to_string(), "R".to_string()],
        };
        let types = vec![
            TypeRow {
                name: "Col".into(),
                consensus_nt: None,
            },
            TypeRow {
                name: "VPN".into(),
                consensus_nt: None,
            },
        ];
        // idx 0..3: columnar (hex), idx 4..6: VPN.
        let mut rows = vec![
            NeuronRow {
                body_id: 1,
                status: Some(0),
                type_id: Some(0),
                instance: None,
                superclass: Some(0),
                class: None,
                subclass: None,
                soma_side: None,
                group: None,
                ol_hex1: Some(1),
                ol_hex2: Some(1),
            },
            NeuronRow {
                body_id: 2,
                status: Some(0),
                type_id: Some(0),
                instance: None,
                superclass: Some(0),
                class: None,
                subclass: None,
                soma_side: None,
                group: None,
                ol_hex1: Some(30),
                ol_hex2: Some(1),
            },
        ];
        // idx 2: VPN L, wired to columnar 0 (hex 1,1 -> low diff), weight 20 (well above default
        // support threshold 10, so it's not a fallback purely by low weight either).
        rows.push(NeuronRow {
            body_id: 10,
            status: Some(0),
            type_id: Some(1),
            instance: None,
            superclass: Some(1),
            class: None,
            subclass: None,
            soma_side: Some(0),
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });
        // idx 3: VPN L, wired to columnar 1 (hex 30,1 -> high diff) -> a real spread within L.
        rows.push(NeuronRow {
            body_id: 11,
            status: Some(0),
            type_id: Some(1),
            instance: None,
            superclass: Some(1),
            class: None,
            subclass: None,
            soma_side: Some(0),
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });
        // idx 4: VPN L, no incoming edge at all -> fallback, spread within L's real range.
        rows.push(NeuronRow {
            body_id: 12,
            status: Some(0),
            type_id: Some(1),
            instance: None,
            superclass: Some(1),
            class: None,
            subclass: None,
            soma_side: Some(0),
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });
        // idx 5: VPN R, no incoming edge and no R-side hex-based sibling -> full-FOV fallback.
        rows.push(NeuronRow {
            body_id: 13,
            status: Some(0),
            type_id: Some(1),
            instance: None,
            superclass: Some(1),
            class: None,
            subclass: None,
            soma_side: Some(1),
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });

        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 2,
                weight: 20,
            },
            Edge {
                pre_idx: 1,
                post_idx: 3,
                weight: 20,
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
                total_input: vec![0; 6],
                total_output: vec![0; 6],
            },
            neuron_nt: vec![NeuronNt::default(); 6],
            edges: EdgesTable {
                edges,
                autapses_dropped: 0,
            },
        }
    }

    const DEFAULT_MIN_SUPPORT: u64 = 10;

    #[test]
    fn hex_based_neurons_get_distinct_azimuths_matching_eye_sign() {
        let tables = tiny_tables();
        let (rf, stats) = compute_receptive_fields(&tables, &[2, 3, 4, 5], DEFAULT_MIN_SUPPORT);
        assert_eq!(stats.total_visual_inputs, 4);
        assert_eq!(stats.fallback_count, 2, "idx 4 (no edge) and idx 5 (no edge) fall back");
        assert_eq!(stats.min_hex_support, DEFAULT_MIN_SUPPORT);

        let rf2 = rf[&2];
        let rf3 = rf[&3];
        assert!(!rf2.is_fallback && !rf3.is_fallback);
        // Both are on side L -> negative azimuth (eye_sign(L) == -1) whenever the underlying
        // magnitude is nonzero.
        assert!(rf2.azimuth_deg <= 0.0, "{rf2:?}");
        assert!(rf3.azimuth_deg <= 0.0, "{rf3:?}");
        // idx 2's presynaptic hex diff (1-1=0) is smaller in magnitude than idx 3's (30-1=29) —
        // over the global calibration range [0, 29], idx 3 must be further from zero.
        assert!(rf3.azimuth_deg.abs() > rf2.azimuth_deg.abs(), "rf2={rf2:?} rf3={rf3:?}");
    }

    #[test]
    fn below_min_support_falls_back_even_with_a_real_nonzero_signal() {
        // idx 2 has real hex support of exactly 20 synapses; raising the threshold above that
        // must force a fallback even though the signal is "real" (nonzero).
        let tables = tiny_tables();
        let (rf, stats) = compute_receptive_fields(&tables, &[2, 3, 4, 5], 25);
        assert!(
            rf[&2].is_fallback,
            "support 20 < threshold 25 must fall back: {:?}",
            rf[&2]
        );
        assert!(rf[&3].is_fallback);
        let l_row = stats
            .per_type
            .iter()
            .find(|s| s.type_name == "VPN" && s.side_label == "L")
            .unwrap();
        assert_eq!(
            l_row.below_threshold_count, 3,
            "all 3 L-side VPNs are below a threshold of 25"
        );
    }

    #[test]
    fn one_hop_derived_hex_is_used_when_the_direct_partner_has_none() {
        // Extend the fixture: a relay neuron (idx 6, type "Relay", no hex of its own) sits between
        // columnar neuron 0 (hex 1,1) and a new VPN (idx 7). The VPN's RF must come out using
        // columnar 0's hex via the relay (one-hop-derived), not fall back for lack of a *direct*
        // hex-carrying partner.
        let mut tables = tiny_tables();
        tables.types.push(TypeRow {
            name: "Relay".into(),
            consensus_nt: None,
        });
        tables.neurons.rows.push(NeuronRow {
            body_id: 20,
            status: Some(0),
            type_id: Some(2),
            instance: None,
            superclass: Some(0),
            class: None,
            subclass: None,
            soma_side: None,
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });
        tables.neurons.rows.push(NeuronRow {
            body_id: 21,
            status: Some(0),
            type_id: Some(1),
            instance: None,
            superclass: Some(1),
            class: None,
            subclass: None,
            soma_side: Some(0),
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        });
        tables.neurons.total_input = vec![0; 8];
        tables.neurons.total_output = vec![0; 8];
        tables.neuron_nt = vec![NeuronNt::default(); 8];
        tables.edges.edges.push(Edge {
            pre_idx: 0,  // columnar (hex 1,1)
            post_idx: 6, // relay (no hex)
            weight: 20,
        });
        tables.edges.edges.push(Edge {
            pre_idx: 6,  // relay
            post_idx: 7, // new VPN
            weight: 20,
        });
        // Re-sort: `tables.edges.edges` must stay sorted by (post_idx, pre_idx) — this fixture
        // pushes new edges after existing ones, so restore the invariant explicitly.
        tables.edges.edges.sort_by_key(|e| (e.post_idx, e.pre_idx));

        let (rf, stats) = compute_receptive_fields(&tables, &[7], DEFAULT_MIN_SUPPORT);
        assert!(
            !rf[&7].is_fallback,
            "must use the one-hop-derived hex, not fall back: {:?}",
            rf[&7]
        );
        assert_eq!(stats.fallback_count, 0);
    }

    #[test]
    fn fallback_within_a_group_spreads_across_that_groups_real_range() {
        let tables = tiny_tables();
        let (rf, _stats) = compute_receptive_fields(&tables, &[2, 3, 4, 5], DEFAULT_MIN_SUPPORT);
        let rf2 = rf[&2];
        let rf3 = rf[&3];
        let rf4 = rf[&4]; // fallback, same (type, side L) group as 2 and 3
        assert!(rf4.is_fallback);
        let lo = rf2.azimuth_deg.min(rf3.azimuth_deg);
        let hi = rf2.azimuth_deg.max(rf3.azimuth_deg);
        assert!(
            rf4.azimuth_deg >= lo - 1e-3 && rf4.azimuth_deg <= hi + 1e-3,
            "fallback azimuth {} must lie within the group's real range [{lo}, {hi}]",
            rf4.azimuth_deg
        );
    }

    #[test]
    fn fallback_with_no_real_sibling_at_all_uses_the_full_fov_and_correct_sign() {
        let tables = tiny_tables();
        let (rf, _stats) = compute_receptive_fields(&tables, &[2, 3, 4, 5], DEFAULT_MIN_SUPPORT);
        let rf5 = rf[&5]; // side R, no R-side hex-based sibling in this fixture
        assert!(rf5.is_fallback);
        assert!(rf5.azimuth_deg >= 0.0, "R side must be non-negative: {rf5:?}");
        assert!(rf5.azimuth_deg <= AZIMUTH_FOV_DEG + 1e-3);
    }

    #[test]
    fn per_type_stats_cover_every_group_and_show_a_real_spread() {
        let tables = tiny_tables();
        let (_rf, stats) = compute_receptive_fields(&tables, &[2, 3, 4, 5], DEFAULT_MIN_SUPPORT);
        let l_row = stats
            .per_type
            .iter()
            .find(|s| s.type_name == "VPN" && s.side_label == "L")
            .unwrap();
        assert_eq!(l_row.count, 3); // idx 2, 3, 4
        assert!(l_row.azimuth_min_deg <= l_row.azimuth_median_deg && l_row.azimuth_median_deg <= l_row.azimuth_max_deg);
        assert!(
            l_row.azimuth_min_deg < l_row.azimuth_max_deg,
            "L side must show a real spread: {l_row:?}"
        );
        let r_row = stats
            .per_type
            .iter()
            .find(|s| s.type_name == "VPN" && s.side_label == "R")
            .unwrap();
        assert_eq!(r_row.count, 1); // idx 5
        assert_eq!(r_row.fallback_count, 1);
    }

    #[test]
    fn normalize_handles_degenerate_range_without_dividing_by_zero() {
        assert_eq!(normalize_signed(5.0, 5.0, 5.0, 90.0), 0.0);
        assert_eq!(normalize_unsigned(5.0, 5.0, 5.0, 90.0), 0.0);
    }

    #[test]
    fn normalize_unsigned_is_monotonic_not_a_folded_abs_of_the_signed_map() {
        // Regression for the bug this split fixes: mapping to [-fov, fov] then taking abs()
        // would make both range endpoints collapse to the same magnitude (fov), which is not
        // monotonic in `value` and destroys exactly the spread this function exists to preserve.
        let lo_mag = normalize_unsigned(0.0, 0.0, 10.0, 90.0);
        let mid_mag = normalize_unsigned(5.0, 0.0, 10.0, 90.0);
        let hi_mag = normalize_unsigned(10.0, 0.0, 10.0, 90.0);
        assert_eq!(lo_mag, 0.0);
        assert_eq!(hi_mag, 90.0);
        assert!(
            lo_mag < mid_mag && mid_mag < hi_mag,
            "must be strictly monotonic: {lo_mag} {mid_mag} {hi_mag}"
        );
    }
}
