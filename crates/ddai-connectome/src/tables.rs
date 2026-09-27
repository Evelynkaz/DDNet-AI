//! `build-tables`: streams the three MaleCNS Feather files batch by batch and turns them into
//! compact internal tables (neurons, types, per-neuron neurotransmitters, edges), serialized as
//! postcard + zstd. See `docs/research/fly-data.md` §1.4 for the source schemas and §5 for the
//! expected counts this is cross-checked against.
//!
//! Design notes (why things are shaped the way they are):
//! - The **neurons** table has one row per `body-annotations` row (all ~211k bodies, every
//!   `status`, not just `Traced`) so `stats` can report counts by status/superclass across the
//!   whole annotation set. Its dense index is bodies sorted by `bodyId` ascending — this makes
//!   the index assignment independent of arrow's internal row order and gives `edges` a stable
//!   target to index into.
//! - Low-cardinality string columns (`status`, `superclass`, `class`, `subclass`, `somaSide`)
//!   are dictionary-encoded into [`Dictionaries`]; `instance` is not (it is close to unique per
//!   neuron) and is stored inline. `type` gets its own dictionary (`types`) because it carries a
//!   payload (`consensus_nt`), not just a name.
//! - **Determinism**: every dictionary is built from a `BTreeSet`/`BTreeMap` (sorted iteration),
//!   never from iterating a `HashMap`. `HashMap`s are used internally only as `body_id -> index`
//!   lookup tables and are never iterated to produce output.
//! - **Edges** only exist between two `Traced` bodies — the traced-only weights file is supposed
//!   to already guarantee this, but it's verified per-row rather than trusted (see
//!   [`read_edges`]'s `is_traced` check) — sorted by `(post_idx, pre_idx)`. Autapses
//!   (`pre_idx == post_idx`) are dropped from the edge *list* and counted, but their weight is
//!   still real synapses on that one neuron, so it still counts in `total_input`/`total_output`
//!   (added to both, since pre==post for a self-loop). Both totals sum over ALL of a neuron's
//!   traced inputs/outputs, with no weight threshold.
//! - **Memory**: files are read one `RecordBatch` at a time and never buffered whole; only the
//!   derived compact structures (roughly one `Edge` per input row, ~12 bytes each) accumulate
//!   across the file. See the crate README for the measured peak RSS on the real 508 MB file.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::ipc::reader::FileReader;
use serde::{Deserialize, Serialize};

use crate::hashing::hash_file;

/// Bumped whenever the *meaning* of a stored field changes (not just its byte layout, which
/// `postcard`/`serde` would already fail to decode mismatched). v2: `total_input`/`total_output`
/// now include autapse weight (v1 excluded it, matching the edge list); see the module docs.
/// v3 (task 6.3): `NeuronRow` gained `ol_hex1`/`ol_hex2` (the source `assignedOlHex1/2` columns)
/// — the hex-grid column coordinates of optic-lobe columnar neurons, needed to compute visual
/// receptive fields for the fly subgraph (`ddai-flyg`); every other neuron has both `None`.
pub const TABLES_FORMAT_VERSION: u32 = 3;
pub const TABLES_FILE_NAME: &str = "connectome.tables";

pub const ANNOTATIONS_FILE_NAME: &str = "body-annotations-male-cns-v1.0-minconf-0.5.feather";
pub const NEUROTRANSMITTERS_FILE_NAME: &str = "body-neurotransmitters-male-cns-v1.0.feather";
pub const WEIGHTS_FILE_NAME: &str = "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather";

pub fn tables_path(out_dir: &Path) -> PathBuf {
    out_dir.join(TABLES_FILE_NAME)
}

// ---------------------------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------------------------

/// One of the 7 neurotransmitter classes the source data predicts, plus `Unclear` for the
/// literal `"unclear"` value (see docs/research/fly-data.md §1.4). A body/type with *no* NT row
/// at all is `None` at the call site, not a variant here — see [`crate::tables::NeuronNt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NtClass {
    Acetylcholine,
    Glutamate,
    Gaba,
    Histamine,
    Dopamine,
    Serotonin,
    Octopamine,
    Unclear,
}

impl NtClass {
    /// Parses one of the source data's lowercase NT strings. Anything else is a hard error: an
    /// unrecognized value means the schema drifted from what this code assumes, and silently
    /// mapping it to `None`/`Unclear` would hide that.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "acetylcholine" => Ok(Self::Acetylcholine),
            "glutamate" => Ok(Self::Glutamate),
            "gaba" => Ok(Self::Gaba),
            "histamine" => Ok(Self::Histamine),
            "dopamine" => Ok(Self::Dopamine),
            "serotonin" => Ok(Self::Serotonin),
            "octopamine" => Ok(Self::Octopamine),
            "unclear" => Ok(Self::Unclear),
            other => Err(anyhow!("unrecognized neurotransmitter class {other:?}")),
        }
    }
}

/// Low-cardinality string columns, dictionary-encoded. Every `Vec` is sorted ascending, so a
/// value's id (its position) is independent of row order in the source file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dictionaries {
    pub statuses: Vec<String>,
    pub superclasses: Vec<String>,
    pub classes: Vec<String>,
    pub subclasses: Vec<String>,
    pub soma_sides: Vec<String>,
}

/// One row of the `types` table (id = its position in `ConnectomeTables::types`), built from
/// `body-annotations`' `type` column plus the per-type consensus NT joined in from
/// `body-neurotransmitters` (majority vote over that type's neurons; see [`build_tables_from_raw`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeRow {
    pub name: String,
    pub consensus_nt: Option<NtClass>,
}

/// One row of the `neurons` table (dense index = its position).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuronRow {
    pub body_id: i64,
    pub status: Option<u16>,
    pub type_id: Option<u32>,
    pub instance: Option<String>,
    pub superclass: Option<u16>,
    pub class: Option<u16>,
    pub subclass: Option<u16>,
    pub soma_side: Option<u16>,
    /// Source `group` column, a `double` in the Feather file but integral by convention; see
    /// [`f64_to_exact_i64`]. `None` when the source value is missing (NaN).
    pub group: Option<i64>,
    /// Source `assignedOlHex1`/`assignedOlHex2` columns (task 6.3): hex-grid column coordinates
    /// of optic-lobe columnar neurons (observed range on the real data: roughly 1..=36 / 1..=39),
    /// `double` in the Feather file but integral by convention; see [`f64_to_exact_i64`]. `None`
    /// for the large majority of neurons that aren't optic-lobe columns (no hex position at
    /// all). Used by `ddai-flyg`'s subgraph builder to compute visual receptive fields: a VPN
    /// neuron's RF center is the synapse-weighted mean of its presynaptic partners' hex
    /// coordinates (see `docs/FLY.md` §5, `crates/ddai-flyg/README.md`).
    pub ol_hex1: Option<i16>,
    pub ol_hex2: Option<i16>,
}

/// Per-neuron predicted NT, parallel to `neurons` (same length, same order/index).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuronNt {
    pub predicted_nt: Option<NtClass>,
    pub predicted_nt_confidence_milli: Option<u16>,
}

impl NeuronNt {
    /// `predicted_nt_confidence` as `f32` (source is a `double` in \[0, 1\]; stored as
    /// milli-units, i.e. thousandths, for a compact fixed-point representation).
    pub fn predicted_nt_confidence(&self) -> Option<f32> {
        self.predicted_nt_confidence_milli.map(|m| f32::from(m) / 1000.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub pre_idx: u32,
    pub post_idx: u32,
    pub weight: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgesTable {
    /// Sorted by `(post_idx, pre_idx)` ascending.
    pub edges: Vec<Edge>,
    pub autapses_dropped: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuronsTable {
    pub rows: Vec<NeuronRow>,
    /// Sum of `weight` over this neuron's incoming traced synapses — parallel to `rows`. Includes
    /// autapse weight (an autapse is still real input, even though it's excluded from
    /// `EdgesTable::edges` — see the module docs).
    pub total_input: Vec<u64>,
    /// Sum of `weight` over this neuron's outgoing traced synapses — parallel to `rows`. Includes
    /// autapse weight, symmetrically with `total_input`.
    pub total_output: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TablesHeader {
    pub format_version: u32,
    /// sha256 (hex) of each input Feather file, keyed by file name.
    pub input_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectomeTables {
    pub header: TablesHeader,
    pub dictionaries: Dictionaries,
    pub types: Vec<TypeRow>,
    pub neurons: NeuronsTable,
    pub neuron_nt: Vec<NeuronNt>,
    pub edges: EdgesTable,
}

/// Summary printed by the `build-tables` CLI command.
#[derive(Debug, Clone)]
pub struct BuildSummary {
    pub neurons: usize,
    pub traced_neurons: usize,
    pub types: usize,
    pub edges: usize,
    pub autapses_dropped: u64,
    pub output_path: PathBuf,
    pub wall_time: std::time::Duration,
}

// ---------------------------------------------------------------------------------------------
// Exact double -> i64 conversion
// ---------------------------------------------------------------------------------------------

/// The largest magnitude at which every integer is still exactly representable as `f64` (2^53:
/// the mantissa has 52 explicit bits + 1 implicit bit). Any `|x|` beyond this may silently
/// collide with a neighboring integer, so a value out here can never be *proven* exact — it must
/// be rejected even if it happens to look integral.
const MAX_EXACTLY_REPRESENTABLE_F64: f64 = 9_007_199_254_740_992.0; // 2^53

/// Converts a source `double` id/group column value to `i64`, exactly. `NaN` (the source's way
/// of saying "missing") maps to `Ok(None)`. Any other value that doesn't round-trip exactly
/// through `i64` (fractional, or too large to represent exactly) is `Err` — that indicates the
/// source data violated the "these doubles are really integers" assumption, which we want to
/// know about rather than silently truncate.
///
/// The magnitude check happens *before* the `as i64` cast: Rust's float-to-int cast saturates
/// out-of-range values (e.g. `2f64.powi(63) as i64 == i64::MAX`), and `i64::MAX as f64` rounds
/// back to exactly `2f64.powi(63)` too — so a naive round-trip check alone would wrongly accept
/// `2^63` (and anything that saturates to `i64::MAX`/`i64::MIN`) as "exact".
pub fn f64_to_exact_i64(x: f64) -> Result<Option<i64>> {
    if x.is_nan() {
        return Ok(None);
    }
    if !(-MAX_EXACTLY_REPRESENTABLE_F64..=MAX_EXACTLY_REPRESENTABLE_F64).contains(&x) {
        return Err(anyhow!(
            "value {x} is outside the range f64 can represent exactly ([-2^53, 2^53])"
        ));
    }
    let as_i64 = x as i64; // safe: |x| <= 2^53, far inside i64's range, so this cast never saturates
    if (as_i64 as f64) == x {
        Ok(Some(as_i64))
    } else {
        Err(anyhow!("value {x} does not convert exactly to i64 (not integral)"))
    }
}

/// Like [`f64_to_exact_i64`], but additionally requires the value to fit in `i16` — used for
/// `assignedOlHex1/2`, whose observed range on the real data (roughly 1..=36 / 1..=39) is nowhere
/// near `i16`'s range, so a value outside it signals the source data no longer matches this
/// assumption rather than a value to silently clamp.
pub fn f64_to_exact_i16(x: f64) -> Result<Option<i16>> {
    match f64_to_exact_i64(x)? {
        None => Ok(None),
        Some(v) => i16::try_from(v)
            .map(Some)
            .with_context(|| format!("value {v} does not fit in i16")),
    }
}

// ---------------------------------------------------------------------------------------------
// Arrow column access helpers
// ---------------------------------------------------------------------------------------------

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a arrow::array::ArrayRef> {
    batch
        .column_by_name(name)
        .with_context(|| format!("column {name:?} not found in batch (schema: {:?})", batch.schema()))
}

fn i64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    let col = column(batch, name)?;
    col.as_any()
        .downcast_ref::<Int64Array>()
        .with_context(|| format!("column {name:?} is not Int64 (got {:?})", col.data_type()))
}

fn f64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    let col = column(batch, name)?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("column {name:?} is not Float64 (got {:?})", col.data_type()))
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let col = column(batch, name)?;
    col.as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("column {name:?} is not Utf8/String (got {:?})", col.data_type()))
}

fn open_ipc_reader(path: &Path, projection: Option<Vec<usize>>) -> Result<FileReader<BufReader<File>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    FileReader::try_new_buffered(file, projection)
        .with_context(|| format!("{}: not a readable Arrow IPC (Feather) file", path.display()))
}

/// Resolves `names` to column indices via a cheap probe open (parses only the footer/schema,
/// no batch bodies), for use as a [`FileReader`] projection. Used so a wide source file with
/// columns we never touch (e.g. the weights file's `type_pre`/`type_post`) doesn't pay to
/// decompress them.
fn column_projection(path: &Path, names: &[&str]) -> Result<Vec<usize>> {
    let probe = open_ipc_reader(path, None)?;
    let schema = probe.schema();
    names
        .iter()
        .map(|n| {
            schema
                .index_of(n)
                .with_context(|| format!("{}: column {n:?} not found", path.display()))
        })
        .collect()
}

fn opt_string(col: &StringArray, i: usize) -> Option<String> {
    if col.is_null(i) {
        None
    } else {
        Some(col.value(i).to_string())
    }
}

fn opt_f64(col: &Float64Array, i: usize) -> Option<f64> {
    if col.is_null(i) { None } else { Some(col.value(i)) }
}

// ---------------------------------------------------------------------------------------------
// Pass 1: body-annotations -> raw rows + dictionaries
// ---------------------------------------------------------------------------------------------

struct RawAnnotationRow {
    body_id: i64,
    status: Option<String>,
    type_name: Option<String>,
    instance: Option<String>,
    superclass: Option<String>,
    class: Option<String>,
    subclass: Option<String>,
    soma_side: Option<String>,
    group: Option<f64>,
    ol_hex1: Option<f64>,
    ol_hex2: Option<f64>,
}

struct RawAnnotations {
    rows: Vec<RawAnnotationRow>,
    dictionaries: Dictionaries,
    /// Distinct `type` names, sorted (this becomes the `types` table's order).
    type_names: Vec<String>,
}

fn read_annotations(path: &Path) -> Result<RawAnnotations> {
    let reader = open_ipc_reader(path, None)?;

    let mut rows = Vec::new();
    let mut statuses = BTreeSet::new();
    let mut superclasses = BTreeSet::new();
    let mut classes = BTreeSet::new();
    let mut subclasses = BTreeSet::new();
    let mut soma_sides = BTreeSet::new();
    let mut type_names = BTreeSet::new();

    for batch_result in reader {
        let batch = batch_result.with_context(|| format!("reading a batch from {}", path.display()))?;
        let body_id_col = i64_column(&batch, "bodyId")?;
        let status_col = string_column(&batch, "status")?;
        let type_col = string_column(&batch, "type")?;
        let instance_col = string_column(&batch, "instance")?;
        let superclass_col = string_column(&batch, "superclass")?;
        let class_col = string_column(&batch, "class")?;
        let subclass_col = string_column(&batch, "subclass")?;
        let soma_side_col = string_column(&batch, "somaSide")?;
        let group_col = f64_column(&batch, "group")?;
        let ol_hex1_col = f64_column(&batch, "assignedOlHex1")?;
        let ol_hex2_col = f64_column(&batch, "assignedOlHex2")?;

        for i in 0..batch.num_rows() {
            if body_id_col.is_null(i) {
                bail!("{}: row {i} has a null bodyId", path.display());
            }
            let status = opt_string(status_col, i);
            let type_name = opt_string(type_col, i);
            let superclass = opt_string(superclass_col, i);
            let class = opt_string(class_col, i);
            let subclass = opt_string(subclass_col, i);
            let soma_side = opt_string(soma_side_col, i);

            if let Some(v) = &status {
                statuses.insert(v.clone());
            }
            if let Some(v) = &type_name {
                type_names.insert(v.clone());
            }
            if let Some(v) = &superclass {
                superclasses.insert(v.clone());
            }
            if let Some(v) = &class {
                classes.insert(v.clone());
            }
            if let Some(v) = &subclass {
                subclasses.insert(v.clone());
            }
            if let Some(v) = &soma_side {
                soma_sides.insert(v.clone());
            }

            rows.push(RawAnnotationRow {
                body_id: body_id_col.value(i),
                status,
                type_name,
                instance: opt_string(instance_col, i),
                superclass,
                class,
                subclass,
                soma_side,
                group: opt_f64(group_col, i),
                ol_hex1: opt_f64(ol_hex1_col, i),
                ol_hex2: opt_f64(ol_hex2_col, i),
            });
        }
    }

    Ok(RawAnnotations {
        rows,
        dictionaries: Dictionaries {
            statuses: statuses.into_iter().collect(),
            superclasses: superclasses.into_iter().collect(),
            classes: classes.into_iter().collect(),
            subclasses: subclasses.into_iter().collect(),
            soma_sides: soma_sides.into_iter().collect(),
        },
        type_names: type_names.into_iter().collect(),
    })
}

fn dict_index_map(dict: &[String]) -> HashMap<&str, u16> {
    dict.iter().enumerate().map(|(i, s)| (s.as_str(), i as u16)).collect()
}

// ---------------------------------------------------------------------------------------------
// Pass 2: body-neurotransmitters -> per-body NT lookup
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct NtSourceRow {
    predicted_nt: Option<NtClass>,
    predicted_nt_confidence: Option<f32>,
    consensus_nt: Option<NtClass>,
}

fn read_neurotransmitters(path: &Path) -> Result<HashMap<i64, NtSourceRow>> {
    let projection = column_projection(
        path,
        &["body", "predicted_nt", "predicted_nt_confidence", "consensus_nt"],
    )?;
    let reader = open_ipc_reader(path, Some(projection))?;
    let mut by_body = HashMap::new();

    for batch_result in reader {
        let batch = batch_result.with_context(|| format!("reading a batch from {}", path.display()))?;
        let body_col = i64_column(&batch, "body")?;
        let predicted_col = string_column(&batch, "predicted_nt")?;
        let confidence_col = f64_column(&batch, "predicted_nt_confidence")?;
        let consensus_col = string_column(&batch, "consensus_nt")?;

        for i in 0..batch.num_rows() {
            if body_col.is_null(i) {
                bail!("{}: row {i} has a null body id", path.display());
            }
            let predicted_nt = opt_string(predicted_col, i).map(|s| NtClass::parse(&s)).transpose()?;
            let consensus_nt = opt_string(consensus_col, i).map(|s| NtClass::parse(&s)).transpose()?;
            // A *non-null* cell can still hold a NaN float (Arrow's null bitmap and IEEE-754 NaN
            // are two independent ways for a float to be "missing"); treat both the same way —
            // otherwise a NaN would silently become 0.0 confidence a few lines down, in
            // `build_tables_from_raw`'s `(c * 1000.0).round().clamp(0.0, 1000.0) as u16`: NaN
            // compares false to both clamp bounds so it falls through unchanged, and `NaN as u16`
            // saturates to 0 — i.e. a real "no confidence value" would look like "0% confidence".
            let predicted_nt_confidence =
                opt_f64(confidence_col, i).and_then(|v| if v.is_nan() { None } else { Some(v as f32) });
            by_body.insert(
                body_col.value(i),
                NtSourceRow {
                    predicted_nt,
                    predicted_nt_confidence,
                    consensus_nt,
                },
            );
        }
    }

    Ok(by_body)
}

// ---------------------------------------------------------------------------------------------
// Pass 3: connectome-weights -> edges
// ---------------------------------------------------------------------------------------------

struct EdgeBuildResult {
    edges: Vec<Edge>,
    autapses_dropped: u64,
    total_input: Vec<u64>,
    total_output: Vec<u64>,
}

/// `is_traced[idx]` says whether the neuron at dense index `idx` has `status == "Traced"`.
/// `read_edges` uses it to verify the traced-only file's own guarantee (both endpoints Traced)
/// rather than silently trusting the file name.
fn read_edges(path: &Path, body_id_to_idx: &HashMap<i64, u32>, is_traced: &[bool]) -> Result<EdgeBuildResult> {
    let num_neurons = is_traced.len();
    let projection = column_projection(path, &["body_pre", "body_post", "weight"])?;
    let reader = open_ipc_reader(path, Some(projection))?;
    let mut edges = Vec::new();
    let mut autapses_dropped = 0u64;
    let mut total_input = vec![0u64; num_neurons];
    let mut total_output = vec![0u64; num_neurons];

    for batch_result in reader {
        let batch = batch_result.with_context(|| format!("reading a batch from {}", path.display()))?;
        let pre_col = i64_column(&batch, "body_pre")?;
        let post_col = i64_column(&batch, "body_post")?;
        let weight_col = i64_column(&batch, "weight")?;

        for i in 0..batch.num_rows() {
            if pre_col.is_null(i) || post_col.is_null(i) || weight_col.is_null(i) {
                bail!("{}: row {i} has a null body_pre/body_post/weight", path.display());
            }
            let body_pre = pre_col.value(i);
            let body_post = post_col.value(i);
            let weight_raw = weight_col.value(i);
            let weight: u32 = weight_raw.try_into().with_context(|| {
                format!(
                    "{}: row {i} has weight {weight_raw}, does not fit in u32",
                    path.display()
                )
            })?;

            let pre_idx = *body_id_to_idx.get(&body_pre).with_context(|| {
                format!(
                    "{}: row {i} references body_pre {body_pre}, not found in body-annotations",
                    path.display()
                )
            })?;
            let post_idx = *body_id_to_idx.get(&body_post).with_context(|| {
                format!(
                    "{}: row {i} references body_post {body_post}, not found in body-annotations",
                    path.display()
                )
            })?;

            // The file's name promises both endpoints are Traced; verify rather than trust,
            // since a violation would silently corrupt the subgraph selection later tasks build
            // on top of this.
            if !is_traced[pre_idx as usize] {
                bail!(
                    "{}: row {i} has non-Traced body_pre {body_pre} (idx {pre_idx})",
                    path.display()
                );
            }
            if !is_traced[post_idx as usize] {
                bail!(
                    "{}: row {i} has non-Traced body_post {body_post} (idx {post_idx})",
                    path.display()
                );
            }

            // Autapses (self-loops) are dropped from the edge *list* (see module docs), but a
            // self-loop's synapses are still real synapses on that neuron, so it still counts
            // towards both `total_input` and `total_output` (same index, both directions).
            total_output[pre_idx as usize] += weight as u64;
            total_input[post_idx as usize] += weight as u64;

            if pre_idx == post_idx {
                autapses_dropped += 1;
                continue;
            }

            edges.push(Edge {
                pre_idx,
                post_idx,
                weight,
            });
        }
    }

    edges.sort_unstable_by_key(|e| (e.post_idx, e.pre_idx));

    Ok(EdgeBuildResult {
        edges,
        autapses_dropped,
        total_input,
        total_output,
    })
}

// ---------------------------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------------------------

/// Builds [`ConnectomeTables`] from the three raw Feather files in `raw_dir` (must be named
/// exactly [`ANNOTATIONS_FILE_NAME`], [`NEUROTRANSMITTERS_FILE_NAME`], [`WEIGHTS_FILE_NAME`]).
pub fn build_tables_from_raw(raw_dir: &Path) -> Result<ConnectomeTables> {
    let annotations_path = raw_dir.join(ANNOTATIONS_FILE_NAME);
    let nt_path = raw_dir.join(NEUROTRANSMITTERS_FILE_NAME);
    let weights_path = raw_dir.join(WEIGHTS_FILE_NAME);

    for p in [&annotations_path, &nt_path, &weights_path] {
        if !p.is_file() {
            bail!("expected input file not found: {}", p.display());
        }
    }

    // Hashes go into the header regardless of how the files got here (fetch's own manifest
    // pinning is a separate, earlier check); this makes the compact tables self-describing.
    let mut input_sha256 = BTreeMap::new();
    input_sha256.insert(
        ANNOTATIONS_FILE_NAME.to_string(),
        hash_file(&annotations_path)?.sha256_hex,
    );
    input_sha256.insert(NEUROTRANSMITTERS_FILE_NAME.to_string(), hash_file(&nt_path)?.sha256_hex);
    input_sha256.insert(WEIGHTS_FILE_NAME.to_string(), hash_file(&weights_path)?.sha256_hex);

    let raw = read_annotations(&annotations_path)?;
    let status_ids = dict_index_map(&raw.dictionaries.statuses);
    let superclass_ids = dict_index_map(&raw.dictionaries.superclasses);
    let class_ids = dict_index_map(&raw.dictionaries.classes);
    let subclass_ids = dict_index_map(&raw.dictionaries.subclasses);
    let soma_side_ids = dict_index_map(&raw.dictionaries.soma_sides);
    let type_ids = raw
        .type_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect::<HashMap<_, _>>();

    // Dense index = position after sorting by bodyId, independent of source row order.
    let mut order: Vec<usize> = (0..raw.rows.len()).collect();
    order.sort_unstable_by_key(|&i| raw.rows[i].body_id);
    // bodyId is documented unique; verify it, since a duplicate would silently corrupt the
    // body_id -> idx map used by edges below.
    for w in order.windows(2) {
        if raw.rows[w[0]].body_id == raw.rows[w[1]].body_id {
            bail!(
                "{}: duplicate bodyId {}",
                annotations_path.display(),
                raw.rows[w[0]].body_id
            );
        }
    }

    let mut neuron_rows = Vec::with_capacity(order.len());
    let mut body_id_to_idx = HashMap::with_capacity(order.len());
    for (idx, &src_i) in order.iter().enumerate() {
        let r = &raw.rows[src_i];
        let group = f64_to_exact_i64(r.group.unwrap_or(f64::NAN))
            .with_context(|| format!("{}: bodyId {} column `group`", annotations_path.display(), r.body_id))?;
        let ol_hex1 = f64_to_exact_i16(r.ol_hex1.unwrap_or(f64::NAN)).with_context(|| {
            format!(
                "{}: bodyId {} column `assignedOlHex1`",
                annotations_path.display(),
                r.body_id
            )
        })?;
        let ol_hex2 = f64_to_exact_i16(r.ol_hex2.unwrap_or(f64::NAN)).with_context(|| {
            format!(
                "{}: bodyId {} column `assignedOlHex2`",
                annotations_path.display(),
                r.body_id
            )
        })?;
        neuron_rows.push(NeuronRow {
            body_id: r.body_id,
            status: r.status.as_deref().and_then(|s| status_ids.get(s)).copied(),
            type_id: r.type_name.as_deref().and_then(|s| type_ids.get(s)).copied(),
            instance: r.instance.clone(),
            superclass: r.superclass.as_deref().and_then(|s| superclass_ids.get(s)).copied(),
            class: r.class.as_deref().and_then(|s| class_ids.get(s)).copied(),
            subclass: r.subclass.as_deref().and_then(|s| subclass_ids.get(s)).copied(),
            soma_side: r.soma_side.as_deref().and_then(|s| soma_side_ids.get(s)).copied(),
            group,
            ol_hex1,
            ol_hex2,
        });
        body_id_to_idx.insert(r.body_id, idx as u32);
    }
    let traced_status_id = status_ids.get("Traced").copied();
    let is_traced: Vec<bool> = neuron_rows
        .iter()
        .map(|n| n.status == traced_status_id && traced_status_id.is_some())
        .collect();

    let nt_by_body = read_neurotransmitters(&nt_path)?;
    let neuron_nt: Vec<NeuronNt> = neuron_rows
        .iter()
        .map(|n| match nt_by_body.get(&n.body_id) {
            Some(src) => NeuronNt {
                predicted_nt: src.predicted_nt,
                predicted_nt_confidence_milli: src
                    .predicted_nt_confidence
                    .map(|c| (c * 1000.0).round().clamp(0.0, 1000.0) as u16),
            },
            None => NeuronNt::default(),
        })
        .collect();

    // Per-type consensus NT: majority vote of the `consensus_nt` values seen among a type's
    // neurons (documented to already agree within a type; voting just makes real-world
    // disagreement deterministic instead of a crash — see docs/research/fly-data.md §1.4).
    let mut type_votes: Vec<BTreeMap<NtClass, u32>> = vec![BTreeMap::new(); raw.type_names.len()];
    for n in &neuron_rows {
        if let Some(type_id) = n.type_id
            && let Some(src) = nt_by_body.get(&n.body_id)
            && let Some(nt) = src.consensus_nt
        {
            *type_votes[type_id as usize].entry(nt).or_insert(0) += 1;
        }
    }
    let types: Vec<TypeRow> = raw
        .type_names
        .into_iter()
        .zip(type_votes)
        .map(|(name, votes)| {
            let consensus_nt = votes.into_iter().max_by_key(|(_, count)| *count).map(|(nt, _)| nt);
            TypeRow { name, consensus_nt }
        })
        .collect();

    let edge_result = read_edges(&weights_path, &body_id_to_idx, &is_traced)?;

    Ok(ConnectomeTables {
        header: TablesHeader {
            format_version: TABLES_FORMAT_VERSION,
            input_sha256,
        },
        dictionaries: raw.dictionaries,
        types,
        neurons: NeuronsTable {
            rows: neuron_rows,
            total_input: edge_result.total_input,
            total_output: edge_result.total_output,
        },
        neuron_nt,
        edges: EdgesTable {
            edges: edge_result.edges,
            autapses_dropped: edge_result.autapses_dropped,
        },
    })
}

/// Runs `build-tables`: reads `raw_dir`, writes the compact tables to `out_dir`, returns a
/// summary for the CLI to print.
pub fn run_build_tables(raw_dir: &Path, out_dir: &Path) -> Result<BuildSummary> {
    let started = Instant::now();
    let tables = build_tables_from_raw(raw_dir)?;

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let output_path = save_tables(&tables, out_dir)?;

    Ok(BuildSummary {
        neurons: tables.neurons.rows.len(),
        traced_neurons: {
            let traced_id = tables
                .dictionaries
                .statuses
                .iter()
                .position(|s| s == "Traced")
                .map(|i| i as u16);
            tables
                .neurons
                .rows
                .iter()
                .filter(|n| n.status == traced_id && traced_id.is_some())
                .count()
        },
        types: tables.types.len(),
        edges: tables.edges.edges.len(),
        autapses_dropped: tables.edges.autapses_dropped,
        output_path,
        wall_time: started.elapsed(),
    })
}

/// Serializes `tables` as postcard, compresses with zstd (level 3 — datasets, not archives; see
/// docs/research/rust-stack.md §2), and writes it to `<out_dir>/connectome.tables`.
pub fn save_tables(tables: &ConnectomeTables, out_dir: &Path) -> Result<PathBuf> {
    let bytes = postcard::to_allocvec(tables).context("serializing tables to postcard")?;
    let compressed = zstd::stream::encode_all(bytes.as_slice(), 3).context("zstd-compressing tables")?;
    let path = tables_path(out_dir);
    // Write to a sibling temp file and rename into place, so a reader (or a crashed/killed
    // build) never sees a half-written `connectome.tables`, and a failed write never clobbers a
    // previously good one. `rename` within the same directory is atomic on the filesystems we
    // target (ext4/xfs/btrfs on Linux).
    let tmp_path = out_dir.join(format!("{TABLES_FILE_NAME}.tmp"));
    std::fs::write(&tmp_path, compressed).with_context(|| format!("writing {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("renaming {} to {}", tmp_path.display(), path.display()))?;
    Ok(path)
}

/// Inverse of [`save_tables`]. Rejects a file written by a different, incompatible
/// [`TABLES_FORMAT_VERSION`] with a clear error instead of returning silently-wrong data (the
/// wire format is still whatever `postcard`/`serde` produced for the *current* Rust struct
/// definitions, so an old file with a different version can still happen to decode — this catch
/// is specifically for cases where the bytes decode fine but *mean* something different now,
/// e.g. v1's `total_input`/`total_output` excluded autapse weight and v2's don't).
pub fn load_tables(out_dir: &Path) -> Result<ConnectomeTables> {
    load_tables_file(&tables_path(out_dir))
}

/// Like [`load_tables`], but takes the exact path to the tables file itself rather than its
/// containing directory (used by `build-subgraph --tables <file>`, whose CLI contract — task
/// 6.3's spec — names the file directly; see [`resolve_tables_file_path`] for the bit of both-ways
/// compatibility that lets a caller pass either).
pub fn load_tables_file(path: &Path) -> Result<ConnectomeTables> {
    let compressed = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes = zstd::stream::decode_all(compressed.as_slice())
        .with_context(|| format!("zstd-decompressing {}", path.display()))?;
    let tables: ConnectomeTables =
        postcard::from_bytes(&bytes).with_context(|| format!("deserializing {}", path.display()))?;
    if tables.header.format_version != TABLES_FORMAT_VERSION {
        bail!(
            "{}: tables format version {} != expected {TABLES_FORMAT_VERSION} — rebuild with `build-tables`",
            path.display(),
            tables.header.format_version
        );
    }
    Ok(tables)
}

/// Resolves a `--tables` argument that may be either a direct path to the tables file (task 6.3's
/// documented CLI contract) or a directory containing [`TABLES_FILE_NAME`] (this crate's other
/// commands' convention, e.g. `stats --tables <dir>`) to the exact file path either way.
pub fn resolve_tables_file_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        tables_path(path)
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integers_round_trip() {
        assert_eq!(f64_to_exact_i64(0.0).unwrap(), Some(0));
        assert_eq!(f64_to_exact_i64(-0.0).unwrap(), Some(0));
        assert_eq!(f64_to_exact_i64(145524.0).unwrap(), Some(145524));
        assert_eq!(f64_to_exact_i64(-42.0).unwrap(), Some(-42));
        // Largest bodyId-scale values we actually see (~1.57e9) are nowhere near where f64
        // loses integer precision (2^53); check a value near that scale round-trips too.
        assert_eq!(f64_to_exact_i64(1_570_000_000.0).unwrap(), Some(1_570_000_000));
    }

    #[test]
    fn nan_means_missing_not_an_error() {
        assert_eq!(f64_to_exact_i64(f64::NAN).unwrap(), None);
    }

    #[test]
    fn out_of_exact_range_values_are_rejected_not_saturated() {
        // Regression: `x as i64` saturates for out-of-range floats (2^63 as i64 == i64::MAX),
        // and i64::MAX as f64 rounds back to exactly 2^63 — so a bare round-trip check alone
        // would have wrongly accepted this as "exact". It must be rejected before ever casting.
        let two_pow_63 = 2f64.powi(63);
        assert_eq!(
            two_pow_63 as i64,
            i64::MAX,
            "sanity: this is the saturating cast the bug hinged on"
        );
        assert!(f64_to_exact_i64(two_pow_63).is_err());
        assert!(f64_to_exact_i64(-two_pow_63 - 1024.0).is_err()); // similarly saturates to i64::MIN
        assert!(f64_to_exact_i64(f64::INFINITY).is_err());
        assert!(f64_to_exact_i64(f64::NEG_INFINITY).is_err());
        // A large-but-still-exactly-representable value must still be accepted.
        assert_eq!(
            f64_to_exact_i64(MAX_EXACTLY_REPRESENTABLE_F64).unwrap(),
            Some(1i64 << 53)
        );
        assert_eq!(
            f64_to_exact_i64(-MAX_EXACTLY_REPRESENTABLE_F64).unwrap(),
            Some(-(1i64 << 53))
        );
        // Just past that boundary must be rejected even though it happens to still look
        // integral (2^53 + 2 is itself exactly representable, but it's outside our conservative
        // cutoff, which is the point: we reject by magnitude, not by "does this specific value
        // happen to survive the round trip").
        assert!(f64_to_exact_i64(MAX_EXACTLY_REPRESENTABLE_F64 + 2.0).is_err());
    }

    #[test]
    fn fractional_value_is_a_hard_error() {
        assert!(f64_to_exact_i64(1.5).is_err());
        assert!(f64_to_exact_i64(-0.001).is_err());
    }

    #[test]
    fn nt_class_parses_all_known_values_and_rejects_unknown() {
        assert_eq!(NtClass::parse("acetylcholine").unwrap(), NtClass::Acetylcholine);
        assert_eq!(NtClass::parse("gaba").unwrap(), NtClass::Gaba);
        assert_eq!(NtClass::parse("unclear").unwrap(), NtClass::Unclear);
        assert!(
            NtClass::parse("ACETYLCHOLINE").is_err(),
            "source data is lowercase; an uppercase variant is a schema-drift signal, not a value to normalize silently"
        );
        assert!(NtClass::parse("").is_err());
        assert!(NtClass::parse("serotonine").is_err()); // typo, not a real category
    }

    #[test]
    fn neuron_nt_confidence_round_trips_through_milli_fixed_point() {
        let nt = NeuronNt {
            predicted_nt: Some(NtClass::Gaba),
            predicted_nt_confidence_milli: Some(934),
        };
        assert!((nt.predicted_nt_confidence().unwrap() - 0.934).abs() < 1e-6);
        assert_eq!(NeuronNt::default().predicted_nt_confidence(), None);
    }

    fn empty_tables_with_version(format_version: u32) -> ConnectomeTables {
        ConnectomeTables {
            header: TablesHeader {
                format_version,
                input_sha256: BTreeMap::new(),
            },
            dictionaries: Dictionaries::default(),
            types: Vec::new(),
            neurons: NeuronsTable::default(),
            neuron_nt: Vec::new(),
            edges: EdgesTable::default(),
        }
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let tables = empty_tables_with_version(TABLES_FORMAT_VERSION);
        let path = save_tables(&tables, dir.path()).unwrap();
        assert!(path.is_file());
        assert!(
            !dir.path().join(format!("{TABLES_FILE_NAME}.tmp")).exists(),
            "temp file must be renamed away, not left behind"
        );
        let loaded = load_tables(dir.path()).unwrap();
        assert_eq!(loaded, tables);
    }

    #[test]
    fn load_tables_rejects_a_mismatched_format_version() {
        let dir = tempfile::tempdir().unwrap();
        let stale = empty_tables_with_version(TABLES_FORMAT_VERSION + 1);
        save_tables(&stale, dir.path()).unwrap();
        let err = load_tables(dir.path()).unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("format version"),
            "expected a format-version error, got: {message}"
        );
    }

    #[test]
    fn save_tables_overwrites_a_previous_version_atomically() {
        let dir = tempfile::tempdir().unwrap();
        save_tables(&empty_tables_with_version(TABLES_FORMAT_VERSION), dir.path()).unwrap();
        let second = {
            let mut t = empty_tables_with_version(TABLES_FORMAT_VERSION);
            t.types.push(TypeRow {
                name: "X".into(),
                consensus_nt: None,
            });
            t
        };
        save_tables(&second, dir.path()).unwrap();
        let loaded = load_tables(dir.path()).unwrap();
        assert_eq!(loaded, second, "second save must fully replace the first");
    }
}
