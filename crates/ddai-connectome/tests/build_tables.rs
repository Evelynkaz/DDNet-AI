//! Offline integration test for acceptance criterion 5(a): build tiny Feather files with the
//! same column names/types subset as the real MaleCNS files (LZ4-compressed, like the real
//! files), run `build_tables_from_raw` on them, and check the resulting tables field by field
//! against hand-computed expected values. No network access.
//!
//! Fixture (see the comment blocks below for the exact rows): 8 bodies, `bodyId` 100..999,
//! mixing `Traced`/`Orphan`/missing status, one untyped Traced neuron, one Traced neuron with no
//! superclass (mirrors the ~516 real Traced bodies with no superclass, docs/research/fly-data.md
//! §6.3.2), one type (`TypeA`) whose 3 neurons disagree on `consensus_nt` (2 ACh vs 1 GABA, to
//! exercise the majority vote), one type (`TypeD`) whose only neuron has no neurotransmitter row
//! at all (consensus_nt must come out `None`, not crash), a `group` value that round-trips
//! exactly through `i64`, and a small edge set with one autapse.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::CompressionType;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};

use ddai_connectome::tables::{
    ANNOTATIONS_FILE_NAME, NEUROTRANSMITTERS_FILE_NAME, NtClass, WEIGHTS_FILE_NAME, build_tables_from_raw,
    f64_to_exact_i16, f64_to_exact_i64,
};

/// Writes `batch` to `path` as an Arrow IPC (Feather v2) file with LZ4 frame compression, the
/// same codec pyarrow's `write_feather` defaults to for the real files (requires arrow's
/// `ipc_compression` feature both to write and to read back — see Cargo.toml).
fn write_feather_lz4(path: &Path, schema: &Schema, batch: &RecordBatch) {
    let mut file = std::fs::File::create(path).unwrap();
    let options = IpcWriteOptions::try_new(8, false, arrow::ipc::MetadataVersion::V5)
        .unwrap()
        .try_with_compression(Some(CompressionType::LZ4_FRAME))
        .unwrap();
    let mut writer = FileWriter::try_new_with_options(&mut file, schema, options).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
}

fn write_annotations(dir: &Path) {
    let schema = Schema::new(vec![
        Field::new("bodyId", DataType::Int64, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("type", DataType::Utf8, true),
        Field::new("instance", DataType::Utf8, true),
        Field::new("superclass", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("subclass", DataType::Utf8, true),
        Field::new("somaSide", DataType::Utf8, true),
        Field::new("group", DataType::Float64, true),
        Field::new("assignedOlHex1", DataType::Float64, true),
        Field::new("assignedOlHex2", DataType::Float64, true),
    ]);

    // bodyId:   100            200            300               400                500      600            998   999
    // status:   Traced         Traced         Traced            Traced             Orphan   Traced         NULL  Traced
    // type:     TypeA          TypeA          TypeB              NULL               NULL     TypeA          NULL  TypeD
    // instance: "TypeA(x)_L"   "TypeA(x)_R"   "TypeB_L"           "AN_untyped"       NULL     "TypeA(y)_L"   NULL  "TypeD_M"
    // super:    descending_neuron × TypeA×3   visual_projection   ascending_neuron   NULL     descending_neuron  NULL  NULL (mirrors real "Traced with no superclass")
    // class:    C1                            C2                 NULL               NULL     C1             NULL  NULL
    // subclass: S1                            S2                 NULL               NULL     S1             NULL  NULL
    // somaSide: L / R / (600)L                L                  M                  NULL     L              NULL  NULL
    // group:    10.0 / 10.0                   NaN                NaN                NaN      NaN            NaN   NaN
    let body_id = Int64Array::from(vec![100, 200, 300, 400, 500, 600, 998, 999]);
    let status = StringArray::from(vec![
        Some("Traced"),
        Some("Traced"),
        Some("Traced"),
        Some("Traced"),
        Some("Orphan"),
        Some("Traced"),
        None,
        Some("Traced"),
    ]);
    let type_col = StringArray::from(vec![
        Some("TypeA"),
        Some("TypeA"),
        Some("TypeB"),
        None,
        None,
        Some("TypeA"),
        None,
        Some("TypeD"),
    ]);
    let instance = StringArray::from(vec![
        Some("TypeA(x)_L"),
        Some("TypeA(x)_R"),
        Some("TypeB_L"),
        Some("AN_untyped"),
        None,
        Some("TypeA(y)_L"),
        None,
        Some("TypeD_M"),
    ]);
    let superclass = StringArray::from(vec![
        Some("descending_neuron"),
        Some("descending_neuron"),
        Some("visual_projection"),
        Some("ascending_neuron"),
        None,
        Some("descending_neuron"),
        None,
        None,
    ]);
    let class = StringArray::from(vec![
        Some("C1"),
        Some("C1"),
        Some("C2"),
        None,
        None,
        Some("C1"),
        None,
        None,
    ]);
    let subclass = StringArray::from(vec![
        Some("S1"),
        Some("S1"),
        Some("S2"),
        None,
        None,
        Some("S1"),
        None,
        None,
    ]);
    let soma_side = StringArray::from(vec![
        Some("L"),
        Some("R"),
        Some("L"),
        Some("M"),
        None,
        Some("L"),
        None,
        None,
    ]);
    let group = Float64Array::from(vec![
        Some(10.0),
        Some(10.0),
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
    ]);
    // Only body 100 has hex coordinates (mirrors the real data: only optic-lobe columnar
    // neurons have them, everyone else is NaN/None).
    let ol_hex1 = Float64Array::from(vec![
        Some(12.0),
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
    ]);
    let ol_hex2 = Float64Array::from(vec![
        Some(26.0),
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
        None::<f64>,
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(body_id),
            Arc::new(status),
            Arc::new(type_col),
            Arc::new(instance),
            Arc::new(superclass),
            Arc::new(class),
            Arc::new(subclass),
            Arc::new(soma_side),
            Arc::new(group),
            Arc::new(ol_hex1),
            Arc::new(ol_hex2),
        ],
    )
    .unwrap();

    write_feather_lz4(&dir.join(ANNOTATIONS_FILE_NAME), &schema, &batch);
}

fn write_neurotransmitters(dir: &Path) {
    let schema = Schema::new(vec![
        Field::new("body", DataType::Int64, false),
        Field::new("predicted_nt", DataType::Utf8, true),
        Field::new("predicted_nt_confidence", DataType::Float64, true),
        Field::new("consensus_nt", DataType::Utf8, true),
    ]);

    // No row at all for 500 (Orphan) or 998/999 — 999 (type TypeD) deliberately has no NT row,
    // so TypeD's consensus_nt must come out `None`, and body 999 itself must show up as
    // "missing NT" (not just "unclear").
    // TypeA neurons (100, 200, 600) disagree: 2×acetylcholine vs 1×gaba -> majority is
    // acetylcholine.
    let body = Int64Array::from(vec![100, 200, 300, 400, 600]);
    let predicted_nt = StringArray::from(vec![
        Some("acetylcholine"),
        Some("acetylcholine"),
        Some("unclear"),
        Some("glutamate"),
        Some("gaba"),
    ]);
    let confidence = Float64Array::from(vec![Some(0.95), Some(0.90), Some(0.40), Some(0.70), Some(0.80)]);
    let consensus_nt = StringArray::from(vec![
        Some("acetylcholine"),
        Some("acetylcholine"),
        Some("unclear"),
        Some("glutamate"),
        Some("gaba"),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(body),
            Arc::new(predicted_nt),
            Arc::new(confidence),
            Arc::new(consensus_nt),
        ],
    )
    .unwrap();

    write_feather_lz4(&dir.join(NEUROTRANSMITTERS_FILE_NAME), &schema, &batch);
}

fn write_weights(dir: &Path) {
    let schema = Schema::new(vec![
        Field::new("body_pre", DataType::Int64, false),
        Field::new("body_post", DataType::Int64, false),
        Field::new("weight", DataType::Int64, false),
    ]);

    // 100 -> 100 is an autapse (dropped, counted separately). Every other pair is between two
    // Traced bodies, as the real traced-only file guarantees.
    let pre = Int64Array::from(vec![100, 200, 100, 300, 600, 999, 100]);
    let post = Int64Array::from(vec![200, 100, 300, 400, 999, 100, 100]);
    let weight = Int64Array::from(vec![5, 3, 10, 2, 7, 1, 4]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(pre), Arc::new(post), Arc::new(weight)],
    )
    .unwrap();

    write_feather_lz4(&dir.join(WEIGHTS_FILE_NAME), &schema, &batch);
}

#[test]
fn build_tables_matches_hand_computed_expectations() {
    let dir = tempfile::tempdir().unwrap();
    write_annotations(dir.path());
    write_neurotransmitters(dir.path());
    write_weights(dir.path());

    let tables = build_tables_from_raw(dir.path()).unwrap();

    // --- header -------------------------------------------------------------------------------
    assert_eq!(
        tables.header.format_version,
        ddai_connectome::tables::TABLES_FORMAT_VERSION
    );
    assert_eq!(tables.header.input_sha256.len(), 3);
    assert!(tables.header.input_sha256.contains_key(ANNOTATIONS_FILE_NAME));
    assert!(tables.header.input_sha256.contains_key(NEUROTRANSMITTERS_FILE_NAME));
    assert!(tables.header.input_sha256.contains_key(WEIGHTS_FILE_NAME));
    // sha256 is 32 bytes -> 64 hex chars.
    for sha in tables.header.input_sha256.values() {
        assert_eq!(sha.len(), 64, "sha256 {sha:?} is not 64 hex chars");
    }

    // --- dictionaries (every one sorted ascending) --------------------------------------------
    assert_eq!(
        tables.dictionaries.statuses,
        vec!["Orphan".to_string(), "Traced".to_string()]
    );
    assert_eq!(
        tables.dictionaries.superclasses,
        vec![
            "ascending_neuron".to_string(),
            "descending_neuron".to_string(),
            "visual_projection".to_string()
        ]
    );
    assert_eq!(tables.dictionaries.classes, vec!["C1".to_string(), "C2".to_string()]);
    assert_eq!(tables.dictionaries.subclasses, vec!["S1".to_string(), "S2".to_string()]);
    assert_eq!(
        tables.dictionaries.soma_sides,
        vec!["L".to_string(), "M".to_string(), "R".to_string()]
    );

    // --- types: sorted by name, consensus_nt via majority vote ---------------------------------
    let type_names: Vec<&str> = tables.types.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(type_names, vec!["TypeA", "TypeB", "TypeD"]);
    let type_a = &tables.types[0];
    assert_eq!(
        type_a.consensus_nt,
        Some(NtClass::Acetylcholine),
        "2 ACh vs 1 GABA must pick ACh"
    );
    let type_b = &tables.types[1];
    assert_eq!(type_b.consensus_nt, Some(NtClass::Unclear));
    let type_d = &tables.types[2];
    assert_eq!(
        type_d.consensus_nt, None,
        "TypeD's only neuron (999) has no NT row at all"
    );

    // --- neurons: dense index = bodyId ascending ------------------------------------------------
    let body_ids: Vec<i64> = tables.neurons.rows.iter().map(|n| n.body_id).collect();
    assert_eq!(body_ids, vec![100, 200, 300, 400, 500, 600, 998, 999]);
    assert_eq!(tables.neurons.rows.len(), 8);

    let idx_of = |body_id: i64| body_ids.iter().position(|&b| b == body_id).unwrap();
    let status_name = |id: Option<u16>| id.map(|i| tables.dictionaries.statuses[i as usize].as_str());
    let superclass_name = |id: Option<u16>| id.map(|i| tables.dictionaries.superclasses[i as usize].as_str());
    let type_name = |id: Option<u32>| id.map(|i| tables.types[i as usize].name.as_str());

    let n100 = &tables.neurons.rows[idx_of(100)];
    assert_eq!(status_name(n100.status), Some("Traced"));
    assert_eq!(type_name(n100.type_id), Some("TypeA"));
    assert_eq!(superclass_name(n100.superclass), Some("descending_neuron"));
    assert_eq!(n100.instance, Some("TypeA(x)_L".to_string()));
    assert_eq!(n100.group, Some(10), "group=10.0 must convert exactly to i64 10");

    let n300 = &tables.neurons.rows[idx_of(300)];
    assert_eq!(n300.group, None, "NaN group must convert to None, not an error or 0");

    assert_eq!(n100.ol_hex1, Some(12), "body 100 is the only one with hex coordinates");
    assert_eq!(n100.ol_hex2, Some(26));
    assert_eq!(n300.ol_hex1, None, "NaN hex must convert to None, not an error or 0");
    assert_eq!(n300.ol_hex2, None);

    let n400 = &tables.neurons.rows[idx_of(400)];
    assert_eq!(type_name(n400.type_id), None, "body 400 has no `type` in the source");
    assert_eq!(superclass_name(n400.superclass), Some("ascending_neuron"));

    let n998 = &tables.neurons.rows[idx_of(998)];
    assert_eq!(status_name(n998.status), None, "body 998 has a null `status`");

    let n999 = &tables.neurons.rows[idx_of(999)];
    assert_eq!(status_name(n999.status), Some("Traced"));
    assert_eq!(
        superclass_name(n999.superclass),
        None,
        "body 999 is Traced but has no superclass (mirrors real data)"
    );

    // --- per-neuron predicted NT, parallel to `neurons.rows` ------------------------------------
    assert_eq!(tables.neuron_nt.len(), tables.neurons.rows.len());
    let nt_of = |body_id: i64| tables.neuron_nt[idx_of(body_id)];
    assert_eq!(nt_of(100).predicted_nt, Some(NtClass::Acetylcholine));
    assert!((nt_of(100).predicted_nt_confidence().unwrap() - 0.95).abs() < 1e-3);
    assert_eq!(nt_of(300).predicted_nt, Some(NtClass::Unclear));
    assert_eq!(nt_of(500).predicted_nt, None, "Orphan body 500 has no NT row");
    assert_eq!(nt_of(500).predicted_nt_confidence(), None);
    assert_eq!(
        nt_of(999).predicted_nt,
        None,
        "Traced body 999 has no NT row either — missing, not unclear"
    );

    // --- edges: only among Traced bodies, sorted by (post_idx, pre_idx), autapse dropped -------
    assert_eq!(tables.edges.autapses_dropped, 1);
    assert_eq!(tables.edges.edges.len(), 6);
    let expected_edges: Vec<(i64, i64, u32)> = vec![
        (200, 100, 3), // post=100 group: pre=200 before pre=999
        (999, 100, 1),
        (100, 200, 5),
        (100, 300, 10),
        (300, 400, 2),
        (600, 999, 7),
    ];
    let actual_edges: Vec<(i64, i64, u32)> = tables
        .edges
        .edges
        .iter()
        .map(|e| (body_ids[e.pre_idx as usize], body_ids[e.post_idx as usize], e.weight))
        .collect();
    assert_eq!(actual_edges, expected_edges);
    // Sort order is on (post_idx, pre_idx), i.e. dense indices, not on bodyId — check that
    // directly too, since the mapping above could theoretically mask a sort bug.
    for pair in tables.edges.edges.windows(2) {
        let key = |e: &ddai_connectome::tables::Edge| (e.post_idx, e.pre_idx);
        assert!(
            key(&pair[0]) < key(&pair[1]),
            "edges must be strictly sorted by (post_idx, pre_idx)"
        );
    }

    // --- total_input / total_output: over ALL traced synapses, INCLUDING autapse weight (the
    // self-loop 100->100 weight=4 is dropped from the edge *list* but still counts as 4 real
    // input synapses and 4 real output synapses on body 100) --------------------------------
    let total_input = |body_id: i64| tables.neurons.total_input[idx_of(body_id)];
    let total_output = |body_id: i64| tables.neurons.total_output[idx_of(body_id)];
    assert_eq!(total_input(100), 8, "3 from 200 + 1 from 999 + 4 autapse");
    assert_eq!(total_output(100), 19, "5 to 200 + 10 to 300 + 4 autapse");
    assert_eq!(total_input(200), 5);
    assert_eq!(total_output(200), 3);
    assert_eq!(total_input(300), 10);
    assert_eq!(total_output(300), 2);
    assert_eq!(total_input(400), 2);
    assert_eq!(total_output(400), 0);
    assert_eq!(total_input(500), 0);
    assert_eq!(total_output(500), 0);
    assert_eq!(total_input(600), 0);
    assert_eq!(total_output(600), 7);
    assert_eq!(total_input(999), 7);
    assert_eq!(total_output(999), 1);
}

#[test]
fn build_tables_rejects_a_fractional_group_value() {
    // Same shape as the main fixture but with exactly one row, whose `group` is 1.5 — not
    // integral, so it must not silently truncate to 1.
    let dir = tempfile::tempdir().unwrap();

    let schema = Schema::new(vec![
        Field::new("bodyId", DataType::Int64, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("type", DataType::Utf8, true),
        Field::new("instance", DataType::Utf8, true),
        Field::new("superclass", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("subclass", DataType::Utf8, true),
        Field::new("somaSide", DataType::Utf8, true),
        Field::new("group", DataType::Float64, true),
        Field::new("assignedOlHex1", DataType::Float64, true),
        Field::new("assignedOlHex2", DataType::Float64, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec![Some("Traced")])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(Float64Array::from(vec![Some(1.5)])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.path().join(ANNOTATIONS_FILE_NAME), &schema, &batch);

    // Minimal, empty-of-rows NT and weights files with the right schema (zero rows is enough:
    // `build_tables_from_raw` must fail while still reading `body-annotations`, before it would
    // even need to look at these).
    let nt_schema = Schema::new(vec![
        Field::new("body", DataType::Int64, false),
        Field::new("predicted_nt", DataType::Utf8, true),
        Field::new("predicted_nt_confidence", DataType::Float64, true),
        Field::new("consensus_nt", DataType::Utf8, true),
    ]);
    let nt_batch = RecordBatch::try_new(
        Arc::new(nt_schema.clone()),
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.path().join(NEUROTRANSMITTERS_FILE_NAME), &nt_schema, &nt_batch);

    let weights_schema = Schema::new(vec![
        Field::new("body_pre", DataType::Int64, false),
        Field::new("body_post", DataType::Int64, false),
        Field::new("weight", DataType::Int64, false),
    ]);
    let weights_batch = RecordBatch::try_new(
        Arc::new(weights_schema.clone()),
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.path().join(WEIGHTS_FILE_NAME), &weights_schema, &weights_batch);

    let err = build_tables_from_raw(dir.path()).unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("does not convert exactly"),
        "expected an exact-conversion error, got: {message}"
    );
}

/// Minimal `body-annotations` fixture: `(bodyId, status)` pairs, everything else null. Shares
/// the schema with [`write_annotations`].
fn write_minimal_annotations(dir: &Path, rows: &[(i64, Option<&str>)]) {
    let schema = Schema::new(vec![
        Field::new("bodyId", DataType::Int64, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("type", DataType::Utf8, true),
        Field::new("instance", DataType::Utf8, true),
        Field::new("superclass", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("subclass", DataType::Utf8, true),
        Field::new("somaSide", DataType::Utf8, true),
        Field::new("group", DataType::Float64, true),
        Field::new("assignedOlHex1", DataType::Float64, true),
        Field::new("assignedOlHex2", DataType::Float64, true),
    ]);
    let n = rows.len();
    let body_id = Int64Array::from(rows.iter().map(|(b, _)| *b).collect::<Vec<_>>());
    let status = StringArray::from(rows.iter().map(|(_, s)| *s).collect::<Vec<_>>());
    let none_str = || StringArray::from(vec![None::<&str>; n]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(body_id),
            Arc::new(status),
            Arc::new(none_str()),
            Arc::new(none_str()),
            Arc::new(none_str()),
            Arc::new(none_str()),
            Arc::new(none_str()),
            Arc::new(none_str()),
            Arc::new(Float64Array::from(vec![None::<f64>; n])),
            Arc::new(Float64Array::from(vec![None::<f64>; n])),
            Arc::new(Float64Array::from(vec![None::<f64>; n])),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.join(ANNOTATIONS_FILE_NAME), &schema, &batch);
}

/// Zero-row `body-neurotransmitters` fixture with the right schema.
fn write_empty_neurotransmitters(dir: &Path) {
    let schema = Schema::new(vec![
        Field::new("body", DataType::Int64, false),
        Field::new("predicted_nt", DataType::Utf8, true),
        Field::new("predicted_nt_confidence", DataType::Float64, true),
        Field::new("consensus_nt", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.join(NEUROTRANSMITTERS_FILE_NAME), &schema, &batch);
}

/// Minimal `connectome-weights` fixture: `(body_pre, body_post, weight)` rows.
fn write_minimal_weights(dir: &Path, edges: &[(i64, i64, i64)]) {
    let schema = Schema::new(vec![
        Field::new("body_pre", DataType::Int64, false),
        Field::new("body_post", DataType::Int64, false),
        Field::new("weight", DataType::Int64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(edges.iter().map(|(a, _, _)| *a).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(edges.iter().map(|(_, b, _)| *b).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(edges.iter().map(|(_, _, w)| *w).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.join(WEIGHTS_FILE_NAME), &schema, &batch);
}

#[test]
fn build_tables_rejects_an_edge_with_a_non_traced_endpoint() {
    // The traced-only weights file is *supposed* to only ever reference Traced bodies; this
    // checks that assumption is verified per-row rather than trusted. Body 2 is Orphan, so the
    // 1 -> 2 edge below violates that assumption and must be a hard error, not silently accepted.
    let dir = tempfile::tempdir().unwrap();
    write_minimal_annotations(dir.path(), &[(1, Some("Traced")), (2, Some("Orphan"))]);
    write_empty_neurotransmitters(dir.path());
    write_minimal_weights(dir.path(), &[(1, 2, 5)]);

    let err = build_tables_from_raw(dir.path()).unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("non-Traced"),
        "expected a non-Traced-endpoint error, got: {message}"
    );
}

#[test]
fn build_tables_accepts_edges_between_traced_endpoints_only() {
    // Sanity check for the test above: the same shape, but both endpoints Traced, must succeed.
    let dir = tempfile::tempdir().unwrap();
    write_minimal_annotations(dir.path(), &[(1, Some("Traced")), (2, Some("Traced"))]);
    write_empty_neurotransmitters(dir.path());
    write_minimal_weights(dir.path(), &[(1, 2, 5)]);

    let tables = build_tables_from_raw(dir.path()).unwrap();
    assert_eq!(tables.edges.edges.len(), 1);
}

#[test]
fn build_tables_treats_a_non_null_nan_confidence_as_missing() {
    // `predicted_nt_confidence` being non-null but holding a NaN payload is a different thing
    // from an Arrow-null cell, but must be treated the same way (missing), not as "0%
    // confidence" (which is what `NaN as u16` would silently produce downstream).
    let dir = tempfile::tempdir().unwrap();
    write_minimal_annotations(dir.path(), &[(1, Some("Traced"))]);

    let nt_schema = Schema::new(vec![
        Field::new("body", DataType::Int64, false),
        Field::new("predicted_nt", DataType::Utf8, true),
        Field::new("predicted_nt_confidence", DataType::Float64, true),
        Field::new("consensus_nt", DataType::Utf8, true),
    ]);
    let nt_batch = RecordBatch::try_new(
        Arc::new(nt_schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec![Some("acetylcholine")])),
            Arc::new(Float64Array::from(vec![Some(f64::NAN)])), // non-null, but NaN
            Arc::new(StringArray::from(vec![Some("acetylcholine")])),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.path().join(NEUROTRANSMITTERS_FILE_NAME), &nt_schema, &nt_batch);
    write_minimal_weights(dir.path(), &[]);

    let tables = build_tables_from_raw(dir.path()).unwrap();
    assert_eq!(tables.neuron_nt.len(), 1);
    assert_eq!(tables.neuron_nt[0].predicted_nt, Some(NtClass::Acetylcholine));
    assert_eq!(
        tables.neuron_nt[0].predicted_nt_confidence_milli, None,
        "a non-null NaN confidence must be stored as missing, not as 0"
    );
}

#[test]
fn f64_to_exact_i64_is_exposed_and_used_by_the_build() {
    // Direct check of the pure helper too (belt-and-suspenders with the end-to-end test above).
    assert_eq!(f64_to_exact_i64(10.0).unwrap(), Some(10));
    assert!(f64_to_exact_i64(10.5).is_err());
}

#[test]
fn f64_to_exact_i16_is_exposed_and_rejects_fractional_and_out_of_range() {
    assert_eq!(f64_to_exact_i16(12.0).unwrap(), Some(12));
    assert_eq!(f64_to_exact_i16(f64::NAN).unwrap(), None);
    assert!(
        f64_to_exact_i16(12.5).is_err(),
        "fractional hex coordinate is a hard error"
    );
    assert!(
        f64_to_exact_i16(100_000.0).is_err(),
        "value outside i16's range must be rejected, not silently truncated"
    );
}

#[test]
fn build_tables_rejects_an_out_of_range_hex_coordinate() {
    // Same idea as `build_tables_rejects_a_fractional_group_value`, but for `assignedOlHex1`:
    // a value that fits in f64-exact-integer range but not in i16 must be a hard error, not a
    // silent truncation.
    let dir = tempfile::tempdir().unwrap();
    let schema = Schema::new(vec![
        Field::new("bodyId", DataType::Int64, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("type", DataType::Utf8, true),
        Field::new("instance", DataType::Utf8, true),
        Field::new("superclass", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("subclass", DataType::Utf8, true),
        Field::new("somaSide", DataType::Utf8, true),
        Field::new("group", DataType::Float64, true),
        Field::new("assignedOlHex1", DataType::Float64, true),
        Field::new("assignedOlHex2", DataType::Float64, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec![Some("Traced")])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
            Arc::new(Float64Array::from(vec![Some(100_000.0)])),
            Arc::new(Float64Array::from(vec![None::<f64>])),
        ],
    )
    .unwrap();
    write_feather_lz4(&dir.path().join(ANNOTATIONS_FILE_NAME), &schema, &batch);
    write_empty_neurotransmitters(dir.path());
    write_minimal_weights(dir.path(), &[]);

    let err = build_tables_from_raw(dir.path()).unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("does not fit in i16"),
        "expected an i16-range error, got: {message}"
    );
}
