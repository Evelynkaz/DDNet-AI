//! Offline tests for acceptance criterion 8's second half: `.flyg` round-trips through
//! save/load byte-for-byte-equivalent data, and `load`/`validate` reject a range of corrupted
//! files rather than silently accepting or panicking on them.

use ddai_flyg::{
    Flyg, FlygEdges, FlygError, FlygHeader, FlygNeuron, FlygType, InputChannelMapping, NeuronInputTotals, NeuronRole,
    NtClassUsed, OutputGroup, OutputMember, ReceptiveField, RoleCounts, Side, Sign, SignCounts, Summary, TypePair,
    load, save, validate,
};

/// A tiny but structurally complete, valid `Flyg`: 4 neurons (1 visual input, 1 ascending input,
/// 1 hidden, 1 output), 2 types, one edge input->hidden and one hidden->output, one output group.
/// Every test below starts from this and corrupts exactly one thing.
fn tiny_valid_flyg() -> Flyg {
    let neurons = vec![
        FlygNeuron {
            body_id: 100,
            type_index: 0, // VisualType
            role: NeuronRole::InputVisual,
            side: Side::L,
            group_id: Some(10),
            rf: Some(ReceptiveField {
                azimuth_deg: -30.0,
                elevation_deg: 5.0,
                is_fallback: false,
            }),
        },
        FlygNeuron {
            body_id: 200,
            type_index: 1, // AscendingType (doubles as a generic "other" type for hidden below too)
            role: NeuronRole::InputAscending,
            side: Side::R,
            group_id: None,
            rf: None,
        },
        FlygNeuron {
            body_id: 300,
            type_index: 1,
            role: NeuronRole::Hidden,
            side: Side::M,
            group_id: None,
            rf: None,
        },
        FlygNeuron {
            body_id: 400,
            type_index: 1,
            role: NeuronRole::Output,
            side: Side::L,
            group_id: Some(40),
            rf: None,
        },
    ];
    let types = vec![
        FlygType {
            name: "VisualType".into(),
            superclass: "visual_projection".into(),
            class: "".into(),
            nt_class_used: NtClassUsed::Acetylcholine,
            sign: Sign::Excitatory,
            nt_confidence: 1.0,
            uncertain: false,
            neuron_count: 1,
        },
        FlygType {
            name: "OtherType".into(),
            superclass: "ascending_neuron".into(),
            class: "".into(),
            nt_class_used: NtClassUsed::Gaba,
            sign: Sign::Inhibitory,
            nt_confidence: 0.9,
            uncertain: false,
            neuron_count: 3,
        },
    ];
    // post=2 (hidden) <- pre=0 (visual input), post=3 (output) <- pre=2 (hidden).
    let edges = FlygEdges {
        row_start: vec![0, 0, 0, 1, 2],
        pre_index: vec![0, 2],
        synapse_count: vec![7, 3],
        type_pair_index: vec![0, 1],
    };
    let type_pairs = vec![
        TypePair {
            pre_type: 0,
            post_type: 1,
            total_synapses: 7,
            shared_param_id: 0,
        },
        TypePair {
            pre_type: 1,
            post_type: 1,
            total_synapses: 3,
            shared_param_id: 1,
        },
    ];
    Flyg {
        header: FlygHeader {
            format_version: ddai_flyg::FLYG_FORMAT_VERSION,
            source_tables_sha256: "a".repeat(64),
            config_sha256: "b".repeat(64),
            generator_version: "test".into(),
        },
        neurons,
        types,
        edges,
        neuron_input_totals: NeuronInputTotals {
            full_connectome: vec![50, 20, 30, 40],
            in_subgraph: vec![0, 0, 7, 3],
        },
        type_pairs,
        input_channels: vec![InputChannelMapping {
            type_index: 0,
            channels: vec!["opponent_position".into()],
        }],
        output_groups: vec![OutputGroup {
            action: "direction_left".into(),
            members: vec![OutputMember {
                neuron_index: 3,
                side: Side::L,
            }],
        }],
        summary: Summary {
            neurons_by_role: RoleCounts {
                input_visual: 1,
                input_ascending: 1,
                hidden: 1,
                output: 1,
            },
            num_types: 2,
            num_edges: 2,
            num_type_pairs: 2,
            shared_param_count: 2,
            sign_counts: SignCounts {
                excitatory: 1,
                inhibitory: 1,
                neutral: 0,
            },
            uncertain_types: 0,
            rf_fallback_count: 0,
        },
    }
}

#[test]
fn tiny_valid_flyg_passes_validation() {
    validate(&tiny_valid_flyg()).expect("the fixture itself must be valid");
}

#[test]
fn save_then_load_round_trips_byte_identically() {
    let flyg = tiny_valid_flyg();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.flyg");
    save(&flyg, &path).unwrap();
    assert!(path.is_file());
    assert!(
        !dir.path().join("test.flyg.tmp").exists(),
        "temp file must be renamed away, not left behind"
    );
    let loaded = load(&path).unwrap();
    assert_eq!(loaded, flyg);

    // Byte-identical on a second save too (acceptance criterion 7: running the builder twice
    // gives a byte-identical file) — this crate's own save() must be a pure function of the
    // data, with no embedded timestamps/random ids.
    let path2 = dir.path().join("test2.flyg");
    save(&flyg, &path2).unwrap();
    let bytes1 = std::fs::read(&path).unwrap();
    let bytes2 = std::fs::read(&path2).unwrap();
    assert_eq!(
        bytes1, bytes2,
        "encoding the same Flyg twice must produce identical bytes"
    );
}

#[test]
fn load_rejects_a_mismatched_format_version() {
    let mut flyg = tiny_valid_flyg();
    flyg.header.format_version = ddai_flyg::FLYG_FORMAT_VERSION + 1;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.flyg");
    save(&flyg, &path).unwrap();
    let err = load(&path).unwrap_err();
    assert!(
        matches!(err, FlygError::FormatVersionMismatch { .. }),
        "expected a format-version error, got: {err}"
    );
}

#[test]
fn load_rejects_truncated_bytes() {
    let flyg = tiny_valid_flyg();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.flyg");
    save(&flyg, &path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&path, &bytes).unwrap();

    let err = load(&path).unwrap_err();
    assert!(
        matches!(err, FlygError::Zstd(_) | FlygError::Decode(_)),
        "expected zstd or postcard to reject truncated bytes, got: {err}"
    );
}

#[test]
fn load_rejects_flipped_bytes() {
    let flyg = tiny_valid_flyg();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.flyg");
    save(&flyg, &path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    // Flip a handful of bytes past the zstd frame header, inside the compressed payload.
    for b in bytes.iter_mut().skip(20).take(10) {
        *b ^= 0xff;
    }
    std::fs::write(&path, &bytes).unwrap();

    // Either zstd's checksum/frame validation catches it, or it decodes to nonsense that
    // postcard or `validate` then rejects — any of those is an acceptable, non-panicking
    // rejection. What must NOT happen is `load` returning `Ok` (silently wrong data).
    let result = load(&path);
    assert!(result.is_err(), "flipped bytes must not be silently accepted");
}

#[test]
fn validate_rejects_out_of_range_type_index() {
    let mut flyg = tiny_valid_flyg();
    flyg.neurons[0].type_index = 99;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("type_index"), "{err}");
}

#[test]
fn validate_rejects_nan_receptive_field() {
    let mut flyg = tiny_valid_flyg();
    flyg.neurons[0].rf = Some(ReceptiveField {
        azimuth_deg: f32::NAN,
        elevation_deg: 0.0,
        is_fallback: false,
    });
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("non-finite"), "{err}");
}

#[test]
fn validate_rejects_infinite_receptive_field() {
    // F5 (review round 1): ±inf must be rejected too, not only NaN.
    let mut flyg = tiny_valid_flyg();
    flyg.neurons[0].rf = Some(ReceptiveField {
        azimuth_deg: f32::INFINITY,
        elevation_deg: 0.0,
        is_fallback: false,
    });
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("non-finite"), "{err}");

    let mut flyg2 = tiny_valid_flyg();
    flyg2.neurons[0].rf = Some(ReceptiveField {
        azimuth_deg: 0.0,
        elevation_deg: f32::NEG_INFINITY,
        is_fallback: false,
    });
    let err2 = validate(&flyg2).unwrap_err();
    assert!(err2.0.contains("non-finite"), "{err2}");
}

#[test]
fn validate_rejects_an_edge_whose_type_pair_does_not_match_its_neurons() {
    // F5 (review round 1): type_pair_index pointing at a pair whose (pre_type, post_type) don't
    // match the edge's actual endpoints must be rejected, not silently accepted.
    let mut flyg = tiny_valid_flyg();
    // Edge 0 is (pre=0 -> post=2), types (0, 1), correctly at type_pairs[0]. Point it at
    // type_pairs[1] instead, which is (1, 1) — a mismatch on pre_type.
    flyg.edges.type_pair_index[0] = 1;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("actual types"), "{err}");
}

#[test]
fn validate_rejects_rf_on_a_non_visual_neuron() {
    let mut flyg = tiny_valid_flyg();
    flyg.neurons[1].rf = Some(ReceptiveField {
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        is_fallback: false,
    });
    let err = validate(&flyg).unwrap_err();
    assert!(
        err.0.contains("InputAscending") || err.0.contains("receptive field"),
        "{err}"
    );
}

#[test]
fn validate_rejects_missing_rf_on_a_visual_neuron() {
    let mut flyg = tiny_valid_flyg();
    flyg.neurons[0].rf = None;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("InputVisual"), "{err}");
}

#[test]
fn validate_rejects_bad_row_start_length() {
    let mut flyg = tiny_valid_flyg();
    flyg.edges.row_start.pop();
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("row_start.len()"), "{err}");
}

#[test]
fn validate_rejects_non_monotonic_row_start() {
    let mut flyg = tiny_valid_flyg();
    flyg.edges.row_start = vec![0, 0, 5, 1, 2]; // 5 then 1: not non-decreasing
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("non-decreasing"), "{err}");
}

#[test]
fn validate_rejects_out_of_range_pre_index() {
    let mut flyg = tiny_valid_flyg();
    flyg.edges.pre_index[0] = 99;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("pre_index"), "{err}");
}

#[test]
fn validate_rejects_an_autapse() {
    let mut flyg = tiny_valid_flyg();
    // Make post=2's edge point at itself instead of pre=0.
    flyg.edges.pre_index[0] = 2;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("autapse"), "{err}");
}

#[test]
fn validate_rejects_zero_synapse_count() {
    let mut flyg = tiny_valid_flyg();
    flyg.edges.synapse_count[0] = 0;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("synapse_count"), "{err}");
}

#[test]
fn validate_rejects_unsorted_row() {
    let mut flyg = tiny_valid_flyg();
    // Give post=3 two edges out of ascending order.
    flyg.edges.row_start = vec![0, 0, 0, 1, 3];
    flyg.edges.pre_index = vec![0, 2, 0];
    flyg.edges.synapse_count = vec![7, 3, 1];
    flyg.edges.type_pair_index = vec![0, 1, 0];
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("ascending"), "{err}");
}

#[test]
fn validate_rejects_out_of_range_type_pair_index() {
    let mut flyg = tiny_valid_flyg();
    flyg.edges.type_pair_index[0] = 99;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("type_pair_index"), "{err}");
}

#[test]
fn validate_rejects_output_group_pointing_at_a_non_output_neuron() {
    let mut flyg = tiny_valid_flyg();
    flyg.output_groups[0].members[0].neuron_index = 0; // neuron 0 is InputVisual, not Output
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("Output"), "{err}");
}

#[test]
fn validate_rejects_input_channel_on_an_ascending_type() {
    // Input channels are "per visual input type" specifically (acceptance criterion 1) — OtherType
    // (type_index 1) only has an InputAscending neuron, no InputVisual one, so it must be rejected
    // even though it does have *some* input-role neuron.
    let mut flyg = tiny_valid_flyg();
    flyg.input_channels[0].type_index = 1;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("input_channels"), "{err}");
}

#[test]
fn validate_rejects_input_channel_on_a_type_with_no_neurons_at_all() {
    let mut flyg = tiny_valid_flyg();
    flyg.types.push(FlygType {
        name: "Unused".into(),
        superclass: "".into(),
        class: "".into(),
        nt_class_used: NtClassUsed::Unknown,
        sign: Sign::Neutral,
        nt_confidence: 0.0,
        uncertain: true,
        neuron_count: 0,
    });
    flyg.summary.num_types = flyg.types.len() as u32;
    flyg.summary.uncertain_types += 1;
    flyg.input_channels[0].type_index = 2;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("input_channels"), "{err}");
}

#[test]
fn validate_rejects_neuron_count_mismatch() {
    let mut flyg = tiny_valid_flyg();
    flyg.types[0].neuron_count = 5;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("neuron_count"), "{err}");
}

#[test]
fn validate_rejects_nt_confidence_out_of_range() {
    let mut flyg = tiny_valid_flyg();
    flyg.types[0].nt_confidence = 1.5;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("nt_confidence"), "{err}");
}

#[test]
fn validate_rejects_summary_mismatch() {
    let mut flyg = tiny_valid_flyg();
    flyg.summary.num_edges = 99;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("num_edges"), "{err}");
}

#[test]
fn validate_rejects_in_subgraph_exceeding_full_connectome_total() {
    let mut flyg = tiny_valid_flyg();
    flyg.neuron_input_totals.in_subgraph[0] = 999;
    let err = validate(&flyg).unwrap_err();
    assert!(err.0.contains("in_subgraph"), "{err}");
}

#[test]
fn load_on_a_saved_then_hand_corrupted_flyg_is_rejected_end_to_end() {
    // Ties save/load together with validate: a file that decodes structurally (valid postcard)
    // but whose *content* is invalid must still be rejected by `load`, not just by a standalone
    // `validate` call.
    let mut flyg = tiny_valid_flyg();
    flyg.edges.pre_index[0] = 99;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.flyg");
    save(&flyg, &path).unwrap(); // save() itself does not validate (see its docs)
    let err = load(&path).unwrap_err();
    assert!(
        matches!(err, FlygError::Validation(_)),
        "expected load() to run validation and reject this, got: {err}"
    );
}
