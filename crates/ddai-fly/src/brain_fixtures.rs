//! Tiny hand-built [`Flyg`] graphs for task 7.3's own tests (encoder/decoder/world-model/brain):
//! unlike [`crate::test_fixtures`] (task 7.1/7.2, kept untouched — it hardcodes every
//! `InputVisual` neuron's receptive field to `(0.0, 0.0)` and never fills `input_channels`/
//! `output_groups`, neither of which task 7.1/7.2's own tests needed), this module's builder lets
//! a test set a real per-neuron receptive field and both of those tables — exactly what the
//! encoder/decoder need to exercise for real. `#[doc(hidden)]`, exported unconditionally (not
//! `#[cfg(test)]`-gated) so both this crate's own unit tests and its `tests/*.rs` integration
//! tests can use it, matching [`crate::test_fixtures`]'s own convention.

use crate::config::FlyConfig;
use ddai_flyg::{
    Flyg, FlygEdges, FlygHeader, FlygNeuron, FlygType, InputChannelMapping, NeuronInputTotals, NeuronRole, NtClassUsed,
    OutputGroup, OutputMember, ReceptiveField, RoleCounts, Side, Sign, SignCounts, Summary, TypePair,
};

#[derive(Debug, Clone, Copy)]
pub struct FxType {
    pub name: &'static str,
    pub sign: Sign,
}

#[derive(Debug, Clone, Copy)]
pub struct FxNeuron {
    pub type_index: u32,
    pub role: NeuronRole,
    pub side: Side,
    pub full_connectome_in: u64,
    /// `(azimuth_deg, elevation_deg)` — only meaningful (and only ever read) for `role ==
    /// InputVisual`; ignored otherwise.
    pub rf: (f32, f32),
}

#[derive(Debug, Clone, Copy)]
pub struct FxEdge {
    pub pre: u32,
    pub post: u32,
    pub synapse_count: u32,
}

/// One `[[input_channels]]` entry, by type name (resolved to `type_index` by [`build_brain_flyg`]).
#[derive(Debug, Clone)]
pub struct FxInputChannel {
    pub type_name: &'static str,
    pub channels: Vec<&'static str>,
}

/// One `[[output_groups]]` entry, by member type names (resolved to `neuron_index` by
/// [`build_brain_flyg`] — every neuron of a named type becomes a member, unless `side_filter`
/// narrows that to one side; a test that wants a specific single neuron just uses a type with
/// exactly one matching neuron).
#[derive(Debug, Clone)]
pub struct FxOutputGroup {
    pub action: &'static str,
    pub member_type_names: Vec<&'static str>,
    /// When `Some`, only members whose own `side` field equals this are included — lets a
    /// fixture split one type's `L`/`R` neurons between two named groups (e.g.
    /// `direction_left`/`direction_right`), matching how the real `.flyg`'s own `output_groups`
    /// are actually populated (task 6.3: each side's own descending neuron is assigned to its
    /// own action name, never both). `None` pools every side together (`jump`/`hook`/`fire`/
    /// `direction_stop`, review round 1's F6 tying).
    pub side_filter: Option<Side>,
}

/// Builds a valid [`Flyg`] from plain descriptions, same spirit as [`crate::test_fixtures::
/// build_flyg`] but with real receptive fields plus `input_channels`/`output_groups`. Panics (via
/// the `validate` call at the end) if the description is internally inconsistent.
pub fn build_brain_flyg(
    types: &[FxType],
    neurons: &[FxNeuron],
    edges: &[FxEdge],
    input_channels: &[FxInputChannel],
    output_groups: &[FxOutputGroup],
) -> Flyg {
    let n = neurons.len();
    let t = types.len();

    let flyg_neurons: Vec<FlygNeuron> = neurons
        .iter()
        .map(|nr| FlygNeuron {
            body_id: 0, // overwritten below with the dense index, ascending
            type_index: nr.type_index,
            role: nr.role,
            side: nr.side,
            group_id: None,
            rf: matches!(nr.role, NeuronRole::InputVisual).then_some(ReceptiveField {
                azimuth_deg: nr.rf.0,
                elevation_deg: nr.rf.1,
                is_fallback: false,
            }),
        })
        .enumerate()
        .map(|(i, mut nr)| {
            nr.body_id = i as i64;
            nr
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
    let type_index_of = |name: &str| -> u32 {
        flyg_types
            .iter()
            .position(|t| t.name == name)
            .unwrap_or_else(|| panic!("no type named '{name}' in this fixture")) as u32
    };

    let mut sorted_edges: Vec<FxEdge> = edges.to_vec();
    sorted_edges.sort_by_key(|e| (e.post, e.pre));

    let mut pair_id: Vec<((u32, u32), u32)> = Vec::new();
    let mut pair_totals: Vec<u64> = Vec::new();
    let type_pair_index_of =
        |pre_type: u32, post_type: u32, pair_id: &mut Vec<((u32, u32), u32)>, pair_totals: &mut Vec<u64>| -> usize {
            if let Some(pos) = pair_id.iter().position(|&(k, _)| k == (pre_type, post_type)) {
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

    let flyg_input_channels: Vec<InputChannelMapping> = input_channels
        .iter()
        .map(|ic| InputChannelMapping {
            type_index: type_index_of(ic.type_name),
            channels: ic.channels.iter().map(|s| s.to_string()).collect(),
        })
        .collect();

    let flyg_output_groups: Vec<OutputGroup> = output_groups
        .iter()
        .map(|og| {
            let members: Vec<OutputMember> = neurons
                .iter()
                .enumerate()
                .filter(|(_, nr)| {
                    nr.role == NeuronRole::Output
                        && og
                            .member_type_names
                            .contains(&flyg_types[nr.type_index as usize].name.as_str())
                        && og.side_filter.is_none_or(|s| s == nr.side)
                })
                .map(|(i, nr)| OutputMember {
                    neuron_index: i as u32,
                    side: nr.side,
                })
                .collect();
            OutputGroup {
                action: og.action.to_string(),
                members,
            }
        })
        .collect();

    let flyg = Flyg {
        header: FlygHeader {
            format_version: ddai_flyg::FLYG_FORMAT_VERSION,
            source_tables_sha256: "0".repeat(64),
            config_sha256: "0".repeat(64),
            generator_version: "ddai-fly brain test fixture".to_string(),
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
        input_channels: flyg_input_channels,
        output_groups: flyg_output_groups,
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

/// A small but fully-wired fixture: 2 VPN types (opponent/wall), 1 AN type (grounded), 2 hidden
/// neurons, and one output neuron per `DecoderConfig::default()` action — everything connected
/// input -> hidden -> output, so a real signal actually reaches every decoder head.
// A `vec![...]` literal here would need every field of every one of ~14 `FxNeuron`s
// spelled out positionally with no per-neuron comment anchor -- individual `.push()`
// calls (each right after its own explanatory comment) stay clearer for a fixture this
// shaped, even though clippy's default heuristic can't tell the difference from
// "just forgot the macro".
pub fn tiny_brain_flyg() -> ddai_flyg::Flyg {
    tiny_brain_flyg_with(false)
}

/// [`tiny_brain_flyg`], optionally with an `AN_HOOK` ascending neuron (the proprioception channel `own_hook`) wired
/// strongly to the `DN_HOOK` neuron, so that the own hook state in the observation moves the hook probability: what a
/// test of the masked hook view (`HookView::MaskedForHookHead`) needs, since masking changes nothing otherwise.
#[allow(clippy::vec_init_then_push)]
pub fn tiny_brain_flyg_with(own_hook: bool) -> ddai_flyg::Flyg {
    let mut type_names = vec![
        "VPN_OPP",
        "VPN_WALL",
        "AN_GROUND",
        "HID",
        "DN_LR",
        "DN_STOP",
        "DN_JUMP",
        "DN_HOOK",
        "DN_FIRE",
        "DN_AIM",
    ];
    if own_hook {
        type_names.push("AN_HOOK");
    }
    let types: Vec<FxType> = type_names
        .iter()
        .map(|&name| FxType {
            name,
            sign: Sign::Excitatory,
        })
        .collect();

    let mut neurons = Vec::new();
    // 0, 1: VPN_OPP L/R
    neurons.push(FxNeuron {
        type_index: 0,
        role: NeuronRole::InputVisual,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (-45.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 0,
        role: NeuronRole::InputVisual,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (45.0, 0.0),
    });
    // 2, 3: VPN_WALL L/R
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::InputVisual,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (-30.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 1,
        role: NeuronRole::InputVisual,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (30.0, 0.0),
    });
    // 4: AN_GROUND
    neurons.push(FxNeuron {
        type_index: 2,
        role: NeuronRole::InputAscending,
        side: Side::M,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    // 5, 6: HID x2
    for _ in 0..2 {
        neurons.push(FxNeuron {
            type_index: 3,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }
    // 7, 8: DN_LR L/R (a tied direction pair -- review round 1, F6).
    neurons.push(FxNeuron {
        type_index: 4,
        role: NeuronRole::Output,
        side: Side::L,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    neurons.push(FxNeuron {
        type_index: 4,
        role: NeuronRole::Output,
        side: Side::R,
        full_connectome_in: 1000,
        rf: (0.0, 0.0),
    });
    // 9..14: one output neuron per remaining DN type.
    for ti in 5..10 {
        neurons.push(FxNeuron {
            type_index: ti as u32,
            role: NeuronRole::Output,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
    }

    let input_indices: Vec<u32> = (0..5).collect(); // VPN_OPP x2, VPN_WALL x2, AN_GROUND
    let hidden_indices: Vec<u32> = (5..7).collect();
    let output_indices: Vec<u32> = (7..14).collect(); // one output neuron per DN type (7 DN types)

    let mut edges = Vec::new();
    for &pre in &input_indices {
        for &post in &hidden_indices {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 5,
            });
        }
    }
    for &pre in &hidden_indices {
        for &post in &output_indices {
            edges.push(FxEdge {
                pre,
                post,
                synapse_count: 5,
            });
        }
    }

    if own_hook {
        // neuron 14 = AN_HOOK (appended last, so no other index moves); neuron 11 = the DN_HOOK output
        neurons.push(FxNeuron {
            type_index: (type_names.len() - 1) as u32,
            role: NeuronRole::InputAscending,
            side: Side::M,
            full_connectome_in: 1000,
            rf: (0.0, 0.0),
        });
        edges.push(FxEdge {
            pre: 14,
            post: 11,
            synapse_count: 400,
        });
    }

    let input_channels = vec![
        FxInputChannel {
            type_name: "VPN_OPP",
            channels: vec!["opponent_position"],
        },
        FxInputChannel {
            type_name: "VPN_WALL",
            channels: vec!["walls"],
        },
    ];
    let output_groups = vec![
        FxOutputGroup {
            action: "direction_left",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::L),
        },
        FxOutputGroup {
            action: "direction_right",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::R),
        },
        FxOutputGroup {
            action: "direction_stop",
            member_type_names: vec!["DN_STOP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "jump",
            member_type_names: vec!["DN_JUMP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "hook",
            member_type_names: vec!["DN_HOOK"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "fire",
            member_type_names: vec!["DN_FIRE"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "aim",
            member_type_names: vec!["DN_AIM"],
            side_filter: None,
        },
    ];

    build_brain_flyg(&types, &neurons, &edges, &input_channels, &output_groups)
}

/// Writes the tiny fixture graph and a fly bundle for it (default parameters, `hook_view` as given) into `dir` and
/// returns `(bundle path, flyg path)`: what a test needs to exercise the real bundle loaders (the arena's
/// `ModelBrains`, the bot's `--fly-bundle`) without the local connectome data.
pub fn write_tiny_fly_bundle(
    dir: &std::path::Path,
    hook_view: crate::bc::HookView,
) -> (std::path::PathBuf, std::path::PathBuf) {
    write_tiny_fly_bundle_with(dir, hook_view, FlyConfig::default().substeps_per_decision)
}

/// [`write_tiny_fly_bundle`] with `substeps_per_decision` set (the work-clock price of a proposal is
/// `nnz x substeps x rate`, which rounds to 0 tee-ticks on the tiny graph unless the substeps are many).
pub fn write_tiny_fly_bundle_with(
    dir: &std::path::Path,
    hook_view: crate::bc::HookView,
    substeps_per_decision: u32,
) -> (std::path::PathBuf, std::path::PathBuf) {
    use crate::bc::HeadThresholds;
    use crate::bundle::{BUNDLE_FORMAT_VERSION, BundleMeta, FlyBundle, save_bundle, sha256_hex_of_file};
    use crate::decoder::{DecoderConfig, DecoderModel, DnCalibration};
    use crate::encoder::{EncoderModel, EncoderParams, ProprioceptionConfig, RayGridConfig};
    use crate::{FlyModel, FlyParams};

    let flyg = tiny_brain_flyg_with(true);
    let flyg_path = dir.join("tiny.flyg");
    ddai_flyg::save(&flyg, &flyg_path).expect("writing the fixture graph");
    let sha = sha256_hex_of_file(&flyg_path).expect("hashing the fixture graph");
    let config = FlyConfig {
        substeps_per_decision,
        ..FlyConfig::default()
    };
    let params = FlyParams::init_default(&flyg, &config, 1);
    let model = FlyModel::new(flyg, config, params.clone()).expect("fixture model");
    let encoder = EncoderModel::new(
        &model,
        RayGridConfig::default(),
        &ProprioceptionConfig {
            grounded: vec!["AN_GROUND".to_string()],
            own_hook: vec!["AN_HOOK".to_string()],
            ..ProprioceptionConfig::default()
        },
    )
    .expect("fixture encoder");
    let decoder = DecoderModel::new(&model, DecoderConfig::default()).expect("fixture decoder");
    // The hook head reads the DN_HOOK rate (which the own-hook input drives): idle -> p ~ 0.3, grabbed -> p ~ 0.8.
    let mut decoder_params = decoder.init_default_params();
    decoder_params.hook_w.fill(3.0);
    decoder_params.hook_b = -2.0;
    let brain_config_toml = {
        #[derive(serde::Serialize)]
        struct File {
            ray_grid: RayGridConfig,
            decoder: DecoderConfig,
            world_model: crate::world_model::WorldModelConfig,
            proprioception: ProprioceptionConfig,
        }
        toml::to_string(&File {
            ray_grid: RayGridConfig::default(),
            decoder: DecoderConfig::default(),
            world_model: crate::world_model::WorldModelConfig::default(),
            proprioception: ProprioceptionConfig {
                grounded: vec!["AN_GROUND".to_string()],
                own_hook: vec!["AN_HOOK".to_string()],
                ..ProprioceptionConfig::default()
            },
        })
        .expect("the fixture brain config serialises")
    };
    let bundle = FlyBundle {
        format_version: BUNDLE_FORMAT_VERSION,
        flyg_sha256: sha,
        flyg_path_hint: flyg_path.to_string_lossy().into_owned(),
        brain_config_toml,
        fly_config: config,
        fly_params: params,
        encoder_params: EncoderParams::init_default(encoder.num_params()),
        decoder_params,
        calibration: DnCalibration {
            mu: vec![0.0; model.num_outputs()],
            sigma: vec![1.0; model.num_outputs()],
        },
        meta: BundleMeta::default(),
        thresholds: HeadThresholds::default(),
        hook_view,
        hook_param: crate::bc::HookParam::Legacy,
        hook_decode: crate::bc::HookDecode::Plain,
        hook_readout: crate::hook_wide::HookReadout::Pooled,
    };
    let bundle_path = dir.join("tiny.bundle");
    save_bundle(&bundle_path, &bundle).expect("writing the fixture bundle");
    (bundle_path, flyg_path)
}
