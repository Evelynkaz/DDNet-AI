//! Per-type synapse sign (acceptance criterion 4): `consensus_nt`, falling back to a
//! confidence-weighted majority vote of `predicted_nt` over the type's neurons when
//! `consensus_nt` is missing or `unclear`.

use ddai_flyg::{NtClassUsed, Sign};

use super::config::NtSignParams;
use crate::tables::{ConnectomeTables, NtClass};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeSign {
    pub nt_class_used: NtClassUsed,
    pub sign: Sign,
    pub confidence: f32,
    pub uncertain: bool,
}

/// The 8 [`NtClass`] variants in their declaration order — matches `NtClass`'s `as usize` cast
/// (a fieldless enum's discriminant is its declaration position unless overridden, and `NtClass`
/// declares no explicit discriminants), used as the fixed vote-bucket index order below.
const NT_CLASS_ORDER: [NtClass; 8] = [
    NtClass::Acetylcholine,
    NtClass::Glutamate,
    NtClass::Gaba,
    NtClass::Histamine,
    NtClass::Dopamine,
    NtClass::Serotonin,
    NtClass::Octopamine,
    NtClass::Unclear,
];

fn nt_class_to_used(nt: NtClass) -> NtClassUsed {
    match nt {
        NtClass::Acetylcholine => NtClassUsed::Acetylcholine,
        NtClass::Glutamate => NtClassUsed::Glutamate,
        NtClass::Gaba => NtClassUsed::Gaba,
        NtClass::Histamine => NtClassUsed::Histamine,
        // Dopamine/serotonin/octopamine (and tyramine, which the MaleCNS NT file does not
        // actually predict as its own class — see this crate's README) are all "modulatory":
        // excluded from fast transmission by default, per FLY.md §4/§7.3 and D-013.
        NtClass::Dopamine | NtClass::Serotonin | NtClass::Octopamine => NtClassUsed::Modulatory,
        NtClass::Unclear => NtClassUsed::Unknown,
    }
}

fn sign_from_raw(raw: i8) -> Sign {
    match raw {
        1 => Sign::Excitatory,
        -1 => Sign::Inhibitory,
        _ => Sign::Neutral,
    }
}

fn sign_for_used(used: NtClassUsed, cfg: &NtSignParams) -> Sign {
    let raw = match used {
        NtClassUsed::Acetylcholine => cfg.ach_sign,
        NtClassUsed::Gaba => cfg.gaba_sign,
        NtClassUsed::Glutamate => cfg.glu_sign,
        NtClassUsed::Histamine => cfg.his_sign,
        NtClassUsed::Modulatory => cfg.modulatory_sign,
        NtClassUsed::Unknown => 0,
    };
    sign_from_raw(raw)
}

/// Computes one [`TypeSign`] per entry of `tables.types` (same index), for every type — callers
/// filter down to the types actually present in the subgraph.
pub fn compute_type_signs(tables: &ConnectomeTables, cfg: &NtSignParams) -> Vec<TypeSign> {
    let traced_id = tables
        .dictionaries
        .statuses
        .iter()
        .position(|s| s == "Traced")
        .map(|i| i as u16);

    // Confidence-weighted vote totals per (type, raw NT class), plus how many neurons of that
    // type cast *any* vote at all — built in one pass over all Traced, typed neurons with an NT
    // row (see `NT_CLASS_ORDER` for the bucket order). `vote_count`, not the weighted sum, is
    // the confidence denominator below: dividing by the weighted sum instead would make a
    // *single* voting neuron's winning share trivially 1.0 regardless of how low its own
    // `predicted_nt_confidence` actually was (it is, definitionally, 100% of a one-neuron vote) —
    // dividing by how many neurons actually voted instead means a lone low-confidence vote (or a
    // handful of neurons that disagree) correctly drags the type's reported confidence down.
    let mut weighted_votes: Vec<[f64; 8]> = vec![[0.0; 8]; tables.types.len()];
    let mut vote_count: Vec<u32> = vec![0; tables.types.len()];
    for (idx, row) in tables.neurons.rows.iter().enumerate() {
        if row.status != traced_id || traced_id.is_none() {
            continue;
        }
        let Some(type_id) = row.type_id else { continue };
        let Some(nt) = tables.neuron_nt[idx].predicted_nt else {
            continue;
        };
        let weight = f64::from(tables.neuron_nt[idx].predicted_nt_confidence().unwrap_or(0.0));
        weighted_votes[type_id as usize][nt as usize] += weight;
        vote_count[type_id as usize] += 1;
    }

    tables
        .types
        .iter()
        .enumerate()
        .map(|(type_id, trow)| {
            if let Some(nt) = trow.consensus_nt
                && nt != NtClass::Unclear
            {
                // Curated consensus, not itself `unclear`: treated as fully confident, per D-013
                // ("consensus_nt приоритетно"). A modulatory/unknown mapping's resulting `sign ==
                // Neutral` here is a deliberate exclusion, not uncertainty.
                let used = nt_class_to_used(nt);
                return TypeSign {
                    nt_class_used: used,
                    sign: sign_for_used(used, cfg),
                    confidence: 1.0,
                    uncertain: false,
                };
            }

            // Fallback: confidence-weighted majority vote over predicted_nt. Ties (and the
            // "no votes at all" case) resolve deterministically to `NT_CLASS_ORDER`'s first
            // entry — using a strict `>` comparison below keeps the first-seen maximum on ties.
            let votes = &weighted_votes[type_id];
            let n_votes = vote_count[type_id];
            if n_votes == 0 {
                return TypeSign {
                    nt_class_used: NtClassUsed::Unknown,
                    sign: Sign::Neutral,
                    confidence: 0.0,
                    uncertain: true,
                };
            }
            let mut best_i = 0usize;
            let mut best_w = votes[0];
            for (i, &w) in votes.iter().enumerate().skip(1) {
                if w > best_w {
                    best_w = w;
                    best_i = i;
                }
            }
            let winning_nt = NT_CLASS_ORDER[best_i];
            let confidence = (best_w / f64::from(n_votes)) as f32;
            let used = nt_class_to_used(winning_nt);
            let uncertain = confidence < cfg.sign_confidence_threshold || winning_nt == NtClass::Unclear;
            let sign = if uncertain && !cfg.use_best_guess_when_uncertain {
                Sign::Neutral
            } else {
                sign_for_used(used, cfg)
            };
            TypeSign {
                nt_class_used: used,
                sign,
                confidence,
                uncertain,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{Dictionaries, EdgesTable, NeuronNt, NeuronRow, NeuronsTable, TablesHeader, TypeRow};
    use std::collections::BTreeMap;

    fn base_cfg() -> NtSignParams {
        NtSignParams::default()
    }

    fn tables_with(types: Vec<TypeRow>, rows: Vec<NeuronRow>, neuron_nt: Vec<NeuronNt>) -> ConnectomeTables {
        ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: BTreeMap::new(),
            },
            dictionaries: Dictionaries {
                statuses: vec!["Traced".to_string()],
                superclasses: vec![],
                classes: vec![],
                subclasses: vec![],
                soma_sides: vec![],
            },
            types,
            neurons: NeuronsTable {
                total_input: vec![0; rows.len()],
                total_output: vec![0; rows.len()],
                rows,
            },
            neuron_nt,
            edges: EdgesTable::default(),
        }
    }

    fn neuron(type_id: u32) -> NeuronRow {
        NeuronRow {
            body_id: 1,
            status: Some(0),
            type_id: Some(type_id),
            instance: None,
            superclass: None,
            class: None,
            subclass: None,
            soma_side: None,
            group: None,
            ol_hex1: None,
            ol_hex2: None,
        }
    }

    #[test]
    fn consensus_ach_maps_to_excitatory_and_confident() {
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            }],
            vec![neuron(0)],
            vec![NeuronNt::default()],
        );
        let signs = compute_type_signs(&tables, &base_cfg());
        assert_eq!(signs[0].sign, Sign::Excitatory);
        assert_eq!(signs[0].nt_class_used, NtClassUsed::Acetylcholine);
        assert_eq!(signs[0].confidence, 1.0);
        assert!(!signs[0].uncertain);
    }

    #[test]
    fn consensus_gaba_glu_his_map_to_inhibitory_by_default() {
        for nt in [NtClass::Gaba, NtClass::Glutamate, NtClass::Histamine] {
            let tables = tables_with(
                vec![TypeRow {
                    name: "T".into(),
                    consensus_nt: Some(nt),
                }],
                vec![neuron(0)],
                vec![NeuronNt::default()],
            );
            let signs = compute_type_signs(&tables, &base_cfg());
            assert_eq!(signs[0].sign, Sign::Inhibitory, "{nt:?} must default to inhibitory");
        }
    }

    #[test]
    fn consensus_modulatory_maps_to_neutral_and_not_uncertain() {
        for nt in [NtClass::Dopamine, NtClass::Serotonin, NtClass::Octopamine] {
            let tables = tables_with(
                vec![TypeRow {
                    name: "T".into(),
                    consensus_nt: Some(nt),
                }],
                vec![neuron(0)],
                vec![NeuronNt::default()],
            );
            let signs = compute_type_signs(&tables, &base_cfg());
            assert_eq!(signs[0].sign, Sign::Neutral);
            assert_eq!(signs[0].nt_class_used, NtClassUsed::Modulatory);
            assert!(
                !signs[0].uncertain,
                "a clean modulatory exclusion is not the same as an uncertain sign"
            );
        }
    }

    #[test]
    fn consensus_unclear_falls_back_to_weighted_predicted_nt_vote() {
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: Some(NtClass::Unclear),
            }],
            vec![neuron(0), neuron(0), neuron(0)],
            vec![
                NeuronNt {
                    predicted_nt: Some(NtClass::Acetylcholine),
                    predicted_nt_confidence_milli: Some(900),
                },
                NeuronNt {
                    predicted_nt: Some(NtClass::Acetylcholine),
                    predicted_nt_confidence_milli: Some(800),
                },
                NeuronNt {
                    predicted_nt: Some(NtClass::Gaba),
                    predicted_nt_confidence_milli: Some(950),
                },
            ],
        );
        let signs = compute_type_signs(&tables, &base_cfg());
        // ACh: 0.9+0.8=1.7 weighted votes vs GABA: 0.95 -> ACh wins.
        assert_eq!(signs[0].nt_class_used, NtClassUsed::Acetylcholine);
        assert_eq!(signs[0].sign, Sign::Excitatory);
        assert!(!signs[0].uncertain, "1.7/3 votes = 0.567 >= default threshold 0.5");
    }

    #[test]
    fn missing_consensus_and_no_predicted_nt_at_all_is_uncertain_neutral() {
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: None,
            }],
            vec![neuron(0)],
            vec![NeuronNt::default()],
        );
        let signs = compute_type_signs(&tables, &base_cfg());
        assert_eq!(signs[0].sign, Sign::Neutral);
        assert_eq!(signs[0].nt_class_used, NtClassUsed::Unknown);
        assert_eq!(signs[0].confidence, 0.0);
        assert!(signs[0].uncertain);
    }

    #[test]
    fn low_confidence_majority_is_uncertain_and_defaults_to_neutral() {
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: None,
            }],
            vec![neuron(0), neuron(0)],
            vec![
                NeuronNt {
                    predicted_nt: Some(NtClass::Acetylcholine),
                    predicted_nt_confidence_milli: Some(300),
                },
                NeuronNt {
                    predicted_nt: Some(NtClass::Gaba),
                    predicted_nt_confidence_milli: Some(290),
                },
            ],
        );
        let signs = compute_type_signs(&tables, &base_cfg());
        // ACh wins (0.3 > 0.29) but 0.3/0.59 = 0.508... actually let's just assert uncertain
        // status directly below rather than pre-computing by hand.
        assert!(signs[0].uncertain);
        assert_eq!(signs[0].sign, Sign::Neutral);
    }

    #[test]
    fn use_best_guess_when_uncertain_uses_the_winner_sign_instead_of_neutral() {
        let mut cfg = base_cfg();
        cfg.use_best_guess_when_uncertain = true;
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: None,
            }],
            vec![neuron(0)],
            vec![NeuronNt {
                predicted_nt: Some(NtClass::Gaba),
                predicted_nt_confidence_milli: Some(100), // low confidence -> still "uncertain"
            }],
        );
        let signs = compute_type_signs(&tables, &cfg);
        assert!(signs[0].uncertain, "still flagged uncertain");
        assert_eq!(
            signs[0].sign,
            Sign::Inhibitory,
            "but the sign uses the best guess (GABA) instead of being forced to Neutral"
        );
    }

    #[test]
    fn configured_sign_overrides_are_respected() {
        let mut cfg = base_cfg();
        cfg.glu_sign = 1; // ablation: "Glu excitatory"
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: Some(NtClass::Glutamate),
            }],
            vec![neuron(0)],
            vec![NeuronNt::default()],
        );
        let signs = compute_type_signs(&tables, &cfg);
        assert_eq!(signs[0].sign, Sign::Excitatory);
    }

    #[test]
    fn tie_breaks_deterministically_to_the_first_class_in_declared_order() {
        // ACh and GABA tie exactly at 0.5 weight each -> NT_CLASS_ORDER puts ACh first.
        let tables = tables_with(
            vec![TypeRow {
                name: "T".into(),
                consensus_nt: None,
            }],
            vec![neuron(0), neuron(0)],
            vec![
                NeuronNt {
                    predicted_nt: Some(NtClass::Acetylcholine),
                    predicted_nt_confidence_milli: Some(500),
                },
                NeuronNt {
                    predicted_nt: Some(NtClass::Gaba),
                    predicted_nt_confidence_milli: Some(500),
                },
            ],
        );
        let signs = compute_type_signs(&tables, &base_cfg());
        assert_eq!(signs[0].nt_class_used, NtClassUsed::Acetylcholine);
    }
}
