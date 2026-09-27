//! `stats --tables <dir> --out <report.md>`: reads the compact tables written by `build-tables`
//! and writes a statistics report **in Russian** (per project convention, docs are Russian, code
//! is English — see `CLAUDE.md`).
//!
//! A note on which neurotransmitter field is used where, since the source data has two (see
//! `docs/research/fly-data.md` §1.4): this report's "missing/unclear NT" and "NT distribution"
//! sections are based on **`predicted_nt`** (the per-neuron field `build-tables` actually stores
//! — see `tables::NeuronNt`), not `consensus_nt` (which `build-tables` only stores per-*type*,
//! as a majority vote — see `tables::TypeRow`). A separate "Сверка" section reports the
//! type-level `consensus_nt` breakdown for DN/VPN/AN so a comparison to the phase-0 research
//! notes (which tally `consensus_nt` directly per body, not grouped by type) is still possible.
//! Three specific, verified reasons the two can disagree by a handful of neurons (see that
//! section's own text in the generated report for the exact numbers): an untyped neuron has no
//! type to inherit a consensus from; a typed neuron with no NT-file row of its own still
//! inherits its type's majority vote here, unlike a per-body tally; and the phase-0 note's own
//! prose can simply omit a rare class from a written sum (confirmed for AN/histamine).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::tables::{ConnectomeTables, NeuronRow, NtClass, load_tables};

#[derive(Debug, Clone, Default)]
pub struct GroupCounts {
    pub neurons: u64,
    pub types: u64,
}

#[derive(Debug, Clone, Default)]
pub struct StatsSummary {
    pub total_neurons: u64,
    pub traced_neurons: u64,
    pub dn: GroupCounts,
    pub vpn: GroupCounts,
    pub an: GroupCounts,
    pub edges_total: u64,
    pub autapses_dropped: u64,
    pub edges_weight_ge3: u64,
    pub edges_weight_ge5: u64,
    pub edges_weight_ge10: u64,
}

const DN_SUPERCLASS: &str = "descending_neuron";
const VPN_SUPERCLASS: &str = "visual_projection";
const AN_SUPERCLASS: &str = "ascending_neuron";
const TRACED_STATUS: &str = "Traced";

pub fn run_stats(tables_dir: &Path, out_path: &Path) -> Result<StatsSummary> {
    let tables = load_tables(tables_dir)?;
    let (markdown, summary) = render_report(&tables);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(out_path, markdown).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(summary)
}

fn dict_id(dict: &[String], name: &str) -> Option<u16> {
    dict.iter().position(|s| s == name).map(|i| i as u16)
}

fn name_or_none(dict: &[String], id: Option<u16>) -> String {
    id.map(|i| dict[i as usize].clone())
        .unwrap_or_else(|| "(нет)".to_string())
}

/// Counts `rows` by the string a `dict`/id-getter pair resolves to, sorted by count descending
/// then name ascending (deterministic, and more readable than plain alphabetical).
fn count_by<'a, F>(rows: &'a [NeuronRow], dict: &[String], get_id: F) -> Vec<(String, u64)>
where
    F: Fn(&'a NeuronRow) -> Option<u16>,
{
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for row in rows {
        *counts.entry(name_or_none(dict, get_id(row))).or_insert(0) += 1;
    }
    let mut v: Vec<(String, u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

fn markdown_table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str("| ");
    out.push_str(&header.join(" | "));
    out.push_str(" |\n|");
    out.push_str(&"---|".repeat(header.len()));
    out.push('\n');
    for row in rows {
        out.push_str("| ");
        out.push_str(&row.join(" | "));
        out.push_str(" |\n");
    }
    out
}

fn nt_class_name_ru(nt: NtClass) -> &'static str {
    match nt {
        NtClass::Acetylcholine => "ацетилхолин",
        NtClass::Glutamate => "глутамат",
        NtClass::Gaba => "ГАМК",
        NtClass::Histamine => "гистамин",
        NtClass::Dopamine => "дофамин",
        NtClass::Serotonin => "серотонин",
        NtClass::Octopamine => "октопамин",
        NtClass::Unclear => "unclear",
    }
}

const ALL_NT_CLASSES: [NtClass; 8] = [
    NtClass::Acetylcholine,
    NtClass::Glutamate,
    NtClass::Gaba,
    NtClass::Histamine,
    NtClass::Dopamine,
    NtClass::Serotonin,
    NtClass::Octopamine,
    NtClass::Unclear,
];

/// A (superclass name, [neuron indices]) group, restricted to Traced, sorted by superclass name.
fn traced_indices_by_superclass(tables: &ConnectomeTables, traced_status: Option<u16>) -> Vec<(String, Vec<usize>)> {
    let mut by_superclass: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, row) in tables.neurons.rows.iter().enumerate() {
        if row.status != traced_status {
            continue;
        }
        let name = name_or_none(&tables.dictionaries.superclasses, row.superclass);
        by_superclass.entry(name).or_default().push(idx);
    }
    by_superclass.into_iter().collect()
}

fn group_counts(tables: &ConnectomeTables, traced_status: Option<u16>, superclass_name: &str) -> GroupCounts {
    let superclass_id = dict_id(&tables.dictionaries.superclasses, superclass_name);
    let mut types = std::collections::BTreeSet::new();
    let mut neurons = 0u64;
    for row in &tables.neurons.rows {
        if row.status == traced_status && row.superclass == superclass_id && superclass_id.is_some() {
            neurons += 1;
            if let Some(t) = row.type_id {
                types.insert(t);
            }
        }
    }
    GroupCounts {
        neurons,
        types: types.len() as u64,
    }
}

pub fn render_report(tables: &ConnectomeTables) -> (String, StatsSummary) {
    let traced_status = dict_id(&tables.dictionaries.statuses, TRACED_STATUS);

    let dn = group_counts(tables, traced_status, DN_SUPERCLASS);
    let vpn = group_counts(tables, traced_status, VPN_SUPERCLASS);
    let an = group_counts(tables, traced_status, AN_SUPERCLASS);

    let traced_neurons = tables
        .neurons
        .rows
        .iter()
        .filter(|r| r.status == traced_status && traced_status.is_some())
        .count() as u64;

    let mut edges_weight_ge3 = 0u64;
    let mut edges_weight_ge5 = 0u64;
    let mut edges_weight_ge10 = 0u64;
    let mut histogram: BTreeMap<&str, u64> = BTreeMap::new();
    let mut weight_sum: u128 = 0;
    let mut max_weight = 0u32;
    for e in &tables.edges.edges {
        if e.weight >= 3 {
            edges_weight_ge3 += 1;
        }
        if e.weight >= 5 {
            edges_weight_ge5 += 1;
        }
        if e.weight >= 10 {
            edges_weight_ge10 += 1;
        }
        weight_sum += e.weight as u128;
        max_weight = max_weight.max(e.weight);
        let bucket = match e.weight {
            0 => "0", // should not occur (no zero-weight edges expected), kept for completeness
            1 => "1",
            2 => "2",
            3..=4 => "3-4",
            5..=9 => "5-9",
            10..=19 => "10-19",
            20..=49 => "20-49",
            50..=99 => "50-99",
            _ => "100+",
        };
        *histogram.entry(bucket).or_insert(0) += 1;
    }
    let edges_total = tables.edges.edges.len() as u64;
    let mean_weight = if edges_total > 0 {
        weight_sum as f64 / edges_total as f64
    } else {
        0.0
    };

    let summary = StatsSummary {
        total_neurons: tables.neurons.rows.len() as u64,
        traced_neurons,
        dn: dn.clone(),
        vpn: vpn.clone(),
        an: an.clone(),
        edges_total,
        autapses_dropped: tables.edges.autapses_dropped,
        edges_weight_ge3,
        edges_weight_ge5,
        edges_weight_ge10,
    };

    // --- markdown -------------------------------------------------------------------------------
    let mut md = String::new();
    md.push_str("# Отчёт по коннектому MaleCNS v1.0\n\n");
    md.push_str("Сгенерировано `ddai-connectome stats` из компактных таблиц (`build-tables`).\n\n");

    md.push_str("## Тела по статусу (все тела из body-annotations)\n\n");
    let status_counts = count_by(&tables.neurons.rows, &tables.dictionaries.statuses, |r| r.status);
    md.push_str(&markdown_table(
        &["status", "тел"],
        &status_counts
            .iter()
            .map(|(k, v)| vec![k.clone(), v.to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push_str(&format!(
        "\nВсего тел: **{}**, из них Traced: **{traced_neurons}**.\n\n",
        tables.neurons.rows.len()
    ));

    md.push_str("## Superclass среди Traced\n\n");
    let traced_rows: Vec<NeuronRow> = tables
        .neurons
        .rows
        .iter()
        .filter(|r| r.status == traced_status && traced_status.is_some())
        .cloned()
        .collect();
    let superclass_counts = count_by(&traced_rows, &tables.dictionaries.superclasses, |r| r.superclass);
    md.push_str(&markdown_table(
        &["superclass", "нейронов (Traced)"],
        &superclass_counts
            .iter()
            .map(|(k, v)| vec![k.clone(), v.to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push('\n');

    md.push_str("## DN / VPN / AN (по superclass, не по префиксу имени типа)\n\n");
    md.push_str(&markdown_table(
        &["группа", "superclass", "нейронов", "типов"],
        &[
            vec![
                "DN".into(),
                DN_SUPERCLASS.into(),
                dn.neurons.to_string(),
                dn.types.to_string(),
            ],
            vec![
                "VPN".into(),
                VPN_SUPERCLASS.into(),
                vpn.neurons.to_string(),
                vpn.types.to_string(),
            ],
            vec![
                "AN".into(),
                AN_SUPERCLASS.into(),
                an.neurons.to_string(),
                an.types.to_string(),
            ],
        ],
    ));
    md.push('\n');

    md.push_str("## Рёбра\n\n");
    md.push_str(&format!(
        "Всего рёбер: **{edges_total}**. Автапсов отброшено: **{}**. Средний вес: {mean_weight:.3}. Максимальный вес: {max_weight}.\n\n",
        tables.edges.autapses_dropped
    ));
    md.push_str(&markdown_table(
        &["порог", "рёбер"],
        &[
            vec!["weight ≥ 3".into(), edges_weight_ge3.to_string()],
            vec!["weight ≥ 5".into(), edges_weight_ge5.to_string()],
            vec!["weight ≥ 10".into(), edges_weight_ge10.to_string()],
        ],
    ));
    md.push_str("\n### Гистограмма весов\n\n");
    let bucket_order = ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"];
    let histogram_rows: Vec<Vec<String>> = bucket_order
        .iter()
        .filter_map(|b| histogram.get(b).map(|c| vec![(*b).to_string(), c.to_string()]))
        .collect();
    md.push_str(&markdown_table(&["weight", "рёбер"], &histogram_rows));
    md.push('\n');

    md.push_str("## Нейромедиатор (predicted_nt, на нейрон)\n\n");
    md.push_str(
        "Ниже — распределение поля `predicted_nt` из таблицы `neuron_nt` (не `consensus_nt` типа — см. пояснение в начале файла `stats.rs`).\n\n",
    );
    let (missing, unclear, predicted_counts) = predicted_nt_breakdown(tables, &traced_rows);
    md.push_str(&format!(
        "Среди Traced: нет строки/значения — **{missing}**, `unclear` — **{unclear}**.\n\n",
    ));
    md.push_str("### Распределение predicted_nt среди Traced (в целом)\n\n");
    md.push_str(&markdown_table(
        &["класс", "нейронов"],
        &predicted_counts
            .iter()
            .map(|(k, v)| vec![k.clone(), v.to_string()])
            .collect::<Vec<_>>(),
    ));

    md.push_str("\n### Распределение predicted_nt по superclass (среди Traced)\n\n");
    let groups = traced_indices_by_superclass(tables, traced_status);
    let mut per_superclass_rows = Vec::new();
    for (superclass, indices) in &groups {
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for &idx in indices {
            let label = match tables.neuron_nt[idx].predicted_nt {
                Some(nt) => nt_class_name_ru(nt),
                None => "(нет)",
            };
            *counts.entry(label).or_insert(0) += 1;
        }
        let mut row = vec![superclass.clone(), indices.len().to_string()];
        for nt in ALL_NT_CLASSES {
            row.push(counts.get(nt_class_name_ru(nt)).copied().unwrap_or(0).to_string());
        }
        row.push(counts.get("(нет)").copied().unwrap_or(0).to_string());
        per_superclass_rows.push(row);
    }
    let mut header = vec!["superclass", "нейронов"];
    let nt_headers: Vec<&str> = ALL_NT_CLASSES.iter().map(|nt| nt_class_name_ru(*nt)).collect();
    header.extend(nt_headers.iter());
    header.push("(нет)");
    md.push_str(&markdown_table(&header, &per_superclass_rows));

    md.push_str("\n## Сверка: consensus_nt типа для DN/VPN/AN\n\n");
    md.push_str(
        "Для сравнения с фазой 0 (`docs/research/fly-data.md` §5) — то же самое, но через `types[type_id].consensus_nt` \
         (агрегат по типу, majority vote), а не через predicted_nt на нейрон. От чисел фазы 0 (взятых напрямую из колонки \
         `consensus_nt` на каждое тело, без группировки по типу) это может отличаться по трём конкретным, проверенным \
         причинам, не являющимся ошибкой: (1) у нейрона без `type` нет типа, через который агрегировать consensus_nt — \
         такие нейроны показаны отдельно как «нет типа», а не смешаны с «unclear»/отсутствием NT; (2) у отдельных \
         типизированных нейронов нет собственной строки в файле нейромедиаторов, но majority-голос их типа всё равно \
         непустой — такой нейрон получает голос своего типа здесь, а не «нет NT», как при подсчёте по значению самого \
         нейрона (у VPN это ровно 1 нейрон типа с консенсусом ACh: 7890 здесь против 7889 у фазы 0); (3) текст фазы 0 \
         для AN не упомянул гистамин явно, хотя данные его содержат: 1271 (ACh) + 407 (GABA) + 127 (Glu) + 37 (unclear) \
         = 1842, а не 1846 — недостающие 4 нейрона гистаминовые (показаны здесь как «гистамин»), это неполнота того \
         текста, а не ошибка подсчёта.\n\n",
    );
    for (label, superclass_name) in [("DN", DN_SUPERCLASS), ("VPN", VPN_SUPERCLASS), ("AN", AN_SUPERCLASS)] {
        let superclass_id = dict_id(&tables.dictionaries.superclasses, superclass_name);
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        let mut untyped = 0u64;
        let mut typed_without_consensus = 0u64;
        for row in &traced_rows {
            if row.superclass != superclass_id {
                continue;
            }
            match row.type_id {
                None => untyped += 1,
                Some(t) => match tables.types[t as usize].consensus_nt {
                    Some(nt) => *counts.entry(nt_class_name_ru(nt)).or_insert(0) += 1,
                    None => typed_without_consensus += 1,
                },
            }
        }
        md.push_str(&format!("**{label}**: "));
        let mut parts: Vec<String> = counts.iter().map(|(k, v)| format!("{k} {v}")).collect();
        if untyped > 0 {
            parts.push(format!("нет типа {untyped}"));
        }
        if typed_without_consensus > 0 {
            parts.push(format!("тип без NT {typed_without_consensus}"));
        }
        md.push_str(&parts.join(", "));
        md.push_str("\n\n");
    }

    (md, summary)
}

/// Returns `(missing, unclear, distribution)` for `predicted_nt` among `traced_rows`.
fn predicted_nt_breakdown(tables: &ConnectomeTables, traced_rows: &[NeuronRow]) -> (u64, u64, Vec<(String, u64)>) {
    let body_to_idx: BTreeMap<i64, usize> = tables
        .neurons
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.body_id, i))
        .collect();
    let mut missing = 0u64;
    let mut unclear = 0u64;
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for row in traced_rows {
        let idx = body_to_idx[&row.body_id];
        match tables.neuron_nt[idx].predicted_nt {
            None => missing += 1,
            Some(NtClass::Unclear) => {
                unclear += 1;
                *counts
                    .entry(nt_class_name_ru(NtClass::Unclear).to_string())
                    .or_insert(0) += 1;
            }
            Some(nt) => {
                *counts.entry(nt_class_name_ru(nt).to_string()).or_insert(0) += 1;
            }
        }
    }
    if missing > 0 {
        counts.insert("(нет)".to_string(), missing);
    }
    let mut v: Vec<(String, u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    (missing, unclear, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{Dictionaries, Edge, EdgesTable, NeuronNt, NeuronsTable, TablesHeader, TypeRow};
    use std::collections::BTreeMap as StdBTreeMap;

    fn tiny_tables() -> ConnectomeTables {
        // 3 Traced neurons: one DN (type X, ACh), one VPN (type Y, unclear), one with no NT row.
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![DN_SUPERCLASS.to_string(), VPN_SUPERCLASS.to_string()],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec![],
        };
        let types = vec![
            TypeRow {
                name: "X".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
            TypeRow {
                name: "Y".into(),
                consensus_nt: Some(NtClass::Unclear),
            },
        ];
        let rows = vec![
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
                ol_hex1: None,
                ol_hex2: None,
            },
            NeuronRow {
                body_id: 2,
                status: Some(0),
                type_id: Some(1),
                instance: None,
                superclass: Some(1),
                class: None,
                subclass: None,
                soma_side: None,
                group: None,
                ol_hex1: None,
                ol_hex2: None,
            },
            NeuronRow {
                body_id: 3,
                status: Some(0),
                type_id: None,
                instance: None,
                superclass: Some(0),
                class: None,
                subclass: None,
                soma_side: None,
                group: None,
                ol_hex1: None,
                ol_hex2: None,
            },
        ];
        let neuron_nt = vec![
            NeuronNt {
                predicted_nt: Some(NtClass::Acetylcholine),
                predicted_nt_confidence_milli: Some(950),
            },
            NeuronNt {
                predicted_nt: Some(NtClass::Unclear),
                predicted_nt_confidence_milli: Some(400),
            },
            NeuronNt::default(),
        ];
        let edges = vec![
            Edge {
                pre_idx: 0,
                post_idx: 1,
                weight: 5,
            },
            Edge {
                pre_idx: 0,
                post_idx: 2,
                weight: 12,
            },
        ];
        ConnectomeTables {
            header: TablesHeader {
                format_version: 1,
                input_sha256: StdBTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input: vec![0, 5, 12],
                total_output: vec![17, 0, 0],
            },
            neuron_nt,
            edges: EdgesTable {
                edges,
                autapses_dropped: 1,
            },
        }
    }

    #[test]
    fn summary_counts_dn_and_vpn_correctly() {
        let tables = tiny_tables();
        let (_md, summary) = render_report(&tables);
        assert_eq!(summary.total_neurons, 3);
        assert_eq!(summary.traced_neurons, 3);
        assert_eq!(summary.dn.neurons, 2); // bodies 1 and 3 share the DN superclass
        assert_eq!(summary.dn.types, 1); // only body 1 has a type
        assert_eq!(summary.vpn.neurons, 1);
        assert_eq!(summary.vpn.types, 1);
        assert_eq!(summary.edges_total, 2);
        assert_eq!(summary.autapses_dropped, 1);
        assert_eq!(summary.edges_weight_ge3, 2);
        assert_eq!(summary.edges_weight_ge5, 2);
        assert_eq!(summary.edges_weight_ge10, 1);
    }

    #[test]
    fn report_is_valid_utf8_markdown_and_mentions_key_sections() {
        let tables = tiny_tables();
        let (md, _summary) = render_report(&tables);
        for heading in [
            "# Отчёт",
            "## Тела по статусу",
            "## DN / VPN / AN",
            "## Рёбра",
            "## Нейромедиатор",
        ] {
            assert!(md.contains(heading), "report is missing section {heading:?}:\n{md}");
        }
    }

    #[test]
    fn missing_nt_is_counted_separately_from_unclear() {
        let tables = tiny_tables();
        let traced_rows: Vec<NeuronRow> = tables.neurons.rows.clone();
        let (missing, unclear, _dist) = predicted_nt_breakdown(&tables, &traced_rows);
        assert_eq!(missing, 1, "body 3 has no NT row at all");
        assert_eq!(unclear, 1, "body 2 is explicitly unclear");
    }

    #[test]
    fn consensus_section_splits_untyped_from_typed_without_consensus() {
        // 3 Traced DN neurons: one with a type that has a consensus, one with a type that has
        // none (all its own neurons disagreed evenly, say), and one with no type at all. The
        // "Сверка" section must tell these apart rather than lumping both into a single "(нет)".
        let dictionaries = Dictionaries {
            statuses: vec!["Traced".to_string()],
            superclasses: vec![DN_SUPERCLASS.to_string()],
            classes: vec![],
            subclasses: vec![],
            soma_sides: vec![],
        };
        let types = vec![
            TypeRow {
                name: "A".into(),
                consensus_nt: Some(NtClass::Acetylcholine),
            },
            TypeRow {
                name: "B".into(),
                consensus_nt: None,
            },
        ];
        let rows = vec![
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
                ol_hex1: None,
                ol_hex2: None,
            },
            NeuronRow {
                body_id: 2,
                status: Some(0),
                type_id: Some(1),
                instance: None,
                superclass: Some(0),
                class: None,
                subclass: None,
                soma_side: None,
                group: None,
                ol_hex1: None,
                ol_hex2: None,
            },
            NeuronRow {
                body_id: 3,
                status: Some(0),
                type_id: None,
                instance: None,
                superclass: Some(0),
                class: None,
                subclass: None,
                soma_side: None,
                group: None,
                ol_hex1: None,
                ol_hex2: None,
            },
        ];
        let tables = ConnectomeTables {
            header: TablesHeader {
                format_version: crate::tables::TABLES_FORMAT_VERSION,
                input_sha256: StdBTreeMap::new(),
            },
            dictionaries,
            types,
            neurons: NeuronsTable {
                rows,
                total_input: vec![0, 0, 0],
                total_output: vec![0, 0, 0],
            },
            neuron_nt: vec![NeuronNt::default(); 3],
            edges: EdgesTable::default(),
        };

        let (md, _summary) = render_report(&tables);
        let dn_line = md
            .lines()
            .find(|l| l.starts_with("**DN**:"))
            .expect("DN line in the Сверка section");
        assert!(
            dn_line.contains("ацетилхолин 1"),
            "body 1's type A has an ACh consensus: {dn_line:?}"
        );
        assert!(
            dn_line.contains("тип без NT 1"),
            "body 2's type B has no consensus: {dn_line:?}"
        );
        assert!(dn_line.contains("нет типа 1"), "body 3 has no type at all: {dn_line:?}");
    }
}
