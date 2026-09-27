//! Renders the `build-subgraph` Markdown report (in Russian, per project convention — see
//! `ddai_connectome::stats`'s report for the same convention) from a built [`Flyg`] and
//! [`BuildReport`].

use std::time::Duration;

use ddai_flyg::{Flyg, NtClassUsed, Sign};

use super::build::BuildReport;
use super::config::SubgraphConfig;

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

fn nt_class_used_ru(nt: NtClassUsed) -> &'static str {
    match nt {
        NtClassUsed::Acetylcholine => "ацетилхолин",
        NtClassUsed::Glutamate => "глутамат",
        NtClassUsed::Gaba => "ГАМК",
        NtClassUsed::Histamine => "гистамин",
        NtClassUsed::Modulatory => "модуляторный (DA/5-HT/OA)",
        NtClassUsed::Unknown => "неизвестен",
    }
}

fn sign_ru(sign: Sign) -> &'static str {
    match sign {
        Sign::Excitatory => "+1 (возбуждающий)",
        Sign::Inhibitory => "−1 (тормозный)",
        Sign::Neutral => "0 (нейтральный)",
    }
}

pub fn render(
    flyg: &Flyg,
    report: &BuildReport,
    config: &SubgraphConfig,
    flyg_sha256: &str,
    wall_time: Duration,
    peak_rss_kb: Option<u64>,
) -> String {
    let mut md = String::new();
    md.push_str("# Отчёт по подграфу мухи (`build-subgraph`)\n\n");
    md.push_str(&format!(
        "sha256(`.flyg`) = `{flyg_sha256}`. Версия формата: {}. Версия генератора: `{}`.\n\n",
        flyg.header.format_version, flyg.header.generator_version
    ));
    md.push_str(&format!(
        "sha256(таблиц) = `{}`. sha256(конфига) = `{}`.\n\n",
        flyg.header.source_tables_sha256, flyg.header.config_sha256
    ));
    md.push_str(&format!(
        "Время сборки: {:.2} с. Пиковая RSS: {}.\n\n",
        wall_time.as_secs_f64(),
        peak_rss_kb
            .map(|kb| format!("{:.0} МиБ", kb as f64 / 1024.0))
            .unwrap_or_else(|| "н/д".to_string())
    ));

    // --- параметры конфига -----------------------------------------------------------------
    md.push_str("## Параметры выбора\n\n");
    md.push_str(&markdown_table(
        &["параметр", "значение"],
        &[
            vec!["k (макс. длина пути, хопов)".into(), config.selection.k.to_string()],
            vec![
                "θ_path (мин. вес ребра для поиска путей)".into(),
                config.selection.theta_path.to_string(),
            ],
            vec![
                "θ_edge (мин. вес ребра в итоговом подграфе)".into(),
                config.selection.theta_edge.to_string(),
            ],
            vec![
                "max_hidden (потолок ядра скрытых)".into(),
                config.selection.max_hidden.to_string(),
            ],
            vec![
                "weak_pair_threshold".into(),
                config.selection.weak_pair_threshold.to_string(),
            ],
            vec!["an_top_n_types".into(), config.selection.an_top_n_types.to_string()],
            vec!["rf.min_hex_support".into(), config.rf.min_hex_support.to_string()],
            vec![
                "sign_confidence_threshold".into(),
                config.nt.sign_confidence_threshold.to_string(),
            ],
            vec![
                "use_best_guess_when_uncertain".into(),
                config.nt.use_best_guess_when_uncertain.to_string(),
            ],
        ],
    ));
    md.push('\n');

    // --- проверка имён типов ------------------------------------------------------------------
    md.push_str("## Проверка имён типов из конфига\n\n");
    if report.missing_visual_types.is_empty() && report.missing_dn_types.is_empty() {
        md.push_str(
            "Все имена визуальных входных типов и типов DN-выходов из конфига найдены в MaleCNS v1.0 \
                      точно как написаны.\n\n",
        );
    } else {
        if !report.missing_visual_types.is_empty() {
            md.push_str(&format!(
                "**Не найдены типы входов (visual_types):** {}\n\n",
                report.missing_visual_types.join(", ")
            ));
        }
        if !report.missing_dn_types.is_empty() {
            md.push_str(&format!(
                "**Не найдены типы выходов (dn_types):** {}\n\n",
                report.missing_dn_types.join(", ")
            ));
        }
    }
    if !report.missing_input_channel_types.is_empty() {
        md.push_str(&format!(
            "**input_channels ссылается на типы без InputVisual-нейрона в итоговом подграфе (пропущены):** {}\n\n",
            report.missing_input_channel_types.join(", ")
        ));
    }
    if !report.missing_output_group_types.is_empty() {
        md.push_str(&format!(
            "**output_groups ссылается на типы, не найденные/не выбранные (пропущены):** {}\n\n",
            report.missing_output_group_types.join(", ")
        ));
    }

    // --- AN (ascending neurons), данные-ориентированный выбор ----------------------------------
    md.push_str("## Восходящие нейроны (AN): выбор по данным\n\n");
    md.push_str(
        "Имена типов AN в MaleCNS v1.0 почти всегда не несут функционального смысла (см. отчёт задачи 6.1) — \
         вместо курированного списка взяты топ-N типов AN по суммарному числу синапсов на выбранные DN-выходы \
         и на «проксі-центральные» нейроны (прямые постсинаптические партнёры визуальных входов с весом ≥ θ_path). \
         Это правило — **С/П** (семантическая аналогия/произвольно), не биологическая функция: см. README крейта.\n\n",
    );
    md.push_str(&markdown_table(
        &["тип AN", "синапсов на цели"],
        &report
            .an_types_picked
            .iter()
            .map(|s| vec![s.name.clone(), s.score.to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push('\n');

    // --- нейроны по ролям --------------------------------------------------------------------
    md.push_str("## Нейроны по ролям\n\n");
    let rc = &flyg.summary.neurons_by_role;
    md.push_str(&markdown_table(
        &["роль", "нейронов"],
        &[
            vec!["input_visual".into(), rc.input_visual.to_string()],
            vec!["input_ascending".into(), rc.input_ascending.to_string()],
            vec!["hidden".into(), rc.hidden.to_string()],
            vec!["output".into(), rc.output.to_string()],
            vec![
                "**всего**".into(),
                (rc.input_visual + rc.input_ascending + rc.hidden + rc.output).to_string(),
            ],
        ],
    ));
    md.push_str(&format!(
        "\nЯдро скрытых нейронов (до двусторонней дополненности): **{}**. Билатерально добавлено: {} (по `group`) \
         + {} (по типу+противоположной стороне, когда `group` не задан) = **{}**. Итоговых скрытых: **{}**.\n\n",
        report.hidden_core_count,
        report.bilateral_added_via_group,
        report.bilateral_added_via_type_fallback,
        report.bilateral_added_via_group + report.bilateral_added_via_type_fallback,
        rc.hidden
    ));
    md.push_str(&format!(
        "«Тупиковые» скрытые нейроны *в пределах подграфа* (не ошибка — вход/выход берутся целыми типами, так что \
         у скрытого нейрона могут быть только внешние партнёры по одной из сторон именно в этом подграфе): без \
         входящих рёбер в подграфе — **{}**; без исходящих рёбер в подграфе — **{}** (из {} скрытых).\n\n",
        report.hidden_dead_end_no_input, report.hidden_dead_end_no_output, rc.hidden
    ));

    // --- рёбра, типы, type_pairs ----------------------------------------------------------------
    md.push_str("## Рёбра, типы, пары типов\n\n");
    md.push_str(&markdown_table(
        &["метрика", "значение"],
        &[
            vec!["рёбер (CSR nnz)".into(), flyg.summary.num_edges.to_string()],
            vec!["типов в подграфе".into(), flyg.summary.num_types.to_string()],
            vec!["пар типов (type_pairs)".into(), flyg.summary.num_type_pairs.to_string()],
            vec![
                "общих параметров (shared_param_id, уникальных)".into(),
                flyg.summary.shared_param_count.to_string(),
            ],
        ],
    ));
    md.push('\n');

    // --- знаки -----------------------------------------------------------------------------
    md.push_str("## Знаки синапсов (по типам)\n\n");
    let sc = &flyg.summary.sign_counts;
    md.push_str(&markdown_table(
        &["знак", "типов"],
        &[
            vec![sign_ru(Sign::Excitatory).into(), sc.excitatory.to_string()],
            vec![sign_ru(Sign::Inhibitory).into(), sc.inhibitory.to_string()],
            vec![sign_ru(Sign::Neutral).into(), sc.neutral.to_string()],
        ],
    ));
    md.push_str(&format!(
        "\n`uncertain = true` (уверенность < {} или NT неизвестен/unclear): **{}** типов из {}.\n\n",
        config.nt.sign_confidence_threshold, flyg.summary.uncertain_types, flyg.summary.num_types
    ));
    if !report.uncertain_type_names.is_empty() {
        md.push_str(&format!(
            "Список неуверенных типов: {}\n\n",
            report.uncertain_type_names.join(", ")
        ));
    }
    md.push_str("### NT-класс, использованный для знака (по типам)\n\n");
    let mut nt_counts: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for t in &flyg.types {
        *nt_counts.entry(nt_class_used_ru(t.nt_class_used)).or_insert(0) += 1;
    }
    md.push_str(&markdown_table(
        &["NT-класс", "типов"],
        &nt_counts
            .iter()
            .map(|(k, v)| vec![(*k).to_string(), v.to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push('\n');

    // --- рецептивные поля ------------------------------------------------------------------
    md.push_str("## Рецептивные поля визуальных входов\n\n");
    md.push_str(&format!(
        "Всего визуальных входных нейронов: **{}**. Из них по запасному пути (гекс-опора ниже порога \
         `min_hex_support={}`, прямая + однохоповая производная): **{}**. Нейронов с гекс-координатами во всём \
         коннектоме (для калибровки карты): **{}**.\n\n",
        report.rf_stats.total_visual_inputs,
        report.rf_stats.min_hex_support,
        report.rf_stats.fallback_count,
        report.rf_stats.hex_neuron_count_full_connectome
    ));
    md.push_str(
        "Гекс-опора (`hex_support`) — сумма веса синапсов от партнёров с гекс-координатами: прямых \
         (assignedOlHex1/2 у самого партнёра) плюс однохоповых (партнёр без гекс-координат сам получает \
         производную гекс-координату как синапс-взвешенное среднее СВОИХ гекс-координированных партнёров — см. \
         README крейта, раздел «Рецептивные поля»). Ниже — по каждой паре (тип, сторона) среди визуальных входов \
         этого подграфа (не только LC10a/LC4/LPLC2 — приёмочный критерий 5 — весь список).\n\n",
    );
    md.push_str(&markdown_table(
        &[
            "тип",
            "сторона",
            "нейронов",
            "fallback",
            "опора: мин",
            "опора: медиана",
            "опора: макс",
            "< порога",
            "азимут: мин°",
            "азимут: медиана°",
            "азимут: макс°",
        ],
        &report
            .rf_stats
            .per_type
            .iter()
            .map(|s| {
                vec![
                    s.type_name.clone(),
                    s.side_label.clone(),
                    s.count.to_string(),
                    s.fallback_count.to_string(),
                    format!("{:.0}", s.hex_support_min),
                    format!("{:.0}", s.hex_support_median),
                    format!("{:.0}", s.hex_support_max),
                    s.below_threshold_count.to_string(),
                    format!("{:.1}", s.azimuth_min_deg),
                    format!("{:.1}", s.azimuth_median_deg),
                    format!("{:.1}", s.azimuth_max_deg),
                ]
            })
            .collect::<Vec<_>>(),
    ));
    md.push('\n');

    // --- выходные группы ---------------------------------------------------------------------
    md.push_str("## Группы выходов (действия)\n\n");
    md.push_str(&markdown_table(
        &["действие", "нейронов"],
        &flyg
            .output_groups
            .iter()
            .map(|g| vec![g.action.clone(), g.members.len().to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push('\n');

    // --- целевые размеры ---------------------------------------------------------------------
    let total_neurons = rc.input_visual + rc.input_ascending + rc.hidden + rc.output;
    md.push_str("## Итог по размеру\n\n");
    md.push_str(&format!(
        "**{total_neurons}** нейронов, **{}** рёбер.\n\n",
        flyg.summary.num_edges
    ));
    md
}
