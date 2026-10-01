//! Per-condition summaries, the run record, the machine stall baseline (D-045) and the Russian
//! markdown table.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use serde::Serialize;

use crate::run::ConditionRun;
use crate::stats::{GameResult, Tally, percentile_sorted};

/// A proportion with its 95% Wilson interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Rate {
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Rate {
    fn from(t: Option<(f64, f64, f64)>) -> Option<Rate> {
        t.map(|(p, lo, hi)| Rate { p, lo, hi })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlayerSummary {
    pub slot: usize,
    pub label: String,
    pub lag: u32,
    pub decisions: u64,
    /// Pooled over all games, wall microseconds.
    pub decide_us_p50: Option<u32>,
    pub decide_us_p99: Option<u32>,
    pub decide_us_max: Option<u32>,
    /// Sums of the numeric fields of the brain's telemetry over all games.
    pub telemetry_sum: BTreeMap<String, f64>,
}

/// Adds every numeric leaf of a telemetry object into `sums` under its dotted path
/// (`totals.extended`). Nested objects are followed (the hybrid brain reports `totals.*`); arrays
/// and the `last` snapshot (a single decision, meaningless summed) are skipped.
fn sum_numeric_leaves(prefix: &str, v: &serde_json::Value, sums: &mut BTreeMap<String, f64>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, child) in m {
                if prefix.is_empty() && k == "last" {
                    continue;
                }
                let path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                sum_numeric_leaves(&path, child, sums);
            }
        }
        other => {
            if let Some(x) = other.as_f64() {
                *sums.entry(prefix.to_string()).or_insert(0.0) += x;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Distribution {
    pub n: u32,
    pub mean: Option<f64>,
    pub median: Option<f64>,
}

impl Distribution {
    fn of(mut v: Vec<f64>) -> Distribution {
        v.sort_by(f64::total_cmp);
        Distribution {
            n: v.len() as u32,
            mean: (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64),
            median: percentile_sorted(&v, 50.0),
        }
    }
}

/// The outcome of the games of one layout half (see `game::Layout`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SplitStats {
    pub games: u32,
    pub tally: Tally,
    /// `W / (W + L + D)` within the half.
    pub win_rate: Option<Rate>,
}

impl SplitStats {
    fn of<'a>(games: impl Iterator<Item = &'a crate::game::GameReport>) -> SplitStats {
        let mut tally = Tally::default();
        for g in games {
            tally.record(g.result);
        }
        SplitStats {
            games: tally.total(),
            tally,
            win_rate: Rate::from(tally.win_rate()),
        }
    }
}

/// Per-layout splits: a large gap between the two halves of either pair is a bias of the
/// *arena*, not of the players (the headline numbers pool all four cells evenly).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayoutSplits {
    /// Spawn order normal (slot 0 spawned first, as before) / reversed (slot 0 spawned last).
    pub order_normal: SplitStats,
    pub order_reversed: SplitStats,
    /// Positions as drawn / focal player and first opponent traded.
    pub position_plain: SplitStats,
    pub position_swapped: SplitStats,
}

/// Everything reported for one condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionSummary {
    pub name: String,
    pub arena: String,
    pub arena_tag: String,
    pub map_sha256: Option<String>,
    pub games: u32,
    pub tally: Tally,
    /// `W / (W + L + D)`.
    pub win_rate: Option<Rate>,
    /// `W / (W + L + D + T)`.
    pub win_rate_all: Option<Rate>,
    /// D-059 headline: `credited_w / games` (games won by the focal player's own credited block,
    /// timeouts counted against it) with its Wilson interval.
    pub credited_win_rate: Option<Rate>,
    pub credited_w: u32,
    pub held_w: u32,
    pub credited_l: u32,
    pub held_l: u32,
    /// Opponent onsets credited to the focal player.
    pub blocks_by_a: u32,
    /// Game time played, minutes (sum of end ticks / 3000).
    pub minutes_played: f64,
    pub blocks_per_min: f64,
    pub a_self_freezes: u32,
    pub self_freezes_per_min: f64,
    /// Wayblock arenas: the share of the focal player's game time spent inside the hall's band
    /// (`wbBand`); `None` on other arenas.
    #[serde(default)]
    pub band_fraction: Option<f64>,
    /// Wayblock arenas: the share of the focal player's game time spent inside the held hall.
    #[serde(default)]
    pub hall_fraction: Option<f64>,
    /// Focal-player survival in seconds (censored at the deciding tick of games it did not lose).
    pub a_survival_s: Distribution,
    /// Games where the focal player never went out before the game was decided.
    pub a_survival_censored: u32,
    /// Seconds from the start to the first credited block, over games with one.
    pub time_to_first_block_s: Distribution,
    pub bystander_outs: u32,
    /// Length of decided games (W/L/D), seconds.
    pub decided_game_s: Distribution,
    /// Win rate per spawn order and per position half.
    pub splits: LayoutSplits,
    pub players: Vec<PlayerSummary>,
    /// Wall-clock cost of the batch.
    pub wall_s: f64,
    pub games_per_s: f64,
}

/// Aggregates one condition.
pub fn summarize(run: &ConditionRun, arena_tag: &str, map_sha256: Option<String>) -> ConditionSummary {
    let games = &run.games;
    let mut tally = Tally::default();
    let (mut credited_w, mut held_w, mut credited_l, mut held_l) = (0, 0, 0, 0);
    let mut blocks = 0u32;
    let mut freezes = 0u32;
    let mut bystanders = 0u32;
    let mut ticks = 0i64;
    let mut survival = Vec::new();
    let mut censored = 0u32;
    let mut first_block = Vec::new();
    let mut decided_len = Vec::new();
    for g in games {
        tally.record(g.result);
        match g.result {
            GameResult::W => {
                credited_w += u32::from(g.credited);
                held_w += u32::from(g.held);
            }
            GameResult::L => {
                credited_l += u32::from(g.credited);
                held_l += u32::from(g.held);
            }
            _ => {}
        }
        if g.result != GameResult::T {
            decided_len.push(f64::from(g.end_tick) / 50.0);
        }
        blocks += g.blocks_by_a;
        freezes += g.a_self_freezes;
        bystanders += g.bystander_outs;
        ticks += i64::from(g.end_tick.max(0));
        survival.push(f64::from(g.a_out_tick.unwrap_or(g.end_tick)) / 50.0);
        censored += u32::from(g.a_out_tick.is_none());
        if let Some(t) = g.first_block_tick {
            first_block.push(f64::from(t) / 50.0);
        }
    }
    let minutes = ticks as f64 / 3000.0;
    let per_min = |n: u32| if minutes > 0.0 { f64::from(n) / minutes } else { 0.0 };
    let n_slots = games.first().map_or(0, |g| g.players.len());
    let mut players = Vec::with_capacity(n_slots);
    for slot in 0..n_slots {
        let mut pooled: Vec<u32> = games.iter().flat_map(|g| g.decide_us[slot].iter().copied()).collect();
        pooled.sort_unstable();
        let mut sums: BTreeMap<String, f64> = BTreeMap::new();
        for g in games {
            if let Some(t @ serde_json::Value::Object(_)) = &g.players[slot].telemetry {
                sum_numeric_leaves("", t, &mut sums);
            }
        }
        let first = games.first().map(|g| &g.players[slot]);
        players.push(PlayerSummary {
            slot,
            label: first.map(|p| p.label.clone()).unwrap_or_default(),
            lag: first.map_or(0, |p| p.lag),
            decisions: games.iter().map(|g| u64::from(g.players[slot].decisions)).sum(),
            decide_us_p50: percentile_sorted(&pooled, 50.0),
            decide_us_p99: percentile_sorted(&pooled, 99.0),
            decide_us_max: pooled.last().copied(),
            telemetry_sum: sums,
        });
    }
    let n = games.len() as u32;
    ConditionSummary {
        name: run.condition.name.clone(),
        arena: run.arena.clone(),
        arena_tag: arena_tag.to_string(),
        map_sha256,
        games: n,
        tally,
        win_rate: Rate::from(tally.win_rate()),
        win_rate_all: Rate::from(tally.win_rate_all()),
        credited_win_rate: Rate::from(tally.credited_win_rate(credited_w)),
        credited_w,
        held_w,
        credited_l,
        held_l,
        blocks_by_a: blocks,
        minutes_played: minutes,
        blocks_per_min: per_min(blocks),
        a_self_freezes: freezes,
        self_freezes_per_min: per_min(freezes),
        band_fraction: {
            let total: u64 = games.iter().map(|g| u64::from(g.a_ticks)).sum();
            (total > 0).then(|| games.iter().map(|g| f64::from(g.a_band_ticks)).sum::<f64>() / total as f64)
        },
        hall_fraction: {
            let total: u64 = games.iter().map(|g| u64::from(g.a_ticks)).sum();
            (total > 0).then(|| games.iter().map(|g| f64::from(g.a_hall_ticks)).sum::<f64>() / total as f64)
        },
        a_survival_s: Distribution::of(survival),
        a_survival_censored: censored,
        time_to_first_block_s: Distribution::of(first_block),
        bystander_outs: bystanders,
        decided_game_s: Distribution::of(decided_len),
        splits: LayoutSplits {
            order_normal: SplitStats::of(games.iter().filter(|g| !g.reverse_order)),
            order_reversed: SplitStats::of(games.iter().filter(|g| g.reverse_order)),
            position_plain: SplitStats::of(games.iter().filter(|g| !g.swap)),
            position_swapped: SplitStats::of(games.iter().filter(|g| g.swap)),
        },
        players,
        wall_s: run.wall_s,
        games_per_s: if run.wall_s > 0.0 {
            f64::from(n) / run.wall_s
        } else {
            0.0
        },
    }
}

impl ConditionSummary {
    /// The summary without anything wall-clock dependent (decision times, batch duration): what
    /// must be identical between two runs of the same config.
    pub fn deterministic_view(&self) -> ConditionSummary {
        let mut s = self.clone();
        s.wall_s = 0.0;
        s.games_per_s = 0.0;
        for p in &mut s.players {
            p.decide_us_p50 = None;
            p.decide_us_p99 = None;
            p.decide_us_max = None;
        }
        s
    }
}

/// How much a busy loop with no work is interrupted on this machine (D-045): the VM pauses
/// threads for ~10 ms around ten times a second, which shows up as decision-time tails that are
/// not search work. Published next to every wall-clock decision time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StallBaseline {
    pub duration_ms: u64,
    pub samples: u64,
    /// Fraction of consecutive-read gaps above 0.5 ms / 2 ms.
    pub frac_over_0_5ms: f64,
    pub frac_over_2ms: f64,
    /// Gaps above 2 ms per second of wall time.
    pub stalls_over_2ms_per_s: f64,
    pub max_ms: f64,
}

/// Spins on `Instant::now()` for `duration_ms` and reports the gap statistics.
pub fn measure_stall_baseline(duration_ms: u64) -> StallBaseline {
    let start = Instant::now();
    let mut last = start;
    let (mut n, mut over_half, mut over_two, mut max) = (0u64, 0u64, 0u64, 0.0f64);
    while start.elapsed().as_millis() < u128::from(duration_ms) {
        let now = Instant::now();
        let gap = now.duration_since(last).as_secs_f64() * 1000.0;
        last = now;
        n += 1;
        over_half += u64::from(gap > 0.5);
        over_two += u64::from(gap > 2.0);
        max = max.max(gap);
    }
    let secs = (duration_ms as f64 / 1000.0).max(1e-9);
    StallBaseline {
        duration_ms,
        samples: n,
        frac_over_0_5ms: over_half as f64 / n.max(1) as f64,
        frac_over_2ms: over_two as f64 / n.max(1) as f64,
        stalls_over_2ms_per_s: over_two as f64 / secs,
        max_ms: max,
    }
}

/// One real map used by a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MapRecord {
    pub arena: String,
    pub source: String,
    pub sha256: String,
}

/// What produced a summary: enough to re-run it and to detect a stale result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunMeta {
    pub run_name: String,
    /// SHA-256 of the effective config (see `RunConfig::hash`).
    pub config_hash: String,
    pub git_commit: String,
    pub git_dirty: bool,
    pub base_seed: u64,
    pub threads: usize,
    pub maps: Vec<MapRecord>,
    pub stall_baseline: Option<StallBaseline>,
    pub total_wall_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunSummary {
    pub meta: RunMeta,
    pub conditions: Vec<ConditionSummary>,
}

fn pct(r: &Option<Rate>) -> String {
    match r {
        Some(r) => format!("{:.1}% [{:.1}; {:.1}]", 100.0 * r.p, 100.0 * r.lo, 100.0 * r.hi),
        None => "—".to_string(),
    }
}

fn ms(us: Option<u32>) -> String {
    us.map_or("—".to_string(), |u| format!("{:.2}", f64::from(u) / 1000.0))
}

fn opt_s(x: Option<f64>) -> String {
    x.map_or("—".to_string(), |v| format!("{v:.1}"))
}

/// The Russian markdown report: run record, then one row per condition.
pub fn markdown(s: &RunSummary) -> String {
    let m = &s.meta;
    let mut out = String::new();
    let _ = writeln!(out, "# Арена: {}\n", m.run_name);
    let _ = writeln!(out, "- конфиг (sha256): `{}`", m.config_hash);
    let _ = writeln!(
        out,
        "- коммит: `{}`{}",
        m.git_commit,
        if m.git_dirty {
            " (+ незакоммиченные правки)"
        } else {
            ""
        }
    );
    let _ = writeln!(out, "- базовый сид: {}, потоков: {}", m.base_seed, m.threads);
    for map in &m.maps {
        let _ = writeln!(
            out,
            "- карта арены `{}`: `{}` sha256 `{}`",
            map.arena, map.source, map.sha256
        );
    }
    if let Some(b) = &m.stall_baseline {
        let _ = writeln!(
            out,
            "- пауз ВМ (холостой цикл {} мс): {:.4}% интервалов > 0,5 мс, {:.4}% > 2 мс ({:.1} в секунду), максимум {:.1} мс",
            b.duration_ms,
            100.0 * b.frac_over_0_5ms,
            100.0 * b.frac_over_2ms,
            b.stalls_over_2ms_per_s,
            b.max_ms
        );
    }
    let _ = writeln!(out, "- общее время: {:.1} с\n", m.total_wall_s);
    let _ = writeln!(
        out,
        "Победа игрока A: он заморозил соперника (или соперник сам замёрз), пока сам не выбыл. Главная метрика (D-059) — «credited-побед» = игры, выигранные собственным засчитанным блоком, / все игры \
         (победа без credit — соперник сам замёрз, её набирает и бездействие); рядом блоки и самозаморозки в минуту. Винрейт = W/(W+L+D) — \
         вторичная колонка, с 95% ДИ Уилсона [нижняя; верхняя]; «W/все» считает и таймауты. credited/held — среди побед W. \
         Решение — реальное время (мс) p50/p99 по всем решениям игрока; хвост p99 включает паузы ВМ (D-045).\n"
    );
    let _ = writeln!(
        out,
        "| Условие | Арена | Игр | W:L:D:T | credited-побед, % [ДИ] | Винрейт, % [ДИ] | W/все, % [ДИ] | credited | held | блоков/мин | самозаморозок/мин | до 1-го блока, с | решение A, мс | решение соперн., мс | игр/с |"
    );
    let _ = writeln!(
        out,
        "|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|"
    );
    for c in &s.conditions {
        let t = c.tally;
        let a = c.players.first();
        let b = c.players.get(1);
        let dec = |p: Option<&PlayerSummary>| {
            p.map_or("—".to_string(), |p| {
                format!("{}/{}", ms(p.decide_us_p50), ms(p.decide_us_p99))
            })
        };
        let _ = writeln!(
            out,
            "| {} | {} ({}) | {} | {}:{}:{}:{} | {} | {} | {} | {}/{} | {}/{} | {:.2} | {:.2} | {} | {} | {} | {:.2} |",
            c.name,
            c.arena,
            c.arena_tag,
            c.games,
            t.w,
            t.l,
            t.d,
            t.t,
            pct(&c.credited_win_rate),
            pct(&c.win_rate),
            pct(&c.win_rate_all),
            c.credited_w,
            t.w,
            c.held_w,
            t.w,
            c.blocks_per_min,
            c.self_freezes_per_min,
            opt_s(c.time_to_first_block_s.median),
            dec(a),
            dec(b),
            c.games_per_s
        );
    }
    if s.conditions.iter().any(|c| c.band_fraction.is_some()) {
        let _ = writeln!(
            out,
            "\nВейблок (задача 4.2): доля игрового времени игрока A в зале (`inWbHall`: зоны плюс 3 тайла) — держит ли он место — и в полосе `wbBand` \
             (прямоугольник у основания трубы, `bot.ts:4799`; планировщик штрафуется за пребывание в нём только при `bandCost > 0`, по умолчанию 0).\n"
        );
        let _ = writeln!(out, "| Условие | в зале, % | в полосе, % |\n|---|---:|---:|");
        for c in &s.conditions {
            if let (Some(f), Some(h)) = (c.band_fraction, c.hall_fraction) {
                let _ = writeln!(out, "| {} | {:.1} | {:.1} |", c.name, 100.0 * h, 100.0 * f);
            }
        }
    }
    let _ = writeln!(
        out,
        "\nРазбивка по раскладке (порядок спавна: слот 0 первым / последним; позиции: как выпали / обменяны). Заметный разрыв между \
         половинами пары — перекос арены, а не игроков; общий винрейт усредняет все четыре ячейки поровну.\n"
    );
    let _ = writeln!(
        out,
        "| Условие | порядок обычный | порядок обратный | позиции как выпали | позиции обменяны |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|");
    for c in &s.conditions {
        let cell = |x: &SplitStats| {
            format!(
                "{} ({}:{}:{}:{})",
                pct(&x.win_rate),
                x.tally.w,
                x.tally.l,
                x.tally.d,
                x.tally.t
            )
        };
        let sp = &c.splits;
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            c.name,
            cell(&sp.order_normal),
            cell(&sp.order_reversed),
            cell(&sp.position_plain),
            cell(&sp.position_swapped)
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stall_baseline_reports_something_sane() {
        let b = measure_stall_baseline(30);
        assert!(b.samples > 100, "a 30 ms spin must take many reads");
        assert!((0.0..=1.0).contains(&b.frac_over_2ms));
        assert!(b.max_ms >= 0.0);
    }

    #[test]
    fn distribution_median_and_mean() {
        let d = Distribution::of(vec![3.0, 1.0, 2.0, 10.0]);
        assert_eq!(d.n, 4);
        assert_eq!(d.mean, Some(4.0));
        assert_eq!(d.median, Some(2.0));
        assert_eq!(Distribution::of(vec![]).mean, None);
    }
}
