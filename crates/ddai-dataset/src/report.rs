//! The aggregate report of a dataset build: usable hours, reconstruction quality (overall, by
//! map, by skill bucket, per input channel), skill summary, technique frequency and success with
//! example hits. Contains no names and no file paths; demos are identified by sha256 prefixes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::dataset::{DemoEntry, MapEntry, SampleTotals};
use crate::demo::DemoOutput;
use crate::pipeline::BuildCounters;
use crate::replay::{Channel, ReplayStats};
use crate::skill::{RankRow, TICK_RATE};
use crate::tags::Technique;
use crate::types::{ReplayClass, SkillBucket};

/// Hours of `ticks` server ticks.
pub fn hours(ticks: i64) -> f64 {
    ticks as f64 / f64::from(TICK_RATE) / 3600.0
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DemoCounts {
    pub total: u64,
    pub ok: u64,
    pub skipped: u64,
    /// Skip reason -> number of demos.
    pub skipped_reasons: BTreeMap<String, u64>,
    pub map_embedded: u64,
    pub map_from_cache: u64,
    /// Demos that ended early on a decode error (frames before it are kept).
    pub decode_errors: u64,
    /// Demos whose bytes are identical to an earlier one (dropped).
    pub duplicates: u64,
    /// Demos in which some snapshot carried tuning other than DDNet's defaults.
    pub tune_nondefault: u64,
    /// Demos in which some snapshot carried non-default tuning in a movement-relevant field.
    pub tune_movement_nondefault: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TimeStats {
    /// Wall-clock time covered by the demos that were read (sum of their snapshot spans).
    pub recorded_hours: f64,
    /// Time covered by snapshots that hold at least one confidently reconstructed sample.
    pub usable_hours: f64,
    /// Character-hours (every visible character counts).
    pub character_hours: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SampleCounts {
    pub total: u64,
    pub confident: u64,
    pub active: u64,
    pub active_confident: u64,
    pub next_fresh: u64,
    pub frozen_actor: u64,
    /// By skill bucket (indexed by [`SkillBucket`] as `u8`).
    pub by_skill: [u64; 4],
    pub confident_by_skill: [u64; 4],
    pub tagged: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupReplay {
    pub name: String,
    pub demos: u64,
    pub hours: f64,
    pub stats: ReplayStats,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillSummary {
    pub players: u64,
    pub ranked: u64,
    pub buckets: [u64; 4],
    pub freeze_entries: u64,
    pub credited_freezes: u64,
    pub blocks: u64,
    pub self_freezes: u64,
    pub kills_credited: u64,
    /// The best ranked players, anonymous.
    pub top: Vec<RankRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    /// First 12 hex characters of the demo's sha256.
    pub demo: String,
    pub tick: i32,
    pub success: bool,
    /// Anonymous per-demo labels of the actor and the other party (for `dataset show`).
    pub actor: u16,
    pub other: Option<u16>,
}

/// Active events of one technique that have one `detail` bit set (the meaning of the bit is
/// technique specific, see `technique.rs`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BitRow {
    pub bit: u8,
    pub active: u64,
    pub success: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TechniqueRow {
    pub code: String,
    pub name: String,
    pub detectable: bool,
    pub not_detected_reason: Option<String>,
    /// Events with the technique actually applied.
    pub active: u64,
    pub success: u64,
    /// At-risk baseline without a counter-measure (defensive techniques only).
    pub passive: u64,
    pub passive_success: u64,
    /// Distinct `(demo, player)` actors of the active events.
    pub actors: u64,
    /// Active events of `Top`/`Mid` players and how many succeeded.
    pub good_active: u64,
    pub good_success: u64,
    /// Samples carrying this tag.
    pub samples: u64,
    /// Per `detail` bit of the active events (only bits that occur).
    pub bits: Vec<BitRow>,
    /// Up to five example hits: a mix of successes (up to three) and failures (the rest).
    pub examples: Vec<Example>,
}

impl TechniqueRow {
    pub fn success_rate(&self) -> f64 {
        if self.active == 0 {
            0.0
        } else {
            self.success as f64 / self.active as f64
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub source: String,
    pub code_commit: String,
    pub config_hash: String,
    pub demos: DemoCounts,
    pub time: TimeStats,
    pub samples: SampleCounts,
    pub chars: BuildCounters,
    pub replay_total: ReplayStats,
    pub replay_by_map: Vec<GroupReplay>,
    pub replay_by_skill: Vec<GroupReplay>,
    pub skill: SkillSummary,
    pub techniques: Vec<TechniqueRow>,
}

/// Picks up to `n` items evenly spread over `pool` (deterministic).
fn spread<T: Clone>(pool: &[T], n: usize) -> Vec<T> {
    if pool.len() <= n {
        return pool.to_vec();
    }
    if n <= 1 {
        return pool.iter().take(n).cloned().collect();
    }
    (0..n).map(|i| pool[i * (pool.len() - 1) / (n - 1)].clone()).collect()
}

/// Up to five examples: three successes and two failures, the short side filled from the other.
fn mix_examples<T: Clone>(ok: &[T], bad: &[T]) -> Vec<T> {
    let n_ok = ok
        .len()
        .min(if bad.is_empty() { 5 } else { 3 })
        .max(5usize.saturating_sub(bad.len()).min(ok.len()));
    let mut ex = spread(ok, n_ok);
    ex.extend(spread(bad, 5 - ex.len()));
    ex
}

/// Builds the report. `outputs[i]` and `totals[i]` belong to `demos[i]` (`None` for skipped
/// demos); `totals` are the sample counters the dataset writer collected while writing the chunks;
/// `ranking` carries the global buckets keyed by `(demo index, player label)`.
pub fn build(
    demos: &[DemoEntry],
    maps: &[MapEntry],
    outputs: &[Option<DemoOutput>],
    totals: &[Option<SampleTotals>],
    ranking: &[RankRow],
    counts: DemoCounts,
    top_players: usize,
) -> Report {
    let bucket_of: BTreeMap<(u32, u16), SkillBucket> =
        ranking.iter().map(|r| ((r.demo, r.skill.player), r.bucket)).collect();
    let mut rep = Report {
        demos: counts,
        ..Report::default()
    };
    let mut by_map: BTreeMap<u32, GroupReplay> = BTreeMap::new();
    let mut by_skill: BTreeMap<u8, ReplayStats> = BTreeMap::new();
    let mut tech: BTreeMap<Technique, TechniqueRow> = Technique::ALL
        .iter()
        .map(|&t| {
            (
                t,
                TechniqueRow {
                    code: t.code().to_string(),
                    name: t.name().to_string(),
                    detectable: t.detectable(),
                    not_detected_reason: t.not_detected_reason().map(str::to_string),
                    active: 0,
                    success: 0,
                    passive: 0,
                    passive_success: 0,
                    actors: 0,
                    good_active: 0,
                    good_success: 0,
                    samples: 0,
                    bits: Vec::new(),
                    examples: Vec::new(),
                },
            )
        })
        .collect();
    let mut actors: BTreeMap<Technique, std::collections::BTreeSet<(u32, u16)>> = BTreeMap::new();
    let mut pools: BTreeMap<Technique, (Vec<Example>, Vec<Example>)> = BTreeMap::new();

    for (di, (entry, out)) in demos.iter().zip(outputs).enumerate() {
        let Some(out) = out else { continue };
        let di = di as u32;
        rep.chars.ext_chars += out.counters.ext_chars;
        rep.chars.inferred_chars += out.counters.inferred_chars;
        rep.chars.frozen_chars += out.counters.frozen_chars;
        rep.chars.chars += out.counters.chars;
        rep.chars.gaps += out.counters.gaps;
        rep.chars.chars_without_info += out.counters.chars_without_info;
        rep.chars.fresh_chars += out.counters.fresh_chars;
        rep.chars.conv_both += out.counters.conv_both;
        rep.chars.conv_ext_only += out.counters.conv_ext_only;
        rep.chars.conv_weapon_only += out.counters.conv_weapon_only;
        rep.chars.conv_neither += out.counters.conv_neither;

        let demo_ticks = if out.frame_count == 0 {
            0
        } else {
            i64::from(out.last_tick) - i64::from(out.first_tick) + 2
        };
        rep.time.recorded_hours += hours(demo_ticks);
        rep.time.character_hours += hours(out.counters.chars as i64 * 2);
        if let Some(t) = totals.get(di as usize).and_then(Option::as_ref) {
            rep.samples.total += t.total;
            for b in 0..4 {
                rep.samples.by_skill[b] += t.by_skill[b];
                rep.samples.confident_by_skill[b] += t.confident_by_skill[b];
            }
            rep.samples.confident += t.confident;
            rep.samples.active += t.active;
            rep.samples.active_confident += t.active_confident;
            rep.samples.next_fresh += t.next_fresh;
            rep.samples.frozen_actor += t.frozen_actor;
            rep.samples.tagged += t.tagged;
            for (technique, n) in &t.by_technique {
                if let Some(row) = tech.get_mut(technique) {
                    row.samples += n;
                }
            }
            rep.time.usable_hours += hours(t.usable_frames as i64 * 2);
        }

        for (label, stats) in &out.replay {
            rep.replay_total.merge(stats);
            let b = bucket_of.get(&(di, *label)).copied().unwrap_or(SkillBucket::Unranked);
            by_skill.entry(b as u8).or_default().merge(stats);
            if let Some(mi) = entry.map {
                let g = by_map.entry(mi).or_insert_with(|| GroupReplay {
                    name: maps
                        .get(mi as usize)
                        .map(|m| format!("{} ({})", m.name, &m.sha256[..8]))
                        .unwrap_or_default(),
                    demos: 0,
                    hours: 0.0,
                    stats: ReplayStats::default(),
                });
                g.stats.merge(stats);
            }
        }
        if let Some(mi) = entry.map
            && let Some(g) = by_map.get_mut(&mi)
        {
            g.demos += 1;
            g.hours += hours(demo_ticks);
        }

        for p in &out.players {
            rep.skill.players += 1;
            rep.skill.self_freezes += u64::from(p.self_freezes);
            rep.skill.kills_credited += u64::from(p.kills_credited);
        }
        rep.skill.freeze_entries += u64::from(out.freeze_entries);
        rep.skill.credited_freezes += u64::from(out.credited_freezes);
        rep.skill.blocks += u64::from(out.blocks);

        for e in &out.events {
            let Some(row) = tech.get_mut(&e.technique) else {
                continue;
            };
            let good = matches!(bucket_of.get(&(di, e.actor)), Some(SkillBucket::Top | SkillBucket::Mid));
            if e.active {
                row.active += 1;
                if e.success {
                    row.success += 1;
                }
                if good {
                    row.good_active += 1;
                    if e.success {
                        row.good_success += 1;
                    }
                }
                actors.entry(e.technique).or_default().insert((di, e.actor));
                for bit in 0..8u8 {
                    if e.detail & (1 << bit) != 0 {
                        match row.bits.iter_mut().find(|b| b.bit == bit) {
                            Some(b) => {
                                b.active += 1;
                                b.success += u64::from(e.success);
                            }
                            None => row.bits.push(BitRow {
                                bit,
                                active: 1,
                                success: u64::from(e.success),
                            }),
                        }
                    }
                }
                let ex = Example {
                    demo: entry.sha256[..12].to_string(),
                    tick: e.start_tick,
                    success: e.success,
                    actor: e.actor,
                    other: e.other,
                };
                let pool = pools.entry(e.technique).or_default();
                if e.success {
                    pool.0.push(ex);
                } else {
                    pool.1.push(ex);
                }
            } else {
                row.passive += 1;
                if e.success {
                    row.passive_success += 1;
                }
            }
        }
    }
    for (t, row) in tech.iter_mut() {
        row.actors = actors.get(t).map_or(0, |s| s.len() as u64);
        if let Some((ok, bad)) = pools.get(t) {
            row.examples = mix_examples(ok, bad);
        }
    }
    rep.techniques = tech.into_values().collect();
    rep.replay_by_map = by_map.into_values().collect();
    rep.replay_by_skill = by_skill
        .into_iter()
        .map(|(b, stats)| GroupReplay {
            name: SkillBucket::from_u8(b).name().to_string(),
            demos: 0,
            hours: 0.0,
            stats,
        })
        .collect();
    rep.skill.ranked = ranking.iter().filter(|r| r.rank > 0).count() as u64;
    for r in ranking {
        rep.skill.buckets[r.bucket as usize] += 1;
    }
    rep.skill.top = ranking
        .iter()
        .filter(|r| r.rank > 0)
        .take(top_players)
        .cloned()
        .collect();
    rep
}

/// Human-readable rendering (English; the Russian write-up quotes these tables).
pub fn render(rep: &Report) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
    let _ = writeln!(s, "source: {}", rep.source);
    let _ = writeln!(s, "code commit: {}   config hash: {}", rep.code_commit, rep.config_hash);
    let d = &rep.demos;
    let _ = writeln!(
        s,
        "demos: {} total, {} ok, {} skipped, {} duplicates, {} with a decode error; maps: {} embedded, {} from the local cache",
        d.total, d.ok, d.skipped, d.duplicates, d.decode_errors, d.map_embedded, d.map_from_cache
    );
    let _ = writeln!(
        s,
        "demos with non-default tuning: {} (movement-relevant fields: {})",
        d.tune_nondefault, d.tune_movement_nondefault
    );
    for (reason, n) in &d.skipped_reasons {
        let _ = writeln!(s, "  skipped: {n} x {reason}");
    }
    let t = &rep.time;
    let _ = writeln!(
        s,
        "time: recorded {:.2} h, usable (>= 1 confident sample) {:.2} h, character-hours {:.1} h",
        t.recorded_hours, t.usable_hours, t.character_hours
    );
    let c = &rep.samples;
    let _ = writeln!(
        s,
        "samples: {} total, {} confident ({:.1}%), {} active, {} active+confident, {} next-fresh, {} with a frozen actor, {} tagged",
        c.total,
        c.confident,
        pct(c.confident, c.total),
        c.active,
        c.active_confident,
        c.next_fresh,
        c.frozen_actor,
        c.tagged
    );
    let ch = &rep.chars;
    let _ = writeln!(
        s,
        "characters: {} snapshots-chars, {} with DDNetCharacter, {} freeze-inferred, {} frozen, {} fresh wire cores, {} gaps",
        ch.chars, ch.ext_chars, ch.inferred_chars, ch.frozen_chars, ch.fresh_chars, ch.gaps
    );
    let conv_frozen = ch.conv_both + ch.conv_ext_only;
    let conv_weapon = ch.conv_both + ch.conv_weapon_only;
    let _ = writeln!(
        s,
        "ninja-weapon freeze convention on {} characters with the real extension: recall {:.1}% ({}/{}), precision {:.1}% ({}/{})",
        ch.ext_chars,
        pct(ch.conv_both, conv_frozen),
        ch.conv_both,
        conv_frozen,
        pct(ch.conv_both, conv_weapon),
        ch.conv_both,
        conv_weapon
    );
    let _ = writeln!(
        s,
        "\nreplay of the reconstructed inputs (2 ticks, vs the demo's next snapshot):"
    );
    let row = |name: &str, st: &ReplayStats| {
        format!(
            "  {name:32} n={:8} exact {:5.1}%  within1px {:5.1}%  off {:5.1}% | active n={:8} exact {:5.1}% conf {:5.1}% | fresh n={:8} conf {:5.1}%",
            st.samples,
            pct(st.by_class[ReplayClass::Exact as usize], st.samples),
            pct(st.by_class[ReplayClass::Within1px as usize], st.samples),
            pct(st.by_class[ReplayClass::Off as usize], st.samples),
            st.active_samples,
            pct(st.active_by_class[ReplayClass::Exact as usize], st.active_samples),
            pct(ReplayStats::confident(&st.active_by_class), st.active_samples),
            st.fresh_samples,
            pct(ReplayStats::confident(&st.fresh_by_class), st.fresh_samples),
        )
    };
    let _ = writeln!(s, "{}", row("all", &rep.replay_total));
    for g in &rep.replay_by_map {
        let _ = writeln!(s, "{}", row(&format!("map {}", g.name), &g.stats));
    }
    for g in &rep.replay_by_skill {
        let _ = writeln!(s, "{}", row(&format!("skill {}", g.name), &g.stats));
    }
    let _ = writeln!(
        s,
        "\nper input channel (active samples; ablation = channel neutralised):"
    );
    for c in Channel::ALL {
        let cs = &rep.replay_total.channels[c as usize];
        let _ = writeln!(
            s,
            "  {:10} active {:8}  confirmed {:5.1}%  unconstrained {:5.1}%  contradicted {:5.1}%  fresh {:5.1}%",
            c.name(),
            cs.active,
            pct(cs.confirmed, cs.active),
            pct(cs.unconstrained, cs.active),
            pct(cs.contradicted, cs.active),
            pct(cs.fresh, cs.active),
        );
    }
    let _ = writeln!(
        s,
        "  fire by attack_tick: the replayed weapon fired on the demo's exact tick in {} of {} fire events ({:.1}%)",
        rep.replay_total.fire_tick_match,
        rep.replay_total.fire_events,
        pct(rep.replay_total.fire_tick_match, rep.replay_total.fire_events)
    );
    let k = &rep.skill;
    let _ = writeln!(
        s,
        "\nskill: {} players, {} ranked (top {} / mid {} / low {}), {} freeze entries, {} credited, {} blocks, {} self-freezes, {} credited kills",
        k.players,
        k.ranked,
        k.buckets[3],
        k.buckets[2],
        k.buckets[1],
        k.freeze_entries,
        k.credited_freezes,
        k.blocks,
        k.self_freezes,
        k.kills_credited
    );
    let _ = writeln!(
        s,
        "\ntechniques (active = technique applied; success = attack froze the target / escape stayed free):"
    );
    for t in &rep.techniques {
        if !t.detectable {
            let _ = writeln!(
                s,
                "  {:3} {:38} not detected: {}",
                t.code,
                t.name,
                t.not_detected_reason.as_deref().unwrap_or("")
            );
            continue;
        }
        let _ = writeln!(
            s,
            "  {:3} {:38} active {:6} success {:6} ({:5.1}%) good-players {:5}/{:5} passive {:5} ({:5.1}% free) actors {:4} samples {:7}",
            t.code,
            t.name,
            t.active,
            t.success,
            100.0 * t.success_rate(),
            t.good_success,
            t.good_active,
            t.passive,
            pct(t.passive_success, t.passive),
            t.actors,
            t.samples
        );
        for b in &t.bits {
            let _ = writeln!(
                s,
                "        detail bit {}: active {:5} success {:5} ({:5.1}%)",
                1u32 << b.bit,
                b.active,
                b.success,
                pct(b.success, b.active)
            );
        }
        for e in &t.examples {
            let _ = writeln!(
                s,
                "        demo {} tick {} players {}{} {}",
                e.demo,
                e.tick,
                e.actor,
                e.other.map(|o| format!(",{o}")).unwrap_or_default(),
                if e.success { "success" } else { "fail" }
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_build_renders_every_section_without_dividing_by_zero() {
        let rep = build(&[], &[], &[], &[], &[], DemoCounts::default(), 5);
        assert_eq!(rep.techniques.len(), Technique::ALL.len());
        let text = render(&rep);
        assert!(text.contains("demos: 0 total"));
        assert!(text.contains("T1 "));
        assert!(text.contains("not detected"));
        assert!(text.contains("recall 0.0%"));
        assert_eq!(rep.samples.total, 0);
    }

    #[test]
    fn examples_mix_successes_and_failures() {
        let ok: Vec<i32> = (0..10).collect();
        let bad: Vec<i32> = (100..110).collect();
        assert_eq!(mix_examples(&ok, &bad), vec![0, 4, 9, 100, 109]);
        assert_eq!(mix_examples(&ok, &bad[..1]), vec![0, 3, 6, 9, 100]);
        assert_eq!(mix_examples(&ok[..1], &bad), vec![0, 100, 103, 106, 109]);
        assert_eq!(mix_examples(&ok, &[]).len(), 5);
        assert_eq!(mix_examples(&[], &bad).len(), 5);
        assert!(mix_examples::<i32>(&[], &[]).is_empty());
        assert_eq!(mix_examples(&ok[..2], &bad[..1]), vec![0, 1, 100]);
    }

    #[test]
    fn spread_is_even_deterministic_and_total() {
        let pool: Vec<u32> = (0..100).collect();
        assert_eq!(spread(&pool, 5), vec![0, 24, 49, 74, 99]);
        assert_eq!(spread(&pool, 1), vec![0]);
        assert!(spread(&pool, 0).is_empty());
        assert_eq!(spread(&[1, 2], 5), vec![1, 2]);
        assert!(spread::<u32>(&[], 5).is_empty());
    }
}
