//! The human-play dataset (`ddai-dataset`, task 8.4c) as training sequences (D-014, D-040).
//!
//! **Sequences.** A recurrent learner needs consecutive decisions, so the reader's shuffled
//! sample stream is not used: each chunk (one demo) is decoded whole and the samples of one
//! anonymous player label are cut into *runs* of consecutive snapshots (a gap of more than
//! `max_gap_ticks` starts a new run). Every snapshot stays in its run as context; only the
//! confidently reconstructed, unfrozen ones of a sufficiently skilled player carry weight.
//!
//! **Weights** (the data's own labels do the filtering, never the player identity - D-040):
//! * `0` unless the physics replay reproduced the next state (`Within1px` or better), the actor
//!   is not frozen (its input is ignored by the server) and its skill bucket is at least
//!   [`HumanConfig::min_skill`];
//! * otherwise `1` for an *active* step (a key held, movement, or a hook out) and
//!   [`HumanConfig::idle_weight`] for a standing-still one (the majority of the data);
//! * multiplied by the largest matching entry of [`HumanConfig::tag_weights`]: techniques (D-048;
//!   above all the wall/ceiling hook under threat) are rare and deserve more draws;
//! * a skill below `Mid` is scaled by [`HumanConfig::low_skill_weight`].
//!
//! **Splits.** Demos on the maps named in [`HumanConfig::exclude_map_names`] (ChillBlock5, the
//! arena's holdout map) never enter training: they form the `holdout_map` set. Of the rest every
//! `val_every`-th demo is validation, so validation is unseen *demos* (never unseen samples of a
//! seen demo).

use std::path::PathBuf;
use std::sync::Arc;

use ddai_dataset::dataset::{DatasetError, DatasetReader};
use ddai_dataset::tags::Technique;
use ddai_dataset::types::{CharRec, SkillBucket};
use ddai_fly::bc::HeadMask;
use rayon::prelude::*;

use crate::seq::{MapEntry, Seq, SeqStep, Source};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HumanConfig {
    /// Human datasets (`ddai-dataset` directories); their sequences are pooled.
    pub dataset_dirs: Vec<PathBuf>,
    /// Maps (by exact name) that are evaluation-only.
    pub exclude_map_names: Vec<String>,
    /// Maps whose name contains one of these words are evaluation-only too (e.g. `Copy` for the
    /// whole Copy Love Box family, when a run must not train on human play of the arena's maps).
    pub exclude_map_substrings: Vec<String>,
    /// Every k-th non-excluded demo is validation (`0` = no validation split).
    pub val_every: u32,
    pub min_skill: String,
    pub low_skill_weight: f32,
    pub idle_weight: f32,
    /// `(technique code, weight)`; codes are `T1`..`T18` and `WH`.
    pub tag_weights: Vec<(String, f32)>,
    pub max_gap_ticks: i32,
    pub min_run: usize,
    /// Other characters farther than this from the actor are dropped (the encoder ignores anything
    /// beyond its 20-tile range; the chosen target is always kept).
    pub others_radius_px: f32,
    /// Load at most this many demos (quick experiments and tests).
    pub max_demos: Option<usize>,
}

impl Default for HumanConfig {
    fn default() -> Self {
        HumanConfig {
            dataset_dirs: Vec::new(),
            exclude_map_names: vec!["ChillBlock5".to_string()],
            exclude_map_substrings: Vec::new(),
            val_every: 10,
            min_skill: "low".to_string(),
            low_skill_weight: 0.5,
            idle_weight: 0.1,
            tag_weights: vec![
                ("WH".into(), 6.0),
                ("T1".into(), 3.0),
                ("T2".into(), 3.0),
                ("T3".into(), 3.0),
                ("T4".into(), 3.0),
                ("T5".into(), 3.0),
                ("T9".into(), 2.0),
                ("T10".into(), 4.0),
                ("T12".into(), 2.0),
                ("T13".into(), 1.5),
                ("T14".into(), 3.0),
                ("T15".into(), 1.5),
            ],
            max_gap_ticks: 3,
            min_run: 8,
            others_radius_px: 800.0,
            max_demos: None,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct HumanStats {
    pub demos_used: usize,
    pub demos_excluded_map: usize,
    pub runs: usize,
    pub steps: usize,
    pub scored_steps: usize,
    /// Scored steps whose weight exceeds `1` (a technique tag multiplied it).
    pub upweighted_steps: usize,
    pub train_runs: usize,
    pub val_runs: usize,
    pub holdout_map_runs: usize,
}

/// Human sequences split for training, validation and map-holdout evaluation.
pub struct HumanData {
    pub train: Vec<Seq>,
    pub val: Vec<Seq>,
    pub holdout_map: Vec<Seq>,
    pub stats: HumanStats,
}

fn parse_skill(name: &str) -> SkillBucket {
    match name.to_ascii_lowercase().as_str() {
        "top" => SkillBucket::Top,
        "mid" => SkillBucket::Mid,
        "low" => SkillBucket::Low,
        _ => SkillBucket::Unranked,
    }
}

fn technique_weights(cfg: &HumanConfig) -> Vec<(u32, f32)> {
    cfg.tag_weights
        .iter()
        .filter_map(|(code, w)| Technique::from_code(code).map(|t| (t.bit(), *w)))
        .collect()
}

/// The sampling weight of one human sample (see the module docs); `0` = context only.
pub fn sample_weight(
    cfg: &HumanConfig,
    tag_bits: &[(u32, f32)],
    confident: bool,
    frozen: bool,
    active: bool,
    skill: SkillBucket,
    tags: u32,
) -> f32 {
    if !confident || frozen || skill < parse_skill(&cfg.min_skill) {
        return 0.0;
    }
    let mut w = if active { 1.0 } else { cfg.idle_weight };
    let tag_mult = tag_bits
        .iter()
        .filter(|(bit, _)| tags & bit != 0)
        .map(|(_, m)| *m)
        .fold(1.0f32, f32::max);
    w *= tag_mult;
    if skill < SkillBucket::Mid {
        w *= cfg.low_skill_weight;
    }
    w
}

fn dist2(a: &CharRec, b: &CharRec) -> f32 {
    let (dx, dy) = (a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]);
    dx * dx + dy * dy
}

/// Cuts one decoded chunk into runs.
fn chunk_runs(
    chunk: &ddai_dataset::types::Chunk,
    map: &Arc<MapEntry>,
    cfg: &HumanConfig,
    tag_bits: &[(u32, f32)],
    demo: u32,
) -> Vec<Seq> {
    let r2 = cfg.others_radius_px * cfg.others_radius_px;
    // Samples per player label, in frame order (the dataset stores them that way).
    let mut per_player: std::collections::BTreeMap<u16, Vec<SeqStep>> = std::collections::BTreeMap::new();
    for s in &chunk.samples {
        let frame = &chunk.frames[s.frame as usize];
        let Some(me) = frame.chars.get(s.slot as usize) else {
            continue;
        };
        let others: Vec<CharRec> = frame
            .chars
            .iter()
            .enumerate()
            .filter(|&(i, c)| {
                i != s.slot as usize && (dist2(me, c) <= r2 || (s.target >= 0 && i16::from(c.id) == s.target))
            })
            .map(|(_, c)| {
                let mut c = *c;
                c.aim = [0, 0];
                c
            })
            .collect();
        let weight = sample_weight(
            cfg,
            tag_bits,
            s.confident(),
            me.frozen(),
            s.active(),
            SkillBucket::from_u8(s.skill),
            s.tags,
        );
        let mut me = *me;
        me.aim = [0, 0];
        let aim_matters = s.action.hook || s.action.fire;
        per_player.entry(me.player).or_default().push(SeqStep {
            tick: frame.tick,
            me,
            others,
            target: s.target,
            label: s.action,
            soft: None,
            weight,
            mask: HeadMask {
                aim: aim_matters,
                ..HeadMask::ALL
            },
        });
    }
    let mut out = Vec::new();
    for (_, steps) in per_player {
        let mut run: Vec<SeqStep> = Vec::new();
        let mut flush = |run: &mut Vec<SeqStep>| {
            if run.len() >= cfg.min_run && run.iter().any(|s| s.weight > 0.0) {
                out.push(Seq {
                    map: map.clone(),
                    steps: std::mem::take(run),
                    source: Source::Human { demo },
                });
            } else {
                run.clear();
            }
        };
        for st in steps {
            if run
                .last()
                .is_some_and(|prev| st.tick - prev.tick > cfg.max_gap_ticks || st.tick <= prev.tick)
            {
                flush(&mut run);
            }
            run.push(st);
        }
        flush(&mut run);
    }
    out
}

#[derive(Debug)]
pub struct HumanError(pub String);

impl std::fmt::Display for HumanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HumanError {}

impl From<DatasetError> for HumanError {
    fn from(e: DatasetError) -> Self {
        HumanError(e.to_string())
    }
}

/// Loads every configured human dataset and pools them into training, validation and
/// map-holdout sequences.
pub fn load_human(cfg: &HumanConfig, threads: usize) -> Result<HumanData, HumanError> {
    if cfg.dataset_dirs.is_empty() {
        return Err(HumanError("no human dataset configured".to_string()));
    }
    let mut out = HumanData {
        train: Vec::new(),
        val: Vec::new(),
        holdout_map: Vec::new(),
        stats: HumanStats::default(),
    };
    for dir in &cfg.dataset_dirs {
        let one = load_human_one(cfg, dir, threads)?;
        out.train.extend(one.train);
        out.val.extend(one.val);
        out.holdout_map.extend(one.holdout_map);
        let (a, b) = (&mut out.stats, one.stats);
        a.demos_used += b.demos_used;
        a.demos_excluded_map += b.demos_excluded_map;
        a.runs += b.runs;
        a.steps += b.steps;
        a.scored_steps += b.scored_steps;
        a.upweighted_steps += b.upweighted_steps;
        a.train_runs += b.train_runs;
        a.val_runs += b.val_runs;
        a.holdout_map_runs += b.holdout_map_runs;
    }
    Ok(out)
}

fn load_human_one(cfg: &HumanConfig, dir: &std::path::Path, threads: usize) -> Result<HumanData, HumanError> {
    let reader = DatasetReader::open(dir)?;
    let manifest = &reader.manifest;
    let tag_bits = technique_weights(cfg);

    // Maps, loaded once.
    let mut maps: Vec<Option<Arc<MapEntry>>> = Vec::with_capacity(manifest.maps.len());
    for m in &manifest.maps {
        let path = dir.join(&m.file);
        let bytes = std::fs::read(&path).map_err(|e| HumanError(format!("{}: {e}", path.display())))?;
        let loaded = ddai_map::load_map(&bytes).map_err(|e| HumanError(format!("{}: {e}", path.display())))?;
        maps.push(Some(MapEntry::new(Arc::new(loaded.data))));
    }

    // Which demos are used, and into which split.
    #[derive(Clone, Copy, PartialEq)]
    enum Split {
        Train,
        Val,
        HoldoutMap,
    }
    let mut kept: Vec<u32> = Vec::new();
    let mut split_of: std::collections::HashMap<u32, Split> = std::collections::HashMap::new();
    let mut stats = HumanStats::default();
    for (i, d) in manifest.demos.iter().enumerate() {
        let (Some(map_idx), true) = (d.map, d.status == "ok") else {
            continue;
        };
        let name = &manifest.maps[map_idx as usize].name;
        if cfg.exclude_map_names.iter().any(|x| x == name)
            || cfg.exclude_map_substrings.iter().any(|x| name.contains(x.as_str()))
        {
            split_of.insert(i as u32, Split::HoldoutMap);
            stats.demos_excluded_map += 1;
        } else {
            kept.push(i as u32);
        }
    }
    for (k, &i) in kept.iter().enumerate() {
        let is_val = cfg.val_every > 0 && (k as u32 + 1).is_multiple_of(cfg.val_every);
        split_of.insert(i, if is_val { Split::Val } else { Split::Train });
    }
    // `max_demos` limits the total (deterministically: lowest demo indices first).
    if let Some(max) = cfg.max_demos {
        let mut ids: Vec<u32> = split_of.keys().copied().collect();
        ids.sort_unstable();
        for id in ids.into_iter().skip(max) {
            split_of.remove(&id);
        }
    }
    stats.demos_used = split_of.len();

    let chunk_ids: Vec<usize> = (0..manifest.chunks.len())
        .filter(|&ci| split_of.contains_key(&manifest.chunks[ci].demo))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| HumanError(format!("thread pool: {e}")))?;
    let results: Vec<Result<(u32, Vec<Seq>), HumanError>> = pool.install(|| {
        chunk_ids
            .par_iter()
            .map(|&ci| {
                let chunk = reader.read_chunk(ci)?;
                let map_idx = manifest.demos[chunk.demo as usize].map.expect("kept demos have a map") as usize;
                let map = maps[map_idx].as_ref().expect("map loaded").clone();
                Ok((chunk.demo, chunk_runs(&chunk, &map, cfg, &tag_bits, chunk.demo)))
            })
            .collect()
    });
    let mut out = HumanData {
        train: Vec::new(),
        val: Vec::new(),
        holdout_map: Vec::new(),
        stats: HumanStats::default(),
    };
    for r in results {
        let (demo, seqs) = r?;
        for s in &seqs {
            stats.runs += 1;
            stats.steps += s.steps.len();
            for st in &s.steps {
                if st.weight > 0.0 {
                    stats.scored_steps += 1;
                    stats.upweighted_steps += usize::from(st.weight > 1.0);
                }
            }
        }
        match split_of[&demo] {
            Split::Train => out.train.extend(seqs),
            Split::Val => out.val.extend(seqs),
            Split::HoldoutMap => out.holdout_map.extend(seqs),
        }
    }
    stats.train_runs = out.train.len();
    stats.val_runs = out.val.len();
    stats.holdout_map_runs = out.holdout_map.len();
    out.stats = stats;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HumanConfig {
        HumanConfig::default()
    }

    #[test]
    fn weights_follow_the_documented_rules() {
        let c = cfg();
        let tags = technique_weights(&c);
        let w =
            |confident, frozen, active, skill, t: u32| sample_weight(&c, &tags, confident, frozen, active, skill, t);
        assert_eq!(
            w(false, false, true, SkillBucket::Top, 0),
            0.0,
            "unconfident reconstruction"
        );
        assert_eq!(w(true, true, true, SkillBucket::Top, 0), 0.0, "frozen actor");
        assert_eq!(w(true, false, true, SkillBucket::Unranked, 0), 0.0, "below min_skill");
        assert_eq!(w(true, false, true, SkillBucket::Top, 0), 1.0);
        assert_eq!(w(true, false, false, SkillBucket::Mid, 0), c.idle_weight);
        assert_eq!(w(true, false, true, SkillBucket::Low, 0), c.low_skill_weight);
        assert_eq!(w(true, false, true, SkillBucket::Top, Technique::WallHook.bit()), 6.0);
        // The largest matching technique weight wins; it is not a product.
        let both = Technique::WallHook.bit() | Technique::T9.bit();
        assert_eq!(w(true, false, true, SkillBucket::Top, both), 6.0);
        // A technique tag on an idle low-skill sample still multiplies the reduced base.
        assert!((w(true, false, false, SkillBucket::Low, Technique::T10.bit()) - 0.1 * 4.0 * 0.5).abs() < 1e-6);
    }

    #[test]
    fn unknown_technique_codes_are_ignored() {
        let mut c = cfg();
        c.tag_weights = vec![("T99".into(), 9.0), ("WH".into(), 2.0)];
        assert_eq!(technique_weights(&c), vec![(Technique::WallHook.bit(), 2.0)]);
    }
}
