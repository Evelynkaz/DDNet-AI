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
//! **Memory** (8.2b). A dataset can hold tens of millions of samples (one 3.3 h or 11 h demo is most of it) and
//! the corpus lives in RAM, so every demo contributes at most [`HumanConfig::max_samples_per_demo`] samples (per
//! map override [`HumanConfig::map_max_samples_per_demo`]): stretches of `segment_len` consecutive steps are kept or dropped by a deterministic
//! hash, with a probability that gives about that many samples per demo (stretches that carry a technique tag are
//! kept with `tagged_keep_boost` times the probability). The mass a demo then has in the corpus is its kept samples, **not** rescaled to its length, so a long
//! demo does not outweigh a hundred short ones. The sets held out by map are capped separately
//! ([`HumanConfig::holdout_max_samples_per_demo`]). Weights are multiplied by the dataset's
//! ([`HumanConfig::dataset_weights`]) and the map's ([`HumanConfig::map_weights`]: the owner's Copy Love Box family
//! is the priority, D-057).
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
    /// Under a demo cap, runs are cut into segments of this many steps before the keep/drop draw.
    pub segment_len: usize,
    /// Under a demo cap, a segment that carries a technique tag is this many times as likely to be kept.
    pub tagged_keep_boost: f64,
    /// Keep at most this many neighbours per step (the nearest ones, the chosen target always); `0` = all. A crowded
    /// hall has dozens of players in range of every sample, and the corpus lives in RAM.
    pub max_others: usize,
    /// Load at most this many demos (quick experiments and tests).
    pub max_demos: Option<usize>,
    /// `(dataset directory name, weight)`: multiplies the weight of every sample of that dataset.
    pub dataset_weights: Vec<(String, f32)>,
    /// `(map name substring, weight)`: multiplies the weight of samples on matching maps (all matching entries
    /// multiply). The owner's Copy Love Box family gets the larger weight.
    pub map_weights: Vec<(String, f32)>,
    /// A demo keeps about this many samples (`0` = all); see the module docs.
    pub max_samples_per_demo: usize,
    /// `(map name substring, cap)` overriding `max_samples_per_demo` for matching maps (the first match wins).
    pub map_max_samples_per_demo: Vec<(String, usize)>,
    /// Cap per demo of the sets held out by map (evaluation only).
    pub holdout_max_samples_per_demo: usize,
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
            segment_len: 128,
            tagged_keep_boost: 5.0,
            max_others: 0,
            max_demos: None,
            dataset_weights: Vec::new(),
            map_weights: Vec::new(),
            max_samples_per_demo: 0,
            map_max_samples_per_demo: Vec::new(),
            holdout_max_samples_per_demo: 20_000,
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
    /// Runs dropped by the per-demo sample cap, and runs kept because they carry a technique tag.
    pub runs_thinned: usize,
    pub tagged_runs_kept: usize,
    /// Demos whose samples were capped.
    pub demos_capped: usize,
    /// Per map name, over the **training** sequences after the caps and weights: scored steps and the sum of their
    /// sampling weights. The share of a map family (e.g. the owner's Copy Love Box family, D-057) in the human
    /// batch is its weight over the total.
    pub train_scored_by_map: std::collections::BTreeMap<String, usize>,
    pub train_weight_by_map: std::collections::BTreeMap<String, f64>,
}

impl HumanStats {
    /// Share of the training sampling weight on maps whose name contains `word` (`0` without data).
    pub fn weight_share_of(&self, word: &str) -> f64 {
        let total: f64 = self.train_weight_by_map.values().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let part: f64 = self
            .train_weight_by_map
            .iter()
            .filter(|(m, _)| m.contains(word))
            .map(|(_, w)| *w)
            .sum();
        part / total
    }

    /// Share of the scored training steps on maps whose name contains `word`.
    pub fn step_share_of(&self, word: &str) -> f64 {
        let total: usize = self.train_scored_by_map.values().sum();
        if total == 0 {
            return 0.0;
        }
        let part: usize = self
            .train_scored_by_map
            .iter()
            .filter(|(m, _)| m.contains(word))
            .map(|(_, n)| *n)
            .sum();
        part as f64 / total as f64
    }
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
/// A deterministic uniform draw in `[0, 1)` for one run (no RNG state, so any thread count thins alike).
fn run_draw(demo: u32, chunk: usize, run: usize) -> f64 {
    let mut z = (u64::from(demo) << 40) ^ ((chunk as u64) << 20) ^ run as u64 ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// How one demo is read: the multiplier of its weights and the fraction of its runs kept.
#[derive(Debug, Clone, Copy)]
struct DemoPlan {
    weight_mult: f32,
    keep: f64,
}

/// One decoded chunk's contribution: its demo, its sequences and the thinning counters.
type ChunkResult = Result<(u32, Vec<Seq>, ThinStats), HumanError>;

/// Outcome counters of one chunk's thinning.
#[derive(Debug, Clone, Copy, Default)]
struct ThinStats {
    thinned: usize,
    tagged_kept: usize,
}

fn chunk_runs(
    chunk: &ddai_dataset::types::Chunk,
    chunk_index: usize,
    map: &Arc<MapEntry>,
    cfg: &HumanConfig,
    tag_bits: &[(u32, f32)],
    demo: u32,
    plan: DemoPlan,
) -> (Vec<Seq>, ThinStats) {
    let r2 = cfg.others_radius_px * cfg.others_radius_px;
    // Only the technique bits that carry an extra weight make a run "tagged" (the tag field also holds other
    // bits, e.g. the skill bucket, which every sample has).
    let tag_mask: u32 = tag_bits.iter().filter(|(_, w)| *w > 1.0).fold(0, |m, (b, _)| m | b);
    // Pass 1 (cheap, allocation-free per sample): per player label the samples in frame order, cut into runs
    // of consecutive snapshots, each with its weights and a "carries a technique tag" flag; the run's keep/drop
    // decision is made here, so the expensive part (the neighbours of every kept step) is only built for kept runs.
    // A crowded hall puts dozens of players within range of every sample: building their records for samples
    // that are then dropped used to cost gigabytes per chunk (8.2b memory finding).
    let mut per_player: std::collections::BTreeMap<u16, Vec<(usize, f32, bool)>> = std::collections::BTreeMap::new();
    for (si, s) in chunk.samples.iter().enumerate() {
        let frame = &chunk.frames[s.frame as usize];
        let Some(me) = frame.chars.get(s.slot as usize) else {
            continue;
        };
        let weight = sample_weight(
            cfg,
            tag_bits,
            s.confident(),
            me.frozen(),
            s.active(),
            SkillBucket::from_u8(s.skill),
            s.tags,
        ) * plan.weight_mult;
        per_player
            .entry(me.player)
            .or_default()
            .push((si, weight, s.tags & tag_mask != 0 && weight > 0.0));
    }
    let tick_of = |si: usize| chunk.frames[chunk.samples[si].frame as usize].tick;
    let mut out = Vec::new();
    let mut thin = ThinStats::default();
    let mut run_no = 0usize;
    for (_, samples) in per_player {
        let mut start = 0usize;
        let mut runs: Vec<&[(usize, f32, bool)]> = Vec::new();
        for k in 1..=samples.len() {
            let cut = k == samples.len() || {
                let (prev, cur) = (tick_of(samples[k - 1].0), tick_of(samples[k].0));
                cur - prev > cfg.max_gap_ticks || cur <= prev
            };
            if cut {
                runs.push(&samples[start..k]);
                start = k;
            }
        }
        // Under a demo cap a long run is cut into segments, so the keep/drop decision is made on stretches of
        // `segment_len` steps, not on a player's whole stay (which would keep or drop up to a chunk at once).
        let mut pieces: Vec<&[(usize, f32, bool)]> = Vec::new();
        for run in runs {
            if plan.keep < 1.0 && cfg.segment_len >= cfg.min_run && run.len() > cfg.segment_len {
                pieces.extend(run.chunks(cfg.segment_len));
            } else {
                pieces.push(run);
            }
        }
        for run in pieces {
            if run.len() < cfg.min_run || !run.iter().any(|r| r.1 > 0.0) {
                continue;
            }
            let tagged = run.iter().any(|r| r.2);
            // A piece with a technique tag is `tagged_keep_boost` times as likely to be kept (not always: in a
            // crowded hall most stretches carry some tag, and "always" kept the whole demo).
            let p = if tagged {
                (plan.keep * cfg.tagged_keep_boost).min(1.0)
            } else {
                plan.keep
            };
            let keep = plan.keep >= 1.0 || run_draw(demo, chunk_index, run_no) < p;
            run_no += 1;
            if !keep {
                thin.thinned += 1;
                continue;
            }
            thin.tagged_kept += usize::from(tagged && plan.keep < 1.0);
            // Pass 2: the steps of a kept run.
            let steps: Vec<SeqStep> = run
                .iter()
                .enumerate()
                .map(|(k, &(si, weight, _))| {
                    let s = &chunk.samples[si];
                    let frame = &chunk.frames[s.frame as usize];
                    let me = &frame.chars[s.slot as usize];
                    let mut near: Vec<(f32, CharRec)> = frame
                        .chars
                        .iter()
                        .enumerate()
                        .filter(|&(i, c)| {
                            i != s.slot as usize
                                && (dist2(me, c) <= r2 || (s.target >= 0 && i16::from(c.id) == s.target))
                        })
                        .map(|(_, c)| {
                            let mut c = *c;
                            c.aim = [0, 0];
                            // The chosen target sorts first so a cap never drops it.
                            let d = if s.target >= 0 && i16::from(c.id) == s.target {
                                -1.0
                            } else {
                                dist2(me, &c)
                            };
                            (d, c)
                        })
                        .collect();
                    if cfg.max_others > 0 && near.len() > cfg.max_others {
                        near.sort_by(|a, b| a.0.total_cmp(&b.0));
                        near.truncate(cfg.max_others);
                    }
                    let mut me = *me;
                    me.aim = [0, 0];
                    SeqStep {
                        tick: frame.tick,
                        me,
                        others: near.into_iter().map(|(_, c)| c).collect(),
                        target: s.target,
                        label: s.action,
                        soft: None,
                        weight,
                        mask: HeadMask {
                            aim: s.action.hook || s.action.fire,
                            ..HeadMask::ALL
                        },
                        // The latch: the previous decision's hook key in this run (the run's first has none).
                        latch: k > 0 && chunk.samples[run[k - 1].0].action.hook,
                    }
                })
                .collect();
            out.push(Seq {
                map: map.clone(),
                steps,
                source: Source::Human { demo },
            });
        }
    }
    (out, thin)
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
        a.runs_thinned += b.runs_thinned;
        a.tagged_runs_kept += b.tagged_runs_kept;
        a.demos_capped += b.demos_capped;
        for (m, n) in b.train_scored_by_map {
            *a.train_scored_by_map.entry(m).or_default() += n;
        }
        for (m, w) in b.train_weight_by_map {
            *a.train_weight_by_map.entry(m).or_default() += w;
        }
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

    // How each used demo is read.
    let dataset_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let dataset_mult = cfg
        .dataset_weights
        .iter()
        .find(|(n, _)| *n == dataset_name)
        .map_or(1.0, |(_, w)| *w);
    let mut plan_of: std::collections::HashMap<u32, DemoPlan> = std::collections::HashMap::new();
    for (&demo, &split) in &split_of {
        let d = &manifest.demos[demo as usize];
        let name = &manifest.maps[d.map.expect("kept demos have a map") as usize].name;
        let cap = if split == Split::HoldoutMap {
            cfg.holdout_max_samples_per_demo
        } else {
            cfg.map_max_samples_per_demo
                .iter()
                .find(|(m, _)| name.contains(m.as_str()))
                .map_or(cfg.max_samples_per_demo, |(_, c)| *c)
        };
        let keep = if cap == 0 || d.samples == 0 {
            1.0
        } else {
            (cap as f64 / d.samples as f64).min(1.0)
        };
        stats.demos_capped += usize::from(keep < 1.0);
        let map_mult: f32 = cfg
            .map_weights
            .iter()
            .filter(|(m, _)| name.contains(m.as_str()))
            .map(|(_, w)| *w)
            .product();
        plan_of.insert(
            demo,
            DemoPlan {
                weight_mult: dataset_mult * map_mult,
                keep,
            },
        );
    }
    // Chunks of the map-held-out demos are evaluation-only: decode just the share the cap keeps (whole chunks).
    let chunk_ids: Vec<usize> = (0..manifest.chunks.len())
        .filter(|&ci| {
            let demo = manifest.chunks[ci].demo;
            match (split_of.get(&demo), plan_of.get(&demo)) {
                (Some(Split::HoldoutMap), Some(plan)) if plan.keep < 1.0 => run_draw(demo, ci, usize::MAX) < plan.keep,
                (Some(_), _) => true,
                _ => false,
            }
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| HumanError(format!("thread pool: {e}")))?;
    let results: Vec<ChunkResult> = pool.install(|| {
        chunk_ids
            .par_iter()
            .map(|&ci| {
                let chunk = reader.read_chunk(ci)?;
                let map_idx = manifest.demos[chunk.demo as usize].map.expect("kept demos have a map") as usize;
                let map = maps[map_idx].as_ref().expect("map loaded").clone();
                let mut plan = plan_of[&chunk.demo];
                if split_of[&chunk.demo] == Split::HoldoutMap {
                    plan.keep = 1.0; // the chunk was the sampling unit
                }
                let (seqs, thin) = chunk_runs(&chunk, ci, &map, cfg, &tag_bits, chunk.demo, plan);
                Ok((chunk.demo, seqs, thin))
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
        let (demo, seqs, thin) = r?;
        stats.runs_thinned += thin.thinned;
        stats.tagged_runs_kept += thin.tagged_kept;
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
        if split_of[&demo] == Split::Train {
            let map_name = manifest.maps[manifest.demos[demo as usize].map.expect("kept demos have a map") as usize]
                .name
                .clone();
            for sq in &seqs {
                for st in sq.steps.iter().filter(|st| st.weight > 0.0) {
                    *stats.train_scored_by_map.entry(map_name.clone()).or_default() += 1;
                    *stats.train_weight_by_map.entry(map_name.clone()).or_default() += f64::from(st.weight);
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

    /// A chunk of `runs` runs of 10 consecutive snapshots of one player each (a tick gap between runs), every
    /// 7th run carrying a technique tag.
    fn chunk_of_runs(runs: usize) -> ddai_dataset::types::Chunk {
        use ddai_dataset::types::{ActionRec, Chunk, FORMAT_VERSION, FrameRec, SampleRec, quality};
        let mut frames = Vec::new();
        let mut samples = Vec::new();
        for r in 0..runs {
            for k in 0..10 {
                let c = CharRec {
                    id: 0,
                    player: 0,
                    team: 0,
                    pos: [100.0, 100.0],
                    vel: [0.0, 0.0],
                    hook_state: 0,
                    hook_pos: [0.0, 0.0],
                    hooked_player: -1,
                    flags: 0,
                    freeze_ticks: 0,
                    jumps_left: 2,
                    jumps_used: 0,
                    weapon: 1,
                    direction: 0,
                    aim: [0, 0],
                };
                frames.push(FrameRec {
                    tick: (r * 1000 + k * 2) as i32,
                    chars: vec![c],
                });
                samples.push(SampleRec {
                    frame: (r * 10 + k) as u32,
                    slot: 0,
                    action: ActionRec {
                        direction: 1,
                        jump: false,
                        hook: false,
                        fire: false,
                        aim: [1, 0],
                    },
                    target: -1,
                    tags: if r % 7 == 3 { Technique::WallHook.bit() } else { 0 },
                    q: 2 | quality::ACTIVE | quality::NEXT_FRESH,
                    skill: SkillBucket::Top as u8,
                });
            }
        }
        Chunk {
            format_version: FORMAT_VERSION,
            demo: 0,
            frames,
            samples,
        }
    }

    fn map() -> Arc<MapEntry> {
        crate::seq::testutil::room_map()
    }

    #[test]
    fn a_demo_cap_keeps_a_fraction_of_the_runs_deterministically_and_boosts_tagged_ones() {
        let c = cfg();
        let tags = technique_weights(&c);
        let chunk = chunk_of_runs(210);
        let full = chunk_runs(
            &chunk,
            0,
            &map(),
            &c,
            &tags,
            0,
            DemoPlan {
                weight_mult: 1.0,
                keep: 1.0,
            },
        );
        assert_eq!((full.0.len(), full.1.thinned), (210, 0));
        let plan = DemoPlan {
            weight_mult: 1.0,
            keep: 0.3,
        };
        let (a, ta) = chunk_runs(&chunk, 0, &map(), &c, &tags, 0, plan);
        let (b, tb) = chunk_runs(&chunk, 0, &map(), &c, &tags, 0, plan);
        let ticks = |s: &[Seq]| s.iter().map(|q| q.steps[0].tick).collect::<Vec<_>>();
        assert_eq!(ticks(&a), ticks(&b), "the same chunk thins the same way every time");
        assert_eq!((ta.thinned, ta.tagged_kept), (tb.thinned, tb.tagged_kept));
        let tagged_runs = (0..210).filter(|r| r % 7 == 3).count();
        assert_eq!(ta.tagged_kept, tagged_runs, "every tagged run is kept");
        assert_eq!(a.len() + ta.thinned, 210);
        let untagged_kept = a.len() - tagged_runs;
        let untagged = 210 - tagged_runs;
        assert!(
            (untagged_kept as f64 / untagged as f64 - 0.3).abs() < 0.12,
            "kept {untagged_kept} of {untagged} untagged runs"
        );
        // A different chunk index draws differently (no two chunks of a demo drop the same runs).
        let (other, _) = chunk_runs(&chunk, 1, &map(), &c, &tags, 0, plan);
        assert_ne!(ticks(&a), ticks(&other));
    }

    #[test]
    fn under_a_cap_a_long_stay_is_cut_into_segments_that_are_kept_or_dropped_separately() {
        use ddai_dataset::types::{Chunk, FORMAT_VERSION, FrameRec, SampleRec, quality};
        let c = cfg();
        let tags = technique_weights(&c);
        let base = chunk_of_runs(1);
        // One player present for 1000 consecutive snapshots.
        let mut frames = Vec::new();
        let mut samples = Vec::new();
        for k in 0..1000u32 {
            frames.push(FrameRec {
                tick: (k * 2) as i32,
                chars: base.frames[0].chars.clone(),
            });
            let mut s: SampleRec = base.samples[0];
            s.frame = k;
            s.tags = 0;
            s.q = 2 | quality::ACTIVE | quality::NEXT_FRESH;
            samples.push(s);
        }
        let chunk = Chunk {
            format_version: FORMAT_VERSION,
            demo: 3,
            frames,
            samples,
        };
        let (whole, _) = chunk_runs(
            &chunk,
            0,
            &map(),
            &c,
            &tags,
            3,
            DemoPlan {
                weight_mult: 1.0,
                keep: 1.0,
            },
        );
        assert_eq!(whole.len(), 1, "an uncapped demo keeps the stay as one run");
        assert_eq!(whole[0].steps.len(), 1000);
        let mut kept_lens = Vec::new();
        for chunk_index in 0..40 {
            let (seqs, _) = chunk_runs(
                &chunk,
                chunk_index,
                &map(),
                &c,
                &tags,
                3,
                DemoPlan {
                    weight_mult: 1.0,
                    keep: 0.3,
                },
            );
            kept_lens.extend(seqs.iter().map(|s| s.steps.len()));
        }
        assert!(
            kept_lens.iter().all(|&n| n == 128 || n == 1000 - 7 * 128),
            "{kept_lens:?}"
        );
        let kept_steps: usize = kept_lens.iter().sum();
        let share = kept_steps as f64 / (40.0 * 1000.0);
        assert!((share - 0.3).abs() < 0.07, "about 30% of the steps are kept: {share}");
    }

    #[test]
    fn a_neighbour_cap_keeps_the_nearest_players_and_the_chosen_target() {
        let c = HumanConfig { max_others: 4, ..cfg() };
        let tags = technique_weights(&c);
        let mut chunk = chunk_of_runs(1);
        // A crowd: 20 more players at growing distances; sample 0..10 target the farthest one (id 20).
        for f in &mut chunk.frames {
            let me = f.chars[0];
            for k in 1..=20u8 {
                let mut other = me;
                other.id = k;
                other.player = u16::from(k);
                other.pos = [me.pos[0] + 10.0 * f32::from(k), me.pos[1]];
                f.chars.push(other);
            }
        }
        for s in &mut chunk.samples {
            s.target = 20;
        }
        let (seqs, _) = chunk_runs(
            &chunk,
            0,
            &map(),
            &c,
            &tags,
            0,
            DemoPlan {
                weight_mult: 1.0,
                keep: 1.0,
            },
        );
        // Only player 0 has samples, so one run.
        let seq = &seqs[0];
        for st in &seq.steps {
            assert_eq!(st.others.len(), 4);
            let ids: Vec<u8> = st.others.iter().map(|o| o.id).collect();
            assert!(ids.contains(&20), "the target survives the cap: {ids:?}");
            assert!(
                ids.contains(&1) && ids.contains(&2) && ids.contains(&3),
                "the nearest survive: {ids:?}"
            );
        }
        // Uncapped keeps all 20 (all within the radius).
        let all = HumanConfig { max_others: 0, ..cfg() };
        let (seqs, _) = chunk_runs(
            &chunk,
            0,
            &map(),
            &all,
            &tags,
            0,
            DemoPlan {
                weight_mult: 1.0,
                keep: 1.0,
            },
        );
        assert!(seqs[0].steps.iter().all(|st| st.others.len() == 20));
    }

    #[test]
    fn the_effective_share_of_a_map_family_is_reported_from_weights_and_steps() {
        let mut st = HumanStats::default();
        st.train_scored_by_map.insert("Copy Love Box".into(), 100);
        st.train_scored_by_map.insert("Copy The Box TF".into(), 50);
        st.train_scored_by_map.insert("BlmapChill".into(), 150);
        st.train_weight_by_map.insert("Copy Love Box".into(), 300.0);
        st.train_weight_by_map.insert("Copy The Box TF".into(), 150.0);
        st.train_weight_by_map.insert("BlmapChill".into(), 150.0);
        assert!((st.step_share_of("Copy") - 0.5).abs() < 1e-9);
        assert!(
            (st.weight_share_of("Copy") - 0.75).abs() < 1e-9,
            "a map weight of 3 moves the sampling share, not the step share"
        );
        assert_eq!(HumanStats::default().weight_share_of("Copy"), 0.0);
    }

    #[test]
    fn dataset_and_map_weights_multiply_the_sample_weights() {
        let c = cfg();
        let tags = technique_weights(&c);
        let chunk = chunk_of_runs(2);
        let (plain, _) = chunk_runs(
            &chunk,
            0,
            &map(),
            &c,
            &tags,
            0,
            DemoPlan {
                weight_mult: 1.0,
                keep: 1.0,
            },
        );
        let (boosted, _) = chunk_runs(
            &chunk,
            0,
            &map(),
            &c,
            &tags,
            0,
            DemoPlan {
                weight_mult: 3.0,
                keep: 1.0,
            },
        );
        for (p, b) in plain.iter().zip(&boosted) {
            for (x, y) in p.steps.iter().zip(&b.steps) {
                assert!((y.weight - 3.0 * x.weight).abs() < 1e-6);
            }
        }
        assert!(plain[0].steps.iter().all(|s| s.weight == 1.0));
    }

    #[test]
    fn unknown_technique_codes_are_ignored() {
        let mut c = cfg();
        c.tag_weights = vec![("T99".into(), 9.0), ("WH".into(), 2.0)];
        assert_eq!(technique_weights(&c), vec![(Technique::WallHook.bit(), 2.0)]);
    }
}
