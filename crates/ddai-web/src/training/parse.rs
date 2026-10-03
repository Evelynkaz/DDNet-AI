//! Parsing of what a training job writes into its run directory (`docs/formats.md` §22.4, §29): `status.json`,
//! `metrics.jsonl`, `config.toml` and the arena `summary.json`.
//!
//! The parsers are forgiving about *content* (an unknown field is ignored, a missing one is `None`, a malformed
//! `metrics.jsonl` line is skipped and counted) and strict about *shape and size*: every string that reaches the page is
//! stripped of control characters and cut, every number must be finite, and the number of records kept is capped. Nothing
//! here touches the filesystem.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use super::Limits;

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

const MAX_TEXT: usize = 120;
/// Distinct phase names kept (a real run has `bc` and `dagger-1…5`).
const MAX_PHASES: usize = 64;

/// Removes control characters and cuts to `max` characters.
pub fn clean_text(s: &str, max: usize) -> String {
    s.chars().filter(|c| !c.is_control()).take(max).collect()
}

fn at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cur = v;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    at(v, path)?.as_f64().filter(|x| x.is_finite())
}

fn uint(v: &Value, path: &[&str]) -> Option<u64> {
    at(v, path)?.as_u64()
}

fn text(v: &Value, path: &[&str]) -> Option<String> {
    at(v, path)?.as_str().map(|s| clean_text(s, MAX_TEXT))
}

fn str_list(v: &Value, path: &[&str], max_items: usize) -> Vec<String> {
    at(v, path)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .take(max_items)
                .map(|s| clean_text(s, MAX_TEXT))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// Wilson interval and `Ci`
// ---------------------------------------------------------------------------------------------

/// A proportion with its 95% confidence interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Ci {
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
}

/// The 95% Wilson score interval of `k` successes in `n` trials (the same one `ddai-env` writes into `summary.json`).
/// `None` for `n == 0` or `k > n`.
pub fn wilson(k: u64, n: u64) -> Option<Ci> {
    if n == 0 || k > n {
        return None;
    }
    const Z: f64 = 1.959_963_984_540_054;
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let z2 = Z * Z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    Some(Ci {
        p,
        lo: (centre - half).max(0.0),
        hi: (centre + half).min(1.0),
    })
}

/// `[p, lo, hi]` (metrics.jsonl) or `{"p":…,"lo":…,"hi":…}` (summary.json).
fn ci_at(v: &Value, path: &[&str]) -> Option<Ci> {
    let x = at(v, path)?;
    let (p, lo, hi) = match x {
        Value::Array(a) if a.len() == 3 => (a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?),
        Value::Object(_) => (num(x, &["p"])?, num(x, &["lo"])?, num(x, &["hi"])?),
        _ => return None,
    };
    [p, lo, hi].iter().all(|n| n.is_finite()).then_some(Ci { p, lo, hi })
}

// ---------------------------------------------------------------------------------------------
// status.json
// ---------------------------------------------------------------------------------------------

/// `status.json`: the phase and step, plus (while the job trains) the progress inside the phase.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct StatusInfo {
    pub phase: Option<String>,
    pub step: Option<u64>,
    pub phase_step: Option<u64>,
    pub phase_steps: Option<u64>,
    pub loss: Option<f64>,
    pub elapsed_s: Option<f64>,
    pub unix_s: Option<u64>,
}

pub fn parse_status(bytes: &[u8]) -> Option<StatusInfo> {
    let v: Value = serde_json::from_slice(bytes).ok()?;
    if !v.is_object() {
        return None;
    }
    Some(StatusInfo {
        phase: text(&v, &["phase"]).filter(|p| !p.is_empty()),
        step: uint(&v, &["step"]),
        phase_step: uint(&v, &["phase_step"]),
        phase_steps: uint(&v, &["phase_steps"]),
        loss: num(&v, &["loss"]),
        elapsed_s: num(&v, &["elapsed_s"]),
        unix_s: uint(&v, &["unix_s"]),
    })
}

// ---------------------------------------------------------------------------------------------
// config.toml
// ---------------------------------------------------------------------------------------------

/// The few facts of `config.toml` the panel shows (a whitelist: paths and everything else stay on the disk).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ConfigSummary {
    pub name: Option<String>,
    /// `fly`, `mlp` or `gru`.
    pub kind: Option<String>,
    pub hidden: Option<u64>,
    pub seed: Option<u64>,
    pub own_hook_mode: Option<String>,
    pub human_fraction: Option<f64>,
    pub bc_steps: Option<u64>,
    pub steps_per_round: Option<u64>,
    /// Number of DAgger rounds (the length of `dagger.betas`).
    pub rounds: Option<u64>,
    /// The total number of steps the config plans (BC + rounds), when all of its parts are there.
    pub planned_steps: Option<u64>,
    pub eval_games: Option<u64>,
    pub eval_arenas: Vec<String>,
}

pub fn parse_config(src: &str) -> Option<ConfigSummary> {
    let doc: toml::Table = src.parse().ok()?;
    let v = serde_json::to_value(&doc).ok()?;
    let bc_steps = uint(&v, &["bc_steps"]);
    let steps_per_round = uint(&v, &["dagger", "steps_per_round"]);
    let rounds = at(&v, &["dagger", "betas"])
        .and_then(Value::as_array)
        .map(|a| a.len() as u64);
    let planned_steps = match (bc_steps, steps_per_round, rounds) {
        (Some(bc), Some(per), Some(r)) => per.checked_mul(r).and_then(|x| x.checked_add(bc)),
        _ => None,
    };
    Some(ConfigSummary {
        name: text(&v, &["name"]),
        kind: text(&v, &["model", "kind"]),
        hidden: uint(&v, &["model", "hidden"]),
        seed: uint(&v, &["train", "seed"]),
        own_hook_mode: text(&v, &["train", "own_hook", "mode"]),
        human_fraction: num(&v, &["train", "human_fraction"]),
        bc_steps,
        steps_per_round,
        rounds,
        planned_steps,
        eval_games: uint(&v, &["dagger", "eval_games"]),
        eval_arenas: str_list(&v, &["dagger", "eval_arenas"], 16),
    })
}

// ---------------------------------------------------------------------------------------------
// metrics.jsonl
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrainPoint {
    pub step: u64,
    pub phase: String,
    pub total: Option<f64>,
    pub dir: Option<f64>,
    pub jump: Option<f64>,
    pub hook: Option<f64>,
    pub fire: Option<f64>,
    pub aim: Option<f64>,
    pub grad_norm: Option<f64>,
}

/// One validation report (`kind = "eval"`): the key numbers per head, for one eval set.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvalPoint {
    pub step: u64,
    pub phase: String,
    pub set: String,
    pub dir_acc: Option<f64>,
    pub dir_top2: Option<f64>,
    pub hook_auroc: Option<f64>,
    pub jump_auroc: Option<f64>,
    pub fire_auroc: Option<f64>,
    /// Hook accuracy where the own hook is not out (starting a hook) and where it is (releasing it).
    pub start_acc: Option<f64>,
    pub release_acc: Option<f64>,
    pub aim_within_15: Option<f64>,
}

/// One arena evaluation of a model (`kind = "arena"` in `metrics.jsonl`, or a condition of a `summary.json`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArenaPoint {
    pub step: Option<u64>,
    pub phase: Option<String>,
    /// The arena (or, for a summary condition, its full name such as `clb-left vs scripted`).
    pub arena: String,
    pub opponents: Vec<String>,
    pub games: Option<u64>,
    pub w: Option<u64>,
    pub l: Option<u64>,
    pub d: Option<u64>,
    pub t: Option<u64>,
    pub credited_w: Option<u64>,
    /// D-059: the headline metric, credited wins / all games, with the Wilson interval.
    pub credited: Option<Ci>,
    pub win_rate: Option<Ci>,
    pub win_rate_all: Option<Ci>,
    pub blocks_per_min: Option<f64>,
    pub self_freezes_per_min: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CollectRec {
    pub round: u64,
    pub beta: Option<f64>,
    pub jobs: u64,
    pub resumed: Option<u64>,
    pub games: u64,
    pub w: u64,
    pub l: u64,
    pub d: u64,
    pub t: u64,
    pub steps: u64,
}

/// `kind = "hook_play"`: how often the student starts / releases a hook against the teacher on the same states.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HookPlayRec {
    pub round: u64,
    pub start_student: Option<f64>,
    pub start_teacher: Option<f64>,
    pub release_student: Option<f64>,
    pub release_teacher: Option<f64>,
    pub steps: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThresholdRec {
    pub step: Option<u64>,
    pub phase: Option<String>,
    pub jump: Option<f64>,
    pub hook: Option<f64>,
    pub fire: Option<f64>,
    pub sets: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionRec {
    pub phase: Option<String>,
    pub arenas: Vec<String>,
    /// `(phase, mean credited win rate over the selection arenas)`.
    pub table: Vec<(String, f64)>,
}

/// Where a phase (`bc`, `dagger-1`, …) starts on the step axis: the DAgger round markers of the charts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PhaseMark {
    pub name: String,
    pub start_step: u64,
    pub end_step: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Metrics {
    pub train: Vec<TrainPoint>,
    pub eval: Vec<EvalPoint>,
    pub arena: Vec<ArenaPoint>,
    pub collect: Vec<CollectRec>,
    pub hook_play: Vec<HookPlayRec>,
    pub thresholds: Vec<ThresholdRec>,
    pub selection: Option<SelectionRec>,
    pub phases: Vec<PhaseMark>,
    /// Lines that were not valid JSON objects, or were longer than the line cap.
    pub skipped_lines: u64,
    /// `train` points were thinned to fit the cap on points sent to the page.
    pub train_thinned: bool,
}

fn push_capped<T>(v: &mut Vec<T>, item: T, cap: usize) {
    if v.len() >= cap {
        v.remove(0);
    }
    v.push(item);
}

/// Keeps at most `max` points by taking every n-th (and always the last), preserving order.
pub fn thin<T: Clone>(points: &[T], max: usize) -> Vec<T> {
    if max == 0 {
        return Vec::new();
    }
    if points.len() <= max {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(max);
    let mut out: Vec<T> = points.iter().step_by(stride).cloned().collect();
    if let Some(last) = points.last()
        && !(points.len() - 1).is_multiple_of(stride)
    {
        if out.len() >= max {
            out.pop();
        }
        out.push(last.clone());
    }
    out
}

fn eval_point(v: &Value) -> Option<EvalPoint> {
    Some(EvalPoint {
        step: uint(v, &["step"])?,
        phase: text(v, &["phase"]).unwrap_or_default(),
        set: text(v, &["set"]).unwrap_or_default(),
        dir_acc: num(v, &["report", "dir", "accuracy"]),
        dir_top2: num(v, &["report", "dir", "top2_accuracy"]),
        hook_auroc: num(v, &["report", "hook", "auroc"]),
        jump_auroc: num(v, &["report", "jump", "auroc"]),
        fire_auroc: num(v, &["report", "fire", "auroc"]),
        start_acc: num(v, &["report", "hook_by_state", "start_accuracy"]),
        release_acc: num(v, &["report", "hook_by_state", "release_accuracy"]),
        aim_within_15: num(v, &["report", "aim", "within_15deg"]),
    })
}

/// The arena fields shared by `metrics.jsonl`'s `eval` object and a `summary.json` condition.
fn arena_fields(e: &Value, arena: String) -> ArenaPoint {
    let games = uint(e, &["games"]);
    let credited_w = uint(e, &["credited_w"]);
    // Old summaries have no `credited_win_rate`: it is `credited_w / games` with the Wilson interval (ddai-env README).
    let credited = ci_at(e, &["credited_win_rate"]).or_else(|| match (credited_w, games) {
        (Some(k), Some(n)) => wilson(k, n),
        _ => None,
    });
    let (w, l, d, t) = if e.get("tally").is_some() {
        (
            uint(e, &["tally", "w"]),
            uint(e, &["tally", "l"]),
            uint(e, &["tally", "d"]),
            uint(e, &["tally", "t"]),
        )
    } else {
        (uint(e, &["w"]), uint(e, &["l"]), uint(e, &["d"]), uint(e, &["t"]))
    };
    ArenaPoint {
        step: None,
        phase: None,
        arena,
        opponents: str_list(e, &["opponents"], 8),
        games,
        w,
        l,
        d,
        t,
        credited_w,
        credited,
        win_rate: ci_at(e, &["win_rate"]),
        win_rate_all: ci_at(e, &["win_rate_all"]),
        blocks_per_min: num(e, &["blocks_per_min"]),
        self_freezes_per_min: num(e, &["self_freezes_per_min"]),
    }
}

fn arena_point(v: &Value) -> Option<ArenaPoint> {
    let e = v.get("eval")?;
    let arena = text(e, &["arena"])?;
    let mut p = arena_fields(e, arena);
    p.step = uint(v, &["step"]);
    p.phase = text(v, &["phase"]);
    Some(p)
}

fn collect_rec(v: &Value) -> Option<CollectRec> {
    let round = uint(v, &["round"])?;
    let mut rec = CollectRec {
        round,
        beta: num(v, &["beta"]),
        jobs: 0,
        resumed: uint(v, &["jobs_resumed"]),
        games: 0,
        w: 0,
        l: 0,
        d: 0,
        t: 0,
        steps: 0,
    };
    for job in v.get("jobs").and_then(Value::as_array).into_iter().flatten().take(256) {
        rec.jobs += 1;
        rec.games = rec.games.saturating_add(uint(job, &["games"]).unwrap_or(0));
        rec.w = rec.w.saturating_add(uint(job, &["w"]).unwrap_or(0));
        rec.l = rec.l.saturating_add(uint(job, &["l"]).unwrap_or(0));
        rec.d = rec.d.saturating_add(uint(job, &["d"]).unwrap_or(0));
        rec.t = rec.t.saturating_add(uint(job, &["t"]).unwrap_or(0));
        rec.steps = rec.steps.saturating_add(uint(job, &["steps"]).unwrap_or(0));
    }
    Some(rec)
}

fn hook_play_rec(v: &Value) -> Option<HookPlayRec> {
    Some(HookPlayRec {
        round: uint(v, &["round"])?,
        start_student: num(v, &["report", "start_student"]),
        start_teacher: num(v, &["report", "start_teacher"]),
        release_student: num(v, &["report", "release_student"]),
        release_teacher: num(v, &["report", "release_teacher"]),
        steps: uint(v, &["report", "steps"]),
    })
}

fn selection_rec(v: &Value) -> SelectionRec {
    let table = v
        .get("table")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(64)
        .filter_map(|row| {
            let row = row.as_array()?;
            let name = clean_text(row.first()?.as_str()?, MAX_TEXT);
            let score = row.get(1)?.as_f64().filter(|x| x.is_finite())?;
            Some((name, score))
        })
        .collect();
    SelectionRec {
        phase: text(v, &["phase"]),
        arenas: str_list(v, &["arenas"], 16),
        table,
    }
}

/// Parses the bytes of a `metrics.jsonl` (or the tail of one, already cut at a line boundary by
/// [`super::read::read_tail`]) into the series the page draws.
pub fn parse_metrics(bytes: &[u8], limits: &Limits) -> Metrics {
    let mut m = Metrics::default();
    // Per phase in order of first appearance: (name, min step, max step).
    let mut phases: Vec<(String, u64, u64)> = Vec::new();
    let cap = limits.max_records;
    let mut all_train: Vec<TrainPoint> = Vec::new();

    for line in bytes.split(|&b| b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if line.len() > limits.max_line_bytes {
            m.skipped_lines += 1;
            continue;
        }
        let Ok(v) = serde_json::from_slice::<Value>(line) else {
            m.skipped_lines += 1;
            continue;
        };
        if !v.is_object() {
            m.skipped_lines += 1;
            continue;
        }
        if let (Some(phase), Some(step)) = (text(&v, &["phase"]), uint(&v, &["step"]))
            && !phase.is_empty()
        {
            match phases.iter().position(|p| p.0 == phase) {
                Some(i) => {
                    phases[i].1 = phases[i].1.min(step);
                    phases[i].2 = phases[i].2.max(step);
                }
                None if phases.len() < MAX_PHASES => phases.push((phase, step, step)),
                None => {}
            }
        }
        match v.get("kind").and_then(Value::as_str) {
            Some("train") => {
                if let (Some(step), Some(phase)) = (uint(&v, &["step"]), text(&v, &["phase"])) {
                    // Not capped here: the tail read already bounds how many there can be, and thinning below keeps the
                    // whole step range in view instead of only the newest records.
                    all_train.push(TrainPoint {
                        step,
                        phase,
                        total: num(&v, &["loss", "total"]),
                        dir: num(&v, &["loss", "dir"]),
                        jump: num(&v, &["loss", "jump"]),
                        hook: num(&v, &["loss", "hook"]),
                        fire: num(&v, &["loss", "fire"]),
                        aim: num(&v, &["loss", "aim"]),
                        grad_norm: num(&v, &["grad_norm"]),
                    });
                }
            }
            Some("eval") => {
                if let Some(p) = eval_point(&v) {
                    push_capped(&mut m.eval, p, cap);
                }
            }
            Some("arena") => {
                if let Some(p) = arena_point(&v) {
                    push_capped(&mut m.arena, p, cap);
                }
            }
            Some("collect") => {
                if let Some(p) = collect_rec(&v) {
                    push_capped(&mut m.collect, p, cap);
                }
            }
            Some("hook_play") => {
                if let Some(p) = hook_play_rec(&v) {
                    push_capped(&mut m.hook_play, p, cap);
                }
            }
            Some("thresholds") => push_capped(
                &mut m.thresholds,
                ThresholdRec {
                    step: uint(&v, &["step"]),
                    phase: text(&v, &["phase"]),
                    jump: num(&v, &["jump"]),
                    hook: num(&v, &["hook"]),
                    fire: num(&v, &["fire"]),
                    sets: str_list(&v, &["sets"], 8),
                },
                cap,
            ),
            Some("selection") => m.selection = Some(selection_rec(&v)),
            _ => {}
        }
    }

    m.train_thinned = all_train.len() > limits.max_series_points;
    m.train = thin(&all_train, limits.max_series_points);

    // Phase marks: a phase starts where the previous one ended (the first one at its smallest step, 0 for `bc`).
    let mut prev_end = 0;
    for (i, (name, min_step, max_step)) in phases.into_iter().enumerate() {
        let start_step = if i == 0 && name != "bc" { min_step } else { prev_end };
        prev_end = max_step;
        m.phases.push(PhaseMark {
            name,
            start_step: start_step.min(max_step),
            end_step: max_step,
        });
    }
    m
}

// ---------------------------------------------------------------------------------------------
// Arena summary.json
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArenaSummary {
    /// The directory the summary came from (`arena`, `arena-half`, …).
    pub source: String,
    pub run_name: Option<String>,
    /// First 12 hex of the commit the arena binary was built from.
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub base_seed: Option<u64>,
    pub conditions: Vec<ArenaPoint>,
    pub conditions_truncated: bool,
}

pub fn parse_arena_summary(bytes: &[u8], source: &str, max_conditions: usize) -> Option<ArenaSummary> {
    let v: Value = serde_json::from_slice(bytes).ok()?;
    let conds = v.get("conditions")?.as_array()?;
    let conditions = conds
        .iter()
        .take(max_conditions)
        .filter_map(|c| {
            let name = text(c, &["name"]).or_else(|| text(c, &["arena"]))?;
            let mut p = arena_fields(c, name);
            // A condition's opponents live in `players`, which is not forwarded; the name already says it.
            p.opponents.clear();
            Some(p)
        })
        .collect();
    Some(ArenaSummary {
        source: clean_text(source, 64),
        run_name: text(&v, &["meta", "run_name"]),
        git_commit: text(&v, &["meta", "git_commit"]).map(|c| c.chars().take(12).collect()),
        git_dirty: at(&v, &["meta", "git_dirty"]).and_then(Value::as_bool),
        base_seed: uint(&v, &["meta", "base_seed"]),
        conditions,
        conditions_truncated: conds.len() > max_conditions,
    })
}

/// The latest arena evaluation per arena (the record with the largest step, then the latest in the file).
pub fn latest_arena_by_name(points: &[ArenaPoint]) -> BTreeMap<String, &ArenaPoint> {
    let mut best: BTreeMap<String, &ArenaPoint> = BTreeMap::new();
    for p in points {
        match best.get(&p.arena) {
            Some(cur) if cur.step > p.step => {}
            _ => {
                best.insert(p.arena.clone(), p);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn limits() -> Limits {
        Limits::default()
    }

    #[test]
    fn wilson_matches_known_values_and_edge_cases() {
        // 10 of 20: centre 0.5, the textbook interval is about 0.299..0.701.
        let ci = wilson(10, 20).unwrap();
        assert!((ci.p - 0.5).abs() < 1e-12);
        assert!((ci.lo - 0.2993).abs() < 1e-3, "{ci:?}");
        assert!((ci.hi - 0.7007).abs() < 1e-3, "{ci:?}");
        // 0 of 1000 (real data: credited 0/300 -> hi 0.0126 in summary.json).
        let zero = wilson(0, 300).unwrap();
        assert_eq!(zero.p, 0.0);
        assert!(zero.lo.abs() < 1e-12 && (zero.hi - 0.012_64).abs() < 1e-4, "{zero:?}");
        // 28 of 1000 against the value `ddai-env` wrote (E-008): 0.019442 .. 0.040170.
        let real = wilson(28, 1000).unwrap();
        assert!(
            (real.lo - 0.019_442).abs() < 1e-5 && (real.hi - 0.040_170).abs() < 1e-5,
            "{real:?}"
        );
        assert!(wilson(0, 0).is_none() && wilson(5, 4).is_none());
        let all = wilson(7, 7).unwrap();
        assert!(all.hi <= 1.0 && all.lo > 0.5);
    }

    #[test]
    fn status_parses_a_done_and_a_running_file_and_rejects_junk() {
        let done = parse_status(br#"{"phase":"done","step":6500}"#).unwrap();
        assert_eq!(done.phase.as_deref(), Some("done"));
        assert_eq!(done.step, Some(6500));
        assert_eq!(done.loss, None);
        let running = parse_status(
            br#"{"elapsed_s":248.5,"loss":1.01,"phase":"dagger-4","phase_step":1000,"phase_steps":1000,"step":7000,"unix_s":1791017330}"#,
        )
        .unwrap();
        assert_eq!(running.phase_step, Some(1000));
        assert_eq!(running.unix_s, Some(1_791_017_330));
        assert!((running.loss.unwrap() - 1.01).abs() < 1e-12);
        assert!(parse_status(b"[1,2]").is_none());
        assert!(parse_status(b"not json").is_none());
        assert!(parse_status(b"").is_none());
        // Control characters and over-long strings are cleaned.
        let weird = parse_status(format!(r#"{{"phase":"a\u0007b{}"}}"#, "x".repeat(500)).as_bytes()).unwrap();
        let phase = weird.phase.unwrap();
        assert!(phase.len() <= MAX_TEXT && !phase.contains('\u{7}'));
    }

    const CONFIG: &str = r#"
name = "e008-p2-mlpw"
flyg = "~/aiddnet/data/connectome/compiled/fly-S-v1.flyg"
bc_steps = 3000
[model]
kind = "mlp"
hidden = 64
[train]
seed = 2
human_fraction = 0.1
[train.own_hook]
mode = "mask_hook_head"
[dagger]
betas = [
  0.5,
  0.4,
  0.3,
  0.2,
  0.15,
]
steps_per_round = 1000
eval_games = 300
eval_arenas = ["clb-left", "pit", "platform"]
"#;

    #[test]
    fn config_summary_is_a_whitelist_and_computes_the_planned_steps() {
        let c = parse_config(CONFIG).unwrap();
        assert_eq!(c.kind.as_deref(), Some("mlp"));
        assert_eq!(c.seed, Some(2));
        assert_eq!(c.own_hook_mode.as_deref(), Some("mask_hook_head"));
        assert_eq!(c.rounds, Some(5));
        assert_eq!(c.planned_steps, Some(8000));
        assert_eq!(c.eval_arenas, ["clb-left", "pit", "platform"]);
        // The paths of the config are not part of the summary type at all.
        let json = serde_json::to_string(&c).unwrap();
        assert!(!json.contains("aiddnet") && !json.contains("flyg"));
        // A partial config still parses; a broken one is None.
        let partial = parse_config("bc_steps = 10\n").unwrap();
        assert_eq!(partial.planned_steps, None);
        assert!(parse_config("this is = = not toml").is_none());
    }

    fn line(v: Value) -> String {
        v.to_string() + "\n"
    }

    fn sample_metrics() -> String {
        let mut s = String::new();
        for step in [50u64, 100, 150] {
            s += &line(json!({"kind":"train","phase":"bc","step":step,"grad_norm":3.5,
                "loss":{"total":3.0,"dir":1.0,"jump":0.9,"hook":0.6,"fire":0.9,"aim":-1.2}}));
        }
        s += &line(
            json!({"kind":"eval","phase":"bc","set":"teacher-val","step":150,"report":{
            "dir":{"accuracy":0.7,"top2_accuracy":0.92},"hook":{"auroc":0.83},"jump":{"auroc":0.67},"fire":{"auroc":0.73},
            "hook_by_state":{"start_accuracy":0.48,"release_accuracy":0.09},"aim":{"within_15deg":0.39}}}),
        );
        s += &line(
            json!({"kind":"arena","phase":"bc","step":150,"eval":{"arena":"clb-left","games":300,"w":117,"l":162,"d":4,"t":17,
            "credited_w":0,"credited_win_rate":[0.0,0.0,0.0126],"win_rate":[0.41,0.36,0.47],"win_rate_all":[0.39,0.34,0.45],
            "blocks_per_min":0.0,"self_freezes_per_min":3.85,"opponents":["scripted"]}}),
        );
        s += &line(json!({"kind":"collect","round":1,"beta":0.5,"jobs_resumed":0,"jobs":[
            {"arena":"clb-left","games":250,"w":222,"l":26,"d":2,"t":0,"steps":38735},
            {"arena":"pit","games":60,"w":29,"l":0,"d":0,"t":31,"steps":34277}]}));
        s += &line(
            json!({"kind":"thresholds","phase":"bc","step":150,"jump":0.5,"hook":0.62,"fire":0.53,"sets":["teacher-val"]}),
        );
        for step in [200u64, 250] {
            s += &line(json!({"kind":"train","phase":"dagger-1","step":step,"loss":{"total":2.0,"dir":0.9}}));
        }
        s += &line(
            json!({"kind":"hook_play","round":1,"report":{"start_student":0.34,"start_teacher":0.4,
            "release_student":0.02,"release_teacher":0.2,"steps":49020}}),
        );
        s += &line(
            json!({"kind":"selection","phase":"dagger-1","arenas":["clb-left"],"table":[["bc",0.0],["dagger-1",0.04]]}),
        );
        s
    }

    #[test]
    fn metrics_are_parsed_into_series_and_phase_marks() {
        let m = parse_metrics(sample_metrics().as_bytes(), &limits());
        assert_eq!(m.skipped_lines, 0);
        assert_eq!(m.train.len(), 5);
        assert_eq!(m.train[0].step, 50);
        assert_eq!(m.train[0].aim, Some(-1.2));
        assert_eq!(m.train[3].fire, None, "a missing head stays None");
        assert_eq!(m.eval.len(), 1);
        assert_eq!(m.eval[0].set, "teacher-val");
        assert_eq!(m.eval[0].hook_auroc, Some(0.83));
        assert_eq!(m.eval[0].start_acc, Some(0.48));
        assert_eq!(m.arena.len(), 1);
        let a = &m.arena[0];
        assert_eq!(
            (a.games, a.w, a.l, a.d, a.t),
            (Some(300), Some(117), Some(162), Some(4), Some(17))
        );
        assert_eq!(a.credited.unwrap().hi, 0.0126);
        assert_eq!(a.opponents, ["scripted"]);
        assert_eq!(m.collect[0].games, 310);
        assert_eq!(m.collect[0].w, 251);
        assert_eq!(m.collect[0].steps, 73012);
        assert_eq!(m.hook_play[0].release_teacher, Some(0.2));
        assert_eq!(m.thresholds[0].sets, ["teacher-val"]);
        assert_eq!(m.selection.as_ref().unwrap().table[1], ("dagger-1".to_string(), 0.04));
        // bc: 0..150, dagger-1 starts where bc ended.
        assert_eq!(
            m.phases,
            vec![
                PhaseMark {
                    name: "bc".into(),
                    start_step: 0,
                    end_step: 150
                },
                PhaseMark {
                    name: "dagger-1".into(),
                    start_step: 150,
                    end_step: 250
                },
            ]
        );
    }

    #[test]
    fn bad_lines_are_skipped_and_counted_not_fatal() {
        let mut s = String::from("not json\n\n[1,2,3]\n\"str\"\n{\"kind\":\"train\"\n");
        s += &line(json!({"kind":"train","phase":"bc","step":10,"loss":{"total":1.0}}));
        s += &line(json!({"kind":"train","phase":"bc","loss":{"total":1.0}})); // no step: ignored, not an error
        s += &line(json!({"kind":"unknown","x":1}));
        s += "{\"kind\":\"train\",\"phase\":\"bc\",\"step\":20,\"loss\":{\"total\":NaN}}\n"; // Python NaN is not JSON
        let m = parse_metrics(s.as_bytes(), &limits());
        assert_eq!(m.train.len(), 1);
        assert_eq!(m.skipped_lines, 5);
    }

    #[test]
    fn an_oversized_line_is_skipped_and_the_rest_still_parse() {
        let limits = Limits {
            max_line_bytes: 200,
            ..Limits::default()
        };
        let huge = json!({"kind":"train","phase":"bc","step":1,"pad":"x".repeat(1000)}).to_string();
        let s = format!(
            "{huge}\n{}",
            line(json!({"kind":"train","phase":"bc","step":2,"loss":{"total":1.0}}))
        );
        let m = parse_metrics(s.as_bytes(), &limits);
        assert_eq!(m.skipped_lines, 1);
        assert_eq!(m.train.len(), 1);
        assert_eq!(m.train[0].step, 2);
    }

    #[test]
    fn train_points_are_thinned_but_keep_the_last_and_phase_marks_use_all_of_them() {
        let limits = Limits {
            max_series_points: 10,
            ..Limits::default()
        };
        let mut s = String::new();
        for i in 1..=95u64 {
            s += &line(
                json!({"kind":"train","phase":if i <= 50 {"bc"} else {"dagger-1"},"step":i*50,"loss":{"total":1.0}}),
            );
        }
        let m = parse_metrics(s.as_bytes(), &limits);
        assert!(m.train.len() <= 10 && m.train_thinned);
        assert_eq!(m.train.last().unwrap().step, 95 * 50);
        assert_eq!(m.train[0].step, 50);
        assert!(m.train.windows(2).all(|w| w[0].step < w[1].step));
        assert_eq!(m.phases[1].name, "dagger-1");
        assert_eq!(m.phases[1].start_step, 2500);
        assert_eq!(m.phases[1].end_step, 95 * 50);
    }

    #[test]
    fn record_caps_keep_the_newest() {
        let limits = Limits {
            max_records: 3,
            ..Limits::default()
        };
        let mut s = String::new();
        for i in 1..=10u64 {
            s += &line(json!({"kind":"eval","phase":"bc","set":"x","step":i,"report":{}}));
        }
        let m = parse_metrics(s.as_bytes(), &limits);
        assert_eq!(m.eval.iter().map(|e| e.step).collect::<Vec<_>>(), [8, 9, 10]);
    }

    #[test]
    fn a_tail_that_starts_in_a_later_phase_marks_it_at_its_first_step() {
        let mut s = String::new();
        for step in [4000u64, 4050] {
            s += &line(json!({"kind":"train","phase":"dagger-3","step":step,"loss":{"total":1.0}}));
        }
        s += &line(json!({"kind":"train","phase":"dagger-4","step":5050,"loss":{"total":1.0}}));
        let m = parse_metrics(s.as_bytes(), &limits());
        assert_eq!(m.phases[0].start_step, 4000);
        assert_eq!(m.phases[1].start_step, 4050);
    }

    #[test]
    fn the_number_of_distinct_phases_is_capped() {
        let mut s = String::new();
        for i in 0..500u64 {
            s += &line(json!({"kind":"train","phase":format!("p{i}"),"step":i,"loss":{"total":1.0}}));
        }
        let m = parse_metrics(s.as_bytes(), &limits());
        assert_eq!(m.phases.len(), MAX_PHASES);
        assert_eq!(m.train.len(), 500);
    }

    #[test]
    fn thin_helper_edge_cases() {
        assert_eq!(thin(&[1, 2, 3], 10), [1, 2, 3]);
        assert!(thin::<u8>(&[], 5).is_empty());
        assert!(thin(&[1, 2, 3], 0).is_empty());
        let v: Vec<u32> = (0..101).collect();
        for max in [1usize, 2, 7, 50, 100] {
            let t = thin(&v, max);
            assert!(t.len() <= max, "{max}: {}", t.len());
            assert_eq!(*t.last().unwrap(), 100);
        }
    }

    #[test]
    fn arena_summary_reads_new_and_old_conditions_and_forwards_no_paths() {
        let summary = json!({
        "meta": {"run_name":"E-008 evaluation","git_commit":"a2b67c8f9eca95b5324c4951e076c88cf28a7120","git_dirty":true,"base_seed":1,
            "maps":[{"arena":"clb-left","source":"/home/ubuntu/secret/path.map","sha256":"x"}]},
        "conditions": [
          {"name":"clb-left vs scripted","arena":"clb-left","games":1000,"tally":{"w":502,"l":419,"d":5,"t":74},
           "win_rate":{"p":0.54,"lo":0.51,"hi":0.57},"win_rate_all":{"p":0.502,"lo":0.47,"hi":0.53},
           "credited_win_rate":{"p":0.028,"lo":0.0194,"hi":0.0402},"credited_w":28,"blocks_per_min":0.19,
           "self_freezes_per_min":2.48,"players":[{"name":"someone"}]},
          {"name":"old condition","arena":"pit","games":200,"tally":{"w":150,"l":40,"d":0,"t":10},
           "win_rate":{"p":0.79,"lo":0.7,"hi":0.85},"win_rate_all":{"p":0.75,"lo":0.7,"hi":0.8},"credited_w":50}
        ]});
        let s = parse_arena_summary(summary.to_string().as_bytes(), "arena", 10).unwrap();
        assert_eq!(s.git_commit.as_deref(), Some("a2b67c8f9eca"));
        assert_eq!(s.git_dirty, Some(true));
        assert_eq!(s.conditions.len(), 2);
        assert_eq!(s.conditions[0].w, Some(502));
        assert_eq!(s.conditions[0].credited.unwrap().hi, 0.0402);
        // No `credited_win_rate`: recomputed from credited_w / games.
        let old = s.conditions[1].credited.unwrap();
        assert_eq!(old, wilson(50, 200).unwrap());
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("secret") && !json.contains("someone"));
        let cut = parse_arena_summary(summary.to_string().as_bytes(), "arena", 1).unwrap();
        assert_eq!(cut.conditions.len(), 1);
        assert!(cut.conditions_truncated);
        assert!(parse_arena_summary(b"{}", "arena", 5).is_none());
        assert!(parse_arena_summary(b"nope", "arena", 5).is_none());
    }

    #[test]
    fn latest_arena_picks_the_largest_step_per_arena() {
        let mut m = parse_metrics(
            [
                line(json!({"kind":"arena","phase":"bc","step":100,"eval":{"arena":"a","games":10,"credited_w":1}})),
                line(json!({"kind":"arena","phase":"dagger-1","step":200,"eval":{"arena":"a","games":10,"credited_w":2}})),
                line(json!({"kind":"arena","phase":"dagger-1","step":200,"eval":{"arena":"b","games":10,"credited_w":3}})),
            ]
            .concat()
            .as_bytes(),
            &limits(),
        );
        let latest = latest_arena_by_name(&m.arena);
        assert_eq!(latest["a"].credited_w, Some(2));
        assert_eq!(latest["b"].credited_w, Some(3));
        m.arena.clear();
        assert!(latest_arena_by_name(&m.arena).is_empty());
    }
}
