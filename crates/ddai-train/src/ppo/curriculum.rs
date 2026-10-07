//! A reverse curriculum from planner demonstrations (task 8.5b; Salimans & Chen 2018, "Learning Montezuma's Revenge from a single
//! demonstration"; Florensa et al. 2017, "Reverse curriculum generation").
//!
//! The held block is too rare for the fly to find by exploration on the starts that matter (about 7% on the victim-escapable ones),
//! while the planner holds 84% of them. A **demonstration** is the planner's play from the handover of a bank start to the end of the window,
//! recorded as its logged actions, kept only when it holds the block (the victim never free, the planner never out). Replaying the source
//! game with the demonstration's actions up to the tick `H = freeze tick + offset` reproduces the planner's state at `H` exactly (the
//! physics and the scripted victim are deterministic), and the fly takes over there, after the usual burn-in on the last 50 ticks. At offset
//! `200` the fly has the last 50 ticks of the window to finish (and the planner has done the hard part); each time the fly holds the block
//! at the current offset often enough the offset moves back by `step` ticks, down to `0` where the fly plays the whole window of the plain
//! bank start. Episodes are played and rewarded exactly like the others (same terminal reward on the whole window, same potential-based
//! shaping): only the starting state is easier. A share of every iteration keeps playing the easier offsets so that nothing is forgotten.

use std::sync::Arc;

use ddai_brain::Brain;
use ddai_env::EnvError;
use ddai_env::config::{PlayerSpec, Rules};
use ddai_fly::bundle::{read_zstd_postcard, write_zstd_postcard};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bank::{Bank, BankStart, LoggedAction, ReplayThenBrain};
use crate::experiment::Env;
use crate::heldblock::{EpisodeOutcome, play_episode};

pub const DEMO_FORMAT_VERSION: u32 = 1;

/// How the curriculum is run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurriculumConfig {
    pub enabled: bool,
    /// Where the demonstrations are kept; built on the first run (the planner plays every training start once) when missing.
    pub demos: String,
    /// The first offset of each class's ladder (V, B, H): ticks of the planner's play before the fly takes over, so the fly finishes the last
    /// `window - offset` ticks. The V ladder starts early (an idle blocker holds a V start from offset 8 on, probe E-027), the B ladder only in the
    /// last ticks (an idle blocker survives a B start only from offset 236 on: the hazard unfolds in the last 30 ticks).
    pub start_offsets: [i32; 3],
    /// How far back one level moves the handover, in ticks (an even number: decisions are 2 ticks apart).
    pub step: i32,
    /// The level moves back when the share of held blocks at it reaches this (over at least `min_episodes` episodes).
    pub threshold: f32,
    pub min_episodes: usize,
    /// Share of an iteration's post-freeze episodes that are curriculum episodes.
    pub share: f32,
    /// Share of the curriculum episodes at the current level; the rest are at a uniformly drawn easier level (later handover).
    pub current_share: f32,
    /// Weights of the start classes the curriculum draws its starts from (V, B, H).
    pub mix: [f32; 3],
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        CurriculumConfig {
            enabled: false,
            demos: String::new(),
            start_offsets: [104, 236, 104],
            step: 12,
            threshold: 0.5,
            min_episodes: 24,
            share: 0.5,
            current_share: 0.6,
            mix: [0.6, 0.4, 0.0],
        }
    }
}

impl CurriculumConfig {
    pub fn validate(&self, window: i32) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.demos.is_empty() {
            return Err("curriculum.demos (a path) is needed".into());
        }
        if self.step <= 0
            || self.step % 2 != 0
            || self.start_offsets.iter().any(|o| *o < 0 || *o >= window || o % 2 != 0)
        {
            return Err("curriculum.step / start_offsets must be even, 0 <= start_offset < window, step > 0".into());
        }
        if !(0.0..=1.0).contains(&self.share)
            || !(0.0..=1.0).contains(&self.current_share)
            || !(0.0..=1.0).contains(&self.threshold)
        {
            return Err("curriculum.share / current_share / threshold must be in [0, 1]".into());
        }
        if self.mix.iter().any(|w| !(*w >= 0.0 && w.is_finite())) || self.mix.iter().all(|w| *w == 0.0) {
            return Err("curriculum.mix needs non-negative weights, one positive at least".into());
        }
        Ok(())
    }
}

/// The planner's play of one start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Demo {
    /// The identity of the start it belongs to.
    pub arena: String,
    pub seed: u64,
    pub swap: bool,
    pub reverse_order: bool,
    pub end_tick: i32,
    /// The planner held the block for the whole window (the victim never free, the planner never out, credited).
    pub held: bool,
    /// The planner's decisions from the handover tick on (`tick >= end_tick`).
    pub actions: Vec<LoggedAction>,
}

impl Demo {
    pub fn matches(&self, s: &BankStart) -> bool {
        self.arena == s.arena
            && self.seed == s.seed
            && self.swap == s.swap
            && self.reverse_order == s.reverse_order
            && self.end_tick == s.end_tick
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DemoSet {
    pub version: u32,
    /// A fingerprint of the bank's starts the demos were recorded from (the demos are only valid for the very same recipes).
    pub fingerprint: String,
    pub window: i32,
    pub demos: Vec<Demo>,
}

pub fn bank_fingerprint(starts: &[&BankStart]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for s in starts {
        h.update(format!("{}|{}|{}|{}|{};", s.arena, s.seed, s.swap, s.reverse_order, s.end_tick).as_bytes());
    }
    ddai_env::arena::hex(&h.finalize())[..16].to_string()
}

impl DemoSet {
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        write_zstd_postcard(path, self, 3).map_err(|e| e.to_string())
    }

    pub fn load(path: &std::path::Path) -> Result<DemoSet, String> {
        let d: DemoSet = read_zstd_postcard(path).map_err(|e| e.to_string())?;
        if d.version != DEMO_FORMAT_VERSION {
            return Err(format!(
                "{}: demo format version {} (expected {DEMO_FORMAT_VERSION})",
                path.display(),
                d.version
            ));
        }
        Ok(d)
    }

    /// The demo of `start`, when the planner held it.
    pub fn held_demo(&self, start: &BankStart) -> Option<&Demo> {
        self.demos.iter().find(|d| d.held && d.matches(start))
    }
}

/// The planner plays the seat after the freeze of `start` once; its decisions are kept with the outcome.
pub fn record_demo(
    env: &Env,
    rules: &Rules,
    start: &BankStart,
    window: i32,
) -> Result<(Demo, EpisodeOutcome), EnvError> {
    let factory = env.models.factory();
    let arena = env
        .arenas
        .get(&start.arena)
        .ok_or_else(|| EnvError::new(format!("unknown arena {:?}", start.arena)))?;
    let planner = factory(&PlayerSpec::simple("planner"))?;
    let (rec, log) = ddai_env::brains::RecordingBrain::new(planner);
    let replay = ReplayThenBrain::new(Box::new(rec), Arc::new(start.actions.clone()), start.end_tick, 0);
    let o = play_episode(
        arena,
        rules,
        start.seed,
        start.layout(),
        Box::new(replay),
        factory(&PlayerSpec::simple("scripted"))?,
        window,
    )?;
    if o.result != ddai_env::stats::GameResult::W || o.end_tick != start.end_tick || !o.credited {
        return Err(EnvError::new(format!(
            "bank start {} seed {} did not replay under the planner",
            start.arena, start.seed
        )));
    }
    let actions: Vec<LoggedAction> = log
        .lock()
        .map_err(|_| EnvError::new("demo log poisoned"))?
        .iter()
        .map(|(t, a)| LoggedAction::new(*t, a))
        .collect();
    Ok((
        Demo {
            arena: start.arena.clone(),
            seed: start.seed,
            swap: start.swap,
            reverse_order: start.reverse_order,
            end_tick: start.end_tick,
            held: o.held_block,
            actions,
        },
        o,
    ))
}

/// Records a demo for every start of `starts` (in that order, whatever the thread count).
pub fn build_demos(
    env: &Env,
    bank: &Bank,
    starts: &[&BankStart],
    pool: &rayon::ThreadPool,
    log: &mut dyn FnMut(&str),
) -> Result<DemoSet, String> {
    let results: Vec<Result<Demo, EnvError>> = pool.install(|| {
        starts
            .par_iter()
            .map(|s| record_demo(env, &bank.rules, s, bank.window_ticks).map(|(d, _)| d))
            .collect()
    });
    let demos: Vec<Demo> = results
        .into_iter()
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    log(&format!(
        "demos: the planner holds {} of {} training starts",
        demos.iter().filter(|d| d.held).count(),
        demos.len()
    ));
    Ok(DemoSet {
        version: DEMO_FORMAT_VERSION,
        fingerprint: bank_fingerprint(starts),
        window: bank.window_ticks,
        demos,
    })
}

/// The actions to replay for a curriculum episode of `start` handed over to the fly at `end_tick + offset`: the source game's decisions of
/// the blocker before the freeze, then the demonstration's from the freeze up to the handover (exclusive). Returns the log and the handover tick.
pub fn resumed_log(start: &BankStart, demo: &Demo, offset: i32) -> (Arc<Vec<LoggedAction>>, i32) {
    let handover = start.end_tick + offset;
    let mut log: Vec<LoggedAction> = start
        .actions
        .iter()
        .filter(|a| a.tick < start.end_tick)
        .copied()
        .collect();
    log.extend(
        demo.actions
            .iter()
            .filter(|a| a.tick >= start.end_tick && a.tick < handover)
            .copied(),
    );
    (Arc::new(log), handover)
}

/// The curriculum's position, saved in the state of a run: one ladder per start class (V, B, H).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurriculumState {
    /// The current offset of each class (ticks of the planner's play before the fly takes over); `0` = the plain start.
    pub offsets: [i32; 3],
    /// `(iteration, class, offset)` every time a level moved.
    pub moves: Vec<(u64, usize, i32)>,
    /// Outcomes (held or not) of the episodes played at the current level of each class since it was entered, newest last (at most
    /// `3 * min_episodes`).
    pub pending: [Vec<bool>; 3],
}

impl CurriculumState {
    pub fn new(cfg: &CurriculumConfig) -> CurriculumState {
        CurriculumState {
            offsets: cfg.start_offsets,
            moves: Vec::new(),
            pending: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// The levels a curriculum episode of `class` may be played at: the current one and the easier ones up to the first.
    pub fn levels(&self, cfg: &CurriculumConfig, class: usize) -> Vec<i32> {
        let mut v = vec![self.offsets[class]];
        let mut o = self.offsets[class];
        while o + cfg.step <= cfg.start_offsets[class] {
            o += cfg.step;
            v.push(o);
        }
        v
    }

    /// Adds the episodes played at the current level (`held` per episode) to the record of the level and moves it back by `step` when the
    /// record has at least `min_episodes` of them and the share of held blocks reaches the threshold. Returns whether it moved.
    pub fn advance(&mut self, cfg: &CurriculumConfig, iteration: u64, class: usize, at_current: &[bool]) -> bool {
        if self.offsets[class] == 0 {
            return false;
        }
        let pending = &mut self.pending[class];
        pending.extend_from_slice(at_current);
        let cap = 3 * cfg.min_episodes.max(1);
        if pending.len() > cap {
            let drop = pending.len() - cap;
            pending.drain(..drop);
        }
        if pending.len() < cfg.min_episodes {
            return false;
        }
        let rate = pending.iter().filter(|&&h| h).count() as f32 / pending.len() as f32;
        if rate >= cfg.threshold {
            self.offsets[class] = (self.offsets[class] - cfg.step).max(0);
            self.moves.push((iteration, class, self.offsets[class]));
            self.pending[class].clear();
            return true;
        }
        false
    }
}

/// Plays `start` with the demonstration handed over at `offset` to `focal` (the fly, or a stand-in) through the whole window. `burn_in_ticks`
/// is the warm-up of the fly's recurrent state before the hand-over (it decides on what it sees, the logged action is played), **the same as
/// in training and evaluation** (50): a cold fly plays much worse (the E-027 ladder was first measured without it, review 8.5b F1).
#[allow(clippy::too_many_arguments)]
pub fn play_resumed(
    env: &Env,
    rules: &Rules,
    start: &BankStart,
    demo: &Demo,
    offset: i32,
    focal: Box<dyn Brain>,
    window: i32,
    burn_in_ticks: i32,
) -> Result<EpisodeOutcome, EnvError> {
    let factory = env.models.factory();
    let (log, handover) = resumed_log(start, demo, offset);
    let replay = ReplayThenBrain::new(focal, log, handover, burn_in_ticks);
    let arena = &env.arenas[&start.arena];
    play_episode(
        arena,
        rules,
        start.seed,
        start.layout(),
        Box::new(replay),
        factory(&PlayerSpec::simple("scripted"))?,
        window,
    )
}

/// What a brain does after the hand-over at one offset, over the starts that have a held demonstration, per class (V, B, H):
/// `(episodes, held blocks, the focal player out in the window)`.
pub type OffsetRow = (i32, [(usize, usize, usize); 3]);

/// The held share of `maker`'s brain at every offset of `offsets` (the curriculum's ladder, read from outside: how far down it can go), with
/// the run's `burn_in_ticks` of warm-up before every hand-over.
#[allow(clippy::too_many_arguments)]
pub fn probe_offsets(
    env: &Env,
    bank: &Bank,
    starts: &[&BankStart],
    demos: &DemoSet,
    offsets: &[i32],
    maker: &(dyn Fn() -> Result<Box<dyn Brain>, EnvError> + Sync),
    burn_in_ticks: i32,
    pool: &rayon::ThreadPool,
) -> Result<Vec<OffsetRow>, String> {
    let with_demo: Vec<(&BankStart, &Demo)> = starts
        .iter()
        .filter_map(|s| demos.held_demo(s).map(|d| (*s, d)))
        .collect();
    let mut rows = Vec::new();
    for &offset in offsets {
        let outcomes: Vec<Result<(usize, EpisodeOutcome), EnvError>> = pool.install(|| {
            with_demo
                .par_iter()
                .map(|(s, d)| {
                    let class = super::start_class(s).map_err(EnvError::new)?.index();
                    play_resumed(
                        env,
                        &bank.rules,
                        s,
                        d,
                        offset,
                        maker()?,
                        bank.window_ticks,
                        burn_in_ticks,
                    )
                    .map(|o| (class, o))
                })
                .collect()
        });
        let mut by = [(0usize, 0usize, 0usize); 3];
        for r in outcomes {
            let (c, o) = r.map_err(|e| e.to_string())?;
            by[c].0 += 1;
            by[c].1 += usize::from(o.held_block);
            by[c].2 += usize::from(o.focal_out_in_window);
        }
        rows.push((offset, by));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CurriculumConfig {
        CurriculumConfig {
            enabled: true,
            demos: "d".into(),
            start_offsets: [200, 200, 200],
            step: 50,
            threshold: 0.5,
            min_episodes: 4,
            ..CurriculumConfig::default()
        }
    }

    #[test]
    fn the_level_moves_back_only_when_the_fly_holds_often_enough_and_stops_at_zero() {
        let c = cfg();
        let mut s = CurriculumState::new(&c);
        assert_eq!(s.levels(&c, 0), vec![200]);
        // Too few episodes (the record accumulates across iterations), then too few successes: it stays.
        assert!(!s.advance(&c, 1, 0, &[true, true, true]));
        assert!(!s.advance(
            &c,
            2,
            0,
            &[false, false, false, false, false, false, false, false, false]
        ));
        assert_eq!(s.offsets[0], 200);
        // Enough: back by one step, the record starts again, and the easier levels join the draw.
        assert!(s.advance(&c, 3, 0, &[true; 12]));
        assert!(s.pending[0].is_empty());
        assert_eq!(s.offsets[0], 150);
        assert_eq!(s.levels(&c, 0), vec![150, 200]);
        for it in 4..8 {
            s.advance(&c, it, 0, &[true; 4]);
        }
        assert_eq!(s.offsets[0], 0);
        assert!(!s.advance(&c, 9, 0, &[true; 4]), "nothing to move at the plain start");
        assert_eq!(s.levels(&c, 0), vec![0, 50, 100, 150, 200]);
        assert_eq!(s.moves.len(), 4);
    }

    /// What the planner does in its first decisions after the freeze, by start class (the diagnosis of E-027: what the V skill is). Prints a table.
    /// `cargo test -p ddai-train --release --lib openings -- --ignored --nocapture` (needs `demos-train-v2.bin` of E-027).
    #[test]
    #[ignore = "needs the recorded demonstrations"]
    fn openings_of_the_planner_demonstrations() {
        let path = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join("aiddnet/data/runs/E-023/demos-train-v2.bin");
        let set = DemoSet::load(&path).unwrap();
        let bank = Bank::load(
            &std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                .join("aiddnet/data/runs/E-022/banks/bank-v2.bank"),
        )
        .unwrap();
        let starts = bank.select(
            &["clb-left".into(), "pit".into(), "platform".into()],
            Some(false),
            false,
        );
        for (name, class) in [("V", 0usize), ("B", 1), ("H", 2)] {
            for (label, from, to) in [
                ("first 4 decisions", 0usize, 4usize),
                ("decisions 4-12", 4, 12),
                ("decisions 12-60", 12, 60),
                ("decisions 60-125", 60, 125),
            ] {
                let (mut n, mut hook, mut fire, mut jump, mut left, mut right) =
                    (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
                for s in &starts {
                    let Some(d) = set.held_demo(s) else { continue };
                    if super::super::start_class(s).unwrap().index() != class {
                        continue;
                    }
                    for a in d.actions.iter().skip(from).take(to - from) {
                        n += 1;
                        hook += usize::from(a.hook);
                        fire += usize::from(a.fire);
                        jump += usize::from(a.jump);
                        left += usize::from(a.direction < 0);
                        right += usize::from(a.direction > 0);
                    }
                }
                let f = |k: usize| 100.0 * k as f64 / n.max(1) as f64;
                println!(
                    "{name} {label:18}: hook {:4.1}%  fire {:4.1}%  jump {:4.1}%  left {:4.1}%  right {:4.1}%  ({n} decisions)",
                    f(hook),
                    f(fire),
                    f(jump),
                    f(left),
                    f(right)
                );
            }
        }
    }

    #[test]
    fn a_config_with_an_odd_step_or_no_demo_path_is_refused() {
        assert!(cfg().validate(250).is_ok());
        assert!(CurriculumConfig { step: 25, ..cfg() }.validate(250).is_err());
        assert!(
            CurriculumConfig {
                start_offsets: [250, 100, 100],
                ..cfg()
            }
            .validate(250)
            .is_err()
        );
        assert!(
            CurriculumConfig {
                demos: String::new(),
                ..cfg()
            }
            .validate(250)
            .is_err()
        );
        assert!(
            CurriculumConfig::default().validate(250).is_ok(),
            "disabled needs nothing"
        );
    }

    #[test]
    fn the_replay_log_is_the_source_before_the_freeze_and_the_demo_up_to_the_handover() {
        let act = |tick| LoggedAction {
            tick,
            direction: 1,
            jump: false,
            hook: false,
            fire: false,
            target: [0, -1],
            weapon: -1,
        };
        let start = BankStart {
            arena: "pit".into(),
            split: "train".into(),
            seed: 1,
            swap: false,
            reverse_order: false,
            end_tick: 100,
            blocker: "x".into(),
            actions: (0..=100).step_by(2).map(act).collect(),
            idle_held: false,
            scripted_held: false,
            victim_escapes_under_idle: Some(true),
            idle_blocker_out: Some(false),
            phi: 0.0,
        };
        let demo = Demo {
            arena: "pit".into(),
            seed: 1,
            swap: false,
            reverse_order: false,
            end_tick: 100,
            held: true,
            actions: (100..350)
                .step_by(2)
                .map(|t| LoggedAction { jump: true, ..act(t) })
                .collect(),
        };
        assert!(demo.matches(&start));
        let (log, h) = resumed_log(&start, &demo, 200);
        assert_eq!(h, 300);
        assert!(log.iter().all(|a| a.tick < 300));
        assert!(
            log.iter().filter(|a| a.tick < 100).all(|a| !a.jump),
            "the source's own decisions before the freeze"
        );
        assert!(
            log.iter().filter(|a| a.tick >= 100).all(|a| a.jump),
            "the demonstration from the freeze on"
        );
        let ticks: Vec<i32> = log.iter().map(|a| a.tick).collect();
        assert!(
            ticks.windows(2).all(|w| w[0] < w[1]),
            "sorted, no tick twice (ReplayThenBrain looks them up by binary search)"
        );
    }
}
