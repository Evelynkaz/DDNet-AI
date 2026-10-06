//! The post-freeze start bank (task 8.5a): states in which the focal player has just frozen the opponent, to start training and
//! evaluation episodes from (the arena used to end every game right there, so the fly never saw what comes next).
//!
//! An entry is **not** a world snapshot: it is the recipe of a game that ended in a credited freeze of the opponent by slot 0,
//! (arena, seed, layout) plus the logged decisions of the blocker (slot 0) up to the freeze. The opponent is the scripted bot, which
//! is deterministic given its seed and what it sees, so replaying the logged blocker against it reproduces the game tick for tick
//! (tested: the replay's decision hashes equal the original game's). At the freeze tick a [`ReplayThenBrain`] hands the seat to the
//! fly. For the last [`DEFAULT_BURN_IN_TICKS`] ticks before the handover the fly is already deciding (its decisions are discarded and
//! the logged ones played) so that its recurrent state is not "from rest" when it takes over; this is the burn-in of
//! `docs/research/fly-training-methods.md` §3.A.3.
//!
//! The bank is stratified by *escapability*: every entry records whether an idle blocker (does nothing from the freeze on) holds the
//! victim for the whole window. Starts where idle holds are those where the geometry does the work; they are kept, tagged, so that
//! a held share can be reported with and without them, next to the `idle` baseline (D-059).

use std::path::Path;
use std::sync::Arc;

use ddai_brain::{Action, Brain, IVec2, Observation, ResetContext, WorldView};
use ddai_env::EnvError;
use ddai_env::arena::Arena;
use ddai_env::brains::RecordingBrain;
use ddai_env::config::Rules;
use ddai_env::game::{Layout, play_game_watched};
use ddai_env::run::layout_of;
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;
use ddai_fly::bundle::{decode_payload, peek_version, read_zstd_bytes, write_zstd_postcard};
use serde::{Deserialize, Serialize};

use crate::heldblock::{EpisodeOutcome, hold_potential, play_episode};

/// Ticks before the handover in which the fly decides without acting.
pub const DEFAULT_BURN_IN_TICKS: i32 = 50;

/// Version 2 stores the rules as JSON text (a positional postcard `Rules` stopped loading when task 3.10 added a field to it), the
/// split escapability tags, and decodes version 1 files (see [`Bank::load`]).
pub const BANK_FORMAT_VERSION: u32 = 2;

/// One logged decision of the blocker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggedAction {
    pub tick: i32,
    pub direction: i8,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    pub target: [i32; 2],
    /// `-1` = no weapon change.
    pub weapon: i8,
}

impl LoggedAction {
    pub fn new(tick: i32, a: &Action) -> Self {
        LoggedAction {
            tick,
            direction: a.direction as i8,
            jump: a.jump,
            hook: a.hook,
            fire: a.fire,
            target: [a.target.x, a.target.y],
            weapon: a.wanted_weapon.map_or(-1, |w| w as i8),
        }
    }

    pub fn action(&self) -> Action {
        Action {
            direction: i32::from(self.direction),
            jump: self.jump,
            hook: self.hook,
            fire: self.fire,
            target: IVec2::new(self.target[0], self.target[1]),
            wanted_weapon: (self.weapon >= 0).then_some(i32::from(self.weapon)),
        }
    }
}

/// One post-freeze start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BankStart {
    pub arena: String,
    /// `train` or `holdout`, as the arena was tagged when the entry was made.
    pub split: String,
    /// The game's seed and layout (`swap`, `reverse_order`).
    pub seed: u64,
    pub swap: bool,
    pub reverse_order: bool,
    /// The tick the opponent's freeze took effect (`GameReport::end_tick`): the handover tick.
    pub end_tick: i32,
    /// The label of the brain that played the blocker's seat in the source game.
    pub blocker: String,
    /// The blocker's decisions up to and including the tick of the freeze.
    pub actions: Vec<LoggedAction>,
    /// An idle blocker from the handover on holds the block: the victim is out for the whole window **and the blocker is never out**.
    /// This is *not* "the geometry holds the victim": it is also false when the idle blocker falls first (see `idle_blocker_out`).
    pub idle_held: bool,
    /// A scripted blocker from the handover on holds the block (the scripted bot is the weak finisher).
    pub scripted_held: bool,
    /// With an idle blocker, the victim gets free during the window (tracked after the blocker went out too): the starts where the
    /// victim really can escape. `None` in a version 1 file that was not re-tagged (`train es retag`).
    pub victim_escapes_under_idle: Option<bool>,
    /// With an idle blocker, the blocker itself goes out (frozen or dead) during the window.
    pub idle_blocker_out: Option<bool>,
    /// The victim's freeze timer and potential at the handover.
    pub phi: f32,
}

impl BankStart {
    pub fn layout(&self) -> Layout {
        Layout {
            swap: self.swap,
            reverse_order: self.reverse_order,
        }
    }

    /// Entries are partitioned by seed into a training and a validation part (the latter is never sampled for training), a
    /// fixed 1 in 4.
    pub fn is_validation(&self) -> bool {
        let mut z = self.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xBA17_BA17;
        z = (z ^ (z >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (z >> 33).is_multiple_of(4)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bank {
    pub version: u32,
    pub rules: Rules,
    pub window_ticks: i32,
    /// The opponent of the source games (and of every episode started from them): `scripted`.
    pub opponent: String,
    /// What the bank was built from, for the record.
    pub notes: String,
    pub starts: Vec<BankStart>,
}

/// Which starts a selection keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartFilter {
    All,
    /// The legacy "escapable" of the E-022 pilot: an idle blocker does not hold the block (`!idle_held`). Mostly starts where the
    /// idle blocker falls (review of 8.5a, F1); kept so the pilot's selections can be reproduced.
    IdleDoesNotHold,
    /// The victim escapes under an idle blocker (`victim_escapes_under_idle`); needs a bank tagged with it.
    VictimEscapes,
}

/// The on-disk form of version 2: the rules as JSON text (a `Rules` field added later by another task is `#[serde(default)]` in JSON, and
/// positional postcard would not survive it), everything else owned by this module.
#[derive(Serialize, Deserialize)]
struct BankFile {
    version: u32,
    rules_json: String,
    window_ticks: i32,
    opponent: String,
    notes: String,
    starts: Vec<BankStart>,
}

/// Version 1 (the E-022 pilot's bank): the rules positional, the starts without the split tags.
mod v1 {
    use super::{LoggedAction, Serialize};
    use serde::Deserialize;

    #[derive(Deserialize, Serialize)]
    pub struct RulesV1 {
        pub max_ticks: i32,
        pub after_ticks: i32,
        pub decide_every: i32,
        pub credit_ticks: i32,
        pub credit_required: Option<bool>,
        pub crowd_from: Option<usize>,
        pub crowd_spacing: Option<f64>,
    }

    #[derive(Deserialize, Serialize)]
    pub struct StartV1 {
        pub arena: String,
        pub split: String,
        pub seed: u64,
        pub swap: bool,
        pub reverse_order: bool,
        pub end_tick: i32,
        pub blocker: String,
        pub actions: Vec<LoggedAction>,
        pub idle_held: bool,
        pub scripted_held: bool,
        pub phi: f32,
    }

    #[derive(Deserialize, Serialize)]
    pub struct BankV1 {
        #[allow(dead_code)]
        pub version: u32,
        pub rules: RulesV1,
        pub window_ticks: i32,
        pub opponent: String,
        pub notes: String,
        pub starts: Vec<StartV1>,
    }
}

impl Bank {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = BankFile {
            version: BANK_FORMAT_VERSION,
            rules_json: serde_json::to_string(&self.rules).map_err(|e| e.to_string())?,
            window_ticks: self.window_ticks,
            opponent: self.opponent.clone(),
            notes: self.notes.clone(),
            starts: self.starts.clone(),
        };
        write_zstd_postcard(path, &file, 3).map_err(|e| e.to_string())
    }

    /// Loads a version 2 bank, or a version 1 bank (converted: the new tags are `None` until `train es retag`).
    pub fn load(path: &Path) -> Result<Bank, String> {
        let bytes = read_zstd_bytes(path).map_err(|e| e.to_string())?;
        match peek_version(path, &bytes).map_err(|e| e.to_string())? {
            2 => {
                let f: BankFile = decode_payload(path, &bytes, "a v2 bank").map_err(|e| e.to_string())?;
                let rules: Rules = serde_json::from_str(&f.rules_json)
                    .map_err(|e| format!("{}: the bank's rules: {e}", path.display()))?;
                Ok(Bank {
                    version: BANK_FORMAT_VERSION,
                    rules,
                    window_ticks: f.window_ticks,
                    opponent: f.opponent,
                    notes: f.notes,
                    starts: f.starts,
                })
            }
            1 => {
                let b: v1::BankV1 = decode_payload(path, &bytes, "a v1 bank").map_err(|e| e.to_string())?;
                let r = &b.rules;
                // `..Rules::default()` is a no-op here but not on a tree where another task added a field to `Rules` (3.10: `hold_target`).
                #[allow(clippy::needless_update)]
                let rules = Rules {
                    max_ticks: r.max_ticks,
                    after_ticks: r.after_ticks,
                    decide_every: r.decide_every,
                    credit_ticks: r.credit_ticks,
                    credit_required: r.credit_required,
                    crowd_from: r.crowd_from,
                    crowd_spacing: r.crowd_spacing,
                    ..Rules::default()
                };
                Ok(Bank {
                    version: BANK_FORMAT_VERSION,
                    rules,
                    window_ticks: b.window_ticks,
                    opponent: b.opponent,
                    notes: b.notes,
                    starts: b
                        .starts
                        .into_iter()
                        .map(|s| BankStart {
                            arena: s.arena,
                            split: s.split,
                            seed: s.seed,
                            swap: s.swap,
                            reverse_order: s.reverse_order,
                            end_tick: s.end_tick,
                            blocker: s.blocker,
                            actions: s.actions,
                            idle_held: s.idle_held,
                            scripted_held: s.scripted_held,
                            victim_escapes_under_idle: None,
                            idle_blocker_out: None,
                            phi: s.phi,
                        })
                        .collect(),
                })
            }
            v => Err(format!(
                "{}: bank format version {v} (this build reads 1 and {BANK_FORMAT_VERSION})",
                path.display()
            )),
        }
    }

    /// The starts on the given arenas, optionally only the validation (`Some(true)`) or training (`Some(false)`) part, optionally only
    /// those where an idle blocker does not hold the block ([`StartFilter::IdleDoesNotHold`], the pilot's "escapable").
    pub fn select(&self, arenas: &[String], validation: Option<bool>, escapable_only: bool) -> Vec<&BankStart> {
        let f = if escapable_only {
            StartFilter::IdleDoesNotHold
        } else {
            StartFilter::All
        };
        self.select_by(arenas, validation, f)
            .expect("these filters need no tags")
    }

    /// [`Bank::select`] with a [`StartFilter`]; `Err` when the filter needs tags the bank does not have.
    pub fn select_by(
        &self,
        arenas: &[String],
        validation: Option<bool>,
        filter: StartFilter,
    ) -> Result<Vec<&BankStart>, String> {
        if filter == StartFilter::VictimEscapes && self.starts.iter().any(|s| s.victim_escapes_under_idle.is_none()) {
            return Err(
                "this bank has no victim-escape tags (a version 1 file): run `ddnet-ai train es retag` first".into(),
            );
        }
        Ok(self
            .starts
            .iter()
            .filter(|s| arenas.iter().any(|a| a == &s.arena))
            .filter(|s| validation.is_none_or(|v| s.is_validation() == v))
            .filter(|s| match filter {
                StartFilter::All => true,
                StartFilter::IdleDoesNotHold => !s.idle_held,
                StartFilter::VictimEscapes => s.victim_escapes_under_idle == Some(true),
            })
            .collect())
    }
}

/// Plays the logged blocker actions up to the handover, then the real brain. Before the burn-in the inner brain is not asked at
/// all; during it, it decides on what it sees but the logged action is played; from the handover on it plays.
pub struct ReplayThenBrain {
    inner: Box<dyn Brain>,
    log: Arc<Vec<LoggedAction>>,
    handover: i32,
    burn_in_from: i32,
    name: String,
}

impl ReplayThenBrain {
    pub fn new(inner: Box<dyn Brain>, log: Arc<Vec<LoggedAction>>, handover: i32, burn_in_ticks: i32) -> Self {
        let name = inner.name().to_string();
        ReplayThenBrain {
            inner,
            log,
            handover,
            burn_in_from: handover - burn_in_ticks.max(0),
            name,
        }
    }

    fn logged(&self, tick: i32) -> Action {
        self.log
            .binary_search_by_key(&tick, |a| a.tick)
            .map_or_else(|_| Action::neutral(), |i| self.log[i].action())
    }
}

impl Brain for ReplayThenBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        if obs.tick >= self.handover {
            return self.inner.decide_in(obs, view);
        }
        if obs.tick >= self.burn_in_from {
            let _ = self.inner.decide_in(obs, view);
        }
        self.logged(obs.tick)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn telemetry(&self) -> Option<String> {
        self.inner.telemetry()
    }
}

/// Plays the episode that starts at `start` with `focal` taking the blocker's seat at the freeze and `opponent` as the victim's brain.
/// Errors when the replay does not reproduce the source game (a different result or freeze tick: the bank no longer matches the
/// arena, the rules or the code).
pub fn play_from_start(
    arena: &Arena,
    rules: &Rules,
    start: &BankStart,
    focal: Box<dyn Brain>,
    opponent: Box<dyn Brain>,
    window: i32,
    burn_in_ticks: i32,
) -> Result<EpisodeOutcome, EnvError> {
    let replay = ReplayThenBrain::new(focal, Arc::new(start.actions.clone()), start.end_tick, burn_in_ticks);
    let o = play_episode(
        arena,
        rules,
        start.seed,
        start.layout(),
        Box::new(replay),
        opponent,
        window,
    )?;
    if o.result != GameResult::W || o.end_tick != start.end_tick || !o.credited {
        return Err(EnvError::new(format!(
            "bank start {} seed {} did not replay: {:?} at tick {} (credited {}), the bank says a credited win at tick {}",
            start.arena, start.seed, o.result, o.end_tick, o.credited, start.end_tick
        )));
    }
    Ok(o)
}

/// How a bank is made.
#[derive(Debug, Clone)]
pub struct BankBuildSpec {
    pub arenas: Vec<String>,
    /// Blocker brains as command-line arguments (`scripted`, `planner`, `fly:<bundle>`, ...) and the games each plays per arena.
    pub blockers: Vec<(String, u32)>,
    pub base_seed: u64,
    pub window_ticks: i32,
    pub threads: usize,
}

/// One source game: slot 0 plays `blocker` (recorded), slot 1 the scripted bot. `Some(start)` when slot 0 froze slot 1 with credit.
fn source_game(
    arena: &Arena,
    rules: &Rules,
    blocker: Box<dyn Brain>,
    blocker_label: &str,
    seed: u64,
    layout: Layout,
    factory: &ddai_env::config::BrainFactory,
) -> Result<Option<BankStart>, EnvError> {
    let (rec, log) = RecordingBrain::new(blocker);
    let scripted = ddai_env::config::PlayerSpec::simple("scripted");
    let players = vec![
        PlayerSetup {
            brain: Box::new(rec),
            lag: 0,
            label: "blocker".into(),
        },
        PlayerSetup {
            brain: factory(&scripted)?,
            lag: 0,
            label: "opponent".into(),
        },
    ];
    let source_rules = Rules {
        after_ticks: 0,
        ..rules.clone()
    };
    let mut phi = 0.0f32;
    let report = play_game_watched(arena, &source_rules, seed, layout, players, &mut |sim, tick| {
        phi = hold_potential(sim.pw.inner(), sim.ids[1]);
        let _ = tick;
        true
    })?;
    if report.result != GameResult::W || !report.credited || report.victim != 1 {
        return Ok(None);
    }
    let actions: Vec<LoggedAction> = {
        let g: std::sync::MutexGuard<'_, Vec<(i32, Action)>> = log.lock().map_err(|_| EnvError::new("log poisoned"))?;
        g.iter()
            // The game plays one more step after the deciding one, so the blocker also decided at `end_tick` itself (a replay that is never
            // handed over reproduces the source game's decision hash to the last decision).
            .filter(|(t, _)| *t <= report.end_tick)
            .map(|(t, a)| LoggedAction::new(*t, a))
            .collect()
    };
    Ok(Some(BankStart {
        arena: arena.name.clone(),
        split: arena.tag.label().to_string(),
        seed,
        swap: layout.swap,
        reverse_order: layout.reverse_order,
        end_tick: report.end_tick,
        blocker: blocker_label.to_string(),
        actions,
        idle_held: false,
        scripted_held: false,
        victim_escapes_under_idle: None,
        idle_blocker_out: None,
        phi,
    }))
}

/// Tags every start with what an idle and a scripted blocker do from the handover on: `idle_held`, `scripted_held`,
/// `victim_escapes_under_idle` and `idle_blocker_out`.
pub fn tag_starts(
    env: &crate::experiment::Env,
    pool: &rayon::ThreadPool,
    rules: &Rules,
    starts: &mut [BankStart],
    window: i32,
) -> Result<(), String> {
    use rayon::prelude::*;
    let factory = env.models.factory();
    let tags: Vec<Result<(EpisodeOutcome, EpisodeOutcome), EnvError>> = pool.install(|| {
        starts
            .par_iter()
            .map(|s| {
                let arena = env
                    .arenas
                    .get(&s.arena)
                    .ok_or_else(|| EnvError::new(format!("unknown arena {:?}", s.arena)))?;
                let with = |brain: &str| -> Result<EpisodeOutcome, EnvError> {
                    play_from_start(
                        arena,
                        rules,
                        s,
                        factory(&ddai_env::config::PlayerSpec::simple(brain))?,
                        factory(&ddai_env::config::PlayerSpec::simple("scripted"))?,
                        window,
                        0,
                    )
                };
                Ok((with("idle")?, with("scripted")?))
            })
            .collect()
    });
    for (s, t) in starts.iter_mut().zip(tags) {
        let (idle, scripted) = t.map_err(|e| e.to_string())?;
        s.idle_held = idle.held_block;
        s.scripted_held = scripted.held_block;
        s.victim_escapes_under_idle = Some(idle.escape_tick.is_some());
        s.idle_blocker_out = Some(idle.focal_out_in_window);
    }
    Ok(())
}

/// Re-tags a whole bank (a version 1 bank, or one made before a change of what the tags mean).
pub fn retag_bank(env: &crate::experiment::Env, bank: &mut Bank, threads: usize) -> Result<(), String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let rules = bank.rules.clone();
    let window = bank.window_ticks;
    tag_starts(env, &pool, &rules, &mut bank.starts, window)
}

/// Builds a bank: for every `(arena, blocker)` the first `games` games (seeds `base_seed + g`, the arena's 4-way layouts), kept when
/// the blocker froze the scripted opponent with credit, then every kept start is replayed with an idle and with a scripted blocker
/// from the handover on to tag its escapability. Entries come out in a fixed order whatever the thread count.
pub fn build_bank(
    env: &crate::experiment::Env,
    spec: &BankBuildSpec,
    log: &mut dyn FnMut(&str),
) -> Result<Bank, String> {
    use rayon::prelude::*;
    let rules = Rules::default();
    let factory = env.models.factory();
    let mut jobs: Vec<(String, String, u32)> = Vec::new(); // (arena, blocker arg, game)
    for arena in &spec.arenas {
        if !env.arenas.contains_key(arena) {
            return Err(format!("unknown arena {arena:?}"));
        }
        for (arg, games) in &spec.blockers {
            jobs.extend((0..*games).map(|g| (arena.clone(), arg.clone(), g)));
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(spec.threads.max(1))
        .stack_size(32 << 20)
        .build()
        .map_err(|e| e.to_string())?;
    let found: Vec<Result<Option<BankStart>, EnvError>> = pool.install(|| {
        jobs.par_iter()
            .map(|(arena_name, arg, g)| {
                let arena = &env.arenas[arena_name];
                let player = ddai_env::models::player_from_arg(arg);
                let blocker = factory(&player)?;
                let seed = spec.base_seed.wrapping_add(u64::from(*g));
                let label = ddai_env::models::label_of_arg(arg);
                source_game(arena, &rules, blocker, &label, seed, layout_of(arena, *g), &factory)
            })
            .collect()
    });
    let mut starts = Vec::new();
    for (job, r) in jobs.iter().zip(found) {
        if let Some(s) = r.map_err(|e| format!("{} {}: {e}", job.0, job.1))? {
            starts.push(s);
        }
    }
    log(&format!(
        "{} source games -> {} credited freezes",
        jobs.len(),
        starts.len()
    ));
    tag_starts(env, &pool, &rules, &mut starts, spec.window_ticks)?;
    let notes = format!(
        "arenas {:?}, blockers {:?}, base seed {}, window {} ticks",
        spec.arenas, spec.blockers, spec.base_seed, spec.window_ticks
    );
    Ok(Bank {
        version: BANK_FORMAT_VERSION,
        rules,
        window_ticks: spec.window_ticks,
        opponent: "scripted".into(),
        notes,
        starts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_start(seed: u64) -> BankStart {
        BankStart {
            arena: "pit".into(),
            split: "train".into(),
            seed,
            swap: true,
            reverse_order: false,
            end_tick: 120,
            blocker: "scripted".into(),
            actions: vec![
                LoggedAction {
                    tick: 0,
                    direction: -1,
                    jump: true,
                    hook: false,
                    fire: false,
                    target: [10, -20],
                    weapon: -1,
                },
                LoggedAction {
                    tick: 2,
                    direction: 1,
                    jump: false,
                    hook: true,
                    fire: true,
                    target: [-5, 7],
                    weapon: 1,
                },
            ],
            idle_held: true,
            scripted_held: false,
            victim_escapes_under_idle: Some(false),
            idle_blocker_out: Some(true),
            phi: 0.25,
        }
    }

    fn sample_bank() -> Bank {
        Bank {
            version: BANK_FORMAT_VERSION,
            rules: Rules::default(),
            window_ticks: 250,
            opponent: "scripted".into(),
            notes: "test".into(),
            starts: vec![sample_start(77)],
        }
    }

    /// The payload of a version 2 bank of `sample_bank()`, as postcard bytes (hex). Fixed: if the layout of `BankFile`, `BankStart` or
    /// `LoggedAction` changes, this fails, and `BANK_FORMAT_VERSION` must be bumped with a reader for the old layout.
    const V2_HEX: &str = "0285017b226d61785f7469636b73223a313530302c2261667465725f7469636b73223a3135302c226465636964655f6576657279223a322c226372656469745f7469636b73223a35302c226372656469745f7265717569726564223a6e756c6c2c2263726f77645f66726f6d223a6e756c6c2c2263726f77645f73706163696e67223a6e756c6c7df4030873637269707465640474657374010370697405747261696e4d0100f0010873637269707465640200ff0100001427ff0401000101090e010100010001010000803e";
    /// The same bank as the E-022 pilot wrote it (version 1: positional `Rules`, no split tags).
    const V1_HEX: &str = "01b817ac020464000000f4030873637269707465640474657374010370697405747261696e4d0100f0010873637269707465640200ff0100001427ff0401000101090e0101000000803e";

    fn hex_of(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn v2_payload() -> Vec<u8> {
        let b = sample_bank();
        let f = BankFile {
            version: 2,
            // A literal, not `to_string(&Rules::default())`: the fixed bytes must not move when a task adds a field to `Rules`.
            rules_json: r#"{"max_ticks":1500,"after_ticks":150,"decide_every":2,"credit_ticks":50,"credit_required":null,"crowd_from":null,"crowd_spacing":null}"#.into(),
            window_ticks: b.window_ticks,
            opponent: b.opponent,
            notes: b.notes,
            starts: b.starts,
        };
        postcard::to_allocvec(&f).unwrap()
    }

    fn v1_payload() -> Vec<u8> {
        let s = sample_start(77);
        let b = v1::BankV1 {
            version: 1,
            rules: v1::RulesV1 {
                max_ticks: 1500,
                after_ticks: 150,
                decide_every: 2,
                credit_ticks: 50,
                credit_required: None,
                crowd_from: None,
                crowd_spacing: None,
            },
            window_ticks: 250,
            opponent: "scripted".into(),
            notes: "test".into(),
            starts: vec![v1::StartV1 {
                arena: s.arena,
                split: s.split,
                seed: s.seed,
                swap: s.swap,
                reverse_order: s.reverse_order,
                end_tick: s.end_tick,
                blocker: s.blocker,
                actions: s.actions,
                idle_held: s.idle_held,
                scripted_held: s.scripted_held,
                phi: s.phi,
            }],
        };
        postcard::to_allocvec(&b).unwrap()
    }

    fn write_payload(dir: &Path, name: &str, payload: &[u8]) -> std::path::PathBuf {
        // The file layout of `write_zstd_postcard`: sha256 of the payload, then the zstd stream.
        use sha2::{Digest, Sha256};
        let mut out = Sha256::digest(payload).to_vec();
        out.extend(zstd::stream::encode_all(payload, 3).unwrap());
        let p = dir.join(name);
        std::fs::write(&p, out).unwrap();
        p
    }

    #[test]
    fn a_bank_with_a_fixed_byte_layout_loads_and_a_version_1_bank_converts() {
        let dir = tempfile::tempdir().unwrap();
        // Version 2: the bytes are those of today's layout, and they load.
        let p2 = write_payload(dir.path(), "v2.bank", &unhex(V2_HEX));
        let b2 = Bank::load(&p2).unwrap();
        assert_eq!(b2, sample_bank());
        assert_eq!(
            hex_of(&v2_payload()),
            V2_HEX,
            "the v2 layout changed: bump BANK_FORMAT_VERSION"
        );
        // Version 1: the pilot's layout still loads, with the new tags unknown and the rules filled with today's defaults for any
        // field that did not exist then.
        let p1 = write_payload(dir.path(), "v1.bank", &unhex(V1_HEX));
        let b1 = Bank::load(&p1).unwrap();
        let mut expect = sample_bank();
        expect.starts[0].victim_escapes_under_idle = None;
        expect.starts[0].idle_blocker_out = None;
        expect.rules.after_ticks = 150;
        assert_eq!(b1, expect);
        assert_eq!(hex_of(&v1_payload()), V1_HEX);
        // An untagged bank refuses the victim-escape filter.
        assert!(b1.select_by(&["pit".into()], None, StartFilter::VictimEscapes).is_err());
        assert_eq!(
            b1.select_by(&["pit".into()], None, StartFilter::IdleDoesNotHold)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            b2.select_by(&["pit".into()], None, StartFilter::VictimEscapes)
                .unwrap()
                .len(),
            0
        );
        // A save and a load keep everything, and a version from the future is refused by name.
        let p = dir.path().join("rt.bank");
        b2.save(&p).unwrap();
        assert_eq!(Bank::load(&p).unwrap(), b2);
        let p9 = write_payload(dir.path(), "v9.bank", &postcard::to_allocvec(&9u32).unwrap());
        assert!(Bank::load(&p9).unwrap_err().contains("version 9"));
    }

    /// The E-022 pilot's banks (version 1 as written, and re-tagged version 2): they must load on any tree, in particular one where task 3.10
    /// added `Rules::hold_target` (the case that made the positional version 1 unreadable). Needs the local data.
    #[test]
    #[ignore = "needs ~/aiddnet/data/runs/E-022/banks"]
    fn the_real_pilot_banks_load() {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap();
        let dir = home.join("aiddnet/data/runs/E-022/banks");
        let v1 = Bank::load(&dir.join("bank-v1.bank")).unwrap();
        let v2 = Bank::load(&dir.join("bank-v2.bank")).unwrap();
        assert_eq!((v1.starts.len(), v2.starts.len()), (1519, 1519));
        assert!(v1.starts.iter().all(|s| s.victim_escapes_under_idle.is_none()));
        assert!(
            v2.starts
                .iter()
                .all(|s| s.victim_escapes_under_idle.is_some() && s.idle_blocker_out.is_some())
        );
        assert_eq!(v1.rules, v2.rules);
        assert_eq!(v2.rules.after_ticks, 150);
        for (a, b) in v1.starts.iter().zip(&v2.starts) {
            assert_eq!(
                (&a.arena, a.seed, a.end_tick, a.idle_held),
                (&b.arena, b.seed, b.end_tick, b.idle_held)
            );
        }
    }

    #[test]
    fn rules_in_a_bank_survive_a_field_added_later() {
        // The rules are JSON: a file written before a field existed still parses (the field takes its default), which is what
        // positional postcard could not do when task 3.10 added `hold_target`.
        let old = r#"{"max_ticks":1500,"after_ticks":250,"decide_every":2,"credit_ticks":50}"#;
        let r: Rules = serde_json::from_str(old).unwrap();
        assert_eq!(r.after_ticks, 250);
    }
}
