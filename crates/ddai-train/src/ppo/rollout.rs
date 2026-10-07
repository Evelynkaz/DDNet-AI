//! Playing one PPO episode: the fly (an [`PpoActor`]) in the focal seat of an arena game, from a bank start or from the spawn, and turning
//! the game into an [`Episode`]: the decisions it made, their rewards, and what happened (task 8.5b).
//!
//! A **bank start** replays its source game up to the handover (the logged blocker against the deterministic scripted bot, so the replay
//! reproduces the source tick for tick, checked here), then the fly plays the blocker's seat for the held-block window. The victim after the
//! handover may be another brain than the scripted one ([`HandoverVictim`]: the scripted bot plays until the handover so that the replay
//! matches, then the chosen opponent takes over, after a burn-in of its own). A **normal start** is a full game from the spawn.
//!
//! The episode ends when the window after the deciding freeze is over, when the game ends without a freeze of the opponent, or one
//! decision after the fly is out (a frozen fly does nothing, and its freeze is already the whole story).

use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_env::EnvError;
use ddai_env::arena::Arena;
use ddai_env::config::Rules;
use ddai_env::game::{Layout, play_game_watched};
use ddai_env::observe;
use ddai_env::sim::PlayerSetup;
use ddai_env::stats::GameResult;
use ddai_fly::bundle::FlyBrainTemplate;

use super::actor::{ActMode, Decision, PpoActor, Sink, WindowGrid};
use super::critic::{EpisodeClock, MapFields};
use super::reward::{EpisodeKind, PpoReward, RewardParts, assemble};
use crate::bank::{BankStart, ReplayThenBrain};
use crate::heldblock::{EpisodeOutcome, TickRecord};

/// One played episode.
#[derive(Debug, Clone)]
pub struct Episode {
    pub kind: EpisodeKind,
    pub arena: String,
    /// The bank start's seed or the game's seed, and a label of the opponent that played the victim after the handover.
    pub seed: u64,
    pub opponent: String,
    /// The class of a post-freeze start (`None` for a game).
    pub class: Option<super::StartClass>,
    /// Ticks of a demonstration's play before the fly took over (`0` for a plain start).
    pub offset: i32,
    /// The decisions, ending with the last one of the episode.
    pub decisions: Vec<Decision>,
    /// The reward after each decision (burn-in decisions `0`).
    pub rewards: Vec<f32>,
    pub parts: RewardParts,
    pub clock: EpisodeClock,
    pub outcome: EpisodeOutcome,
}

impl Episode {
    /// Index of the first acted decision.
    pub fn first_acted(&self) -> usize {
        self.decisions
            .iter()
            .position(|d| d.acted)
            .unwrap_or(self.decisions.len())
    }

    pub fn acted_decisions(&self) -> usize {
        self.decisions.len() - self.first_acted()
    }

    pub fn total_reward(&self) -> f32 {
        self.rewards.iter().sum()
    }
}

/// What to play.
#[derive(Debug, Clone)]
pub enum EpisodeSpec<'a> {
    Post(&'a BankStart),
    /// A bank start whose seat after the freeze was played by a demonstration up to `handover` (the reverse curriculum, [`super::curriculum`]).
    Resumed {
        start: &'a BankStart,
        log: Arc<Vec<crate::bank::LoggedAction>>,
        handover: i32,
        offset: i32,
    },
    Game {
        arena: &'a str,
        seed: u64,
        layout: Layout,
    },
}

/// Plays the scripted bot (any brain) until `handover`, then `second` (which decides, unseen, from `burn_ticks` ticks before).
pub struct HandoverVictim {
    first: Box<dyn Brain>,
    second: Box<dyn Brain>,
    handover: i32,
    burn_from: i32,
}

impl HandoverVictim {
    pub fn new(first: Box<dyn Brain>, second: Box<dyn Brain>, handover: i32, burn_ticks: i32) -> Self {
        HandoverVictim {
            first,
            second,
            handover,
            burn_from: handover - burn_ticks.max(0),
        }
    }
}

impl Brain for HandoverVictim {
    fn reset(&mut self, ctx: &ResetContext) {
        self.first.reset(ctx);
        self.second.reset(ctx);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        if obs.tick >= self.handover {
            return self.second.decide_in(obs, view);
        }
        if obs.tick >= self.burn_from {
            let _ = self.second.decide_in(obs, view);
        }
        self.first.decide_in(obs, view)
    }

    fn name(&self) -> &str {
        self.second.name()
    }
}

/// Makes a brain (the opponent).
pub type BrainMaker<'a> = dyn Fn() -> Result<Box<dyn Brain>, EnvError> + 'a;

/// Everything an episode needs that does not change from one to the next.
pub struct RolloutCtx<'a> {
    pub arenas: &'a std::collections::BTreeMap<String, Arena>,
    pub fields: &'a std::collections::BTreeMap<String, MapFields>,
    pub template: &'a FlyBrainTemplate,
    pub rules: &'a Rules,
    pub window: i32,
    pub burn_in_ticks: i32,
    pub grid: WindowGrid,
    pub aim_kappa: f32,
    pub temps: ddai_fly::policy::Temperatures,
    pub gamma: f32,
    pub reward: &'a PpoReward,
    pub mode: ActMode,
}

/// Plays one episode. `scripted` builds the scripted bot (the victim before any handover), `opponent` the brain that plays the
/// victim after the handover of a bank start / the opponent of a game (`None` = the scripted bot).
pub fn play_episode_ppo(
    ctx: &RolloutCtx<'_>,
    spec: &EpisodeSpec<'_>,
    scripted: &dyn Fn() -> Result<Box<dyn Brain>, EnvError>,
    opponent: Option<(&str, &BrainMaker<'_>)>,
) -> Result<Episode, String> {
    let (arena_name, seed, layout, kind) = match spec {
        EpisodeSpec::Post(s) | EpisodeSpec::Resumed { start: s, .. } => {
            (s.arena.as_str(), s.seed, s.layout(), EpisodeKind::Post)
        }
        EpisodeSpec::Game { arena, seed, layout } => (*arena, *seed, *layout, EpisodeKind::Game),
    };
    let arena = ctx
        .arenas
        .get(arena_name)
        .ok_or_else(|| format!("unknown arena {arena_name:?}"))?;
    let fields = ctx
        .fields
        .get(arena_name)
        .ok_or_else(|| format!("no map fields for {arena_name:?}"))?;
    let sink: Sink = Arc::new(Mutex::new(Vec::new()));
    let (acting_from, offset) = match spec {
        EpisodeSpec::Post(s) => (s.end_tick, 0),
        EpisodeSpec::Resumed { handover, offset, .. } => (*handover, *offset),
        EpisodeSpec::Game { .. } => (0, 0),
    };
    let actor = PpoActor::new(
        ctx.template,
        ctx.mode,
        ctx.aim_kappa,
        ctx.temps,
        acting_from,
        ctx.grid,
        sink.clone(),
    );
    let focal: Box<dyn Brain> = match spec {
        EpisodeSpec::Post(s) => Box::new(ReplayThenBrain::new(
            Box::new(actor),
            Arc::new(s.actions.clone()),
            s.end_tick,
            ctx.burn_in_ticks,
        )),
        EpisodeSpec::Resumed { log, handover, .. } => Box::new(ReplayThenBrain::new(
            Box::new(actor),
            log.clone(),
            *handover,
            ctx.burn_in_ticks,
        )),
        EpisodeSpec::Game { .. } => Box::new(actor),
    };
    let opp_label = opponent.as_ref().map_or("scripted", |(l, _)| l).to_string();
    let victim: Box<dyn Brain> = match (spec, &opponent) {
        (EpisodeSpec::Post(_) | EpisodeSpec::Resumed { .. }, Some((_, make))) => Box::new(HandoverVictim::new(
            scripted().map_err(|e| e.to_string())?,
            make().map_err(|e| e.to_string())?,
            acting_from,
            ctx.burn_in_ticks,
        )),
        (EpisodeSpec::Game { .. }, Some((_, make))) => make().map_err(|e| e.to_string())?,
        (_, None) => scripted().map_err(|e| e.to_string())?,
    };
    let players = vec![
        PlayerSetup {
            brain: focal,
            lag: 0,
            label: "focal".into(),
        },
        PlayerSetup {
            brain: victim,
            lag: 0,
            label: "opponent".into(),
        },
    ];
    let rules = Rules {
        after_ticks: ctx.window,
        ..ctx.rules.clone()
    };
    let mut records: Vec<TickRecord> = Vec::with_capacity((rules.max_ticks + ctx.window) as usize);
    let report = play_game_watched(arena, &rules, seed, layout, players, &mut |sim, tick| {
        let w = sim.pw.inner();
        let out = [observe::is_out(w, sim.ids[0]), observe::is_out(w, sim.ids[1])];
        // The fly out for two ticks in a row: its episode is over (the result of the game was decided at the onset).
        if out[0] && records.last().is_some_and(|r| r.out[0]) {
            return false;
        }
        records.push(TickRecord { tick, out, phi: 0.0 });
        true
    })
    .map_err(|e| e.to_string())?;
    if let EpisodeSpec::Post(s) | EpisodeSpec::Resumed { start: s, .. } = spec
        && (report.result != GameResult::W || report.end_tick != s.end_tick || !report.credited)
    {
        return Err(format!(
            "bank start {} seed {} did not replay: {:?} at tick {} (credited {}), the bank says a credited win at tick {}",
            s.arena, s.seed, report.result, report.end_tick, report.credited, s.end_tick
        ));
    }
    let outcome = EpisodeOutcome::from_records(&report, &records, ctx.window);
    // Trim the decisions: nothing after the one whose two ticks hold the onset of the fly's own freeze.
    let mut decisions = std::mem::take(&mut *sink.lock().map_err(|_| "sink poisoned".to_string())?);
    if let Some(onset) = records.iter().find(|r| r.out[0]).map(|r| r.tick) {
        decisions.retain(|d| d.obs.tick < onset);
    }
    // A game decided without a credited freeze of the opponent has no window worth playing: nothing after the deciding tick.
    if kind == EpisodeKind::Game && !(report.result == GameResult::W && report.credited) {
        decisions.retain(|d| d.obs.tick <= report.end_tick);
    }
    let freeze_tick = (report.result == GameResult::W && report.credited).then_some(report.end_tick);
    let clock = EpisodeClock {
        freeze_tick,
        window: ctx.window,
        max_ticks: ctx.rules.max_ticks,
    };
    let (rewards, parts) = assemble(
        &decisions,
        fields,
        &outcome,
        kind,
        ctx.rules.decide_every,
        ctx.gamma,
        ctx.reward,
    );
    Ok(Episode {
        kind,
        arena: arena_name.to_string(),
        seed,
        opponent: opp_label,
        offset,
        class: match spec {
            EpisodeSpec::Post(s) | EpisodeSpec::Resumed { start: s, .. } => Some(super::start_class(s)?),
            EpisodeSpec::Game { .. } => None,
        },
        decisions,
        rewards,
        parts,
        clock,
        outcome,
    })
}
