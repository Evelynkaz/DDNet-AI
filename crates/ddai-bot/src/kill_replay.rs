//! Offline replay of the self-kills in recorded clips (task 4.12, D-108): what would the smart policy have done instead?
//!
//! `ddnet-ai clip selfkill <clip>...` feeds a clip's recorded frames, one by one, to the bot's own pipeline pieces (the `LiveWorld`, the
//! tee set, the activity clock, [`crate::unstick::Unstick`]) twice: once under the legacy timers and once under the smart policy
//! ([`crate::smartkill`]), and compares them at every `Cl_Kill` the clip recorded ([`ClipEvent::KillSent`]):
//!
//! * [`Outcome::Same`]: the smart policy kills at the same frame (within 2 frames), [`Outcome::Earlier`]: sooner,
//! * [`Outcome::Skipped`]: the smart policy would not have killed. The **counterfactual** then runs on: our tee is carried on
//!   the real physics from the kill frame with no input (a frozen tee cannot act; the other tees come from the recording, so a hook or a
//!   push that was recorded still acts) and the smart policy keeps judging every frame. [`Fate`] says what became of the tee:
//!   thawed by itself, died, was killed by the smart policy later, or was still frozen when the clip ended.
//!
//! **Limits.** The clip has no relation lists (nobody is a friend), no navigator (the dead zone is rebuilt from the map, no routes; the wayblock's own request is
//! taken from the recording), only the nearest tees, and ends 30 s after it began. Kills for a route's respawn step or a
//! trek ([`KillWhy::Navigation`], [`KillWhy::Trek`]) and the owner's console `!kill` are listed, not judged: the smart policy does not
//! change them (only their *planning*: a respawn step only without a foot route). The legacy timers are replayed from the frozen run's
//! start (or 500 ticks before a kill of a free tee) and the replay checks that they fire where the recording says they did.

use std::sync::Arc;

use ddai_clip::format::{Clip, ClipEvent, Frame, KillWhy};
use ddai_net::generated::objects;
use ddai_net::tuning::{DEFAULT_TUNE_PARAMS, NUM_TUNE_PARAMS, TeamsState, TuneParams, from_array};
use ddai_net::view::{CharacterView, PlayerView};
use ddai_physics::core::PlayerInput;
use ddai_physics::map::MapData;
use ddai_planner::plan_world::PlanWorld;
use ddai_world::{LiveWorld, OwnState, SnapshotInput, player_input_from_net};

use crate::activity::ActivityClock;
use crate::duel::{ChatSignal, DuelChange, DuelDetector, DuelWhy};
use crate::mapgrid::MapGrid;
use crate::planning::PlanScratch;
use crate::players::PlayerTable;
use crate::relations::Relations;
use crate::smartkill::{self, SelfKillPolicy, SmartWhy};
use crate::tees::TeeSet;
use crate::unstick::{KillReason, Unstick, UnstickCtx, Verdict};

/// How far before a kill of a free tee the replay starts (the wedge window is 200 ticks, the stuck one 450).
const LEAD_TICKS: i32 = 500;
/// The counterfactual runs this long after a skipped kill at most.
pub const COUNTERFACTUAL_TICKS: i32 = 800;
/// Frames closer than this to the legacy kill count as "the same moment".
const SAME_FRAMES: usize = 2;

/// What the smart policy would have done at a recorded kill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Not judged: a route's or trek's respawn step, the owner's `!kill` (see the module docs).
    NotJudged(&'static str),
    /// The replayed legacy timers did not fire there (the clip is too short to hold the whole frozen run, or the recorded kill came
    /// from state the clip does not hold): nothing can be said.
    LegacyNotReproduced,
    Same {
        why: SmartWhy,
    },
    Earlier {
        ticks: i32,
        why: SmartWhy,
    },
    Skipped {
        why: Option<SmartWhy>,
        fate: Fate,
    },
}

/// What became of a tee whose kill the smart policy skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    /// Free again by itself `after` ticks after the legacy kill.
    Thawed { after: i32 },
    /// Died (a kill tile) `after` ticks after it.
    Died { after: i32 },
    /// The smart policy killed it later, `after` ticks after the legacy kill.
    KilledLater { after: i32, why: SmartWhy },
    /// Still frozen when the clip ended, `ticks` after the legacy kill: the smart policy kept waiting and the upper bounds (frozen
    /// 400 ticks, a friend 1500) had not been reached.
    StillFrozen { ticks: i32 },
    /// The clip ended before anything was decided (`ticks` after the kill, the tee free).
    ClipEnded { ticks: i32 },
}

/// One recorded kill and what the smart policy says about it.
#[derive(Debug, Clone)]
pub struct Case {
    pub clip: String,
    pub frame: usize,
    pub tick: i32,
    pub why: u8,
    /// Ticks the tee had been frozen when the kill went out (0: free).
    pub frozen_for: i32,
    pub deep: bool,
    pub verdict: Outcome,
    /// The smart policy's call frame by frame from the start of the replayed run to the kill (`--trace`).
    pub trace: Vec<String>,
}

fn kill_why_name(why: u8) -> &'static str {
    match why {
        w if w == KillWhy::Unstick as u8 => "unstick",
        w if w == KillWhy::WayBlockLying as u8 => "wayblock",
        w if w == KillWhy::Navigation as u8 => "navigation",
        w if w == KillWhy::Trek as u8 => "trek",
        w if w == KillWhy::Console as u8 => "console",
        _ => "?",
    }
}

impl Case {
    pub fn why_name(&self) -> &'static str {
        kill_why_name(self.why)
    }
}

fn tuning_at(clip: &Clip, tick: i32) -> TuneParams {
    let mut t = DEFAULT_TUNE_PARAMS;
    for c in &clip.header.tuning {
        if c.from_tick <= tick && c.values.len() == NUM_TUNE_PARAMS {
            let mut a = [0i32; NUM_TUNE_PARAMS];
            a.copy_from_slice(&c.values);
            t = from_array(c.received as usize, a);
        }
    }
    t
}

/// The teams message in force at `tick`.
pub fn teams_at(clip: &Clip, tick: i32) -> Option<TeamsState> {
    let mut out = None;
    for c in &clip.header.teams {
        if c.from_tick <= tick {
            let mut teams = [0i32; 128];
            for (i, t) in c.teams.iter().take(128).enumerate() {
                teams[i] = *t;
            }
            out = Some(TeamsState {
                teams,
                received: (c.received as usize).min(128),
            });
        }
    }
    out
}

fn views(f: &Frame) -> Vec<CharacterView> {
    f.tees
        .iter()
        .map(|t| CharacterView {
            id: t.id,
            character: t.ch.to_net(),
            ddnet: t.dd.map(|d| d.to_net()),
        })
        .collect()
}

fn players_of(clip: &Clip) -> Vec<PlayerView> {
    clip.header
        .players
        .iter()
        .map(|p| PlayerView {
            id: p.id,
            info: objects::PlayerInfo {
                local: i32::from(p.id == clip.header.own_id),
                client_id: p.id,
                team: 0,
                score: 0,
                latency: 20,
            },
            client_info: Some(objects::ClientInfo {
                name: p.tag.clone(),
                clan: String::new(),
                country: -1,
                skin: "default".to_string(),
                use_custom_color: 0,
                color_body: 0,
                color_feet: 0,
            }),
            ddnet: Some(objects::DDNetPlayer {
                flags: 0,
                auth_level: 0,
                finish_time_seconds: 0,
                finish_time_millis: 0,
            }),
        })
        .collect()
}

/// The replay's machinery: a world, the tee set, the clock, the players and one legacy and one smart [`Unstick`].
struct Rig<'a> {
    clip: &'a Clip,
    lw: LiveWorld,
    grid: MapGrid,
    plan: PlanScratch,
    spawns: Vec<(f64, f64)>,
    /// The navigator's dead zone (`Navigator::in_dead_zone`): tiles with no way back to the game.
    dead: Option<Vec<u8>>,
    width: i32,
    players: PlayerTable,
    tees: TeeSet,
    clock: ActivityClock,
    legacy: Unstick,
    smart: Unstick,
}

impl<'a> Rig<'a> {
    fn new(clip: &'a Clip, map: &Arc<MapData>) -> Rig<'a> {
        let mut smart = Unstick::new();
        smart.set_policy(SelfKillPolicy::Smart);
        let mut players = PlayerTable::new([3; 16]);
        players.update(&players_of(clip), &Relations::new());
        let spawns = ddai_nav::route::spawn_tiles(map);
        let world = ddai_planner::physics_adapter::PhysicsWorld::new(Arc::clone(map), 1);
        let router = ddai_nav::route::Router::new(world.collision(), &spawns);
        let dead = (!spawns.is_empty()).then(|| ddai_nav::route::dead_zone(&router.grid, &spawns));
        Rig {
            clip,
            lw: LiveWorld::new(Arc::clone(map), clip.header.own_id, clip.header.world_seed),
            grid: MapGrid::new(map),
            plan: PlanScratch::new(Arc::clone(map)),
            spawns,
            dead,
            width: map.width as i32,
            players,
            tees: TeeSet::new(),
            clock: ActivityClock::new(),
            legacy: Unstick::new(),
            smart,
        }
    }

    fn feed(&mut self, f: &Frame) -> Vec<CharacterView> {
        let characters = views(f);
        let projectiles: Vec<_> = f.projectiles.iter().map(|p| p.to_view()).collect();
        let switches: Vec<(i32, objects::SwitchState)> = f.switches.iter().map(|s| s.to_net()).collect();
        let teams = teams_at(self.clip, f.tick);
        self.lw.on_snapshot(SnapshotInput {
            tick: f.tick,
            characters: &characters,
            tuning: tuning_at(self.clip, f.tick),
            switch_states: &switches,
            teams: teams.as_ref(),
            own_input_at_tick: f.sent.last().map(|s| player_input_from_net(s.input.to_net())),
            projectiles: &projectiles,
        });
        characters
    }

    /// One frame through the clock and both state machines. `wb_kill_recorded`: the recorded kill was the wayblock's own request and
    /// this is its frame (the clip does not hold the navigator's zones, so the rule is not rebuilt: it is taken from the recording).
    /// Returns the two verdicts.
    fn step(
        &mut self,
        f: &Frame,
        characters: &[CharacterView],
        wb_kill_recorded: bool,
    ) -> Option<(Verdict, Verdict, i32, bool)> {
        let own_id = self.clip.header.own_id;
        self.tees.rebuild(self.lw.base_world(), characters);
        self.clock.update(f.tick, &self.tees, &self.players, own_id);
        let own = *self.tees.get(own_id)?;
        let target = f.bot.target;
        let cost = smartkill::kill_cost_ticks(&self.spawns, (own.pos.x, own.pos.y));
        let forecast = self.smart.wants_forecast(f.tick, &own, wb_kill_recorded).then(|| {
            self.plan
                .own_forecast(self.lw.base_world(), own_id, smartkill::FORECAST_HORIZON_TICKS)
        });
        let holding = self.clock.holding_block();
        let in_dead = self.dead.as_ref().is_some_and(|d| {
            let i = i64::from((own.pos.y / 32.0).trunc() as i32) * i64::from(self.width)
                + i64::from((own.pos.x / 32.0).trunc() as i32);
            i >= 0 && (i as usize) < d.len() && d[i as usize] == 1
        });
        let mk = |forecast| UnstickCtx {
            tick: f.tick,
            own: &own,
            tees: &self.tees,
            players: &self.players,
            grid: &self.grid,
            target,
            acting: true,
            in_dead_zone: in_dead,
            wayblock_wants_kill: wb_kill_recorded,
            forecast,
            cost_ticks: cost,
            holding_block: holding,
        };
        let vl = self.legacy.step(&mk(None));
        let vs = self.smart.step(&mk(forecast));
        Some((vl, vs, self.smart.frozen_for(f.tick, &own), own.deep_frozen))
    }
}

/// How long the own tee had been frozen at frame `k` (consecutive frames with the tee present and frozen), and the first such frame.
fn frozen_run(frames: &[Frame], own_id: i32, k: usize) -> (usize, i32) {
    let mut s = k;
    while s > 0
        && frames[s - 1].own_alive
        && frames[s - 1].tee(own_id).is_some_and(|t| t.frozen)
        && frames[k].tee(own_id).is_some_and(|t| t.frozen)
    {
        s -= 1;
    }
    let ticks = if frames[k].tee(own_id).is_some_and(|t| t.frozen) {
        frames[k].tick - frames[s].tick
    } else {
        0
    };
    (s, ticks)
}

/// Every recorded unstick kill of the clip and what the smart policy says about it.
pub fn analyse(clip: &Clip, map: &Arc<MapData>, clip_name: &str) -> Vec<Case> {
    let own_id = clip.header.own_id;
    let mut out = Vec::new();
    for (k, f) in clip.frames.iter().enumerate() {
        for e in &f.events {
            let ClipEvent::KillSent { why } = *e else { continue };
            let (run_start, frozen_ticks) = frozen_run(&clip.frames, own_id, k);
            let deep = f.tee(own_id).is_some_and(|t| t.deep_frozen);
            let mut case = Case {
                clip: clip_name.to_string(),
                frame: k,
                tick: f.tick,
                why,
                frozen_for: frozen_ticks,
                deep,
                verdict: Outcome::LegacyNotReproduced,
                trace: Vec::new(),
            };
            case.verdict = if why == KillWhy::Navigation as u8 {
                Outcome::NotJudged("route respawn step")
            } else if why == KillWhy::Trek as u8 {
                Outcome::NotJudged("trek respawn step")
            } else if why == KillWhy::Console as u8 {
                Outcome::NotJudged("owner's !kill")
            } else {
                judge_case(
                    clip,
                    map,
                    k,
                    run_start,
                    why == KillWhy::WayBlockLying as u8,
                    &mut case.trace,
                )
            };
            out.push(case);
        }
    }
    out
}

fn judge_case(
    clip: &Clip,
    map: &Arc<MapData>,
    k: usize,
    run_start: usize,
    wb_kill: bool,
    trace: &mut Vec<String>,
) -> Outcome {
    let frames = &clip.frames;
    let own_id = clip.header.own_id;
    // Start: the frozen run's first frame, or LEAD_TICKS before the kill of a free tee (never before the clip).
    let free = !frames[k].tee(own_id).is_some_and(|t| t.frozen);
    let mut start = if free { k } else { run_start };
    if free {
        while start > 0 && frames[k].tick - frames[start - 1].tick <= LEAD_TICKS {
            start -= 1;
        }
    }
    // A run that began before the clip did cannot be replayed from its start: the frozen clock would be too young.
    let run_complete = free || run_start > 0;
    let mut rig = Rig::new(clip, map);
    let (mut legacy_at, mut smart_at, mut smart_why): (Option<usize>, Option<usize>, Option<SmartWhy>) =
        (None, None, None);
    let mut last_skip: Option<SmartWhy> = None;
    for (i, f) in frames.iter().enumerate().take(k + 1).skip(start) {
        let chars = rig.feed(f);
        if !f.own_alive {
            rig.legacy.reset_ticks();
            rig.smart.reset_ticks();
            continue;
        }
        let Some((vl, vs, _, _)) = rig.step(f, &chars, wb_kill && i == k) else {
            continue;
        };
        if matches!(vl, Verdict::Kill(_)) && legacy_at.is_none() {
            legacy_at = Some(i);
        }
        if matches!(vs, Verdict::Kill(_)) && smart_at.is_none() {
            smart_at = Some(i);
            smart_why = rig.smart.last_why();
        }
        if let Some((_, why)) = rig.smart.take_skip() {
            last_skip = Some(why);
        }
        if smart_at.is_none_or(|s| s == i)
            && let Some(own) = rig.tees.get(own_id)
        {
            trace.push(format!(
                "  tick {} frozen_for {:>4} pos ({:.0},{:.0}) v ({:.1},{:.1}) left {} hooked {} call {:?}",
                f.tick,
                rig.smart.frozen_for(f.tick, own),
                own.pos.x,
                own.pos.y,
                own.vel.x,
                own.vel.y,
                own.freeze_ticks_left,
                rig.tees.iter().any(|o| o.id != own_id && o.hooked_player == own_id),
                rig.smart.last_call()
            ));
        }
    }
    let Some(l) = legacy_at else {
        return Outcome::LegacyNotReproduced;
    };
    if !run_complete && l != k {
        return Outcome::LegacyNotReproduced;
    }
    match smart_at {
        Some(s) if s + SAME_FRAMES >= l => Outcome::Same {
            why: smart_why.unwrap_or(SmartWhy::UpperBound),
        },
        Some(s) => Outcome::Earlier {
            ticks: frames[l].tick - frames[s].tick,
            why: smart_why.unwrap_or(SmartWhy::UpperBound),
        },
        None => {
            let fate = counterfactual(&mut rig, clip, l);
            Outcome::Skipped { why: last_skip, fate }
        }
    }
}

/// Carries our tee on from frame `k` (where the smart policy did not kill) with no input and keeps judging.
fn counterfactual(rig: &mut Rig<'_>, clip: &Clip, k: usize) -> Fate {
    let frames = &clip.frames;
    let own_id = clip.header.own_id;
    let kill_tick = frames[k].tick;
    let Some(mut carried): Option<OwnState> = rig.lw.export_own(rig.lw.base_world()) else {
        return Fate::ClipEnded { ticks: 0 };
    };
    let mut last_state_frozen = true;
    for next in frames.iter().skip(k + 1) {
        let after = next.tick - kill_tick;
        if after > COUNTERFACTUAL_TICKS {
            break;
        }
        // Our tee, alone on the physics, with the recorded world around it, from the previous frame to this one.
        let inputs: Vec<(i32, PlayerInput)> = next.sent.iter().map(|s| (s.tick, PlayerInput::default())).collect();
        let predicted_alive = rig
            .lw
            .predict(next.tick, &inputs)
            .characters
            .get(own_id as usize)
            .is_some_and(Option::is_some);
        if !predicted_alive {
            return Fate::Died { after };
        }
        if let Some(c) = rig.lw.export_own_predicted() {
            carried = c;
        }
        let chars = rig.feed(next);
        // The recorded frame holds the new life; put the counterfactual tee in its place.
        if !rig.lw.import_own(&carried) {
            continue;
        }
        let Some((_, vs, _, _)) = rig.step(next, &chars, false) else {
            continue;
        };
        let own = rig.tees.get(own_id).copied();
        let Some(own) = own else { continue };
        last_state_frozen = own.frozen;
        if let Verdict::Kill(_) = vs {
            return Fate::KilledLater {
                after,
                why: rig.smart.last_why().unwrap_or(SmartWhy::UpperBound),
            };
        }
        if !own.frozen {
            return Fate::Thawed { after };
        }
    }
    let ticks = frames.last().map_or(0, |f| f.tick) - kill_tick;
    if last_state_frozen {
        Fate::StillFrozen { ticks }
    } else {
        Fate::ClipEnded { ticks }
    }
}

/// The duel detector over a clip: for every frame, whether it says duel; the first tick it does and by what. `f_ddrace`: the clip is from an
/// F-DDrace server (the clip has no chat, so the evidence the team signal needs is given from the first frame).
#[derive(Debug, Clone, Default)]
pub struct DuelScan {
    pub frames: usize,
    pub frames_in_duel: usize,
    pub first: Option<(i32, DuelWhy)>,
    /// Ticks of the clip's recorded `Cl_Kill`s that fall inside a duel (the detector would have stopped these).
    pub kills_in_duel: Vec<i32>,
    pub kills_outside: Vec<i32>,
    /// Our DDRace team over the clip: `(tick, team, players in it besides us)`, one entry per change.
    pub own_team: Vec<(i32, i32, usize)>,
}

pub fn scan_duel(clip: &Clip, f_ddrace: bool) -> DuelScan {
    let own_id = clip.header.own_id;
    let mut players = PlayerTable::new([3; 16]);
    players.update(&players_of(clip), &Relations::new());
    let mut det = DuelDetector::new();
    let mut scan = DuelScan::default();
    if f_ddrace {
        // A clip holds no chat: for a clip from an F-DDrace server the evidence the team signal needs is given from the start.
        det.on_chat(clip.frames.first().map_or(0, |f| f.tick), ChatSignal::Invited);
    }
    for f in &clip.frames {
        scan.frames += 1;
        let teams = teams_at(clip, f.tick);
        if let Some(t) = &teams
            && let Some(&team) = t.teams.get(own_id as usize).filter(|_| (own_id as usize) < t.received)
        {
            let others = (0..t.received)
                .filter(|&i| {
                    i as i32 != own_id && t.teams[i] == team && players.get(i as i32).is_some_and(|s| s.present)
                })
                .count();
            if scan.own_team.last().is_none_or(|l| (l.1, l.2) != (team, others)) {
                scan.own_team.push((f.tick, team, others));
            }
        }
        if let Some(DuelChange::Started(why)) = det.update(f.tick, own_id, teams.as_ref(), &players)
            && scan.first.is_none()
        {
            scan.first = Some((f.tick, why));
        }
        let on = det.active().is_some();
        scan.frames_in_duel += usize::from(on);
        for e in &f.events {
            if let ClipEvent::KillSent { .. } = e {
                if on {
                    scan.kills_in_duel.push(f.tick);
                } else {
                    scan.kills_outside.push(f.tick);
                }
            }
        }
    }
    scan
}

/// `KillReason` of a verdict, for the CLI's table.
pub fn reason_name(r: KillReason) -> &'static str {
    match r {
        KillReason::Overdue => "overdue",
        KillReason::WayBlockLying => "wayblock",
        KillReason::Stuck => "stuck",
    }
}

/// One probe of the forecast against what the clip shows happened.
#[derive(Debug, Clone)]
pub struct ForecastSample {
    pub clip: String,
    /// Tick of the first frozen frame of the run.
    pub run_start: i32,
    /// Ticks into the run when the forecast was made.
    pub age: i32,
    /// `Forecast::free_in` (ticks until free from the probe frame), `None`: held for the horizon.
    pub predicted: Option<i32>,
    pub died: bool,
    /// Ticks from the probe frame to the first free frame the clip shows.
    pub actual: i32,
    /// Somebody hooked us, or stood within 60 px, between the probe and the thaw: the forecast leaves the other tees out on purpose.
    pub touched: bool,
}

/// How good the passive forecast is, on the clip's frozen runs that ended by themselves (the tee free again in the next frame, no
/// death or `Cl_Kill` in between): at each of `ages` ticks into the run, what it said against what happened.
pub fn forecast_samples(clip: &Clip, map: &Arc<MapData>, clip_name: &str, ages: &[i32]) -> Vec<ForecastSample> {
    let frames = &clip.frames;
    let own_id = clip.header.own_id;
    let mut out = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        let frozen = |f: &Frame| f.own_alive && f.tee(own_id).is_some_and(|t| t.frozen);
        if !frozen(&frames[i]) || (i > 0 && frozen(&frames[i - 1])) {
            i += 1;
            continue;
        }
        // A run begins at `i` (the frame before it, if any, had the tee free: the run is whole).
        let s = i;
        let mut e = s;
        while e + 1 < frames.len() && frozen(&frames[e + 1]) {
            e += 1;
        }
        i = e + 1;
        let thawed = frames
            .get(e + 1)
            .is_some_and(|f| f.own_alive && f.tee(own_id).is_some_and(|t| !t.frozen));
        let killed = frames[s..=e + 1.min(frames.len() - 1 - e)].iter().any(|f| {
            f.events
                .iter()
                .any(|ev| matches!(ev, ClipEvent::KillSent { .. } | ClipEvent::Respawn { .. }))
        });
        if s == 0 || !thawed || killed {
            continue;
        }
        let mut rig = Rig::new(clip, map);
        let mut probes: Vec<(usize, i32)> = Vec::new();
        for &age in ages {
            if let Some(p) = (s..=e).find(|&p| frames[p].tick - frames[s].tick >= age) {
                probes.push((p, age));
            }
        }
        for (j, f) in frames.iter().enumerate().take(e + 1).skip(s.saturating_sub(1)) {
            rig.feed(f);
            for &(p, age) in probes.iter().filter(|&&(p, _)| p == j) {
                let fc = rig.plan_forecast(own_id);
                let touched = frames[p..=e + 1].iter().any(|fr| {
                    let Some(me) = fr.tee(own_id) else { return false };
                    fr.tees.iter().filter(|t| t.id != own_id).any(|t| {
                        t.ch.hooked_player == own_id
                            || me.ch.hooked_player == t.id
                            || (f64::from(t.ch.x - me.ch.x)).hypot(f64::from(t.ch.y - me.ch.y)) < 60.0
                    })
                });
                out.push(ForecastSample {
                    clip: clip_name.to_string(),
                    run_start: frames[s].tick,
                    age,
                    predicted: fc.free_in,
                    died: fc.died,
                    actual: frames[e + 1].tick - frames[p].tick,
                    touched,
                });
            }
        }
    }
    out
}

impl Rig<'_> {
    fn plan_forecast(&mut self, own_id: i32) -> ddai_planner::forecast::Forecast {
        self.plan
            .own_forecast(self.lw.base_world(), own_id, smartkill::FORECAST_HORIZON_TICKS)
    }
}
