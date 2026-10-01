//! Synthetic snapshots and a probing brain for the offline scenario tests: a small closed room with a
//! floor, tees standing on it, `LiveWorldSnapshot`s built the way the driver builds them. No network,
//! no real nicknames (names are `p<id>`-style test strings).
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use ddai_bot::{Bot, BotConfig, BrainKind, Hooks, Output, Relations};
use ddai_brain::{Action, Brain, LiveContext, Observation, ResetContext, WorldView};
use ddai_client::LiveWorldSnapshot;
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::{CharacterView, PlayerView};
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};

pub const W: u32 = 120;
pub const H: u32 = 40;

/// A `W x H` room: solid border; `extra` tiles `(x, y, tile index)`.
pub fn room(extra: &[(u32, u32, u8)]) -> Arc<MapData> {
    let mut game = vec![Tile::default(); (W * H) as usize];
    for x in 0..W {
        game[x as usize].index = TILE_SOLID;
        game[((H - 1) * W + x) as usize].index = TILE_SOLID;
    }
    for y in 0..H {
        game[(y * W) as usize].index = TILE_SOLID;
        game[(y * W + W - 1) as usize].index = TILE_SOLID;
    }
    for &(x, y, index) in extra {
        game[(y * W + x) as usize].index = index;
    }
    Arc::new(MapData {
        width: W,
        height: H,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

/// The y (px) of a tee resting on the bottom border (box bottom 1 px above the floor tile: a tee
/// placed at exactly `floor - 14` would be embedded in it and never move).
pub const FLOOR_Y: i32 = (H as i32 - 1) * 32 - 15;

/// One tee of a scenario.
#[derive(Debug, Clone)]
pub struct TeeSpec {
    pub id: i32,
    pub x: i32,
    pub y: i32,
    pub frozen: bool,
    pub angle: i32,
    pub attack_tick: i32,
    pub weapon: i32,
    pub hooked_player: i32,
    pub hook_state: i32,
    pub direction: i32,
}

pub fn tee(id: i32, x: i32) -> TeeSpec {
    TeeSpec {
        id,
        x,
        y: FLOOR_Y,
        frozen: false,
        angle: 0,
        attack_tick: 0,
        weapon: 0,
        hooked_player: -1,
        hook_state: 0,
        direction: 0,
    }
}

#[derive(Debug, Clone)]
pub struct PlayerSpec {
    pub id: i32,
    pub name: String,
    pub clan: String,
    pub team: i32,
    pub ex_flags: i32,
}

pub fn player(id: i32, name: &str) -> PlayerSpec {
    PlayerSpec {
        id,
        name: name.to_string(),
        clan: String::new(),
        team: 0,
        ex_flags: 0,
    }
}

pub fn character_view(t: &TeeSpec, tick: i32) -> CharacterView {
    CharacterView {
        id: t.id,
        character: objects::Character {
            tick: 0,
            x: t.x,
            y: t.y,
            vel_x: 0,
            vel_y: 0,
            angle: t.angle,
            direction: t.direction,
            jumped: 0,
            hooked_player: t.hooked_player,
            hook_state: t.hook_state,
            hook_tick: 0,
            hook_x: t.x,
            hook_y: t.y,
            hook_dx: 0,
            hook_dy: 0,
            player_flags: playerflagflag::PLAYING,
            health: 10,
            armor: 0,
            ammo_count: -1,
            weapon: t.weapon,
            emote: 0,
            attack_tick: t.attack_tick,
        },
        ddnet: Some(objects::DDNetCharacter {
            flags: 0,
            freeze_end: if t.frozen { tick + 120 } else { 0 },
            jumps: 2,
            tele_checkpoint: -1,
            strong_weak_id: t.id,
            jumped_total: -1,
            ninja_activation_tick: -1,
            freeze_start: if t.frozen { tick } else { -1 },
            target_x: 0,
            target_y: 0,
            tune_zone_override: -1,
        }),
    }
}

pub fn player_view(p: &PlayerSpec, own_id: i32) -> PlayerView {
    PlayerView {
        id: p.id,
        info: objects::PlayerInfo {
            local: i32::from(p.id == own_id),
            client_id: p.id,
            team: p.team,
            score: 0,
            latency: 20,
        },
        client_info: Some(objects::ClientInfo {
            name: p.name.clone(),
            clan: p.clan.clone(),
            country: -1,
            skin: "default".to_string(),
            use_custom_color: 0,
            color_body: 0,
            color_feet: 0,
        }),
        ddnet: Some(objects::DDNetPlayer {
            flags: p.ex_flags,
            auth_level: 0,
            finish_time_seconds: 0,
            finish_time_millis: 0,
        }),
    }
}

/// A scenario: tees and players, advanced tick by tick.
pub struct Scenario {
    pub map: Arc<MapData>,
    pub own_id: i32,
    pub tick: i32,
    pub tees: Vec<TeeSpec>,
    pub players: Vec<PlayerSpec>,
    /// `pred_tick` the driver would report, relative to the snapshot tick.
    pub pred_ahead: i32,
    /// An absolute `pred_tick` instead (0 = "the timing bootstrap has not happened yet").
    pub pred_tick_fixed: Option<i32>,
    /// When the driver's next input is due, after the snapshot (`None`: immediately).
    pub next_input_in: Option<std::time::Duration>,
}

impl Scenario {
    pub fn new(map: Arc<MapData>, tees: Vec<TeeSpec>) -> Self {
        let players = tees.iter().map(|t| player(t.id, &format!("p{}", t.id))).collect();
        Scenario {
            map,
            own_id: 0,
            tick: 1000,
            tees,
            players,
            pred_ahead: 3,
            pred_tick_fixed: None,
            next_input_in: Some(std::time::Duration::from_millis(15)),
        }
    }

    pub fn tee_mut(&mut self, id: i32) -> &mut TeeSpec {
        self.tees.iter_mut().find(|t| t.id == id).expect("tee")
    }

    pub fn player_mut(&mut self, id: i32) -> &mut PlayerSpec {
        self.players.iter_mut().find(|p| p.id == id).expect("player")
    }

    pub fn snapshot(&self) -> LiveWorldSnapshot {
        LiveWorldSnapshot {
            tick: self.tick,
            own_id: Some(self.own_id),
            characters: self.tees.iter().map(|t| character_view(t, self.tick)).collect(),
            tuning: DEFAULT_TUNE_PARAMS,
            switch_states: Vec::new(),
            teams: None,
            projectiles: Vec::new(),
            players: self.players.iter().map(|p| player_view(p, self.own_id)).collect(),
            pred_tick: self.pred_tick_fixed.unwrap_or(self.tick + self.pred_ahead),
            next_input_in: self.next_input_in,
            arrived: Instant::now(),
        }
    }
}

/// What a probing brain saw on one decision.
#[derive(Debug, Clone)]
pub struct Seen {
    pub obs_tick: i32,
    pub world_tick: i32,
    pub lag_ticks: u32,
    pub in_flight: usize,
    pub self_id: i32,
    pub target: Option<i32>,
    pub others: Vec<i32>,
    pub world_ids: Vec<i32>,
    pub spares: Vec<(f32, f32)>,
    /// `LiveContext::spare_ids` of that decision.
    pub spare_ids: Vec<i32>,
    /// Our own predicted position in the observation.
    pub self_x: f32,
    /// Whether that predicted tee is frozen.
    pub self_frozen: bool,
}

/// A brain that records what it is given and returns a fixed action.
pub struct Probe {
    pub log: Rc<RefCell<Vec<Seen>>>,
    pub resets: Rc<RefCell<Vec<ResetContext>>>,
    pub action: Rc<RefCell<Action>>,
    spares: Vec<(f32, f32)>,
    spare_ids: Vec<i32>,
}

impl Probe {
    #[allow(clippy::type_complexity)]
    pub fn new(
        action: Action,
    ) -> (
        Probe,
        Rc<RefCell<Vec<Seen>>>,
        Rc<RefCell<Vec<ResetContext>>>,
        Rc<RefCell<Action>>,
    ) {
        let log = Rc::new(RefCell::new(Vec::new()));
        let resets = Rc::new(RefCell::new(Vec::new()));
        let action = Rc::new(RefCell::new(action));
        (
            Probe {
                log: Rc::clone(&log),
                resets: Rc::clone(&resets),
                action: Rc::clone(&action),
                spares: Vec::new(),
                spare_ids: Vec::new(),
            },
            log,
            resets,
            action,
        )
    }
}

impl Brain for Probe {
    fn reset(&mut self, ctx: &ResetContext) {
        self.resets.borrow_mut().push(ctx.clone());
    }

    fn decide(&mut self, _obs: &Observation) -> Action {
        *self.action.borrow()
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let view = view.expect("the live bot always passes the exact world");
        let world_ids = (0..128)
            .filter(|&i| view.world.characters[i as usize].is_some())
            .collect::<Vec<i32>>();
        self.log.borrow_mut().push(Seen {
            obs_tick: obs.tick,
            world_tick: view.world.tick,
            lag_ticks: view.lag_ticks,
            in_flight: view.in_flight.len(),
            self_id: view.self_id,
            target: obs.target_id,
            others: obs.others.iter().map(|o| o.id).collect(),
            world_ids,
            spares: self.spares.clone(),
            spare_ids: self.spare_ids.clone(),
            self_x: obs.self_state.pos.x,
            self_frozen: obs.self_state.is_frozen,
        });
        *self.action.borrow()
    }

    fn set_live_context(&mut self, ctx: &LiveContext<'_>) {
        self.spares = ctx.spares.iter().map(|(p, _)| (p.x, p.y)).collect();
        self.spare_ids = ctx.spare_ids.to_vec();
    }

    fn name(&self) -> &str {
        "probe"
    }
}

pub fn bot_with(brain: Box<dyn Brain>, cfg: BotConfig, rel: Relations) -> Bot {
    Bot::new(cfg, brain, Hooks::default(), rel)
}

pub fn cfg(kind: BrainKind) -> BotConfig {
    BotConfig {
        brain: kind,
        salt: [7; 16],
        // Deterministic slot choice: the scenarios assert exact prediction ticks, and the real decision
        // and queue times of the host must not move them (3.5b review F8).
        decision_time_override: Some(std::time::Duration::from_millis(1)),
        ..BotConfig::default()
    }
}

/// Runs `n` snapshots (2 ticks apart), feeding back `InputSent` like the driver would, and returns
/// the outputs.
pub fn run(bot: &mut Bot, sc: &mut Scenario, n: usize) -> Vec<Output> {
    let mut outs = Vec::new();
    for _ in 0..n {
        let snap = sc.snapshot();
        let out = bot.on_snapshot(&snap);
        if let Some(input) = out.input {
            // The driver sends this decision's input for the next predicted tick.
            bot.on_input_sent(snap.tick + sc.pred_ahead + 1, &input);
        }
        outs.push(out);
        sc.tick += 2;
    }
    outs
}

pub const FREEZE: u8 = TILE_FREEZE;
pub const SOLID: u8 = TILE_SOLID;

/// Runs `f` on a thread with a 64 MB stack: the planner's helpers (shield, seal) copy whole
/// `World<f32>`s by value, which does not fit libtest's 2 MB default in an unoptimised-ish build
/// (3.5's review hit the same; the live runner gives the bot thread the same generous stack).
pub fn big_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e));
}
