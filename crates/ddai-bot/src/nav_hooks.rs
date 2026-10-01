//! The real bodies of the navigation hooks (task 4.2): goto / follow / trek / seek / home, the wayblock
//! (WB) on Copy Love Box, the freeze memory, and the reachability search — over `ddai-nav`.
//!
//! One [`Core`] holds all of it behind an `Rc<RefCell<…>>`; the four hook objects ([`Hooks`]) are thin
//! views of it, so the walk, the WB and the trek can see each other the way the TS `DdnetBot` fields do.
//! The bot thread is the only user (the brain is not `Send` either), so nothing here is `Sync`.
//!
//! **Who drives (task 4.2, spec 7).** Every snapshot the bot calls, in this order:
//!
//! 1. [`Navigator::poll`] — commands, side of the WB hall, freeze memory, the end of a walk; it may
//!    change the mode (`goto` while a walk runs, back to the mode it began from afterwards);
//! 2. [`Navigator::drive`] — while a walk runs it is the **navigator** that decides the input
//!    (`driveNav`, `bot.ts:1948`): the bot sends it (guarded by the shield unless the navigator is
//!    crossing a tube or walks a planned freeze), the brain is not asked and no target is picked. A walk
//!    that asks for a respawn gets one `Cl_Kill` (never chat) when the 500-tick cooldown allows;
//! 3. when nothing walks: the target pick, then [`Trek::steer`] (walk to where the game is, seek, home,
//!    back to the WB spot) which may *start* a walk for the next snapshot; the **brain** decides the
//!    input and gets the trek's/path's next point as `LiveContext::travel_goal` and the WB hints.
//!
//! Commands come from outside (the CLI now, the chat commands of task 4.3) through a [`NavHandle`]: a
//! `Send` queue of [`NavCommand`]s and a queue of replies, both drained by the bot thread at `poll`.
//!
//! **Differences from TS** (all listed in `docs/research/nav.md`): the freeze memory is keyed by the map's
//! sha256 (not its name); the planner gets *copies* of the memory and the dead zone
//! ([`Brain::set_map_knowledge`]) refreshed every few seconds instead of sharing a live object; the
//! partner/duel/rescue-friend branches of D-021 are gone; a goto that ends "arrived" within 2 tiles but
//! more than 40 px from the goal walks the last stretch (`finish_approach`, measured in
//! `ddai-nav/tests/arrival_vs_ts.rs`).

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ddai_brain::{Action, FreezeMemoryData, MapKnowledge, WbHints};
use ddai_nav::crossing::Crossing;
use ddai_nav::follow::{FOLLOW_ARRIVED_PX, Follow, FollowCtx, FollowVerdict, follow_tile};
use ddai_nav::home::{GO_HOME_AFTER_TICKS, Home};
use ddai_nav::memory::{MemoryStore, default_memory_dir};
use ddai_nav::navigator::{NavCtx, NavGoal, NavOpts, Navigator as TsNavigator, default_tele_goals, tile_goal};
use ddai_nav::route::{RouteOpts, Router, dead_zone, spawn_tiles};
use ddai_nav::trek::{ACTION_MEMORY_TICKS, CROWD_RADIUS_PX, PathGoal, Trek as TsTrek, game_spot};
pub use ddai_nav::wayblock::WbMode;
use ddai_nav::wayblock::{WB_NO_CLIMB_TILES, WB_RETURN_TICKS, WbState, on_wb_spot, wayblock_for, wb_spot};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_planner::brains::action_from_input;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::TeeState;
use ddai_planner::vmath::Vec2 as Vec2d;

use crate::bot::Mode;
use crate::consts::*;
use crate::hooks::{HookContext, Hooks, MapIdent, NavStep, Navigator, Poll, Trek, WanderHint, WayBlock, WbFilter};
use crate::mapgrid::MapGrid;
use crate::reach::{RouteFinder, Tile};
use crate::tees::{HOOK_FLYING, Tee, dist};

/// `TRAVEL_RETRY_TICKS` (`bot.ts:235`).
const TRAVEL_RETRY_TICKS: i64 = 5 * 50;
/// `SEEK_MARGIN`, `SEEK_PATIENCE_TICKS`, `SEEK_ARRIVED_PX` (`bot.ts:240-242`).
const SEEK_MARGIN: i32 = 3;
const SEEK_PATIENCE_TICKS: i64 = 4 * 50;
const SEEK_ARRIVED_PX: f32 = 500.0;
/// `COUNTER_REACH_PX` (`bot.ts:230`).
const COUNTER_REACH_PX: f32 = 64.0;
/// How often the status text is rebuilt (ticks).
const STATUS_EVERY_TICKS: i32 = 25;

/// Settings of the navigation hooks.
#[derive(Debug, Clone)]
pub struct NavConfig {
    /// Where the freeze memory lives (an absolute path; a relative one is refused); `None` keeps no
    /// memory at all (tests, or `HOME` unset).
    pub memory_dir: Option<PathBuf>,
    /// `!wb`: which hall to hold (default `auto`).
    pub wb_mode: WbMode,
    /// Strong mode (`--strong`): inside a hall the planner searches wider (`STRONG_WB`).
    pub strong: bool,
    /// `seekEnabled()`: walk to where the game is when it is dull here.
    pub seek: bool,
    /// Wall-clock budget of one crossing search step in milliseconds (`navBudgetMs`; 0 = unbounded).
    pub cross_budget_ms: f64,
}

impl Default for NavConfig {
    fn default() -> Self {
        NavConfig {
            memory_dir: default_memory_dir(),
            wb_mode: WbMode::Auto,
            strong: false,
            seek: true,
            cross_budget_ms: 10.0,
        }
    }
}

/// What a console command or the CLI asks of the navigation (the API task 4.3 builds its chat commands
/// on). Replies come back through [`NavHandle::drain_replies`].
#[derive(Debug, Clone, PartialEq)]
pub enum NavCommand {
    /// `?goto <x> <y>` (tiles). `through_freeze: false` is "without crossing freeze".
    Goto {
        tx: i32,
        ty: i32,
        through_freeze: bool,
    },
    /// `?goto tele`: to the nearest reachable teleporter.
    GotoTele,
    /// `?goto <nick>` / `@nick`: follow this client while it moves.
    Follow {
        client_id: i32,
    },
    /// `?stop` / `?goto stop`: call the walk off (back to the mode it began from).
    Stop,
    /// `!home <x> <y>`, `!home` (here), `!home off`.
    SetHome {
        tx: i32,
        ty: i32,
    },
    HomeHere,
    HomeOff,
    /// `!wb off | left | right | auto`.
    Wb(WbMode),
    /// `!seek on|off`.
    Seek(bool),
}

/// A snapshot of what the navigation is doing, for status lines and the web unit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NavStatus {
    /// A walk is running.
    pub walking: bool,
    /// The walk's progress text (`nav.progress`).
    pub progress: String,
    /// The WB line (`!wb` with no argument).
    pub wb: String,
    pub home: Option<(i32, i32)>,
    /// Freezes noted in the memory of this map.
    pub memory_events: i64,
    /// Our tee's tile at the last snapshot.
    pub tile: Option<(i32, i32)>,
    /// Walks that ended (arrived, blocked or cancelled) since the start.
    pub walks_ended: u32,
    /// How the last walk ended (`arrived: walked the route to …`, `blocked: …`).
    pub last_walk: String,
    /// `Cl_Kill`s the navigation asked for (a respawn step of a route) that went out.
    pub nav_kills: u32,
}

#[derive(Default)]
struct Shared {
    inbox: VecDeque<NavCommand>,
    replies: VecDeque<String>,
    status: NavStatus,
}

/// The `Send + Clone` handle to the navigation hooks of a running bot.
#[derive(Clone, Default)]
pub struct NavHandle {
    shared: Arc<Mutex<Shared>>,
}

impl NavHandle {
    pub fn new() -> NavHandle {
        NavHandle::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Queues a command; the bot thread takes it at its next snapshot with a live tee.
    pub fn send(&self, cmd: NavCommand) {
        self.lock().inbox.push_back(cmd);
    }

    pub fn goto_tile(&self, tx: i32, ty: i32) {
        self.send(NavCommand::Goto {
            tx,
            ty,
            through_freeze: true,
        });
    }

    pub fn goto_tele(&self) {
        self.send(NavCommand::GotoTele);
    }

    pub fn follow(&self, client_id: i32) {
        self.send(NavCommand::Follow { client_id });
    }

    pub fn stop(&self) {
        self.send(NavCommand::Stop);
    }

    /// The replies the bot thread produced since the last call (what TS returned as the command's text).
    pub fn drain_replies(&self) -> Vec<String> {
        self.lock().replies.drain(..).collect()
    }

    pub fn status(&self) -> NavStatus {
        self.lock().status.clone()
    }
}

/// Per-map state.
struct MapState {
    /// The live world copy, synced from the snapshot's base world when the navigation needs tees.
    world: PhysicsWorld,
    synced_tick: i32,
    /// The factory of scratch worlds for the crossing search.
    template: PhysicsWorld,
    router: Router,
    /// Built once per map and shared with the brain (`Arc`: never copied again).
    dead: Option<Arc<Vec<u8>>>,
    width: i32,
    height: i32,
}

type TsNav = TsNavigator<PhysicsWorld>;

struct Core {
    cfg: NavConfig,
    handle: NavHandle,
    started: Instant,
    ms: Option<MapState>,
    wb: WbState,
    memory: Option<MemoryStore>,
    /// New knowledge for the brain is waiting (a map load, or we froze).
    knowledge_due: bool,
    was_frozen: bool,
    last_tile: i32,

    /// The mode we believe the bot is in, and what we last told it.
    mode: Mode,
    synced_mode: Mode,
    nav_return: Mode,
    mode_request: Option<Mode>,

    nav: Option<TsNav>,
    follow: Option<Follow>,
    wb_walk: bool,
    seeking_game: bool,
    trek: Option<TsTrek>,
    trek_avoid: HashSet<i32>,
    path: PathGoal,
    home: Option<Home>,
    map_name: String,
    idle_since: i64,
    travel_since: i64,
    dull_since: i64,
    kill_wanted: bool,
    walks_ended: u32,
    last_walk: String,
    nav_kills: u32,
    last_tile_pos: Option<(i32, i32)>,
    last_kill_tick: i32,
    route_kill_tick: i32,
    last_status_tick: i32,
    seek_enabled: bool,
}

impl Core {
    fn new(cfg: NavConfig, handle: NavHandle) -> Core {
        let wb = WbState::new(None, cfg.wb_mode);
        let seek_enabled = cfg.seek;
        Core {
            cfg,
            handle,
            started: Instant::now(),
            ms: None,
            wb,
            memory: None,
            knowledge_due: false,
            was_frozen: false,
            last_tile: -1,
            mode: Mode::Fight,
            synced_mode: Mode::Fight,
            nav_return: Mode::Fight,
            mode_request: None,
            nav: None,
            follow: None,
            wb_walk: false,
            seeking_game: false,
            trek: None,
            trek_avoid: HashSet::new(),
            path: PathGoal::default(),
            home: None,
            map_name: String::new(),
            idle_since: -1,
            travel_since: i64::MIN / 2,
            dull_since: -1,
            kill_wanted: false,
            walks_ended: 0,
            last_walk: String::new(),
            nav_kills: 0,
            last_tile_pos: None,
            last_kill_tick: i32::MIN / 2,
            route_kill_tick: i32::MIN / 2,
            last_status_tick: i32::MIN / 2,
            seek_enabled,
        }
    }

    fn now_ms(&self) -> i64 {
        self.started.elapsed().as_millis() as i64
    }

    fn reply(&self, text: impl Into<String>) {
        let text = text.into();
        tracing::info!(target: "nav", "{text}");
        self.handle.lock().replies.push_back(text);
    }

    fn log(&self, text: &str) {
        tracing::info!(target: "nav", "{text}");
    }

    // ---- map -------------------------------------------------------------------------------------

    fn on_map(&mut self, map: &Arc<MapData>, ident: &MapIdent) {
        self.save_memory();
        let world = PhysicsWorld::new(Arc::clone(map), 1);
        let spawns = spawn_tiles(map);
        let router = Router::new(world.collision(), &spawns);
        let dead = (!spawns.is_empty()).then(|| Arc::new(dead_zone(&router.grid, &spawns)));
        let (width, height) = (world.collision().width(), world.collision().height());
        let def = wayblock_for(&ident.name, Some(world.collision()));
        self.wb.on_map(def);
        let template = world.new_scratch();
        self.ms = Some(MapState {
            world,
            synced_tick: i32::MIN,
            template,
            router,
            dead,
            width,
            height,
        });
        // An absolute directory only: a relative one would land wherever the process happens to run.
        self.memory = self.cfg.memory_dir.as_ref().filter(|d| d.is_absolute()).map(|dir| {
            let hex: String = ident.sha256.iter().map(|b| format!("{b:02x}")).collect();
            MemoryStore::open(dir, &hex, width, height)
        });
        if self.memory.is_none() && self.cfg.memory_dir.is_some() {
            tracing::warn!("the freeze memory directory is not an absolute path: running without a freeze memory");
        }
        // A new map ends every walk, trek and the path; the home survives only on the same map.
        self.drop_walk();
        self.trek = None;
        self.trek_avoid.clear();
        self.path.reset();
        if let Some(h) = &mut self.home
            && h.on_map_change(&ident.name)
        {
            self.home = None;
        }
        self.map_name = ident.name.clone();
        self.touch_status();
        self.idle_since = -1;
        self.knowledge_due = true;
        self.last_tile = -1;
        self.was_frozen = false;
    }

    fn save_memory(&mut self) {
        if let Some(m) = &mut self.memory {
            m.save();
        }
    }

    fn knowledge(&self) -> MapKnowledge {
        let Some(ms) = &self.ms else {
            return MapKnowledge::default();
        };
        MapKnowledge {
            width: ms.width,
            height: ms.height,
            dead_zone: ms.dead.clone(),
            // A deep copy, made when it is handed over (at a map load and when we freeze, i.e. when the
            // decision is of no use anyway): the live memory keeps mutating with every tile entered, and
            // a snapshot the brain shares must not be copied-on-write back on the decision path.
            freeze_memory: self.memory.as_ref().map(|m| FreezeMemoryData {
                cells: Arc::new(m.mem.cells().to_vec()),
                passes: Arc::new(m.mem.passes().to_vec()),
                events: m.mem.noted(),
            }),
        }
    }

    /// The live world copy at this snapshot's tick.
    fn sync(&mut self, ctx: &HookContext<'_>) {
        if let Some(ms) = &mut self.ms
            && ms.synced_tick != ctx.tick
        {
            ms.world.sync_from(ctx.world);
            ms.synced_tick = ctx.tick;
        }
    }

    fn tee_state(&self, id: i32) -> Option<TeeState> {
        self.ms.as_ref().and_then(|m| m.world.get_tee(id))
    }

    // ---- mode ------------------------------------------------------------------------------------

    fn fights(&self) -> bool {
        self.mode == Mode::Fight || (self.mode == Mode::Goto && self.nav_return == Mode::Fight)
    }

    /// `wbHolding()`.
    fn wb_holding(&self) -> bool {
        self.wb
            .holding(self.now_ms(), self.fights(), self.home.is_some(), false)
            .is_some()
    }

    fn set_mode(&mut self, m: Mode) {
        self.mode = m;
        self.mode_request = Some(m);
    }

    /// Keeps the cheap parts of the shared status current between the periodic rebuilds.
    fn touch_status(&self) {
        let mut sh = self.handle.lock();
        sh.status.walking = self.nav.is_some();
        if self.nav.is_none() {
            sh.status.progress.clear();
        }
        sh.status.home = self.home.as_ref().map(|h| (h.tx, h.ty));
        sh.status.walks_ended = self.walks_ended;
        sh.status.last_walk.clone_from(&self.last_walk);
        sh.status.nav_kills = self.nav_kills;
        sh.status.tile = self.last_tile_pos;
    }

    fn end_nav(&mut self) {
        self.nav = None;
        self.follow = None;
        self.seeking_game = false;
        self.wb_walk = false;
        let back = self.nav_return;
        self.set_mode(back);
        self.touch_status();
    }

    /// `dropNav` without a mode change (the map changed under it).
    fn drop_walk(&mut self) {
        if self.nav.take().is_some() && self.mode == Mode::Goto {
            let back = self.nav_return;
            self.set_mode(back);
        }
        self.follow = None;
        self.seeking_game = false;
        self.wb_walk = false;
        self.touch_status();
    }

    fn cancel_nav(&mut self, why: &str) -> String {
        if let Some(n) = &mut self.nav {
            n.cancel(why);
            self.last_walk = format!("cancelled: {why}");
            self.walks_ended += 1;
        }
        let back = self.nav_return;
        self.end_nav();
        format!("goto: {why}, back to {}", back.name())
    }

    fn nav_opts(&self, through_freeze: bool, crossings: Option<Vec<Crossing>>) -> NavOpts {
        let crossings =
            crossings.unwrap_or_else(|| self.wb.def.as_ref().map(|d| d.crossings.clone()).unwrap_or_default());
        NavOpts {
            through_freeze,
            crossings,
            finish_approach: true,
            ..NavOpts::default()
        }
    }

    fn start_nav(&mut self, goals: Vec<NavGoal>, opts: NavOpts) -> String {
        if self.nav.is_some() {
            self.log("goto: replaced by a new destination");
        }
        if self.mode != Mode::Goto {
            self.nav_return = self.mode;
        }
        self.seeking_game = false;
        self.follow = None;
        self.wb_walk = false;
        let first = goals.first().map(|g| g.label.clone()).unwrap_or_default();
        let rest = if goals.len() > 1 {
            format!(
                ", then {} more candidate{} if it is not a doorway",
                goals.len() - 1,
                if goals.len() > 2 { "s" } else { "" }
            )
        } else {
            String::new()
        };
        self.nav = Some(TsNav::new(goals, opts));
        self.set_mode(Mode::Goto);
        self.touch_status();
        format!("goto: {first}{rest}")
    }

    // ---- commands --------------------------------------------------------------------------------

    fn handle_command(&mut self, cmd: NavCommand, ctx: &HookContext<'_>) {
        match cmd {
            NavCommand::Stop => {
                let r = if self.nav.is_none() {
                    "not going anywhere".to_string()
                } else {
                    self.cancel_nav("cancelled")
                };
                self.reply(r);
            }
            NavCommand::Goto { tx, ty, through_freeze } => {
                let r = self.goto_tile(tx, ty, through_freeze);
                self.reply(r);
            }
            NavCommand::GotoTele => {
                let r = self.goto_tele(ctx);
                self.reply(r);
            }
            NavCommand::Follow { client_id } => {
                let r = self.goto_player(ctx, client_id);
                self.reply(r);
            }
            NavCommand::SetHome { tx, ty } => {
                self.home = Some(Home::new(tx, ty, &self.map_name));
                self.touch_status();
                self.reply(format!("home: ({tx},{ty}); the WB is not held while it is set"));
            }
            NavCommand::HomeHere => {
                let (tx, ty) = tile_of(ctx.own.pos);
                self.home = Some(Home::new(tx, ty, &self.map_name));
                self.touch_status();
                self.reply(format!("home: here ({tx},{ty}); the WB is not held while it is set"));
            }
            NavCommand::HomeOff => {
                self.home = None;
                self.touch_status();
                self.reply("home: off");
            }
            NavCommand::Wb(mode) => {
                self.wb.set_mode(mode);
                let mut gave = String::new();
                if mode == WbMode::Off && self.wb_walk && self.nav.is_some() {
                    gave = format!("{}; ", self.cancel_nav("the WB is off"));
                }
                let r = if self.wb.def.is_none() {
                    format!(
                        "{gave}WB: {} (this map has none; it applies on Copy Love Box)",
                        mode.name()
                    )
                } else if mode == WbMode::Off {
                    format!("{gave}WB: off -- it stays wherever the fight is")
                } else {
                    format!("{gave}WB: {}", mode.name())
                };
                self.reply(r);
            }
            NavCommand::Seek(on) => {
                self.seek_enabled = on;
                self.reply(format!("seek: {}", if on { "on" } else { "off" }));
            }
        }
    }

    fn goto_tile(&mut self, tx: i32, ty: i32, through_freeze: bool) -> String {
        let Some(ms) = &self.ms else {
            return "no map yet: without a collision grid there is nowhere to walk to".to_string();
        };
        let col = ms.world.collision();
        let (w, h) = (PlanCollision::width(col), PlanCollision::height(col));
        if tx < 0 || ty < 0 || tx >= w || ty >= h {
            return format!("({tx},{ty}) is off the map -- it is {w}x{h} tiles");
        }
        let (px, py) = (f64::from(tx * 32 + 16), f64::from(ty * 32 + 16));
        if PlanCollision::is_solid(col, px, py) {
            return format!("({tx},{ty}) is a wall");
        }
        if PlanCollision::is_death(col, px, py) {
            return format!("({tx},{ty}) is a death tile");
        }
        if PlanCollision::is_freeze(col, px, py) {
            return format!("({tx},{ty}) is freeze -- walking into it on purpose is not a plan");
        }
        let goal = tile_goal(col, tx, ty);
        let opts = self.nav_opts(through_freeze, None);
        self.start_nav(vec![goal], opts)
    }

    fn goto_tele(&mut self, ctx: &HookContext<'_>) -> String {
        self.sync(ctx);
        let Some(ms) = &self.ms else {
            return "no map yet".to_string();
        };
        let col = ms.world.collision();
        if !col.has_tele() {
            return "this map has no teleport layer".to_string();
        }
        let goals = default_tele_goals(col, f64::from(ctx.own.pos.x), f64::from(ctx.own.pos.y));
        if goals.is_empty() {
            return "no teleporter on this map can be reached from here by walking".to_string();
        }
        let opts = self.nav_opts(true, None);
        self.start_nav(goals, opts)
    }

    fn follow_goal(tx: i32, ty: i32, who: &str) -> NavGoal {
        NavGoal {
            tx,
            ty,
            label: format!("{who} at ({tx},{ty})"),
            tele: None,
        }
    }

    fn goto_player(&mut self, ctx: &HookContext<'_>, id: i32) -> String {
        self.sync(ctx);
        if id == ctx.own.id {
            return "that is the bot itself".to_string();
        }
        let who = format!("c{id}");
        if ctx.players.get(id).is_none_or(|s| !s.present) {
            return format!("nobody with client id {id} is on the server");
        }
        if ctx.players.get(id).is_some_and(|s| s.not_playing()) {
            return format!("{who} is not in the game (spectating or paused)");
        }
        let Some(tee) = self.tee_state(id).filter(|t| t.alive) else {
            return format!("{who} has no tee on the map right now");
        };
        let me = Vec2d {
            x: f64::from(ctx.own.pos.x),
            y: f64::from(ctx.own.pos.y),
        };
        if ddai_planner::vmath::vdistance(me, tee.pos) <= FOLLOW_ARRIVED_PX {
            return format!("already next to {who}");
        }
        let col = self.ms.as_ref().expect("map").world.collision();
        let goal =
            follow_tile(col, tee.pos).unwrap_or(((tee.pos.x / 32.0).trunc() as i32, (tee.pos.y / 32.0).trunc() as i32));
        let opts = self.nav_opts(true, None);
        let reply = self.start_nav(vec![Self::follow_goal(goal.0, goal.1, &who)], opts);
        self.follow = Some(Follow::new(id, goal, me, tee.pos, i64::from(ctx.tick)));
        format!(
            "{reply}, following them as they move; {}",
            self.after_arrival(ctx, id, &who)
        )
    }

    fn after_arrival(&self, ctx: &HookContext<'_>, id: i32, who: &str) -> String {
        if self.nav_return != Mode::Fight {
            return format!(
                "nobody is touched on the way, and it goes back to {} there, which fights nobody",
                self.nav_return.name()
            );
        }
        let flags = ctx.players.get(id).map(|s| s.flags).unwrap_or_default();
        if flags.friendly() {
            return format!("{who} is a friend, never touched on the way or after");
        }
        if flags.ignore {
            return format!("{who} is ignored, never touched on the way or after");
        }
        if flags.at_war() {
            return format!("not touched on the way; after arriving {who} is fought, being on the war list");
        }
        format!(
            "not touched on the way; after arriving it is back to fight, and {who} is a target there like anybody else who is playing"
        )
    }

    // ---- awake / crowd helpers ---------------------------------------------------------------------

    fn parked_in_freeze(ctx: &HookContext<'_>, t: &Tee) -> bool {
        t.frozen && (t.deep_frozen || ctx.clock.frozen_for(t, ctx.tick) > CROWD_FROZEN_TICKS)
    }

    /// `!afk(t, strict) && !parkedInFreeze(t)`.
    fn awake(ctx: &HookContext<'_>, t: &Tee) -> bool {
        !ctx.clock.afk(t.id, ctx.tick, ctx.players, true) && !Self::parked_in_freeze(ctx, t)
    }

    /// `someoneWorthFighting(ownId, at, within)`: an unfrozen, awake tee we could fight within reach.
    /// (TS asks `pickTarget` as well when it cuts a walk; friends, ignored and out-of-game tees are
    /// excluded here directly, which is what makes `pickTarget` skip them.)
    fn someone_worth_fighting(ctx: &HookContext<'_>, within: f32) -> bool {
        ctx.tees.iter().any(|t| {
            if t.id == ctx.own.id || t.frozen || dist(ctx.own.pos, t.pos) > within {
                return false;
            }
            let slot = ctx.players.get(t.id);
            if slot.is_some_and(|s| s.flags.never_target() || s.not_playing()) {
                return false;
            }
            !ctx.clock.afk(t.id, ctx.tick, ctx.players, false)
        })
    }

    fn crowd_at(ctx: &HookContext<'_>, at: Vec2<f32>) -> (i32, i32) {
        let (mut tees, mut busy) = (0, 0);
        for t in ctx.tees.iter() {
            if t.id == ctx.own.id || dist(at, t.pos) > CROWD_RADIUS_PX as f32 || !Self::awake(ctx, t) {
                continue;
            }
            tees += 1;
            if t.hook_state >= HOOK_FLYING || ctx.tick - t.attack_tick < ACTION_MEMORY_TICKS as i32 {
                busy += 1;
            }
        }
        (tees, busy)
    }

    fn engaged_now(ctx: &HookContext<'_>) -> bool {
        let me = ctx.own;
        if me.frozen || me.hooked_player >= 0 {
            return true;
        }
        ctx.tees.iter().any(|t| {
            t.id != me.id
                && (t.hooked_player == me.id
                    || (!t.frozen
                        && dist(me.pos, t.pos) < ENGAGED_PX
                        && !ctx.clock.afk(t.id, ctx.tick, ctx.players, false)))
        })
    }

    /// `gameSpot(ownId, from)` over the alive tees.
    fn game_spot(&self, ctx: &HookContext<'_>) -> Option<ddai_nav::trek::Spot> {
        let tees: Vec<TeeState> = ctx
            .tees
            .iter()
            .filter(|t| t.id != ctx.own.id)
            .map(|t| self.tee_state_of(t))
            .collect();
        let awake = |s: &TeeState| ctx.tees.get(s.id).is_some_and(|t| Self::awake(ctx, t));
        game_spot(
            self.wb.def.as_ref(),
            Vec2d {
                x: f64::from(ctx.own.pos.x),
                y: f64::from(ctx.own.pos.y),
            },
            &tees,
            i64::from(ctx.tick),
            &awake,
        )
    }

    /// The few fields of a [`TeeState`] the spot search reads, from the bot's own flat tee.
    fn tee_state_of(&self, t: &Tee) -> TeeState {
        let mut s = ddai_planner::types::blank_tee_state();
        s.id = t.id;
        s.alive = t.alive;
        s.pos = Vec2d {
            x: f64::from(t.pos.x),
            y: f64::from(t.pos.y),
        };
        s.hook_state = t.hook_state;
        s.attack_tick = i64::from(t.attack_tick);
        s.frozen = t.frozen;
        s
    }

    // ---- poll --------------------------------------------------------------------------------------

    fn poll(&mut self, ctx: &HookContext<'_>) -> Poll {
        if self.ms.is_none() {
            return Poll::default();
        }
        let tick = ctx.tick;
        self.last_tile_pos = Some(tile_of(ctx.own.pos));
        // 1. someone else changed the mode (`!mode`, `!stop`): a walk does not outlive that.
        if ctx.mode != self.synced_mode && self.mode_request.is_none() {
            self.mode = ctx.mode;
            self.synced_mode = ctx.mode;
            if self.nav.is_some() && ctx.mode != Mode::Goto {
                self.drop_walk_quiet();
            }
        }
        // 2. commands.
        loop {
            let cmd = self.handle.lock().inbox.pop_front();
            let Some(cmd) = cmd else { break };
            self.handle_command(cmd, ctx);
        }
        let own = ctx.own;
        let tile = tile_of(own.pos);
        // 3. the WB: side, forgiveness, the walk that cannot climb.
        if self.wb.def.is_some() {
            let holding = self.wb_holding();
            let counts = self.wb_counts(ctx);
            let own_free = own.alive && !own.frozen;
            let change = self.wb.update_side(
                tile,
                f64::from(own.pos.x) / 32.0,
                own_free,
                counts,
                i64::from(tick),
                holding,
            );
            if let Some(c) = change {
                self.log(&format!(
                    "WB: over to the {} ({} playing on the left, {} on the right)",
                    c.to.name(),
                    counts.0,
                    counts.1
                ));
                if self.wb_walk && self.nav.is_some() {
                    if let Some(n) = &mut self.nav {
                        n.cancel("the WB side changed");
                    }
                    self.end_nav();
                    self.idle_since = i64::from(tick) - WB_RETURN_TICKS - 1;
                }
            }
            if self.wb_walk
                && !own.frozen
                && let (Some(def), Some(side), Some(nav)) = (&self.wb.def, self.wb.side(), &self.nav)
                && let Some(goal) = nav.goal()
                && def.in_hall(side, tile.0, tile.1)
                && tile.1 - goal.ty >= WB_NO_CLIMB_TILES
            {
                let msg = format!(
                    "below the WB spot ({},{}) inside the hall: no climb up to it",
                    goal.tx, goal.ty
                );
                let line = self.cancel_nav(&msg);
                self.log(&line);
                self.idle_since = i64::from(tick) - WB_RETURN_TICKS - 1;
            }
            self.wb.arrived_in_hall(tile, own.frozen);
        }
        // 4. the freeze memory.
        self.note_memory(ctx, tile);
        // 5. status.
        if tick - self.last_status_tick >= STATUS_EVERY_TICKS || tick < self.last_status_tick {
            self.last_status_tick = tick;
            self.publish_status(ctx);
        }
        // 6. what the bot should know.
        let mut poll = Poll::default();
        if self.knowledge_due {
            self.knowledge_due = false;
            poll.knowledge = Some(self.knowledge());
        }
        if let Some(m) = self.mode_request.take() {
            self.synced_mode = m;
            poll.mode = Some(m);
        }
        poll
    }

    /// Somebody else changed the mode: end the walk without touching the mode again.
    fn drop_walk_quiet(&mut self) {
        if let Some(n) = &mut self.nav {
            n.cancel("the mode changed");
        }
        self.nav = None;
        self.follow = None;
        self.seeking_game = false;
        self.wb_walk = false;
        self.touch_status();
    }

    fn wb_counts(&self, ctx: &HookContext<'_>) -> (i32, i32) {
        let Some(def) = &self.wb.def else { return (0, 0) };
        let (mut l, mut r) = (0, 0);
        for t in ctx.tees.iter() {
            if t.id == ctx.own.id || !Self::awake(ctx, t) {
                continue;
            }
            let (tx, ty) = tile_of(t.pos);
            match def.side_at(tx, ty) {
                Some(ddai_nav::wayblock::WbSide::Left) => l += 1,
                Some(ddai_nav::wayblock::WbSide::Right) => r += 1,
                None => {}
            }
        }
        (l, r)
    }

    fn note_memory(&mut self, ctx: &HookContext<'_>, tile: (i32, i32)) {
        let own = ctx.own;
        let Some(ms) = &self.ms else { return };
        let Some(mem) = &mut self.memory else {
            self.was_frozen = own.frozen;
            return;
        };
        if own.alive && !own.frozen {
            let idx = tile.1 * ms.width + tile.0;
            if idx != self.last_tile {
                self.last_tile = idx;
                mem.note_pass(f64::from(own.pos.x), f64::from(own.pos.y));
            }
        }
        if own.frozen != self.was_frozen {
            if own.frozen {
                mem.note_freeze(f64::from(own.pos.x), f64::from(own.pos.y));
                // A freeze is the moment the planner gets the memory (the snapshot costs a map-sized copy;
                // a frozen tee decides nothing anyway): at most once per freeze, never per tile.
                self.knowledge_due = true;
            }
            self.was_frozen = own.frozen;
        }
    }

    fn publish_status(&mut self, ctx: &HookContext<'_>) {
        let me = self.tee_state(ctx.own.id);
        let progress = match (&self.nav, &self.follow) {
            (Some(n), Some(f)) if f.waiting => format!("waiting for c{}: no tee on the map", f.id),
            (Some(n), Some(f)) => format!("following c{}: {}", f.id, n.progress(me.as_ref())),
            (Some(n), None) => n.progress(me.as_ref()),
            _ => String::new(),
        };
        let wb = self.wb_line(ctx);
        let mut sh = self.handle.lock();
        sh.status = NavStatus {
            walking: self.nav.is_some(),
            progress,
            wb,
            home: self.home.as_ref().map(|h| (h.tx, h.ty)),
            memory_events: self.memory.as_ref().map_or(0, |m| m.mem.noted()),
            tile: self.last_tile_pos,
            walks_ended: self.walks_ended,
            last_walk: self.last_walk.clone(),
            nav_kills: self.nav_kills,
        };
    }

    /// `wbCommand` with no argument.
    fn wb_line(&self, ctx: &HookContext<'_>) -> String {
        let _ = ctx;
        if self.wb.def.is_none() {
            return format!("WB: '{}' has none", self.map_name);
        }
        let now = self.now_ms();
        let held = self.wb_holding();
        let paused = self.wb.paused_min(now);
        let why = if self.wb.mode == WbMode::Off {
            "off".to_string()
        } else if self.home.is_some() {
            "not held while home is set (home off)".to_string()
        } else if paused > 0 {
            format!(
                "{}, left alone for {paused} more min after dying on the way in",
                self.wb.mode.name()
            )
        } else if !held {
            format!("{}, not held in mode {}", self.wb.mode.name(), self.mode.name())
        } else {
            format!(
                "{}, holding the {}",
                self.wb.mode.name(),
                self.wb.side().map_or("?", |s| s.name())
            )
        };
        format!(
            "WB: {why}; playing: {} on the left, {} on the right",
            self.wb.counts.0, self.wb.counts.1
        )
    }

    // ---- drive (`driveNav`) ------------------------------------------------------------------------

    fn drive(&mut self, ctx: &HookContext<'_>) -> Option<NavStep> {
        self.nav.as_ref()?;
        self.sync(ctx);
        let me = self.tee_state(ctx.own.id)?;
        let tick = i64::from(ctx.tick);

        // `seekingGame` walks are cut short when a fight is at hand.
        if self.seeking_game
            && (!self.wb_walk
                || self
                    .wb
                    .walk_cuttable(tile_of(ctx.own.pos), ctx.own.frozen, self.wb_holding()))
            && Self::someone_worth_fighting(ctx, SEEK_ARRIVED_PX)
        {
            self.seeking_game = false;
            self.end_nav();
            self.log("found a game on the way; stopping the walk");
            return None;
        }

        let following = self.follow.is_some();
        if following {
            match self.steer_follow(ctx, &me) {
                FollowStep::Wait => {
                    return Some(NavStep::Input {
                        action: Action::neutral(),
                        guard: false,
                    });
                }
                FollowStep::Over => {
                    return Some(NavStep::Input {
                        action: Action::neutral(),
                        guard: false,
                    });
                }
                FollowStep::Go => {}
            }
        }
        let budget = self.cfg.cross_budget_ms;
        let ms = self.ms.as_mut()?;
        let others: Vec<TeeState> = ms
            .world
            .all_tees()
            .into_iter()
            .filter(|t| t.id != me.id && t.alive)
            .collect();
        let MapState {
            world,
            template,
            router,
            ..
        } = ms;
        let mut make = || template.new_scratch();
        let nav = self.nav.as_mut()?;
        nav.cross_budget_ms = budget;
        let want = {
            let mut nctx = NavCtx {
                col: world.collision(),
                router,
                make_sim: &mut make,
            };
            nav.step(&mut nctx, &me, tick, &others, i64::from(ctx.lag_ticks))
        };
        let guard = !(nav.crossing() || nav.planned_freeze());
        let notes = nav.take_notes();
        let kill = nav.take_kill();
        let done = nav.done();
        for n in notes {
            self.log(&format!("goto: {n}"));
        }
        let action = action_from_input(&want);
        if kill {
            // The bot applies the cooldown and reports back through `kill_sent`.
            return Some(NavStep::Kill { action });
        }
        if done && !following {
            let (phase, outcome) = self
                .nav
                .as_ref()
                .map(|n| (n.phase().name(), n.outcome().to_string()))
                .unwrap_or(("?", String::new()));
            self.last_walk = format!("{phase}: {outcome}");
            self.walks_ended += 1;
            let back = self.nav_return;
            self.end_nav();
            self.log(&format!(
                "goto: {}",
                if back == Mode::Hold {
                    "standing by".to_string()
                } else {
                    format!("back to {}", back.name())
                }
            ));
        }
        Some(NavStep::Input { action, guard })
    }

    fn steer_follow(&mut self, ctx: &HookContext<'_>, me: &TeeState) -> FollowStep {
        let Some(f) = &self.follow else { return FollowStep::Go };
        let id = f.id;
        let target = self.tee_state(id);
        let slot = ctx.players.get(id);
        let col = self.ms.as_ref().expect("map").world.collision();
        let goal = target.as_ref().and_then(|t| follow_tile(col, t.pos));
        let (phase, outcome) = {
            let n = self.nav.as_ref().expect("nav");
            (n.phase(), n.outcome().to_string())
        };
        let fc = FollowCtx {
            tick: i64::from(ctx.tick),
            me,
            target: target.as_ref(),
            target_away: slot.is_some_and(|s| s.not_playing()),
            target_on_server: slot.is_some_and(|s| s.present),
            nav_phase: phase,
            nav_outcome: &outcome,
            goal,
        };
        let verdict = self.follow.as_mut().expect("follow").steer(&fc);
        match verdict {
            FollowVerdict::End(why) => {
                let back = self.nav_return;
                self.last_walk = format!("follow: {why}");
                self.walks_ended += 1;
                self.end_nav();
                self.log(&format!("goto: c{id}: {why} -- back to {}", back.name()));
                FollowStep::Over
            }
            FollowVerdict::Wait => FollowStep::Wait,
            FollowVerdict::Reroute(g) => {
                let opts = self.nav_opts(true, None);
                self.nav = Some(TsNav::new(vec![Self::follow_goal(g.0, g.1, &format!("c{id}"))], opts));
                if self.nav.as_ref().is_some_and(|n| n.done()) {
                    FollowStep::Wait
                } else {
                    FollowStep::Go
                }
            }
            FollowVerdict::Go => {
                if self.nav.as_ref().is_none_or(|n| n.done()) {
                    FollowStep::Wait
                } else {
                    FollowStep::Go
                }
            }
        }
    }

    // ---- trek, seek, home, back to the WB spot ---------------------------------------------------------

    fn start_trek(&mut self, ctx: &HookContext<'_>, to: (f64, f64)) -> String {
        let Some(ms) = &mut self.ms else {
            return "no map".to_string();
        };
        let from = Vec2d {
            x: f64::from(ctx.own.pos.x),
            y: f64::from(ctx.own.pos.y),
        };
        match TsTrek::start(
            &mut ms.router,
            ms.world.collision(),
            from,
            to,
            &self.trek_avoid,
            i64::from(ctx.tick),
        ) {
            Ok(t) => {
                let line = t.describe();
                self.trek = Some(t);
                self.seeking_game = true;
                line
            }
            Err(e) => {
                self.trek = None;
                self.seeking_game = false;
                format!("{e} (map {})", self.map_name)
            }
        }
    }

    fn end_trek(&mut self) {
        self.trek = None;
        self.seeking_game = false;
    }

    fn steer(&mut self, ctx: &HookContext<'_>, target: i32) {
        if self.ms.is_none() || self.nav.is_some() {
            return;
        }
        let tick = i64::from(ctx.tick);
        if self.trek.is_some() && Self::someone_worth_fighting(ctx, SEEK_ARRIVED_PX) {
            self.end_trek();
            self.log("found a game on the way; stopping the walk");
        }
        let holding = self.wb_holding();
        if self.mode == Mode::Fight
            && target != -1
            && self.seek_enabled
            && !holding
            && tick - self.travel_since > TRAVEL_RETRY_TICKS
            && !Self::engaged_now(ctx)
            && !Self::someone_worth_fighting(ctx, SEEK_ARRIVED_PX)
        {
            let here = Self::crowd_at(ctx, ctx.own.pos);
            if let Some(spot) = self.game_spot(ctx)
                && spot.tees + spot.busy >= here.0 + here.1 + SEEK_MARGIN
            {
                if self.dull_since < 0 {
                    self.dull_since = tick;
                }
                if tick - self.dull_since > SEEK_PATIENCE_TICKS {
                    self.travel_since = tick;
                    self.dull_since = -1;
                    let reply = self.start_trek(ctx, (spot.x, spot.y));
                    self.log(&format!(
                        "here: {} tees, {} fighting; there: {}/{} at ({},{}), {}px -- {reply}",
                        here.0,
                        here.1,
                        spot.tees,
                        spot.busy,
                        (spot.x / 32.0).trunc() as i32,
                        (spot.y / 32.0).trunc() as i32,
                        spot.dist.round()
                    ));
                }
            } else {
                self.dull_since = -1;
            }
        }
        if target != -1 {
            self.idle_since = -1;
            return;
        }
        // Nobody to fight.
        if self.trek.is_some() {
            self.end_trek();
        }
        if self.idle_since < 0 {
            self.idle_since = tick;
        }
        if !holding && self.nav.is_none() && tick - self.travel_since > TRAVEL_RETRY_TICKS {
            self.walk_to_game(ctx, tick);
        }
        let here = tile_of(ctx.own.pos);
        if let Some(h) = self.home.as_ref().filter(|_| self.nav.is_none())
            && tick - self.idle_since > GO_HOME_AFTER_TICKS
        {
            let (hx, hy) = (h.tx, h.ty);
            if h.due(tick, self.idle_since, here) {
                let reply = self.goto_tile(hx, hy, true);
                self.log(&format!("nobody to fight: walking home to ({hx},{hy}) -- {reply}"));
            }
            self.idle_since = tick;
        } else if holding && self.nav.is_none() && tick - self.idle_since > WB_RETURN_TICKS {
            self.walk_to_wb(ctx);
            self.idle_since = tick;
        }
    }

    /// "nobody in reach; walk to where the game is" (`bot.ts:2548-2564`).
    fn walk_to_game(&mut self, ctx: &HookContext<'_>, tick: i64) {
        let Some(spot) = self.game_spot(ctx) else { return };
        self.travel_since = tick;
        let (tx, ty) = ((spot.x / 32.0).trunc() as i32, (spot.y / 32.0).trunc() as i32);
        let Some(ms) = &mut self.ms else { return };
        let way = ms.router.find_route(
            (f64::from(ctx.own.pos.x), f64::from(ctx.own.pos.y)),
            (spot.x, spot.y),
            &RouteOpts {
                near_tiles: 3,
                partial: false,
                allow_kill: true,
                through_freeze: false,
                max_nodes: REACH_MAX_NODES,
                ..RouteOpts::default()
            },
        );
        if way.is_none() {
            self.log(&format!(
                "nobody in reach; the game is at ({tx},{ty}) but there is no way there without freeze -- staying"
            ));
            return;
        }
        let reply = self.goto_tile(tx, ty, false);
        self.seeking_game = self.nav.is_some() && self.nav_return == Mode::Fight;
        self.log(&format!(
            "nobody in reach; walking to where the game is: ({tx},{ty}), {}px, {} tees, {} of them fighting -- {reply}",
            spot.dist.round(),
            spot.tees,
            spot.busy
        ));
    }

    /// `walkToWb`.
    fn walk_to_wb(&mut self, ctx: &HookContext<'_>) {
        if !self.wb_holding() {
            return;
        }
        self.sync(ctx);
        let (Some(def), Some(side)) = (self.wb.def.clone(), self.wb.side()) else {
            return;
        };
        let (tx, ty) = tile_of(ctx.own.pos);
        let inside = def.in_hall(side, tx, ty);
        let tees: Vec<TeeState> = ctx.tees.iter().map(|t| self.tee_state_of(t)).collect();
        let is_friend = |id: i32| ctx.players.get(id).is_some_and(|s| s.flags.friendly());
        let first = wb_spot(ctx.own.id, &tees, &is_friend, &def, side, Some((tx, ty)));
        let spots: Vec<(i32, i32)> = std::iter::once(first)
            .chain(def.side(side).spots.iter().copied())
            .collect();
        let mut spot = None;
        let Some(ms) = &mut self.ms else { return };
        for p in spots {
            if on_wb_spot((tx, ty), p) {
                return;
            }
            if !inside {
                spot = Some(p);
                break;
            }
            if ty - p.1 >= WB_NO_CLIMB_TILES {
                continue;
            }
            let way = ms.router.find_route(
                (f64::from(ctx.own.pos.x), f64::from(ctx.own.pos.y)),
                (f64::from(p.0 * 32 + 16), f64::from(p.1 * 32 + 16)),
                &RouteOpts {
                    near_tiles: 1,
                    partial: false,
                    allow_kill: false,
                    through_freeze: false,
                    max_nodes: REACH_MAX_NODES,
                    ..RouteOpts::default()
                },
            );
            if way.is_some() {
                spot = Some(p);
                break;
            }
        }
        let Some(spot) = spot else {
            self.log(&format!(
                "WB {}: no way back to its spots from ({tx},{ty}) without the freeze; holding here",
                side.name()
            ));
            return;
        };
        let goal = tile_goal(self.ms.as_ref().expect("map").world.collision(), spot.0, spot.1);
        let opts = if inside {
            self.nav_opts(false, None)
        } else {
            self.nav_opts(true, Some(def.crossings.clone()))
        };
        let reply = self.start_nav(vec![goal], opts);
        self.wb_walk = self.nav.is_some();
        self.seeking_game = self.nav.is_some() && self.nav_return == Mode::Fight;
        self.log(&format!(
            "WB {}: back to ({},{}) -- {reply}",
            side.name(),
            spot.0,
            spot.1
        ));
    }

    fn trek_goal(&mut self, ctx: &HookContext<'_>, target: &Tee) -> Option<Vec2<f32>> {
        let tick = i64::from(ctx.tick);
        let me = Vec2d {
            x: f64::from(ctx.own.pos.x),
            y: f64::from(ctx.own.pos.y),
        };
        let kill_ready = ctx.tick - self.last_kill_tick >= KILL_COOLDOWN_TICKS;
        if let Some(trek) = &mut self.trek {
            let step = trek.goal(me, tick, kill_ready, &mut self.trek_avoid);
            if step.kill {
                self.kill_wanted = true;
            }
            if let Some(n) = &step.note {
                tracing::info!(target: "nav", "{n}");
            }
            if step.ended {
                self.end_trek();
            }
            if let Some(g) = step.goal {
                return Some(Vec2::new(g.x as f32, g.y as f32));
            }
        }
        let ms = self.ms.as_mut()?;
        let tpos = Vec2d {
            x: f64::from(target.pos.x),
            y: f64::from(target.pos.y),
        };
        self.path
            .goal(&mut ms.router, ms.world.collision(), me, target.id, tpos, tick)
            .map(|g| Vec2::new(g.x as f32, g.y as f32))
    }

    // ---- wayblock ------------------------------------------------------------------------------------

    fn wb_filter(&self, ctx: &HookContext<'_>, cand: &Tee) -> WbFilter {
        let (Some(def), Some(side)) = (&self.wb.def, self.wb.side()) else {
            return WbFilter::default();
        };
        let own = ctx.own;
        let (tx, ty) = tile_of(cand.pos);
        let me_in_leash = {
            let (ox, oy) = tile_of(own.pos);
            def.in_hall(side, ox, oy)
        };
        let flags = ctx.players.get(cand.id).map(|s| s.flags).unwrap_or_default();
        let roped = cand.hooked_player == own.id || own.hooked_player == cand.id;
        let at_us = ctx.clock.at_us_within(cand.id, ctx.tick, AGGRESSOR_MEMORY_TICKS);
        let d = dist(own.pos, cand.pos);
        let in_leash = def.in_leash(side, tx, ty);
        let counter = me_in_leash && at_us && d <= HOOK_LENGTH_PX + COUNTER_REACH_PX && !in_leash;
        let skip = !roped && !flags.at_war() && !counter && if me_in_leash { !in_leash } else { !at_us };
        if skip {
            return WbFilter {
                skip: true,
                ..WbFilter::default()
            };
        }
        let in_zone = def.in_zone(side, tx, ty);
        WbFilter {
            skip: false,
            in_zone,
            finish_zone: me_in_leash && in_zone,
        }
    }

    fn wb_wants_kill(&self, ctx: &HookContext<'_>, frozen_for: i32) -> bool {
        let Some(def) = &self.wb.def else { return false };
        let own = ctx.own;
        let (tx, ty) = tile_of(own.pos);
        if def
            .left
            .zone
            .iter()
            .chain(def.right.zone.iter())
            .any(|b| b.contains(tx, ty))
        {
            return false;
        }
        let hooked = ctx.tees.iter().any(|o| o.id != own.id && o.hooked_player == own.id);
        own.frozen
            && ctx.grid.is_freeze(own.pos.x, own.pos.y)
            && !hooked
            && !crate::unstick::helper_near(own, ctx.tees, ctx.players)
            && own.vel.x.hypot(own.vel.y) < 0.5
            && frozen_for >= WB_LYING_TICKS
    }

    fn wb_brain_hints(&self, ctx: &HookContext<'_>) -> WbHints {
        let tile = tile_of(ctx.own.pos);
        match self.wb.hall_hints(tile, self.wb_holding()) {
            Some(band) => WbHints {
                in_hall: true,
                strong: self.cfg.strong,
                band: Some(band),
            },
            None => WbHints::default(),
        }
    }

    fn wb_wander_hint(&self, ctx: &HookContext<'_>) -> Option<WanderHint> {
        let (def, side) = (self.wb.def.as_ref()?, self.wb.side()?);
        let (tx, ty) = tile_of(ctx.own.pos);
        if !def.in_hall(side, tx, ty) {
            return None;
        }
        let tees: Vec<TeeState> = ctx.tees.iter().map(|t| self.tee_state_of(t)).collect();
        let is_friend = |id: i32| ctx.players.get(id).is_some_and(|s| s.flags.friendly());
        let spot = wb_spot(ctx.own.id, &tees, &is_friend, def, side, Some((tx, ty)));
        let watch = def.side(side).watch;
        Some(WanderHint {
            anchor_x: (spot.0 * 32 + 16) as f32,
            look_at: Some(((watch.0 * 32 + 16) as f32, (watch.1 * 32 + 16) as f32)),
        })
    }

    // ---- kills ---------------------------------------------------------------------------------------

    fn kill_sent(&mut self, tick: i32, by_route: bool) {
        self.last_kill_tick = tick;
        if by_route {
            self.nav_kills += 1;
            self.route_kill_tick = tick;
        }
    }

    fn on_kill(&mut self, victim: i32, own_id: i32, tick: i32) {
        if let Some(f) = &mut self.follow {
            f.on_kill(victim, own_id, i64::from(tick), i64::from(self.last_kill_tick));
        }
        if victim == own_id && self.wb_walk && tick - self.route_kill_tick > 50 {
            let now = self.now_ms();
            if let Some(ms) = self.wb.note_walk_death(now) {
                if self.nav.is_some() && self.wb_walk {
                    let line = self.cancel_nav("the WB cannot be reached");
                    self.log(&line);
                }
                self.log(&format!(
                    "WB: {} deaths on the way in a row -- playing where it stands for {} min",
                    ddai_nav::wayblock::WB_WALK_MAX_FAILS,
                    ms / 60_000
                ));
            }
        }
    }

    fn reachable(&mut self, from: Tile, to: Tile, near_tiles: i32, max_nodes: usize) -> bool {
        let Some(ms) = &mut self.ms else { return true };
        let c = |t: i32| f64::from(t * 32 + 16);
        ms.router
            .find_route(
                (c(from.0), c(from.1)),
                (c(to.0), c(to.1)),
                &RouteOpts {
                    near_tiles,
                    max_nodes,
                    through_freeze: false,
                    partial: false,
                    ..RouteOpts::default()
                },
            )
            .is_some()
    }
}

enum FollowStep {
    Go,
    Wait,
    Over,
}

fn tile_of(p: Vec2<f32>) -> (i32, i32) {
    ((p.x / 32.0).trunc() as i32, (p.y / 32.0).trunc() as i32)
}

// ---- the hook objects --------------------------------------------------------------------------------

type Shared2 = Rc<RefCell<Core>>;

struct NavHook(Shared2);
struct WbHook(Shared2);
struct TrekHook(Shared2);
struct RouteHook(Shared2);

impl Navigator for NavHook {
    fn on_map(&mut self, map: &Arc<MapData>, ident: &MapIdent) {
        self.0.borrow_mut().on_map(map, ident);
    }
    fn on_map_changing(&mut self) {
        let mut c = self.0.borrow_mut();
        c.save_memory();
        c.drop_walk();
        c.trek = None;
        c.ms = None;
    }
    fn respawned(&mut self) {
        let mut c = self.0.borrow_mut();
        if let Some(n) = &mut c.nav {
            n.respawned();
        }
    }
    fn poll(&mut self, ctx: &HookContext<'_>) -> Poll {
        self.0.borrow_mut().poll(ctx)
    }
    fn drive(&mut self, ctx: &HookContext<'_>) -> Option<NavStep> {
        self.0.borrow_mut().drive(ctx)
    }
    fn vetoed(&mut self) {
        if let Some(n) = &mut self.0.borrow_mut().nav {
            n.vetoed();
        }
    }
    fn kill_sent(&mut self, tick: i32, by_route: bool) {
        self.0.borrow_mut().kill_sent(tick, by_route);
    }
    fn on_kill(&mut self, victim: i32, own_id: i32, tick: i32) {
        self.0.borrow_mut().on_kill(victim, own_id, tick);
    }
    fn in_dead_zone(&self, pos: Vec2<f32>) -> bool {
        let c = self.0.borrow();
        let Some(ms) = &c.ms else { return false };
        let Some(dead) = &ms.dead else { return false };
        // `inDeadZone` of the planner (`planner.ts:512`): the combined index only is bounds-checked.
        let i =
            i64::from((pos.y / 32.0).trunc() as i32) * i64::from(ms.width) + i64::from((pos.x / 32.0).trunc() as i32);
        i >= 0 && (i as usize) < dead.len() && dead[i as usize] == 1
    }
    fn stop(&mut self) {
        self.0.borrow_mut().save_memory();
    }
}

impl WayBlock for WbHook {
    fn holding(&self) -> bool {
        self.0.borrow().wb_holding()
    }
    fn filter(&mut self, ctx: &HookContext<'_>, candidate: &Tee) -> WbFilter {
        self.0.borrow().wb_filter(ctx, candidate)
    }
    fn wants_kill(&mut self, ctx: &HookContext<'_>, frozen_for: i32) -> bool {
        self.0.borrow().wb_wants_kill(ctx, frozen_for)
    }
    fn brain_hints(&mut self, ctx: &HookContext<'_>) -> WbHints {
        self.0.borrow().wb_brain_hints(ctx)
    }
    fn wander_hint(&mut self, ctx: &HookContext<'_>) -> Option<WanderHint> {
        self.0.borrow().wb_wander_hint(ctx)
    }
}

impl Trek for TrekHook {
    fn goal(&mut self, ctx: &HookContext<'_>, target: &Tee) -> Option<Vec2<f32>> {
        self.0.borrow_mut().trek_goal(ctx, target)
    }
    fn steer(&mut self, ctx: &HookContext<'_>, target: i32) {
        self.0.borrow_mut().steer(ctx, target);
    }
    fn take_kill(&mut self) -> bool {
        std::mem::take(&mut self.0.borrow_mut().kill_wanted)
    }
}

impl RouteFinder for RouteHook {
    fn reachable(&mut self, _grid: &MapGrid, from: Tile, to: Tile, near_tiles: i32, max_nodes: usize) -> bool {
        self.0.borrow_mut().reachable(from, to, near_tiles, max_nodes)
    }
}

/// The hooks of a bot with real navigation. `handle` is how commands reach it and replies leave it.
pub fn nav_hooks(cfg: NavConfig, handle: NavHandle) -> Hooks {
    let core = Rc::new(RefCell::new(Core::new(cfg, handle)));
    Hooks {
        navigator: Box::new(NavHook(Rc::clone(&core))),
        wayblock: Box::new(WbHook(Rc::clone(&core))),
        trek: Box::new(TrekHook(Rc::clone(&core))),
        route: Box::new(RouteHook(core)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activity::ActivityClock;
    use crate::mapgrid::test_maps::{FREEZE, room};
    use crate::players::{PlayerTable, Salt, test_support::player};
    use crate::relations::Relations;
    use crate::tees::TeeSet;
    use ddai_planner::brains::input_from_action;
    use ddai_planner::types::empty_input;

    const SALT: Salt = [7; 16];

    /// A live bot's view of a closed-loop world: `PhysicsWorld` as the "server", the hooks as the bot.
    struct Fx {
        pw: PhysicsWorld,
        map: Arc<MapData>,
        hooks: Hooks,
        handle: NavHandle,
        tees: TeeSet,
        players: PlayerTable,
        clock: ActivityClock,
        grid: MapGrid,
        mode: Mode,
        tick: i32,
        prev: ddai_planner::types::PlayerInput,
        kills: u32,
    }

    impl Fx {
        fn new(map: MapData, cfg: NavConfig, sha: u8) -> Fx {
            let map = Arc::new(map);
            let handle = NavHandle::new();
            let mut hooks = nav_hooks(cfg, handle.clone());
            hooks.navigator.on_map(
                &map,
                &MapIdent {
                    name: "test room".to_string(),
                    sha256: [sha; 32],
                },
            );
            let mut players = PlayerTable::new(SALT);
            let views = [
                player(0, "bot", "", true, 0, None),
                player(1, "other", "", false, 0, None),
            ];
            players.update(&views, &Relations::new());
            Fx {
                pw: PhysicsWorld::new(Arc::clone(&map), 1),
                grid: MapGrid::new(&map),
                map,
                hooks,
                handle,
                tees: TeeSet::new(),
                players,
                clock: ActivityClock::new(),
                mode: Mode::Fight,
                tick: 1000,
                prev: empty_input(),
                kills: 0,
            }
        }

        fn place(&mut self, id: i32, tx: i32, ty: i32) {
            self.pw.add_tee(
                id,
                Vec2d {
                    x: f64::from(tx * 32 + 16),
                    y: f64::from(ty * 32 + 16),
                },
            );
        }

        fn refresh_tees(&mut self) {
            let w = self.pw.inner();
            for id in 0..2usize {
                let (Some(core), Some(ch)) = (w.cores.get(id as u8), w.characters[id].as_ref()) else {
                    continue;
                };
                self.tees.set_for_test(Tee {
                    id: id as i32,
                    alive: true,
                    pos: core.pos,
                    vel: core.vel,
                    frozen: ch.freeze_time > 0,
                    freeze_ticks_left: ch.freeze_time.max(0),
                    hook_state: core.hook_state,
                    hooked_player: core.hooked_player(),
                    direction: core.direction,
                    jumped: core.jumped,
                    weapon: core.active_weapon,
                    ..Tee::DEAD
                });
            }
        }

        /// One bot tick: poll, drive, apply the input to the world, step it.
        fn tick(&mut self) -> Option<NavStep> {
            self.refresh_tees();
            let own = *self.tees.get(0).expect("own tee");
            let world = self.pw.inner().clone();
            let ctx = HookContext {
                tick: self.tick,
                own: &own,
                tees: &self.tees,
                players: &self.players,
                grid: &self.grid,
                clock: &self.clock,
                world: &world,
                lag_ticks: 0,
                mode: self.mode,
            };
            let poll = self.hooks.navigator.poll(&ctx);
            if let Some(m) = poll.mode {
                self.mode = m;
            }
            let ctx = HookContext { mode: self.mode, ..ctx };
            let step = self.hooks.navigator.drive(&ctx);
            let input = match step {
                Some(NavStep::Input { action, .. }) => input_from_action(&action, &self.prev),
                Some(NavStep::Kill { action }) => {
                    self.kills += 1;
                    input_from_action(&action, &self.prev)
                }
                None => empty_input(),
            };
            self.prev = input;
            self.pw.set_input(0, input);
            self.pw.step();
            self.tick += 1;
            step
        }

        fn run_until(&mut self, max: i32, mut stop: impl FnMut(&mut Fx) -> bool) -> i32 {
            for t in 0..max {
                self.tick();
                if stop(self) {
                    return t;
                }
            }
            max
        }

        fn own_pos(&self) -> Vec2<f32> {
            self.pw.inner().cores.get(0).expect("tee").pos
        }

        fn replies(&self) -> Vec<String> {
            self.handle.drain_replies()
        }
    }

    fn no_memory_dir_for_knowledge() -> NavConfig {
        // The memory exists (so there is something to hand over) but lives in a throw-away directory.
        let dir = std::env::temp_dir().join(format!("ddai-nav-knowledge-{}", std::process::id()));
        NavConfig {
            memory_dir: Some(dir),
            ..NavConfig::default()
        }
    }

    fn no_memory() -> NavConfig {
        NavConfig {
            memory_dir: None,
            ..NavConfig::default()
        }
    }

    fn px(t: i32) -> f32 {
        (t * 32 + 16) as f32
    }

    #[test]
    fn a_goto_walks_the_tee_to_the_tile_and_hands_the_mode_back() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.place(0, 4, 10);
        f.handle.goto_tile(50, 10);
        let took = f.run_until(1500, |f| f.mode == Mode::Fight && f.tick > 1010);
        assert!(took < 1500, "the walk ended");
        let p = f.own_pos();
        assert!(
            (p.x - px(50)).abs() < 64.0 && (p.y - px(10)).abs() < 64.0,
            "arrived near (50,10): {p:?}"
        );
        let replies = f.replies();
        assert!(replies.iter().any(|r| r.starts_with("goto: ")), "{replies:?}");
        assert_eq!(f.mode, Mode::Fight, "back to the mode it began from");
        assert!(!f.handle.status().walking);
    }

    #[test]
    fn a_goto_that_started_in_hold_returns_to_hold() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.mode = Mode::Hold;
        f.place(0, 4, 10);
        f.handle.goto_tile(20, 10);
        f.tick();
        assert_eq!(f.mode, Mode::Goto);
        f.run_until(1500, |f| f.mode != Mode::Goto);
        assert_eq!(f.mode, Mode::Hold);
    }

    #[test]
    fn a_goto_to_a_wall_a_freeze_or_off_the_map_is_refused_with_a_reason() {
        let mut f = Fx::new(room(40, 12, &[(20, 8, FREEZE)]), no_memory(), 1);
        f.place(0, 4, 10);
        f.handle.goto_tile(0, 5);
        f.handle.goto_tile(20, 8);
        f.handle.goto_tile(99, 5);
        f.tick();
        let r = f.replies();
        assert_eq!(r.len(), 3, "{r:?}");
        assert!(r[0].contains("is a wall"), "{r:?}");
        assert!(r[1].contains("is freeze"), "{r:?}");
        assert!(r[2].contains("is off the map"), "{r:?}");
        assert_eq!(f.mode, Mode::Fight, "a refused goto changes nothing");
    }

    #[test]
    fn stop_calls_a_walk_off_and_a_second_goto_replaces_the_first() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.place(0, 4, 10);
        f.handle.goto_tile(50, 10);
        f.run_until(20, |_| false);
        assert_eq!(f.mode, Mode::Goto);
        f.handle.stop();
        f.tick();
        assert_eq!(f.mode, Mode::Fight);
        assert!(f.replies().iter().any(|r| r.contains("cancelled")));
        f.handle.goto_tile(50, 10);
        f.handle.goto_tile(10, 10);
        f.run_until(1500, |f| f.mode == Mode::Fight && f.tick > 1030);
        let p = f.own_pos();
        assert!(
            (p.x - px(10)).abs() < 64.0,
            "went to the second destination, not the first: {p:?}"
        );
    }

    #[test]
    fn following_a_standing_player_ends_next_to_them() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.place(0, 4, 10);
        f.place(1, 40, 10);
        f.handle.follow(1);
        f.run_until(1500, |f| f.mode == Mode::Fight && f.tick > 1030);
        let (me, other) = (f.own_pos(), f.pw.inner().cores.get(1).expect("other").pos);
        assert!(dist(me, other) <= 64.0 + 8.0, "next to them: {}", dist(me, other));
        let r = f.replies();
        assert!(r.iter().any(|x| x.contains("following them")), "{r:?}");
        assert!(
            !r.iter().any(|x| x.contains("other")),
            "no nickname in any reply: {r:?}"
        );
    }

    #[test]
    fn our_own_kills_of_any_kind_are_not_deaths_on_the_way_for_a_follow() {
        // TS `onKill`: a death of ours counts only when it is more than 50 ticks after our own Cl_Kill
        // (`lastKillTick`), and that includes the unstick's kills.
        let walk_survives = |announce_kill: bool| {
            let mut f = Fx::new(room(80, 12, &[]), no_memory(), 1);
            f.place(0, 4, 10);
            f.place(1, 70, 10);
            f.handle.follow(1);
            f.run_until(5, |_| false);
            for k in 0..3 {
                let t = f.tick + k * 100;
                if announce_kill {
                    f.hooks.navigator.kill_sent(t, false);
                }
                f.hooks.navigator.on_kill(0, 0, t + 2);
            }
            f.run_until(3, |_| false);
            f.handle.status().walking
        };
        assert!(
            walk_survives(true),
            "three deaths right after our own kills: not counted"
        );
        assert!(
            !walk_survives(false),
            "three deaths that were not our kill end the follow"
        );
    }

    #[test]
    fn following_somebody_who_is_not_there_is_refused() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.place(0, 4, 10);
        f.handle.follow(9);
        f.handle.follow(0);
        f.tick();
        let r = f.replies();
        assert!(r[0].contains("nobody with client id 9"), "{r:?}");
        assert!(r[1].contains("the bot itself"), "{r:?}");
        assert_eq!(f.mode, Mode::Fight);
    }

    #[test]
    fn the_first_poll_hands_the_brain_the_map_knowledge_and_the_memory_is_keyed_by_the_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = NavConfig {
            memory_dir: Some(dir.path().to_path_buf()),
            ..NavConfig::default()
        };
        let mut f = Fx::new(room(40, 12, &[]), cfg, 0xab);
        f.place(0, 4, 10);
        f.refresh_tees();
        let own = *f.tees.get(0).unwrap();
        let world = f.pw.inner().clone();
        let ctx = HookContext {
            tick: 1000,
            own: &own,
            tees: &f.tees,
            players: &f.players,
            grid: &f.grid,
            clock: &f.clock,
            world: &world,
            lag_ticks: 0,
            mode: Mode::Fight,
        };
        let poll = f.hooks.navigator.poll(&ctx);
        let k = poll.knowledge.expect("knowledge on the first poll");
        assert_eq!((k.width, k.height), (40, 12));
        let mem = k.freeze_memory.expect("a memory");
        assert_eq!(mem.cells.len(), 40 * 12);
        assert_eq!(mem.events, 0);
        // A freeze is noted, and stop() saves it under the hash.
        let mut frozen = own;
        frozen.frozen = true;
        let ctx2 = HookContext {
            own: &frozen,
            tick: 1001,
            ..ctx
        };
        f.hooks.navigator.poll(&ctx2);
        f.hooks.navigator.stop();
        let file = dir.path().join(format!("{}.json", "ab".repeat(32)));
        assert!(
            file.exists(),
            "memory file keyed by the sha256: {:?}",
            std::fs::read_dir(dir.path()).map(|d| d.count())
        );
        let text = std::fs::read_to_string(&file).expect("read");
        assert!(
            text.contains("\"events\":1") || text.contains("\"events\": 1"),
            "{text}"
        );
    }

    #[test]
    fn map_knowledge_goes_to_the_brain_at_the_map_load_and_when_we_freeze_never_per_tile() {
        let mut f = Fx::new(room(40, 12, &[]), no_memory_dir_for_knowledge(), 5);
        f.place(0, 4, 10);
        f.refresh_tees();
        let world = f.pw.inner().clone();
        let mut own = *f.tees.get(0).unwrap();
        let mut knowledge_at = Vec::new();
        for (i, (x, frozen)) in [
            (4, false),
            (5, false),
            (6, false),
            (7, true),
            (7, true),
            (8, false),
            (9, false),
        ]
        .into_iter()
        .enumerate()
        {
            own.pos = Vec2::new(px(x), px(10));
            own.frozen = frozen;
            let ctx = HookContext {
                tick: 1000 + i as i32,
                own: &own,
                tees: &f.tees,
                players: &f.players,
                grid: &f.grid,
                clock: &f.clock,
                world: &world,
                lag_ticks: 0,
                mode: Mode::Fight,
            };
            if let Some(k) = f.hooks.navigator.poll(&ctx).knowledge {
                knowledge_at.push((i, k));
            }
        }
        let at: Vec<usize> = knowledge_at.iter().map(|(i, _)| *i).collect();
        assert_eq!(
            at,
            vec![0, 3],
            "the first poll (the map load) and the tick the tee froze, nothing per tile"
        );
        // The freeze snapshot carries what was noted so far and is its own copy.
        let m = knowledge_at[1].1.freeze_memory.as_ref().expect("memory");
        assert_eq!(m.events, 1);
        assert!(m.passes.iter().any(|&v| v > 0.0), "the tiles walked so far");
    }

    /// Review F4: the cost of the map-knowledge hand-off on a ChillBlock5-size grid (943 x 1075 = 1.01M
    /// tiles). `cargo test --release -p ddai-bot --lib knowledge_handoff -- --ignored --nocapture`.
    #[test]
    #[ignore = "a timing measurement, prints its result"]
    fn knowledge_handoff_cost_on_a_million_tile_map() {
        let mut f = Fx::new(room(943, 1075, &[]), no_memory_dir_for_knowledge(), 9);
        f.place(0, 10, 1073);
        f.refresh_tees();
        let world = f.pw.inner().clone();
        let mut own = *f.tees.get(0).unwrap();
        let mut steady = Vec::new();
        let mut handoff = Vec::new();
        let mut tick = 1000;
        for round in 0..30 {
            // 60 polls while walking over new tiles (no hand-off may happen), then one freeze.
            for k in 0..60 {
                own.frozen = false;
                own.pos = Vec2::new(px(10 + round * 3 + k % 7), px(1073));
                tick += 1;
                let ctx = HookContext {
                    tick,
                    own: &own,
                    tees: &f.tees,
                    players: &f.players,
                    grid: &f.grid,
                    clock: &f.clock,
                    world: &world,
                    lag_ticks: 0,
                    mode: Mode::Fight,
                };
                let t0 = std::time::Instant::now();
                let p = f.hooks.navigator.poll(&ctx);
                let d = t0.elapsed();
                if round > 0 || k > 0 {
                    assert!(p.knowledge.is_none(), "no hand-off per tile");
                    steady.push(d);
                }
            }
            own.frozen = true;
            tick += 1;
            let ctx = HookContext {
                tick,
                own: &own,
                tees: &f.tees,
                players: &f.players,
                grid: &f.grid,
                clock: &f.clock,
                world: &world,
                lag_ticks: 0,
                mode: Mode::Fight,
            };
            let t0 = std::time::Instant::now();
            let p = f.hooks.navigator.poll(&ctx);
            handoff.push(t0.elapsed());
            assert!(p.knowledge.is_some(), "a hand-off at the freeze");
        }
        steady.sort();
        handoff.sort();
        let q = |v: &[std::time::Duration], p: f64| v[((v.len() - 1) as f64 * p) as usize];
        eprintln!(
            "poll per tick (steady): p50 {:?} p99 {:?} max {:?}; hand-off at a freeze: p50 {:?} max {:?}",
            q(&steady, 0.5),
            q(&steady, 0.99),
            steady.last().unwrap(),
            q(&handoff, 0.5),
            handoff.last().unwrap()
        );
    }

    #[test]
    fn a_map_change_forgets_the_walk_and_a_home_on_another_map() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.place(0, 4, 10);
        f.handle.send(NavCommand::SetHome { tx: 30, ty: 10 });
        f.handle.goto_tile(50, 10);
        f.run_until(10, |_| false);
        assert_eq!(f.mode, Mode::Goto);
        f.hooks.navigator.on_map_changing();
        f.hooks.navigator.on_map(
            &f.map,
            &MapIdent {
                name: "another room".to_string(),
                sha256: [2; 32],
            },
        );
        f.tick();
        assert_eq!(f.mode, Mode::Fight, "the walk ended with the map");
        assert!(f.handle.status().home.is_none(), "the home is forgotten on another map");
    }

    #[test]
    fn an_unknown_map_has_no_wayblock_and_wb_commands_say_so() {
        let mut f = Fx::new(room(60, 12, &[]), no_memory(), 1);
        f.place(0, 4, 10);
        f.handle.send(NavCommand::Wb(WbMode::Left));
        f.tick();
        let r = f.replies();
        assert!(r[0].contains("this map has none"), "{r:?}");
        assert!(!f.hooks.wayblock.holding());
    }

    // ---- Copy Love Box (needs the map file; skipped when it is not there) -----------------------------

    fn clb_map() -> Option<MapData> {
        let path = PathBuf::from(std::env::var("HOME").ok()?)
            .join("aiddnet/data/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map");
        Some(ddai_map::load_map(&std::fs::read(path).ok()?).ok()?.data)
    }

    fn clb_fx() -> Option<Fx> {
        let map = Arc::new(clb_map()?);
        let handle = NavHandle::new();
        let mut hooks = nav_hooks(no_memory(), handle.clone());
        hooks.navigator.on_map(
            &map,
            &MapIdent {
                name: "Copy Love Box".to_string(),
                sha256: [3; 32],
            },
        );
        let mut players = PlayerTable::new(SALT);
        let views = [
            player(0, "bot", "", true, 0, None),
            player(1, "a", "", false, 0, None),
            player(2, "b", "", false, 0, None),
        ];
        players.update(&views, &Relations::new());
        Some(Fx {
            pw: PhysicsWorld::new(Arc::clone(&map), 1),
            grid: MapGrid::new(&map),
            map,
            hooks,
            handle,
            tees: TeeSet::new(),
            players,
            clock: ActivityClock::new(),
            mode: Mode::Fight,
            tick: 1000,
            prev: empty_input(),
            kills: 0,
        })
    }

    fn tee_at(id: i32, tx: i32, ty: i32) -> Tee {
        Tee {
            id,
            alive: true,
            pos: Vec2::new(px(tx), px(ty)),
            ..Tee::DEAD
        }
    }

    #[test]
    fn on_copy_love_box_the_bot_holds_the_wb_and_filters_candidates_by_the_leash() {
        let Some(mut f) = clb_fx() else {
            eprintln!("skipping: the Copy Love Box map is not present");
            return;
        };
        let def = ddai_nav::wayblock::wayblocks().into_iter().next().expect("CLB");
        let spot = def.left.spots[0];
        f.tees.set_for_test(tee_at(0, spot.0, spot.1));
        let leash_tile = {
            let b = def.left.leash[0];
            (b.x0 + 1, b.y0 + 1)
        };
        f.tees.set_for_test(tee_at(1, leash_tile.0, leash_tile.1));
        // Far from every hall, not attacking us.
        f.tees.set_for_test(tee_at(2, 300, 200));
        let own = *f.tees.get(0).unwrap();
        let world = f.pw.inner().clone();
        let ctx = HookContext {
            tick: 1000,
            own: &own,
            tees: &f.tees,
            players: &f.players,
            grid: &f.grid,
            clock: &f.clock,
            world: &world,
            lag_ticks: 0,
            mode: Mode::Fight,
        };
        f.hooks.navigator.poll(&ctx);
        assert!(
            f.hooks.wayblock.holding(),
            "the WB is held in fight mode on Copy Love Box"
        );
        let near = *f.tees.get(1).unwrap();
        let far = *f.tees.get(2).unwrap();
        let a = f.hooks.wayblock.filter(&ctx, &near);
        assert!(
            !a.skip && a.in_zone == def.left.zone.iter().any(|b| b.contains(leash_tile.0, leash_tile.1)),
            "{a:?}"
        );
        let b = f.hooks.wayblock.filter(&ctx, &far);
        assert!(
            b.skip,
            "a tee outside the leash that is not at us is not ours to chase from the hall: {b:?}"
        );
        // The brain is told the hall's overrides and band while we stand in it.
        let hints = f.hooks.wayblock.brain_hints(&ctx);
        assert!(hints.in_hall && hints.band.is_some(), "{hints:?}");
        // ...and not when the WB is off.
        f.handle.send(NavCommand::Wb(WbMode::Off));
        f.hooks.navigator.poll(&ctx);
        assert!(!f.hooks.wayblock.holding());
        assert_eq!(f.hooks.wayblock.brain_hints(&ctx), WbHints::default());
    }

    #[test]
    fn lying_frozen_in_the_tiles_outside_the_zone_asks_for_a_kill_after_25_ticks_only() {
        let Some(mut f) = clb_fx() else {
            eprintln!("skipping: the Copy Love Box map is not present");
            return;
        };
        let def = ddai_nav::wayblock::wayblocks().into_iter().next().expect("CLB");
        // A freeze tile of the map that lies outside both zones.
        let (tx, ty) = (0..f.grid.width())
            .flat_map(|x| (0..f.grid.height()).map(move |y| (x, y)))
            .find(|&(x, y)| {
                f.grid.is_freeze(px(x), px(y))
                    && !def
                        .left
                        .zone
                        .iter()
                        .chain(def.right.zone.iter())
                        .any(|b| b.contains(x, y))
            })
            .expect("a freeze tile outside the zones");
        let mut own = tee_at(0, tx, ty);
        own.frozen = true;
        f.tees.set_for_test(own);
        let world = f.pw.inner().clone();
        let ctx = HookContext {
            tick: 1000,
            own: &own,
            tees: &f.tees,
            players: &f.players,
            grid: &f.grid,
            clock: &f.clock,
            world: &world,
            lag_ticks: 0,
            mode: Mode::Fight,
        };
        f.hooks.navigator.poll(&ctx);
        assert!(!f.hooks.wayblock.wants_kill(&ctx, WB_LYING_TICKS - 1), "not yet");
        assert!(
            f.hooks.wayblock.wants_kill(&ctx, WB_LYING_TICKS),
            "25 ticks lying in the freeze tiles"
        );
        // A rope on us, or moving, or inside a zone: no.
        let mut moving = own;
        moving.vel = Vec2::new(2.0, 0.0);
        assert!(!f.hooks.wayblock.wants_kill(&HookContext { own: &moving, ..ctx }, 100));
        let zone = def.left.zone[0];
        let in_zone = Tee {
            frozen: true,
            ..tee_at(0, zone.x0 + 1, zone.y0 + 1)
        };
        assert!(!f.hooks.wayblock.wants_kill(&HookContext { own: &in_zone, ..ctx }, 100));
    }
}
