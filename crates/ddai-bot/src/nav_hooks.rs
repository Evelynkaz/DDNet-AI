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
use ddai_nav::wayblock::{WB_NO_CLIMB_TILES, WB_RETURN_TICKS, WbState, on_wb_spot, wayblock_for, wb_walk_allowed};
use ddai_physics::map::MapData;
use ddai_physics::vmath::Vec2;
use ddai_planner::brains::action_from_input;
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::{PlanCollision, PlanWorld};
use ddai_planner::types::TeeState;
use ddai_planner::vmath::Vec2 as Vec2d;

use crate::bot::Mode;
use crate::consts::*;
use crate::hooks::{
    HookContext, Hooks, MapIdent, NavClipState, NavStep, Navigator, Poll, Trek, WanderHint, WayBlock, WbFilter,
};
use crate::mapgrid::MapGrid;
use crate::reach::{RouteFinder, Tile};
use crate::tees::{HOOK_FLYING, Tee, dist};

mod in_the_way;
mod wb_extra;

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
    /// Task 3.12 (`--wb-smart`, opt-in, off by default; D-103): on Copy Love Box the hall is chosen by the number of **blockable** targets on
    /// each side (the more the better, a tie at random, with hysteresis and a memory of failed tube crossings), and an idle (AFK) tee is
    /// fought when he is in the way (`in_the_way`).
    pub wb_smart: bool,
    /// The seed of the side choice's tie-break (`wb_smart`). `None`: a fresh one per process (the live default: another side each
    /// session); tests pass one.
    pub seed: Option<u64>,
}

impl Default for NavConfig {
    fn default() -> Self {
        NavConfig {
            memory_dir: default_memory_dir(),
            wb_mode: WbMode::Auto,
            strong: false,
            seek: true,
            cross_budget_ms: 10.0,
            wb_smart: false,
            seed: None,
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
    /// `!strong on|off`: inside a wayblock hall the planner searches wider (`STRONG_WB`).
    Strong(bool),
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
    /// `--wb-smart` (3.12b, review F12): the life has just begun (a join, a respawn) and the walk to the WB has not been tried yet: that
    /// first walk starts at once ([`Core::wb_return_ticks`]); any later one waits [`WB_RETURN_TICKS`] again.
    fresh_life: bool,
    /// Walks to the WB begun from the idle branch of the steering (tests).
    wb_walk_tries: u32,
    /// The target the steering last saw and when (`-1` none): the fight test of `--wb-smart` ([`Core::fighting_here`]).
    target_seen: (i32, i32),
    travel_since: i64,
    dull_since: i64,
    kill_wanted: bool,
    /// `--no-selfkill` (D-102): no route or trek with a respawn (kill) step, no kill asked for.
    no_selfkill: bool,
    /// The last `crossing()` / `planned_freeze()` of the running walk, for the clip frame.
    clip_crossing: bool,
    clip_planned: bool,
    /// A cross-fail note waiting for the bot (`Navigator::take_cross_fail`).
    cross_fail: Option<String>,
    walks_ended: u32,
    last_walk: String,
    nav_kills: u32,
    last_tile_pos: Option<(i32, i32)>,
    last_kill_tick: i32,
    route_kill_tick: i32,
    last_status_tick: i32,
    seek_enabled: bool,
    /// The foe of the WB walk, route 2, the guard (`wb_extra`).
    x: wb_extra::WbExtra,
    wb_route2_on: bool,
    wb_route2_crowd_on: bool,
}

impl Core {
    fn new(cfg: NavConfig, handle: NavHandle) -> Core {
        let mut wb = WbState::new(None, cfg.wb_mode);
        if cfg.wb_smart {
            let seed = cfg.seed.unwrap_or_else(|| {
                let salt = crate::players::random_salt();
                u64::from_le_bytes(salt[..8].try_into().expect("a salt of at least 8 bytes"))
            });
            wb.chooser.set_smart(true, seed);
        }
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
            fresh_life: true,
            wb_walk_tries: 0,
            target_seen: (-1, i32::MIN / 2),
            travel_since: i64::MIN / 2,
            dull_since: -1,
            kill_wanted: false,
            no_selfkill: false,
            clip_crossing: false,
            clip_planned: false,
            cross_fail: None,
            walks_ended: 0,
            last_walk: String::new(),
            nav_kills: 0,
            last_tile_pos: None,
            last_kill_tick: i32::MIN / 2,
            route_kill_tick: i32::MIN / 2,
            last_status_tick: i32::MIN / 2,
            seek_enabled,
            x: wb_extra::WbExtra::default(),
            wb_route2_on: wb_extra::route2_on(),
            wb_route2_crowd_on: wb_extra::route2_crowd_on(),
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
        // The same map again (its name and size): the deaths on the way and the pause of the WB stay.
        let key = format!("{}|{}x{}", ident.name, width, height);
        let same = key == self.x.pause_key;
        self.x.reset();
        self.x.pause_key = key;
        if let Some(d) = &def {
            let named = d.name.to_lowercase() == ident.name.trim().to_lowercase();
            self.log(&format!(
                "WB: this map has one{}; {}",
                if named { String::new() } else { format!(" ({})", d.name) },
                "holding it when there is nobody to fight (!wb off to stop)"
            ));
        } else if ddai_nav::wayblock::has_wayblock_named(&ident.name) {
            self.log(&format!(
                "WB: '{}' here is another version of the map the WB was measured on (no hall like it in its tiles, or a spot, a rope anchor of the tube, the swing's start or the passage's exit differs); not holding it",
                ident.name
            ));
        }
        self.wb.on_map_keeping(def, same);
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
        let mut nav = TsNav::new(goals, opts);
        nav.set_allow_kill(!self.no_selfkill);
        self.nav = Some(nav);
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
                        "{gave}WB: {} (this map has none; it applies on Copy Love Box and its copies with the same hall)",
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
            NavCommand::Strong(on) => {
                self.cfg.strong = on;
                self.reply(if on {
                    "strong mode: on (inside a wayblock hall the planner searches wider: more CPU)".to_string()
                } else {
                    "strong mode: off (the plain search)".to_string()
                });
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

    fn engaged_now(ctx: &HookContext<'_>, target: i32) -> bool {
        let me = ctx.own;
        if me.frozen || me.hooked_player >= 0 {
            return true;
        }
        ctx.tees.iter().any(|t| {
            t.id != me.id
                && (t.hooked_player == me.id
                    || (!t.frozen
                        && dist(me.pos, t.pos) < ENGAGED_PX
                        // A paused or spectating tee we are fighting (the target) counts as engaged too.
                        && (!ctx.clock.afk(t.id, ctx.tick, ctx.players, false)
                            || (t.id == target && ctx.players.get(t.id).is_some_and(|s| s.not_playing())))))
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
            self.wb.fight_here = self.cfg.wb_smart && self.fighting_here(ctx);
            let change = self.wb.update_side(
                tile,
                f64::from(own.pos.x) / 32.0,
                own_free,
                counts,
                i64::from(tick),
                holding,
            );
            if let Some(c) = change {
                let what = if self.cfg.wb_smart {
                    "blockable targets"
                } else {
                    "playing"
                };
                self.log(&format!(
                    "WB: over to the {} ({} {what} on the left, {} on the right)",
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
            // Who froze us on the way (three times: a grudge), then the foe of the walk.
            for by in std::mem::take(&mut self.x.blocked_by) {
                self.note_wb_freeze(ctx, by);
            }
            self.update_wb_foe(ctx);
            if (self.wb.walk_fails() > 0 || self.x.route2_crowd)
                && !own.frozen
                && self.wb.def.as_ref().is_some_and(|d| {
                    d.in_hall(ddai_nav::wayblock::WbSide::Left, tile.0, tile.1)
                        || d.in_hall(ddai_nav::wayblock::WbSide::Right, tile.0, tile.1)
                })
            {
                self.x.route2_crowd = false;
            }
            self.wb.arrived_in_hall(tile, own.frozen);
            self.update_wb_route2(ctx);
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
        if self.cfg.wb_smart {
            return self.wb_blockable_counts(ctx);
        }
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
            "WB: {why}; {}: {} on the left, {} on the right",
            if self.cfg.wb_smart {
                "blockable targets"
            } else {
                "playing"
            },
            self.wb.counts.0,
            self.wb.counts.1
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
            && (Self::someone_worth_fighting(ctx, SEEK_ARRIVED_PX) || self.afk_blocks_the_walk(ctx))
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
        nav.smart = ddai_nav::crossing::CrossSmart {
            others: self.cfg.wb_smart,
            unblock: self.cfg.wb_smart,
        };
        let want = {
            let mut nctx = NavCtx {
                col: world.collision(),
                router,
                make_sim: &mut make,
            };
            nav.step(&mut nctx, &me, tick, &others, i64::from(ctx.lag_ticks))
        };
        let (crossing, planned) = (nav.crossing(), nav.planned_freeze());
        let guard = !(crossing || planned);
        let notes = nav.take_notes();
        let kill = nav.take_kill();
        let done = nav.done();
        self.clip_crossing = crossing;
        self.clip_planned = planned;
        for n in notes {
            self.log(&format!("goto: {n}"));
            if ddai_clip::store::is_cross_fail_note(&n) {
                if self.cfg.wb_smart
                    && let Some(def) = &self.wb.def
                    && let Some(side) = [ddai_nav::wayblock::WbSide::Left, ddai_nav::wayblock::WbSide::Right]
                        .into_iter()
                        .find(|&s| n.starts_with(&def.side(s).crossing.label))
                {
                    self.wb.chooser.note_cross_fail(side, tick);
                }
                self.cross_fail = Some(n);
            }
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
                let mut nav = TsNav::new(vec![Self::follow_goal(g.0, g.1, &format!("c{id}"))], opts);
                nav.set_allow_kill(!self.no_selfkill);
                self.nav = Some(nav);
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
        let allow_kill = !self.no_selfkill;
        let Some(ms) = &mut self.ms else {
            return "no map".to_string();
        };
        let from = Vec2d {
            x: f64::from(ctx.own.pos.x),
            y: f64::from(ctx.own.pos.y),
        };
        match TsTrek::start_with(
            &mut ms.router,
            ms.world.collision(),
            from,
            to,
            &self.trek_avoid,
            i64::from(ctx.tick),
            allow_kill,
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
        if target != -1 {
            self.target_seen = (target, ctx.tick);
        }
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
            && !Self::engaged_now(ctx, target)
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
        } else if holding && self.nav.is_none() && tick - self.idle_since > self.wb_return_ticks(ctx) {
            self.wb_walk_tries += 1;
            self.fresh_life = false;
            self.walk_to_wb(ctx);
            self.idle_since = tick;
        }
    }

    /// "nobody in reach; walk to where the game is" (`bot.ts:2548-2564`).
    fn walk_to_game(&mut self, ctx: &HookContext<'_>, tick: i64) {
        let Some(spot) = self.game_spot(ctx) else { return };
        self.travel_since = tick;
        let (tx, ty) = ((spot.x / 32.0).trunc() as i32, (spot.y / 32.0).trunc() as i32);
        let allow_kill = !self.no_selfkill;
        let Some(ms) = &mut self.ms else { return };
        let way = ms.router.find_route(
            (f64::from(ctx.own.pos.x), f64::from(ctx.own.pos.y)),
            (spot.x, spot.y),
            &RouteOpts {
                near_tiles: 3,
                partial: false,
                allow_kill,
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

    /// How long the tee idles outside the spot before it walks back to it: [`WB_RETURN_TICKS`] (TS). Under `--wb-smart` a tee that is
    /// not in the hall goes at once (task 3.12b), **for the first walk of a life only** (review F12): it has just respawned, and the spawns
    /// of Copy Love Box hang over the freeze chamber or stand at the edge of its ledge. A walk that ends without arriving (no route, a
    /// route that needs a kill under `--no-selfkill`) is not restarted every tick: the second try waits the full second again.
    fn wb_return_ticks(&self, ctx: &HookContext<'_>) -> i64 {
        if self.cfg.wb_smart
            && self.fresh_life
            && let (Some(def), Some(side)) = (&self.wb.def, self.wb.side())
        {
            let (tx, ty) = tile_of(ctx.own.pos);
            if !def.in_hall(side, tx, ty) {
                return 0;
            }
        }
        WB_RETURN_TICKS
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
        let first = self.wb_spot_for(ctx, &def, side, (tx, ty));
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
        let kill_ready = !self.no_selfkill && ctx.tick - self.last_kill_tick >= KILL_COOLDOWN_TICKS;
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
        self.wb_filter_guard(ctx, cand)
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

    fn wb_wander_hint(&mut self, ctx: &HookContext<'_>) -> Option<WanderHint> {
        let (def, side) = (self.wb.def.clone()?, self.wb.side()?);
        let (tx, ty) = tile_of(ctx.own.pos);
        if !def.in_hall(side, tx, ty) {
            // Task 3.12b (`--wb-smart`): outside the hall with the WB held (just respawned in the top room, the walk not begun) the tee
            // stands where it is. The free wander walks off the ledge of the tube's start into the freeze chamber (a drop it takes
            // with probability 0.3): 6 of the 34 failed crossings of the 2026-10-06 session ended that way.
            if self.cfg.wb_smart && self.wb_holding() {
                return Some(WanderHint {
                    anchor_x: ctx.own.pos.x,
                    look_at: None,
                    still: true,
                });
            }
            return None;
        }
        let spot = self.wb_spot_for(ctx, &def, side, (tx, ty));
        let watch = def.side(side).watch;
        // The guard on its spot stands still there (`wander(…, still)`).
        let still = ddai_nav::wayblock::wb_guard() && !self.role_lower(ctx);
        Some(WanderHint {
            anchor_x: (spot.0 * 32 + 16) as f32,
            look_at: Some(((watch.0 * 32 + 16) as f32, (watch.1 * 32 + 16) as f32)),
            still,
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
    fn set_no_selfkill(&mut self, off: bool) {
        let mut c = self.0.borrow_mut();
        c.no_selfkill = off;
        if let Some(n) = &mut c.nav {
            n.set_allow_kill(!off);
        }
    }
    fn on_map(&mut self, map: &Arc<MapData>, ident: &MapIdent) {
        self.0.borrow_mut().on_map(map, ident);
    }
    fn on_map_changing(&mut self) {
        let mut c = self.0.borrow_mut();
        c.save_memory();
        c.drop_walk();
        c.trek = None;
        c.ms = None;
        c.x.reset();
        c.fresh_life = true;
    }
    fn respawned(&mut self) {
        let mut c = self.0.borrow_mut();
        if let Some(n) = &mut c.nav {
            n.respawned();
        }
        c.fresh_life = true;
        // A new try: route 2 is said (and the crowd judged) again.
        c.x.route2_crowd = false;
        c.x.route2_said = false;
    }
    fn blocked_by(&mut self, by: i32, _tick: i32) {
        let mut c = self.0.borrow_mut();
        if c.x.blocked_by.len() < 16 {
            c.x.blocked_by.push(by);
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
    fn clip_state(&self, label: &mut String) -> NavClipState {
        use std::fmt::Write;
        label.clear();
        let c = self.0.borrow();
        let Some(n) = &c.nav else {
            return NavClipState::default();
        };
        if let Some(f) = &c.follow {
            let _ = write!(label, "follow c{}", f.id);
        } else if let Some(g) = n.goal() {
            label.push_str(&g.label);
        } else {
            label.push_str("walk");
        }
        NavClipState {
            walking: true,
            crossing: c.clip_crossing,
            planned_freeze: c.clip_planned,
        }
    }
    fn take_cross_fail(&mut self) -> Option<String> {
        self.0.borrow_mut().cross_fail.take()
    }
    fn resend_knowledge(&mut self) {
        self.0.borrow_mut().knowledge_due = true;
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
        self.0.borrow_mut().wb_wander_hint(ctx)
    }
    fn begin_pick(&mut self, ctx: &HookContext<'_>, target: i32, sealed: &mut dyn FnMut(&Tee) -> bool) {
        self.0.borrow_mut().begin_pick(ctx, target, sealed);
    }
    fn walk_allowed(&self, tx: i32, ty: i32) -> bool {
        wb_walk_allowed(self.0.borrow().wb.def.as_ref(), tx, ty)
    }
    fn foe_target(&mut self) -> Option<i32> {
        self.0.borrow().x.foe.as_ref().map(|f| f.id)
    }
    fn afk_in_the_way(&mut self, ctx: &HookContext<'_>, candidate: &Tee, current: bool) -> bool {
        self.0.borrow().afk_in_the_way(ctx, candidate, current).is_some()
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
    use crate::mapgrid::test_maps::{FREEZE, SOLID, room};
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
                fixed_target: false,
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
            fixed_target: false,
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
                fixed_target: false,
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
                    fixed_target: false,
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
                fixed_target: false,
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
            fixed_target: false,
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
            fixed_target: false,
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

    // ---- the WB update of af49dfb: route 2, the foe of the walk, the guard ---------------------------

    /// Runs a test body on a thread with room for the worlds (several copies of a `World` are big).
    fn on_big_stack(body: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(body)
            .expect("thread")
            .join()
            .expect("the test body");
    }

    /// A bare `Core` on a map, for the pieces the hooks only expose through a whole bot.
    struct CoreFx {
        core: Core,
        env: CoreEnv,
    }

    /// What the hooks look at (kept apart from the core so a context can be held while the core changes).
    struct CoreEnv {
        map: Arc<MapData>,
        pw: PhysicsWorld,
        grid: MapGrid,
        players: PlayerTable,
        clock: ActivityClock,
        tees: TeeSet,
        mode: Mode,
    }

    impl CoreFx {
        fn on(map: MapData, name: &str, relations: &Relations) -> CoreFx {
            Self::with(no_memory(), map, name, relations)
        }

        fn with(cfg: NavConfig, map: MapData, name: &str, relations: &Relations) -> CoreFx {
            let map = Arc::new(map);
            let mut core = Core::new(cfg, NavHandle::new());
            core.on_map(
                &map,
                &MapIdent {
                    name: name.to_string(),
                    sha256: [9; 32],
                },
            );
            core.mode = Mode::Fight;
            let mut players = PlayerTable::new(SALT);
            let views = [
                player(0, "bot", "", true, 0, None),
                player(1, "foe", "", false, 0, None),
                player(2, "pal", "", false, 0, None),
                player(3, "p3", "", false, 0, None),
                player(4, "p4", "", false, 0, None),
                player(5, "p5", "", false, 0, None),
            ];
            players.update(&views, relations);
            CoreFx {
                core,
                env: CoreEnv {
                    pw: PhysicsWorld::new(Arc::clone(&map), 1),
                    grid: MapGrid::new(&map),
                    map,
                    players,
                    clock: ActivityClock::new(),
                    tees: TeeSet::new(),
                    mode: Mode::Fight,
                },
            }
        }
    }

    impl CoreEnv {
        fn ctx<'a>(&'a self, own: &'a Tee, tick: i32, world: &'a ddai_physics::world::World<f32>) -> HookContext<'a> {
            HookContext {
                tick,
                own,
                tees: &self.tees,
                players: &self.players,
                grid: &self.grid,
                clock: &self.clock,
                world,
                lag_ticks: 0,
                mode: self.mode,
                fixed_target: false,
            }
        }
    }

    #[test]
    fn route_2_comes_after_two_failed_tries_and_not_before_or_without_a_walk() {
        on_big_stack(|| {
            use super::wb_extra::wb_route2_why;
            assert_eq!(wb_route2_why(0, 0, false), None);
            assert_eq!(wb_route2_why(1, 1, false), None);
            assert_eq!(
                wb_route2_why(2, 0, false).as_deref(),
                Some("the last 2 tries to get in failed")
            );
            assert_eq!(
                wb_route2_why(0, 3, false).as_deref(),
                Some("the last 3 tries to get in failed")
            );
            assert_eq!(wb_route2_why(0, 0, true).as_deref(), Some("a crowd at the tube"));
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            let mut f = CoreFx::on(map, "Copy Love Box", &Relations::new());
            let spawn = ddai_nav::route::spawn_tiles(&f.env.map)[0];
            let (tx, ty) = ((spawn.0 / 32.0) as i32, (spawn.1 / 32.0) as i32);
            f.env.tees.set_for_test(tee_at(0, tx, ty));
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 1000, &world);
            f.core.poll(&ctx);
            f.core.walk_to_wb(&ctx);
            assert!(f.core.wb_walk && f.core.nav.is_some(), "the walk to the WB runs");
            f.core.update_wb_route2(&ctx);
            assert!(!f.core.nav.as_ref().unwrap().wall_route, "route 1 first");
            f.core.wb.note_walk_death(0);
            f.core.wb.note_walk_death(0);
            f.core.update_wb_route2(&ctx);
            assert!(
                f.core.nav.as_ref().unwrap().wall_route,
                "two deaths on the way: route 2"
            );
            // Not on a walk that is not the WB's.
            f.core.wb_walk = false;
            f.core.update_wb_route2(&ctx);
            assert!(!f.core.nav.as_ref().unwrap().wall_route);
        });
    }

    #[test]
    fn outside_the_leash_only_the_war_list_is_fought_and_inside_the_guard_skips_the_ones_falling_past() {
        on_big_stack(|| {
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            let mut rel = Relations::new();
            rel.add(crate::relations::ListKind::War, "pal");
            let mut f = CoreFx::on(map, "Copy Love Box", &rel);
            let def = ddai_nav::wayblock::wayblocks().remove(0);
            let spawn = ddai_nav::route::spawn_tiles(&f.env.map)[0];
            let (sx, sy) = ((spawn.0 / 32.0) as i32, (spawn.1 / 32.0) as i32);
            // On the way in (at the spawn): nobody but the war list, even one that hooks us.
            let mut hooker = tee_at(1, sx + 3, sy);
            hooker.hooked_player = 0;
            f.env.tees.set_for_test(tee_at(0, sx, sy));
            f.env.tees.set_for_test(hooker);
            f.env.tees.set_for_test(tee_at(2, sx + 4, sy));
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 1000, &world);
            f.core.poll(&ctx);
            assert!(f.core.wb_holding());
            let wb = |c: &CoreFx, t: &Tee| c.core.wb_filter(&c.env.ctx(&own, 1000, &world), t);
            assert!(
                wb(&f, f.env.tees.get(1).unwrap()).skip,
                "a tee that hooks us on the way is not fought (af49dfb)"
            );
            assert!(!wb(&f, f.env.tees.get(2).unwrap()).skip, "the war list is");
            // In the hall: one that falls past the hall's zone is no target yet; one in it is.
            let side = f.core.wb.side().expect("a side");
            let zone = def.side(side).zone[0];
            let spot = def.side(side).spots[0];
            let mut f2 = CoreFx::on(clb_map().unwrap(), "Copy Love Box", &Relations::new());
            let falling_outside = {
                let b = def.side(side).approach[0];
                let mut t = tee_at(1, b.x0 + 1, b.y0 + 1);
                t.vel.y = 6.0;
                t
            };
            let standing_in = tee_at(2, zone.x0 + 3, zone.y0 + 2);
            f2.env.tees.set_for_test(tee_at(0, spot.0, spot.1));
            f2.env.tees.set_for_test(falling_outside);
            f2.env.tees.set_for_test(standing_in);
            let own2 = *f2.env.tees.get(0).unwrap();
            let world2 = f2.env.pw.inner().clone();
            let ctx2 = f2.env.ctx(&own2, 1000, &world2);
            f2.core.poll(&ctx2);
            f2.core.wb.chooser.adopt(side);
            let mut sealed_calls = 0;
            f2.core.begin_pick(&ctx2, -1, &mut |_| {
                sealed_calls += 1;
                false
            });
            let in_zone = f2.core.wb_filter(&ctx2, f2.env.tees.get(2).unwrap());
            assert!(!in_zone.skip && in_zone.in_zone, "{in_zone:?}");
            let fall = f2.core.wb_filter(&ctx2, f2.env.tees.get(1).unwrap());
            assert!(
                fall.skip || !fall.in_zone,
                "a tee falling past the approach is not a target for the guard: {fall:?}"
            );
        });
    }

    #[test]
    fn a_grudge_is_three_freezes_on_the_way_and_the_walk_stops_for_him() {
        on_big_stack(|| {
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            let mut f = CoreFx::on(map, "Copy Love Box", &Relations::new());
            // The start of the left tube: standable, outside the hall's zones, on the way in.
            let (sx, sy) = ddai_nav::wayblock::wayblocks().remove(0).left.crossing.start;
            let foe_x = (1..8)
                .map(|d| sx + d)
                .find(|&x| {
                    !f.env.grid.is_solid(px(x), px(sy)) && !f.env.grid.is_solid(px(x), px(sy) + 32.0 - 16.0 + 3.0)
                })
                .expect("an open tile next to the start");
            f.env.tees.set_for_test(tee_at(0, sx, sy));
            f.env.tees.set_for_test(tee_at(1, foe_x, sy));
            // Active (its aim changes), so it is not AFK.
            f.env.clock.update(990, &f.env.tees, &f.env.players, 0);
            let mut moved = tee_at(1, foe_x, sy);
            moved.angle = 100;
            f.env.tees.set_for_test(moved);
            f.env.clock.update(995, &f.env.tees, &f.env.players, 0);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 1000, &world);
            f.core.poll(&ctx);
            f.core.walk_to_wb(&ctx);
            assert!(f.core.wb_walk);
            // Two freezes are not enough; the third (inside three minutes) is.
            f.core.note_wb_freeze(&ctx, 1);
            let ctx = f.env.ctx(&own, 1100, &world);
            f.core.note_wb_freeze(&ctx, 1);
            f.core.update_wb_foe(&ctx);
            assert!(f.core.x.foe.is_none(), "two freezes");
            let ctx = f.env.ctx(&own, 1200, &world);
            f.core.note_wb_freeze(&ctx, 1);
            f.core.update_wb_foe(&ctx);
            let foe = f.core.x.foe.as_ref().expect("a grudge");
            assert!(foe.grudge && foe.id == 1);
            assert!(f.core.nav.is_none(), "the walk is called off while he is dealt with");
        });
    }

    #[test]
    fn the_guard_stands_at_the_job_spot_over_a_frozen_tee_on_the_lower_shelf_and_the_hall_is_found_by_its_tiles() {
        on_big_stack(|| {
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            // Another name: the hall is found by its tiles.
            let mut f = CoreFx::on(map, "Some Other Name", &Relations::new());
            let def = f
                .core
                .wb
                .def
                .clone()
                .expect("the WB of a map with the hall, whatever its name");
            assert_eq!(def.name, "Copy Love Box hall at +0,+0");
            let side = ddai_nav::wayblock::WbSide::Left;
            let g = def.guard_geom(side);
            let first = def.side(side).spots[0];
            f.env.tees.set_for_test(tee_at(0, first.0, first.1));
            let mut frozen = tee_at(1, g.shelf.x0 + 3, g.shelf.y0 + 1);
            frozen.frozen = true;
            f.env.tees.set_for_test(frozen);
            // Active (its aim changes): a tee that never moved is AFK and no job.
            f.env.clock.update(990, &f.env.tees, &f.env.players, 0);
            frozen.angle = 7;
            f.env.tees.set_for_test(frozen);
            f.env.clock.update(995, &f.env.tees, &f.env.players, 0);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 1000, &world);
            f.core.poll(&ctx);
            f.core.wb.chooser.adopt(side);
            f.core.begin_pick(&ctx, -1, &mut |_| false);
            assert_eq!(
                f.core.wb_spot_for(&ctx, &def, side, (first.0, first.1)),
                g.job,
                "over the one on the shelf"
            );
            // Sealed: no job, the first spot.
            f.core.begin_pick(&ctx, -1, &mut |_| true);
            let ctx = f.env.ctx(&own, 1001, &world);
            f.core.x.memo = None;
            f.core.begin_pick(&ctx, -1, &mut |_| true);
            assert_eq!(f.core.wb_spot_for(&ctx, &def, side, (first.0, first.1)), first);
        });
    }

    #[test]
    fn reloading_the_same_map_keeps_the_pause_of_the_wb_and_another_size_forgets_it() {
        on_big_stack(|| {
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            let map = Arc::new(map);
            let ident = MapIdent {
                name: "Copy Love Box".to_string(),
                sha256: [3; 32],
            };
            let mut core = Core::new(no_memory(), NavHandle::new());
            core.on_map(&map, &ident);
            for _ in 0..ddai_nav::wayblock::WB_WALK_MAX_FAILS {
                core.wb.note_walk_death(0);
            }
            assert!(core.wb.paused_min(0) > 0, "paused after the deaths");
            core.on_map(&map, &ident);
            assert!(core.wb.paused_min(0) > 0, "the same map again: the pause stays");
            let other = Arc::new(room(60, 30, &[]));
            core.on_map(
                &other,
                &MapIdent {
                    name: "Copy Love Box".to_string(),
                    sha256: [4; 32],
                },
            );
            assert_eq!(core.wb.paused_min(0), 0, "another size: forgotten");
        });
    }

    #[test]
    fn the_crowd_at_the_tube_counts_awake_foes_in_the_chamber_and_a_foe_acts_when_it_aims_at_us_after_an_action() {
        on_big_stack(|| {
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            let def = ddai_nav::wayblock::wayblocks().remove(0);
            let tube = def.left.crossing.clone();
            let mut rel = Relations::new();
            rel.add(crate::relations::ListKind::Friend, "pal");
            let mut f = CoreFx::on(map, "Copy Love Box", &rel);
            // We stand at the tube's start; "foe" (id 1) and "pal" (id 2, a friend) in the chamber next to it.
            let (sx, sy) = tube.start;
            let chamber = (tube.chamber.x0 + 3, tube.chamber.y0 + 3);
            f.env.tees.set_for_test(tee_at(0, sx, sy));
            f.env.tees.set_for_test(tee_at(1, chamber.0, chamber.1));
            f.env.tees.set_for_test(tee_at(2, chamber.0 + 1, chamber.1));
            // Both active (their aim changes), so neither is AFK.
            f.env.clock.update(990, &f.env.tees, &f.env.players, 0);
            for id in [1, 2] {
                let mut t = *f.env.tees.get(id).unwrap();
                t.angle = 100;
                f.env.tees.set_for_test(t);
            }
            f.env.clock.update(995, &f.env.tees, &f.env.players, 0);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 1000, &world);
            let crowd = Core::wb_crowd(&ctx, &tube);
            assert_eq!(crowd, vec![1], "a friend is no foe: only the one in the chamber counts");
            // He acts: swung at us just now, aiming at us -> acting; aiming away -> not.
            let mut him = *f.env.tees.get(1).unwrap();
            him.attack_tick = 990;
            let dx = own.pos.x - him.pos.x;
            let dy = own.pos.y - him.pos.y;
            him.angle = ((dy.atan2(dx) + std::f32::consts::TAU) % std::f32::consts::TAU * 256.0) as i32;
            assert!(f.core.foe_acting(&ctx, &him), "aims at us after an action");
            him.angle =
                ((dy.atan2(dx) + std::f32::consts::PI + std::f32::consts::TAU) % std::f32::consts::TAU * 256.0) as i32;
            assert!(!f.core.foe_acting(&ctx, &him), "aims away");
            him.attack_tick = 0;
            him.angle = ((dy.atan2(dx) + std::f32::consts::TAU) % std::f32::consts::TAU * 256.0) as i32;
            assert!(!f.core.foe_acting(&ctx, &him), "no action in the last 2 s");
        });
    }
    // ---- task 3.12: `--wb-smart` -------------------------------------------------------------------------------------

    fn smart_cfg() -> NavConfig {
        NavConfig {
            wb_smart: true,
            seed: Some(1),
            ..no_memory()
        }
    }

    /// Updates the clock so that the tees in `idle` never change (AFK after 10 s) and the others turn their aim every snapshot.
    fn age_the_clock(env: &mut CoreEnv, idle: &[i32], from: i32, to: i32) {
        for t in (from..=to).step_by(2) {
            for id in 1..6 {
                if idle.contains(&id) {
                    continue;
                }
                if let Some(mut tee) = env.tees.get(id).copied() {
                    tee.angle = t * 7;
                    env.tees.set_for_test(tee);
                }
            }
            env.clock.update(t, &env.tees, &env.players, 0);
        }
    }

    /// Task 3.12b: a tee that has just respawned on the ledge of the right tube's start (outside the hall, the WB held, no walk yet)
    /// stands still and walks at once under `--wb-smart`; the port's rule (free wander, a second of idling first) is the default.
    #[test]
    fn outside_the_hall_with_the_wb_held_the_smart_tee_stands_still_and_starts_its_walk_at_once() {
        on_big_stack(|| {
            if clb_map().is_none() {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            }
            let run = |cfg: NavConfig, at: (i32, i32)| {
                let mut f = CoreFx::with(cfg, clb_map().unwrap(), "Copy Love Box", &Relations::new());
                f.core.wb.chooser.adopt(ddai_nav::wayblock::WbSide::Right);
                f.env.tees.set_for_test(tee_at(0, at.0, at.1));
                let own = *f.env.tees.get(0).unwrap();
                let world = f.env.pw.inner().clone();
                let ctx = f.env.ctx(&own, 1000, &world);
                assert!(f.core.wb_holding(), "the WB is held");
                (
                    f.core.wb_wander_hint(&ctx).map(|h| h.still),
                    f.core.wb_return_ticks(&ctx),
                )
            };
            let ledge = (131, 35);
            assert_eq!(
                run(no_memory(), ledge),
                (None, WB_RETURN_TICKS),
                "default: the port's rule"
            );
            assert_eq!(
                run(smart_cfg(), ledge),
                (Some(true), 0),
                "smart: still, and the walk at once"
            );
            // In the hall nothing changes: the guard's own hint and the second of idling.
            let spot = ddai_nav::wayblock::wayblocks().remove(0).right.spots[0];
            let (still, wait) = run(smart_cfg(), spot);
            assert!(still.is_some(), "the hall has its own hint");
            assert_eq!(wait, WB_RETURN_TICKS);
        });
    }

    /// Review F12: the walk to the WB starts at once only on the first try of a life. A walk that ends without arriving (here it is ended
    /// by hand on every look: no route, or under `--no-selfkill` a route that needs a kill) is tried again after [`WB_RETURN_TICKS`], not
    /// on every look.
    #[test]
    fn the_immediate_walk_is_for_the_first_try_of_a_life_only() {
        on_big_stack(|| {
            if clb_map().is_none() {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            }
            let tries = |cfg: NavConfig, ticks: i32, respawn_at: Option<i32>| {
                let mut f = CoreFx::with(cfg, clb_map().unwrap(), "Copy Love Box", &Relations::new());
                f.core.wb.chooser.adopt(ddai_nav::wayblock::WbSide::Right);
                f.core.no_selfkill = true;
                f.env.tees.set_for_test(tee_at(0, 131, 35)); // the ledge of the right tube: outside the hall
                let own = *f.env.tees.get(0).unwrap();
                let world = f.env.pw.inner().clone();
                let mut first = None;
                for tick in (1000..1000 + ticks).step_by(2) {
                    if respawn_at == Some(tick) {
                        f.core.fresh_life = true; // `Navigator::respawned`
                    }
                    let ctx = f.env.ctx(&own, tick, &world);
                    f.core.sync(&ctx);
                    f.core.steer(&ctx, -1);
                    if f.core.nav.is_some() {
                        first.get_or_insert(tick);
                        f.core.end_nav(); // the walk ended without arriving
                    }
                }
                (f.core.wb_walk_tries, first)
            };
            let (smart, first) = tries(smart_cfg(), 200, None);
            assert!(
                first.is_some_and(|t| t <= 1004),
                "the first walk of the life starts at once: {first:?}"
            );
            assert!(
                smart <= 4,
                "then one per second at most: {smart} walks in 200 ticks (it was 100)"
            );
            let (_, _) = (smart, first);
            // A respawn starts a new life: the walk is at once again.
            let (with_respawn, _) = tries(smart_cfg(), 200, Some(1030));
            assert!(
                with_respawn == smart + 1,
                "one more try after the respawn: {with_respawn} against {smart}"
            );
            // Not on: the second of idling first, as before.
            let (default, first) = tries(no_memory(), 200, None);
            assert!(
                default <= 4 && first.is_some_and(|t| t >= 1000 + WB_RETURN_TICKS as i32),
                "default: {default}, first {first:?}"
            );
        });
    }

    #[test]
    fn an_idle_tee_is_in_the_way_in_the_held_hall_and_not_in_the_other_one_or_when_the_option_is_off() {
        on_big_stack(|| {
            if clb_map().is_none() {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            }
            let def = ddai_nav::wayblock::wayblocks().remove(0);
            let spot = def.right.spots[0];
            // The shelf of the live clips: the left end of the right hall's lower shelf, and the mirrored one on the left.
            let (shelf_right, shelf_left) = ((134, 84), (234 - 134, 84));
            let run = |cfg: NavConfig| {
                let mut f = CoreFx::with(cfg, clb_map().unwrap(), "Copy Love Box", &Relations::new());
                f.env.tees.set_for_test(tee_at(0, spot.0, spot.1));
                f.env.tees.set_for_test(tee_at(1, shelf_right.0, shelf_right.1));
                f.env.tees.set_for_test(tee_at(2, shelf_left.0, shelf_left.1));
                f.env.tees.set_for_test(tee_at(3, 146, 56)); // in the passage of the right hall, 24 tiles above
                age_the_clock(&mut f.env, &[1, 2, 3], 0, 600);
                let own = *f.env.tees.get(0).unwrap();
                let world = f.env.pw.inner().clone();
                let ctx = f.env.ctx(&own, 602, &world);
                f.core.poll(&ctx);
                assert_eq!(
                    f.core.wb.side(),
                    Some(ddai_nav::wayblock::WbSide::Right),
                    "we stand in the right hall"
                );
                for id in [1, 2, 3] {
                    assert!(ctx.clock.away_in_game(id, 602, ctx.players), "tee {id} is idle");
                }
                let way = |id: i32| f.core.afk_in_the_way(&ctx, f.env.tees.get(id).unwrap(), false);
                (way(1), way(2), way(3))
            };
            use super::in_the_way::Way;
            assert_eq!(
                run(no_memory()),
                (None, None, None),
                "off: nobody idle is in the way (the port's rule)"
            );
            assert_eq!(
                run(smart_cfg()),
                (Some(Way::Hall), None, None),
                "on: the idle tee on our hall's lower shelf; not the one in the other hall, not the one far up the passage"
            );
        });
    }

    #[test]
    fn an_idle_tee_next_to_us_or_on_the_route_of_the_walk_is_in_the_way_one_off_to_the_side_is_not() {
        on_big_stack(|| {
            use super::in_the_way::Way;
            let mut f = CoreFx::with(smart_cfg(), room(60, 12, &[]), "test room", &Relations::new());
            f.env.tees.set_for_test(tee_at(0, 4, 10));
            f.env.tees.set_for_test(tee_at(1, 5, 10)); // 32 px: next to us
            f.env.tees.set_for_test(tee_at(2, 9, 10)); // 5 tiles on, on the line to the goal
            f.env.tees.set_for_test(tee_at(3, 9, 4)); // 5 tiles on but 6 rows off the line
            f.env.tees.set_for_test(tee_at(4, 40, 10)); // on the line, 36 tiles away
            age_the_clock(&mut f.env, &[1, 2, 3, 4], 0, 600);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 602, &world);
            f.core.poll(&ctx);
            let way = |f: &CoreFx, id: i32| f.core.afk_in_the_way(&ctx, f.env.tees.get(id).unwrap(), false);
            assert_eq!(way(&f, 1), Some(Way::NextTo));
            assert_eq!(way(&f, 2), None, "no walk yet: nothing is on a route");
            f.core.handle.goto_tile(50, 10);
            f.core.poll(&ctx);
            assert!(f.core.nav.is_some(), "the walk began");
            assert_eq!(way(&f, 2), Some(Way::Route));
            assert_eq!(way(&f, 3), None, "off to the side");
            assert_eq!(way(&f, 4), None, "too far ahead to be in the way now");
            // A wall across the room: the walk goes around it; the idle tee on the line to the goal is still in the way (the branch with a
            // route runner's steps is not reached by these room walks: no runner is made, so only the goal line is tested).
            let wall: Vec<(u32, u32, u8)> = (3..=10u32).map(|y| (20, y, SOLID)).collect();
            let mut g = CoreFx::with(smart_cfg(), room(60, 12, &wall), "test room", &Relations::new());
            g.env.tees.set_for_test(tee_at(0, 4, 10));
            g.env.tees.set_for_test(tee_at(2, 9, 10));
            g.env.tees.set_for_test(tee_at(3, 9, 4));
            age_the_clock(&mut g.env, &[2, 3], 0, 600);
            let own = *g.env.tees.get(0).unwrap();
            let world = g.env.pw.inner().clone();
            let ctx = g.env.ctx(&own, 602, &world);
            g.core.poll(&ctx);
            g.core.handle.goto_tile(50, 10);
            g.core.poll(&ctx);
            for _ in 0..40 {
                g.core.drive(&ctx);
            }
            let way = |id: i32| g.core.afk_in_the_way(&ctx, g.env.tees.get(id).unwrap(), false);
            assert_eq!(way(2), Some(Way::Route), "on the line to the goal");
            assert_eq!(way(3), None, "6 rows off it");
            assert!(
                !g.core.afk_blocks_the_walk(&ctx),
                "on the route is not at our feet: the walk is not cut for him"
            );
            let off = CoreFx::with(no_memory(), room(60, 12, &[]), "test room", &Relations::new());
            assert!(!off.core.afk_blocks_the_walk(&ctx), "off: never");
        });
    }

    #[test]
    fn blockable_targets_count_the_active_around_a_hall_and_the_idle_only_inside_it_not_friends_parked_or_spectators() {
        on_big_stack(|| {
            let Some(map) = clb_map() else {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            };
            let def = ddai_nav::wayblock::wayblocks().remove(0);
            let mut rel = Relations::new();
            rel.add(crate::relations::ListKind::Friend, "pal");
            let mut f = CoreFx::with(smart_cfg(), map, "Copy Love Box", &rel);
            let spawn = ddai_nav::route::spawn_tiles(&f.env.map)[0];
            f.env
                .tees
                .set_for_test(tee_at(0, (spawn.0 / 32.0) as i32, (spawn.1 / 32.0) as i32));
            // Right hall: an active tee in the zone, an idle one on the lower shelf, a friend (never), a parked frozen one (never).
            let z = def.right.zone[0];
            f.env.tees.set_for_test(tee_at(1, z.x0 + 3, z.y0 + 2));
            f.env.tees.set_for_test(tee_at(2, 134, 84)); // "pal": a friend
            f.env.tees.set_for_test(tee_at(3, 140, 84)); // idle on the shelf
            let mut parked = tee_at(4, 142, 84);
            parked.frozen = true;
            parked.deep_frozen = true;
            f.env.tees.set_for_test(parked);
            // Left hall: an idle tee in the approach (the passage), not in the hall: not counted; an active one there: counted.
            let a = def.left.approach[0];
            f.env.tees.set_for_test(tee_at(5, a.x0 + 2, a.y0 + 2));
            age_the_clock(&mut f.env, &[3, 5], 0, 600);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 602, &world);
            assert_eq!(
                f.core.wb_blockable_counts(&ctx),
                (0, 2),
                "right: the active one and the idle one on the shelf"
            );
            // The active one in the left approach joins the left.
            let mut t5 = *f.env.tees.get(5).unwrap();
            t5.angle = 12345;
            f.env.tees.set_for_test(t5);
            f.env.clock.update(604, &f.env.tees, &f.env.players, 0);
            t5.angle = 54321;
            f.env.tees.set_for_test(t5);
            f.env.clock.update(606, &f.env.tees, &f.env.players, 0);
            let ctx = f.env.ctx(&own, 606, &world);
            assert_eq!(f.core.wb_blockable_counts(&ctx), (1, 2));
            // A spectator or a paused player is no target.
            let views = [
                player(0, "bot", "", true, 0, None),
                player(1, "foe", "", false, 0, None),
                player(2, "pal", "", false, 0, None),
                player(3, "p3", "", false, 0, Some(-1)),
                player(4, "p4", "", false, 0, None),
                player(5, "p5", "", false, 0, None),
            ];
            f.env.players.update(&views, &rel);
            let ctx = f.env.ctx(&own, 606, &world);
            assert_eq!(
                f.core.wb_blockable_counts(&ctx),
                (1, 1),
                "the spectator on the shelf is out"
            );
        });
    }

    #[test]
    fn the_smart_side_follows_the_blockable_targets_and_the_default_goes_to_the_emptier_hall() {
        on_big_stack(|| {
            if clb_map().is_none() {
                eprintln!("skipping: the Copy Love Box map is not present");
                return;
            }
            let def = ddai_nav::wayblock::wayblocks().remove(0);
            let side_of = |cfg: NavConfig| {
                let mut f = CoreFx::with(cfg, clb_map().unwrap(), "Copy Love Box", &Relations::new());
                let spawn = ddai_nav::route::spawn_tiles(&f.env.map)[0];
                f.env
                    .tees
                    .set_for_test(tee_at(0, (spawn.0 / 32.0) as i32, (spawn.1 / 32.0) as i32));
                // Three players on the left, one on the right, all playing.
                let l = def.left.zone[0];
                let r = def.right.zone[0];
                f.env.tees.set_for_test(tee_at(1, l.x0 + 3, l.y0 + 2));
                f.env.tees.set_for_test(tee_at(2, l.x0 + 5, l.y0 + 2));
                f.env.tees.set_for_test(tee_at(3, l.x0 + 7, l.y0 + 2));
                f.env.tees.set_for_test(tee_at(4, r.x0 + 3, r.y0 + 2));
                age_the_clock(&mut f.env, &[], 0, 100);
                let own = *f.env.tees.get(0).unwrap();
                let world = f.env.pw.inner().clone();
                let ctx = f.env.ctx(&own, 102, &world);
                f.core.poll(&ctx);
                f.core.wb.side().expect("a side")
            };
            use ddai_nav::wayblock::WbSide;
            assert_eq!(side_of(no_memory()), WbSide::Right, "the port's rule: the emptier hall");
            assert_eq!(
                side_of(smart_cfg()),
                WbSide::Left,
                "--wb-smart: the hall with more targets"
            );
        });
    }

    #[test]
    fn a_cross_fail_note_names_the_tube_of_its_side() {
        let def = ddai_nav::wayblock::wayblocks().remove(0);
        let n = "the right freeze tube: lies frozen at (159,102); trying again from the spawn";
        assert!(n.starts_with(&def.right.crossing.label) && !n.starts_with(&def.left.crossing.label));
        let n = "the left freeze tube: no swing through from where it got to; trying again from the spawn";
        assert!(n.starts_with(&def.left.crossing.label) && !n.starts_with(&def.right.crossing.label));
    }
    #[test]
    fn a_walk_is_cut_only_for_an_idle_tee_at_our_feet_and_he_stays_the_target_after_the_cut() {
        use super::in_the_way::{AFK_KEEP_PX, AFK_NEXT_TO_PX, Way};
        on_big_stack(|| {
            let mut f = CoreFx::with(smart_cfg(), room(60, 12, &[]), "test room", &Relations::new());
            f.env.tees.set_for_test(tee_at(0, 4, 10));
            f.env.tees.set_for_test(tee_at(2, 9, 10)); // idle, on the route, 5 tiles on
            age_the_clock(&mut f.env, &[2], 0, 600);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 602, &world);
            f.core.poll(&ctx);
            f.core.handle.goto_tile(50, 10);
            f.core.poll(&ctx);
            assert_eq!(
                f.core.afk_in_the_way(&ctx, f.env.tees.get(2).unwrap(), false),
                Some(Way::Route)
            );
            assert!(
                !f.core.afk_blocks_the_walk(&ctx),
                "on the route is not at our feet: the walk goes on"
            );
            // He steps up to us: the walk is cut for him, and once it is gone he is still in the way (he is next to us).
            f.env.tees.set_for_test(tee_at(2, 5, 10));
            age_the_clock(&mut f.env, &[2], 604, 1300);
            let ctx = f.env.ctx(&own, 1302, &world);
            assert!(f.core.afk_blocks_the_walk(&ctx));
            f.core.nav = None;
            assert_eq!(
                f.core.afk_in_the_way(&ctx, f.env.tees.get(2).unwrap(), false),
                Some(Way::NextTo),
                "no walk, no route, still in the way: no cut-and-retry loop"
            );
            // Hysteresis: 80 px away he is not in the way, unless he is our target now; 100 px: neither.
            const { assert!(80.0 > AFK_NEXT_TO_PX && 80.0 < AFK_KEEP_PX) };
            let mut far = tee_at(2, 4, 10);
            far.pos.x += 80.0;
            f.env.tees.set_for_test(far);
            let ctx = f.env.ctx(&own, 1302, &world);
            assert_eq!(f.core.afk_in_the_way(&ctx, f.env.tees.get(2).unwrap(), false), None);
            assert_eq!(
                f.core.afk_in_the_way(&ctx, f.env.tees.get(2).unwrap(), true),
                Some(Way::NextTo)
            );
            far.pos.x += 20.0;
            f.env.tees.set_for_test(far);
            let ctx = f.env.ctx(&own, 1302, &world);
            assert_eq!(f.core.afk_in_the_way(&ctx, f.env.tees.get(2).unwrap(), true), None);
        });
    }

    #[test]
    fn a_fight_where_we_stand_is_a_hook_either_way_our_target_close_by_or_somebody_who_attacked_us() {
        on_big_stack(|| {
            let mut rel = Relations::new();
            rel.add(crate::relations::ListKind::Friend, "pal");
            let mut f = CoreFx::with(smart_cfg(), room(60, 12, &[]), "test room", &rel);
            f.env.tees.set_for_test(tee_at(0, 4, 10));
            f.env.tees.set_for_test(tee_at(2, 6, 10)); // the friend, close
            f.env.tees.set_for_test(tee_at(3, 7, 10)); // idle, close
            age_the_clock(&mut f.env, &[3], 0, 600);
            let own = *f.env.tees.get(0).unwrap();
            let world = f.env.pw.inner().clone();
            let ctx = f.env.ctx(&own, 602, &world);
            assert!(!f.core.fighting_here(&ctx), "a friend and a sleeper are no fight");
            // Review F8: a free awake foe that merely stands within 420 px is no fight (it used to hold a switch for good).
            let mut foe = tee_at(1, 8, 10);
            foe.angle = 99;
            f.env.tees.set_for_test(foe);
            let ctx = f.env.ctx(&own, 602, &world);
            assert!(
                !f.core.fighting_here(&ctx),
                "an awake foe standing near is not an engagement"
            );
            // He is the player we are fighting now (the steering saw him as the target), within four tiles ...
            f.core.target_seen = (1, 600);
            let ctx = f.env.ctx(&own, 602, &world);
            assert!(f.core.fighting_here(&ctx), "our target within four tiles");
            // ... but not when he is farther, or when the target is old news.
            f.env.tees.set_for_test(tee_at(1, 12, 10)); // 8 tiles
            let ctx = f.env.ctx(&own, 602, &world);
            assert!(!f.core.fighting_here(&ctx), "our target 8 tiles away");
            f.env.tees.set_for_test(tee_at(1, 8, 10));
            let ctx = f.env.ctx(&own, 700, &world);
            assert!(!f.core.fighting_here(&ctx), "a target the steering saw 100 ticks ago");
            f.core.target_seen = (-1, i32::MIN / 2);
            // He hooks us from afar.
            let mut far = tee_at(1, 40, 10);
            far.hooked_player = 0;
            f.env.tees.set_for_test(far);
            let ctx = f.env.ctx(&own, 602, &world);
            assert!(f.core.fighting_here(&ctx), "he hooks us from afar");
            // Somebody swung at us a moment ago and is gone.
            let mut f2 = CoreFx::with(smart_cfg(), room(60, 12, &[]), "test room", &Relations::new());
            f2.env.tees.set_for_test(tee_at(0, 4, 10));
            f2.env.tees.set_for_test(tee_at(1, 6, 10));
            f2.env.clock.update(600, &f2.env.tees, &f2.env.players, 0);
            let mut swinger = tee_at(1, 6, 10);
            swinger.attack_tick = 601;
            f2.env.tees.set_for_test(swinger);
            f2.env.clock.update(602, &f2.env.tees, &f2.env.players, 0);
            f2.env.tees.set_for_test(tee_at(1, 30, 10));
            let own = *f2.env.tees.get(0).unwrap();
            let world = f2.env.pw.inner().clone();
            let ctx = f2.env.ctx(&own, 610, &world);
            assert!(f2.core.fighting_here(&ctx), "he swung at us 8 ticks ago");
            let ctx = f2.env.ctx(&own, 610 + AGGRESSOR_MEMORY_TICKS, &world);
            assert!(!f2.core.fighting_here(&ctx), "and not 150 ticks later");
        });
    }
}
