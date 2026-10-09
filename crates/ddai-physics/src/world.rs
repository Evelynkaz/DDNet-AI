// Ported from DDNet 20.1 `src/game/server/{gameworld,player,gamecontext,teams,gamecontroller}.cpp`
// and `src/game/server/entities/{character,projectile,pickup,door}.cpp` and
// `src/game/server/gamemodes/ddnet.cpp`. DDNet's zlib-style license notice for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//   /* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the       */
//   /* distribution for more information. Based on Race mod stuff and tweaked by GreYFoX@GTi and */
//   /* others to fit our DDRace needs.                                                            */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`, no `unsafe`. See this module's
// doc comment for the scope this covers (task 1.6, Stage A) and the free-function pattern it
// reuses from `core.rs` (extract a `Copy` snapshot, mutate/read siblings through `world`, write
// the snapshot back) for the same borrow-checker reason `core.rs`'s module doc comment explains.

//! `World<R>`: the server-level DDRace simulation task 1.6 (Stage A) adds on top of task 1.3's
//! core (`core::WorldCore`/`CharacterCore`, `collision::Collision`, `tuning::TuningParams`).
//! Covers everything `docs/formats.md`'s task 1.6 spec lists for Stage A: DDRace tile logic
//! (freeze/deep/live/unfreeze, death, every tele kind, speedups, stoppers, the switch layer
//! including `CDoor` collision, timed switches, switch-freeze/deep/jump/hit, tune zones, endless
//! hook, jump/walljump/refill tiles, NPC/NPH/HIT, solo, teams), hammer, `CProjectile` (gun and
//! grenade — and the `ENTITY_CRAZY_SHOTGUN[_EX]` map fixture, which is a real `CProjectile` too),
//! pickups, `CPlayer` spawn/respawn/kill (including the kill bit through the
//! `OnKillNetMessage` path), and the server's per-tick input/order handling.
//!
//! **Stage B** (task 1.6's second half, merged with Stage A in this module): `CLaser` (rifle and
//! shotgun — DDNet's shotgun fires a laser, not a projectile; `laser`), `CDragger`/
//! `CDraggerBeam`, `CGun` (turret)/`CPlasma`, `CLight` (`fixtures`) and ninja (`ninja`,
//! `HandleNinja` plus the activation in `FireWeapon`). The five classes (plus `CDoor`) share
//! DDNet's single `ENTTYPE_LASER` entity list, whose tick order this module reproduces: dynamic
//! entities (lasers, beams, plasma — new ones always go to the list head) in
//! [`World::lasers`], then the static map fixtures in [`World::fixtures`] (creation order,
//! newest first). With stage B the "Stage-B cut" of the Oracle B parity test is gone: every
//! corpus tick is compared. Also out of scope, with no observable effect on any compared field (see
//! this module's `BUILD REPORT` for the evidence for each): `/rescue` (`TrySetRescue`), `/pause`,
//! team locking/flocking/practice mode (`CGameTeams::m_aTeamLocked`/`m_aTeamFlock`/`m_aPractice`
//! — none of the three can ever become `true` through anything this harness/scenario format can
//! drive: no chat commands, no `sv_practice_by_default` cfg line anywhere in the corpus), and
//! save/load-team (`GetSaving`, always `false` for the same reason).

mod fixtures;
mod laser;
mod ninja;

pub use fixtures::{Dragger, DraggerBeam, DraggerState, Fixture, Gun, GunState, Light, MOVER_PERIOD, Plasma};
pub use laser::{Laser, LaserList, LaserSlot, add_velocity};
pub use ninja::handle_ninja;

use crate::collision::Collision;
use crate::core::{
    self, CharacterCore, MAX_CLIENTS, NUM_DDRACE_TEAMS, NUM_WEAPONS, PlayerInput, Switcher, TEAM_FLOCK, TEAM_SUPER,
    TeamsCore, WEAPON_GRENADE, WEAPON_GUN, WEAPON_HAMMER, WEAPON_LASER, WEAPON_NINJA, WEAPON_SHOTGUN, WorldCore,
};
use crate::map::{self, MapData};
use crate::prng::Prng;
use crate::real::Real;
use crate::switch;
use crate::tuning::TuningParams;
use crate::vmath::{self, Vec2};

// --- Dev-only phase profiling (task 1.10), `--features phase_profile` only ---------------------

/// Per-phase wall-clock time accumulated inside `World::world_tick` (private), read (and reset to zero)
/// via [`take_phase_profile`]. Only compiled in under `--features phase_profile`; every normal
/// build (including every other crate's default build, which never enables this feature) pays
/// nothing for it — not even a branch, since the `Instant::now()` call sites don't exist at all.
/// A coarse `std::time::Instant`-based breakdown (task 1.10 acceptance criterion 1's documented
/// fallback for when a real sampling profiler — `perf` — isn't usable: this box's
/// `perf_event_paranoid=4` blocks it without a system-wide capability change this task doesn't
/// make), not a substitute for `perf`/`callgrind` — each phase's own `Instant::now()` pair adds a
/// few ns of overhead per call, negligible next to the phases being measured (all >= tens of ns)
/// but enough that this must never be enabled in a build anything else measures throughput on.
#[cfg(feature = "phase_profile")]
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseProfile {
    /// The projectile-tick loop (`world_tick`'s `ENTTYPE_PROJECTILE` pass).
    pub projectiles: std::time::Duration,
    /// The `ENTTYPE_LASER` pass: lasers, dragger beams, turret shots, then draggers/turrets/lights.
    pub fixtures: std::time::Duration,
    /// The pickup-tick loop (`world_tick`'s `ENTTYPE_PICKUP` pass, including each pickup's
    /// `find_characters_in_range_into` scan).
    pub pickups: std::time::Duration,
    /// The `no_weak_hook` `PreTick` pre-pass (usually a no-op loop — zero on every corpus
    /// scenario this crate has seen where `sv_no_weak_hook` is off).
    pub character_pre_tick_pass: std::time::Duration,
    /// The main per-character `Tick()` pass (`character_tick`: DDRace tiles, weapons, core tick).
    pub character_tick_pass: std::time::Duration,
    /// The `TickDeferred()` pass (`Move()` + `Quantize()`).
    pub character_deferred_pass: std::time::Duration,
    /// `projectiles.retain(...)`.
    pub retain: std::time::Duration,
    /// The `m_StrongWeakId` assignment pass.
    pub strong_weak_id_pass: std::time::Duration,
    /// [`switch::tick_switch_expiry`] (called from [`World::step`], not `world_tick`, but
    /// accumulated into the same thread-local for one combined report).
    pub switch_expiry: std::time::Duration,
}

#[cfg(feature = "phase_profile")]
thread_local! {
    static PHASE_PROFILE: std::cell::RefCell<PhaseProfile> = std::cell::RefCell::new(PhaseProfile::default());
}

/// Returns the [`PhaseProfile`] accumulated since the last call (or since startup), zeroing it
/// back out — so a caller can bracket exactly the ticks it wants measured. `--features
/// phase_profile` only.
#[cfg(feature = "phase_profile")]
pub fn take_phase_profile() -> PhaseProfile {
    PHASE_PROFILE.with(|p| p.replace(PhaseProfile::default()))
}

/// Adds `dt` to one [`PhaseProfile`] field, selected by `$field`. `--features phase_profile` only
/// (every call site below is itself `#[cfg(feature = "phase_profile")]`-gated, so this macro is
/// never invoked, let alone defined, in a normal build).
#[cfg(feature = "phase_profile")]
macro_rules! phase_time {
    ($field:ident, $body:expr) => {{
        let __start = std::time::Instant::now();
        let __result = $body;
        PHASE_PROFILE.with(|p| p.borrow_mut().$field += __start.elapsed());
        __result
    }};
}
#[cfg(not(feature = "phase_profile"))]
macro_rules! phase_time {
    ($field:ident, $body:expr) => {
        $body
    };
}

/// `TuneZone::NUM` (`engine/shared/protocol.h`): `CGameContext::m_aTuningList`'s size.
pub const TUNE_ZONE_COUNT: usize = 256;

/// `CGameContext::m_aTuningList[TuneZone::NUM]`: one [`TuningParams`] per tune zone (index `0` is
/// the default tuning used outside any zone). See [`TuningList::reset_to_baseline`] for the
/// exact baseline every zone starts at.
///
/// `zones` is reference-counted (`Arc`), not owned outright (task 1.10, acceptance criterion 2's
/// "copy-on-write map", the same idea [`World::collision`] already applies, generalized to this
/// field too): every one of this crate's own scenarios spends its entire run at whatever tuning
/// `World::init` left it at (`tune`/`tune_zone`/`sv_solo_server` config-time writes only — never a
/// per-tick mutation), so a `World::clone()`/`World::restore_from()` that never diverges a
/// world's tuning from its saved baseline should cost nothing at all for this field, not a
/// `TUNE_ZONE_COUNT * size_of::<TuningParams>()` (tens of KB) copy every time. `Clone` is left
/// derived: the default `clone_from` (`*self = source.clone()`) already just clones the `Arc`
/// handle (an `O(1)` refcount bump — `Arc<T>::clone()` never deep-copies `T`), which is exactly
/// the fast path wanted here. [`TuningList::zone_mut`]/[`TuningList::zero_player_collision_and_hooking_everywhere`]
/// use [`std::sync::Arc::make_mut`], which *does* deep-copy — but only the first time a given
/// `Arc` allocation is actually mutated while shared (`make_mut`'s own documented behavior), so
/// the cost of a genuine tuning change still lands exactly once, on whichever `World` writes it.
///
/// Task 1.10b, finding F2: `Clone` is hand-written (not derived) purely to give `clone_from` an
/// `Arc::ptr_eq` short-circuit — the default `clone_from` (`*self = source.clone()`) would still
/// do two atomic refcount ops (an increment cloning `source.zones`, then a decrement dropping
/// `self`'s old one) even when both already point at the very same allocation, the common
/// steady-state case for a search loop's `World::restore_from` that never diverges tuning between
/// saves. `clone()` itself is unchanged from what `#[derive]` would generate.
#[derive(Debug)]
pub struct TuningList {
    zones: std::sync::Arc<[TuningParams; TUNE_ZONE_COUNT]>,
}

impl Clone for TuningList {
    fn clone(&self) -> Self {
        TuningList {
            zones: self.zones.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        if !std::sync::Arc::ptr_eq(&self.zones, &source.zones) {
            self.zones = std::sync::Arc::clone(&source.zones);
        }
    }
}

impl TuningList {
    /// `CGameContext::OnInit`'s tune-zone reset loop (`gamecontext.cpp:4110-4119`): every zone
    /// (including zone 0) gets `CTuningParams::DEFAULT` plus 5 weapon-tuning overrides
    /// (`gun_curvature=0`, `gun_speed=1400`, `shotgun_curvature=0`, `shotgun_speed=500`,
    /// `shotgun_speeddiff=0`) that also get (re)applied to zone 0 alone right after, via either
    /// `ResetTuning()` or the `else` branch (`gamecontext.cpp:4127-4139`) — both produce the same
    /// net values for zone 0 as this loop already gives every zone, so this crate doesn't model
    /// `sv_tune_reset` as a separate step (see the module's `BUILD REPORT` for that equivalence
    /// argument).
    pub fn reset_to_baseline() -> Self {
        let mut zones: [TuningParams; TUNE_ZONE_COUNT] = [TuningParams::default(); TUNE_ZONE_COUNT];
        for zone in zones.iter_mut() {
            zone.set_by_name("gun_curvature", 0.0);
            zone.set_by_name("gun_speed", 1400.0);
            zone.set_by_name("shotgun_curvature", 0.0);
            zone.set_by_name("shotgun_speed", 500.0);
            zone.set_by_name("shotgun_speeddiff", 0.0);
        }
        TuningList {
            zones: std::sync::Arc::new(zones),
        }
    }

    /// Zone `n`'s tuning (`0..256`).
    pub fn zone(&self, n: i32) -> &TuningParams {
        &self.zones[n as usize]
    }

    /// Mutable zone `n`'s tuning. Copy-on-write: see the struct doc comment.
    pub fn zone_mut(&mut self, n: i32) -> &mut TuningParams {
        &mut std::sync::Arc::make_mut(&mut self.zones)[n as usize]
    }

    /// `sv_solo_server`'s tuning side effect (`gamecontext.cpp:4171-4178`): zeroes
    /// `player_collision`/`player_hooking` in every zone, including zone 0
    /// (`GlobalTuning()->Set(...)`, applied identically inside the same `if`). Copy-on-write: see
    /// the struct doc comment.
    pub fn zero_player_collision_and_hooking_everywhere(&mut self) {
        for zone in std::sync::Arc::make_mut(&mut self.zones).iter_mut() {
            zone.set_by_name("player_collision", 0.0);
            zone.set_by_name("player_hooking", 0.0);
        }
    }
}

/// The subset of `g_Config`/tuning-adjacent server settings this corpus's scenarios and map
/// "Settings" strings actually use, plus every command acceptance criterion 2 explicitly names
/// (`sv_no_weak_hook`, `sv_hit`, `sv_solo_server`, `sv_team`, `sv_deepfly`, `sv_endless_drag`).
/// See [`crate::world::apply_command`] for the recognized command set and its "unknown command
/// -> explicit error" behavior, and [`World::init`] for the two-phase (pre-init/post-init)
/// timing model this struct's `game_settings_locked` field exists to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerConfig {
    /// `g_Config.m_SvHit` (`sv_hit`, default `1`, `CFGFLAG_SERVER | CFGFLAG_GAME` —
    /// `config_variables.h:541`).
    pub sv_hit: bool,
    /// `g_Config.m_SvTeam` (`sv_team`, default `1` = `SV_TEAM_ALLOWED`, `CFGFLAG_GAME` —
    /// `config_variables.h:611`). `3` = `SV_TEAM_FORCED_SOLO`.
    pub sv_team: i32,
    /// `g_Config.m_SvSoloServer` (`sv_solo_server`, default `0`, `CFGFLAG_GAME` —
    /// `config_variables.h:711`).
    pub sv_solo_server: bool,
    /// `g_Config.m_SvNoWeakHook` (`sv_no_weak_hook`, default `0`, `CFGFLAG_GAME` —
    /// `config_variables.h:623`).
    pub sv_no_weak_hook: bool,
    /// `g_Config.m_SvDeepfly` (`sv_deepfly`, default `1`, `CFGFLAG_GAME` —
    /// `config_variables.h:295`).
    pub sv_deepfly: bool,
    /// `g_Config.m_SvEndlessDrag` (`sv_endless_drag`, default `0`, `CFGFLAG_GAME` —
    /// `config_variables.h:542`) — also forced to `1` map-wide by a `TILE_EHOOK` tile anywhere on
    /// the game/front layer (`gamecontext.cpp:4298-4302`, a direct engine-side write, never
    /// subject to the read-only lock below).
    pub sv_endless_drag: bool,
    /// `g_Config.m_SvOldTeleportHook` (`sv_old_teleport_hook`, default `0`, `CFGFLAG_GAME` —
    /// `config_variables.h:292`).
    pub sv_old_teleport_hook: bool,
    /// `g_Config.m_SvOldTeleportWeapons` (`sv_old_teleport_weapons`, default `0`,
    /// `CFGFLAG_GAME` — `config_variables.h:291`).
    pub sv_old_teleport_weapons: bool,
    /// `g_Config.m_SvTeleportHoldHook` (`sv_teleport_hold_hook`, default `0`, `CFGFLAG_GAME` —
    /// `config_variables.h:293`).
    pub sv_teleport_hold_hook: bool,
    /// `g_Config.m_SvTeleportLoseWeapons` (`sv_teleport_lose_weapons`, default `0`,
    /// `CFGFLAG_GAME` — `config_variables.h:294`).
    pub sv_teleport_lose_weapons: bool,
    /// `g_Config.m_SvDestroyBulletsOnDeath` (`sv_destroy_bullets_on_death`, default **`1`**,
    /// `CFGFLAG_SERVER | CFGFLAG_GAME` — `config_variables.h:296`). Read by
    /// [`projectile_tick`]'s "owner not alive" check (`projectile.cpp:125`): found empirically —
    /// an earlier revision of this port hardcoded this `false` (with a doc comment that
    /// incorrectly claimed the *default* was `0`), so a dead owner's grenade never got destroyed
    /// by default, when in fact (at the DDNet default) every projectile type does.
    pub sv_destroy_bullets_on_death: bool,
    /// `g_Config.m_SvOldLaser` (`sv_old_laser`, default `0`, `CFGFLAG_GAME` —
    /// `config_variables.h:620`). Read by `CLaser` (self-hit rule, shotgun pull direction); also
    /// forced on map-wide by a `TILE_OLDLASER` tile, and covered by the reset rule
    /// ([`World::init`]) and the lock ([`apply_command`]).
    pub sv_old_laser: bool,
    /// `g_Config.m_SvDestroyLasersOnDeath` (`sv_destroy_lasers_on_death`, default `0`,
    /// `CFGFLAG_SERVER | CFGFLAG_GAME` — `config_variables.h:297`): `CLaser::Tick` destroys a
    /// laser whose owner is no longer alive (`laser.cpp:265-272`).
    pub sv_destroy_lasers_on_death: bool,
    /// `g_Config.m_SvDraggerRange` (`sv_dragger_range`, default `700`, `1..=99999`,
    /// `CFGFLAG_GAME` — `config_variables.h:691`): how far a dragger tracks tees.
    pub sv_dragger_range: i32,
    /// `g_Config.m_SvPlasmaRange` (`sv_plasma_range`, default `700`, `1..=99999`, `CFGFLAG_GAME`
    /// — `config_variables.h:689`): how far a turret tracks tees.
    pub sv_plasma_range: i32,
    /// `g_Config.m_SvPlasmaPerSec` (`sv_plasma_per_sec`, default `3`, `0..=50`, `CFGFLAG_GAME` —
    /// `config_variables.h:690`): turret shots per second (`0` = turrets never fire).
    pub sv_plasma_per_sec: i32,
    /// `g_Config.m_SvShowOthersDefault` (`sv_show_others_default`, default `0` = `SHOW_OTHERS_OFF`,
    /// `CFGFLAG_GAME` — `config_variables.h:686`). Network/HUD-only (never read by any traced
    /// field) — tracked for the same completeness reason as `sv_old_laser`.
    pub sv_show_others_default: i32,
    /// `g_Config.m_SvFreezeDelay` (`sv_freeze_delay`, default `3` seconds, `CFGFLAG_GAME` —
    /// `config_variables.h:544`).
    pub sv_freeze_delay: i32,
    /// `g_Config.m_SvKillDelay` (`sv_kill_delay`, default `1` second, `CFGFLAG_SERVER` only —
    /// `config_variables.h:573`) — `OnKillNetMessage`. Never locked.
    pub sv_kill_delay: i32,
    /// `g_Config.m_SvKillProtection` (`sv_kill_protection`, default `20` minutes, `0` disables,
    /// `CFGFLAG_SERVER` only — `config_variables.h:710`). Never locked.
    pub sv_kill_protection: i32,
    /// `g_Config.m_SvTuneReset` (`sv_tune_reset`, default `1`, `CFGFLAG_SERVER` only —
    /// `config_variables.h:694`). Gates `OnInit()`'s `ResetTuning()` vs the 5-value-only `else`
    /// branch (`gamecontext.cpp:4126-4137`) — provably a no-op either way for this crate's
    /// *full* [`TuningList`] (both branches are strict subsets of the unconditional per-zone
    /// reset [`World::init`] already performs — `gamecontext.cpp:4108-4120` — since that loop
    /// already resets *every* zone, including zone 0, to `DEFAULT` plus those same 5 values).
    /// Tracked (and its command recognized) for completeness even though [`World::init`] does
    /// not need to branch on it.
    pub sv_tune_reset: bool,
    /// `g_Config.m_SvDDRaceTuneReset` (`sv_ddrace_tune_reset`, default `1`, `CFGFLAG_SERVER`
    /// only — `config_variables.h:697`). Gates the `sv_hit`/`sv_endless_drag`/`sv_old_laser`/
    /// `sv_old_teleport_hook`/`sv_old_teleport_weapons`/`sv_teleport_hold_hook`/`sv_team`/
    /// `sv_show_others_default` reset-to-default block, and every switcher's `m_Initial = true`
    /// reset (`gamecontext.cpp:4139-4154`) — this one is *not* a no-op, so [`World::init`]
    /// honors its current value.
    pub sv_ddrace_tune_reset: bool,
    /// Mirrors every `CFGFLAG_GAME` variable's `SConfigVariable::m_ReadOnly`
    /// (`engine/shared/config.h:124`) collectively: `false` before
    /// `CConfigManager::SetGameSettingsReadOnly(true)` (called once, inside
    /// `CGameContext::OnInit()`, right after `LoadMapSettings()` —
    /// `gamecontext.cpp:4157`), `true` after. [`World::init`] flips this at exactly that point;
    /// [`apply_command`] consults it to reject further writes to any `CFGFLAG_GAME` variable
    /// (`SConfigVariable::CheckReadOnly`, `engine/shared/config.cpp:29-34`) exactly like the
    /// real server does, instead of silently applying them.
    pub game_settings_locked: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            sv_hit: true,
            sv_team: 1,
            sv_solo_server: false,
            sv_no_weak_hook: false,
            sv_deepfly: true,
            sv_endless_drag: false,
            sv_old_teleport_hook: false,
            sv_old_teleport_weapons: false,
            sv_teleport_hold_hook: false,
            sv_teleport_lose_weapons: false,
            sv_destroy_bullets_on_death: true,
            sv_old_laser: false,
            sv_destroy_lasers_on_death: false,
            sv_dragger_range: 700,
            sv_plasma_range: 700,
            sv_plasma_per_sec: 3,
            sv_show_others_default: 0,
            sv_freeze_delay: 3,
            sv_kill_delay: 1,
            sv_kill_protection: 20,
            sv_tune_reset: true,
            sv_ddrace_tune_reset: true,
            game_settings_locked: false,
        }
    }
}

/// `SV_TEAM_FORCED_SOLO` (`teamscore.h`).
pub const SV_TEAM_FORCED_SOLO: i32 = 3;
/// `SV_TEAM_ALLOWED` (`teamscore.h`) — `sv_team`'s default, and what `OnInit()`'s
/// `sv_ddrace_tune_reset` block resets it back to (`gamecontext.cpp:4145`).
pub const SV_TEAM_ALLOWED: i32 = 1;

/// Every `CFGFLAG_GAME` console command this crate recognizes (`config_variables.h`, grepped for
/// `CFGFLAG_GAME`, restricted to the ones any `MACRO_CONFIG_INT`-backed `g_Config.m_Sv*` this
/// crate models could plausibly appear as in a corpus `--cfg` file or map "Settings" string).
/// **Not** included: `tune`/`tune_zone`/`tune_zone_enter`/`tune_zone_leave`/`switch_open`/
/// `mapbug` — these are plain `Console()->Register(...)` *commands* (also tagged
/// `CFGFLAG_GAME`, but that flag means something different there: it is read by
/// `IConsole`'s own dispatch, not by `CConfigManager`), not `SConfigVariable`s, so
/// `SetGameSettingsReadOnly`/`CheckReadOnly` never applies to them — confirmed by reading every
/// one of their callbacks (`ConTuneParam`/`ConTuneZone`/`ConSwitchOpen`/...,
/// `gamecontext.cpp:3040+`), none of which calls `CheckReadOnly` or consults any read-only
/// state. They can always be set, pre- or post-init, cfg or map settings alike.
fn is_cfgflag_game(name: &str) -> bool {
    matches!(
        name,
        "sv_hit"
            | "sv_team"
            | "sv_solo_server"
            | "sv_no_weak_hook"
            | "sv_deepfly"
            | "sv_endless_drag"
            | "sv_old_teleport_hook"
            | "sv_old_teleport_weapons"
            | "sv_teleport_hold_hook"
            | "sv_teleport_lose_weapons"
            | "sv_old_laser"
            | "sv_show_others_default"
            | "sv_freeze_delay"
            | "sv_destroy_bullets_on_death"
            | "sv_destroy_lasers_on_death"
            | "sv_dragger_range"
            | "sv_plasma_range"
            | "sv_plasma_per_sec"
    )
}

/// One error produced by [`apply_command`]: an unrecognized console/settings command — see
/// acceptance criterion 2 ("unknown commands -> explicit error listing them").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownCommand {
    /// The full source line (as given), for diagnostics.
    pub line: String,
}

/// Applies one console-command line (from a scenario `cfg_lines` entry or a map "Settings"
/// string) to `config`/`tuning`/`switchers`. Recognizes: `sv_hit`, `sv_team`, `sv_solo_server`,
/// `sv_no_weak_hook`, `sv_deepfly`, `sv_endless_drag`, `sv_old_teleport_hook`,
/// `sv_old_teleport_weapons`, `sv_teleport_hold_hook`, `sv_teleport_lose_weapons`,
/// `sv_old_laser`, `sv_show_others_default`, `sv_freeze_delay`, `sv_kill_delay`,
/// `sv_kill_protection`, `sv_tune_reset`, `sv_ddrace_tune_reset`, `sv_destroy_bullets_on_death`,
/// `sv_destroy_lasers_on_death`, `sv_dragger_range`, `sv_plasma_range`, `sv_plasma_per_sec`
/// (all `ConInt`-style:
/// `NAME VALUE`), `tune NAME VALUE` (`ConTuneParam`, zone 0 only), `tune_zone N NAME VALUE`
/// (`ConTuneZone`), `switch_open N` (`ConSwitchOpen`), and `tune_zone_enter`/`tune_zone_leave`
/// (zone-change chat messages — recognized as a no-op; cosmetic, no physics effect). Any other
/// command is an error (acceptance criterion 2) rather than being silently ignored.
///
/// **The `CFGFLAG_GAME` lock** (`config.cpp:29-34`, `SConfigVariable::CheckReadOnly`): when
/// `config.game_settings_locked` is set (by [`World::init`], mirroring
/// `CConfigManager::SetGameSettingsReadOnly(true)`, `gamecontext.cpp:4157`) and `cmd` is one of
/// `is_cfgflag_game`'s (private below) names with at least one argument (a *set*, not a bare "print current
/// value" query — matching `SIntConfigVariable::CommandCallback`'s own
/// `if(pResult->NumArguments())` gate before it even calls `CheckReadOnly`), the write is
/// rejected: a human-readable line is appended to `command_log` (mirroring the real server's
/// `log_error("config", "The config variable '%s' cannot be changed right now.", ...)`) and the
/// field is left untouched, but this still returns `Ok(())` — a locked-out write is a normal,
/// expected outcome (recorded as a warning), not the "genuinely unrecognized command" error
/// [`UnknownCommand`] exists for. `tune`/`tune_zone`/`switch_open`/`tune_zone_enter`/
/// `tune_zone_leave` are never subject to this (see `is_cfgflag_game`'s doc comment, private below).
pub fn apply_command(
    config: &mut ServerConfig,
    tuning: &mut TuningList,
    switchers: &mut [Switcher],
    command_log: &mut Vec<String>,
    line: &str,
) -> Result<(), UnknownCommand> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(());
    }
    let mut parts = trimmed.split_whitespace();
    let Some(cmd) = parts.next() else { return Ok(()) };
    let rest: Vec<&str> = parts.collect();
    let unknown = || Err(UnknownCommand { line: line.to_string() });

    if config.game_settings_locked && !rest.is_empty() && is_cfgflag_game(cmd) {
        command_log.push(format!(
            "warning: config variable '{cmd}' cannot be changed right now (post-init, CFGFLAG_GAME) -- line: {line:?}"
        ));
        return Ok(());
    }

    match cmd {
        "sv_hit" => set_bool(&mut config.sv_hit, &rest, line),
        "sv_solo_server" => set_bool(&mut config.sv_solo_server, &rest, line),
        "sv_no_weak_hook" => set_bool(&mut config.sv_no_weak_hook, &rest, line),
        "sv_deepfly" => set_bool(&mut config.sv_deepfly, &rest, line),
        "sv_endless_drag" => set_bool(&mut config.sv_endless_drag, &rest, line),
        "sv_old_teleport_hook" => set_bool(&mut config.sv_old_teleport_hook, &rest, line),
        "sv_old_teleport_weapons" => set_bool(&mut config.sv_old_teleport_weapons, &rest, line),
        "sv_teleport_hold_hook" => set_bool(&mut config.sv_teleport_hold_hook, &rest, line),
        "sv_teleport_lose_weapons" => set_bool(&mut config.sv_teleport_lose_weapons, &rest, line),
        "sv_old_laser" => set_bool(&mut config.sv_old_laser, &rest, line),
        "sv_destroy_bullets_on_death" => set_bool(&mut config.sv_destroy_bullets_on_death, &rest, line),
        "sv_destroy_lasers_on_death" => set_bool(&mut config.sv_destroy_lasers_on_death, &rest, line),
        "sv_dragger_range" => set_int(&mut config.sv_dragger_range, &rest, line, 1, 99999),
        "sv_plasma_range" => set_int(&mut config.sv_plasma_range, &rest, line, 1, 99999),
        "sv_plasma_per_sec" => set_int(&mut config.sv_plasma_per_sec, &rest, line, 0, 50),
        // Ranges match `config_variables.h`'s own `MACRO_CONFIG_INT(..., min, max, ...)` for
        // each variable exactly (`sv_team` 0-3, `sv_show_others_default` 0-2, `sv_freeze_delay`
        // 1-30, `sv_kill_delay`/`sv_kill_protection` 0-9999).
        "sv_team" => set_int(&mut config.sv_team, &rest, line, 0, 3),
        "sv_show_others_default" => set_int(&mut config.sv_show_others_default, &rest, line, 0, 2),
        "sv_freeze_delay" => set_int(&mut config.sv_freeze_delay, &rest, line, 1, 30),
        "sv_kill_delay" => set_int(&mut config.sv_kill_delay, &rest, line, 0, 9999),
        "sv_kill_protection" => set_int(&mut config.sv_kill_protection, &rest, line, 0, 9999),
        "sv_tune_reset" => set_bool(&mut config.sv_tune_reset, &rest, line),
        "sv_ddrace_tune_reset" => set_bool(&mut config.sv_ddrace_tune_reset, &rest, line),
        "tune" => {
            if rest.len() != 2 {
                return unknown();
            }
            let value: f32 = rest[1].parse().map_err(|_| UnknownCommand { line: line.to_string() })?;
            if !tuning.zone_mut(0).set_by_name(rest[0], value) {
                return unknown();
            }
            Ok(())
        }
        "tune_zone" => {
            if rest.len() != 3 {
                return unknown();
            }
            let zone: i32 = rest[0].parse().map_err(|_| UnknownCommand { line: line.to_string() })?;
            let value: f32 = rest[2].parse().map_err(|_| UnknownCommand { line: line.to_string() })?;
            if !(0..TUNE_ZONE_COUNT as i32).contains(&zone) || !tuning.zone_mut(zone).set_by_name(rest[1], value) {
                return unknown();
            }
            Ok(())
        }
        "switch_open" => {
            if rest.len() != 1 {
                return unknown();
            }
            let n: usize = rest[0].parse().map_err(|_| UnknownCommand { line: line.to_string() })?;
            if n < switchers.len() {
                switchers[n].initial = false;
            }
            Ok(())
        }
        "tune_zone_enter" | "tune_zone_leave" => Ok(()),
        _ => unknown(),
    }
}

/// Every `bool`-backed `ServerConfig` field this crate models is, in real DDNet, a plain
/// `MACRO_CONFIG_INT` with `min=0, max=1` (`config_variables.h`) — see [`set_int`] for the
/// clamping this shares with it (`sv_hit 2` silently becomes `1`, it is not a parse error).
fn set_bool(field: &mut bool, rest: &[&str], line: &str) -> Result<(), UnknownCommand> {
    let mut value = 0i32;
    set_int(&mut value, rest, line, 0, 1)?;
    *field = value != 0;
    Ok(())
}

// --- `CCharacter`'s DDRace-level state (`character.h`) ------------------------------------------

/// DDRace-level per-character state — everything `CCharacter` (`character.h`) has beyond its
/// embedded [`CharacterCore`] (which lives in [`World::cores`] instead, keyed the same way).
/// `Default` (all-zero/`false`) matches `MACRO_ALLOC_POOL_ID_IMPL`'s `mem_zero` on every `new`
/// (`game/alloc.h:52-59`) — DDNet's pool allocator zeroes a character's *entire* memory block on
/// every spawn *and* every respawn (not just the first time), so every field `Spawn`/
/// `DDRaceInit` don't explicitly set genuinely does start at zero each (re)spawn, not "whatever
/// was left over from this client id's previous life" — see the task's `BUILD REPORT` for the
/// `alloc.h` citation this rests on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Character<R: Real> {
    /// `m_Alive`.
    pub alive: bool,
    /// `CCharacter::m_MoveRestrictions` (`character.h:136`) — a *separate* field from
    /// `CCharacterCore::m_MoveRestrictions` (`gamecore.h:282`, [`CharacterCore::move_restrictions`]):
    /// this one is written by `HandleTiles(int Index)` (`character.cpp:1638-1642`), called from
    /// the DDRace tile anti-skip loop with a `MapIndex` override (whatever intermediate tile the
    /// skip loop is currently checking, *not* necessarily `GetPureMapIndex(m_Pos)`), while the
    /// core one is written once per tick by `CCharacterCore::Tick()` (`gamecore.cpp:197`) from
    /// `m_Pos` alone, no override. Found empirically (root-caused by a parallel diagnosis agent
    /// against an instrumented Oracle B binary, `recipe_front_seed20004` character 1 tick 5):
    /// this crate's `handle_tiles` was overwriting the *core* field via
    /// `CharacterCore::set_move_restrictions`, conflating the two — the reference trace dumps
    /// the core field directly (`oracle_server.cpp:2375`, `pChar->m_Core.m_MoveRestrictions`),
    /// so every consumer of the *character* field (`TakeDamage`, `SetVelocity`/
    /// `ApplyMoveRestrictions`, the speedup tiles, the stopper jump-reset, and the *other*
    /// character's field in the hammer-hit force clamp, `character.cpp:1059,2595,2612,1572,1592,
    /// 1824,550`) must read *this* field, never the core one (which is consumed *only* by the
    /// hook-drag clamp, `gamecore.cpp:517,521` — `core.rs`'s own `move_restrictions()` getter
    /// stays as-is for that). Initial value `0`, matching `MACRO_ALLOC_POOL_ID_IMPL`'s zeroing.
    pub move_restrictions: i32,
    /// `m_DDRaceState` (`ERaceState`: `0`=None, `1`=Started, `2`=Cheated, `3`=Finished).
    pub ddrace_state: i32,
    /// `m_FreezeTime`.
    pub freeze_time: i32,
    /// `m_FrozenLastTick`.
    pub frozen_last_tick: bool,
    /// `m_TuneZone`.
    pub tune_zone: i32,
    /// `m_TuneZoneOld`.
    pub tune_zone_old: i32,
    /// `m_StartTime`.
    pub start_time: i32,
    /// `CEntity::m_Pos` — the *entity* position every other entity and every tile lookup sees,
    /// distinct from `CCharacterCore::m_Pos` ([`CharacterCore::pos`]): only `Spawn()` and
    /// `TickDeferred()` copy the core position into it, so it lags a core position moved *within*
    /// a tick (`HandleNinja`'s dash, a teleport) until that tick's deferred pass. [`World::step`]
    /// re-syncs it from the core at the start of every tick (the value `TickDeferred` left it at,
    /// unless something outside `step` moved the core).
    pub pos: Vec2<R>,
    /// `m_PrevPos` — the position `HandleTiles`'s anti-skip loop diffs against; lags the real
    /// position by one server tick (see this module's doc comment / `BUILD REPORT`).
    pub prev_pos: Vec2<R>,
    /// `m_TeleCheckpoint`.
    pub tele_checkpoint: i32,
    /// `m_TileIndex`.
    pub tile_index: i32,
    /// `m_TileFIndex`.
    pub tile_findex: i32,
    /// `m_LastRefillJumps`.
    pub last_refill_jumps: bool,
    /// `m_LastPenalty`.
    pub last_penalty: bool,
    /// `m_LastBonus`.
    pub last_bonus: bool,
    /// `m_TeleGunPos`.
    pub tele_gun_pos: Vec2<R>,
    /// `m_TeleGunTeleport`.
    pub tele_gun_teleport: bool,
    /// `m_IsBlueTeleGunTeleport`.
    pub is_blue_tele_gun_teleport: bool,
    /// `m_StrongWeakId`.
    pub strong_weak_id: i32,
    /// `m_SpawnTick`.
    pub spawn_tick: i32,
    /// `m_NumInputs` (`CCharacter`'s own counter — distinct from `CPlayer::m_NumInputs`).
    pub num_inputs: i32,
    /// `m_LastWeapon`.
    pub last_weapon: i32,
    /// `m_QueuedWeapon` (`-1` = no queued switch).
    pub queued_weapon: i32,
    /// `m_ReloadTimer`.
    pub reload_timer: i32,
    /// `m_AttackTick`.
    pub attack_tick: i32,
    /// `m_Health` — never compared directly, but its value gates `IncreaseHealth`'s return
    /// (unused here) and, via `Unfreeze`, `m_Armor`'s reset to `10`.
    pub health: i32,
    /// `m_Armor` — same scope note as `m_Health`.
    pub armor: i32,
    /// `m_NumObjectsHit` — ninja-hit counter (reset by `FireWeapon`'s `WEAPON_NINJA` case, bumped
    /// by `HandleNinja` per distinct hit tee).
    pub num_objects_hit: i32,
    /// `m_aHitObjects[..m_NumObjectsHit]` as a set of client ids (bit `c` = client `c`): the
    /// ninja's "already hit this tee" list. Only membership is ever read, so a bit set replaces the
    /// array (two `u64` words, not a `u128`: the latter would raise `Character`'s alignment to 16
    /// and its size on the hot path; bit `c` is word `c / 64`, bit `c % 64`).
    pub hit_objects: [u64; 2],

    // Input snapshots (`CNetObj_PlayerInput`, 10 fields each — see `core::PlayerInput`).
    /// `m_Input`.
    pub input: PlayerInput,
    /// `m_LatestInput`.
    pub latest_input: PlayerInput,
    /// `m_LatestPrevInput`.
    pub latest_prev_input: PlayerInput,
    /// `m_PrevInput`.
    pub prev_input: PlayerInput,
    /// `m_SavedInput`.
    pub saved_input: PlayerInput,
}

impl<R: Real> Default for Character<R> {
    fn default() -> Self {
        Character {
            alive: false,
            move_restrictions: 0,
            ddrace_state: 0,
            freeze_time: 0,
            frozen_last_tick: false,
            tune_zone: 0,
            tune_zone_old: 0,
            start_time: 0,
            pos: Vec2::zero(),
            prev_pos: Vec2::zero(),
            tele_checkpoint: 0,
            tile_index: 0,
            tile_findex: 0,
            last_refill_jumps: false,
            last_penalty: false,
            last_bonus: false,
            tele_gun_pos: Vec2::zero(),
            tele_gun_teleport: false,
            is_blue_tele_gun_teleport: false,
            strong_weak_id: 0,
            spawn_tick: 0,
            num_inputs: 0,
            last_weapon: WEAPON_HAMMER,
            queued_weapon: -1,
            reload_timer: 0,
            attack_tick: 0,
            health: 0,
            armor: 0,
            num_objects_hit: 0,
            hit_objects: [0; 2],
            input: PlayerInput::default(),
            latest_input: PlayerInput::default(),
            latest_prev_input: PlayerInput::default(),
            prev_input: PlayerInput::default(),
            saved_input: PlayerInput::default(),
        }
    }
}

// --- `CPlayer`'s spawn/respawn/kill bookkeeping (`player.h`) -----------------------------------

/// `CPlayer`-level state this crate needs: spawn/respawn/kill timing. `Default` matches a fresh
/// `CPlayer` (`CPlayer::Reset`, `player.cpp:49-156` — the fields this crate needs all start at
/// `Server()->Tick()` there, *not* zero; see [`Player::new`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    /// `m_DieTick`.
    pub die_tick: i32,
    /// `m_PreviousDieTick`.
    pub previous_die_tick: i32,
    /// `m_Spawning`: waiting for `TryRespawn` to succeed.
    pub spawning: bool,
    /// `m_WeakHookSpawn`: never set `true` by anything this crate drives (`CPlayer::Respawn`'s
    /// `WeakHook` parameter, called only from a chat command's dummy-spawn path — unreachable by
    /// this harness); kept so [`try_respawn`]'s branch structure mirrors `player.cpp` exactly.
    pub weak_hook_spawn: bool,
    /// `m_LastKill`: last tick `OnKillNetMessage` actually processed a kill for this player
    /// (`0` = never — matches a fresh `CPlayer`'s zero-initialized `int`, since `Reset()` doesn't
    /// set it either).
    pub last_kill: i32,
    /// `m_Team` — always `TEAM_GAME` once spawned in this harness (never spectator); kept for
    /// `CanSpawn`'s `Team == TEAM_SPECTATORS` guard.
    pub team: i32,
    /// `CPlayer::m_TuneZone` (`player.cpp:278-280`): the tune zone of the player's *view position*
    /// (`m_ViewPos`, the character's `m_Pos` as of the last [`player_tick`] it was alive in) —
    /// distinct from [`Character::tune_zone`], which `HandleTuneLayer` refreshes at the start of the
    /// character's own tick and which therefore lags one tick of movement behind this one.
    /// `CGameContext::CreateExplosion` reads *this* one for the owner's `explosion_strength`
    /// (`gamecontext.cpp:393-396`). `0` until the first alive tick, like a fresh `CPlayer`.
    pub tune_zone: i32,
    /// Mirrors `m_pCharacter != nullptr` — **not** the same thing as `world.characters[id]
    /// .alive`. A normal in-world death (`Die()`, called from `HandleSkippableTiles`/
    /// `HandleTiles`) sets `alive = false` but does *not* delete the C++ `CCharacter` object
    /// (and so does not touch this flag) until the *next* `CPlayer::Tick()` notices
    /// (`player.cpp:252-264`) — deferring any respawn attempt to (at the earliest) the *next*
    /// server tick. `OnKillNetMessage`'s `KillCharacter()` (`player.cpp:712-721`), by contrast,
    /// deletes the object immediately (clears this flag right away), which is exactly how a
    /// kill-bit death and its respawn can land in the *same* tick (`docs/formats.md`, finding
    /// F12) while a normal death never can. See [`die`] vs [`kill_character`].
    pub has_character: bool,
}

/// `TEAM_GAME` (any non-negative, non-`TEAM_SPECTATORS` value; DDNet's actual constant, from
/// `gamecontroller.h`).
pub const TEAM_GAME: i32 = 0;
/// `TEAM_SPECTATORS` (`gamecontroller.h`).
pub const TEAM_SPECTATORS: i32 = -1;

impl Player {
    /// `CPlayer::CPlayer`/`Reset()` at tick `now` (`m_DieTick = m_PreviousDieTick = Server()->
    /// Tick()`, `player.cpp:51-52` — *not* zero, since `CPlayer` (unlike `CCharacter`) is a plain
    /// heap `new`, not a zeroing pool allocator).
    pub fn new(now: i32) -> Self {
        Player {
            die_tick: now,
            previous_die_tick: now,
            spawning: false,
            weak_hook_spawn: false,
            last_kill: 0,
            team: TEAM_SPECTATORS,
            tune_zone: 0,
            has_character: false,
        }
    }
}

// --- `CProjectile` (`entities/projectile.h`/`.cpp`) — gun/grenade, and the
// `ENTITY_CRAZY_SHOTGUN[_EX]` map fixture (a real `CProjectile`, `WEAPON_SHOTGUN`-typed). --------

/// `CProjectile`. Generic over weapon type (`m_Type`) — this is the same struct/`tick` logic for
/// a player-fired gun/grenade shot and for an `ENTITY_CRAZY_SHOTGUN[_EX]` map fixture's
/// permanently-bouncing `WEAPON_SHOTGUN` shot (`gamecontroller.cpp:214-244`); only weapon-typed
/// tuning lookups (`GetPos`) and `m_Explosive`/`m_Freeze`/`m_Bouncing` flags actually vary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projectile<R: Real> {
    /// `m_Type` (`WEAPON_*`).
    pub weapon_type: i32,
    /// `m_Owner` (client id, `-1` = a map fixture's shot).
    pub owner: i32,
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_Direction`.
    pub direction: Vec2<R>,
    /// `m_InitDir`.
    pub init_dir: Vec2<R>,
    /// `m_LifeSpan` (`-1` = infinite, matching `m_Bouncing != 0` map fixtures).
    pub life_span: i32,
    /// `m_StartTick`.
    pub start_tick: i32,
    /// `m_Freeze`.
    pub freeze: bool,
    /// `m_Explosive`.
    pub explosive: bool,
    /// `m_Bouncing` (`0` = none, `1` = horizontal, `2` = vertical — `SetBouncing`).
    pub bouncing: i32,
    /// `m_TuneZone`, fixed at creation (`GetPos` uses `GetTuning(m_TuneZone)`, never
    /// recomputed).
    pub tune_zone: i32,
    /// `m_Layer` (`LAYER_GAME` unless spawned from the switch layer).
    pub layer: Layer,
    /// `m_Number` (switch number, for `m_Layer == LAYER_SWITCH` fixtures).
    pub number: i32,
    /// Marked for removal at the end of this tick (`m_MarkedForDestroy`; `CGameWorld::
    /// RemoveEntities`).
    pub marked_for_destroy: bool,
}

/// Capacity reserved once for [`World::lasers`] (the 1.6 nit "reserve the capacity of
/// `active_timed_switchers`/`projectiles` once"): 64 live lasers/beams/turret shots at the same
/// time is far more than a handful of tees ever produce on a block map; a turret-heavy map under a
/// crowd can exceed it, in which case the vector grows once by doubling (the only allocation
/// `World::step` can ever make). [`LaserList`] keeps the reservation across `Clone`.
pub const LASER_CAPACITY: usize = 64;
/// Extra capacity reserved once for [`World::projectiles`] beyond the map's own crazy-shotgun
/// fixtures: a gun is reloaded every ~8 ticks and a grenade every 15, with lifetimes of 2 s, so
/// a few tees keep well under this.
pub const PROJECTILE_HEADROOM: usize = 32;

/// `LAYER_GAME`/`LAYER_FRONT`/`LAYER_SWITCH` (`mapitems.h`) — the three variants any Stage A
/// entity actually needs (tele/speedup/tune layers never host a `CProjectile`/`CPickup`/fixture).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// `LAYER_GAME`.
    Game,
    /// `LAYER_FRONT`.
    Front,
    /// `LAYER_SWITCH`.
    Switch,
}

/// One `ENTITY_CRAZY_SHOTGUN[_EX]` scan hit (`gamecontroller.cpp:214-244`), collected during
/// [`World::from_map`]'s map scan and turned into a permanently-bouncing [`Projectile`] once the
/// scan finishes: `(pos, direction, explosive, bouncing, layer, switch_number)`.
type CrazyShotgunSpawn<R> = (Vec2<R>, Vec2<R>, bool, i32, Layer, u8);

// --- `CPickup` (`entities/pickup.h`/`.cpp`) -----------------------------------------------------

/// `CPickup::Type()` (this crate's own internal enum for pickup kinds — DDNet's `POWERUP_*`
/// values are a *network* enum this crate never serializes, so it doesn't need to match them
/// numerically, only distinguish the same cases `IGameController::OnEntity`/`CPickup::Tick` do).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickupKind {
    /// `POWERUP_FREEZE` (from `ENTITY_HEALTH_1` — DDRace repurposes the "health" pickup as a
    /// freeze trap).
    Freeze,
    /// `POWERUP_ARMOR` (`ENTITY_ARMOR_1`).
    Armor,
    /// `POWERUP_ARMOR_SHOTGUN`.
    ArmorShotgun,
    /// `POWERUP_ARMOR_GRENADE`.
    ArmorGrenade,
    /// `POWERUP_ARMOR_NINJA`.
    ArmorNinja,
    /// `POWERUP_ARMOR_LASER`.
    ArmorLaser,
    /// `POWERUP_WEAPON` (`SubType` = `WEAPON_*`).
    Weapon(i32),
    /// `POWERUP_NINJA`.
    Ninja,
}

/// `CPickup`. Map fixtures only in this harness (never dynamically created/destroyed — DDRace
/// pickups don't despawn on pickup, matching `pickup.cpp`'s lack of any `m_MarkedForDestroy`
/// write outside `Reset()`, which this crate's `World` never calls).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pickup<R: Real> {
    /// `m_Pos` — updated every tick divisible by 7 (`pickup_tick`'s own `Move()` port) by
    /// whatever `mcore` currently holds.
    pub pos: Vec2<R>,
    /// `Type()`.
    pub kind: PickupKind,
    /// `m_Layer`.
    pub layer: Layer,
    /// `m_Number`.
    pub number: i32,
    /// `CPickup::m_Core` (`pickup.cpp:18`, initial value `(0,0)`) — the mover-tile velocity a
    /// `TILE_CP`/`TILE_CP_F` tile last set (`Move()`, `pickup.cpp:191-198`,
    /// `CCollision::MoverSpeed`, `collision.cpp:791-826`). *Persists* once the pickup leaves the
    /// mover tile (`MoverSpeed` returns `0`/leaves its out-param untouched off such a tile, so
    /// the pickup keeps drifting at its last mover-assigned velocity forever, never reset to
    /// zero) — found empirically (a parallel diagnosis agent, against an instrumented Oracle B):
    /// several corpus maps have a freeze/armor pickup sitting on a `TILE_CP`/`TILE_CP_F`
    /// conveyor, oscillating back and forth every ~0.15s, and this crate's pickups never moved
    /// at all, so a character could be in/out of a moving pickup's radius at the wrong ticks
    /// (`ChillBlock5__seed10007`'s tick-136 freeze/velocity mismatch, `BlmapChill_seed10038`'s
    /// freeze-timing mismatch, `Blockdale__seed10001`'s `active_weapon` mismatch: an armor
    /// pickup that moved into range earlier than a static-position model ever would).
    pub mcore: Vec2<R>,
}

/// `CPickup::Tick()`'s own search radius term (`pickup.cpp:14,41`): `GetProximityRadius() +
/// ms_CollisionExtraSize` = `PICKUP_PHYSICS_RADIUS (14) + ms_CollisionExtraSize (6)` = `20`.
/// [`find_characters_in_range`] adds the *character's* own `m_ProximityRadius` (28) on top of
/// whatever is passed here, matching `CGameWorld::FindEntities`'s `Radius + pEnt->
/// m_ProximityRadius` check (`gameworld.cpp:58-77`) — so the combined search radius actually used
/// is `20 + 28 = 48`, not `14 + 28 = 42`. Found empirically: dozens of corpus traces showed a
/// character with fewer `weapon_got_mask` bits set than the reference, always a strict subset,
/// often from tick 0 — a character spawning 43-48 units from a weapon pickup (just past this
/// crate's old too-small 42-unit radius, but within the real server's 48-unit one) never picked
/// it up here.
pub const PICKUP_PROXIMITY_RADIUS: f32 = 20.0;

// --- `CDoor` map fixture (position only: it never ticks) -----------------------------------------

/// A Stage-A `CDoor` fixture: position only (see `docs/formats.md` §11.2, finding F6 — kind 2's
/// dynamic state, incl. `m_To`, isn't accessible to the real Oracle B harness either; only
/// `CEntity::m_Pos` is dumped/compared).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoorFixture<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
}

// --- `CGameTeams` (`teams.h`/`.cpp`) — DDRace-team-level state, simplified. ---------------------
//
// `m_aTeamLocked`/`m_aTeamFlock`/`m_aPractice` are not modeled at all: nothing this harness can
// drive ever sets any of the three to `true` (no chat commands exist in this scenario format;
// `sv_practice_by_default` never appears in the corpus, and even if it did it also requires
// `sv_testing_commands`, likewise absent). Every `CGameTeams` branch gated on one of them takes
// its "false" arm unconditionally here — see each function's doc comment for the specific
// citation. `m_apSaveTeamResult`/`GetSaving` (save/load-team) is the same story (only reachable
// via `/save`/`/load` chat commands) and is hardcoded `false` (never even given a field).

/// `ETeamState` (`team_state.h`) — only the values reachable without team-locking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeamState {
    /// `ETeamState::EMPTY`.
    Empty,
    /// `ETeamState::OPEN`.
    Open,
    /// `ETeamState::STARTED`.
    Started,
    /// `ETeamState::FINISHED`.
    Finished,
}

/// `CGameTeams`'s per-(DDRace team) and per-client state this crate needs.
#[derive(Debug, Clone)]
pub struct RaceTeams {
    team_state: [TeamState; MAX_CLIENTS + 1],
    tee_started: [bool; MAX_CLIENTS],
    tee_finished: [bool; MAX_CLIENTS],
}

impl Default for RaceTeams {
    fn default() -> Self {
        RaceTeams {
            team_state: [TeamState::Empty; MAX_CLIENTS + 1],
            tee_started: [false; MAX_CLIENTS],
            tee_finished: [false; MAX_CLIENTS],
        }
    }
}

impl RaceTeams {
    /// `CGameTeams::TeeFinished(int ClientId)` (`teams.cpp:1459-1462`).
    pub fn tee_finished(&self, client_id: i32) -> bool {
        self.tee_finished[client_id as usize]
    }

    /// `CGameTeams::GetTeamState(int Team)` (`teams.cpp:1464-1467`).
    pub fn team_state(&self, team: i32) -> TeamState {
        self.team_state[team as usize]
    }

    /// `CGameTeams::TeamSize(int Team)` (`teams.cpp:509`, `Team != TEAM_SUPER` case): how many
    /// players in this scenario (`world.characters[i].is_some()` — a slot this scenario ever
    /// registered a client id at, matching `CPlayer` existing, *not* `CCharacter`/`IsAlive()` —
    /// `CPlayer` outlives any one of its character's deaths) are on `team`.
    fn team_size<R: Real>(&self, teams: &TeamsCore, characters: &[Option<Character<R>>], team: i32) -> i32 {
        (0..MAX_CLIENTS as i32)
            .filter(|&i| characters[i as usize].is_some() && teams.team(i) == team)
            .count() as i32
    }

    /// `CGameTeams::OnCharacterSpawn(int ClientId)` (`teams.cpp:1277-1294`), with
    /// `m_aTeamLocked`/`m_aTeamFlock` hardcoded `false` (see this module's section doc comment):
    /// unconditionally resets `ClientId`'s DDRace team to [`TEAM_FLOCK`] (or, under
    /// `sv_team == SV_TEAM_FORCED_SOLO`, to `ClientId` itself) via [`set_force_character_team`],
    /// then (since `!m_aTeamFlock[Team]` is always true) checks whether that team just finished.
    /// Returns whether a genuine team change occurred (see
    /// [`set_force_character_team`]'s return value), for [`spawn_character`] to apply
    /// [`World::team_changed_this_pass`]'s side effects.
    #[allow(clippy::too_many_arguments)]
    pub fn on_character_spawn<R: Real>(
        &mut self,
        teams: &mut TeamsCore,
        switchers: &mut [Switcher],
        characters: &mut [Option<Character<R>>],
        client_id: i32,
        sv_team_forced_solo: bool,
    ) -> bool {
        teams.set_solo(client_id, false);
        let target = if sv_team_forced_solo { client_id } else { TEAM_FLOCK };
        let changed =
            self.set_force_character_team(teams, switchers, characters, client_id, target, sv_team_forced_solo);
        self.check_team_finished(teams, characters, teams.team(client_id));
        changed
    }

    /// `CGameTeams::SetForceCharacterTeam` (`teams.cpp:462-507`), with locking/flocking removed
    /// (see this module's section doc comment): unsets started/finished for `client_id`, moves
    /// it to `team` in `teams`, and — only when `team`'s state is [`TeamState::Empty`] (its very
    /// first use) — opens the team and resets every switcher's status for it back to `initial`
    /// (`ResetSwitchers`, `teams.cpp:75-83`). When `client_id` was the *last* character in its
    /// old team, that old team also empties and gets its own switchers reset the same way
    /// (`ResetRoundState(OldTeam)` -> `ResetSwitchers(OldTeam)`, `teams.cpp:471-479` — every other
    /// effect of `ResetRoundState` is locking/practice/vote/swap bookkeeping this crate's scope
    /// doesn't model, per this module's section doc comment; found empirically missing entirely
    /// in an earlier revision of this port — see this crate's `BUILD REPORT`). `sv_team_forced_solo`
    /// widens the old-team-emptying gate exactly like the real `(OldTeam != TEAM_FLOCK ||
    /// g_Config.m_SvTeam == SV_TEAM_FORCED_SOLO)` term does: under forced solo, even a
    /// `TEAM_FLOCK` character leaving can empty that "team".
    ///
    /// Returns `Team != OldTeam` (`teams.cpp:486`) — real DDNet gates an unconditional
    /// `m_pGameContext->m_World.RemoveEntitiesFromPlayer(ClientId)` call (`teams.cpp:497`) on
    /// exactly this condition, whenever this function is called from *any* path (character
    /// spawn or death). This crate doesn't have access to `World::projectiles` from inside
    /// `RaceTeams` (a separate, `World`-independent struct — see this module's section doc
    /// comment), so every caller of this method is responsible for applying that removal
    /// itself using this return value; see [`die`], [`spawn_character`], and the free-standing
    /// [`set_force_character_team`] wrapper.
    #[allow(clippy::too_many_arguments)]
    pub fn set_force_character_team<R: Real>(
        &mut self,
        teams: &mut TeamsCore,
        switchers: &mut [Switcher],
        characters: &[Option<Character<R>>],
        client_id: i32,
        team: i32,
        sv_team_forced_solo: bool,
    ) -> bool {
        self.tee_started[client_id as usize] = false;
        self.tee_finished[client_id as usize] = false;
        let old_team = teams.team(client_id);
        if team != old_team
            && (old_team != TEAM_FLOCK || sv_team_forced_solo)
            && old_team != teams.team_super()
            && self.team_state[old_team as usize] != TeamState::Empty
            && self.team_size(teams, characters, old_team) <= 1
        {
            self.team_state[old_team as usize] = TeamState::Empty;
            reset_switchers_for_team(switchers, old_team);
        }
        teams.set_team(client_id, team);
        if team != teams.team_super() && self.team_state[team as usize] == TeamState::Empty {
            self.team_state[team as usize] = TeamState::Open;
            reset_switchers_for_team(switchers, team);
        }
        team != old_team
    }

    /// `CGameTeams::OnCharacterStart(int ClientId)` (`teams.cpp:85-201`), with locking/flocking
    /// removed and every chat-only branch (`SendChatTarget`/`SendChatTeam`) dropped (no
    /// observable physics effect). `sv_team_forced_solo`: `g_Config.m_SvTeam ==
    /// SV_TEAM_FORCED_SOLO`.
    pub fn on_character_start<R: Real>(
        &mut self,
        teams: &TeamsCore,
        characters: &mut [Option<Character<R>>],
        client_id: i32,
        tick: i32,
        sv_team_forced_solo: bool,
    ) {
        let Some(starting) = characters[client_id as usize].as_mut() else {
            return;
        };
        let team = teams.team(client_id);
        if sv_team_forced_solo && starting.ddrace_state == 1 {
            return;
        }
        if (sv_team_forced_solo || team != TEAM_FLOCK) && starting.ddrace_state == 3 {
            return;
        }
        if !sv_team_forced_solo && (team == TEAM_FLOCK || team == teams.team_super()) {
            self.tee_started[client_id as usize] = true;
            starting.ddrace_state = 1;
            starting.start_time = tick;
            return;
        }
        // Non-flock, non-solo team: wait for every teammate that already finished (and hasn't
        // restarted) — `teams.cpp:106-147` (chat messages dropped).
        let mut waiting = false;
        for i in 0..MAX_CLIENTS as i32 {
            if teams.team(client_id) != teams.team(i) {
                continue;
            }
            let Some(other) = characters[i as usize] else { continue };
            if !other.alive || other.ddrace_state != 3 {
                continue;
            }
            waiting = true;
        }
        if waiting {
            characters[client_id as usize].as_mut().unwrap().ddrace_state = 0;
            return;
        }
        self.tee_started[client_id as usize] = true;
        if self.team_state[team as usize] != TeamState::Started {
            self.team_state[team as usize] = TeamState::Started;
            for i in 0..MAX_CLIENTS as i32 {
                if teams.team(i) != team {
                    continue;
                }
                if let Some(c) = characters[i as usize].as_mut() {
                    self.tee_started[i as usize] = true;
                    c.ddrace_state = 1;
                    c.start_time = tick;
                }
            }
        }
    }

    /// `CGameTeams::OnCharacterFinish(int ClientId)` (`teams.cpp:203-227`), with flocking removed
    /// and score/chat bookkeeping (`OnFinish`) dropped — the only physics-observable effect
    /// either branch has is `SetDDRaceState(..., ERaceState::FINISHED)`, which this reproduces
    /// directly.
    pub fn on_character_finish<R: Real>(
        &mut self,
        teams: &TeamsCore,
        characters: &mut [Option<Character<R>>],
        client_id: i32,
        sv_team_forced_solo: bool,
    ) {
        let team = teams.team(client_id);
        if (team == TEAM_FLOCK && !sv_team_forced_solo) || team == teams.team_super() {
            if let Some(c) = characters[client_id as usize].as_mut() {
                c.ddrace_state = 3;
            }
            return;
        }
        if self.tee_started[client_id as usize] {
            self.tee_finished[client_id as usize] = true;
        }
        self.check_team_finished(teams, characters, team);
    }

    /// `CGameTeams::OnCharacterDeath(int ClientId, int Weapon)` (`teams.cpp:1296-1385`), with
    /// team-locking removed (`Locked = TeamLocked(Team) && Weapon != WEAPON_GAME` is always
    /// `false` — see this module's section doc comment) and chat-only bookkeeping dropped: under
    /// `sv_team == SV_TEAM_FORCED_SOLO`, only marks the team `Open` and resets its round state
    /// (switchers back to `initial`); otherwise resets the character's team to [`TEAM_FLOCK`]
    /// (`SetForceCharacterTeam`) and, since `!m_aTeamFlock[Team]` is always true, checks whether
    /// the team the character *left* just finished.
    ///
    /// Returns whether [`set_force_character_team`] reported a genuine team change (`false`
    /// under the `sv_team_forced_solo` branch, which — matching real DDNet exactly, see
    /// `teams.cpp:1305-1323` — never calls it at all). [`die`] uses this to apply
    /// [`World::team_changed_this_pass`]'s side effects (`teams.cpp:497`'s
    /// `RemoveEntitiesFromPlayer`, and the character-loop cutoff).
    pub fn on_character_death<R: Real>(
        &mut self,
        teams: &mut TeamsCore,
        switchers: &mut [Switcher],
        characters: &mut [Option<Character<R>>],
        client_id: i32,
        sv_team_forced_solo: bool,
    ) -> bool {
        teams.set_solo(client_id, false);
        let team = teams.team(client_id);
        if sv_team_forced_solo && team != teams.team_super() {
            self.team_state[team as usize] = TeamState::Open;
            reset_switchers_for_team(switchers, team);
            false
        } else {
            let changed =
                self.set_force_character_team(teams, switchers, characters, client_id, TEAM_FLOCK, sv_team_forced_solo);
            self.check_team_finished(teams, characters, team);
            changed
        }
    }

    /// `CGameTeams::CheckTeamFinished`/`TeamFinished` (`teams.cpp:328-389,584-594`), with
    /// practice mode removed (always takes the non-practice branch) and score/chat/save
    /// bookkeeping (`OnFinish`/`OnTeamFinish`) dropped.
    fn check_team_finished<R: Real>(&mut self, teams: &TeamsCore, characters: &mut [Option<Character<R>>], team: i32) {
        if self.team_state[team as usize] != TeamState::Started {
            return;
        }
        for i in 0..MAX_CLIENTS as i32 {
            if teams.team(i) == team && !self.tee_finished[i as usize] {
                return;
            }
        }
        for i in 0..MAX_CLIENTS as i32 {
            if teams.team(i) == team {
                self.tee_started[i as usize] = false;
                self.tee_finished[i as usize] = false;
                if let Some(c) = characters[i as usize].as_mut() {
                    c.ddrace_state = 3;
                }
            }
        }
        self.team_state[team as usize] = TeamState::Finished;
    }
}

/// `CGameTeams::ResetSwitchers(int Team)` (`teams.cpp:75-83`): every switcher's status for `team`
/// reverts to its `initial` value, `end_tick` clears, and `kind` resets to `TILE_SWITCHOPEN`.
fn reset_switchers_for_team(switchers: &mut [Switcher], team: i32) {
    for switcher in switchers.iter_mut() {
        switcher.status[team as usize] = switcher.initial;
        switcher.end_tick[team as usize] = 0;
        switcher.kind[team as usize] = map::TILE_SWITCHOPEN as i32;
    }
}

// --- `World<R>`: everything above, tied together. -----------------------------------------------

/// The server-level DDRace world. See the module doc comment for scope. `Clone` gives a full
/// snapshot suitable for save/restore (acceptance criterion 1): every genuinely *mutable* field
/// is deep-copied (`Copy`/plain data, or a `Vec`/`Box` with its own heap allocation, so nothing
/// mutable is shared between a clone and its original), while [`World::collision`] — the one
/// large, immutable-after-construction field — is reference-counted (`Arc`) instead. Review
/// round 1, finding F10: an earlier revision of this port deep-copied `collision` too, making a
/// single `World::clone()` cost milliseconds and megabytes on a large map (measured: ~3.25 ms /
/// 21.7 MB for `BlmapChill`) — roughly 350 `World::step` calls' worth of cost, making
/// save/restore-heavy use (a planner's search, or an arena that resets between rounds)
/// impractical.
///
/// Task 1.10: `collision` aside, `.clone()` is still `O(map size)`, not `O(character count)` —
/// several fields (`pickups`/`fixtures`/`doors`/`projectiles`/`tuning`) scale with the *map*
/// (`BlmapChill`: 140/89/102/12 vs. `CopyLoveBox`'s 47/0/0/0 — measured, this task's `BUILD
/// REPORT`), not the character count, so a search loop calling `.clone()` (or worse, assigning
/// `*scratch = source.clone()`) once per rollout still pays that cost every time. Use
/// [`World::restore_from`] instead for that access pattern: same result, but every `Vec`/`Box`
/// field reuses its *destination*'s existing allocation once grown to fit, so repeated restores
/// against one scratch `World` allocate nothing at all (measured effect: same `BUILD REPORT`).
#[derive(Debug, Clone)]
pub struct World<R: Real> {
    /// The task-1.3 core array (`CCharacterCore` + id bookkeeping), always sized to
    /// [`MAX_CLIENTS`] capacity.
    pub cores: WorldCore<R, MAX_CLIENTS>,
    /// The loaded map's collision (task 1.3/1.4) — mutated only at world construction (door
    /// collision setup), never during a tick, so every `World` produced from the same
    /// [`World::from_map`] call (a fresh load, or a `Clone`) can safely share one `Arc`'d copy
    /// instead of each holding its own (review round 1, finding F10). `Deref`s transparently to
    /// `&Collision<R>`, so every existing read-only call site (`world.collision.method(...)`, or
    /// a function taking `&Collision<R>` and called with `&world.collision`) needed no change.
    pub collision: std::sync::Arc<Collision<R>>,
    /// Low-level per-client team/solo state (`CTeamsCore`, task 1.3).
    pub teams_core: TeamsCore,
    /// DDRace-team-level state (`CGameTeams`, simplified — see [`RaceTeams`]).
    pub race_teams: RaceTeams,
    /// DDRace-level per-character state, indexed by client id directly (`Some` for every client
    /// id this scenario ever registers, for its entire duration — see [`Character`]'s doc
    /// comment on why entries never need to become `None` again after a death).
    pub characters: [Option<Character<R>>; MAX_CLIENTS],
    /// `CPlayer`-level state, same indexing convention as [`World::characters`].
    pub players: [Option<Player>; MAX_CLIENTS],
    /// Character client ids in `CGameWorld`'s entity-list order (index `0` = most recently
    /// (re)spawned = processed first this tick — see the module doc comment / `docs/formats.md`
    /// §5.1).
    pub entity_order: Vec<u8>,
    /// Scratch storage for [`World::for_each_in_entity_order`]'s stable per-pass snapshot of
    /// [`World::entity_order`] — reused (never shrunk, only `clear()`ed) across calls instead of
    /// a fresh `entity_order.clone()` each time, for zero heap allocations in steady state
    /// (acceptance criterion 1; review round 1, finding F9). Always empty except while a
    /// `for_each_in_entity_order` call has temporarily taken it via `std::mem::take` — a caller
    /// outside that method must never read this field.
    entity_order_scratch: Vec<u8>,
    /// Scratch storage for `find_characters_in_range_into`'s result, reused (never shrunk, only
    /// `clear()`ed) the same way [`World::entity_order_scratch`] is, by every hot-path caller
    /// (`pickup_tick`, `projectile_tick`'s freeze branch, `create_explosion`) instead of each
    /// allocating its own fresh `Vec` via `find_characters_in_range` every call (review round 1,
    /// finding F9). Always empty except while temporarily taken via `std::mem::take`.
    range_scratch: Vec<i32>,
    /// Scratch storage for `fire_hammer`'s two-pass target list (found, then hit) — reused the
    /// same way [`World::range_scratch`] is (review round 1, finding F9). Always empty except
    /// while temporarily taken via `std::mem::take`.
    hammer_scratch: Vec<usize>,
    /// Scratch storage for `Collision::get_map_indices_into`'s result (the DDRace anti-skip
    /// tile-visiting loop, `ddrace_post_core_tick`) — reused the same way
    /// [`World::range_scratch`] is (review round 2, finding F9 \[CONFIRMED\]: called every tick
    /// a character is on or passes over any `tile_exists` tile — freeze/speedup/stopper/tele/
    /// switch/kill/etc, not a rare event, unlike `can_spawn`'s own allocation). Always empty
    /// except while temporarily taken via `std::mem::take`.
    map_indices_scratch: Vec<i32>,
    /// Live projectiles (gun/grenade shots, and `ENTITY_CRAZY_SHOTGUN[_EX]` fixtures), in
    /// entity-list order (index `0` = newest).
    pub projectiles: Vec<Projectile<R>>,
    /// Set whenever [`RaceTeams::set_force_character_team`] reports a genuine team change
    /// (`Team != OldTeam`, `teams.cpp:486`), which in real DDNet unconditionally calls
    /// `CGameWorld::RemoveEntitiesFromPlayer(ClientId)` (`teams.cpp:497`). That function
    /// (`gameworld.cpp:160-182`) walks *every* entity type with its own fresh sub-loop, but
    /// writes the result into the *same* `m_pNextTraverseEntity` member `CGameWorld::Tick()`'s
    /// own per-type sub-loop uses to advance (`gameworld.cpp:225-231`). When a mid-loop
    /// `Die()` call (reached from a character's own `Tick()`, via `HandleSkippableTiles`)
    /// triggers this, `RemoveEntitiesFromPlayer` leaves `m_pNextTraverseEntity == nullptr`
    /// once it finishes its own full sweep — so the *outer* character sub-loop's `pEnt =
    /// m_pNextTraverseEntity;` reads `nullptr` and ends right there, silently skipping every
    /// character still left in that pass, for this tick only (the separate `TickDeferred()`
    /// pass re-fetches its own list head fresh and is unaffected). This crate models only the
    /// `ENTTYPE_CHARACTER` sub-loop instance of this bug (the one with verified trace
    /// evidence — see `World::world_tick`'s character loop): [`die`] and
    /// [`set_force_character_team`] (both call sites: [`RaceTeams::on_character_death`] and
    /// [`RaceTeams::on_character_spawn`]) set this flag whenever they report a team change;
    /// `World::world_tick` resets it to `false` immediately before that same loop starts (so a
    /// team change from *outside* the loop — the kill-message path, or respawn bookkeeping,
    /// both of which run in their own separate passes — has no effect on it) and checks it
    /// after each iteration, stopping the pass early once set.
    pub team_changed_this_pass: bool,
    /// Map-fixture pickups (never created/destroyed after world construction).
    pub pickups: Vec<Pickup<R>>,
    /// `CDoor` fixtures — position only (see [`DoorFixture`]). Reference-counted (task 1.10b,
    /// speed-up 4): never mutated after [`World::from_map`] (`CDoor` has no `Tick()` override at
    /// all — see [`Fixture`]'s for the contrast with `CDragger`/`CGun`/
    /// `CLight`, which *do* drift and so stay a plain owned `Vec` below), so a `World::clone()`/
    /// `World::restore_from()` that never touches it can share one allocation instead of copying
    /// it — same idea as [`World::collision`]/`TuningList`'s own `zones` field.
    pub doors: std::sync::Arc<Vec<DoorFixture<R>>>,
    /// The static, ticking members of DDNet's `ENTTYPE_LASER` list — draggers, turrets and
    /// lights — in list order (index `0` = list head = created last by the map scan, ticked
    /// first). They sit *behind* every entity of [`World::lasers`] in the real list (those are
    /// only ever inserted at the head), so a tick visits [`World::lasers`] newest-first, then this
    /// vector front to back. Never grows or shrinks after [`World::from_map`].
    pub fixtures: Vec<Fixture<R>>,
    /// `CDragger::m_aTargetIdInTeam`/beam registry, one per dragger ([`Dragger::state`]).
    pub dragger_states: Vec<DraggerState>,
    /// `CGun::m_aLastFireTeam`/`m_aLastFireSolo`, one per turret ([`Gun::state`]).
    pub gun_states: Vec<GunState>,
    /// The dynamic members of the `ENTTYPE_LASER` list: lasers (rifle and shotgun shots), dragger
    /// beams and turret shots. **Stored oldest first**: the real list inserts at the head, so the
    /// head is the *last* element here (`push` = `InsertEntity`); a tick and the trace-b dump walk
    /// it back to front. Entities are only ever removed in bulk at the end of a tick
    /// (`CGameWorld::RemoveEntities`), so indices stay valid during a pass. Capacity is reserved
    /// once ([`LASER_CAPACITY`]); growing past it allocates (amortized doubling).
    pub lasers: LaserList<R>,
    /// `CGameContext::m_aTuningList`.
    pub tuning: TuningList,
    /// Server settings this crate models (see [`ServerConfig`]).
    pub config: ServerConfig,
    /// `IServer::Tick()` — this world's current game tick, incremented by [`World::step`].
    pub tick: i32,
    /// Default-type spawn points (`m_avSpawnPoints[SPAWNTYPE_DEFAULT]`, `ENTITY_SPAWN`), in
    /// map-scan order (row-major, `y` outer — matching `CreateAllEntities`). Reference-counted
    /// (task 1.10b, speed-up 4): set once in [`World::from_map`], read-only ever after (no
    /// in-crate code writes it outside a test directly poking the field — see this field's own
    /// note in the task's `BUILD REPORT` on why that stays source-compatible via `Arc::new`/
    /// `Arc::make_mut`) — same "never diverges, so share instead of copy" reasoning as
    /// [`World::doors`].
    pub spawn_points: std::sync::Arc<Vec<Vec2<R>>>,
    /// `m_avSpawnPoints[SPAWNTYPE_RED]` (`ENTITY_SPAWN_RED`), same ordering. Empty for every map
    /// this crate has seen so far but, unlike an earlier revision of this port, no longer
    /// assumed to *always* be empty: see `World::evaluate_spawn_type`'s doc comment for why
    /// concatenating this into [`World::spawn_points`] was a real, found-empirically bug.
    pub spawn_points_red: std::sync::Arc<Vec<Vec2<R>>>,
    /// `m_avSpawnPoints[SPAWNTYPE_BLUE]` (`ENTITY_SPAWN_BLUE`); see [`World::spawn_points_red`].
    pub spawn_points_blue: std::sync::Arc<Vec<Vec2<R>>>,
    /// The map's own embedded "Settings" strings (rawmap §1), copied out of the source
    /// [`crate::map::MapData`] at [`World::from_map`] time so [`World::init`] can apply them at
    /// the exact point `CGameContext::OnInit()` calls `LoadMapSettings()`
    /// (`gamecontext.cpp:4159`) without the caller having to keep the original `MapData` around.
    /// Reference-counted (task 1.10b, speed-up 4): its *content* never changes after
    /// `World::from_map` (`World::init` briefly clones this `Arc`, not the `Vec` it points to, to
    /// satisfy the borrow checker while applying each line — see `World::init`'s own body).
    pub map_settings: std::sync::Arc<Vec<String>>,
    /// Human-readable warnings for rejected commands (currently: a `CFGFLAG_GAME` config
    /// variable write attempted after [`World::init`] locked it — see [`apply_command`]'s doc
    /// comment) and any other non-fatal command-application diagnostics this crate ever adds.
    /// Mirrors the real server's `log_error("config", ...)` calls, which this harness has no
    /// stdout/log stream to send to instead.
    pub command_log: Vec<String>,
    /// Perf side list (task 1.6, coordinator follow-up item 3): indices into `cores.switchers`
    /// with at least one team currently in a timed kind (`TILE_SWITCHTIMEDOPEN`/
    /// `TILE_SWITCHTIMEDCLOSE`) — see [`switch::tick_switch_expiry`]'s doc comment for why this
    /// exists (it turns that function's per-tick cost from `O(every switch on the map)` into
    /// `O(switches actually touched)`). Maintained by `handle_switch_tiles` (private, pushes) and
    /// `tick_switch_expiry` itself (drops an index once none of its teams are timed any more) —
    /// never read/written anywhere else.
    pub active_timed_switchers: Vec<u8>,
}

/// Whether `sv_team == SV_TEAM_FORCED_SOLO` — a tiny helper so call sites read as the C++
/// condition they mirror.
fn sv_team_forced_solo(config: &ServerConfig) -> bool {
    config.sv_team == SV_TEAM_FORCED_SOLO
}

// --- Map scan: `CGameContext::CreateAllEntities`/`IGameController::OnEntity`
// (`gamecontext.cpp:4273-4365`, `gamecontroller.cpp:181-395`) --------------------------------

impl<R: Real> World<R> {
    /// Builds a fresh `World` from a loaded map: scans the game/front/switch layers exactly like
    /// `CreateAllEntities(true)` does (row-major, `y` outer — `gamecontext.cpp:4279-4281`),
    /// registering spawn points, pickups, `CDoor` collision, and Stage-B fixture positions, and
    /// applying the map-wide global tile effects (`TILE_OLDLASER`/`NPC`/`EHOOK`/`NOHIT`/`NPH`,
    /// `gamecontext.cpp:4288-4312`). `seed`: the same value the Oracle B harness reseeds
    /// `CWorldCore::m_pPrng` from right after `OnInit()` (`docs/formats.md` §12.2) —
    /// `[seed, seed ^ 0x9E3779B97F4A7C15]`.
    pub fn from_map(map: &MapData, seed: u64) -> Self {
        let mut collision: Collision<R> = Collision::new(map);
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let highest_switch_number = collision.highest_switch_number();
        let mut switchers: Vec<Switcher> = if highest_switch_number > 0 {
            (0..=highest_switch_number).map(|_| Switcher::default()).collect()
        } else {
            Vec::new()
        };
        for s in switchers.iter_mut() {
            s.initial = true;
            for j in 0..NUM_DDRACE_TEAMS as usize {
                s.status[j] = true;
                s.end_tick[j] = 0;
                s.kind[j] = 0;
                s.last_update_tick[j] = 0;
            }
        }

        // `[SPAWNTYPE_DEFAULT, SPAWNTYPE_RED, SPAWNTYPE_BLUE]` — see [`scan_entity`]'s doc
        // comment on why these stay separate through the scan, concatenated into
        // `World::spawn_points` (in that exact order) only afterward.
        let mut spawn_points_by_type: [Vec<Vec2<R>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut pickups: Vec<Pickup<R>> = Vec::new();
        let mut doors: Vec<DoorFixture<R>> = Vec::new();
        let mut fixture_scan: FixtureScan<R> = FixtureScan::default();
        let mut crazy_shotguns: Vec<CrazyShotgunSpawn<R>> = Vec::new();

        let width = map.width as i32;
        let height = map.height as i32;
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                let game_raw = map.game[index].index;
                scan_global_tile(game_raw, &mut config, &mut tuning);
                if game_raw >= map::ENTITY_OFFSET {
                    scan_entity(
                        &mut collision,
                        game_raw - map::ENTITY_OFFSET,
                        map.game[index].flags,
                        x,
                        y,
                        Layer::Game,
                        0,
                        map,
                        &mut spawn_points_by_type,
                        &mut pickups,
                        &mut doors,
                        &mut fixture_scan,
                        &mut crazy_shotguns,
                    );
                }
                if let Some(front) = &map.front {
                    let front_raw = front[index].index;
                    scan_global_tile(front_raw, &mut config, &mut tuning);
                    if front_raw >= map::ENTITY_OFFSET {
                        scan_entity(
                            &mut collision,
                            front_raw - map::ENTITY_OFFSET,
                            front[index].flags,
                            x,
                            y,
                            Layer::Front,
                            0,
                            map,
                            &mut spawn_points_by_type,
                            &mut pickups,
                            &mut doors,
                            &mut fixture_scan,
                            &mut crazy_shotguns,
                        );
                    }
                }
                if let Some(sw) = &map.switch {
                    let sw_raw = sw[index].kind;
                    if sw_raw >= map::ENTITY_OFFSET {
                        scan_entity(
                            &mut collision,
                            sw_raw - map::ENTITY_OFFSET,
                            sw[index].flags,
                            x,
                            y,
                            Layer::Switch,
                            sw[index].number,
                            map,
                            &mut spawn_points_by_type,
                            &mut pickups,
                            &mut doors,
                            &mut fixture_scan,
                            &mut crazy_shotguns,
                        );
                    }
                }
            }
        }

        // The scan above places `CDoor` collision (`scan_entity` -> `place_door_collision` ->
        // `set_door_collision_at`) *after* `Collision::new()` already built its own initial
        // `tile_exists_cache` from the pre-door-placement `door` layer. Task 1.10b, finding F1:
        // `set_door_collision_at` now keeps `tile_exists_cache` in sync itself (recomputing the
        // touched cell and its `tile_exists_next` neighbors on every call — see its own doc
        // comment), so an explicit whole-map `recompute_tile_exists_cache()` call here is no
        // longer needed (task 1.10 had one; removed as redundant, not merely dead, per the
        // review — every door placement above already left the cache correct).
        // Task 3.6: the door cells just placed changed a layer the derived fast-path tables read.
        collision.recompute_derived();
        let switcher_count = switchers.len();
        let mut cores: WorldCore<R, MAX_CLIENTS> = WorldCore::new();
        cores.switchers = switchers;
        cores.prng = Some({
            let mut p = Prng::new();
            p.seed([seed, seed ^ 0x9E3779B97F4A7C15]);
            p
        });

        // `doors`/`fixtures` (`CDoor`/`CDragger`/`CGun`/`CLight`, entity-dump kinds 2/3/5/6) were
        // pushed in the map scan's own row-major order above, but each one's real constructor
        // self-inserts via `CEntity`'s -> `GameWorld()->InsertEntity(this)`, prepending to its
        // type's list head every time (the exact same mechanism [`World::entity_order`]/crazy-
        // shotgun projectiles below already account for) — so the *last*-scanned one of each kind
        // ends up *first* in the dumped entity order. Reverse both here, once, rather than at
        // every one of `scan_entity`'s several `push` call sites. Found empirically: the
        // reference's kind-2/3/5/6 dump order was the exact reverse of this crate's own
        // (`ents0.txt`-style entity dumps, this crate's `BUILD REPORT`).
        doors.reverse();
        fixture_scan.fixtures.reverse();

        // `ENTITY_CRAZY_SHOTGUN[_EX]` (`gamecontroller.cpp:214-244`): a permanently-bouncing
        // `WEAPON_SHOTGUN` `CProjectile`, `m_LifeSpan == -2` (never decremented — `Tick()` only
        // decrements `> -1` values — so never destroyed via the life-span path either).
        //
        // `CreateAllEntities`'s map scan visits these in row-major order and calls `new
        // CCrazyShotgun(...)` for each as it goes, which self-inserts via `CEntity`'s constructor
        // -> `GameWorld()->InsertEntity(this)`, prepending to the type's list head every time
        // (matching `World::entity_order`'s own "index 0 = newest" convention — see its doc
        // comment). So the *last*-scanned fixture ends up *first* in the dumped entity order, not
        // the first-scanned one: iterate `crazy_shotguns` in reverse here (an earlier revision of
        // this port pushed in scan order, i.e. oldest-first — found empirically against
        // BlmapChill's 12 fixtures, dumped reference-oldest-last at tick 0).
        let mut projectiles: Vec<Projectile<R>> = Vec::with_capacity(crazy_shotguns.len() + PROJECTILE_HEADROOM);
        for (pos, direction, explosive, bouncing, layer, number) in crazy_shotguns.into_iter().rev() {
            let tune_zone = collision.is_tune(collision.get_map_index(pos));
            projectiles.push(Projectile {
                weapon_type: WEAPON_SHOTGUN,
                owner: -1,
                pos,
                direction,
                init_dir: direction,
                life_span: -2,
                start_tick: 0,
                freeze: true,
                explosive,
                bouncing,
                tune_zone,
                layer,
                number: number as i32,
                marked_for_destroy: false,
            });
        }

        World {
            cores,
            collision: std::sync::Arc::new(collision),
            teams_core: TeamsCore::new(),
            race_teams: RaceTeams::default(),
            characters: [None; MAX_CLIENTS],
            players: [None; MAX_CLIENTS],
            entity_order: Vec::with_capacity(MAX_CLIENTS),
            entity_order_scratch: Vec::with_capacity(MAX_CLIENTS),
            range_scratch: Vec::with_capacity(MAX_CLIENTS),
            hammer_scratch: Vec::with_capacity(MAX_CLIENTS),
            map_indices_scratch: Vec::with_capacity(8),
            team_changed_this_pass: false,
            projectiles,
            pickups,
            doors: std::sync::Arc::new(doors),
            fixtures: fixture_scan.fixtures,
            dragger_states: fixture_scan.dragger_states,
            gun_states: fixture_scan.gun_states,
            lasers: LaserList::new(),
            tuning,
            config,
            tick: 0,
            spawn_points: std::sync::Arc::new(spawn_points_by_type[0].clone()),
            spawn_points_red: std::sync::Arc::new(spawn_points_by_type[1].clone()),
            spawn_points_blue: std::sync::Arc::new(spawn_points_by_type[2].clone()),
            map_settings: std::sync::Arc::new(map.settings.clone()),
            command_log: Vec::new(),
            active_timed_switchers: Vec::with_capacity(switcher_count),
        }
    }

    /// Models `CGameContext::OnInit()`'s config/tuning/switcher setup (`gamecontext.cpp:4066-
    /// 4185`), run exactly once, in this exact order, between [`World::from_map`] and any
    /// character spawning:
    ///
    /// 1. `pre_init_cfg` (the pre-`OnInit()` `--cfg`/scenario-cfg pass, `docs/formats.md` §12.1)
    ///    is applied — every config variable, including `CFGFLAG_GAME` ones, is still writable.
    /// 2. The unconditional per-tune-zone reset (`gamecontext.cpp:4108-4120`: every zone ->
    ///    `DEFAULT` plus `gun_curvature=0,gun_speed=1400,shotgun_curvature=0,shotgun_speed=500,
    ///    shotgun_speeddiff=0`) — this *wipes* anything `pre_init_cfg` just set via `tune`/
    ///    `tune_zone`. `sv_tune_reset`'s own branch (`gamecontext.cpp:4126-4137`) is not
    ///    separately modeled: both its branches are strict subsets of this same reset (see
    ///    [`ServerConfig::sv_tune_reset`]'s doc comment) — provably a no-op to add.
    /// 3. If `sv_ddrace_tune_reset` (default `true`): reset `sv_hit=true`,
    ///    `sv_endless_drag=false`, `sv_old_laser=false`, `sv_old_teleport_hook=false`,
    ///    `sv_old_teleport_weapons=false`, `sv_teleport_hold_hook=false`,
    ///    `sv_team=SV_TEAM_ALLOWED`, `sv_show_others_default=0`, and every switcher's
    ///    `initial=true` (`gamecontext.cpp:4139-4154`) — this *is* observable (unlike step 2's
    ///    redundant `sv_tune_reset` gate), and specifically wipes whatever `pre_init_cfg` just
    ///    set `sv_hit`/etc. to.
    /// 4. `map_settings` (`LoadMapSettings()`, `gamecontext.cpp:4159`) is applied — still fully
    ///    writable (this is what actually restores a map-authored `sv_hit 0`/`switch_open N`/
    ///    `tune ...` after step 2/3 wiped it, and is why `switch_open` "sticks": it runs after
    ///    the blanket switcher-initial reset).
    /// 5. `config.game_settings_locked = true` (`SetGameSettingsReadOnly(true)`,
    ///    `gamecontext.cpp:4161`) — every `CFGFLAG_GAME` variable becomes read-only from this
    ///    point on; see [`apply_command`]'s doc comment for what that means for a *later*
    ///    `apply_commands` call (the post-init `--cfg` pass).
    /// 6. The `sv_solo_server` check (`gamecontext.cpp:4165-4177`) — reads whichever value
    ///    survived steps 1/3/4 (locking in step 5 doesn't block this *read*) and, if set, forces
    ///    `sv_team=SV_TEAM_FORCED_SOLO` and zeroes `player_collision`/`player_hooking` in every
    ///    tune zone.
    ///
    /// Returns *every* unrecognized command from either `pre_init_cfg` or `map_settings`, if any
    /// (both still run *before* the lock, so neither can ever produce a locked-write warning —
    /// only a later `apply_commands` call can) — real DDNet's `IConsole::ExecuteLine` dispatches
    /// (and, on failure, logs) one line at a time and keeps going, it never aborts the rest of a
    /// `--cfg`/`Settings` file on the first bad line, so this crate must not either: an earlier
    /// revision of this port stopped at the first unrecognized command via `?`, silently never
    /// even trying the lines after it (task spec acceptance criterion 2: "unknown commands ->
    /// explicit error listing them" — plural).
    pub fn init<'a>(&mut self, pre_init_cfg: impl IntoIterator<Item = &'a str>) -> Result<(), Vec<UnknownCommand>> {
        let mut errors = Vec::new();
        // Step 1: the pre-`OnInit()` `--cfg` pass — this genuinely runs *before* `OnInit()` in
        // the real server (`oracle_server.cpp`'s first `ExecuteFile`, before `LoadMap`/
        // `OnInit()`), so it must be applied before steps 2/3 wipe whatever it just set.
        for line in pre_init_cfg {
            if let Err(e) = apply_command(
                &mut self.config,
                &mut self.tuning,
                &mut self.cores.switchers,
                &mut self.command_log,
                line,
            ) {
                errors.push(e);
            }
        }
        // Step 2: unconditional per-zone reset — wipes any `tune`/`tune_zone` step 1 just set.
        self.tuning = TuningList::reset_to_baseline();
        // Step 3: `sv_ddrace_tune_reset`-gated reset (reads step 1's value of the flag itself)
        // — wipes any `sv_hit`/etc. step 1 just set.
        if self.config.sv_ddrace_tune_reset {
            self.config.sv_hit = true;
            self.config.sv_endless_drag = false;
            self.config.sv_old_laser = false;
            self.config.sv_old_teleport_hook = false;
            self.config.sv_old_teleport_weapons = false;
            self.config.sv_teleport_hold_hook = false;
            self.config.sv_team = SV_TEAM_ALLOWED;
            self.config.sv_show_others_default = 0;
            for switcher in self.cores.switchers.iter_mut() {
                switcher.initial = true;
            }
        }
        // Step 4: map settings, still writable. Clones the `Arc` (an O(1) refcount bump, not the
        // `Vec<String>` it points to — task 1.10b, speed-up 4) rather than `mem::take`-ing the
        // field, since `self.map_settings`'s *content* is never written here (or anywhere else),
        // only read while other `self` fields are mutated — an owned handle to the same
        // allocation satisfies the borrow checker just as well as swapping the field out and back
        // in did, without needing `Arc<Vec<String>>: Default` for the temporary placeholder.
        let map_settings = self.map_settings.clone();
        for line in map_settings.iter() {
            if let Err(e) = apply_command(
                &mut self.config,
                &mut self.tuning,
                &mut self.cores.switchers,
                &mut self.command_log,
                line,
            ) {
                errors.push(e);
            }
        }
        // Step 5: lock every `CFGFLAG_GAME` variable.
        self.config.game_settings_locked = true;
        // Step 6: solo check (a read of the now-locked `sv_solo_server`, not a write).
        if self.config.sv_solo_server {
            self.config.sv_team = SV_TEAM_FORCED_SOLO;
            self.tuning.zero_player_collision_and_hooking_everywhere();
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Applies a (post-init) `--cfg`/scenario `cfg_lines` pass via [`apply_command`], in order.
    /// Returns *every* genuinely-unrecognized command, if any (see [`World::init`]'s doc comment
    /// for why this doesn't stop at the first one); a `CFGFLAG_GAME` variable this crate *does*
    /// recognize but that [`World::init`] has already locked is not an error — see
    /// [`apply_command`]'s doc comment — it is recorded in [`World::command_log`] instead.
    pub fn apply_commands<'a>(&mut self, lines: impl IntoIterator<Item = &'a str>) -> Result<(), Vec<UnknownCommand>> {
        let mut errors = Vec::new();
        for line in lines {
            if let Err(e) = apply_command(
                &mut self.config,
                &mut self.tuning,
                &mut self.cores.switchers,
                &mut self.command_log,
                line,
            ) {
                errors.push(e);
            }
        }
        if self.config.sv_solo_server {
            self.config.sv_team = SV_TEAM_FORCED_SOLO;
            self.tuning.zero_player_collision_and_hooking_everywhere();
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Task 1.10, acceptance criterion 2: "save/restore without cloning Vecs" — makes `self` an
    /// exact copy of every mutable field of `source` (the *result* is identical to `*self =
    /// source.clone()`), but reuses `self`'s own existing heap allocations wherever the standard
    /// library's `Vec`/`Box<[T]>` `clone_from` specializations (or this crate's own hand-written
    /// ones on [`core::WorldCore`]/[`TuningList`], for the same reason) can, instead of freeing
    /// them and allocating fresh ones: once `self`'s buffers have already grown to fit `source`'s
    /// (the common case for a search/planner loop calling this repeatedly to reset the *same*
    /// scratch `World` back to one saved baseline before every rollout — `source` is typically a
    /// `World` kept around just as that baseline, itself produced by an earlier `.clone()`), a
    /// `restore_from` call allocates nothing at all. Measured effect: task's `BUILD REPORT`.
    ///
    /// [`World::collision`] is synced too (task 1.10b, finding F2): an `Arc::ptr_eq` check first,
    /// so the common case — `source` built on the same map as `self` (typically `self`'s own
    /// earlier `.clone()`) — costs one pointer comparison and nothing else; only a `source` built
    /// on a genuinely different map's `Collision` pays for the `Arc::clone` (still just a refcount
    /// bump, never a deep copy — see the struct's own doc comment).
    pub fn restore_from(&mut self, source: &World<R>) {
        self.restore_from_bounded(source, MAX_CLIENTS);
    }

    /// [`World::restore_from`] for a caller that knows every `Some` entry of [`World::characters`] and
    /// [`World::players`], in `self` as well as in `source`, has an index below `id_hi`: only that prefix of the two
    /// arrays is looked at and copied (task 4.13; [`World::restore_from`] finds the high-water marks by scanning all
    /// 128 slots of four arrays, which showed up in the profile of the search's per-rollout restore). With
    /// `id_hi = MAX_CLIENTS` this is exactly [`World::restore_from`]. A caller that gets the bound wrong (a `Some`
    /// entry at or above it) leaves such entries of `self` stale, so it must be an upper bound.
    pub fn restore_from_bounded(&mut self, source: &World<R>, id_hi: usize) {
        let id_hi = id_hi.min(MAX_CLIENTS);
        debug_assert!(
            self.characters[id_hi..].iter().all(Option::is_none)
                && self.players[id_hi..].iter().all(Option::is_none)
                && source.characters[id_hi..].iter().all(Option::is_none)
                && source.players[id_hi..].iter().all(Option::is_none),
            "restore_from_bounded: an entry at or above id_hi = {id_hi}"
        );
        if !std::sync::Arc::ptr_eq(&self.collision, &source.collision) {
            self.collision = std::sync::Arc::clone(&source.collision);
        }
        self.cores.clone_from(&source.cores);
        self.teams_core = source.teams_core;
        self.race_teams.clone_from(&source.race_teams);
        if id_hi == MAX_CLIENTS {
            restore_id_indexed_array(&mut self.characters, &source.characters);
            restore_id_indexed_array(&mut self.players, &source.players);
        } else {
            self.characters[..id_hi].copy_from_slice(&source.characters[..id_hi]);
            self.players[..id_hi].copy_from_slice(&source.players[..id_hi]);
        }
        self.entity_order.clone_from(&source.entity_order);
        // Scratch buffers are never part of the saved state (see each one's own doc comment on
        // `World`): always empty except transiently inside a call already in progress on this
        // very `World`, which `restore_from` cannot itself be called from. Cleared defensively
        // rather than left as whatever `self` already had (a no-op in every real scenario, since
        // they're already empty whenever `restore_from` could possibly be called).
        self.entity_order_scratch.clear();
        self.range_scratch.clear();
        self.hammer_scratch.clear();
        self.map_indices_scratch.clear();
        self.projectiles.clone_from(&source.projectiles);
        self.team_changed_this_pass = source.team_changed_this_pass;
        self.pickups.clone_from(&source.pickups);
        self.doors.clone_from(&source.doors);
        self.fixtures.clone_from(&source.fixtures);
        self.dragger_states.clone_from(&source.dragger_states);
        self.gun_states.clone_from(&source.gun_states);
        self.lasers.clone_from(&source.lasers);
        self.tuning.clone_from(&source.tuning);
        self.config = source.config;
        self.tick = source.tick;
        self.spawn_points.clone_from(&source.spawn_points);
        self.spawn_points_red.clone_from(&source.spawn_points_red);
        self.spawn_points_blue.clone_from(&source.spawn_points_blue);
        self.map_settings.clone_from(&source.map_settings);
        self.command_log.clone_from(&source.command_log);
        self.active_timed_switchers.clone_from(&source.active_timed_switchers);
    }
}

/// [`World::restore_from`]'s helper for its two `[Option<T>; MAX_CLIENTS]` fields
/// ([`World::characters`]/[`World::players`], indexed directly by client id — unlike
/// [`core::WorldCore`]'s compact-by-slot arrays, so [`core::WorldCore::clone_from`]'s
/// "just bound by `len`" trick doesn't apply here directly). Both fields' own doc comments state
/// the invariant this relies on: once a client id's slot becomes `Some`, it stays `Some` for the
/// rest of that `World`'s life (a death clears `alive`/`has_character`, never the slot itself) —
/// so every index past the highest `Some` one in *either* array is already (and stays) `None` on
/// both sides, and copying only the shared prefix up to that high-water mark reproduces `source`
/// in `dst` exactly, without reading or writing anything past it. `rposition` scans from the
/// high end, so this costs at most `MAX_CLIENTS` (128) cheap `is_some()` checks — negligible next
/// to the `Vec`/array copies elsewhere in `restore_from`, and a large win over always copying the
/// full 128-slot array (`size_of::<Option<Character<R>>>() * 128` = tens of KB) when, as in every
/// realistic search/rollout use case, only a handful of ids are ever in play.
fn restore_id_indexed_array<T: Copy>(dst: &mut [Option<T>; MAX_CLIENTS], source: &[Option<T>; MAX_CLIENTS]) {
    let dst_hi = dst.iter().rposition(Option::is_some);
    let src_hi = source.iter().rposition(Option::is_some);
    if let Some(hi) = dst_hi.into_iter().chain(src_hi).max() {
        dst[..=hi].copy_from_slice(&source[..=hi]);
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_entity<R: Real>(
    collision: &mut Collision<R>,
    entity: u8,
    flags: u8,
    x: i32,
    y: i32,
    layer: Layer,
    number: u8,
    map: &MapData,
    spawn_points: &mut [Vec<Vec2<R>>; 3],
    pickups: &mut Vec<Pickup<R>>,
    doors: &mut Vec<DoorFixture<R>>,
    fixture_scan: &mut FixtureScan<R>,
    crazy_shotguns: &mut Vec<CrazyShotgunSpawn<R>>,
) {
    let pos = Vec2::new(R::from_i32(x * 32 + 16), R::from_i32(y * 32 + 16));
    let entity_at = |dx: i32, dy: i32, layer: Layer| -> i32 {
        let nx = x + dx;
        let ny = y + dy;
        if nx < 0 || ny < 0 || nx >= map.width as i32 || ny >= map.height as i32 {
            return 0;
        }
        let idx = (ny as u32 * map.width + nx as u32) as usize;
        let raw = match layer {
            Layer::Game => map.game[idx].index,
            Layer::Front => map.front.as_ref().map(|f| f[idx].index).unwrap_or(0),
            Layer::Switch => map.switch.as_ref().map(|s| s[idx].kind).unwrap_or(0),
        };
        raw as i32 - map::ENTITY_OFFSET as i32
    };

    if (map::ENTITY_SPAWN..=map::ENTITY_SPAWN_BLUE).contains(&entity) {
        // `m_avSpawnPoints[SPAWNTYPE_DEFAULT/_RED/_BLUE]` (`gamecontroller.cpp:157`) — three
        // separate lists, each in its own map-scan order; `CanSpawn` evaluates them one whole
        // list at a time (`gamecontroller.cpp:167-180`: `EvaluateSpawnType(DEFAULT)` fully, then
        // `EvaluateSpawnType(RED)` fully, then `_BLUE`), not interleaved by map position. A
        // self-check found this crate initially merged all three into one position-scan-ordered
        // list, both missing `ENTITY_SPAWN_RED`/`_BLUE` entirely in an earlier revision and (once
        // added) getting their relative order wrong — reproduced on `BlmapChill` (`--seed
        // 10001`, which has 8 `ENTITY_SPAWN_RED` and 10 `_BLUE` tiles alongside a single
        // `ENTITY_SPAWN`): a same-tick kill+respawn at `game_tick 319` picked a different (but
        // each individually valid) spawn point until both were fixed, exactly matching the real
        // server's choice once they were.
        spawn_points[(entity - map::ENTITY_SPAWN) as usize].push(pos);
    } else if entity == map::ENTITY_DOOR {
        // `IGameController::OnEntity`'s door branch reads its 8 neighbors from the *same* layer
        // this call came from — `layer` here is always `Layer::Switch` for a real door (`kind ==
        // ENTITY_DOOR` only ever appears via the switch layer's raw type, per this module's map
        // constant doc comments), matching `character.cpp`'s convention of placing the
        // length/direction marker on the same layer as the door tile itself.
        for i in 0..8i32 {
            let (dx, dy) = NEIGHBOR_OFFSETS[i as usize];
            let side = entity_at(dx, dy, layer);
            if (map::ENTITY_LASER_SHORT as i32..=map::ENTITY_LASER_LONG as i32).contains(&side) {
                let rotation = R::from_f64(std::f64::consts::PI) / R::from_i32(4) * R::from_i32(i);
                let length = 32 * 3 + 32 * (side - map::ENTITY_LASER_SHORT as i32) * 3;
                let direction = Vec2::new(rotation.sin(), rotation.cos()); // libm-census: `Real::sin`/`cos`
                switch::place_door_collision(collision, pos, direction, length, number);
                doors.push(DoorFixture { pos });
            }
        }
    } else if entity == map::ENTITY_CRAZY_SHOTGUN_EX || entity == map::ENTITY_CRAZY_SHOTGUN {
        // `gamecontroller.cpp:214-244`: `Dir` from `Flags`, identically for both variants
        // (`ROTATION_90 == TILEFLAG_ROTATE`, `ROTATION_180 == (XFLIP|YFLIP)`, numerically).
        let dir = if flags == 0 {
            0
        } else if flags == map::ROTATION_90 {
            1
        } else if flags == map::ROTATION_180 {
            2
        } else {
            3
        };
        let deg = R::from_f64(std::f64::consts::FRAC_PI_2) * R::from_i32(dir);
        let direction = Vec2::new(deg.sin(), deg.cos()); // libm-census: `Real::sin`/`cos`
        crazy_shotguns.push((
            pos,
            direction,
            entity == map::ENTITY_CRAZY_SHOTGUN_EX,
            2 - (dir % 2),
            layer,
            number,
        ));
    } else if (map::ENTITY_ARMOR_1..=map::ENTITY_WEAPON_LASER).contains(&entity)
        || (map::ENTITY_ARMOR_SHOTGUN..=map::ENTITY_ARMOR_LASER).contains(&entity)
    {
        let kind = pickup_kind_for_entity(entity);
        if let Some(kind) = kind {
            pickups.push(Pickup {
                pos,
                kind,
                layer,
                number: number as i32,
                mcore: Vec2::zero(),
            });
        }
    } else if (map::ENTITY_DRAGGER_WEAK..=map::ENTITY_DRAGGER_STRONG_NW).contains(&entity) {
        // `new CDragger(.., Index - ENTITY_DRAGGER_WEAK[_NW] + 1, IgnoreWalls, Layer, Number)`
        // (`gamecontroller.cpp:343-350`): the strength parameter is a `float`.
        let (strength, ignore_walls) = if entity <= map::ENTITY_DRAGGER_STRONG {
            (entity - map::ENTITY_DRAGGER_WEAK + 1, false)
        } else {
            (entity - map::ENTITY_DRAGGER_WEAK_NW + 1, true)
        };
        let state = fixture_scan.dragger_states.len() as u16;
        fixture_scan.dragger_states.push(DraggerState::default());
        fixture_scan.fixtures.push(Fixture::Dragger(Dragger {
            pos,
            core: Vec2::zero(),
            strength: R::from_i32(i32::from(strength)),
            ignore_walls,
            layer,
            number: i32::from(number),
            state,
        }));
    } else if (map::ENTITY_PLASMAE..=map::ENTITY_PLASMAU).contains(&entity) {
        // `new CGun(.., Freeze, Explosive, Layer, Number)` (`gamecontroller.cpp:351-366`).
        let (freeze, explosive) = match entity {
            map::ENTITY_PLASMAE => (false, true),
            map::ENTITY_PLASMAF => (true, false),
            map::ENTITY_PLASMA => (true, true),
            _ => (false, false), // ENTITY_PLASMAU
        };
        let state = fixture_scan.gun_states.len() as u16;
        fixture_scan.gun_states.push(GunState::default());
        fixture_scan.fixtures.push(Fixture::Gun(Gun {
            pos,
            core: Vec2::zero(),
            freeze,
            explosive,
            layer,
            number: i32::from(number),
            state,
        }));
    } else if (map::ENTITY_LASER_FAST_CCW..=map::ENTITY_LASER_FAST_CW).contains(&entity) {
        // `gamecontroller.cpp:290-343`. Everything is `float`/`int` exactly as there: `pi / 360`
        // is a `float` division (`pi` is a `float` constant), `AngularSpeed *= M` multiplies by an
        // `int`.
        let ind = i32::from(entity) - i32::from(map::ENTITY_LASER_STOP);
        let (ind_abs, m) = if ind < 0 {
            (-ind, 1)
        } else if ind == 0 {
            (0, 0)
        } else {
            (ind, -1)
        };
        let mut angular_speed = R::ZERO;
        if ind_abs == 1 {
            angular_speed = R::PI / R::from_i32(360);
        } else if ind_abs == 2 {
            angular_speed = R::PI / R::from_i32(180);
        } else if ind_abs == 3 {
            angular_speed = R::PI / R::from_i32(90);
        }
        angular_speed *= R::from_i32(m);
        for i in 0..8i32 {
            let (dx, dy) = NEIGHBOR_OFFSETS[i as usize];
            let side = entity_at(dx, dy, layer);
            if (i32::from(map::ENTITY_LASER_SHORT)..=i32::from(map::ENTITY_LASER_LONG)).contains(&side) {
                // `aSides2[i]`: the same neighbor, two cells out.
                let side2 = entity_at(dx * 2, dy * 2, layer);
                let rotation = R::PI / R::from_i32(4) * R::from_i32(i);
                let length = 32 * 3 + 32 * (side - i32::from(map::ENTITY_LASER_SHORT)) * 3;
                let (speed, curve_length) = if (i32::from(map::ENTITY_LASER_C_SLOW)
                    ..=i32::from(map::ENTITY_LASER_C_FAST))
                    .contains(&side2)
                {
                    (1 + (side2 - i32::from(map::ENTITY_LASER_C_SLOW)) * 2, length)
                } else if (i32::from(map::ENTITY_LASER_O_SLOW)..=i32::from(map::ENTITY_LASER_O_FAST)).contains(&side2) {
                    (1 + (side2 - i32::from(map::ENTITY_LASER_O_SLOW)) * 2, 0)
                } else {
                    (0, length)
                };
                fixture_scan.fixtures.push(Fixture::Light(fixtures::new_light(
                    pos,
                    rotation,
                    length,
                    layer,
                    i32::from(number),
                    angular_speed,
                    speed,
                    curve_length,
                )));
            }
        }
    }
}

/// What the map scan collects for [`World::fixtures`] and its two state pools.
struct FixtureScan<R: Real> {
    fixtures: Vec<Fixture<R>>,
    dragger_states: Vec<DraggerState>,
    gun_states: Vec<GunState>,
}

impl<R: Real> Default for FixtureScan<R> {
    fn default() -> Self {
        FixtureScan {
            fixtures: Vec::new(),
            dragger_states: Vec::new(),
            gun_states: Vec::new(),
        }
    }
}

/// The 8 cell offsets `aSides[0..8]` scans, in that exact order (`gamecontroller.cpp:186-193`):
/// S, SE, E, NE, N, NW, W, SW.
const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

fn pickup_kind_for_entity(entity: u8) -> Option<PickupKind> {
    match entity {
        map::ENTITY_ARMOR_1 => Some(PickupKind::Armor),
        map::ENTITY_HEALTH_1 => Some(PickupKind::Freeze),
        map::ENTITY_WEAPON_SHOTGUN => Some(PickupKind::Weapon(WEAPON_SHOTGUN)),
        map::ENTITY_WEAPON_GRENADE => Some(PickupKind::Weapon(WEAPON_GRENADE)),
        map::ENTITY_POWERUP_NINJA => Some(PickupKind::Ninja),
        map::ENTITY_WEAPON_LASER => Some(PickupKind::Weapon(WEAPON_LASER)),
        map::ENTITY_ARMOR_SHOTGUN => Some(PickupKind::ArmorShotgun),
        map::ENTITY_ARMOR_GRENADE => Some(PickupKind::ArmorGrenade),
        map::ENTITY_ARMOR_NINJA => Some(PickupKind::ArmorNinja),
        map::ENTITY_ARMOR_LASER => Some(PickupKind::ArmorLaser),
        _ => None,
    }
}

// --- `CCharacter`'s small helpers (`character.cpp`) --------------------------------------------

/// `CCharacter::Team()` (`character.cpp:1397-1400`): `Teams()->m_Core.Team(ClientId)`.
pub fn character_team(teams: &TeamsCore, id: i32) -> i32 {
    teams.team(id)
}

/// `CCharacter::CanCollide(int ClientId)` (`character.cpp:1388-1391`).
pub fn character_can_collide(teams: &TeamsCore, self_id: i32, other_id: i32) -> bool {
    teams.can_collide(self_id, other_id)
}

/// `CCharacter::ApplyMoveRestrictions()` (`character.cpp:2610-2613`): `m_Core.m_Vel =
/// ClampVel(m_MoveRestrictions, m_Core.m_Vel)` — `m_MoveRestrictions` there is the
/// *character*-level field ([`Character::move_restrictions`], not `CharacterCore`'s own), so
/// `move_restrictions` is a parameter here rather than read off `core`.
pub fn apply_move_restrictions<R: Real>(core: &mut CharacterCore<R>, move_restrictions: i32) {
    core.vel = crate::collision::clamp_vel(move_restrictions, core.vel);
}

/// `CCharacter::TakeDamage(vec2 Force, int Dmg, int From, int Weapon)` (`character.cpp:1051-
/// 1062`): `m_Core.m_Vel = ClampVel(m_MoveRestrictions, Temp)` — the *character*-level field
/// (see [`apply_move_restrictions`]'s doc comment). `Dmg` only gates a cosmetic emote in the C++
/// source (never read here — no compared field depends on it), so it isn't a parameter.
pub fn take_damage<R: Real>(core: &mut CharacterCore<R>, force: Vec2<R>, move_restrictions: i32) {
    let temp = core.vel + force;
    core.vel = crate::collision::clamp_vel(move_restrictions, temp);
}

/// `CCharacter::Freeze(int Seconds)` (`character.cpp:2367-2379`). Returns whether it actually
/// froze/extended (matching the C++ `bool` return, e.g. for `hammer_hits`/`freeze_entries`
/// bookkeeping — unused by `World` itself, but kept since callers may want it).
pub fn freeze<R: Real>(character: &mut Character<R>, core: &mut CharacterCore<R>, tick: i32, seconds: i32) -> bool {
    if seconds <= 0 || core.is_super || core.invincible || character.freeze_time > seconds * core::SERVER_TICK_SPEED {
        return false;
    }
    if character.freeze_time == 0 || core.freeze_start < tick - core::SERVER_TICK_SPEED {
        character.armor = 0;
        character.freeze_time = seconds * core::SERVER_TICK_SPEED;
        core.freeze_start = tick;
        return true;
    }
    false
}

/// `CCharacter::Freeze()` (`character.cpp:2381-2384`): `Freeze(g_Config.m_SvFreezeDelay)`.
pub fn freeze_default<R: Real>(
    character: &mut Character<R>,
    core: &mut CharacterCore<R>,
    tick: i32,
    sv_freeze_delay: i32,
) -> bool {
    freeze(character, core, tick, sv_freeze_delay)
}

/// `CCharacter::Unfreeze()` (`character.cpp:2386-2399`).
pub fn unfreeze<R: Real>(character: &mut Character<R>, core: &mut CharacterCore<R>) -> bool {
    if character.freeze_time > 0 {
        character.armor = 10;
        if core.active_weapon >= 0 && !core.weapons[core.active_weapon as usize].got {
            core.active_weapon = WEAPON_GUN;
        }
        character.freeze_time = 0;
        core.freeze_start = 0;
        character.frozen_last_tick = true;
        return true;
    }
    false
}

/// `CCharacter::GiveWeapon(int Weapon, bool Remove)` (`character.cpp:2407-2429`). The
/// `WEAPON_NINJA` branch calls [`give_ninja`]/[`remove_ninja`] — see those functions' doc
/// comments for Stage A's deliberately-partial ninja handling.
pub fn give_weapon<R: Real>(
    character: &mut Character<R>,
    core: &mut CharacterCore<R>,
    tick: i32,
    weapon: i32,
    remove: bool,
) {
    if weapon == WEAPON_NINJA {
        if remove {
            remove_ninja(character, core);
        } else {
            give_ninja(character, core, tick);
        }
        return;
    }
    if remove {
        if core.active_weapon == weapon {
            core.active_weapon = WEAPON_GUN;
        }
    } else {
        core.weapons[weapon as usize].ammo = -1;
    }
    core.weapons[weapon as usize].got = !remove;
}

/// `CCharacter::GiveAllWeapons()` (`character.cpp:2431-2437`) — unused by anything Stage A
/// drives (no map/scenario mechanism this crate models ever calls it — it's a debug/rcon
/// command), kept for completeness.
pub fn give_all_weapons<R: Real>(character: &mut Character<R>, core: &mut CharacterCore<R>, tick: i32) {
    for w in WEAPON_GUN..NUM_WEAPONS as i32 - 1 {
        give_weapon(character, core, tick, w, false);
    }
}

/// `CCharacter::ResetPickups()` (`character.cpp:2439-2447`).
pub fn reset_pickups<R: Real>(core: &mut CharacterCore<R>) {
    for w in WEAPON_SHOTGUN..NUM_WEAPONS as i32 - 1 {
        core.weapons[w as usize].got = false;
        if core.active_weapon == w {
            core.active_weapon = WEAPON_GUN;
        }
    }
}

/// `CCharacter::SetEndlessHook(bool Enable)` (`character.cpp:2449-2458`) — the chat message is
/// dropped (cosmetic); the early-return-if-unchanged is preserved even though it has no
/// observable effect here (documents the exact C++ shape).
pub fn set_endless_hook<R: Real>(core: &mut CharacterCore<R>, enable: bool) {
    if core.endless_hook == enable {
        return;
    }
    core.endless_hook = enable;
}

/// `CCharacter::GiveNinja()` (`character.cpp:678-689`); the dash itself is [`handle_ninja`].
pub fn give_ninja<R: Real>(character: &mut Character<R>, core: &mut CharacterCore<R>, tick: i32) {
    core.ninja.activation_tick = tick;
    core.weapons[WEAPON_NINJA as usize].got = true;
    core.weapons[WEAPON_NINJA as usize].ammo = -1;
    if core.active_weapon != WEAPON_NINJA {
        character.last_weapon = core.active_weapon;
    }
    core.active_weapon = WEAPON_NINJA;
}

/// `CCharacter::RemoveNinja()` (`character.cpp:691-702`).
pub fn remove_ninja<R: Real>(character: &mut Character<R>, core: &mut CharacterCore<R>) {
    core.ninja = core::NinjaState::default();
    core.weapons[WEAPON_NINJA as usize].got = false;
    core.weapons[WEAPON_NINJA as usize].ammo = 0;
    // `m_Core.m_ActiveWeapon = m_LastWeapon; SetWeapon(m_Core.m_ActiveWeapon);` — the `SetWeapon`
    // call compares its argument with the (just assigned) active weapon and returns at once
    // (`character.cpp:154-156`), so neither `m_LastWeapon` nor the range clamp is touched. (Stage
    // A had this wrong — it never ran, the cut stopped the comparison at the ninja pickup.)
    core.active_weapon = character.last_weapon;
}

/// `CCharacter::ReleaseHook()` (`character.cpp:766-771`). `self_id`/`self_slot`: this
/// character's own id and its current [`WorldCore`] slot.
pub fn release_hook<R: Real>(cores: &mut WorldCore<R, MAX_CLIENTS>, self_id: u8, self_slot: usize) {
    let mut me = *cores.core_at(self_slot);
    core::set_hooked_player(cores, &mut me, self_id, -1);
    me.hook_state = core::HOOK_RETRACTED;
    me.triggered_events |= core::COREEVENT_HOOK_RETRACT;
    *cores.core_at_mut(self_slot) = me;
}

/// `CCharacter::ResetHook()` (`character.cpp:773-777`).
pub fn reset_hook<R: Real>(cores: &mut WorldCore<R, MAX_CLIENTS>, self_id: u8, self_slot: usize) {
    release_hook(cores, self_id, self_slot);
    let pos = cores.core_at(self_slot).pos;
    cores.core_at_mut(self_slot).hook_pos = pos;
}

/// `CCharacter::ResetInput()` (`character.cpp:779-788`) — used by the tele-checkpoint/evil-tele
/// paths. `INPUT_STATE_MASK` (`generated/protocol.h`) is `0x3f`.
pub fn reset_input<R: Real>(core: &mut CharacterCore<R>, character: &mut Character<R>) {
    core.input.direction = 0;
    if core.input.fire & 1 != 0 {
        core.input.fire += 1;
    }
    core.input.fire &= 0x3f;
    core.input.jump = 0;
    character.latest_prev_input = core.input;
    character.latest_input = core.input;
}

// --- `IGameController::CanSpawn`/`EvaluateSpawnType`/`EvaluateSpawnPos`
// (`gamecontroller.cpp:89-181`). -------------------------------------------------------------

impl<R: Real> World<R> {
    /// `IGameController::CanSpawn` (`gamecontroller.cpp:161-173`): evaluates
    /// [`World::spawn_points`] (`SPAWNTYPE_DEFAULT`), then [`World::spawn_points_red`]
    /// (`SPAWNTYPE_RED`), then [`World::spawn_points_blue`] (`SPAWNTYPE_BLUE`) — three genuinely
    /// separate passes over one running best-so-far (`CSpawnEval`), *not* one pool of every
    /// type's points scored together; see `World::evaluate_spawn_type`'s doc comment for why
    /// that distinction is observable. Returns `None` only when every one of the three lists is
    /// empty (`!pEval->m_Got` after all three calls).
    pub fn can_spawn(&self, for_client: i32) -> Option<Vec2<R>> {
        let mut eval: Option<(Vec2<R>, R)> = None;
        self.evaluate_spawn_type(&self.spawn_points, for_client, &mut eval);
        self.evaluate_spawn_type(&self.spawn_points_red, for_client, &mut eval);
        self.evaluate_spawn_type(&self.spawn_points_blue, for_client, &mut eval);
        eval.map(|(pos, _)| pos)
    }

    /// `IGameController::EvaluateSpawnType` (`gamecontroller.cpp:104-158`). `eval` is the shared
    /// `CSpawnEval` (`None` == `!m_Got`), threaded through all three [`World::can_spawn`] calls
    /// in one fixed order.
    ///
    /// `if(!PlayerCollision && pEval->m_Got) return;` (`gamecontroller.cpp:110-111`): once *any*
    /// earlier pass (of this type or an earlier one) already found a spot, and global player
    /// collision is off (`sv_solo_server`, or any other zone-0 `player_collision 0`), every
    /// later type is skipped outright — a genuinely different result from scoring every type's
    /// points together in one combined pool whenever more than one type has points at all. An
    /// earlier revision of this port concatenated `[DEFAULT, RED, BLUE]` into one list and
    /// scored all of them as a single pass — an empirically found bug (see this crate's `BUILD
    /// REPORT`), fixed by keeping the three lists separate and gating each call this way.
    fn evaluate_spawn_type(&self, points: &[Vec2<R>], for_client: i32, eval: &mut Option<(Vec2<R>, R)>) {
        let player_collision = self.tuning.zone(0).player_collision::<R>() != R::ZERO;
        // `PlayerCollisionDisabled = GetPlayerChar(ClientId)->GetCore().m_CollisionDisabled`
        // (`gamecontroller.cpp:98-101`) — `false` when `for_client` has no character yet (the
        // `TryRespawn` call path, always), and `collision_disabled_ticks == 0` for this entire
        // corpus even on the `HandleTiles` tele-checkpoint-fallback call path (where a live
        // character does exist), so hardcoding `false` is exact for every scenario this crate
        // runs, not just a simplification. Distinct from the *nearby candidate's own*
        // `m_CollisionDisabled`, checked per-candidate below.
        let player_collision_disabled = false;

        if !player_collision && eval.is_some() {
            return;
        }

        // j == 0: an empty slot (one of the 5 offset positions around a spawn point that no
        // nearby character currently occupies/overlaps).
        for &spawn in points {
            let offsets: [Vec2<R>; 5] = [
                Vec2::zero(),
                Vec2::new(R::from_i32(-32), R::ZERO),
                Vec2::new(R::ZERO, R::from_i32(-32)),
                Vec2::new(R::from_i32(32), R::ZERO),
                Vec2::new(R::ZERO, R::from_i32(32)),
            ];
            // `FindEntities(SpawnPoint, 64, ..., ENTTYPE_CHARACTER)` (`gamecontroller.cpp:127`):
            // a *fixed* radius around the base spawn point itself, queried once — not
            // re-queried per offset candidate, and not "every character on the map" (this
            // port's own prior bug: it scanned every core and pre-filtered by `CanCollide`
            // before ever reaching the `CheckPoint` half of the condition below — see the
            // per-candidate loop's own comment for why that's also wrong).
            // Allocates (once per spawn point candidate) — acceptable per acceptance criterion
            // 1's own "document any unavoidable allocation": `can_spawn` only runs on an actual
            // respawn (`CPlayer::TryRespawn`), a discrete, comparatively rare event, not part of
            // the zero-allocation *steady-state* tick loop review round 1's finding F9 is about
            // (`World::step`/`world_tick`'s own per-tick hot path no longer allocates at all —
            // see `for_each_in_entity_order`/`find_characters_in_range_into`/`fire_hammer`'s own
            // scratch buffers). Not worth a dedicated scratch field for a call this infrequent.
            let nearby: Vec<(i32, Vec2<R>, bool)> = characters_in_entity_order(self)
                .filter(|&(_, pos, _)| vmath::distance(pos, spawn) <= R::from_i32(64))
                .map(|(id, pos, core)| (id, pos, core.collision_disabled))
                .collect();

            let mut result: Option<usize> = None;
            for (idx, &off) in offsets.iter().enumerate() {
                result = Some(idx);
                if !player_collision || player_collision_disabled {
                    break;
                }
                let candidate = spawn + off;
                let mut occupied = false;
                for &(id, pos, other_collision_disabled) in &nearby {
                    // `GameServer()->Collision()->CheckPoint(...)` runs unconditionally for
                    // *every* nearby entity, regardless of `CanCollide` — only the distance
                    // half of the `||` is gated by it (`gamecontroller.cpp:135-137`). This
                    // port previously `continue`d past non-collidable entities before ever
                    // reaching `CheckPoint`, silently skipping it whenever the *first* nearby
                    // entities happened to be non-collidable teammates.
                    let can_collide =
                        character_can_collide(&self.teams_core, id, for_client) && !other_collision_disabled;
                    if self.collision.check_point_vec(candidate)
                        || (can_collide && vmath::distance(pos, candidate) <= core::physical_size::<R>())
                    {
                        occupied = true;
                        break;
                    }
                }
                if occupied {
                    result = None;
                } else {
                    break;
                }
            }
            if let Some(idx) = result {
                let pos = spawn + offsets[idx];
                let score = self.evaluate_spawn_pos(pos, for_client);
                if eval.is_none() || score < eval.unwrap().1 {
                    *eval = Some((pos, score));
                }
            }
        }

        // j == 1: no empty slot found by any pass so far (this type's own j == 0, or an
        // earlier type entirely) — take *this type's* very first spawn point unconditionally.
        // Every point in `points` after the first is a provable no-op here: the real update
        // condition is `!pEval->m_Got || (j == 0 && ...)`, and `j == 1` makes the second half
        // always `false`, so only the very first point, while `pEval->m_Got` is still `false`,
        // can ever update `pEval` during this pass.
        if eval.is_none()
            && let Some(&first) = points.first()
        {
            let score = self.evaluate_spawn_pos(first, for_client);
            *eval = Some((first, score));
        }
    }

    /// `IGameController::EvaluateSpawnPos` (`gamecontroller.cpp:89-99`): sum of `1/distance` to
    /// every collidable existing character (`1e9` if exactly on top of one).
    fn evaluate_spawn_pos(&self, pos: Vec2<R>, for_client: i32) -> R {
        let mut score = R::ZERO;
        for (id, other_pos, _) in characters_in_entity_order(self) {
            if !character_can_collide(&self.teams_core, for_client, id) {
                continue;
            }
            let d = vmath::distance(pos, other_pos);
            score += if d == R::ZERO {
                R::from_i32(1_000_000_000)
            } else {
                R::ONE / d
            };
        }
        score
    }
}

// --- `CCharacter::HandleTuneLayer`/`HandleSkippableTiles`/`HandleTiles`/`DDRaceTick`/
// `DDRacePostCoreTick` (`character.cpp`) -----------------------------------------------------

/// `CCharacter::HandleTuneLayer()` (`character.cpp:2126-2138`), minus the zone-message send
/// (cosmetic).
pub fn handle_tune_layer<R: Real>(
    character: &mut Character<R>,
    core: &mut CharacterCore<R>,
    collision: &Collision<R>,
    tuning: &TuningList,
) {
    character.tune_zone_old = character.tune_zone;
    let current_index = collision.get_map_index(core.pos);
    character.tune_zone = collision.is_tune(current_index);
    core.tuning = *tuning.zone(character.tune_zone);
}

/// `CCharacter::DDRaceTick()` (`character.cpp:2228-2286`), minus `SetArmorProgress` (cosmetic
/// HUD, `gamemodes/ddnet.cpp:120-123` — `m_Armor` isn't a compared field) and `TrySetRescue`
/// (`docs`/`BUILD REPORT`: no observable effect in this crate's scope). Returns whether the
/// character froze-to-death via the freeze-damage-indicator tick this call may trigger — no,
/// actually: [`freeze`]/[`unfreeze`] never kill; this always returns normally.
pub fn ddrace_tick<R: Real>(
    character: &mut Character<R>,
    core: &mut CharacterCore<R>,
    collision: &Collision<R>,
    tuning: &TuningList,
) {
    // `mem_copy(&m_Input, &m_SavedInput, ...)` (`character.cpp:2230`) — `m_Input` here is
    // `CCharacter`'s own field, copied into `m_Core.m_Input` separately by `PreTick()`
    // (`character.cpp:814`, after this function returns — see [`character_pre_tick`]).
    character.input = character.saved_input;
    if character.input.direction != 0 || character.input.jump != 0 {
        // `m_LastMove` — cosmetic (`DetermineEyeEmote` only), not tracked.
    }

    if core.live_frozen && !core.is_super && !core.invincible {
        character.input.direction = 0;
        character.input.jump = 0;
        // Hook is still possible while live-frozen.
    }
    if character.freeze_time > 0 {
        character.freeze_time -= 1;
        character.input.direction = 0;
        character.input.jump = 0;
        character.input.hook = 0;
        if character.freeze_time == 1 {
            unfreeze(character, core);
        }
    }

    handle_tune_layer(character, core, collision, tuning);

    let index = collision.get_pure_map_index_vec(core.pos);
    let tiles = [
        collision.get_tile_index(index as i32),
        collision.get_front_tile_index(index as i32),
        collision.get_switch_type(index as i32),
    ];
    core.is_in_freeze = tiles.iter().any(|&t| {
        t == map::TILE_FREEZE as i32
            || t == map::TILE_DFREEZE as i32
            || t == map::TILE_LFREEZE as i32
            || t == map::TILE_DEATH as i32
    });
    if !core.is_in_freeze {
        core.is_in_freeze = is_on_death_tile(collision, core.pos);
    }
}

/// The 4-corner `TILE_DEATH` sensitivity check `HandleSkippableTiles`/`DDRaceTick` both use
/// (`character.cpp:1477-1484,2272-2279`) — factored out since it's the exact same expression
/// twice in the C++ source too, just inlined both times.
fn is_on_death_tile<R: Real>(collision: &Collision<R>, pos: Vec2<R>) -> bool {
    // Task 3.6: no death tile within a tile of `pos`'s own tile means none of the eight lookups can hit.
    if !collision.death_possibly_near(pos) {
        return false;
    }
    let r = core::physical_size::<R>() / R::from_i32(3);
    let corners = [(r, -r), (r, r), (-r, -r), (-r, r)];
    corners.iter().any(|&(dx, dy)| {
        collision.get_collision_at(pos.x + dx, pos.y + dy) == map::TILE_DEATH as i32
            || collision.get_front_collision_at(pos.x + dx, pos.y + dy) == map::TILE_DEATH as i32
    })
}

/// `CCharacter::Die(int Killer, int Weapon, bool SendKillMsg)` (`character.cpp:1018-1049`),
/// minus recording/logging/chat/sound (cosmetic) and `CancelSwapRequests` (unreachable —
/// `/swap` is a chat command). `killer`/`weapon`: unused beyond matching the C++ signature (no
/// compared field depends on either — `OnCharacterDeath`'s `ModeSpecial` is always `0` for the
/// base `IGameController`, and `SendDeathMessage`'s fields are network-only).
/// Unlinks `id` from [`World::entity_order`] immediately. Verified empirically against the
/// corpus (`ChillBlock5__seed10001` at tick 200: two characters die simultaneously — the
/// reference's `m_StrongWeakId` for the still-alive characters already reflects the *post-death*
/// entity count in the very same tick the death happened, not one tick later) — so, unlike this
/// crate's own doc-comment history once assumed, `CCharacter::Die()`'s removal from
/// `CGameWorld`'s `ENTTYPE_CHARACTER` list is *not* deferred to the next `CPlayer::Tick()`; it is
/// visible to that same tick's `m_StrongWeakId` assignment pass (`gameworld.cpp:202-263`, which
/// runs *after* every entity's `Tick()`/`TickDeferred()` — i.e. after `Die()` could have already
/// fired this tick — but *before* `CPlayer::Tick()`, `gamecontext.cpp:23-47`).
pub fn die<R: Real>(world: &mut World<R>, id: i32, _killer: i32, _weapon: i32) {
    let forced_solo = sv_team_forced_solo(&world.config);
    let tick = world.tick;
    if let Some(player) = world.players[id as usize].as_mut() {
        player.previous_die_tick = player.die_tick;
        player.die_tick = tick;
    }
    if let Some(c) = world.characters[id as usize].as_mut() {
        c.alive = false;
    }
    world.teams_core.set_solo(id, false);
    if let Some(slot) = world.cores.slot_of(id as u8) {
        world.cores.core_at_mut(slot).solo = false;
    }
    world.cores.remove(id as u8);
    world.entity_order.retain(|&x| x != id as u8);
    let changed = world.race_teams.on_character_death(
        &mut world.teams_core,
        &mut world.cores.switchers,
        &mut world.characters,
        id,
        forced_solo,
    );
    apply_team_change_side_effects(world, id, changed);
}

/// Shared post-processing for any [`RaceTeams::set_force_character_team`] call that reports a
/// genuine team change, applying real DDNet's `RemoveEntitiesFromPlayer(ClientId)`
/// (`teams.cpp:497`, `gameworld.cpp:160-182`): marks every one of `id`'s own live projectiles
/// for removal (any weapon type, including grenade — unlike the unrelated, weapon-type-gated
/// "owner not alive" lazy check already in [`projectile_tick`], `projectile.cpp:121-129`, this
/// one is unconditional) via the existing deferred [`Projectile::marked_for_destroy`] mechanism
/// (safe against index invalidation when called mid a `for i in 0..projectiles.len()` pass;
/// real DDNet's own removal is synchronous, but nothing observes the difference within the same
/// tick — see [`World::projectiles`]'s neighboring fields), and sets
/// [`World::team_changed_this_pass`] so [`World::world_tick`]'s character loop stops early if
/// this was a mid-loop call. A no-op when `changed` is `false`.
fn apply_team_change_side_effects<R: Real>(world: &mut World<R>, id: i32, changed: bool) {
    if changed {
        for p in world.projectiles.iter_mut().filter(|p| p.owner == id) {
            p.marked_for_destroy = true;
        }
        laser::remove_lasers_of_player(world, id);
        world.team_changed_this_pass = true;
    }
}

/// `CCharacter::HandleSkippableTiles(int Index)` (`character.cpp:1474-1595`), minus the
/// old-type-speedup's chat/sound-only branches kept intact (they *do* affect `m_Core.m_Vel`).
/// Returns whether the character is still alive after this call (`Die()` may fire; `HandleTiles`
/// callers must stop touching this character's fields immediately if this returns `false`,
/// matching `if(!m_Alive) return;` at every C++ call site).
pub fn handle_skippable_tiles<R: Real>(world: &mut World<R>, id: i32, index: i32) -> bool {
    let team = world.teams_core.team(id);
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return false;
    };
    let pos = world.characters[id as usize].as_ref().unwrap().pos;
    let is_super = world.cores.core_at(slot).is_super;
    let invincible = world.cores.core_at(slot).invincible;
    let on_death = is_on_death_tile(&world.collision, pos);

    // `Team() && Teams()->TeeFinished(...)`: `Team() == TEAM_FLOCK` (`0`) is falsy in C++, so a
    // `TEAM_FLOCK` character's death-tile check is unconditional; a non-flock character is
    // spared only once `OnCharacterFinish` has marked it finished (`character.cpp:1477-1502`).
    let finished = team != TEAM_FLOCK && world.race_teams.tee_finished(id);
    if on_death && !is_super && !invincible && !finished {
        die(world, id, id, WEAPON_WORLD);
        return false;
    }

    if game_layer_clipped(&world.collision, pos) {
        die(world, id, id, WEAPON_WORLD);
        return false;
    }

    if index < 0 {
        return true;
    }

    if world.collision.is_speedup(index)
        && let Some(info) = world.collision.get_speedup(index)
    {
        let core = world.cores.core_at_mut(slot);
        let mut character = world.characters[id as usize].unwrap();
        apply_speedup(core, &mut character, &info);
        world.characters[id as usize] = Some(character);
    }
    true
}

/// `WEAPON_WORLD` (`generated/protocol.h`): world-caused death (falling off/death tiles).
pub const WEAPON_WORLD: i32 = -2;
/// `WEAPON_SELF` (`generated/protocol.h`): self-inflicted (`/kill`).
pub const WEAPON_SELF: i32 = -1;
/// `WEAPON_GAME` (`generated/protocol.h`): mode-caused death (e.g. leaving to spectators).
pub const WEAPON_GAME: i32 = -3;

/// `CEntity::GameLayerClipped(vec2 CheckPos)` (`entity.cpp`, via `CCharacter`): true once far
/// enough outside the map that DDNet considers the character to have left the game layer.
/// `entity.cpp`'s exact bound is `-200`/`+200` past the map's pixel size.
fn game_layer_clipped<R: Real>(collision: &Collision<R>, pos: Vec2<R>) -> bool {
    // `entity.cpp:53-57`: `round_to_int(CheckPos.x) / 32 < -200 || ... > GetWidth() + 200 || ...`
    // — the `200` margin is in *tile* units, not pixels (a bug found empirically: this crate's
    // previous version compared raw pixel coordinates against a `+ 201.0` pixel margin, making
    // the effective out-of-world margin ~6 tiles instead of 200 — several corpus traces showed a
    // character falling for a few dozen ticks past the map's bottom edge get killed roughly a
    // hundred ticks too early, while the reference correctly keeps falling until it's actually
    // ~200 tiles past the border). `round_to_int(x)/32` is integer division (truncating toward
    // zero) in both C++ and Rust, so this must divide in `i32`, not keep dividing in `R`.
    let tx = vmath::round_to_int(pos.x) / 32;
    let ty = vmath::round_to_int(pos.y) / 32;
    tx < -200 || tx > collision.width() + 200 || ty < -200 || ty > collision.height() + 200
}

/// The `TILE_SPEED_BOOST_OLD`/`TILE_SPEED_BOOST` branch of `HandleSkippableTiles`
/// (`character.cpp:1515-1594`).
fn apply_speedup<R: Real>(
    core: &mut CharacterCore<R>,
    character: &mut Character<R>,
    info: &crate::collision::SpeedupInfo<R>,
) {
    let direction = info.dir;
    let mut temp_vel = core.vel;
    if info.kind == map::TILE_SPEED_BOOST_OLD as i32 {
        if info.force == 255 && info.max_speed > 0 {
            // `Direction * (MaxSpeed / 5)` (`character.cpp:1521`): `MaxSpeed`/`5` are both `int`,
            // so this is *integer* division (truncating toward zero), not `R`-precision division
            // — an earlier revision of this port divided in `R` directly, found empirically
            // (`speedway_v1.trb` corridor 0 tick 36: reference `vel_x` `118.80078` == `594/5`
            // computed as `118` (int) then converted, not `118.8` computed in float).
            core.vel = direction * R::from_i32(info.max_speed / 5);
            return;
        }
        let mut max_speed = info.max_speed;
        if max_speed > 0 && max_speed < 5 {
            max_speed = 5;
        }
        if max_speed > 0 {
            let speeder_angle = old_speedup_angle(direction.x, direction.y);
            let tee_angle = old_speedup_angle(temp_vel.x, temp_vel.y);
            // `TeeSpeed = std::sqrt(std::pow(TempVel.x, 2) + std::pow(TempVel.y, 2))`
            // (`character.cpp:1554`): `std::pow(float, int)` resolves to the generic
            // `<cmath>` overload that promotes both arguments (and its result) to `double`, so
            // the whole sum-then-sqrt happens in `double` precision, only rounding to `float` at
            // the final assignment to `TeeSpeed` — not squaring/summing/rooting in `float`
            // throughout, which an earlier revision of this port did (found empirically: the
            // `f32` path differed from a `double`-throughout one on 16% of 1M random inputs).
            let tee_speed_f64 = temp_vel.x.to_f64().powi(2) + temp_vel.y.to_f64().powi(2);
            let tee_speed = R::from_f64(tee_speed_f64.sqrt());
            let diff_angle = speeder_angle - tee_angle;
            // libm-census: `Real::cos` (ddai_libm for f32)
            let speed_left = R::from_i32(max_speed) / R::from_i32(5) - diff_angle.cos() * tee_speed;
            let force = R::from_i32(info.force);
            // `absolute((int)SpeedLeft)` (`character.cpp:1562-1567`): truncate to `int` *then*
            // take the absolute value of that integer — not the other way around.
            let speed_left_abs_trunc = speed_left.to_i32_trunc().abs();
            if speed_left_abs_trunc > info.force && speed_left > R::from_f64(0.0000001) {
                temp_vel += direction * force;
            } else if speed_left_abs_trunc > info.force {
                temp_vel += direction * -force;
            } else {
                temp_vel += direction * speed_left;
            }
        } else {
            temp_vel += direction * R::from_i32(info.force);
        }
        core.vel = crate::collision::clamp_vel(character.move_restrictions, temp_vel);
    } else if info.kind == map::TILE_SPEED_BOOST as i32 {
        let max_speed_scale = R::from_i32(5);
        let max_speed = if info.max_speed == 0 {
            // `MaxSpeed = std::max(MaxRampSpeed, GetTuning(...)->m_VelrampStart / 50) *
            // MaxSpeedScale;` (`character.cpp:1573-1575`) — the right-hand side is `float`, but
            // `MaxSpeed` itself is declared `int` (`character.cpp:1516`), so this assignment
            // *truncates* toward zero; the truncated `int` is what `TempMaxSpeed = MaxSpeed /
            // MaxSpeedScale` (line 1577) then divides — not the untruncated float value, which
            // an earlier revision of this port used directly.
            let ramp = max_ramp_speed(core.tuning.velramp_range::<R>(), core.tuning.velramp_curvature::<R>());
            let computed = (ramp.max(core.tuning.velramp_start::<R>() / R::from_i32(50))) * max_speed_scale;
            R::from_i32(computed.to_i32_trunc())
        } else {
            R::from_i32(info.max_speed)
        };
        let current_directional_speed = vmath::dot(direction, core.vel);
        let temp_max_speed = max_speed / max_speed_scale;
        if current_directional_speed + R::from_i32(info.force) > temp_max_speed {
            temp_vel += direction * (temp_max_speed - current_directional_speed);
        } else {
            temp_vel += direction * R::from_i32(info.force);
        }
        core.vel = crate::collision::clamp_vel(character.move_restrictions, temp_vel);
    }
}

/// `m_VelrampRange / (50 * log(std::max((float)Curvature, 1.01f)))` (`character.cpp:1580`),
/// i.e. `MaxRampSpeed`'s *entire* right-hand side, not just the `log` call: `log(...)` here is
/// the bare, unqualified C library function (no `std::` prefix in the source) — `<cmath>`'s
/// `float log(float)` overload is only found via `std::log`/unqualified lookup after a `using
/// std::log;`, neither of which applies here, so this resolves to plain `double log(double)`
/// (confirmed by review round 2's disassembly of the real Oracle B binary:
/// `cvtss2sd` -> `call log@plt` -> `mulsd` (by `50.0`) -> `divsd` (by `(double)m_VelrampRange`)
/// -> `cvtsd2ss`, once, at the very end). So `50 * log(...)` and the outer `/` both compute in
/// `double`, promoting the bare `int` `50` and the `float` `m_VelrampRange` along with it — only
/// the initial `std::max((float)Curvature, 1.01f)` is genuine `float` precision (an explicit
/// `(float)` cast and an `f` literal), its result losslessly widened to `double` for everything
/// after. An earlier revision of this port computed the division/multiplication in `R`'s own
/// native precision (even after routing the `log` call itself through `f64`), which is wrong for
/// the `f32` instantiation — found via review (`speedtune_v0.trb`: `tune velramp_range 227.73`/
/// `velramp_curvature 1.02`, tick 4 `vel_x` `230.0` (reference, `MaxSpeed` `1150`) vs `229.80078`
/// (this port's old, `f32`-throughout-past-the-`log`-call computation, `MaxSpeed` `1149`)).
fn max_ramp_speed<R: Real>(velramp_range: R, velramp_curvature: R) -> R {
    let clamped_curvature = velramp_curvature.max(R::from_f64(1.01));
    // The bare C `log` is glibc's double-precision one (D-127: `ddai-libm`'s port, the same bits everywhere).
    let denom = 50.0 * ddai_libm::log(clamped_curvature.to_f64());
    R::from_f64(velramp_range.to_f64() / denom)
}

/// The angle formula `character.cpp:1533-1544`/`1546-1557` repeats verbatim for the speeder
/// direction and the tee's own velocity — `std::atan`/`std::asin(1.0f)` (`pi/2`), always `f32`
/// regardless of `R` (the C++ source never widens this to `double`).
fn old_speedup_angle<R: Real>(x: R, y: R) -> R {
    let half_pi = R::from_f64(std::f64::consts::FRAC_PI_2);
    let mut angle = if x > R::from_f64(0.0000001) {
        -(y / x).atan() // libm-census: `Real::atan`
    } else if x < R::from_f64(0.0000001) {
        (y / x).atan() + R::from_i32(2) * half_pi // libm-census: `Real::atan`
    } else if y > R::from_f64(0.0000001) {
        half_pi
    } else {
        -half_pi
    };
    if angle < R::ZERO {
        angle = R::from_i32(4) * half_pi + angle;
    }
    angle
}

/// Regression tests for review round 1's F5 (three C++ int/double-promotion semantics
/// `apply_speedup` must match exactly, not compute in plain `R` precision throughout). Expected
/// values were independently derived (a standalone `f32`/`f64` program outside this crate,
/// reproducing each C++ expression's exact type at every step — not by calling this crate's own
/// code) — see this crate's `BUILD REPORT` for the derivation.
#[cfg(test)]
mod speedup_tests {
    use super::*;

    fn info(dir: Vec2<f32>, force: i32, max_speed: i32, kind: i32) -> crate::collision::SpeedupInfo<f32> {
        crate::collision::SpeedupInfo {
            dir,
            force,
            max_speed,
            kind,
        }
    }

    /// `Direction * (MaxSpeed / 5)` (`character.cpp:1521`, `Force == 255` branch): `MaxSpeed`/`5`
    /// are both C++ `int`, so this is integer (truncating) division — `594 / 5 == 118`, not the
    /// `118.8` plain-float division an earlier revision of this port computed.
    #[test]
    fn old_type_force_255_divides_max_speed_by_5_as_integers() {
        let mut core = CharacterCore::<f32>::default();
        let mut character = Character::<f32>::default();
        apply_speedup(
            &mut core,
            &mut character,
            &info(Vec2::new(1.0, 0.0), 255, 594, map::TILE_SPEED_BOOST_OLD as i32),
        );
        assert_eq!(
            core.vel,
            Vec2::new(118.0, 0.0),
            "594 / 5 must truncate to 118, not stay 118.8"
        );
    }

    /// `MaxSpeed = std::max(MaxRampSpeed, VelrampStart / 50) * MaxSpeedScale` (`character.cpp:
    /// 1573-1575`) assigns a `float` right-hand side to the `int` `MaxSpeed`, truncating it
    /// *before* `TempMaxSpeed = MaxSpeed / MaxSpeedScale` (line 1577) divides it back down —
    /// not the untruncated float value an earlier revision of this port used directly.
    /// `velramp_range = 0` forces `MaxRampSpeed` itself to `0`, so `max(...)` always picks
    /// `VelrampStart / 50` here, keeping the scenario simple: `101 / 50 == 2.02`, `* 5 == 10.1`,
    /// truncated to `10` — vs. `10.1` unclamped.
    #[test]
    fn new_type_max_speed_zero_truncates_computed_max_speed_to_an_integer() {
        let mut core = CharacterCore::<f32>::default();
        core.tuning.set_by_name("velramp_range", 0.0);
        core.tuning.set_by_name("velramp_curvature", 1.5);
        core.tuning.set_by_name("velramp_start", 101.0);
        let mut character = Character::<f32>::default();
        apply_speedup(
            &mut core,
            &mut character,
            &info(Vec2::new(1.0, 0.0), 100, 0, map::TILE_SPEED_BOOST as i32),
        );
        // `TempMaxSpeed = 10 / 5 = 2.0`; `CurrentDirectionalSpeed (0) + Force (100) > 2.0`, so
        // `TempVel += Direction * (TempMaxSpeed - 0) = (2.0, 0.0)` — not `(2.02, 0.0)`, which an
        // unclamped `MaxSpeed = 10.1` would have produced instead.
        assert_eq!(
            core.vel,
            Vec2::new(2.0, 0.0),
            "MaxSpeed must truncate to the integer 10 (not stay 10.1) before TempMaxSpeed divides it"
        );
    }

    /// Review round 2, finding F5 [CONFIRMED]: `MaxRampSpeed`'s `log(...)` call (bare, not
    /// `std::log` — see [`max_ramp_speed`]'s own doc comment) is `double`-only, so the whole
    /// `range / (50 * log(...))` expression must compute in `f64`, not `R`'s own precision, with
    /// exactly one final rounding to `R`. Exact reproduction of `speedtune_v0.trb`'s evidence
    /// (`tune velramp_range 227.73`, `tune velramp_curvature 1.02`): the reference's `MaxSpeed`
    /// is `1150` (`vel_x` `230.0`) — independently re-derived in Python (not by calling this
    /// crate's own code) for this test's expected value; see this crate's `BUILD REPORT` for the
    /// derivation. Computing `50 * log(...)` and the outer `/` in `f32` instead gives a result
    /// just *under* `230.0`, which truncates to the integer `229` (`MaxSpeed` `1145`) — an
    /// off-by-one this class of bug produces whenever the true result sits extremely close to an
    /// integer boundary, exactly the case here.
    #[test]
    fn new_type_max_ramp_speed_computes_log_in_f64_not_f32() {
        let mut core = CharacterCore::<f32>::default();
        core.tuning.set_by_name("velramp_range", 227.73);
        core.tuning.set_by_name("velramp_curvature", 1.02);
        core.tuning.set_by_name("velramp_start", 0.0);
        let mut character = Character::<f32>::default();
        apply_speedup(
            &mut core,
            &mut character,
            &info(Vec2::new(1.0, 0.0), 1000, 0, map::TILE_SPEED_BOOST as i32),
        );
        assert_eq!(
            core.vel,
            Vec2::new(230.0, 0.0),
            "MaxRampSpeed must round to exactly 230.0 (MaxSpeed 1150), not 229.99998... (MaxSpeed 1145 once truncated)"
        );
    }

    /// `TeeSpeed = std::sqrt(std::pow(TempVel.x, 2) + std::pow(TempVel.y, 2))`
    /// (`character.cpp:1554`): `std::pow(float, int)` is promoted to `double` by `<cmath>`'s
    /// generic overload (mismatched argument types — unlike [`Real::powf`]'s own
    /// `pow(float,float)` use elsewhere), so the whole sum-then-sqrt happens in `double`
    /// precision, only rounding to `float` once, at the end. `direction == vel` here (by
    /// construction) makes `DiffAngle == 0` exactly, so `SpeedLeft = MaxSpeed/5 - TeeSpeed`
    /// exposes `TeeSpeed`'s own precision directly, with `max_speed`/`force` chosen so the
    /// `else { TempVel += Direction * SpeedLeft }` branch (`character.cpp:1567`) is the one
    /// taken — the branch most sensitive to `SpeedLeft`'s exact value.
    #[test]
    fn old_type_tee_speed_computes_via_f64_sum_and_sqrt_not_f32_throughout() {
        let mut core = CharacterCore::<f32>::default();
        let vel = Vec2::new(-899.882_75, -1_107.157_1);
        core.vel = vel;
        let mut character = Character::<f32>::default();
        apply_speedup(
            &mut core,
            &mut character,
            &info(vel, 5000, 7134, map::TILE_SPEED_BOOST_OLD as i32),
        );
        // Independently derived (see this test module's doc comment): the `f64`-sum-and-sqrt
        // path gives `(-954.258, -1174.0569)`; an all-`f32` `TeeSpeed` gives `(-954.36786,
        // -1174.192)` instead for these exact inputs.
        assert_eq!(core.vel, Vec2::new(-954.258, -1174.0569));
    }
}

/// `CGameControllerDDNet::HandleCharacterTiles` (`gamemodes/ddnet.cpp:36-118`), minus
/// start/finish's chat-only warnings, minus `TILE_UNLOCK_TEAM` (dead — team-locking is never
/// reachable, see this module's doc comment), minus solo's chat message (the `SetSolo` call
/// itself is kept, it's the only physics-observable part).
pub fn handle_character_tiles<R: Real>(world: &mut World<R>, id: i32, map_index: i32) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    let pos = world.characters[id as usize].as_ref().unwrap().pos;
    let r3 = core::physical_size::<R>() / R::from_i32(3);
    let sample = |dx: R, dy: R| world.collision.get_pure_map_index(pos.x + dx, pos.y + dy) as i32;
    let s = [sample(r3, -r3), sample(r3, r3), sample(-r3, -r3), sample(-r3, r3)];
    let tile_at = |i: i32| world.collision.get_tile_index(i);
    let ftile_at = |i: i32| world.collision.get_front_tile_index(i);
    let tile_index = tile_at(map_index);
    let tile_findex = ftile_at(map_index);
    let is_start = tile_index == map::TILE_START as i32
        || tile_findex == map::TILE_START as i32
        || s.iter()
            .any(|&i| tile_at(i) == map::TILE_START as i32 || ftile_at(i) == map::TILE_START as i32);
    let is_finish = tile_index == map::TILE_FINISH as i32
        || tile_findex == map::TILE_FINISH as i32
        || s.iter()
            .any(|&i| tile_at(i) == map::TILE_FINISH as i32 || ftile_at(i) == map::TILE_FINISH as i32);

    let ddrace_state = world.characters[id as usize].map(|c| c.ddrace_state).unwrap_or(0);
    if is_start && ddrace_state != 2 {
        world.race_teams.on_character_start(
            &world.teams_core,
            &mut world.characters,
            id,
            world.tick,
            sv_team_forced_solo(&world.config),
        );
    } else if is_finish && ddrace_state == 1 {
        world.race_teams.on_character_finish(
            &world.teams_core,
            &mut world.characters,
            id,
            sv_team_forced_solo(&world.config),
        );
    }

    if tile_index == map::TILE_SOLO_ENABLE as i32 || tile_findex == map::TILE_SOLO_ENABLE as i32 {
        if !world.teams_core.get_solo(id) {
            world.teams_core.set_solo(id, true);
            world.cores.core_at_mut(slot).solo = true;
        }
    } else if (tile_index == map::TILE_SOLO_DISABLE as i32 || tile_findex == map::TILE_SOLO_DISABLE as i32)
        && world.teams_core.get_solo(id)
    {
        world.teams_core.set_solo(id, false);
        world.cores.core_at_mut(slot).solo = false;
    }
}

/// `CCharacter::HandleTiles(int Index)` (`character.cpp:1637-2124`), minus every
/// `SendChatTarget` (cosmetic) and `SetTimeCheckpoint`/broadcast bookkeeping (no compared field).
#[allow(clippy::too_many_lines)]
pub fn handle_tiles<R: Real>(world: &mut World<R>, id: i32, map_index: i32) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    let tile_index = world.collision.get_tile_index(map_index);
    let tile_findex = world.collision.get_front_tile_index(map_index);
    {
        let c = world.characters[id as usize].as_mut().unwrap();
        c.tile_index = tile_index;
        c.tile_findex = tile_findex;
    }

    // `m_MoveRestrictions = Collision()->GetMoveRestrictions(IsSwitchActiveCb, this, m_Pos,
    // 18.0f, MapIndex)` (`character.cpp:1642`) — the switch-aware, `MapIndex`-overridden version
    // (see [`core::tick`]'s doc comment for the same computation without the override). Written
    // to `Character::move_restrictions` (the *character*-level field, `character.h:136`) — see
    // that field's doc comment for why this must be kept separate from `CharacterCore`'s own
    // `move_restrictions` (a real, found-empirically DDNet distinction, not a duplicate).
    {
        let team = world.teams_core.team(id);
        let team_super = world.teams_core.team_super();
        let switchers = &world.cores.switchers;
        let pos = world.characters[id as usize].as_ref().unwrap().pos;
        let restrictions = world.collision.get_move_restrictions(
            Some(|number: u8| {
                team != team_super
                    && (number as usize) < switchers.len()
                    && switchers[number as usize].status[team as usize]
            }),
            pos,
            R::from_i32(18),
            Some(map_index),
        );
        world.characters[id as usize].as_mut().unwrap().move_restrictions = restrictions;
    }

    if map_index < 0 {
        let c = world.characters[id as usize].as_mut().unwrap();
        c.last_refill_jumps = false;
        c.last_penalty = false;
        c.last_bonus = false;
        return;
    }

    let tele_checkpoint = world.collision.is_tele_checkpoint(map_index);
    if tele_checkpoint != 0 {
        world.characters[id as usize].as_mut().unwrap().tele_checkpoint = tele_checkpoint;
    }

    handle_character_tiles(world, id, map_index);
    if !world.characters[id as usize].unwrap().alive {
        return;
    }

    let core_is_super = world.cores.core_at(slot).is_super;
    let core_invincible = world.cores.core_at(slot).invincible;
    let core_deep_frozen = world.cores.core_at(slot).deep_frozen;

    // freeze / unfreeze
    if (tile_index == map::TILE_FREEZE as i32 || tile_findex == map::TILE_FREEZE as i32)
        && !core_is_super
        && !core_invincible
        && !core_deep_frozen
    {
        let sv_freeze_delay = world.config.sv_freeze_delay;
        let tick = world.tick;
        let mut character = world.characters[id as usize].unwrap();
        let mut core = *world.cores.core_at(slot);
        freeze_default(&mut character, &mut core, tick, sv_freeze_delay);
        world.characters[id as usize] = Some(character);
        *world.cores.core_at_mut(slot) = core;
    } else if (tile_index == map::TILE_UNFREEZE as i32 || tile_findex == map::TILE_UNFREEZE as i32) && !core_deep_frozen
    {
        let mut character = world.characters[id as usize].unwrap();
        let mut core = *world.cores.core_at(slot);
        unfreeze(&mut character, &mut core);
        world.characters[id as usize] = Some(character);
        *world.cores.core_at_mut(slot) = core;
    }

    // deep freeze
    let core = world.cores.core_at_mut(slot);
    if (tile_index == map::TILE_DFREEZE as i32 || tile_findex == map::TILE_DFREEZE as i32)
        && !core.is_super
        && !core.invincible
        && !core.deep_frozen
    {
        core.deep_frozen = true;
    } else if (tile_index == map::TILE_DUNFREEZE as i32 || tile_findex == map::TILE_DUNFREEZE as i32)
        && !core.is_super
        && !core.invincible
        && core.deep_frozen
    {
        core.deep_frozen = false;
    }

    // live freeze
    if (tile_index == map::TILE_LFREEZE as i32 || tile_findex == map::TILE_LFREEZE as i32)
        && !core.is_super
        && !core.invincible
    {
        core.live_frozen = true;
    } else if (tile_index == map::TILE_LUNFREEZE as i32 || tile_findex == map::TILE_LUNFREEZE as i32)
        && !core.is_super
        && !core.invincible
    {
        core.live_frozen = false;
    }

    // endless hook
    if tile_index == map::TILE_EHOOK_ENABLE as i32 || tile_findex == map::TILE_EHOOK_ENABLE as i32 {
        set_endless_hook(core, true);
    } else if tile_index == map::TILE_EHOOK_DISABLE as i32 || tile_findex == map::TILE_EHOOK_DISABLE as i32 {
        set_endless_hook(core, false);
    }

    // hit others (all 4 at once)
    if (tile_index == map::TILE_HIT_DISABLE as i32 || tile_findex == map::TILE_HIT_DISABLE as i32)
        && (!core.hammer_hit_disabled
            || !core.shotgun_hit_disabled
            || !core.grenade_hit_disabled
            || !core.laser_hit_disabled)
    {
        core.hammer_hit_disabled = true;
        core.shotgun_hit_disabled = true;
        core.grenade_hit_disabled = true;
        core.laser_hit_disabled = true;
    } else if (tile_index == map::TILE_HIT_ENABLE as i32 || tile_findex == map::TILE_HIT_ENABLE as i32)
        && (core.hammer_hit_disabled
            || core.shotgun_hit_disabled
            || core.grenade_hit_disabled
            || core.laser_hit_disabled)
    {
        core.shotgun_hit_disabled = false;
        core.grenade_hit_disabled = false;
        core.hammer_hit_disabled = false;
        core.laser_hit_disabled = false;
    }

    // collide with others
    if (tile_index == map::TILE_NPC_DISABLE as i32 || tile_findex == map::TILE_NPC_DISABLE as i32)
        && !core.collision_disabled
    {
        core.collision_disabled = true;
    } else if (tile_index == map::TILE_NPC_ENABLE as i32 || tile_findex == map::TILE_NPC_ENABLE as i32)
        && core.collision_disabled
    {
        core.collision_disabled = false;
    }

    // hook others
    if (tile_index == map::TILE_NPH_DISABLE as i32 || tile_findex == map::TILE_NPH_DISABLE as i32)
        && !core.hook_hit_disabled
    {
        core.hook_hit_disabled = true;
    } else if (tile_index == map::TILE_NPH_ENABLE as i32 || tile_findex == map::TILE_NPH_ENABLE as i32)
        && core.hook_hit_disabled
    {
        core.hook_hit_disabled = false;
    }

    // unlimited air jumps
    if (tile_index == map::TILE_UNLIMITED_JUMPS_ENABLE as i32 || tile_findex == map::TILE_UNLIMITED_JUMPS_ENABLE as i32)
        && !core.endless_jump
    {
        core.endless_jump = true;
    } else if (tile_index == map::TILE_UNLIMITED_JUMPS_DISABLE as i32
        || tile_findex == map::TILE_UNLIMITED_JUMPS_DISABLE as i32)
        && core.endless_jump
    {
        core.endless_jump = false;
    }

    // walljump
    if (tile_index == map::TILE_WALLJUMP as i32 || tile_findex == map::TILE_WALLJUMP as i32)
        && core.vel.y > R::ZERO
        && core.colliding != 0
        && core.left_wall
    {
        core.left_wall = false;
        core.jumped_total = if core.jumps >= 2 { core.jumps - 2 } else { 0 };
        core.jumped = 1;
    }

    // jetpack gun
    if (tile_index == map::TILE_JETPACK_ENABLE as i32 || tile_findex == map::TILE_JETPACK_ENABLE as i32)
        && !core.jetpack
    {
        core.jetpack = true;
    } else if (tile_index == map::TILE_JETPACK_DISABLE as i32 || tile_findex == map::TILE_JETPACK_DISABLE as i32)
        && core.jetpack
    {
        core.jetpack = false;
    }

    // refill jumps
    let mut character = world.characters[id as usize].unwrap();
    if (tile_index == map::TILE_REFILL_JUMPS as i32 || tile_findex == map::TILE_REFILL_JUMPS as i32)
        && !character.last_refill_jumps
    {
        core.jumped_total = 0;
        core.jumped = 0;
        character.last_refill_jumps = true;
    }
    if tile_index != map::TILE_REFILL_JUMPS as i32 && tile_findex != map::TILE_REFILL_JUMPS as i32 {
        character.last_refill_jumps = false;
    }

    // telegun tiles
    if (tile_index == map::TILE_TELE_GUN_ENABLE as i32 || tile_findex == map::TILE_TELE_GUN_ENABLE as i32)
        && !core.has_telegun_gun
    {
        core.has_telegun_gun = true;
    } else if (tile_index == map::TILE_TELE_GUN_DISABLE as i32 || tile_findex == map::TILE_TELE_GUN_DISABLE as i32)
        && core.has_telegun_gun
    {
        core.has_telegun_gun = false;
    }
    if (tile_index == map::TILE_TELE_GRENADE_ENABLE as i32 || tile_findex == map::TILE_TELE_GRENADE_ENABLE as i32)
        && !core.has_telegun_grenade
    {
        core.has_telegun_grenade = true;
    } else if (tile_index == map::TILE_TELE_GRENADE_DISABLE as i32
        || tile_findex == map::TILE_TELE_GRENADE_DISABLE as i32)
        && core.has_telegun_grenade
    {
        core.has_telegun_grenade = false;
    }
    if (tile_index == map::TILE_TELE_LASER_ENABLE as i32 || tile_findex == map::TILE_TELE_LASER_ENABLE as i32)
        && !core.has_telegun_laser
    {
        core.has_telegun_laser = true;
    } else if (tile_index == map::TILE_TELE_LASER_DISABLE as i32 || tile_findex == map::TILE_TELE_LASER_DISABLE as i32)
        && core.has_telegun_laser
    {
        core.has_telegun_laser = false;
    }

    // stopper
    if core.vel.y > R::ZERO && (character.move_restrictions & crate::collision::CANTMOVE_DOWN) != 0 {
        core.jumped = 0;
        core.jumped_total = 0;
    }
    apply_move_restrictions(core, character.move_restrictions);
    world.characters[id as usize] = Some(character);

    handle_switch_tiles(world, id, slot, map_index);
    handle_tele_tiles(world, id, slot, map_index);
}

/// The switch-tile dispatch half of `CCharacter::HandleTiles` (`character.cpp:1831-2014`), minus
/// every `SendChatTarget` (cosmetic).
fn handle_switch_tiles<R: Real>(world: &mut World<R>, id: i32, slot: usize, map_index: i32) {
    let switch_type = world.collision.get_switch_type(map_index);
    let switch_number = world.collision.get_switch_number(map_index);
    let switch_delay = world.collision.get_switch_delay(map_index);
    let team = world.teams_core.team(id);
    let tick = world.tick;
    let team_active = |world: &World<R>| {
        switch_number > 0
            && (switch_number as usize) < world.cores.switchers.len()
            && world.cores.switchers[switch_number as usize].status[team as usize]
    };

    if switch_type == map::TILE_SWITCHOPEN as i32 && team != TEAM_SUPER && switch_number > 0 {
        let s = &mut world.cores.switchers[switch_number as usize];
        s.status[team as usize] = true;
        s.end_tick[team as usize] = 0;
        s.kind[team as usize] = map::TILE_SWITCHOPEN as i32;
        s.last_update_tick[team as usize] = tick;
    } else if switch_type == map::TILE_SWITCHTIMEDOPEN as i32 && team != TEAM_SUPER && switch_number > 0 {
        let s = &mut world.cores.switchers[switch_number as usize];
        s.status[team as usize] = true;
        s.end_tick[team as usize] = tick + 1 + switch_delay * core::SERVER_TICK_SPEED;
        s.kind[team as usize] = map::TILE_SWITCHTIMEDOPEN as i32;
        s.last_update_tick[team as usize] = tick;
        switch::mark_switcher_timed(&mut world.active_timed_switchers, switch_number as u8);
    } else if switch_type == map::TILE_SWITCHTIMEDCLOSE as i32 && team != TEAM_SUPER && switch_number > 0 {
        let s = &mut world.cores.switchers[switch_number as usize];
        s.status[team as usize] = false;
        s.end_tick[team as usize] = tick + 1 + switch_delay * core::SERVER_TICK_SPEED;
        s.kind[team as usize] = map::TILE_SWITCHTIMEDCLOSE as i32;
        s.last_update_tick[team as usize] = tick;
        switch::mark_switcher_timed(&mut world.active_timed_switchers, switch_number as u8);
    } else if switch_type == map::TILE_SWITCHCLOSE as i32 && team != TEAM_SUPER && switch_number > 0 {
        let s = &mut world.cores.switchers[switch_number as usize];
        s.status[team as usize] = false;
        s.end_tick[team as usize] = 0;
        s.kind[team as usize] = map::TILE_SWITCHCLOSE as i32;
        s.last_update_tick[team as usize] = tick;
    } else if switch_type == map::TILE_FREEZE as i32 && team != TEAM_SUPER && !world.cores.core_at(slot).invincible {
        if switch_number == 0 || team_active(world) {
            let sv_freeze_delay = switch_delay;
            let mut character = world.characters[id as usize].unwrap();
            let mut core = *world.cores.core_at(slot);
            freeze(&mut character, &mut core, tick, sv_freeze_delay);
            world.characters[id as usize] = Some(character);
            *world.cores.core_at_mut(slot) = core;
        }
    } else if switch_type == map::TILE_DFREEZE as i32 && team != TEAM_SUPER && !world.cores.core_at(slot).invincible {
        if switch_number == 0 || team_active(world) {
            world.cores.core_at_mut(slot).deep_frozen = true;
        }
    } else if switch_type == map::TILE_DUNFREEZE as i32 && team != TEAM_SUPER && !world.cores.core_at(slot).invincible {
        if switch_number == 0 || team_active(world) {
            world.cores.core_at_mut(slot).deep_frozen = false;
        }
    } else if switch_type == map::TILE_LFREEZE as i32 && team != TEAM_SUPER && !world.cores.core_at(slot).invincible {
        if switch_number == 0 || team_active(world) {
            world.cores.core_at_mut(slot).live_frozen = true;
        }
    } else if switch_type == map::TILE_LUNFREEZE as i32 && team != TEAM_SUPER && !world.cores.core_at(slot).invincible {
        if switch_number == 0 || team_active(world) {
            world.cores.core_at_mut(slot).live_frozen = false;
        }
    } else if switch_type == map::TILE_HIT_ENABLE as i32
        && world.cores.core_at(slot).hammer_hit_disabled
        && switch_delay == WEAPON_HAMMER
    {
        world.cores.core_at_mut(slot).hammer_hit_disabled = false;
    } else if switch_type == map::TILE_HIT_DISABLE as i32
        && !world.cores.core_at(slot).hammer_hit_disabled
        && switch_delay == WEAPON_HAMMER
    {
        world.cores.core_at_mut(slot).hammer_hit_disabled = true;
    } else if switch_type == map::TILE_HIT_ENABLE as i32
        && world.cores.core_at(slot).shotgun_hit_disabled
        && switch_delay == WEAPON_SHOTGUN
    {
        world.cores.core_at_mut(slot).shotgun_hit_disabled = false;
    } else if switch_type == map::TILE_HIT_DISABLE as i32
        && !world.cores.core_at(slot).shotgun_hit_disabled
        && switch_delay == WEAPON_SHOTGUN
    {
        world.cores.core_at_mut(slot).shotgun_hit_disabled = true;
    } else if switch_type == map::TILE_HIT_ENABLE as i32
        && world.cores.core_at(slot).grenade_hit_disabled
        && switch_delay == WEAPON_GRENADE
    {
        world.cores.core_at_mut(slot).grenade_hit_disabled = false;
    } else if switch_type == map::TILE_HIT_DISABLE as i32
        && !world.cores.core_at(slot).grenade_hit_disabled
        && switch_delay == WEAPON_GRENADE
    {
        world.cores.core_at_mut(slot).grenade_hit_disabled = true;
    } else if switch_type == map::TILE_HIT_ENABLE as i32
        && world.cores.core_at(slot).laser_hit_disabled
        && switch_delay == WEAPON_LASER
    {
        world.cores.core_at_mut(slot).laser_hit_disabled = false;
    } else if switch_type == map::TILE_HIT_DISABLE as i32
        && !world.cores.core_at(slot).laser_hit_disabled
        && switch_delay == WEAPON_LASER
    {
        world.cores.core_at_mut(slot).laser_hit_disabled = true;
    } else if switch_type == map::TILE_JUMP as i32 {
        let new_jumps = if switch_delay == 255 { -1 } else { switch_delay };
        if new_jumps != world.cores.core_at(slot).jumps {
            world.cores.core_at_mut(slot).jumps = new_jumps;
        }
    } else if switch_type == map::TILE_ADD_TIME as i32 && !world.characters[id as usize].unwrap().last_penalty {
        propagate_start_time(
            world,
            id,
            team,
            (switch_delay * 60 + switch_number) * core::SERVER_TICK_SPEED,
            true,
        );
        world.characters[id as usize].as_mut().unwrap().last_penalty = true;
    } else if switch_type == map::TILE_SUBTRACT_TIME as i32 && !world.characters[id as usize].unwrap().last_bonus {
        propagate_start_time(
            world,
            id,
            team,
            (switch_delay * 60 + switch_number) * core::SERVER_TICK_SPEED,
            false,
        );
        world.characters[id as usize].as_mut().unwrap().last_bonus = true;
    }

    if switch_type != map::TILE_ADD_TIME as i32 {
        world.characters[id as usize].as_mut().unwrap().last_penalty = false;
    }
    if switch_type != map::TILE_SUBTRACT_TIME as i32 {
        world.characters[id as usize].as_mut().unwrap().last_bonus = false;
    }
}

/// `TILE_ADD_TIME`/`TILE_SUBTRACT_TIME`'s shared "adjust `m_StartTime`, then propagate to every
/// other character on the same non-flock, non-super team" logic (`character.cpp:1955-2004`).
/// `add`: `true` for `TILE_ADD_TIME` (subtracts from `m_StartTime`, making the elapsed time
/// larger), `false` for `TILE_SUBTRACT_TIME` (adds, clamped to not exceed the current tick).
fn propagate_start_time<R: Real>(world: &mut World<R>, id: i32, team: i32, delta_ticks: i32, add: bool) {
    let tick = world.tick;
    let me = world.characters[id as usize].as_mut().unwrap();
    if add {
        me.start_time -= delta_ticks;
    } else {
        me.start_time += delta_ticks;
        if me.start_time > tick {
            me.start_time = tick;
        }
    }
    let new_start_time = me.start_time;
    // `(SvTeam == SV_TEAM_FORCED_SOLO || (Team != TEAM_FLOCK && !TeamFlock(Team))) && Team !=
    // TEAM_SUPER` (`character.cpp:1963,1989`) — `TeamFlock` is always `false` here (see this
    // module's doc comment), so this simplifies to the condition below.
    if (sv_team_forced_solo(&world.config) || team != TEAM_FLOCK) && team != world.teams_core.team_super() {
        for i in 0..MAX_CLIENTS as i32 {
            if world.teams_core.team(i) == team
                && i != id
                && let Some(other) = world.characters[i as usize].as_mut()
            {
                other.start_time = new_start_time;
            }
        }
    }
}

/// The teleport half of `CCharacter::HandleTiles` (`character.cpp:2016-2123`): plain, evil,
/// check-evil, and check teleports, in that exact order/priority.
fn handle_tele_tiles<R: Real>(world: &mut World<R>, id: i32, slot: usize, map_index: i32) {
    let is_super = world.cores.core_at(slot).is_super;
    let invincible = world.cores.core_at(slot).invincible;

    let z = world.collision.is_teleport(map_index);
    if !world.config.sv_old_teleport_hook
        && !world.config.sv_old_teleport_weapons
        && z != 0
        && !world.collision.tele_outs((z - 1) as u8).is_empty()
    {
        if is_super || invincible {
            return;
        }
        let outs = world.collision.tele_outs((z - 1) as u8);
        let out = world.cores.random_or_0(outs.len() as i32);
        world.cores.core_at_mut(slot).pos = outs[out as usize];
        if !world.config.sv_teleport_hold_hook {
            reset_hook(&mut world.cores, id as u8, slot);
        }
        if world.config.sv_teleport_lose_weapons {
            reset_pickups(world.cores.core_at_mut(slot));
        }
        return;
    }

    let evil = world.collision.is_evil_teleport(map_index);
    if evil != 0 && !world.collision.tele_outs((evil - 1) as u8).is_empty() {
        if is_super || invincible {
            return;
        }
        let outs = world.collision.tele_outs((evil - 1) as u8);
        let out = world.cores.random_or_0(outs.len() as i32);
        world.cores.core_at_mut(slot).pos = outs[out as usize];
        if !world.config.sv_old_teleport_hook && !world.config.sv_old_teleport_weapons {
            world.cores.core_at_mut(slot).vel = Vec2::zero();
            if !world.config.sv_teleport_hold_hook {
                reset_hook(&mut world.cores, id as u8, slot);
                release_hooked_by_others(world, id);
            }
            if world.config.sv_teleport_lose_weapons {
                reset_pickups(world.cores.core_at_mut(slot));
            }
        }
        return;
    }

    if world.collision.is_check_evil_teleport(map_index) {
        if is_super || invincible {
            return;
        }
        let checkpoint = world.characters[id as usize].unwrap().tele_checkpoint;
        for k in (0..checkpoint).rev() {
            let outs = world.collision.tele_check_outs(k as u8);
            if !outs.is_empty() {
                let out = world.cores.random_or_0(outs.len() as i32);
                world.cores.core_at_mut(slot).pos = outs[out as usize];
                world.cores.core_at_mut(slot).vel = Vec2::zero();
                if !world.config.sv_teleport_hold_hook {
                    reset_hook(&mut world.cores, id as u8, slot);
                    release_hooked_by_others(world, id);
                }
                return;
            }
        }
        if let Some(pos) = world.can_spawn(id) {
            world.cores.core_at_mut(slot).pos = pos;
            world.cores.core_at_mut(slot).vel = Vec2::zero();
            if !world.config.sv_teleport_hold_hook {
                reset_hook(&mut world.cores, id as u8, slot);
                release_hooked_by_others(world, id);
            }
        }
        return;
    }

    if world.collision.is_check_teleport(map_index) {
        if is_super || invincible {
            return;
        }
        let checkpoint = world.characters[id as usize].unwrap().tele_checkpoint;
        for k in (0..checkpoint).rev() {
            let outs = world.collision.tele_check_outs(k as u8);
            if !outs.is_empty() {
                let out = world.cores.random_or_0(outs.len() as i32);
                world.cores.core_at_mut(slot).pos = outs[out as usize];
                if !world.config.sv_teleport_hold_hook {
                    reset_hook(&mut world.cores, id as u8, slot);
                }
                return;
            }
        }
        if let Some(pos) = world.can_spawn(id) {
            world.cores.core_at_mut(slot).pos = pos;
            if !world.config.sv_teleport_hold_hook {
                reset_hook(&mut world.cores, id as u8, slot);
            }
        }
    }
}

/// `CGameWorld::ReleaseHooked(int ClientId)` (`gameworld.cpp:382-392`): releases every *other*
/// character currently hooking `id` (never `id` itself if it's `IsSuper()`, matching the C++
/// `&& !pChr->IsSuper()` guard — but the target of a hook is always non-super here anyway since
/// this is only called right after teleporting `id` itself, not `id`'s hooker).
fn release_hooked_by_others<R: Real>(world: &mut World<R>, id: i32) {
    for other_slot in 0..world.cores.len() {
        if world.cores.core_at(other_slot).hooked_player() == id && !world.cores.core_at(other_slot).is_super {
            let other_id = world.cores.id_at(other_slot);
            release_hook(&mut world.cores, other_id, other_slot);
        }
    }
}

/// `CCharacter::DDRacePostCoreTick()` (`character.cpp:2288-2365`), minus `m_Time`'s computation
/// (only read by `HandleBroadcast`, cosmetic) and the teleport-gun `CreateDeath`/`CreateSound`
/// calls (cosmetic; the position/velocity changes themselves are kept). Returns whether the
/// character is still alive (`HandleSkippableTiles`/`HandleTiles` may kill it).
pub fn ddrace_post_core_tick<R: Real>(world: &mut World<R>, id: i32) -> bool {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return false;
    };
    {
        let core = world.cores.core_at_mut(slot);
        if core.endless_hook {
            core.hook_tick = 0;
        }
    }
    world.characters[id as usize].as_mut().unwrap().frozen_last_tick = false;

    let core = world.cores.core_at_mut(slot);
    if core.deep_frozen && !core.is_super && !core.invincible {
        let mut character = world.characters[id as usize].unwrap();
        let mut core_copy = *core;
        let tick = world.tick;
        let sv_freeze_delay = world.config.sv_freeze_delay;
        freeze_default(&mut character, &mut core_copy, tick, sv_freeze_delay);
        world.characters[id as usize] = Some(character);
        *world.cores.core_at_mut(slot) = core_copy;
    }

    // Jump-count bookkeeping (`character.cpp:2300-2326`) — can be overridden by tiles like
    // Refill Jumps/Stopper/Wall Jump (already applied by [`handle_tiles`] before this runs, per
    // `DDRacePostCoreTick`'s own place in `CCharacter::Tick`).
    let core = world.cores.core_at_mut(slot);
    // Kept as 3 branches (the first two share a body) rather than merged with `||`, matching
    // `character.cpp:2299-2318`'s own 4-branch `if`/`else if` chain letter for letter (each has
    // its own distinct comment there: "only one ground jump", "no jumps at all", "each jump is
    // the last one").
    #[allow(clippy::if_same_then_else)]
    if core.jumps == -1 || core.jumps == 0 {
        core.jumped |= 2;
    } else if core.jumps == 1 && core.jumped > 0 {
        core.jumped |= 2;
    } else if core.jumped_total < core.jumps - 1 && core.jumped > 1 {
        core.jumped = 1;
    }
    if (core.is_super || core.endless_jump) && core.jumped > 1 {
        core.jumped = 1;
    }

    // `CCharacter::DDRacePostCoreTick` reads `m_Pos` (`character.cpp:2328,2334`): the entity
    // position, which a ninja dash earlier in this very tick has not yet moved.
    let pos = world.characters[id as usize].as_ref().unwrap().pos;
    let current_index = world.collision.get_map_index(pos);
    if !handle_skippable_tiles(world, id, current_index) {
        return false;
    }

    let prev_pos = world.characters[id as usize].as_ref().unwrap().prev_pos;
    let mut indices = std::mem::take(&mut world.map_indices_scratch);
    world.collision.get_map_indices_into(prev_pos, pos, 0, &mut indices);
    let mut alive = true;
    if indices.is_empty() {
        handle_tiles(world, id, current_index);
        alive = world.characters[id as usize].unwrap().alive;
    } else {
        for &index in &indices {
            handle_tiles(world, id, index);
            if !world.characters[id as usize].unwrap().alive {
                alive = false;
                break;
            }
        }
    }
    indices.clear();
    world.map_indices_scratch = indices;
    if !alive {
        return false;
    }

    // Teleport gun (`character.cpp:2351-2362`) — cosmetic `CreateDeath`/`CreateSound` dropped.
    let character = world.characters[id as usize].as_mut().unwrap();
    if character.tele_gun_teleport {
        let dest = character.tele_gun_pos;
        let blue = character.is_blue_tele_gun_teleport;
        character.tele_gun_teleport = false;
        character.is_blue_tele_gun_teleport = false;
        world.cores.core_at_mut(slot).pos = dest;
        if !blue {
            world.cores.core_at_mut(slot).vel = Vec2::zero();
        }
    }
    true
}

// --- Weapons: `HandleWeaponSwitch`/`DoWeaponSwitch`/`FireWeapon`/`HandleWeapons`/`HandleJetpack`
// (`character.cpp:253-676`) --------------------------------------------------------------------

/// `INPUT_STATE_MASK` (`generated/protocol.h`, `datasrc/network.py`).
pub const INPUT_STATE_MASK: i32 = 0x3f;

/// `CountInput(int Prev, int Cur)` (`gamecore.h:296-309`): presses/releases between two masked
/// press-counters.
pub fn count_input_presses(prev: i32, cur: i32) -> i32 {
    let prev = prev & INPUT_STATE_MASK;
    let cur = cur & INPUT_STATE_MASK;
    let mut i = prev;
    let mut presses = 0;
    while i != cur {
        i = (i + 1) & INPUT_STATE_MASK;
        if i & 1 != 0 {
            presses += 1;
        }
    }
    presses
}

/// `CCharacter::OnPredictedInput(const CNetObj_PlayerInput*)` (`character.cpp:728-742`): copies
/// the input into `m_Input`/`m_SavedInput`, with the "never aim exactly at the center" fix — no
/// weapon-switch/fire logic here (that's [`on_direct_input`], called earlier in the tick — see
/// [`World::step`]).
pub fn on_predicted_input<R: Real>(character: &mut Character<R>, mut new_input: PlayerInput) {
    if new_input.target_x == 0 && new_input.target_y == 0 {
        new_input.target_y = -1;
    }
    character.input = new_input;
    character.saved_input = new_input;
}

/// `CCharacter::OnDirectInput(const CNetObj_PlayerInput*)` (`character.cpp:744-764`), minus
/// antibot. Also calls [`handle_weapon_switch`]/[`fire_weapon`] when `m_NumInputs > 1` (always
/// true from the second tick a character exists onward, given this harness's idealized —
/// never-null — per-tick input) — see this function's citation for exactly why `m_LatestPrevInput`
/// ends up holding the *previous* tick's input by the time those two read it.
pub fn on_direct_input<R: Real>(world: &mut World<R>, id: i32, new_input: PlayerInput) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    {
        let c = world.characters[id as usize].as_mut().unwrap();
        c.latest_prev_input = c.latest_input;
        c.latest_input = new_input;
        if c.latest_input.target_x == 0 && c.latest_input.target_y == 0 {
            c.latest_input.target_y = -1;
        }
        c.num_inputs += 1;
    }
    let num_inputs = world.characters[id as usize].unwrap().num_inputs;
    if num_inputs > 1 {
        // `m_pPlayer->GetTeam() != TEAM_SPECTATORS`: always true (this harness never has
        // spectators).
        handle_weapon_switch(world, id, slot);
        fire_weapon(world, id, slot);
    }
    let c = world.characters[id as usize].as_mut().unwrap();
    c.latest_prev_input = c.latest_input;
}

/// `CCharacter::HandleWeaponSwitch()` (`character.cpp:408-453`).
pub fn handle_weapon_switch<R: Real>(world: &mut World<R>, id: i32, slot: usize) {
    // Task 3.6: read through references (nothing below writes the world until the queue update).
    let character = world.characters[id as usize].as_ref().unwrap();
    let core = world.cores.core_at(slot);
    let mut wanted_weapon = core.active_weapon;
    if character.queued_weapon != -1 {
        wanted_weapon = character.queued_weapon;
    }
    if !(0..NUM_WEAPONS as i32 - 1).any(|i| core.weapons[i as usize].got) {
        return;
    }
    let next = count_input_presses(
        character.latest_prev_input.next_weapon,
        character.latest_input.next_weapon,
    );
    let prev = count_input_presses(
        character.latest_prev_input.prev_weapon,
        character.latest_input.prev_weapon,
    );
    if next < 128 {
        let mut n = next;
        while n != 0 {
            wanted_weapon = (wanted_weapon + 1) % NUM_WEAPONS as i32;
            if core.weapons[wanted_weapon as usize].got {
                n -= 1;
            }
        }
    }
    if prev < 128 {
        let mut n = prev;
        while n != 0 {
            wanted_weapon = if wanted_weapon - 1 < 0 {
                NUM_WEAPONS as i32 - 1
            } else {
                wanted_weapon - 1
            };
            if core.weapons[wanted_weapon as usize].got {
                n -= 1;
            }
        }
    }
    if character.latest_input.wanted_weapon != 0 {
        // `m_Input.m_WantedWeapon - 1` (not `m_LatestInput`!) — matches `character.cpp:446`
        // exactly (a long-standing DDNet quirk this port preserves bit-for-bit).
        wanted_weapon = core.input.wanted_weapon - 1;
    }
    let set_queue = (0..NUM_WEAPONS as i32).contains(&wanted_weapon)
        && wanted_weapon != core.active_weapon
        && core.weapons[wanted_weapon as usize].got;
    if set_queue {
        world.characters[id as usize].as_mut().unwrap().queued_weapon = wanted_weapon;
    }
    do_weapon_switch(world, id, slot);
}

/// `CCharacter::DoWeaponSwitch()` (`character.cpp:396-406`). Relies entirely on `set_weapon`
/// (private below) to clear `m_QueuedWeapon` (matching the C++ source, which has no explicit
/// `m_QueuedWeapon = -1` of its own here — review self-check: an earlier revision of this port *did* add one,
/// using a `character` snapshot taken *before* calling `set_weapon`, and then wrote that stale
/// snapshot back *after* it — silently reverting `set_weapon`'s own `m_LastWeapon` update. Fixed
/// by not touching `world.characters[id]` again after `set_weapon` returns.)
pub fn do_weapon_switch<R: Real>(world: &mut World<R>, id: i32, slot: usize) {
    let character = world.characters[id as usize].as_ref().unwrap();
    let (reload_timer, queued_weapon) = (character.reload_timer, character.queued_weapon);
    if reload_timer != 0 || queued_weapon == -1 {
        return;
    }
    let core = world.cores.core_at(slot);
    if core.weapons[WEAPON_NINJA as usize].got || !core.weapons[queued_weapon as usize].got {
        return;
    }
    set_weapon(world, id, slot, queued_weapon);
}

/// `CCharacter::SetWeapon(int W)` (`character.cpp:154-166`), minus the switch sound.
fn set_weapon<R: Real>(world: &mut World<R>, id: i32, slot: usize, w: i32) {
    let core = world.cores.core_at_mut(slot);
    if w == core.active_weapon {
        return;
    }
    let mut character = world.characters[id as usize].unwrap();
    character.last_weapon = core.active_weapon;
    character.queued_weapon = -1;
    core.active_weapon = w;
    if core.active_weapon < 0 || core.active_weapon >= NUM_WEAPONS as i32 {
        core.active_weapon = 0;
    }
    world.characters[id as usize] = Some(character);
}

/// `CCharacter::FireWeapon()` (`character.cpp:455-656`): every weapon — hammer, gun, shotgun and
/// laser (a `CLaser`, `laser::laser_new`), grenade, ninja (the dash activation,
/// `ninja::ninja_activate`).
#[allow(clippy::too_many_lines)]
pub fn fire_weapon<R: Real>(world: &mut World<R>, id: i32, slot: usize) {
    if world.characters[id as usize].as_ref().unwrap().reload_timer != 0 {
        return;
    }

    do_weapon_switch(world, id, slot);
    // Task 3.6: the common tick does not fire. Decide that from references, before copying the 300 B
    // character and the 488 B core; the checks are the ones below, and every early return below only
    // wrote the unchanged copy back, so returning here leaves the world exactly as it would be.
    {
        let character = world.characters[id as usize].as_ref().unwrap();
        let core = world.cores.core_at(slot);
        let mut full_auto = core.active_weapon == WEAPON_GRENADE
            || core.active_weapon == WEAPON_SHOTGUN
            || core.active_weapon == WEAPON_LASER;
        if core.jetpack && core.active_weapon == WEAPON_GUN {
            full_auto = true;
        }
        if character.frozen_last_tick {
            full_auto = true;
        }
        if !world.config.sv_deepfly && core.active_weapon == WEAPON_HAMMER && core.deep_frozen {
            return;
        }
        let mut will_fire = count_input_presses(character.latest_prev_input.fire, character.latest_input.fire) != 0;
        if full_auto
            && (character.latest_input.fire & 1) != 0
            && core.active_weapon >= 0
            && core.weapons[core.active_weapon as usize].ammo != 0
        {
            will_fire = true;
        }
        if !will_fire {
            return;
        }
    }
    let mut character = world.characters[id as usize].unwrap();
    let mut core = *world.cores.core_at(slot);

    let mouse_target = Vec2::new(
        R::from_i32(character.latest_input.target_x),
        R::from_i32(character.latest_input.target_y),
    );
    let direction = vmath::normalize(mouse_target);

    let mut full_auto = core.active_weapon == WEAPON_GRENADE
        || core.active_weapon == WEAPON_SHOTGUN
        || core.active_weapon == WEAPON_LASER;
    if core.jetpack && core.active_weapon == WEAPON_GUN {
        full_auto = true;
    }
    if character.frozen_last_tick {
        full_auto = true;
    }

    if !world.config.sv_deepfly && core.active_weapon == WEAPON_HAMMER && core.deep_frozen {
        world.characters[id as usize] = Some(character);
        return;
    }

    let mut will_fire = count_input_presses(character.latest_prev_input.fire, character.latest_input.fire) != 0;
    if full_auto
        && (character.latest_input.fire & 1) != 0
        && core.active_weapon >= 0
        && core.weapons[core.active_weapon as usize].ammo != 0
    {
        will_fire = true;
    }
    if !will_fire {
        world.characters[id as usize] = Some(character);
        return;
    }

    if character.freeze_time != 0 {
        // Pain-sound timer bookkeeping dropped (cosmetic only).
        world.characters[id as usize] = Some(character);
        return;
    }

    if core.active_weapon < 0 || core.weapons[core.active_weapon as usize].ammo == 0 {
        world.characters[id as usize] = Some(character);
        return;
    }

    // `m_Pos + Direction * GetProximityRadius() * 0.75f` (`character.cpp:510`) — left-associative,
    // i.e. `m_Pos + ((Direction * GetProximityRadius()) * 0.75f)`: *two* separate roundings per
    // component, not one multiplication by a precomputed `28.0 * 0.75 == 21.0` constant (found
    // empirically: the precomputed-constant form was 1 ulp off from the reference on 52/2453
    // corpus projectiles, even though `28.0 * 0.75` is itself exact — `direction`'s own components
    // generally aren't, so rounding `direction * 28.0` *before* the second multiply can differ
    // from rounding `direction * 21.0` directly).
    let proj_start_pos = character.pos + direction * core::physical_size::<R>() * R::from_f64(0.75);

    match core.active_weapon {
        WEAPON_HAMMER => {
            if !core.hammer_hit_disabled {
                fire_hammer(world, id, slot, proj_start_pos);
                // `fire_hammer` may have set `m_ReloadTimer` itself (a hit delay); re-read.
                character = world.characters[id as usize].unwrap();
                core = *world.cores.core_at(slot);
            }
        }
        WEAPON_GUN => {
            let lifetime = (core::SERVER_TICK_SPEED as f32 * core.tuning.gun_lifetime()) as i32;
            world.projectiles.insert(
                0,
                Projectile {
                    weapon_type: WEAPON_GUN,
                    owner: id,
                    pos: proj_start_pos,
                    direction,
                    init_dir: mouse_target,
                    life_span: lifetime,
                    start_tick: world.tick,
                    freeze: false,
                    explosive: false,
                    bouncing: 0,
                    tune_zone: character.tune_zone,
                    layer: Layer::Game,
                    number: 0,
                    marked_for_destroy: false,
                },
            );
        }
        WEAPON_GRENADE => {
            let lifetime = (core::SERVER_TICK_SPEED as f32 * core.tuning.grenade_lifetime()) as i32;
            world.projectiles.insert(
                0,
                Projectile {
                    weapon_type: WEAPON_GRENADE,
                    owner: id,
                    pos: proj_start_pos,
                    direction,
                    init_dir: mouse_target,
                    life_span: lifetime,
                    start_tick: world.tick,
                    freeze: false,
                    explosive: true,
                    bouncing: 0,
                    tune_zone: character.tune_zone,
                    layer: Layer::Game,
                    number: 0,
                    marked_for_destroy: false,
                },
            );
        }
        WEAPON_SHOTGUN | WEAPON_LASER => {
            // `new CLaser(.., m_Pos, Direction, LaserReach, owner, type)` (`character.cpp:587-626`);
            // its constructor already traces the first segment (and can hit anyone, change the
            // owner's own telegun fields, ...), so the working copies are flushed first and
            // re-read afterwards, like the hammer branch does.
            let laser_reach: R = world.tuning.zone(character.tune_zone).laser_reach();
            world.characters[id as usize] = Some(character);
            *world.cores.core_at_mut(slot) = core;
            laser::laser_new(world, character.pos, direction, laser_reach, id, core.active_weapon);
            character = world.characters[id as usize].unwrap();
            core = *world.cores.core_at(slot);
        }
        WEAPON_NINJA => ninja::ninja_activate(&mut character, &mut core, direction),
        _ => {}
    }

    character.attack_tick = world.tick;
    if character.reload_timer == 0 && core.active_weapon != -1 {
        character.reload_timer =
            (core.tuning.get_weapon_fire_delay(core.active_weapon) * core::SERVER_TICK_SPEED as f32) as i32;
    }
    world.characters[id as usize] = Some(character);
    *world.cores.core_at_mut(slot) = core;
}

/// `FireWeapon`'s `WEAPON_HAMMER` case (`character.cpp:514-568`), minus sound/antibot.
fn fire_hammer<R: Real>(world: &mut World<R>, id: i32, slot: usize, proj_start_pos: Vec2<R>) {
    // `GameServer()->m_World.FindEntities(ProjStartPos, GetProximityRadius()*0.5f, apEnts,
    // MAX_CLIENTS, ENTTYPE_CHARACTER)` (`character.cpp:525-526`) — `FindEntities`
    // (`gameworld.cpp:58-77`) checks `distance < Radius + pEnt->m_ProximityRadius`, i.e. the
    // *target's own* 28-unit proximity radius is added on top of the 14-unit search radius (a
    // self-check review round found: an earlier revision of this port compared against the bare
    // 14-unit radius alone, silently halving the effective hammer reach and missing hits a real
    // server would land — see [`find_characters_in_range`], which already gets this right, for
    // the pattern this now matches).
    let search_radius = core::physical_size::<R>() * R::from_f64(0.5);
    let mut hits = 0;
    let mut targets = std::mem::take(&mut world.hammer_scratch);
    targets.clear();
    for other_slot in 0..world.cores.len() {
        if other_slot == slot {
            continue;
        }
        let other_id = world.cores.id_at(other_slot) as i32;
        let alive = world.characters[other_id as usize].is_some_and(|c| c.alive);
        if alive && !character_can_collide(&world.teams_core, id, other_id) {
            continue;
        }
        // `pTarget->m_Pos`: the entity position (see [`Character::pos`]).
        let other_entity_pos =
            world.characters[other_id as usize].map_or(world.cores.core_at(other_slot).pos, |c| c.pos);
        if vmath::distance(other_entity_pos, proj_start_pos) < search_radius + character_proximity_radius::<R>() {
            targets.push(other_slot);
        }
    }
    for &other_slot in &targets {
        let other_id = world.cores.id_at(other_slot) as i32;
        let other_pos = world.characters[other_id as usize].map_or(world.cores.core_at(other_slot).pos, |c| c.pos);
        let self_pos = world.characters[id as usize].map_or(world.cores.core_at(slot).pos, |c| c.pos);

        let dir = if vmath::length(other_pos - self_pos) > R::ZERO {
            vmath::normalize(other_pos - self_pos)
        } else {
            Vec2::new(R::ZERO, -R::ONE)
        };
        let strength = world.cores.core_at(slot).tuning.hammer_strength::<R>();
        let other_vel = world.cores.core_at(other_slot).vel;
        // `pTarget->m_MoveRestrictions` (`character.cpp:550`) — the *target*'s own
        // character-level field (see [`apply_move_restrictions`]'s doc comment), not the
        // hammering character's.
        let other_move_restrictions = world.characters[other_id as usize].unwrap().move_restrictions;
        let mut temp = other_vel + vmath::normalize(dir + Vec2::new(R::ZERO, R::from_f64(-1.1))) * R::from_i32(10);
        temp = crate::collision::clamp_vel(other_move_restrictions, temp);
        temp = temp - other_vel;
        let force = (Vec2::new(R::ZERO, -R::ONE) + temp) * strength;
        {
            let other_core = world.cores.core_at_mut(other_slot);
            take_damage(other_core, force, other_move_restrictions);
        }
        {
            let mut other_char = world.characters[other_id as usize].unwrap();
            let mut other_core = *world.cores.core_at(other_slot);
            unfreeze(&mut other_char, &mut other_core);
            world.characters[other_id as usize] = Some(other_char);
            *world.cores.core_at_mut(other_slot) = other_core;
        }
        hits += 1;
    }
    targets.clear();
    world.hammer_scratch = targets;
    if hits > 0 {
        let fire_delay = world.cores.core_at(slot).tuning.hammer_hit_fire_delay_ms();
        let mut character = world.characters[id as usize].unwrap();
        character.reload_timer = (fire_delay * core::SERVER_TICK_SPEED as f32 / 1000.0) as i32;
        world.characters[id as usize] = Some(character);
    }
}

/// `CCharacter::HandleJetpack()` (`character.cpp:253-294`). `jetpack_ticks == 0` for this entire
/// corpus (no map/scenario ever sets `m_Core.m_Jetpack`), so this is dead code in practice for
/// Stage A's corpus, but implemented in full since it's self-contained and cheap.
pub fn handle_jetpack<R: Real>(world: &mut World<R>, id: i32, slot: usize) {
    let core = *world.cores.core_at(slot);
    if core.active_weapon < 0 {
        return;
    }
    let character = world.characters[id as usize].unwrap();
    let direction = vmath::normalize(Vec2::new(
        R::from_i32(character.latest_input.target_x),
        R::from_i32(character.latest_input.target_y),
    ));

    let mut full_auto = false;
    if core.active_weapon == WEAPON_GRENADE
        || core.active_weapon == WEAPON_SHOTGUN
        || core.active_weapon == WEAPON_LASER
    {
        full_auto = true;
    }
    if core.jetpack && core.active_weapon == WEAPON_GUN {
        full_auto = true;
    }

    let will_fire = count_input_presses(character.latest_prev_input.fire, character.latest_input.fire) != 0
        || (full_auto && (character.latest_input.fire & 1) != 0 && core.weapons[core.active_weapon as usize].ammo != 0);
    if !will_fire {
        return;
    }
    if core.weapons[core.active_weapon as usize].ammo == 0 || character.freeze_time != 0 {
        return;
    }
    if core.active_weapon == WEAPON_GUN && core.jetpack {
        let strength = core.tuning.jetpack_strength::<R>();
        let force = direction * R::from_i32(-1) * (strength / R::from_i32(100) / R::from_f64(6.11));
        take_damage(world.cores.core_at_mut(slot), force, character.move_restrictions);
    }
}

/// `CCharacter::HandleWeapons()` (`character.cpp:658-676`), minus the pain-sound timer
/// (cosmetic).
pub fn handle_weapons<R: Real>(world: &mut World<R>, id: i32, slot: usize) {
    handle_ninja(world, id, slot);
    handle_jetpack(world, id, slot);
    let reload = world.characters[id as usize].unwrap().reload_timer;
    if reload != 0 {
        world.characters[id as usize].as_mut().unwrap().reload_timer -= 1;
        return;
    }
    fire_weapon(world, id, slot);
}

// --- `CPlayer` spawn/respawn/kill (`player.cpp`), `CCharacter::Spawn`/`DDRaceInit`
// (`character.cpp:70-145,2487-2537`) --------------------------------------------------------

/// `CCharacter::DDRaceInit()` (`character.cpp:2487-2537`), minus team-locked start/DDRaceState
/// inheritance (dead — team-locking never reachable) and the `sv_team == SV_TEAM_MANDATORY`
/// chat warning.
fn ddrace_init<R: Real>(core: &mut CharacterCore<R>, sv_endless_drag: bool, sv_hit: bool) {
    core.endless_hook = sv_endless_drag;
    core.hammer_hit_disabled = !sv_hit;
    core.shotgun_hit_disabled = !sv_hit;
    core.grenade_hit_disabled = !sv_hit;
    core.laser_hit_disabled = !sv_hit;
    core.jumps = 2;
}

/// `CCharacter::Spawn(CPlayer*, vec2)` + `IGameController::OnCharacterSpawn` (`character.cpp:70-
/// 145`, `gamecontroller.cpp:512-521`), minus everything Stage A's scope excludes (rescue,
/// recording, zone messages, saved-team loading — the last unreachable in this harness, see this
/// module's doc comment).
pub fn spawn_character<R: Real>(world: &mut World<R>, id: i32, pos: Vec2<R>) {
    let mut character = Character::<R> {
        spawn_tick: world.tick,
        pos,
        prev_pos: pos,
        ..Default::default()
    };

    let mut core = CharacterCore::<R>::default();
    core.reset();
    core.id = id;
    core.active_weapon = WEAPON_GUN;
    core.pos = pos;
    let tune_zone = world.collision.is_tune(world.collision.get_map_index(pos));
    core.tuning = *world.tuning.zone(tune_zone);

    world.cores.insert(id as u8, core);
    world.entity_order.insert(0, id as u8);
    character.alive = true;
    if let Some(player) = world.players[id as usize].as_mut() {
        player.has_character = true;
        // `CPlayer::ForceSpawn`/`TryRespawn` (`player.cpp:737,825`): `m_Team = TEAM_GAME` —
        // needed so `player_early_input`'s `Team != TEAM_SPECTATORS` guard (and
        // `CanSpawn`'s/`EvaluateSpawnType`'s own) reads correctly after the *first* spawn.
        player.team = TEAM_GAME;
    }

    let forced_solo = sv_team_forced_solo(&world.config);
    let changed = world.race_teams.on_character_spawn(
        &mut world.teams_core,
        &mut world.cores.switchers,
        &mut world.characters,
        id,
        forced_solo,
    );
    apply_team_change_side_effects(world, id, changed);
    character.health = 10;
    give_weapon(&mut character, &mut core, world.tick, WEAPON_HAMMER, false);
    give_weapon(&mut character, &mut core, world.tick, WEAPON_GUN, false);

    ddrace_init(&mut core, world.config.sv_endless_drag, world.config.sv_hit);
    character.tune_zone = tune_zone;
    character.tune_zone_old = -1;

    world.characters[id as usize] = Some(character);
    let slot = world.cores.slot_of(id as u8).unwrap();
    *world.cores.core_at_mut(slot) = core;
}

/// `World`-level wrapper around [`give_weapon`], for callers (test/replay harnesses) that only
/// have a client id, not an extracted `(Character, CharacterCore)` pair.
pub fn give_weapon_to<R: Real>(world: &mut World<R>, id: i32, weapon: i32) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    let mut character = world.characters[id as usize].unwrap();
    let mut core = *world.cores.core_at(slot);
    give_weapon(&mut character, &mut core, world.tick, weapon, false);
    world.characters[id as usize] = Some(character);
    *world.cores.core_at_mut(slot) = core;
}

/// `World`-level wrapper around [`RaceTeams::set_force_character_team`] — a scenario's per-
/// character `team` field (v3, `docs/formats.md` §12.6) is applied this way, directly after the
/// character's initial spawn (`oracle_server.cpp`'s character-creation loop; see
/// `tests/parity_oracle_b.rs`).
pub fn set_force_character_team<R: Real>(world: &mut World<R>, id: i32, team: i32) {
    let forced_solo = sv_team_forced_solo(&world.config);
    let World {
        race_teams,
        teams_core,
        cores,
        characters,
        ..
    } = world;
    let changed =
        race_teams.set_force_character_team(teams_core, &mut cores.switchers, characters, id, team, forced_solo);
    apply_team_change_side_effects(world, id, changed);
}

/// `IGameController::CanSpawn`/`CPlayer::TryRespawn` (`player.cpp:816-832`). Returns whether it
/// succeeded (a spawn point was found).
pub fn try_respawn<R: Real>(world: &mut World<R>, id: i32) -> bool {
    let Some(pos) = world.can_spawn(id) else { return false };
    if let Some(player) = world.players[id as usize].as_mut() {
        player.weak_hook_spawn = false;
        player.spawning = false;
    }
    spawn_character(world, id, pos);
    // `m_ViewPos = SpawnPos` (`player.cpp:826`) and `CPlayer::Tick` recomputes `m_TuneZone` from it
    // later in the same call (`player.cpp:278-280`): the player's zone is the spawn zone at once.
    let spawn_zone = world.collision.is_tune(world.collision.get_map_index(pos));
    if let Some(player) = world.players[id as usize].as_mut() {
        player.tune_zone = spawn_zone;
    }
    if sv_team_forced_solo(&world.config) {
        world.teams_core.set_solo(id, true);
        if let Some(slot) = world.cores.slot_of(id as u8) {
            world.cores.core_at_mut(slot).solo = true;
        }
    }
    true
}

/// `CPlayer::KillCharacter(int Weapon, bool SendKillMsg)` (`player.cpp:712-721`): `Die()`, then
/// (unlike a normal in-world death) *immediately* clears [`Player::has_character`] — see that
/// field's doc comment.
pub fn kill_character<R: Real>(world: &mut World<R>, id: i32, weapon: i32) {
    if world.players[id as usize].is_some_and(|p| p.has_character) {
        die(world, id, id, weapon);
        world.players[id as usize].as_mut().unwrap().has_character = false;
    }
}

/// `CPlayer::Respawn(bool WeakHook)` (`player.cpp:723-730`): request a respawn (actually spawning
/// happens on a later `player_tick`/`try_respawn` call). `weak_hook`: always `false` from
/// [`on_kill_net_message`] (matches the plain `Respawn()` call there, `gamecontext.cpp:2985`).
pub fn request_respawn<R: Real>(world: &mut World<R>, id: i32, weak_hook: bool) {
    if let Some(player) = world.players[id as usize].as_mut() {
        player.weak_hook_spawn = weak_hook;
        player.spawning = true;
    }
}

/// `CGameContext::OnKillNetMessage` (`gamecontext.cpp:2955-2986`), minus vote/pause checks
/// (unreachable — no votes/pauses in this harness) and chat. Applies the scenario's per-tick
/// `kill` bit exactly like the real server applies a client's `CNetMsg_Cl_Kill`
/// (`docs/formats.md` §11/§12.3, review round 2 finding F12): `sv_kill_delay` and
/// `sv_kill_protection` still gate it.
pub fn on_kill_net_message<R: Real>(world: &mut World<R>, id: i32) {
    let Some(player) = world.players[id as usize] else {
        return;
    };
    if player.last_kill != 0 && player.last_kill + world.config.sv_kill_delay * core::SERVER_TICK_SPEED > world.tick {
        return;
    }
    if !player.has_character || !world.characters[id as usize].is_some_and(|c| c.alive) {
        return;
    }
    let start_time = world.characters[id as usize].unwrap().start_time;
    let curr_time = (world.tick - start_time) / core::SERVER_TICK_SPEED;
    let ddrace_state = world.characters[id as usize].unwrap().ddrace_state;
    if world.config.sv_kill_protection != 0 && curr_time >= 60 * world.config.sv_kill_protection && ddrace_state == 1 {
        return;
    }
    world.players[id as usize].as_mut().unwrap().last_kill = world.tick;
    kill_character(world, id, WEAPON_SELF);
    request_respawn(world, id, false);
}

/// `CPlayer::OnPredictedEarlyInput`'s spawn-on-fire side effect (`player.cpp:678-691`, the
/// `m_Spawning = true` line only — the rest of that function is cosmetic/prediction plumbing out
/// of this crate's scope). Called for every player, in ascending client id order, *before* the
/// world tick and *before* the tick counter itself increments (`docs/formats.md` §12.1's
/// `OnClientPredictedEarlyInput` step).
pub fn player_early_input<R: Real>(world: &mut World<R>, id: i32, fire_bit: i32) {
    let Some(player) = world.players[id as usize].as_mut() else {
        return;
    };
    let has_character = player.has_character;
    if !has_character && player.team != TEAM_SPECTATORS && (fire_bit & 1) != 0 {
        player.spawning = true;
    }
}

/// `CPlayer::Tick()` (`player.cpp:186-299`), minus everything cosmetic/network (latency, AFK,
/// tune-zone messages, Halloween emote) — only the respawn-timing state machine remains. Called
/// for every player, in ascending client id order, *after* the world tick this same server tick
/// (`docs/formats.md` §12.1) — see [`Player::has_character`]'s doc comment for why that ordering
/// is exactly what makes a normal death's respawn wait at least one tick while a kill-bit death's
/// respawn can land the same tick.
pub fn player_tick<R: Real>(world: &mut World<R>, id: i32) {
    let Some(player) = world.players[id as usize] else {
        return;
    };
    // `if(m_pCharacter->IsAlive()) { ...; m_ViewPos = m_pCharacter->m_Pos; }` and, at the end of
    // `CPlayer::Tick`, `m_TuneZone = IsTune(GetMapIndex(m_ViewPos))` (`player.cpp:253-258,278-280`):
    // the zone only changes when the view position does, so it is refreshed exactly there.
    if player.has_character
        && let Some(c) = world.characters[id as usize].filter(|c| c.alive)
    {
        let zone = world.collision.is_tune(world.collision.get_map_index(c.pos));
        world.players[id as usize].as_mut().unwrap().tune_zone = zone;
    }
    let earliest_respawn_tick = player.previous_die_tick + 3 * core::SERVER_TICK_SPEED;
    let respawn_tick = player.die_tick.max(earliest_respawn_tick) + 2;
    if !player.has_character && respawn_tick <= world.tick {
        world.players[id as usize].as_mut().unwrap().spawning = true;
    }

    if player.has_character {
        let alive = world.characters[id as usize].is_some_and(|c| c.alive);
        if !alive {
            world.players[id as usize].as_mut().unwrap().has_character = false;
        }
    } else {
        let player = world.players[id as usize].unwrap();
        if player.spawning && !player.weak_hook_spawn {
            try_respawn(world, id);
        }
    }
}

// --- `CGameWorld::IntersectCharacter`/`FindEntities` (`gameworld.cpp:58-77,292-295,297-332`),
// as far as `CProjectile`/`CPickup` need them. -------------------------------------------------

/// `CCharacterCore::PhysicalSize()`/`CEntity::m_ProximityRadius` for a `CCharacter` — the raw
/// value (`28.0`), *not* halved (`character.h`'s constructor passes it as-is to `CEntity`).
fn character_proximity_radius<R: Real>() -> R {
    core::physical_size::<R>()
}

/// `(client_id, core)` pairs in real DDNet's actual `ENTTYPE_CHARACTER` entity-list order —
/// newest-spawned first, matching [`World::entity_order`] exactly (see that field's doc
/// comment). This is the order `IntersectEntity`/`FindEntities`'s own internal walk
/// (`FindFirst(ENTTYPE_CHARACTER); pC; pC = pC->TypeNext()`, `gameworld.cpp:58-77,297-332`) and
/// `IGameController::EvaluateSpawnPos`/`EvaluateSpawnType`'s occupancy scan
/// (`gamecontroller.cpp:89-99,127-141`) actually visit characters in — *not*
/// [`WorldCore::iter`]'s ascending-slot/client-id order, a storage-layout artifact with no
/// counterpart in the C++ source. This matters for: (a) exact-distance ties in
/// `IntersectEntity`'s strict `<` "closer so far" comparison (`gameworld.cpp:315`) — which
/// specific character wins a tie depends on find-order; (b) `EvaluateSpawnPos`'s `Score +=
/// 1/distance` accumulation (`gamecontroller.cpp:89-99`) — floating-point addition isn't
/// associative, so a different visiting order can produce a bit-different sum even summing the
/// exact same set of terms. An earlier revision of this port used `WorldCore::iter()`
/// (ascending order) for both — an empirically found bug (see this crate's `BUILD REPORT`).
fn characters_in_entity_order<R: Real>(world: &World<R>) -> impl Iterator<Item = (i32, Vec2<R>, &CharacterCore<R>)> {
    world.entity_order.iter().map(move |&id| {
        let slot = world.cores.slot_of(id).expect("entity_order id must have a live core");
        let pos = world.characters[id as usize]
            .as_ref()
            .expect("entity_order id must have a character")
            .pos;
        (id as i32, pos, world.cores.core_at(slot))
    })
}

/// Task 1.10b, speed-up 2 (the "same bbox early-out" applied to [`intersect_character`], which
/// never touches `Collision`/solid tiles at all, so a solid-tile summed-area table doesn't apply
/// to it — this is [`could_be_in_range`]'s idea one level up instead, a point against a *segment*
/// rather than a point against another point): could `p` be within `bound` of *any* point on the
/// segment `[seg0, seg1]`? Every point on that segment lies within its own axis-aligned bounding
/// box (`[min(seg0, seg1), max(seg0, seg1)]` — a convex combination never leaves that range), and
/// distance from a point to a point *inside* a set is always `>=` distance from that point to the
/// set itself, so `p`'s distance to the segment's bbox lower-bounds its distance to the segment's
/// own closest point — same monotonic-rounding argument [`could_be_in_range`]'s doc comment makes,
/// applied to a bbox instead of a single point.
fn could_be_near_segment<R: Real>(seg0: Vec2<R>, seg1: Vec2<R>, p: Vec2<R>, bound: R) -> bool {
    let limit = bound + R::from_f64(RANGE_PREFILTER_SLACK);
    let min_x = seg0.x.min(seg1.x);
    let max_x = seg0.x.max(seg1.x);
    let min_y = seg0.y.min(seg1.y);
    let max_y = seg0.y.max(seg1.y);
    p.x >= min_x - limit && p.x <= max_x + limit && p.y >= min_y - limit && p.y <= max_y + limit
}

/// `CGameWorld::IntersectCharacter` (`gameworld.cpp:292-295`, delegating to `IntersectEntity`,
/// `gameworld.cpp:297-332`, `Type == ENTTYPE_CHARACTER`): the alive character whose position is
/// closest to `pos0` among those within `radius` of the segment `(pos0, pos1)`, excluding
/// `exclude_id`, and passing `collide_with`'s `CanCollide` check (`-1` = no check). Returns
/// `(target_id, intersect_pos)`.
pub fn intersect_character<R: Real>(
    world: &World<R>,
    pos0: Vec2<R>,
    pos1: Vec2<R>,
    radius: R,
    exclude_id: i32,
    collide_with: i32,
) -> Option<(i32, Vec2<R>)> {
    intersect_character_ex(world, pos0, pos1, radius, exclude_id, collide_with, -1)
}

/// [`intersect_character`] with `IntersectEntity`'s `pThisOnly` parameter (`gameworld.cpp:308`):
/// when `this_only != -1`, only that character can be found. `exclude_id`/`collide_with`/
/// `this_only` use `-1` for "none", exactly like the C++ `nullptr`/`-1`.
pub fn intersect_character_ex<R: Real>(
    world: &World<R>,
    pos0: Vec2<R>,
    pos1: Vec2<R>,
    radius: R,
    exclude_id: i32,
    collide_with: i32,
    this_only: i32,
) -> Option<(i32, Vec2<R>)> {
    let mut closest_len = vmath::distance(pos0, pos1) * R::from_i32(100);
    let mut result: Option<(i32, Vec2<R>)> = None;
    let bound = character_proximity_radius::<R>() + radius;
    for (cid, core_pos, _) in characters_in_entity_order(world) {
        if cid == exclude_id {
            continue;
        }
        if this_only != -1 && cid != this_only {
            continue;
        }
        if !world.characters[cid as usize].is_some_and(|c| c.alive) {
            continue;
        }
        if collide_with != -1 && !character_can_collide(&world.teams_core, cid, collide_with) {
            continue;
        }
        // Task 1.10b, speed-up 2: skip the (pricier) closest-point projection entirely for a
        // character nowhere near the segment's own bounding box.
        if !could_be_near_segment(pos0, pos1, core_pos, bound) {
            continue;
        }
        let Some(intersect_pos) = vmath::closest_point_on_line(pos0, pos1, core_pos) else {
            continue;
        };
        let len = vmath::distance(core_pos, intersect_pos);
        if len < bound {
            let len = vmath::distance(pos0, intersect_pos);
            if len < closest_len {
                closest_len = len;
                result = Some((cid, intersect_pos));
            }
        }
    }
    result
}

/// Slack (in map pixels) [`could_be_in_range`]'s cheap pre-filter adds on top of the exact search
/// bound. At this crate's coordinate scale (map extents in the thousands to tens of thousands of
/// pixels; `f32`'s ULP there is well under 0.01 px) this is enormously more slack than any
/// conceivable floating-point rounding difference between a Chebyshev (max-abs-coordinate) bound
/// and the true Euclidean `sqrt`-based one — see [`could_be_in_range`]'s doc comment for why the
/// filter is exact (not approximate) regardless, and this task's `BUILD REPORT` for the full
/// argument. Chosen for a comfortable, legible margin, not tightness.
const RANGE_PREFILTER_SLACK: f64 = 8.0;

/// A cheap, always-safe pre-filter for "could `a` be within `bound` of `b`?" (task 1.10
/// acceptance criterion 2's "early exits only where it is provable that the result is identical",
/// applied to the character-range scans below rather than `MoveBox`/`IntersectLine`): `distance(a,
/// b) = sqrt(dx*dx + dy*dy) >= max(|dx|, |dy|)` always holds, for any correctly-rounded `+`/`*`/
/// `sqrt` (`dx*dx`/`dy*dy` are both `>= 0`, so their correctly-rounded sum can only round to a
/// value `>= dx*dx` — rounding is monotonic — and correctly-rounded `sqrt` is monotonic too, so
/// `sqrt(dx*dx+dy*dy) >= sqrt(dx*dx) ~= |dx|`, off by at most the double-rounding of squaring
/// `dx` itself — far below [`RANGE_PREFILTER_SLACK`]'s margin at this crate's coordinate scale).
/// So whenever `|dx| > bound + slack` or `|dy| > bound + slack`, `distance(a, b) < bound` is
/// certain to be `false` — this can return `false` (skip the expensive exact check) with no
/// possible effect on any comparison's result, while `true` only means "run the real check",
/// never "is in range".
fn could_be_in_range<R: Real>(a: Vec2<R>, b: Vec2<R>, bound: R) -> bool {
    let limit = bound + R::from_f64(RANGE_PREFILTER_SLACK);
    (a.x - b.x).abs() <= limit && (a.y - b.y).abs() <= limit
}

/// Task 1.10b, speed-up 1: the union bounding box (min, max corner) of every *alive* character's
/// position, in entity order — `None` when there are no alive characters at all (in which case
/// nothing this tick can possibly be near anything: every caller should skip its own range scan
/// unconditionally). Meant to be computed once per pickup-tick pass (`World::world_tick`'s pickup
/// phase), not once per pickup — see `could_be_near_alive_characters_bbox`'s (private) doc comment for
/// why recomputing it per pickup would defeat the entire point (`O(pickups)` calls to something
/// itself `O(characters)` is right back to `O(pickups * characters)`, the cost being cut here).
pub fn alive_characters_bbox<R: Real>(world: &World<R>) -> Option<(Vec2<R>, Vec2<R>)> {
    characters_in_entity_order(world)
        .filter(|&(cid, _, _)| world.characters[cid as usize].is_some_and(|c| c.alive))
        .fold(None, |acc, (_, pos, _)| match acc {
            None => Some((pos, pos)),
            Some((min, max)) => Some((
                Vec2::new(min.x.min(pos.x), min.y.min(pos.y)),
                Vec2::new(max.x.max(pos.x), max.y.max(pos.y)),
            )),
        })
}

/// A cheap, always-safe pre-filter for "could any alive character be within `reach` of `pos`?" —
/// the same idea as [`could_be_in_range`] applied once per *pickup* against the whole character
/// roster's bounding box, instead of once per (pickup, character) pair: if `pos` is more than
/// `reach` outside `bbox` on any side, then for every character `c` actually inside `bbox`
/// (`bbox` is an exact min/max reduction over real positions, no rounding involved in building
/// it), `distance(c, pos) >= max(|dx|, |dy|) > reach` — the same monotonic-rounding argument
/// [`could_be_in_range`]'s doc comment makes, one level up. `bbox = None` (no alive characters at
/// all) always returns `false`.
fn could_be_near_alive_characters_bbox<R: Real>(bbox: Option<(Vec2<R>, Vec2<R>)>, pos: Vec2<R>, reach: R) -> bool {
    let Some((min, max)) = bbox else { return false };
    pos.x >= min.x - reach && pos.x <= max.x + reach && pos.y >= min.y - reach && pos.y <= max.y + reach
}

/// `CGameWorld::FindEntities` (`gameworld.cpp:58-77`), `Type == ENTTYPE_CHARACTER`: every alive
/// character within `radius + pEnt->m_ProximityRadius` of `pos`, in the entity-list order
/// [`intersect_character`]'s sibling scan would use — but here, unlike `IntersectCharacter`,
/// order genuinely doesn't matter to any Stage A caller (hammer/ninja/spawn-eval hit everyone in
/// range, order-independently), so this returns them in ascending client-id order for simplicity.
pub fn find_characters_in_range<R: Real>(world: &World<R>, pos: Vec2<R>, radius: R) -> Vec<i32> {
    let bound = radius + character_proximity_radius::<R>();
    characters_in_entity_order(world)
        .filter(|&(cid, core_pos, _)| {
            world.characters[cid as usize].is_some_and(|c| c.alive)
                && could_be_in_range(core_pos, pos, bound)
                && vmath::distance(core_pos, pos) < bound
        })
        .map(|(cid, _, _)| cid)
        .collect()
}

/// [`find_characters_in_range`], writing into `out` (cleared first) instead of returning a
/// freshly allocated `Vec` — zero heap allocations in steady state when `out` is a reused
/// scratch buffer whose capacity has already grown to fit (acceptance criterion 1, review round
/// 1 finding F9: this is the version every call site inside `World::step`'s own hot path now
/// uses; the allocating [`find_characters_in_range`] above stays for any caller — none, inside
/// this crate — that doesn't have a reusable buffer of its own to pass in).
///
/// Task 1.10: [`could_be_in_range`]'s cheap Chebyshev pre-filter runs before the exact
/// `sqrt`-based [`vmath::distance`] check — at real map scale (pickups/projectiles scattered over
/// a map thousands of pixels wide, only 2-8 characters clustered in one area) most pairs fail it
/// and never touch `dot`/`sqrt` at all, which is where the bulk of `pickup_tick`'s/
/// `projectile_tick`'s per-tick cost was going (measured: task's `BUILD REPORT`).
fn find_characters_in_range_into<R: Real>(world: &World<R>, pos: Vec2<R>, radius: R, out: &mut Vec<i32>) {
    let bound = radius + character_proximity_radius::<R>();
    out.clear();
    out.extend(characters_in_entity_order(world).filter_map(|(cid, core_pos, _)| {
        (world.characters[cid as usize].is_some_and(|c| c.alive)
            && could_be_in_range(core_pos, pos, bound)
            && vmath::distance(core_pos, pos) < bound)
            .then_some(cid)
    }));
}

// --- `CProjectile` (`entities/projectile.cpp:15-253`) -------------------------------------------

/// `CalcPos(vec2 Pos, vec2 Velocity, float Curvature, float Speed, float Time)` (`gamecore.h:77-
/// 84`) — always plain `f32` in the C++ source; ported generically over `R` since every caller
/// here already works in `R`, but every intermediate stays consistent with `R`'s own precision
/// (this crate's `f32` instantiation is bit-exact either way; only `f64` would ever differ from
/// literally re-deriving the C++ `float` expression, and `f64` has no parity requirement — see
/// the task spec).
fn calc_pos<R: Real>(pos: Vec2<R>, velocity: Vec2<R>, curvature: R, speed: R, time: R) -> Vec2<R> {
    let t = time * speed;
    Vec2::new(
        pos.x + velocity.x * t,
        pos.y + velocity.y * t + curvature / R::from_i32(10000) * (t * t),
    )
}

/// `CProjectile::GetPos(float Time)` (`projectile.cpp:61-86`).
fn projectile_get_pos<R: Real>(p: &Projectile<R>, tuning: &TuningParams, time: R) -> Vec2<R> {
    let (curvature, speed) = match p.weapon_type {
        WEAPON_GRENADE => (tuning.grenade_curvature::<R>(), tuning.grenade_speed::<R>()),
        WEAPON_SHOTGUN => (tuning.shotgun_curvature::<R>(), tuning.shotgun_speed::<R>()),
        WEAPON_GUN => (tuning.gun_curvature::<R>(), tuning.gun_speed::<R>()),
        _ => (R::ZERO, R::ZERO),
    };
    calc_pos(p.pos, p.direction, curvature, speed, time)
}

/// `CProjectile::Tick()` (`projectile.cpp:88-253`), minus sound/damage-indicator/telegun-visual
/// cosmetics kept only insofar as they change physical state (the telegun teleport itself is
/// kept — it moves `pOwnerChar`). `BUG_GRENADE_DOUBLEEXPLOSION` is never emulated (`EmulateBug`
/// always `false` — see this crate's `BUILD REPORT` for why none of this corpus's maps match any
/// entry in DDNet's hardcoded `CMapBugs` table). `g_Config.m_SvDestroyBulletsOnDeath` is read
/// from [`ServerConfig::sv_destroy_bullets_on_death`] (default `true`).
pub fn projectile_tick<R: Real>(world: &mut World<R>, index: usize) {
    let p = world.projectiles[index];
    let tick = world.tick;
    let tuning = *world.tuning.zone(p.tune_zone);

    let pt = R::from_i32(tick - p.start_tick - 1) / R::from_i32(core::SERVER_TICK_SPEED);
    let ct = R::from_i32(tick - p.start_tick) / R::from_i32(core::SERVER_TICK_SPEED);
    let prev_pos = projectile_get_pos(&p, &tuning, pt);
    let cur_pos = projectile_get_pos(&p, &tuning, ct);
    let hit = world.collision.intersect_line(prev_pos, cur_pos);
    let collide = hit.hit != 0;
    let mut col_pos = hit.collision;
    let new_pos = hit.before_collision;

    let owner_alive = p.owner >= 0 && world.characters[p.owner as usize].is_some_and(|c| c.alive);
    let owner_grenade_hit_disabled = if p.owner >= 0 {
        world.cores.get(p.owner as u8).map(|c| c.grenade_hit_disabled)
    } else {
        None
    };
    let hit_check = owner_grenade_hit_disabled.map(|d| !d).unwrap_or(world.config.sv_hit);

    let target = if hit_check {
        intersect_character(
            world,
            prev_pos,
            col_pos,
            if p.freeze { R::ONE } else { R::from_i32(6) },
            p.owner,
            p.owner,
        )
    } else {
        None
    };
    // `IntersectCharacter(PrevPos, ColPos, ..., ColPos, ...)` (`projectile.cpp:105`) passes
    // `ColPos` as both the segment's end point *and* its own `NewPos` out-parameter, so a found
    // target overwrites `ColPos` with the hit point (`gameworld.cpp:323`'s `NewPos =
    // IntersectPos` inside `IntersectEntity`) — every later use of `ColPos` this tick
    // (`CreateExplosion`, `GameLayerClipped` at `projectile.cpp:142,160`) sees that hit point,
    // not the original wall-collision position. [`intersect_character`]'s returned
    // `intersect_pos` is exactly that `IntersectPos`/hit point, computed but never written back
    // — so mirror the overwrite explicitly here.
    if let Some((_, intersect_pos)) = target {
        col_pos = intersect_pos;
    }

    let mut new_life_span = p.life_span;
    if new_life_span > -1 {
        new_life_span -= 1;
    }

    // `IsWeaponCollide` (`projectile.cpp:112-120`): `pTargetChr`/`pOwnerChar` being non-null
    // already implies both are alive (`GetPlayerChar`/`IntersectCharacter` only ever return
    // alive characters), so the C++ source's redundant `IsAlive()` checks are omitted here.
    let mut is_weapon_collide = false;
    if owner_alive
        && let Some((target_id, _)) = target
        && !character_can_collide(&world.teams_core, target_id, p.owner)
    {
        is_weapon_collide = true;
    }
    // `else if(m_Owner >= 0 && (m_Type != WEAPON_GRENADE || SvDestroyBulletsOnDeath ||
    // BelongsToPracticeTeam))` (`projectile.cpp:121-129`) — this whole `else` only runs when
    // `pOwnerChar` is null, i.e. `!owner_alive` (covers both `m_Owner == -1` and "owner died");
    // `BelongsToPracticeTeam` is always `false` here (see this function's doc comment), but
    // `SvDestroyBulletsOnDeath` defaults to `true` (`config_variables.h:296` — an earlier
    // revision of this port hardcoded it `false`, with a doc comment that incorrectly claimed
    // the default was `0`), so at the default config *every* weapon type is destroyed once its
    // owner is gone, not just non-grenades.
    let marked_for_destroy =
        !owner_alive && p.owner >= 0 && (p.weapon_type != WEAPON_GRENADE || world.config.sv_destroy_bullets_on_death);

    if !marked_for_destroy {
        let target_exists = target.is_some();
        let will_act = (target_exists && hit_check) || collide || game_layer_clipped(&world.collision, cur_pos);
        if will_act && !is_weapon_collide {
            if p.explosive && (!target_exists || (!p.freeze || (p.weapon_type == WEAPON_SHOTGUN && collide))) {
                create_explosion(
                    world,
                    col_pos,
                    p.owner,
                    p.weapon_type,
                    p.owner == -1,
                    target.map(|(t, _)| character_team(&world.teams_core, t)).unwrap_or(-1),
                );
            } else if p.freeze {
                let mut targets = std::mem::take(&mut world.range_scratch);
                find_characters_in_range_into(world, cur_pos, R::ONE, &mut targets);
                for &target_id in &targets {
                    let ok = p.layer != Layer::Switch
                        || (p.layer == Layer::Switch
                            && p.number > 0
                            && (p.number as usize) < world.cores.switchers.len()
                            && world.cores.switchers[p.number as usize].status
                                [character_team(&world.teams_core, target_id) as usize]);
                    if ok {
                        let slot = world.cores.slot_of(target_id as u8).unwrap();
                        let tick = world.tick;
                        let sv_freeze_delay = world.config.sv_freeze_delay;
                        let mut tc = world.characters[target_id as usize].unwrap();
                        let mut tcore = *world.cores.core_at(slot);
                        freeze_default(&mut tc, &mut tcore, tick, sv_freeze_delay);
                        world.characters[target_id as usize] = Some(tc);
                        *world.cores.core_at_mut(slot) = tcore;
                    }
                }
                targets.clear();
                world.range_scratch = targets;
            } else if let Some((target_id, _)) = target {
                let slot = world.cores.slot_of(target_id as u8).unwrap();
                let move_restrictions = world.characters[target_id as usize].unwrap().move_restrictions;
                take_damage(world.cores.core_at_mut(slot), Vec2::zero(), move_restrictions);
            }

            // Telegun (`projectile.cpp:160-199`) — position/velocity effect only.
            if p.owner >= 0 && !game_layer_clipped(&world.collision, col_pos) {
                let has_telegun = world.cores.get(p.owner as u8).is_some_and(|c| {
                    (p.weapon_type == WEAPON_GRENADE && c.has_telegun_grenade)
                        || (p.weapon_type == WEAPON_GUN && c.has_telegun_gun)
                });
                if has_telegun {
                    // `pTargetChr ? pTargetChr->m_Pos : ColPos` (`projectile.cpp:163,188`) — the
                    // target character's own *current* position, not the segment/target
                    // intersect point `ColPos` was just overwritten with above (those are
                    // generally different points: the intersect point is the closest point on
                    // the projectile's travel segment to the target, not the target's own `m_Pos`).
                    let target_pos = target
                        .and_then(|(t, _)| world.cores.get(t as u8).map(|c| c.pos))
                        .unwrap_or(col_pos);
                    let map_index = world.collision.get_pure_map_index_vec(target_pos);
                    let front_index = world.collision.get_front_tile_index(map_index as i32);
                    let is_switch_tele =
                        world.collision.get_switch_type(map_index as i32) == map::TILE_ALLOW_TELE_GUN as i32;
                    let is_blue_switch_tele =
                        world.collision.get_switch_type(map_index as i32) == map::TILE_ALLOW_BLUE_TELE_GUN as i32;
                    let mut is_switch_tele = is_switch_tele;
                    let mut is_blue_switch_tele = is_blue_switch_tele;
                    if is_switch_tele || is_blue_switch_tele {
                        let delay = world.collision.get_switch_delay(map_index as i32);
                        if delay == 1 && p.weapon_type != WEAPON_GUN {
                            is_switch_tele = false;
                            is_blue_switch_tele = false;
                        }
                        if delay == 2 && p.weapon_type != WEAPON_GRENADE {
                            is_switch_tele = false;
                            is_blue_switch_tele = false;
                        }
                    }
                    let front_is_tele = front_index == map::TILE_ALLOW_TELE_GUN as i32
                        || front_index == map::TILE_ALLOW_BLUE_TELE_GUN as i32;
                    if front_is_tele || is_switch_tele || is_blue_switch_tele || target.is_some() {
                        // `!Collide -> GetNearestAirPosPlayer(target_pos, ...)`; `Collide ->
                        // GetNearestAirPos(NewPos, CurPos, ...)` (`projectile.cpp:187-190`).
                        let found_pos = if !collide {
                            get_nearest_air_pos_player(&world.collision, target_pos)
                        } else {
                            get_nearest_air_pos(&world.collision, new_pos, cur_pos)
                        };
                        if let Some(possible_pos) = found_pos {
                            let owner_slot = world.cores.slot_of(p.owner as u8).unwrap();
                            let mut owner_char = world.characters[p.owner as usize].unwrap();
                            owner_char.tele_gun_pos = possible_pos;
                            owner_char.tele_gun_teleport = true;
                            owner_char.is_blue_tele_gun_teleport =
                                front_index == map::TILE_ALLOW_BLUE_TELE_GUN as i32 || is_blue_switch_tele;
                            world.characters[p.owner as usize] = Some(owner_char);
                            let _ = owner_slot;
                        }
                    }
                }
            }

            // `projectile.cpp:201-227`'s own `if`/`else if`/`else { if }` chain, kept as separate
            // branches even though the `WEAPON_GUN` and `!freeze` bodies coincide here: the real
            // `WEAPON_GUN` branch also calls `CreateDamageInd` first (network-only, no compared
            // field) — a `WEAPON_GUN` projectile always destroys on collision, while any other
            // weapon only does when *not* frozen.
            #[allow(clippy::if_same_then_else)]
            if collide && p.bouncing != 0 {
                let mut p2 = p;
                p2.start_tick = world.tick;
                p2.pos = new_pos - p.direction * R::from_i32(4);
                if p2.bouncing == 1 {
                    p2.direction.x = -p2.direction.x;
                } else if p2.bouncing == 2 {
                    p2.direction.y = -p2.direction.y;
                }
                if p2.direction.x.abs() < R::from_f64(1e-6) {
                    p2.direction.x = R::ZERO;
                }
                if p2.direction.y.abs() < R::from_f64(1e-6) {
                    p2.direction.y = R::ZERO;
                }
                p2.pos += p2.direction;
                p2.life_span = new_life_span;
                world.projectiles[index] = p2;
                return;
            } else if p.weapon_type == WEAPON_GUN {
                world.projectiles[index].marked_for_destroy = true;
                return;
            } else if !p.freeze {
                world.projectiles[index].marked_for_destroy = true;
                return;
            }
        }
    } else {
        world.projectiles[index].marked_for_destroy = true;
        return;
    }

    if new_life_span == -1 {
        if p.explosive {
            create_explosion(
                world,
                col_pos,
                p.owner,
                p.weapon_type,
                p.owner == -1,
                if owner_alive {
                    character_team(&world.teams_core, p.owner)
                } else {
                    -1
                },
            );
        }
        world.projectiles[index].marked_for_destroy = true;
        return;
    }

    let x = world.collision.get_index_along(prev_pos, cur_pos);
    let z = if world.config.sv_old_teleport_weapons {
        world.collision.is_teleport(x)
    } else {
        world.collision.is_teleport_weapon(x)
    };
    let mut result = p;
    result.life_span = new_life_span;
    if z != 0 && !world.collision.tele_outs((z - 1) as u8).is_empty() {
        let outs = world.collision.tele_outs((z - 1) as u8);
        let out = world.cores.random_or_0(outs.len() as i32);
        result.pos = outs[out as usize];
        result.start_tick = world.tick;
    }
    world.projectiles[index] = result;
}

/// `CEntity::GetNearestAirPos(vec2 Pos, vec2 PrevPos, vec2 *pOutPos)` (`entity.cpp:59-80`) — used
/// by the telegun when the projectile *collided* (`projectile.cpp:190`, `GetNearestAirPos(NewPos,
/// CurPos, ...)`, note the swapped argument names relative to this function's own parameter
/// names: the caller's `NewPos` is this function's `Pos`, `CurPos` is `PrevPos`).
fn get_nearest_air_pos<R: Real>(collision: &Collision<R>, mut pos: Vec2<R>, prev_pos: Vec2<R>) -> Option<Vec2<R>> {
    for _ in 0..16 {
        if !collision.check_point_vec(pos) {
            break;
        }
        pos = pos - vmath::normalize(prev_pos - pos);
    }
    let round_x = vmath::round_to_int(pos.x);
    let round_y = vmath::round_to_int(pos.y);
    let pos_in_block = Vec2::new(R::from_i32(round_x % 32), R::from_i32(round_y % 32));
    let block_center = Vec2::new(R::from_i32(round_x), R::from_i32(round_y)) - pos_in_block
        + Vec2::new(R::from_i32(16), R::from_i32(16));
    let off = |v: R| if v < R::from_i32(16) { R::from_i32(-2) } else { R::ONE };

    let candidate = Vec2::new(block_center.x + off(pos_in_block.x), pos.y);
    if !collision.test_box(candidate, core::physical_size_vec2()) {
        return Some(candidate);
    }
    let candidate = Vec2::new(pos.x, block_center.y + off(pos_in_block.y));
    if !collision.test_box(candidate, core::physical_size_vec2()) {
        return Some(candidate);
    }
    let candidate = Vec2::new(
        block_center.x + off(pos_in_block.x),
        block_center.y + off(pos_in_block.y),
    );
    if !collision.test_box(candidate, core::physical_size_vec2()) {
        Some(candidate)
    } else {
        None
    }
}

/// `CEntity::GetNearestAirPosPlayer(vec2 PlayerPos, vec2 *pOutPos)` (`entity.cpp:82-93`) — used
/// by the telegun when the projectile did *not* collide (`projectile.cpp:188`,
/// `GetNearestAirPosPlayer(pTargetChr ? pTargetChr->m_Pos : ColPos, ...)`).
fn get_nearest_air_pos_player<R: Real>(collision: &Collision<R>, player_pos: Vec2<R>) -> Option<Vec2<R>> {
    for distance in (-1..=5).rev() {
        let candidate = Vec2::new(player_pos.x, player_pos.y - R::from_i32(distance));
        if !collision.test_box(candidate, core::physical_size_vec2()) {
            return Some(candidate);
        }
    }
    None
}

/// `CGameContext::CreateExplosion` (`gamecontext.cpp:372-428`), minus the network event and
/// per-team damage-dedup mask (`sv_hit`-disabled/`NoDamage` fan-out — never exercised in this
/// corpus: every explosive Stage A weapon here is a live `pOwnerChar` firing with `sv_hit`
/// enabled, the common case, which always takes the *first* branch of that dedup — `NoDamage`
/// requires `Owner == -1`, i.e. a fixture-spawned explosive, which none of the crazy-shotgun
/// fixtures are (`explosive` only on the `_EX` variant, itself never given `NoDamage` semantics
/// since it always has a "no owner" `-1`... see the module's `BUILD REPORT` for the exact
/// coverage argument). `activated_team`: `-1` unless `owner == -1`.
fn create_explosion<R: Real>(
    world: &mut World<R>,
    pos: Vec2<R>,
    owner: i32,
    weapon: i32,
    no_damage: bool,
    activated_team: i32,
) {
    let radius = R::from_i32(135);
    let inner_radius = R::from_i32(48);
    let owner_grenade_hit_disabled = if owner >= 0 {
        world.cores.get(owner as u8).map(|c| c.grenade_hit_disabled)
    } else {
        None
    };
    // `m_apPlayers[Owner]->m_TuneZone` — the *player's* zone ([`Player::tune_zone`]), not the
    // character's (`gamecontext.cpp:393-396`).
    let owner_tune_zone = if owner >= 0 {
        world.players[owner as usize].map(|p| p.tune_zone)
    } else {
        None
    };
    let strength = if owner == -1 || owner_tune_zone.is_none_or(|z| z == 0) {
        world.tuning.zone(0).explosion_strength::<R>()
    } else {
        world.tuning.zone(owner_tune_zone.unwrap()).explosion_strength::<R>()
    };

    // A fixed-size array, not a `HashSet` (review round 1, finding F9: zero heap allocations in
    // steady state) — `player_team` is always `< team_super()` by the time it's indexed here
    // (the `team_super()` case `continue`s just above), so this is always in bounds.
    let mut team_seen = [false; NUM_DDRACE_TEAMS as usize];
    let mut targets = std::mem::take(&mut world.range_scratch);
    find_characters_in_range_into(world, pos, radius, &mut targets);
    for &target_id in &targets {
        let target_pos = world.characters[target_id as usize]
            .expect("in-range characters exist")
            .pos;
        let diff = target_pos - pos;
        let l = vmath::length(diff);
        let force_dir = if l != R::ZERO {
            vmath::normalize(diff)
        } else {
            Vec2::new(R::ZERO, R::ONE)
        };
        let l = R::ONE - ((l - inner_radius) / (radius - inner_radius)).clamp(R::ZERO, R::ONE);
        let dmg = strength * l;
        if dmg.to_i32_trunc() == 0 {
            continue;
        }
        let hit_check =
            owner_grenade_hit_disabled.map(|d| !d).unwrap_or(world.config.sv_hit) || no_damage || owner == target_id;
        if !hit_check {
            continue;
        }
        if owner != -1 {
            let target_alive = world.characters[target_id as usize].is_some_and(|c| c.alive);
            if target_alive && !character_can_collide(&world.teams_core, target_id, owner) {
                continue;
            }
        }
        if owner == -1 && activated_team != -1 {
            let target_alive = world.characters[target_id as usize].is_some_and(|c| c.alive);
            let target_team = character_team(&world.teams_core, target_id);
            if target_alive && target_team != activated_team {
                continue;
            }
        }
        let player_team = character_team(&world.teams_core, target_id);
        let grenade_hit_disabled_for_dedup = owner_grenade_hit_disabled.unwrap_or(!world.config.sv_hit);
        if grenade_hit_disabled_for_dedup || no_damage {
            if player_team == world.teams_core.team_super() {
                continue;
            }
            if team_seen[player_team as usize] {
                continue;
            }
            team_seen[player_team as usize] = true;
        }
        let force = force_dir * dmg * R::from_i32(2);
        let slot = world.cores.slot_of(target_id as u8).unwrap();
        let move_restrictions = world.characters[target_id as usize].unwrap().move_restrictions;
        take_damage(world.cores.core_at_mut(slot), force, move_restrictions);
    }
    targets.clear();
    world.range_scratch = targets;
    let _ = weapon;
}

// --- `CPickup::Tick()` (`entities/pickup.cpp:35-162`) -------------------------------------------

/// `CPickup::Tick()`, including `Move()` (`pickup.cpp:35-37,191-198`: on a tick divisible by
/// `(int)(TickSpeed * 0.15f) == 7`, `MoverSpeed` may update `m_Core`, which is *always* then
/// added to `m_Pos` — see [`Pickup::mcore`]'s doc comment for the persistence semantics), minus
/// every `CreateSound`/`SendWeaponPickup` (cosmetic).
///
/// `characters_bbox`: [`alive_characters_bbox`]'s result, computed *once* for this whole
/// pickup-tick pass by the caller (`World::world_tick`'s pickup phase) rather than once per
/// pickup — task 1.10b, speed-up 1. When this pickup's own position can't possibly be within
/// reach of any alive character (`could_be_near_alive_characters_bbox`), the entire range scan
/// (and its `range_scratch` take/give-back) is skipped outright — measured effect: this task's
/// `BUILD REPORT`.
pub fn pickup_tick<R: Real>(world: &mut World<R>, index: usize, characters_bbox: Option<(Vec2<R>, Vec2<R>)>) {
    if world.tick % 7 == 0 {
        let mut p = world.pickups[index];
        if let Some((tile, speed)) = world
            .collision
            .mover_speed(p.pos.x.to_i32_trunc(), p.pos.y.to_i32_trunc())
        {
            let _ = tile;
            p.mcore = speed;
        }
        p.pos += p.mcore;
        world.pickups[index] = p;
    }
    let pickup = world.pickups[index];
    let reach = R::from_f64(PICKUP_PROXIMITY_RADIUS as f64)
        + character_proximity_radius::<R>()
        + R::from_f64(RANGE_PREFILTER_SLACK);
    if !could_be_near_alive_characters_bbox(characters_bbox, pickup.pos, reach) {
        return;
    }
    let mut targets = std::mem::take(&mut world.range_scratch);
    find_characters_in_range_into(
        world,
        pickup.pos,
        R::from_f64(PICKUP_PROXIMITY_RADIUS as f64),
        &mut targets,
    );
    for &target_id in &targets {
        let team = character_team(&world.teams_core, target_id);
        if pickup.layer == Layer::Switch
            && pickup.number > 0
            && !(world
                .cores
                .switchers
                .get(pickup.number as usize)
                .is_some_and(|s| s.status[team as usize]))
        {
            continue;
        }
        let slot = world.cores.slot_of(target_id as u8).unwrap();
        match pickup.kind {
            PickupKind::Freeze => {
                let tick = world.tick;
                let sv_freeze_delay = world.config.sv_freeze_delay;
                let mut c = world.characters[target_id as usize].unwrap();
                let mut core = *world.cores.core_at(slot);
                freeze_default(&mut c, &mut core, tick, sv_freeze_delay);
                world.characters[target_id as usize] = Some(c);
                *world.cores.core_at_mut(slot) = core;
            }
            PickupKind::Armor => {
                if team == world.teams_core.team_super() {
                    continue;
                }
                let core = world.cores.core_at_mut(slot);
                let mut stripped_any = false;
                // `for(int j = WEAPON_SHOTGUN; j < NUM_WEAPONS; j++)` (`pickup.cpp:62`) — an
                // earlier revision of this port stopped one short (`NUM_WEAPONS - 1`), so an
                // armor pickup never stripped an active ninja pickup's `got`/ammo state (nor
                // counted it toward `Sound`/`SetLastWeapon(WEAPON_GUN)` below). The unconditional
                // `core.active_weapon >= WEAPON_SHOTGUN` check further down still forced
                // `WEAPON_HAMMER` either way (ninja is the highest-numbered weapon), so this only
                // affected `weapons[WEAPON_NINJA].got`/`ammo` and `last_weapon`, not
                // `active_weapon` itself.
                for w in WEAPON_SHOTGUN..NUM_WEAPONS as i32 {
                    if core.weapons[w as usize].got {
                        core.weapons[w as usize].got = false;
                        core.weapons[w as usize].ammo = 0;
                        stripped_any = true;
                    }
                }
                core.ninja.activation_dir = Vec2::zero();
                core.ninja.activation_tick = -500;
                core.ninja.current_move_time = 0;
                // `pickup.cpp:73-79`: `SetLastWeapon(WEAPON_GUN)` only when `Sound` (i.e. at
                // least one weapon was actually stripped) — matching `Sound`'s own guard exactly
                // (not e.g. unconditionally, and not merely "armor pickup touched", found
                // empirically: several traces showed `last_weapon` staying at its spawn default
                // in this crate while the reference had already flipped to `WEAPON_GUN` after an
                // armor pickup stripped a bonus shotgun/grenade/laser this character never
                // otherwise touched via `SetWeapon`).
                if stripped_any {
                    let mut character = world.characters[target_id as usize].unwrap();
                    character.last_weapon = WEAPON_GUN;
                    world.characters[target_id as usize] = Some(character);
                }
                let core = world.cores.core_at_mut(slot);
                if core.active_weapon >= WEAPON_SHOTGUN {
                    core.active_weapon = WEAPON_HAMMER;
                }
            }
            PickupKind::ArmorShotgun | PickupKind::ArmorGrenade | PickupKind::ArmorLaser => {
                if team == world.teams_core.team_super() {
                    continue;
                }
                let w = match pickup.kind {
                    PickupKind::ArmorShotgun => WEAPON_SHOTGUN,
                    PickupKind::ArmorGrenade => WEAPON_GRENADE,
                    _ => WEAPON_LASER,
                };
                let core = world.cores.core_at_mut(slot);
                if core.weapons[w as usize].got {
                    core.weapons[w as usize].got = false;
                    core.weapons[w as usize].ammo = 0;
                    // `pickup.cpp:83-108`: `SetLastWeapon(WEAPON_GUN)` inside the same `if
                    // (GetWeaponGot(...))` block that strips the weapon — see the sibling
                    // `PickupKind::Armor` arm's comment for the empirical finding this fixes.
                    let mut character = world.characters[target_id as usize].unwrap();
                    character.last_weapon = WEAPON_GUN;
                    world.characters[target_id as usize] = Some(character);
                }
                let core = world.cores.core_at_mut(slot);
                if core.active_weapon == w {
                    core.active_weapon = WEAPON_HAMMER;
                }
            }
            PickupKind::ArmorNinja => {
                if team == world.teams_core.team_super() {
                    continue;
                }
                let core = world.cores.core_at_mut(slot);
                core.ninja.activation_dir = Vec2::zero();
                core.ninja.activation_tick = -500;
                core.ninja.current_move_time = 0;
            }
            PickupKind::Weapon(w) => {
                let core = world.cores.core_at(slot);
                if !core.weapons[w as usize].got || core.weapons[w as usize].ammo != -1 {
                    let tick = world.tick;
                    let mut c = world.characters[target_id as usize].unwrap();
                    let mut core = *world.cores.core_at(slot);
                    give_weapon(&mut c, &mut core, tick, w, false);
                    world.characters[target_id as usize] = Some(c);
                    *world.cores.core_at_mut(slot) = core;
                }
            }
            PickupKind::Ninja => {
                let tick = world.tick;
                let mut c = world.characters[target_id as usize].unwrap();
                let mut core = *world.cores.core_at(slot);
                give_ninja(&mut c, &mut core, tick);
                world.characters[target_id as usize] = Some(c);
                *world.cores.core_at_mut(slot) = core;
            }
        }
    }
    targets.clear();
    world.range_scratch = targets;
}

// --- `World::step`: the server's per-tick sequence, tying every piece above together
// (`docs/formats.md` §12.1, `engine/server/server.cpp:3547-3597`, `gamecontext.cpp:1200-1249`
// (`OnTick`), `gameworld.cpp:202-263` (`CGameWorld::Tick`)). ------------------------------------

/// One character's resolved input for one tick, plus the scenario's `kill` bit
/// (`docs/formats.md` §11/§12.3, finding F12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickInput {
    /// Client id.
    pub id: u8,
    /// The already-resolved (`ddai_trace::scenario::resolve_input`, or equivalent) input to
    /// apply this tick.
    pub input: PlayerInput,
    /// Whether this tick's input requests a kill (`OnKillNetMessage`, applied before anything
    /// else this tick — see [`World::step`]).
    pub kill: bool,
}

impl<R: Real> World<R> {
    /// Runs exactly one server tick: kill-bit application, early input, the tick counter
    /// increment, direct input (weapon switch/fire), the world tick (`CGameWorld::Tick`,
    /// including the `no_weak_hook` `PreTick` variant), every player's respawn bookkeeping, and
    /// the switch-expiry pass — in that exact order (`docs/formats.md` §12.1). `inputs` **must**
    /// be sorted by ascending client id (matching the real server's `for(c = 0; c < MAX_CLIENTS;
    /// c++)` client loop) — this is a caller contract, not re-sorted here, to keep this on the
    /// zero-heap-allocation hot path (acceptance criterion 1).
    ///
    /// `no_weak_hook` is *not* a parameter here: real DDNet reads `g_Config.m_SvNoWeakHook`
    /// directly wherever it's needed (`gameworld.cpp:214`, `character.cpp:815,820`), a single
    /// `CFGFLAG_GAME` variable, not something a caller passes in per tick. An earlier revision of
    /// this port took it as an argument, and every one of this crate's own test/harness call
    /// sites passed a hardcoded `false` — silently ignoring whatever a scenario's own cfg had set
    /// `sv_no_weak_hook` to, so the `no_weak_hook` code path never actually ran against the real
    /// corpus. `World::world_tick` now reads `self.config.sv_no_weak_hook` itself.
    ///
    /// # Panics
    ///
    /// In debug builds, if `inputs` isn't sorted by ascending `id`.
    pub fn step(&mut self, inputs: &[TickInput]) {
        self.step_bounded(inputs, MAX_CLIENTS);
    }

    /// [`World::step`] for a caller that knows every `Some` entry of [`World::players`] has an index below `id_hi`:
    /// the per-player pass (step 6) then visits only that prefix instead of all 128 slots (task 4.13: the scan was
    /// about a twentieth of a rollout tick). Identical to [`World::step`] when the bound holds; with
    /// `id_hi = MAX_CLIENTS` it is [`World::step`].
    pub fn step_bounded(&mut self, inputs: &[TickInput], id_hi: usize) {
        let id_hi = id_hi.min(MAX_CLIENTS);
        debug_assert!(
            self.players[id_hi..].iter().all(Option::is_none),
            "step_bounded: a player at or above id_hi = {id_hi}"
        );
        debug_assert!(
            inputs.windows(2).all(|w| w[0].id < w[1].id),
            "World::step: inputs must be sorted by ascending client id"
        );

        // 0. `CEntity::m_Pos` of every living character is whatever the previous tick's
        //    `TickDeferred` left it at, i.e. its core position — unless the core was moved from
        //    outside `step` (a test harness or a live-state injection), which this re-sync
        //    absorbs. See [`Character::pos`].
        for slot in 0..self.cores.len() {
            let id = self.cores.id_at(slot) as usize;
            if let Some(c) = self.characters[id].as_mut() {
                c.pos = self.cores.core_at(slot).pos;
            }
        }

        // 1. Kill bit, applied exactly like a real client's `CNetMsg_Cl_Kill`
        //    (`OnKillNetMessage`), before anything else this tick.
        for ti in inputs {
            if ti.kill {
                on_kill_net_message(self, ti.id as i32);
            }
        }

        // 2. `OnClientPredictedEarlyInput` -> `CPlayer::OnPredictedEarlyInput` ->
        //    `CCharacter::OnDirectInput` (`player.cpp:678-691`, `character.cpp:744-764`): the
        //    fire-while-dead spawn trigger, *and* weapon switch/fire (reads/updates
        //    `m_LatestInput`/`m_LatestPrevInput`) — both still *before* the tick counter
        //    increments. This is the phase that actually calls `HandleWeaponSwitch`/`FireWeapon`
        //    — *not* `OnPredictedInput` below, despite the name (see this function's citations;
        //    an earlier revision of this port had the two swapped).
        for ti in inputs {
            let id = ti.id as i32;
            player_early_input(self, id, ti.input.fire);
            on_direct_input(self, id, ti.input);
        }

        // 3. Tick counter increment (`server.cpp:3547-3597`'s `m_CurrentGameTick++`, between
        //    early and direct input).
        self.tick += 1;

        // 4. `OnClientPredictedInput` -> `CPlayer::OnPredictedInput` -> `CCharacter::
        //    OnPredictedInput` (`player.cpp:634-650`, `character.cpp:728-742`): *only* copies
        //    the input into `m_Input`/`m_SavedInput` (no weapon logic) — `m_SavedInput` is what
        //    this same tick's `DDRaceTick` (called from the world tick, step 5 below) reads.
        for ti in inputs {
            let id = ti.id as i32;
            if let Some(character) = self.characters[id as usize].as_mut() {
                on_predicted_input(character, ti.input);
            }
        }

        // 5. The world tick (`CGameWorld::Tick`, `gameworld.cpp:202-263`).
        self.world_tick(self.config.sv_no_weak_hook);

        // 6. Every player's own `Tick()` (respawn bookkeeping), ascending client id
        //    (`gamecontext.cpp:1227-1249`), *after* the world tick. Real DDNet runs this for
        //    *every* connected player (`for(int i = 0; i < MAX_CLIENTS; i++) if(m_apPlayers[i])
        //    m_apPlayers[i]->Tick();`, `gamecontext.cpp:1227-1231`) — not only the ones with a
        //    `TickInput` entry this tick. An earlier revision of this port iterated `inputs`
        //    instead, so a player who sent no input this tick (an idle spectator-turned-player,
        //    or simply every id `step`'s caller didn't bother building an entry for) never got
        //    its own respawn bookkeeping run and could never respawn.
        for id in 0..id_hi {
            if self.players[id].is_some() {
                player_tick(self, id as i32);
            }
        }

        // 7. Switch-timer expiry (`gamecontext.cpp:1460-1476`), after every player's `Tick()`.
        let tick = self.tick;
        phase_time!(
            switch_expiry,
            switch::tick_switch_expiry(&mut self.cores.switchers, &mut self.active_timed_switchers, tick)
        );
    }

    /// `CGameWorld::Tick()` (`gameworld.cpp:202-263`): projectiles, the `ENTTYPE_LASER` list
    /// (lasers, beams, turret shots, then draggers/turrets/lights), pickups, characters, the
    /// `TickDeferred()` pass, `RemoveEntities()` and the strong/weak ids. (`ENTTYPE_FLAG` is
    /// never populated in DDRace.)
    fn world_tick(&mut self, no_weak_hook: bool) {
        // Entity-type order: `ENTTYPE_PROJECTILE` (0), `ENTTYPE_LASER` (1 — every `CDragger`/
        // `CGun`/`CLight`/`CDoor`/`CDraggerBeam`/`CLaser` shares this slot), `ENTTYPE_PICKUP`
        // (2), `ENTTYPE_CHARACTER` (4) — `gameworld.h:24-30`.
        phase_time!(projectiles, {
            for i in 0..self.projectiles.len() {
                if !self.projectiles[i].marked_for_destroy {
                    projectile_tick(self, i);
                }
            }
        });
        phase_time!(fixtures, {
            // `ENTTYPE_LASER`: the dynamic entities newest first (index `len - 1` is the list
            // head — see [`World::lasers`]), then the static fixtures in list order. Entities
            // pushed during the pass (dragger beams, turret shots) go to the head, i.e. *behind*
            // the traversal cursor, exactly like `m_pNextTraverseEntity` keeps the C++ pass from
            // visiting them this tick, so iterating the starting length is enough.
            for i in (0..self.lasers.len()).rev() {
                match self.lasers[i] {
                    LaserSlot::Laser(_) => laser::laser_tick(self, i),
                    LaserSlot::Beam(_) => fixtures::beam_tick(self, i),
                    LaserSlot::Plasma(_) => fixtures::plasma_tick(self, i),
                }
            }
            // No character moves during this pass, so one bounding box serves every fixture.
            let characters_bbox = if self.fixtures.is_empty() {
                None
            } else {
                alive_characters_bbox(self)
            };
            // A dragger only acts on the movers' tick (`Tick % 7 == 0`); a turret needs either that
            // tick (it may drift) or a possible target; a light always hits (but cannot hit anyone
            // without an alive character). Hoisting those tests keeps a map with ~90 idle fixtures
            // down to a few comparisons per fixture per tick.
            let mover_tick = self.tick % MOVER_PERIOD == 0;
            let guns_may_fire = self.config.sv_plasma_per_sec > 0 && characters_bbox.is_some();
            for i in 0..self.fixtures.len() {
                match self.fixtures[i] {
                    Fixture::Dragger(_) => {
                        if mover_tick {
                            fixtures::dragger_tick(self, i, characters_bbox);
                        }
                    }
                    Fixture::Gun(_) => {
                        if mover_tick || guns_may_fire {
                            fixtures::gun_tick(self, i, characters_bbox);
                        }
                    }
                    Fixture::Light(_) => {
                        if mover_tick || characters_bbox.is_some() {
                            fixtures::light_tick(self, i);
                        }
                    }
                }
            }
        });
        phase_time!(pickups, {
            // Task 1.10b, speed-up 1: computed once for the whole pass, not once per pickup.
            let characters_bbox = alive_characters_bbox(self);
            if self.tick % 7 == 0 {
                // Movers drift on these ticks: every pickup runs its own `Move()` first.
                for i in 0..self.pickups.len() {
                    pickup_tick(self, i, characters_bbox);
                }
            } else if characters_bbox.is_some() {
                // Task 3.6: on the other six ticks `pickup_tick` returns at once for a pickup no
                // character can be near; test that here from the position alone (hoisting the reach)
                // instead of copying the pickup and calling. Same pickups, same order.
                let reach = R::from_f64(PICKUP_PROXIMITY_RADIUS as f64)
                    + character_proximity_radius::<R>()
                    + R::from_f64(RANGE_PREFILTER_SLACK);
                for i in 0..self.pickups.len() {
                    if could_be_near_alive_characters_bbox(characters_bbox, self.pickups[i].pos, reach) {
                        pickup_tick(self, i, characters_bbox);
                    }
                }
            }
        });

        // Both loops below reset then check `team_changed_this_pass` around each of their own
        // passes (not once for both): real DDNet's `m_pNextTraverseEntity` clobber
        // (`World::team_changed_this_pass`'s doc comment) only ever truncates the *one*
        // `for(;pEnt;)` sub-loop that was mid-iteration when the nested `Die()` fired: the
        // `SvNoWeakHook` `PreTick()` pre-loop (`gameworld.cpp:216-222`) and the regular per-type
        // `Tick()` loop (`gameworld.cpp:225-231`) each start fresh from `m_apFirstEntityTypes[i]`,
        // so a truncation in one never carries into the other.
        if no_weak_hook {
            self.team_changed_this_pass = false;
            phase_time!(
                character_pre_tick_pass,
                self.for_each_in_entity_order(|world, _i, id| {
                    // `m_Core.Tick(true, !g_Config.m_SvNoWeakHook)` (`character.cpp:815`): under
                    // `sv_no_weak_hook`, `PreTick()`'s own core tick must *not* run the deferred
                    // (hook-drag) body — `character_tick`'s own `no_weak_hook` branch runs it
                    // separately via `core::tick_deferred`. Passing `true` here (an earlier bug)
                    // applied hook forces twice.
                    character_pre_tick(world, id as i32, false);
                    !world.team_changed_this_pass
                })
            );
        }
        self.team_changed_this_pass = false;
        phase_time!(
            character_tick_pass,
            self.for_each_in_entity_order(|world, _i, id| {
                character_tick(world, id as i32, no_weak_hook);
                !world.team_changed_this_pass
            })
        );

        // `TickDeferred()`, a separate pass over every entity type (`gameworld.cpp:234-240`) —
        // only characters do anything here in Stage A's scope (`CProjectile`/`CPickup` don't
        // override `TickDeferred`, inheriting `CEntity`'s no-op).
        phase_time!(
            character_deferred_pass,
            self.for_each_in_entity_order(|world, _i, id| {
                character_tick_deferred(world, id as i32);
                true
            })
        );

        // `RemoveEntities()` (`gameworld.cpp:186-200`): destroy projectiles marked this tick.
        phase_time!(retain, {
            self.projectiles.retain(|p| !p.marked_for_destroy);
            if !self.lasers.is_empty() {
                self.lasers.retain(|l| !l.marked_for_destroy());
            }
        });

        // `m_StrongWeakId` assignment (`gameworld.cpp:256-262`): entity-list order, head first.
        phase_time!(
            strong_weak_id_pass,
            self.for_each_in_entity_order(|world, i, id| {
                if let Some(c) = world.characters[id as usize].as_mut() {
                    c.strong_weak_id = i as i32;
                }
                true
            })
        );
    }

    /// Runs `f(self, index, id)` for each id currently in [`World::entity_order`], over a
    /// *stable* snapshot taken via the reusable [`World::entity_order_scratch`] buffer instead
    /// of `entity_order.clone()` — zero heap allocations in steady state (the scratch buffer's
    /// capacity, once grown to fit the character count, is never shrunk, only `clear()`ed;
    /// acceptance criterion 1, review round 1 finding F9). Safe to call while `f` mutates
    /// `self`, including `self.entity_order` itself (e.g. a mid-loop death, per
    /// [`World::team_changed_this_pass`]'s own doc comment) — the snapshot `f` iterates doesn't
    /// change underfoot. `f` returns whether to keep going; returning `false` stops the pass
    /// early (mirroring the two `world_tick` call sites that break on
    /// `self.team_changed_this_pass`).
    fn for_each_in_entity_order(&mut self, mut f: impl FnMut(&mut Self, usize, u8) -> bool) {
        let mut order = std::mem::take(&mut self.entity_order_scratch);
        order.clear();
        order.extend_from_slice(&self.entity_order);
        for (i, &id) in order.iter().enumerate() {
            if !f(self, i, id) {
                break;
            }
        }
        order.clear();
        self.entity_order_scratch = order;
    }
}

/// `CCharacter::PreTick()` (`character.cpp:790-816`), minus the "You died of old age" guard
/// (`m_StartTime > Server()->Tick()` — see this crate's `BUILD REPORT` for why this can never
/// trigger for a freshly-spawned `m_StartTime == 0`… `Server()->Tick()` starting at `1`, and
/// every subsequent write to `m_StartTime` this crate ports only ever sets a value `<=` the
/// current tick) and the emote-reset (cosmetic). `do_deferred_tick`: `!SvNoWeakHook` — see
/// [`character_tick`]'s doc comment for why this is a separate function from it (so the
/// `no_weak_hook` pre-pass can call *only* this half).
pub fn character_pre_tick<R: Real>(world: &mut World<R>, id: i32, do_deferred_tick: bool) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    {
        // Task 3.6: in place (character and core are separate fields of the world), not on copies
        // of the 300 B character and 488 B core written back afterwards.
        let character = world.characters[id as usize].as_mut().unwrap();
        let core = world.cores.core_at_mut(slot);
        ddrace_tick(character, core, &world.collision, &world.tuning);
        // `m_Core.m_Input = m_Input;` (`character.cpp:814`, right before `m_Core.Tick(...)`).
        core.input = character.input;
    }
    core::tick(
        &mut world.cores,
        slot,
        &world.collision,
        &world.teams_core,
        true,
        do_deferred_tick,
    );
}

/// `CCharacter::Tick()` (`character.cpp:818-855`), minus antibot hook-attach notifications
/// (cosmetic). Runs [`character_pre_tick`] (or, under `no_weak_hook`, only the deferred-hook
/// core tick — `PreTick` itself already ran in `World::world_tick`'s separate pre-pass),
/// [`handle_weapons`], [`ddrace_post_core_tick`], then the tail input/position bookkeeping
/// (`m_PrevInput`/`m_PrevPos`).
pub fn character_tick<R: Real>(world: &mut World<R>, id: i32, no_weak_hook: bool) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    if no_weak_hook {
        core::tick_deferred(&mut world.cores, slot, &world.teams_core);
    } else {
        character_pre_tick(world, id, true);
    }

    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    handle_weapons(world, id, slot);
    if !world.characters[id as usize].is_some_and(|c| c.alive) {
        return;
    }
    if !ddrace_post_core_tick(world, id) {
        return;
    }

    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    let pos = world.cores.core_at(slot).pos;
    let character = world.characters[id as usize].as_mut().unwrap();
    character.prev_input = character.input;
    character.prev_pos = pos;
}

/// `CCharacter::TickDeferred()` (`character.cpp:857-971`), minus the dead-reckoning/`m_SendCore`
/// bookkeeping and stuck-detection logging (both cosmetic/debug — no compared field). The
/// physical effect is exactly `m_Core.Move()` + `m_Core.Quantize()`.
pub fn character_tick_deferred<R: Real>(world: &mut World<R>, id: i32) {
    let Some(slot) = world.cores.slot_of(id as u8) else {
        return;
    };
    core::move_character(&mut world.cores, slot, &world.collision, &world.teams_core);
    core::quantize(world.cores.core_at_mut(slot));
    // `m_Pos = m_Core.m_Pos;` (`character.cpp:880`)
    let pos = world.cores.core_at(slot).pos;
    if let Some(c) = world.characters[id as usize].as_mut() {
        c.pos = pos;
    }
}

/// `CreateAllEntities`'s 5 map-wide global tile effects (`gamecontext.cpp:4288-4312`), checked
/// against a raw game/front tile index (before subtracting `ENTITY_OFFSET`).
fn scan_global_tile(raw: u8, config: &mut ServerConfig, tuning: &mut TuningList) {
    if raw == map::TILE_NPC {
        tuning.zone_mut(0).set_by_name("player_collision", 0.0);
        // Only zone 0 in the C++ source (`GlobalTuning()->Set(...)`) — *not* every zone (unlike
        // `sv_solo_server`'s equivalent effect); characters in another tune zone are unaffected.
    } else if raw == map::TILE_EHOOK {
        config.sv_endless_drag = true;
    } else if raw == map::TILE_NOHIT {
        config.sv_hit = false;
    } else if raw == map::TILE_NPH {
        tuning.zone_mut(0).set_by_name("player_hooking", 0.0);
    }
    // `TILE_OLDLASER` (`sv_old_laser`) has no effect on anything Stage A models (laser is Stage
    // B) — recognized implicitly by simply not matching any arm above (not an unknown-command
    // error path; this is a tile scan, not a console command).
}

/// `SIntConfigVariable::CommandCallback` (`config.cpp:39-64`): clamps into `[min, max]`
/// (`if(Value < Min) Value = Min; if(Max != 0 && Value > Max) Value = Max;` — none of this
/// crate's modeled variables have `max == 0`, so the "unbounded above" special case that skips
/// the upper clamp never applies here). Real DDNet never rejects an in-range-*format* numeric
/// value for being out of the variable's own range — it silently clamps (found empirically:
/// `sv_hit 2` becomes `1`, not a rejected/unrecognized command — an earlier revision of this
/// port rejected any out-of-`{0,1}` value for a bool-backed field, and stored an out-of-range
/// int-backed field's raw value unclamped).
fn set_int(field: &mut i32, rest: &[&str], line: &str, min: i32, max: i32) -> Result<(), UnknownCommand> {
    if rest.len() != 1 {
        return Err(UnknownCommand { line: line.to_string() });
    }
    let value: i32 = rest[0].parse().map_err(|_| UnknownCommand { line: line.to_string() })?;
    *field = value.clamp(min, max);
    Ok(())
}

#[cfg(test)]
mod command_tests {
    use super::*;

    #[test]
    fn tuning_list_baseline_matches_every_zone() {
        let list = TuningList::reset_to_baseline();
        for &zone in &[0, 1, 255] {
            assert_eq!(list.zone(zone).get_by_name("gun_speed"), Some(1400.0));
            assert_eq!(list.zone(zone).get_by_name("shotgun_speeddiff"), Some(0.0));
        }
    }

    #[test]
    fn zero_player_collision_and_hooking_everywhere_hits_every_zone() {
        let mut list = TuningList::reset_to_baseline();
        list.zero_player_collision_and_hooking_everywhere();
        assert_eq!(list.zone(0).get_by_name("player_collision"), Some(0.0));
        assert_eq!(list.zone(200).get_by_name("player_hooking"), Some(0.0));
    }

    #[test]
    fn apply_command_recognizes_sv_solo_server() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_solo_server 1").unwrap();
        assert!(config.sv_solo_server);
        assert!(log.is_empty());
    }

    /// `SIntConfigVariable::CommandCallback` (`config.cpp:39-64`): an out-of-range but
    /// well-formed numeric value clamps into `[min, max]` rather than being rejected — matching
    /// real DDNet (`sv_hit 2` -> `1`, `sv_freeze_delay 0` -> `1` since its `min` is `1`, not
    /// `0`).
    #[test]
    fn apply_command_clamps_out_of_range_numeric_values_instead_of_rejecting() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit 2").unwrap();
        assert!(config.sv_hit, "sv_hit 2 must clamp to 1 (true), not error or store 2");
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit -3").unwrap();
        assert!(!config.sv_hit, "sv_hit -3 must clamp to 0 (false)");
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_freeze_delay 0").unwrap();
        assert_eq!(
            config.sv_freeze_delay, 1,
            "sv_freeze_delay's own min is 1, not 0 (config_variables.h:544)"
        );
        apply_command(
            &mut config,
            &mut tuning,
            &mut switchers,
            &mut log,
            "sv_freeze_delay 999",
        )
        .unwrap();
        assert_eq!(config.sv_freeze_delay, 30, "sv_freeze_delay's own max is 30");
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_team 99").unwrap();
        assert_eq!(config.sv_team, 3, "sv_team's own max is 3");
        // Still a genuine syntax error for non-numeric/missing arguments (clamping only applies
        // to a value that *did* parse).
        assert!(apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit maybe").is_err());
        assert!(log.is_empty(), "none of the above are locked-write warnings");
    }

    /// Unlocked (`game_settings_locked == false`, the default): a `CFGFLAG_GAME` variable like
    /// `sv_hit` sets normally, exactly like any other recognized command, and a malformed line
    /// (missing/non-bool argument) is still a syntax error.
    #[test]
    fn apply_command_sv_hit_sets_normally_when_unlocked_but_still_validates_syntax() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        assert!(config.sv_hit);
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit 0").unwrap();
        assert!(!config.sv_hit);
        assert!(log.is_empty());
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit 1").unwrap();
        assert!(config.sv_hit);
        let err = apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit maybe").unwrap_err();
        assert_eq!(err.line, "sv_hit maybe");
        let err = apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit").unwrap_err();
        assert_eq!(err.line, "sv_hit");
    }

    /// Locked (`game_settings_locked == true`, mirroring post-`OnInit()`): a `CFGFLAG_GAME`
    /// write is rejected — the field is untouched, a warning lands in `command_log`, and this
    /// is `Ok(())`, not an [`UnknownCommand`] error — while `sv_kill_delay` (`CFGFLAG_SERVER`
    /// only) and `tune`/`switch_open` (plain commands, never gated by this lock at all — see
    /// [`is_cfgflag_game`]'s doc comment) still apply.
    #[test]
    fn apply_command_rejects_cfgflag_game_writes_once_locked() {
        let mut config = ServerConfig {
            game_settings_locked: true,
            ..ServerConfig::default()
        };
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers = vec![Switcher::default()];
        let mut log = Vec::new();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_hit 0").unwrap();
        assert!(config.sv_hit, "a locked sv_hit write must be rejected, not applied");
        assert_eq!(log.len(), 1);
        assert!(
            log[0].contains("sv_hit"),
            "warning should name the rejected variable: {log:?}"
        );
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "sv_kill_delay 5").unwrap();
        assert_eq!(config.sv_kill_delay, 5, "CFGFLAG_SERVER-only vars are never locked");
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "tune gravity 0").unwrap();
        assert_eq!(
            tuning.zone(0).get_by_name("gravity"),
            Some(0.0),
            "plain commands are never locked"
        );
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "switch_open 0").unwrap();
        assert!(!switchers[0].initial, "switch_open is never locked");
        assert_eq!(log.len(), 1, "only the sv_hit write should have produced a warning");
    }

    #[test]
    fn apply_command_rejects_unknown_command() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        let err = apply_command(
            &mut config,
            &mut tuning,
            &mut switchers,
            &mut log,
            "sv_totally_made_up 1",
        )
        .unwrap_err();
        assert_eq!(err.line, "sv_totally_made_up 1");
    }

    #[test]
    fn apply_command_tune_and_tune_zone() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "tune gravity 0").unwrap();
        assert_eq!(tuning.zone(0).get_by_name("gravity"), Some(0.0));
        // Zone 1 is untouched by a plain `tune` (that's `GlobalTuning()`, zone 0 only).
        assert_ne!(tuning.zone(1).get_by_name("gravity"), Some(0.0));
        apply_command(
            &mut config,
            &mut tuning,
            &mut switchers,
            &mut log,
            "tune_zone 1 gravity 0",
        )
        .unwrap();
        assert_eq!(tuning.zone(1).get_by_name("gravity"), Some(0.0));
    }

    #[test]
    fn apply_command_switch_open_sets_initial_false() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers = vec![Switcher::default(), Switcher::default(), Switcher::default()];
        switchers[2].initial = true;
        let mut log = Vec::new();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "switch_open 2").unwrap();
        assert!(!switchers[2].initial);
    }

    #[test]
    fn apply_command_ignores_blank_and_comment_lines() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "").unwrap();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "   ").unwrap();
        apply_command(&mut config, &mut tuning, &mut switchers, &mut log, "# a comment").unwrap();
    }

    #[test]
    fn apply_command_tune_zone_enter_leave_are_recognized_as_no_ops() {
        let mut config = ServerConfig::default();
        let mut tuning = TuningList::reset_to_baseline();
        let mut switchers: Vec<Switcher> = Vec::new();
        let mut log = Vec::new();
        apply_command(
            &mut config,
            &mut tuning,
            &mut switchers,
            &mut log,
            "tune_zone_enter 3 \"Enabled checkpoint 'Useless'.\"",
        )
        .unwrap();
        apply_command(
            &mut config,
            &mut tuning,
            &mut switchers,
            &mut log,
            "tune_zone_leave 3 \"bye\"",
        )
        .unwrap();
    }

    // `World::init`'s end-to-end behavior (needs `ddai_trace::synthetic`, so it lives in
    // `tests/world_smoke.rs` instead — see that file's doc comment on the dev-dependency-cycle
    // reason a `src/world.rs` unit test can't use `ddai_trace` directly):
    // `world_init_models_sv_hit_ending_up_true_for_a_pre_init_only_nohit_cfg`,
    // `world_init_lets_a_map_setting_make_sv_hit_stick`.
}
