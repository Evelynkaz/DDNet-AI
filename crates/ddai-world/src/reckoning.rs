//! Reckoning-core extrapolation: reconstructs the *exact* server-tick state of a character's
//! physics core from the lossy, dead-reckoned data a `CNetObj_Character` actually carries.
//!
//! Per DDNet 20.1's server (`character.cpp:953-970`, `CCharacter::TickDeferred`), the snapshot
//! never contains the character's *current* core — it contains `m_SendCore`, a copy taken at
//! `m_ReckoningTick` and kept as-is until the server itself notices its own idealized
//! ("no new input, empty world, default tuning") continuation of that copy has diverged from the
//! true core, at which point it resyncs both to the true value
//! (`m_SendCore = m_Core; m_ReckoningTick = Server()->Tick();`, `character.cpp:964-967`) or after
//! `m_ReckoningTick` gets more than 3 seconds stale. `SnapCharacter` (`character.cpp:1094-1103`)
//! writes `m_Tick = m_ReckoningTick` (or `0`, meaning "this *is* the current tick, no
//! extrapolation needed" — sent whenever `m_ReckoningTick` is still `0` or the game is paused).
//!
//! The client reconstructs the true tick-`T` core by replaying that same idealized continuation
//! itself, for `T - m_Tick` ticks: this is `CGameClient::OnNewSnapshot`'s `Evolve` lambda
//! (`gameclient.cpp:1725-1742`), ported field-for-field below as [`evolve_character_core`]. This
//! works for *every* character the same way, ours as much as anyone else's — the server does not
//! special-case the local player in this mechanism at all (dead reckoning is decided purely from
//! `m_Core.m_Reset`/divergence/staleness, never from "is this the requesting client's own tee").

use ddai_net::generated::objects;
use ddai_physics::collision::Collision;
use ddai_physics::core::{self, CharacterCore, NetCharacterCore, NetDDNetCharacter, TeamsCore, WorldCore};

/// The client's own dead-reckoning evolve cap (`gameclient.cpp:1893-1908`'s `EvolvePrev`/
/// `EvolveCur` guards: `PrevGameTick - Prev.m_Tick <= 3 * TickSpeed`) — 3 seconds at the standard
/// 50-tick server. Review round 1, finding F7: a snapshot claiming an implausibly old `net.tick`
/// (hostile or simply corrupt data) must not turn [`evolve_character_core`] into an
/// effectively-unbounded loop; the real server's own resync guarantee
/// (`m_ReckoningTick + 3 * TickSpeed < Tick` forces `m_SendCore = m_Core`, `character.cpp:964`)
/// means honest data never actually needs to evolve further than this anyway.
pub const MAX_EVOLVE_AGE_TICKS: i32 = 3 * core::SERVER_TICK_SPEED;

/// `CNetObj_DDNetCharacter` -> `ddai_physics::core::NetDDNetCharacter`: the subset
/// `CharacterCore::read_ddnet` actually reads (`m_TeleCheckpoint`/`m_StrongWeakId`/`m_TargetX`/
/// `m_TargetY`/`m_TuneZoneOverride` exist on the wire object but aren't core-level state — see
/// that method's own doc comment; `crate::live_world` reads them itself, straight off `net`, for
/// the DDRace-level `Character` fields they actually belong to).
pub fn to_net_ddnet_character(net: &objects::DDNetCharacter) -> NetDDNetCharacter {
    NetDDNetCharacter {
        flags: net.flags,
        freeze_end: net.freeze_end,
        jumps: net.jumps,
        jumped_total: net.jumped_total,
        ninja_activation_tick: net.ninja_activation_tick,
        freeze_start: net.freeze_start,
    }
}

/// `gameclient.cpp:1727-1742`'s `Evolve` lambda body, i.e. `CCharacterCore::Tick(false)` +
/// `Move()` + `Quantize()`, run once per elapsed tick in an otherwise-empty world (no sibling
/// characters — `Evolve`'s own `CWorldCore TempWorld;`/`CTeamsCore TempTeams;` are both freshly
/// default-constructed, empty) with **default tuning** (`CCharacterCore TempCore = CCharacterCore();`
/// — the real client never assigns `m_Tuning` at all here, so it keeps `CTuningParams`'s compile-time
/// defaults; [`CharacterCore::default`] matches that). Real map collision *is* still used
/// (`TempCore.Init(&TempWorld, Collision(), &TempTeams)` passes the real `Collision()`), so this
/// still reproduces gravity/ground friction/wall collision/hook-vs-map exactly — only
/// player-vs-player interaction and tuning-zone overrides are excluded, matching the server's own
/// idealized continuation exactly (see the module doc comment: those are exactly the two things
/// that make the server itself resync `m_SendCore` the moment they'd actually change anything).
///
/// `net`: the character's `CNetObj_Character` as received (`net.tick` is `m_Tick`: `0` means "this
/// already **is** the state at `target_tick`, do not evolve" — matches `if(...Cur.m_Tick)` guards
/// in `OnNewSnapshot`, which skip `Evolve` entirely when `m_Tick == 0`).
/// `target_tick`: the tick to reconstruct the core *at* — normally the snapshot's own game tick.
///
/// Review round 1, finding F1 (confirmed live: no weapon could ever fire in any prediction — the
/// hammer's own -1/infinite ammo never blocks it, but every other weapon's `ammo == 0` default
/// made `world::fire_weapon` return immediately). `active_weapon` itself
/// (`TempCore.m_ActiveWeapon = pCharacter->m_Weapon;`, `gameclient.cpp:1731`) is set the same way
/// as the real client here; **ammo itself is deliberately not touched by this function anymore**
/// (review round 3, finding F1, reopened): the naive `me.weapons[net.weapon].ammo = net.ammo_count`
/// this used to do here was wrong wherever `net.ammo_count` is the server's own "no information"
/// sentinel rather than a real reading — the server sends `AmmoCount == 0` for every *other*
/// player, and for our *own* tee while frozen, unconditionally (`character.cpp:1091,1140-1148`),
/// never a genuine "zero ammo" (DDRace weapons are never actually ammo-limited). Distinguishing a
/// real `0` from that sentinel needs to know which weapons this character has ever picked up
/// (`CNetObj_DDNetCharacter::m_Flags`' per-weapon `CHARACTERFLAG_WEAPON_*` bits,
/// i.e. `core.weapons[*].got` after [`core::CharacterCore::read_ddnet`] has run) — information
/// this function, called *before* `read_ddnet` (see `crate::live_world::upsert_character`'s call
/// order), does not have. See [`crate::live_world::reconstruct_weapon_ammo`] for the corrected
/// logic (moved there, where `got` is already known) and this crate's `BUILD REPORT` (round 3) for
/// the live repro this fixes (`GUN_START=1 fire_first_tick 2`/`4` in the reviewer's terms).
///
/// Review round 1, finding F7: never loops more than [`MAX_EVOLVE_AGE_TICKS`] times — a snapshot
/// claiming an implausibly stale `net.tick` (hostile or corrupt) evolves only that many ticks
/// (from `target_tick - MAX_EVOLVE_AGE_TICKS`, not from the claimed `net.tick`) instead of an
/// unbounded loop; matches the real client's own cap (`gameclient.cpp:1893-1908`) skipping
/// `Evolve` past 3 seconds of claimed age. `target_tick < net.tick` (a caller asking to "evolve"
/// backward, or the same hostile-data case) is likewise handled without panicking (review round 1,
/// finding F7's "replace the assert with an early return"): returns the raw, un-evolved core.
pub fn evolve_character_core(
    net: &objects::Character,
    target_tick: i32,
    collision: &Collision<f32>,
) -> CharacterCore<f32> {
    let mut me = CharacterCore::<f32>::default();
    me.init(); // `m_Id = -1`, matching `Init()` — never reassigned below (see this fn's BUILD REPORT note).
    me.read(&NetCharacterCore {
        x: net.x,
        y: net.y,
        vel_x: net.vel_x,
        vel_y: net.vel_y,
        angle: net.angle,
        direction: net.direction,
        jumped: net.jumped,
        hooked_player: net.hooked_player,
        hook_state: net.hook_state,
        hook_tick: net.hook_tick,
        hook_x: net.hook_x,
        hook_y: net.hook_y,
        hook_dx: net.hook_dx,
        hook_dy: net.hook_dy,
    });
    me.active_weapon = net.weapon;

    if net.tick == 0 || net.tick >= target_tick {
        return me;
    }

    // A single-slot isolated world, like the client's `Evolve`, whose `TempWorld` is empty apart
    // from the character itself. `WorldCore::from_characters` forces `core.id = key` (the slot's
    // array key), unlike the real `Init()` (which leaves `m_Id == -1`), so `id` is reset to `-1`
    // right after construction. The slot key must not be the character's own `hooked_player`:
    // in DDNet `m_apCharacters[m_HookedPlayer]` is null in the empty temp world, so a grabbed
    // player-hook is always released (`HOOK_RETRACTED`, `hooked_player = -1`, `hook_pos = pos`,
    // `gamecore.cpp:416`). Keying the slot 0 made `slot_of(0)` resolve to the character itself
    // for a hook on client 0, which panicked in `TeamsCore::can_keep_hook(0, -1)` (found on real
    // demos, task 8.4c). A tee cannot hook itself, so any key different from `net.hooked_player`
    // is alias-free.
    let key: u8 = if net.hooked_player == 0 { 1 } else { 0 };
    let mut temp: WorldCore<f32, 1> = WorldCore::from_characters(&[(key, me)]);
    if let Some(c) = temp.get_mut(key) {
        c.id = -1;
    }
    let empty_teams = TeamsCore::new();

    let start_tick = net.tick.max(target_tick.saturating_sub(MAX_EVOLVE_AGE_TICKS));
    let mut tick = start_tick;
    while tick < target_tick {
        tick += 1;
        core::tick(&mut temp, 0, collision, &empty_teams, false, true);
        core::move_character(&mut temp, 0, collision, &empty_teams);
        if let Some(c) = temp.get_mut(key) {
            core::quantize(c);
        }
    }
    *temp.get(key).expect("single-slot world always keeps its one character")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::core::WEAPON_NINJA;
    use ddai_physics::map::{MapData, TILE_SOLID, Tile};
    use ddai_physics::vmath::Vec2;

    /// A flat 20x10 room: solid floor at y=8, air everywhere else, solid border — enough for
    /// gravity/ground-friction extrapolation without any interaction this module excludes anyway.
    fn flat_room() -> MapData {
        let (w, h) = (20i32, 10i32);
        let mut game = vec![Tile::default(); (w * h) as usize];
        for x in 0..w {
            game[(8 * w + x) as usize] = Tile {
                index: TILE_SOLID,
                ..Default::default()
            };
        }
        MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    fn net_at(tick: i32, x: i32, y: i32) -> objects::Character {
        objects::Character {
            tick,
            x,
            y,
            vel_x: 0,
            vel_y: 0,
            angle: 0,
            direction: 0,
            jumped: 0,
            hooked_player: -1,
            hook_state: -1, // HOOK_IDLE
            hook_tick: 0,
            hook_x: x,
            hook_y: y,
            hook_dx: 0,
            hook_dy: 0,
            player_flags: 0,
            health: 0,
            armor: 0,
            ammo_count: 0,
            weapon: 0,
            emote: 0,
            attack_tick: 0,
        }
    }

    #[test]
    fn a_player_hook_is_released_for_every_hooked_id_including_zero() {
        // DDNet's `Evolve` runs in an empty temp world, so the hooked player is never found and
        // the hook is released: HOOK_RETRACTED, hooked_player -1, hook_pos = pos. Client id 0 used
        // to panic (task 8.4c regression) and then, in a first fix, kept the hook.
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let mut reference = None;
        for hooked in [5, 0, 1, 63] {
            let mut net = net_at(10, 100, 100);
            net.hook_state = ddai_physics::core::HOOK_GRABBED;
            net.hooked_player = hooked;
            net.hook_x = 100;
            net.hook_y = 125;
            let core = evolve_character_core(&net, 20, &collision);
            assert_eq!(core.hook_state, ddai_physics::core::HOOK_RETRACTED, "hooked {hooked}");
            assert_eq!(core.hooked_player(), -1, "hooked {hooked}");
            assert_ne!(core.hook_pos, Vec2::new(100.0, 125.0), "the old hook point is gone");
            assert_eq!(core.id, -1, "the returned core keeps the Init() id");
            // The outcome must not depend on which client id was hooked (no slot aliasing).
            let key = (core.pos, core.vel, core.hook_pos, core.hook_state);
            assert_eq!(*reference.get_or_insert(key), key, "hooked {hooked}");
        }
    }

    #[test]
    fn tick_zero_means_no_extrapolation_needed() {
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let mut net = net_at(0, 100, 100);
        net.vel_y = 123; // an arbitrary "current" velocity a `Tick=0` snapshot would carry as-is.
        let core = evolve_character_core(&net, 500, &collision);
        assert_eq!(core.pos, Vec2::new(100.0, 100.0));
        assert_eq!(core.vel.y, 123.0 / 256.0);
    }

    #[test]
    fn evolving_zero_ticks_is_a_pure_quantize_round_trip() {
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let net = net_at(10, 100, 100);
        let core = evolve_character_core(&net, 10, &collision);
        assert_eq!(core.pos, Vec2::new(100.0, 100.0));
    }

    #[test]
    fn free_fall_matches_stepping_an_isolated_world_the_same_number_of_ticks() {
        // Cross-check against the *other* available API for the same computation
        // (`core::tick`/`move_character`/`quantize`, driven directly instead of through
        // `evolve_character_core`) rather than a hand-derived closed-form displacement — this
        // is exactly what `evolve_character_core` itself does internally, so this test mainly
        // guards against a copy/paste divergence (e.g. a wrong `use_input`/`do_deferred_tick`
        // flag) between the two call sites.
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let net = net_at(100, 100, 50);
        let target = 137;

        let evolved = evolve_character_core(&net, target, &collision);

        let mut me = CharacterCore::<f32>::default();
        me.init();
        me.read(&NetCharacterCore {
            x: net.x,
            y: net.y,
            vel_x: net.vel_x,
            vel_y: net.vel_y,
            angle: net.angle,
            direction: net.direction,
            jumped: net.jumped,
            hooked_player: net.hooked_player,
            hook_state: net.hook_state,
            hook_tick: net.hook_tick,
            hook_x: net.hook_x,
            hook_y: net.hook_y,
            hook_dx: net.hook_dx,
            hook_dy: net.hook_dy,
        });
        let mut temp: WorldCore<f32, 1> = WorldCore::from_characters(&[(0u8, me)]);
        if let Some(c) = temp.get_mut(0) {
            c.id = -1;
        }
        let teams = TeamsCore::new();
        for _ in net.tick..target {
            core::tick(&mut temp, 0, &collision, &teams, false, true);
            core::move_character(&mut temp, 0, &collision, &teams);
            if let Some(c) = temp.get_mut(0) {
                core::quantize(c);
            }
        }
        let reference = *temp.get(0).unwrap();

        assert_eq!(evolved.pos, reference.pos);
        assert_eq!(evolved.vel, reference.vel);
    }

    #[test]
    fn evolution_is_deterministic() {
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let net = net_at(50, 100, 50);
        let a = evolve_character_core(&net, 200, &collision);
        let b = evolve_character_core(&net, 200, &collision);
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.vel, b.vel);
        assert_eq!(a.hook_state, b.hook_state);
    }

    /// Review round 1, finding F7/N1: hostile/malformed data (`target_tick < net.tick`, a caller
    /// "evolving" backward) must never panic, in debug *or* release builds — returns the raw,
    /// un-evolved core instead (this used to `debug_assert!`, which both panicked under `cargo
    /// test`'s default debug profile and silently did nothing in `--release`, i.e. a test that
    /// passed for the wrong reason in a release build).
    #[test]
    fn target_before_net_tick_never_panics_and_returns_the_raw_core() {
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let net = net_at(100, 111, 222);
        let core = evolve_character_core(&net, 10, &collision);
        assert_eq!(core.pos, Vec2::new(111.0, 222.0));
    }

    /// Review round 1, finding F7: an implausibly old `net.tick` (hostile or corrupt — the real
    /// server's own resync guarantee means honest data is never more than
    /// `MAX_EVOLVE_AGE_TICKS` stale) evolves only that many ticks, not
    /// `target_tick - net.tick` of them — this test's own `net.tick` is `i32::MIN`, which an
    /// unbounded loop would never finish.
    #[test]
    fn hostile_ancient_net_tick_is_capped_not_unbounded() {
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        let mut hostile = net_at(i32::MIN, 100, 50);
        hostile.vel_y = 0;
        let target = 10_000;
        let capped = evolve_character_core(&hostile, target, &collision);

        let mut honest = net_at(target - MAX_EVOLVE_AGE_TICKS, 100, 50);
        honest.vel_y = 0;
        let reference = evolve_character_core(&honest, target, &collision);

        assert_eq!(capped.pos, reference.pos);
        assert_eq!(capped.vel, reference.vel);
    }

    /// Review round 3, finding F1: `evolve_character_core` no longer touches ammo at all (moved to
    /// `crate::live_world::reconstruct_weapon_ammo`, see this function's own doc comment for why) —
    /// this pins that down for every weapon, including ninja, so a future change here can't
    /// silently reintroduce the old (wrong) direct copy without this test catching it.
    #[test]
    fn ammo_is_never_touched_regardless_of_weapon_or_wire_value() {
        let map = flat_room();
        let collision: Collision<f32> = Collision::new(&map);
        for (weapon, ammo_count) in [
            (ddai_physics::core::WEAPON_GUN, 7),
            (ddai_physics::core::WEAPON_HAMMER, -1),
            (WEAPON_NINJA, 12345),
        ] {
            let mut net = net_at(0, 100, 100);
            net.weapon = weapon;
            net.ammo_count = ammo_count;
            let core = evolve_character_core(&net, 500, &collision);
            assert_eq!(core.active_weapon, weapon);
            assert_eq!(core.weapons[weapon as usize].ammo, 0, "weapon {weapon}");
        }
    }
}
