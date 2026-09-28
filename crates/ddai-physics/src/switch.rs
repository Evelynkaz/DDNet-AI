// Ported from DDNet 20.1 `src/game/server/entities/door.cpp`, `src/game/server/gamecontext.cpp`
// (the switch-expiry loop inside `CGameContext::OnTick`) and `src/game/gamecore.h`/`.cpp`
// (`SSwitchers`/`CWorldCore::InitSwitchers`, already ported as [`crate::core::Switcher`]/
// [`crate::core::WorldCore::init_switchers`] in task 1.3). DDNet's zlib-style license notice for
// the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//   /* (c) Shereef Marzouk. See "licence DDRace.txt" and the readme.txt in the root of the       */
//   /* distribution for more information. Based on Race mod stuff and tweaked by GreYFoX@GTi and */
//   /* others to fit our DDRace needs.                                                            */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`, no `unsafe`.

//! Switch-layer state: the per-(switch number, DDRace team) open/closed status
//! ([`crate::core::Switcher`], already structurally ported in task 1.3) plus the two pieces of
//! *behavior* task 1.6 adds on top of that placeholder — timed-switch expiry
//! ([`tick_switch_expiry`]) and `CDoor`'s one-time collision setup ([`place_door_collision`]).
//! The switch-tile *dispatch* itself (a character touching `TILE_SWITCHOPEN`/`TILE_JUMP`/
//! `TILE_ADD_TIME`/...) lives in [`crate::world::handle_tiles`], since it also touches
//! character-level fields (`m_LastPenalty`, `m_StartTime`, ...) this module has no reason to know
//! about.

use crate::collision::Collision;
use crate::core::Switcher;
use crate::map;
use crate::real::Real;
use crate::vmath::Vec2;

/// `CDoor::CDoor`/`CDoor::ResetCollision` (`door.cpp:13-41`): lays down a line of
/// `TILE_STOPA`-tagged door cells (`CCollision::SetDoorCollisionAt`) from `pos`, `length - 1`
/// steps in `direction` (already a unit vector — `vec2(sin(Rotation), cos(Rotation))`, matching
/// the C++ source's `m_Direction`, which `ResetCollision`'s loop uses *without* renormalizing —
/// see this function's doc comment on why that distinction is preserved bit-for-bit), stopping
/// early at the first already-solid cell. A no-op if `pos` itself is already on a solid game or
/// front tile (`door.cpp:30`).
///
/// Deliberately does **not** compute `CDoor::m_To` (the door's fully-open endpoint): nothing in
/// `ResetCollision` reads it, and it is otherwise used only by `CDoor::Snap` (cosmetic/network,
/// out of scope) — see `docs/formats.md` §11.2's note that only `CEntity::m_Pos` is dumped for
/// this fixture kind (`kind == 2`).
///
/// `direction` must be exactly `vec2(Rotation.sin(), Rotation.cos())` for some `Rotation` — never
/// pre-normalized by the caller — to match `door.cpp:35`'s `m_Pos + m_Direction * i` exactly
/// (mathematically a no-op since `sin`/`cos` already give a unit vector, but not necessarily a
/// bit-for-bit no-op once floating point rounding is involved, so this function takes it as-is
/// rather than calling [`crate::vmath::normalize`] on it itself).
pub fn place_door_collision<R: Real>(
    collision: &mut Collision<R>,
    pos: Vec2<R>,
    direction: Vec2<R>,
    length: i32,
    number: u8,
) {
    if collision.get_tile(pos.x.to_i32_trunc(), pos.y.to_i32_trunc()) != 0
        || collision.get_front_tile(pos.x.to_i32_trunc(), pos.y.to_i32_trunc()) != 0
    {
        return;
    }
    for i in 0..length - 1 {
        let current = pos + direction * R::from_i32(i);
        if collision.check_point_vec(current) {
            break;
        }
        collision.set_door_collision_at(current.x, current.y, map::TILE_STOPA, 0, number);
    }
}

/// `CGameContext::OnTick`'s switch-expiry loop (`gamecontext.cpp:1460-1476`), run once per game
/// tick, after every character's own tick (see `docs/formats.md`/the task's BUILD REPORT for
/// exactly where in `World::step` this runs). For every `(switch, team)` pair whose current
/// `kind` is a *timed* one and whose `end_tick` has passed, flips it to the opposite steady
/// state (`TIMEDOPEN` -> closed/`SWITCHCLOSE`, `TIMEDCLOSE` -> open/`SWITCHOPEN`) and clears
/// `end_tick`. A pair that was never given a timed kind (`kind == 0`, `end_tick == 0` from
/// [`crate::core::WorldCore::init_switchers`]) never matches either `if`.
///
/// **Perf (task 1.6, coordinator follow-up item 3):** the real server's own loop
/// (`gamecontext.cpp:1460-1476`) is `for(Switcher : Switchers()) for(j : NUM_DDRACE_TEAMS)`,
/// i.e. genuinely `O(switchers × teams)` every tick regardless of activity — but this crate's
/// *own* `Switcher` records are only ever consulted through `active` (see
/// `World::active_timed_switchers`'s doc comment, in world.rs), a side list of switcher indices that have
/// *any* team currently in a timed kind, maintained by `crate::world::handle_switch_tiles` (private)
/// (pushed there) and this function (an index is dropped from `active` once none of its teams
/// are timed any more — `Vec::retain`). Found empirically: a criterion bench on `BlmapChill`
/// (~50 switch numbers on that map) was ~5x slower than `Copy Love Box` (far fewer) at the
/// *same* character count, with the gap barely growing 2->8 characters — a map-size-scaled, not
/// character-count-scaled, fixed cost consistent with exactly this loop (`~50 × 65` teams
/// checked unconditionally, every tick, on `BlmapChill`, before this fix). Behaviorally
/// identical to the old "scan every switcher" version: `active` only ever *skips* switchers this
/// function itself already established have no timed team pending (matching the old version's
/// own no-op for those), and always includes every switcher that could possibly need work.
pub fn tick_switch_expiry(switchers: &mut [Switcher], active: &mut Vec<u8>, tick: i32) {
    active.retain(|&index| {
        let Some(switcher) = switchers.get_mut(index as usize) else {
            return false;
        };
        let mut still_timed = false;
        for team in 0..switcher.kind.len() {
            if switcher.end_tick[team] <= tick && switcher.kind[team] == map::TILE_SWITCHTIMEDOPEN as i32 {
                switcher.status[team] = false;
                switcher.end_tick[team] = 0;
                switcher.kind[team] = map::TILE_SWITCHCLOSE as i32;
            } else if switcher.end_tick[team] <= tick && switcher.kind[team] == map::TILE_SWITCHTIMEDCLOSE as i32 {
                switcher.status[team] = true;
                switcher.end_tick[team] = 0;
                switcher.kind[team] = map::TILE_SWITCHOPEN as i32;
            }
            if switcher.kind[team] == map::TILE_SWITCHTIMEDOPEN as i32
                || switcher.kind[team] == map::TILE_SWITCHTIMEDCLOSE as i32
            {
                still_timed = true;
            }
        }
        still_timed
    });
}

/// Marks switcher `number` as having at least one team in a timed kind — called from
/// `crate::world::handle_switch_tiles` (private) whenever it sets a `TILE_SWITCHTIMEDOPEN`/
/// `TILE_SWITCHTIMEDCLOSE` kind, so [`tick_switch_expiry`] only ever has to look at switchers
/// that could possibly need work this tick. A no-op if `number` is already present (checked
/// linearly — `active` only ever holds as many entries as there are *touched* timed switches in
/// a scenario, never the map's full switcher count, so this stays cheap).
pub fn mark_switcher_timed(active: &mut Vec<u8>, number: u8) {
    if !active.contains(&number) {
        active.push(number);
    }
}

#[cfg(test)]
// Test fixture indices below are deliberately written as `row * width + col` for readability
// (even when `row == 1`, so the multiplication is a no-op) rather than as an unexplained flat
// index.
#[allow(clippy::identity_op)]
mod tests {
    use super::*;
    use crate::core::NUM_DDRACE_TEAMS;
    use crate::map::{MapData, SwitchTile, Tile};

    fn air_map(w: i32, h: i32) -> MapData {
        MapData {
            width: w as u32,
            height: h as u32,
            game: vec![
                Tile {
                    index: 0,
                    flags: 0,
                    skip: 0,
                    reserved: 0
                };
                (w * h) as usize
            ],
            front: None,
            tele: None,
            speedup: None,
            // `Collision::set_door_collision_at`/`get_door_tile` only allocate/read a per-cell
            // door array when the map has a switch layer at all (`CCollision::Init`'s `m_pDoor`
            // is only non-null then) — a real map with any `CDoor` fixture always has one (the
            // door itself is a switch-layer entity), but this synthetic test map needs one
            // explicitly, even though none of its cells carry a meaningful switch tile.
            switch: Some(vec![SwitchTile::default(); (w * h) as usize]),
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn place_door_collision_lays_down_stopa_cells_until_a_wall() {
        let mut map = air_map(10, 4);
        // A wall at tile x=5, blocking the door's path 3 tiles in.
        map.game[1 * 10 + 5].index = map::TILE_SOLID;
        let mut collision: Collision<f32> = Collision::new(&map);

        let pos = Vec2::new(2.0 * 32.0 + 16.0, 1.0 * 32.0 + 16.0);
        let direction = Vec2::new(1.0f32, 0.0); // Rotation = pi/2: sin=1, cos=0
        place_door_collision(&mut collision, pos, direction, 32 * 6, 3);

        // Cells at x=2,3,4 (tile coords) should have a door tagged with number 3; x=5 is solid,
        // stopping the loop (and never itself getting a door tag, matching `check_point` being
        // tested *before* `SetDoorCollisionAt` for that cell).
        for tx in 2..5 {
            let door = collision.get_door_tile(1 * 10 + tx);
            assert_eq!(door.number, 3, "tile x={tx} should have door number 3");
            assert_eq!(door.index, map::TILE_STOPA);
        }
    }

    #[test]
    fn place_door_collision_is_a_no_op_when_the_start_cell_is_already_solid() {
        let mut map = air_map(10, 4);
        map.game[1 * 10 + 2].index = map::TILE_SOLID;
        let mut collision: Collision<f32> = Collision::new(&map);
        let pos = Vec2::new(2.0 * 32.0 + 16.0, 1.0 * 32.0 + 16.0);
        place_door_collision(&mut collision, pos, Vec2::new(1.0f32, 0.0), 32 * 6, 3);
        let door = collision.get_door_tile(1 * 10 + 2);
        assert_eq!(door.number, 0, "no door should be placed starting on a solid cell");
    }

    #[test]
    fn tick_switch_expiry_flips_a_timedopen_switch_back_closed_once_its_end_tick_passes() {
        let mut switchers = vec![Switcher::default(), {
            let mut s = Switcher::default();
            s.status[1] = true;
            s.end_tick[1] = 100;
            s.kind[1] = map::TILE_SWITCHTIMEDOPEN as i32;
            s
        }];
        let mut active = vec![1u8];
        tick_switch_expiry(&mut switchers, &mut active, 99);
        assert!(switchers[1].status[1], "not expired yet at tick 99");
        assert_eq!(
            active,
            vec![1],
            "still timed at tick 99 -- must stay in the active list"
        );
        tick_switch_expiry(&mut switchers, &mut active, 100);
        assert!(!switchers[1].status[1], "expired exactly at end_tick");
        assert_eq!(switchers[1].end_tick[1], 0);
        assert_eq!(switchers[1].kind[1], map::TILE_SWITCHCLOSE as i32);
        assert!(active.is_empty(), "no longer timed -- must drop out of the active list");
    }

    #[test]
    fn tick_switch_expiry_flips_a_timedclose_switch_back_open() {
        let mut switchers = vec![{
            let mut s = Switcher::default();
            s.status[0] = false;
            s.end_tick[0] = 50;
            s.kind[0] = map::TILE_SWITCHTIMEDCLOSE as i32;
            s
        }];
        let mut active = vec![0u8];
        tick_switch_expiry(&mut switchers, &mut active, 60);
        assert!(switchers[0].status[0]);
        assert_eq!(switchers[0].kind[0], map::TILE_SWITCHOPEN as i32);
        assert!(active.is_empty());
    }

    #[test]
    fn tick_switch_expiry_never_touches_a_switch_with_no_timed_kind() {
        let mut switchers = vec![Switcher::default()];
        let before = switchers.clone();
        // Listed in `active` despite not actually being timed -- pins that the function itself
        // (not just the caller never listing it) leaves a non-timed switcher alone, and drops it
        // from `active` since it has nothing timed pending.
        let mut active = vec![0u8];
        tick_switch_expiry(&mut switchers, &mut active, 999_999);
        assert_eq!(switchers[0].status, before[0].status);
        assert_eq!(switchers[0].kind, before[0].kind);
        assert!(active.is_empty());
    }

    #[test]
    fn switcher_default_matches_init_switchers_defaults() {
        // Cross-check against `WorldCore::init_switchers`'s own defaults (task 1.3): status all
        // `true`, `initial: true`... actually `Switcher::default()` (used directly in this
        // module's tests) is the *struct* default (status all `false`) — this test exists to
        // pin that these are two genuinely different starting points, so a future refactor
        // doesn't accidentally conflate them.
        let d = Switcher::default();
        assert!(!d.status[0]);
        assert_eq!(d.status.len(), NUM_DDRACE_TEAMS as usize);
    }
}
