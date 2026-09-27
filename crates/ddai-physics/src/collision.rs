// Ported from DDNet 20.1 `src/game/collision.{h,cpp}` (`CCollision` and the free functions
// declared alongside it). DDNet's zlib-style license notice for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`; no `unsafe`, so `CCollision`'s
// raw-pointer fields (`m_pTiles`/`m_pFront`/...) become owned `Vec`s built once in [`Collision::new`]
// (`CCollision::Init`'s job); the `CALLBACK_SWITCHACTIVE`/`pUser` pair becomes a generic closure
// parameter (see [`Collision::get_move_restrictions`]) since Rust closures already capture their
// own state, and `g_Config.m_SvOldTeleportHook`/`m_SvOldTeleportWeapons` (server console
// variables `CCollision` reads as globals in C++) become explicit `bool` parameters on
// [`Collision::intersect_line_tele_hook`]/[`Collision::intersect_line_tele_weapon`] instead —
// this crate has no global config system, and Oracle A's harness runs with a
// zero-initialized `CConfig` (i.e. both flags `false`), which callers get by passing `false`.
//
// Deliberately not ported (out of scope, no bearing on physics simulation): `FillAntibot`
// (antibot integration — this crate has no antibot system) and `Unload`/the destructor (Rust
// drops `Collision` normally; there is no "loaded but reset to empty" state to model — call
// [`Collision::empty`] for the analogous "freshly constructed, `Init` never called" state).

use crate::map::{self, MapData, SpeedupTile, SwitchTile, TeleTile, Tile, TuneTile};
use crate::real::Real;
use crate::vmath::{self, Vec2};
use std::collections::HashMap;

/// `CDoorTile` (`mapitems.h`): `CCollision::Init`'s precomputed per-cell door record, derived
/// from the switch layer. `index` (`m_Index`, the door's *tile* id) is only ever set by
/// `SetDoorCollisionAt` — a DDRace game-logic call (`character.cpp`, switch-door entities), out
/// of this crate's scope — so it stays `0` for every cell this crate ever builds a [`Collision`]
/// from; `number` is populated by [`Collision::new`] exactly like `CCollision::Init` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DoorTile {
    /// `m_Index`: the door's tile id (`0` = no door here).
    pub index: u8,
    /// `m_Flags`: rotation/flip bits.
    pub flags: u8,
    /// `m_Number`: the switch number this door belongs to.
    pub number: u8,
}

// --- `CANTMOVE_*` (`collision.h`) -------------------------------------------------------------

/// Blocks moving left.
pub const CANTMOVE_LEFT: i32 = 1 << 0;
/// Blocks moving right.
pub const CANTMOVE_RIGHT: i32 = 1 << 1;
/// Blocks moving up.
pub const CANTMOVE_UP: i32 = 1 << 2;
/// Blocks moving down.
pub const CANTMOVE_DOWN: i32 = 1 << 3;

/// `ClampVel(int MoveRestriction, vec2 Vel)` (`collision.cpp`).
pub fn clamp_vel<R: Real>(move_restriction: i32, mut vel: Vec2<R>) -> Vec2<R> {
    if vel.x > R::ZERO && (move_restriction & CANTMOVE_RIGHT) != 0 {
        vel.x = R::ZERO;
    }
    if vel.x < R::ZERO && (move_restriction & CANTMOVE_LEFT) != 0 {
        vel.x = R::ZERO;
    }
    if vel.y > R::ZERO && (move_restriction & CANTMOVE_DOWN) != 0 {
        vel.y = R::ZERO;
    }
    if vel.y < R::ZERO && (move_restriction & CANTMOVE_UP) != 0 {
        vel.y = R::ZERO;
    }
    vel
}

// --- `GetMoveRestrictionsRaw`/`GetMoveRestrictionsMask`/`GetMoveRestrictions` (file-local free
// functions in `collision.cpp`). -----------------------------------------------------------

const MR_DIR_HERE: usize = 0;
const MR_DIR_RIGHT: usize = 1;
const MR_DIR_DOWN: usize = 2;
const MR_DIR_LEFT: usize = 3;
const MR_DIR_UP: usize = 4;
const NUM_MR_DIRS: usize = 5;

fn move_restrictions_raw(tile: u8, flags: u8) -> i32 {
    let flags = flags & (map::TILEFLAG_XFLIP | map::TILEFLAG_YFLIP | map::TILEFLAG_ROTATE);
    if tile == map::TILE_STOP {
        return match flags {
            map::ROTATION_0 => CANTMOVE_DOWN,
            map::ROTATION_90 => CANTMOVE_LEFT,
            map::ROTATION_180 => CANTMOVE_UP,
            map::ROTATION_270 => CANTMOVE_RIGHT,
            f if f == (map::TILEFLAG_YFLIP ^ map::ROTATION_0) => CANTMOVE_UP,
            f if f == (map::TILEFLAG_YFLIP ^ map::ROTATION_90) => CANTMOVE_RIGHT,
            f if f == (map::TILEFLAG_YFLIP ^ map::ROTATION_180) => CANTMOVE_DOWN,
            f if f == (map::TILEFLAG_YFLIP ^ map::ROTATION_270) => CANTMOVE_LEFT,
            _ => 0,
        };
    }
    if tile == map::TILE_STOPS {
        return match flags {
            f if f == map::ROTATION_0
                || f == map::ROTATION_180
                || f == (map::TILEFLAG_YFLIP ^ map::ROTATION_0)
                || f == (map::TILEFLAG_YFLIP ^ map::ROTATION_180) =>
            {
                CANTMOVE_DOWN | CANTMOVE_UP
            }
            f if f == map::ROTATION_90
                || f == map::ROTATION_270
                || f == (map::TILEFLAG_YFLIP ^ map::ROTATION_90)
                || f == (map::TILEFLAG_YFLIP ^ map::ROTATION_270) =>
            {
                CANTMOVE_LEFT | CANTMOVE_RIGHT
            }
            _ => 0,
        };
    }
    if tile == map::TILE_STOPA {
        return CANTMOVE_LEFT | CANTMOVE_RIGHT | CANTMOVE_UP | CANTMOVE_DOWN;
    }
    0
}

fn move_restrictions_mask(direction: usize) -> i32 {
    match direction {
        MR_DIR_HERE => 0,
        MR_DIR_RIGHT => CANTMOVE_RIGHT,
        MR_DIR_DOWN => CANTMOVE_DOWN,
        MR_DIR_LEFT => CANTMOVE_LEFT,
        MR_DIR_UP => CANTMOVE_UP,
        _ => unreachable!("invalid move-restriction direction {direction}"),
    }
}

fn move_restrictions_for(direction: usize, tile: u8, flags: u8) -> i32 {
    let result = move_restrictions_raw(tile, flags);
    // "Generally, stoppers only have an effect if they block us from moving *onto* them. The one
    // exception is one-way blockers, they can also block us from moving if we're on top of
    // them." (`collision.cpp` comment, kept verbatim below.)
    if direction == MR_DIR_HERE && tile == map::TILE_STOP {
        return result;
    }
    result & move_restrictions_mask(direction)
}

/// A tele layer number's 0-indexed record key (`Number - 1`, matching `m_TeleIns`/`m_TeleOuts`/
/// `m_TeleCheckOuts`/`m_TeleOthers`'s `std::map<int, ...>` keys in `collision.cpp`).
type TeleNumber = u8;

/// Port of 20.1 `CCollision`: the map's tile layers plus everything `CCollision::Init`
/// precomputes from them (tele in/out/checkpoint tables, the door array, the highest switch
/// number). Built once from a [`MapData`] via [`Collision::new`] and shared (by reference) across
/// every tick — never cloned per tick, unlike [`crate::core::CharacterCore`].
#[derive(Debug, Clone)]
pub struct Collision<R: Real> {
    width: i32,
    height: i32,
    /// `IsSolid`/`CheckPoint`'s answer for every cell, precomputed once here instead of
    /// recomputed (`GetTile`'s range check + two equality comparisons) on every call — review
    /// round 2, finding F4 ("cheap, bit-exactness-safe wins"): `CheckPoint` is the single
    /// hottest function in this crate (`TestBox` calls it up to 4 times per call, `MoveBox` calls
    /// `TestBox` up to 3 times per step, and `MoveBox` steps up to `(int)|Vel|` times per
    /// `Move()`). Exactly equivalent to `get_tile(x, y) == TILE_SOLID || get_tile(x, y) ==
    /// TILE_NOHOOK` for every `(x, y)` — this changes nothing about *what* is computed, only
    /// *when* (once, in [`Collision::new`], instead of on every call).
    solid: Vec<bool>,
    game: Vec<Tile>,
    front: Option<Vec<Tile>>,
    tele: Option<Vec<TeleTile>>,
    speedup: Option<Vec<SpeedupTile>>,
    switch: Option<Vec<SwitchTile>>,
    tune: Option<Vec<TuneTile>>,
    door: Option<Vec<DoorTile>>,
    highest_switch_number: i32,
    tele_ins: HashMap<TeleNumber, Vec<Vec2<R>>>,
    tele_outs: HashMap<TeleNumber, Vec<Vec2<R>>>,
    tele_check_outs: HashMap<TeleNumber, Vec<Vec2<R>>>,
    tele_others: HashMap<TeleNumber, Vec<Vec2<R>>>,
    has_hook_tele_ins: bool,
}

impl<R: Real> Collision<R> {
    /// `CCollision()` immediately followed by `Unload()` in the C++ source: a `Collision` with
    /// no map loaded (`0x0`, no tiles). Exists so [`Collision`]'s "not yet initialized" state is
    /// representable, matching what a freshly-constructed, never-`Init`-ed `CCollision` reads as
    /// (`GetTile`/`IsSolid`/... all defensively return `0`/`false` — see e.g. `GetTile`'s
    /// `if(!m_pTiles) return 0;`).
    pub fn empty() -> Self {
        Collision {
            width: 0,
            height: 0,
            solid: Vec::new(),
            game: Vec::new(),
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            door: None,
            highest_switch_number: 0,
            tele_ins: HashMap::new(),
            tele_outs: HashMap::new(),
            tele_check_outs: HashMap::new(),
            tele_others: HashMap::new(),
            has_hook_tele_ins: false,
        }
    }

    /// `CCollision::Init(CLayers *pLayers)`. `map` must satisfy [`MapData::validate`] (every
    /// present layer has exactly `width * height` tiles) — this is not re-checked here (mirrors
    /// `Init`, which trusts `CLayers`/the datafile reader to have already enforced it).
    pub fn new(map: &MapData) -> Self {
        let width = map.width as i32;
        let height = map.height as i32;
        let n = map.cell_count();

        let mut door: Option<Vec<DoorTile>> = None;
        let mut highest_switch_number: i32 = 0;
        let mut switch = map.switch.clone();
        if let Some(switch_tiles) = switch.as_mut() {
            let mut doors = vec![DoorTile::default(); n];
            for (i, s) in switch_tiles.iter_mut().enumerate() {
                if i32::from(s.number) > highest_switch_number {
                    highest_switch_number = i32::from(s.number);
                }
                doors[i].number = s.number;

                let index = s.kind;
                if index <= map::TILE_NPH_ENABLE {
                    let keep = (map::TILE_JUMP..=map::TILE_SUBTRACT_TIME).contains(&index)
                        || index == map::TILE_ALLOW_TELE_GUN
                        || index == map::TILE_ALLOW_BLUE_TELE_GUN;
                    if !keep {
                        s.kind = 0;
                    }
                }
            }
            door = Some(doors);
        }

        let mut tele_ins: HashMap<TeleNumber, Vec<Vec2<R>>> = HashMap::new();
        let mut tele_outs: HashMap<TeleNumber, Vec<Vec2<R>>> = HashMap::new();
        let mut tele_check_outs: HashMap<TeleNumber, Vec<Vec2<R>>> = HashMap::new();
        let mut tele_others: HashMap<TeleNumber, Vec<Vec2<R>>> = HashMap::new();
        let mut has_hook_tele_ins = false;
        if let Some(tele) = &map.tele {
            for (i, t) in tele.iter().enumerate() {
                if t.number != 0 && t.kind != 0 {
                    let x = (i as i32 % width) as f64 * 32.0 + 16.0;
                    let y = (i as i32 / width) as f64 * 32.0 + 16.0;
                    let pos = Vec2::new(R::from_f64(x), R::from_f64(y));
                    let key = t.number - 1;
                    match t.kind {
                        k if k == map::TILE_TELEIN => tele_ins.entry(key).or_default().push(pos),
                        k if k == map::TILE_TELEOUT => tele_outs.entry(key).or_default().push(pos),
                        k if k == map::TILE_TELECHECKOUT => tele_check_outs.entry(key).or_default().push(pos),
                        k => {
                            tele_others.entry(key).or_default().push(pos);
                            if k == map::TILE_TELEINHOOK {
                                has_hook_tele_ins = true;
                            }
                        }
                    }
                }
            }
        }

        let solid = map
            .game
            .iter()
            .map(|t| t.index == map::TILE_SOLID || t.index == map::TILE_NOHOOK)
            .collect();

        Collision {
            width,
            height,
            solid,
            game: map.game.clone(),
            front: map.front.clone(),
            tele: map.tele.clone(),
            speedup: map.speedup.clone(),
            switch,
            tune: map.tune.clone(),
            door,
            highest_switch_number,
            tele_ins,
            tele_outs,
            tele_check_outs,
            tele_others,
            has_hook_tele_ins,
        }
    }

    /// The map's width, in tiles.
    pub fn width(&self) -> i32 {
        self.width
    }
    /// The map's height, in tiles.
    pub fn height(&self) -> i32 {
        self.height
    }

    // --- Basic point/tile queries -------------------------------------------------------------

    /// `CCollision::GetTile(int x, int y)`: the game-layer tile id at pixel `(x, y)` if it's in
    /// `[TILE_SOLID, TILE_NOLASER]`, else `0`. `x`/`y` are divided by 32 directly (no rounding).
    pub fn get_tile(&self, x: i32, y: i32) -> i32 {
        if self.game.is_empty() {
            return 0;
        }
        let nx = (x / 32).clamp(0, self.width - 1);
        let ny = (y / 32).clamp(0, self.height - 1);
        let index = (ny * self.width + nx) as usize;
        let idx = self.game[index].index;
        if (map::TILE_SOLID..=map::TILE_NOLASER).contains(&idx) {
            idx as i32
        } else {
            0
        }
    }

    /// `CCollision::GetFrontTile(int x, int y)`: `TILE_DEATH`/`TILE_NOLASER` only (the only two
    /// front-layer ids `GetFrontTile` — as opposed to `GetFrontTileIndex` — ever returns).
    pub fn get_front_tile(&self, x: i32, y: i32) -> i32 {
        let Some(front) = &self.front else { return 0 };
        let nx = (x / 32).clamp(0, self.width - 1);
        let ny = (y / 32).clamp(0, self.height - 1);
        let idx = front[(ny * self.width + nx) as usize].index;
        if idx == map::TILE_DEATH || idx == map::TILE_NOLASER {
            idx as i32
        } else {
            0
        }
    }

    /// `CCollision::IsSolid(int x, int y)`. See the `solid` field's doc comment (private —
    /// `Collision`'s struct definition, above) for why this reads a precomputed table instead of
    /// calling [`Collision::get_tile`] — behaviorally identical (same clamp, same result), just
    /// not recomputing `get_tile`'s range check and two comparisons on every call.
    pub fn is_solid(&self, x: i32, y: i32) -> bool {
        if self.solid.is_empty() {
            return false;
        }
        let nx = (x / 32).clamp(0, self.width - 1);
        let ny = (y / 32).clamp(0, self.height - 1);
        self.solid[(ny * self.width + nx) as usize]
    }

    /// `CCollision::CheckPoint(float x, float y)`: `IsSolid(round_to_int(x), round_to_int(y))`.
    pub fn check_point(&self, x: R, y: R) -> bool {
        self.is_solid(vmath::round_to_int(x), vmath::round_to_int(y))
    }

    /// `CCollision::CheckPoint(vec2 Pos)`.
    pub fn check_point_vec(&self, pos: Vec2<R>) -> bool {
        self.check_point(pos.x, pos.y)
    }

    /// `CCollision::GetCollisionAt(float x, float y)`.
    pub fn get_collision_at(&self, x: R, y: R) -> i32 {
        self.get_tile(vmath::round_to_int(x), vmath::round_to_int(y))
    }

    /// `CCollision::GetFrontCollisionAt(float x, float y)`.
    pub fn get_front_collision_at(&self, x: R, y: R) -> i32 {
        self.get_front_tile(vmath::round_to_int(x), vmath::round_to_int(y))
    }

    /// `CCollision::TestBox(vec2 Pos, vec2 Size)`.
    pub fn test_box(&self, pos: Vec2<R>, size: Vec2<R>) -> bool {
        let half = size * R::from_f64(0.5);
        self.check_point(pos.x - half.x, pos.y - half.y)
            || self.check_point(pos.x + half.x, pos.y - half.y)
            || self.check_point(pos.x - half.x, pos.y + half.y)
            || self.check_point(pos.x + half.x, pos.y + half.y)
    }

    /// `CCollision::IsOnGround(vec2 Pos, float Size)`.
    pub fn is_on_ground(&self, pos: Vec2<R>, size: R) -> bool {
        let half = size / R::from_i32(2);
        self.check_point(pos.x + half, pos.y + half + R::from_i32(5))
            || self.check_point(pos.x - half, pos.y + half + R::from_i32(5))
    }

    // --- Pure map index / tile lookups by index --------------------------------------------

    /// `CCollision::GetPureMapIndex(float x, float y)`: `round_to_int` *then* divide by 32
    /// (unlike [`Collision::get_tile`], which divides the raw pixel coordinate directly) —
    /// this distinction matters and is preserved exactly.
    pub fn get_pure_map_index(&self, x: R, y: R) -> usize {
        let nx = (vmath::round_to_int(x) / 32).clamp(0, self.width.max(1) - 1);
        let ny = (vmath::round_to_int(y) / 32).clamp(0, self.height.max(1) - 1);
        (ny * self.width + nx) as usize
    }

    /// `CCollision::GetPureMapIndex(vec2 Pos)`.
    pub fn get_pure_map_index_vec(&self, pos: Vec2<R>) -> usize {
        self.get_pure_map_index(pos.x, pos.y)
    }

    /// The `int`-argument overload C++ callers get "for free" via implicit `int -> float`
    /// conversion before reaching `GetPureMapIndex(float, float)` — spelled out explicitly here
    /// since Rust has no implicit numeric conversions. Used by [`Collision::is_through`]/
    /// [`Collision::is_hook_blocker`], which take already-rounded pixel coordinates.
    fn pure_map_index_from_ints(&self, x: i32, y: i32) -> usize {
        // `vmath::round_to_int(R::from_i32(v))` is provably exactly `v` again for every `v` this
        // function is ever called with (pixel coordinates ± a 32px tile offset — always tiny
        // compared to `f32`'s 24-bit exact-integer range, per `Real::from_i32`'s doc comment):
        // `from_i32` widens exactly, and `round_to_int` of an already-integer value adds/
        // subtracts `0.5` then truncates, landing back on the same integer. So this divides `x`/
        // `y` directly instead of taking the `int -> float -> round_to_int` round trip through
        // [`Collision::get_pure_map_index`] (review round 2, finding F4) — same result, skips
        // work that was provably a no-op.
        let nx = (x / 32).clamp(0, self.width.max(1) - 1);
        let ny = (y / 32).clamp(0, self.height.max(1) - 1);
        (ny * self.width + nx) as usize
    }

    /// `CCollision::GetTileIndex(int Index)`.
    pub fn get_tile_index(&self, index: i32) -> i32 {
        if index < 0 {
            0
        } else {
            self.game[index as usize].index as i32
        }
    }

    /// `CCollision::GetFrontTileIndex(int Index)`.
    pub fn get_front_tile_index(&self, index: i32) -> i32 {
        if index < 0 {
            return 0;
        };
        match &self.front {
            Some(f) => f[index as usize].index as i32,
            None => 0,
        }
    }

    /// `CCollision::GetTileFlags(int Index)`.
    pub fn get_tile_flags(&self, index: i32) -> i32 {
        if index < 0 {
            0
        } else {
            self.game[index as usize].flags as i32
        }
    }

    /// `CCollision::GetFrontTileFlags(int Index)`.
    pub fn get_front_tile_flags(&self, index: i32) -> i32 {
        if index < 0 {
            return 0;
        };
        match &self.front {
            Some(f) => f[index as usize].flags as i32,
            None => 0,
        }
    }

    /// `CCollision::GetIndex(int Nx, int Ny)`.
    pub fn get_index(&self, nx: i32, ny: i32) -> i32 {
        self.game[(ny * self.width + nx) as usize].index as i32
    }

    /// `CCollision::GetFrontIndex(int Nx, int Ny)`.
    pub fn get_front_index(&self, nx: i32, ny: i32) -> i32 {
        match &self.front {
            Some(f) => f[(ny * self.width + nx) as usize].index as i32,
            None => 0,
        }
    }

    /// `CCollision::GetPos(int Index)`.
    pub fn get_pos(&self, index: i32) -> Vec2<R> {
        if index < 0 {
            return Vec2::zero();
        }
        let x = index % self.width;
        let y = index / self.width;
        Vec2::new(R::from_i32(x * 32 + 16), R::from_i32(y * 32 + 16))
    }

    /// `CCollision::GetDoorTile(int Index, CDoorTile *pDoorTile)`.
    pub fn get_door_tile(&self, index: i32) -> DoorTile {
        match &self.door {
            Some(doors) if index >= 0 && doors[index as usize].index != 0 => doors[index as usize],
            _ => DoorTile::default(),
        }
    }

    /// `CCollision::SetCollisionAt(float x, float y, int Index)`.
    pub fn set_collision_at(&mut self, x: R, y: R, index: u8) {
        let nx = (vmath::round_to_int(x) / 32).clamp(0, self.width - 1);
        let ny = (vmath::round_to_int(y) / 32).clamp(0, self.height - 1);
        let i = (ny * self.width + nx) as usize;
        self.game[i].index = index;
        // Keep the `solid` cache (see its doc comment) in sync — this is the only method that
        // ever mutates `game[..].index` after construction.
        self.solid[i] = index == map::TILE_SOLID || index == map::TILE_NOHOOK;
    }

    /// `CCollision::SetDoorCollisionAt(float x, float y, unsigned char Type, unsigned char
    /// Flags, unsigned char Number)`. A no-op if the map has no switch layer (`!m_pDoor`).
    pub fn set_door_collision_at(&mut self, x: R, y: R, kind: u8, flags: u8, number: u8) {
        let Some(door) = self.door.as_mut() else { return };
        let nx = (vmath::round_to_int(x) / 32).clamp(0, self.width - 1);
        let ny = (vmath::round_to_int(y) / 32).clamp(0, self.height - 1);
        let d = &mut door[(ny * self.width + nx) as usize];
        d.index = kind;
        d.flags = flags;
        d.number = number;
    }

    // --- `GetMoveRestrictions` --------------------------------------------------------------

    /// `CCollision::GetMoveRestrictions(CALLBACK_SWITCHACTIVE, void*, vec2, float, int)`. The
    /// callback (`Option<F>`, `F: Fn(u8) -> bool`, called with a door's switch number) replaces
    /// the C++ `CALLBACK_SWITCHACTIVE pfnSwitchActive, void *pUser` pair — see the module doc
    /// comment. `override_center_tile_index: Some(i)` mirrors `OverrideCenterTileIndex >= 0`.
    ///
    /// # Panics
    ///
    /// If `distance` is outside `0.0..=32.0` (matches the C++ `dbg_assert`).
    pub fn get_move_restrictions<F: Fn(u8) -> bool>(
        &self,
        switch_active: Option<F>,
        pos: Vec2<R>,
        distance: R,
        override_center_tile_index: Option<i32>,
    ) -> i32 {
        assert!(
            distance >= R::ZERO && distance <= R::from_i32(32),
            "invalid distance {distance:?}"
        );
        let directions: [Vec2<R>; NUM_MR_DIRS] = [
            Vec2::new(R::ZERO, R::ZERO),
            Vec2::new(R::ONE, R::ZERO),
            Vec2::new(R::ZERO, R::ONE),
            Vec2::new(-R::ONE, R::ZERO),
            Vec2::new(R::ZERO, -R::ONE),
        ];
        let mut restrictions = 0;
        for (d, dir) in directions.iter().enumerate() {
            let mod_pos = pos + *dir * distance;
            let mut mod_map_index = self.get_pure_map_index_vec(mod_pos) as i32;
            if d == MR_DIR_HERE
                && let Some(over) = override_center_tile_index
            {
                mod_map_index = over;
            }
            for front in [false, true] {
                let (tile, flags) = if !front {
                    (self.get_tile_index(mod_map_index), self.get_tile_flags(mod_map_index))
                } else {
                    (
                        self.get_front_tile_index(mod_map_index),
                        self.get_front_tile_flags(mod_map_index),
                    )
                };
                restrictions |= move_restrictions_for(d, tile as u8, flags as u8);
            }
            if let Some(active) = &switch_active {
                let door = self.get_door_tile(mod_map_index);
                if i32::from(door.number) <= self.highest_switch_number && active(door.number) {
                    restrictions |= move_restrictions_for(d, door.index, door.flags);
                }
            }
        }
        restrictions
    }

    /// The 2-argument overload (`GetMoveRestrictions(vec2, float)`): no switch callback.
    pub fn get_move_restrictions_simple(&self, pos: Vec2<R>, distance: R) -> i32 {
        self.get_move_restrictions::<fn(u8) -> bool>(None, pos, distance, None)
    }

    // --- Line intersection ------------------------------------------------------------------

    /// `CCollision::IntersectLine`'s three output values, always computed (the C++ out-params
    /// are all optional pointers; here every field is always populated — callers that don't need
    /// `before_collision`/`tele_nr` simply ignore them).
    pub fn intersect_line(&self, pos0: Vec2<R>, pos1: Vec2<R>) -> LineHit<R> {
        let distance = vmath::distance(pos0, pos1);
        let end = (distance + R::ONE).to_i32_trunc();
        let mut last = pos0;
        for i in 0..=end {
            let a = R::from_i32(i) / R::from_i32(end);
            let pos = vmath::mix(pos0, pos1, a);
            let (ix, iy) = (vmath::round_to_int(pos.x), vmath::round_to_int(pos.y));
            if self.is_solid(ix, iy) {
                return LineHit {
                    hit: self.get_tile(ix, iy),
                    collision: pos,
                    before_collision: last,
                };
            }
            last = pos;
        }
        LineHit {
            hit: 0,
            collision: pos1,
            before_collision: pos1,
        }
    }

    /// `ThroughOffset(vec2 Pos0, vec2 Pos1, int *pOffsetX, int *pOffsetY)`: the tile offset (in
    /// pixels, `±32` on one axis) in the direction of travel from `pos0` to `pos1`.
    pub fn through_offset(pos0: Vec2<R>, pos1: Vec2<R>) -> (i32, i32) {
        let x = pos0.x - pos1.x;
        let y = pos0.y - pos1.y;
        if x.abs() > y.abs() {
            if x < R::ZERO { (-32, 0) } else { (32, 0) }
        } else if y < R::ZERO {
            (0, -32)
        } else {
            (0, 32)
        }
    }

    /// `CCollision::IsThrough(int x, int y, int OffsetX, int OffsetY, vec2 Pos0, vec2 Pos1)`.
    pub fn is_through(&self, x: i32, y: i32, offset_x: i32, offset_y: i32, pos0: Vec2<R>, pos1: Vec2<R>) -> bool {
        let index = self.pure_map_index_from_ints(x, y);
        if let Some(front) = &self.front {
            let f = front[index];
            if f.index == map::TILE_THROUGH_ALL || f.index == map::TILE_THROUGH_CUT {
                return true;
            }
            if f.index == map::TILE_THROUGH_DIR
                && ((f.flags == map::ROTATION_0 && pos0.y > pos1.y)
                    || (f.flags == map::ROTATION_90 && pos0.x < pos1.x)
                    || (f.flags == map::ROTATION_180 && pos0.y < pos1.y)
                    || (f.flags == map::ROTATION_270 && pos0.x > pos1.x))
            {
                return true;
            }
        }
        let offset_index = self.pure_map_index_from_ints(x + offset_x, y + offset_y);
        self.game[offset_index].index == map::TILE_THROUGH
            || self
                .front
                .as_ref()
                .is_some_and(|f| f[offset_index].index == map::TILE_THROUGH)
    }

    /// `CCollision::IsHookBlocker(int x, int y, vec2 Pos0, vec2 Pos1)`.
    pub fn is_hook_blocker(&self, x: i32, y: i32, pos0: Vec2<R>, pos1: Vec2<R>) -> bool {
        let index = self.pure_map_index_from_ints(x, y);
        let game = self.game[index];
        if game.index == map::TILE_THROUGH_ALL
            || self
                .front
                .as_ref()
                .is_some_and(|f| f[index].index == map::TILE_THROUGH_ALL)
        {
            return true;
        }
        if game.index == map::TILE_THROUGH_DIR
            && ((game.flags == map::ROTATION_0 && pos0.y < pos1.y)
                || (game.flags == map::ROTATION_90 && pos0.x > pos1.x)
                || (game.flags == map::ROTATION_180 && pos0.y > pos1.y)
                || (game.flags == map::ROTATION_270 && pos0.x < pos1.x))
        {
            return true;
        }
        if let Some(front) = &self.front {
            let f = front[index];
            if f.index == map::TILE_THROUGH_DIR
                && ((f.flags == map::ROTATION_0 && pos0.y < pos1.y)
                    || (f.flags == map::ROTATION_90 && pos0.x > pos1.x)
                    || (f.flags == map::ROTATION_180 && pos0.y > pos1.y)
                    || (f.flags == map::ROTATION_270 && pos0.x < pos1.x))
            {
                return true;
            }
        }
        false
    }

    /// `CCollision::IntersectLineTeleHook`. `sv_old_teleport_hook` stands in for
    /// `g_Config.m_SvOldTeleportHook` — pass `false` to match Oracle A's zero-initialized config
    /// (see the module doc comment).
    pub fn intersect_line_tele_hook(&self, pos0: Vec2<R>, pos1: Vec2<R>, sv_old_teleport_hook: bool) -> HookHit<R> {
        let distance = vmath::distance(pos0, pos1);
        let end = (distance + R::ONE).to_i32_trunc();
        let mut last = pos0;
        let (dx, dy) = Self::through_offset(pos0, pos1);
        for i in 0..=end {
            let a = R::from_i32(i) / R::from_i32(end);
            let pos = vmath::mix(pos0, pos1, a);
            let (ix, iy) = (vmath::round_to_int(pos.x), vmath::round_to_int(pos.y));

            let index = self.get_pure_map_index_vec(pos) as i32;
            let tele_nr = if sv_old_teleport_hook {
                self.is_teleport(index)
            } else {
                self.is_teleport_hook(index)
            };
            if tele_nr != 0 {
                return HookHit {
                    hit: map::TILE_TELEINHOOK as i32,
                    collision: pos,
                    before_collision: last,
                    tele_nr,
                };
            }

            let mut hit = 0;
            if self.is_solid(ix, iy) {
                if !self.is_through(ix, iy, dx, dy, pos0, pos1) {
                    hit = self.get_tile(ix, iy);
                }
            } else if self.is_hook_blocker(ix, iy, pos0, pos1) {
                hit = map::TILE_NOHOOK as i32;
            }
            if hit != 0 {
                return HookHit {
                    hit,
                    collision: pos,
                    before_collision: last,
                    tele_nr: 0,
                };
            }
            last = pos;
        }
        HookHit {
            hit: 0,
            collision: pos1,
            before_collision: pos1,
            tele_nr: 0,
        }
    }

    /// `CCollision::IntersectLineTeleWeapon`. `sv_old_teleport_weapons` stands in for
    /// `g_Config.m_SvOldTeleportWeapons` (see the module doc comment).
    pub fn intersect_line_tele_weapon(
        &self,
        pos0: Vec2<R>,
        pos1: Vec2<R>,
        sv_old_teleport_weapons: bool,
    ) -> HookHit<R> {
        let distance = vmath::distance(pos0, pos1);
        let end = (distance + R::ONE).to_i32_trunc();
        let mut last = pos0;
        for i in 0..=end {
            let a = R::from_i32(i) / R::from_i32(end);
            let pos = vmath::mix(pos0, pos1, a);
            let (ix, iy) = (vmath::round_to_int(pos.x), vmath::round_to_int(pos.y));

            let index = self.get_pure_map_index_vec(pos) as i32;
            let tele_nr = if sv_old_teleport_weapons {
                self.is_teleport(index)
            } else {
                self.is_teleport_weapon(index)
            };
            if tele_nr != 0 {
                return HookHit {
                    hit: map::TILE_TELEINWEAPON as i32,
                    collision: pos,
                    before_collision: last,
                    tele_nr,
                };
            }

            if self.is_solid(ix, iy) {
                return HookHit {
                    hit: self.get_tile(ix, iy),
                    collision: pos,
                    before_collision: last,
                    tele_nr: 0,
                };
            }
            last = pos;
        }
        HookHit {
            hit: 0,
            collision: pos1,
            before_collision: pos1,
            tele_nr: 0,
        }
    }

    /// `CCollision::IntersectNoLaser`.
    pub fn intersect_no_laser(&self, pos0: Vec2<R>, pos1: Vec2<R>) -> LineHit<R> {
        let distance = vmath::distance(pos0, pos1);
        let mut last = pos0;
        // `const int DistanceRounded = std::ceil(Distance);`.
        let end = <f64 as Real>::to_i32_trunc(distance.to_f64().ceil());
        for i in 0..end {
            let a = R::from_i32(i) / distance;
            let pos = vmath::mix(pos0, pos1, a);
            let nx = (vmath::round_to_int(pos.x) / 32).clamp(0, self.width - 1);
            let ny = (vmath::round_to_int(pos.y) / 32).clamp(0, self.height - 1);
            let tile = self.get_index(nx, ny);
            let front_tile = self.get_front_index(nx, ny);
            if tile == map::TILE_SOLID as i32
                || tile == map::TILE_NOHOOK as i32
                || tile == map::TILE_NOLASER as i32
                || front_tile == map::TILE_NOLASER as i32
            {
                let hit = if front_tile == map::TILE_NOLASER as i32 {
                    self.get_front_collision_at(pos.x, pos.y)
                } else {
                    self.get_collision_at(pos.x, pos.y)
                };
                return LineHit {
                    hit,
                    collision: pos,
                    before_collision: last,
                };
            }
            last = pos;
        }
        LineHit {
            hit: 0,
            collision: pos1,
            before_collision: pos1,
        }
    }

    /// `CCollision::IntersectNoLaserNoWalls`.
    pub fn intersect_no_laser_no_walls(&self, pos0: Vec2<R>, pos1: Vec2<R>) -> LineHit<R> {
        let distance = vmath::distance(pos0, pos1);
        let mut last = pos0;
        let end = <f64 as Real>::to_i32_trunc(distance.to_f64().ceil());
        for i in 0..end {
            let a = R::from_i32(i) / distance;
            let pos = vmath::mix(pos0, pos1, a);
            let (ix, iy) = (vmath::round_to_int(pos.x), vmath::round_to_int(pos.y));
            let no_laser = self.is_no_laser(ix, iy);
            let front_no_laser = self.is_front_no_laser(ix, iy);
            if no_laser || front_no_laser {
                let hit = if no_laser {
                    self.get_collision_at(pos.x, pos.y)
                } else {
                    self.get_front_collision_at(pos.x, pos.y)
                };
                return LineHit {
                    hit,
                    collision: pos,
                    before_collision: last,
                };
            }
            last = pos;
        }
        LineHit {
            hit: 0,
            collision: pos1,
            before_collision: pos1,
        }
    }

    /// `CCollision::IntersectAir`. `-1` means "left the map" (matches the C++ magic return
    /// value), distinct from `0` ("no hit").
    pub fn intersect_air(&self, pos0: Vec2<R>, pos1: Vec2<R>) -> LineHit<R> {
        let distance = vmath::distance(pos0, pos1);
        let mut last = pos0;
        let end = <f64 as Real>::to_i32_trunc(distance.to_f64().ceil());
        for i in 0..end {
            let a = R::from_i32(i) / distance;
            let pos = vmath::mix(pos0, pos1, a);
            let (ix, iy) = (vmath::round_to_int(pos.x), vmath::round_to_int(pos.y));
            let tile = self.get_tile(ix, iy);
            let front_tile = self.get_front_tile(ix, iy);
            if self.is_solid(ix, iy) || (tile == 0 && front_tile == 0) {
                let hit = if tile == 0 && front_tile == 0 {
                    -1
                } else if tile == 0 {
                    tile
                } else {
                    front_tile
                };
                return LineHit {
                    hit,
                    collision: pos,
                    before_collision: last,
                };
            }
            last = pos;
        }
        LineHit {
            hit: 0,
            collision: pos1,
            before_collision: pos1,
        }
    }

    // --- Movement ----------------------------------------------------------------------------

    /// `CCollision::MovePoint(vec2*, vec2*, float, int*)`.
    pub fn move_point(&self, pos: Vec2<R>, vel: Vec2<R>, elasticity: R) -> (Vec2<R>, Vec2<R>, i32) {
        let mut bounces = 0;
        if self.check_point_vec(pos + vel) {
            let mut affected = 0;
            let mut out_vel = vel;
            if self.check_point(pos.x + vel.x, pos.y) {
                out_vel.x *= -elasticity;
                bounces += 1;
                affected += 1;
            }
            if self.check_point(pos.x, pos.y + vel.y) {
                out_vel.y *= -elasticity;
                bounces += 1;
                affected += 1;
            }
            if affected == 0 {
                out_vel.x *= -elasticity;
                out_vel.y *= -elasticity;
            }
            (pos, out_vel, bounces)
        } else {
            (pos + vel, vel, bounces)
        }
    }

    /// `CCollision::MoveBox(vec2*, vec2*, vec2, vec2, bool*)`. Returns `(new_pos, new_vel,
    /// grounded)` — `grounded` mirrors the C++ `bool *pGrounded` output param, always computed
    /// (the C++ caller may pass `nullptr` to skip it; here it's simply ignored if unwanted).
    pub fn move_box(&self, pos: Vec2<R>, vel: Vec2<R>, size: Vec2<R>, elasticity: Vec2<R>) -> (Vec2<R>, Vec2<R>, bool) {
        let mut pos = pos;
        let mut vel = vel;
        let mut grounded = false;

        let distance = vmath::length(vel);
        let max = distance.to_i32_trunc();

        if distance > R::from_f64(0.00001) {
            let fraction = R::ONE / R::from_i32(max + 1);
            let elasticity_x = elasticity.x.clamp(-R::ONE, R::ONE);
            let elasticity_y = elasticity.y.clamp(-R::ONE, R::ONE);

            for _ in 0..=max {
                if vel == Vec2::zero() {
                    break;
                }
                let mut new_pos = pos + vel * fraction;
                if new_pos == pos {
                    break;
                }
                if self.test_box(new_pos, size) {
                    let mut hits = 0;
                    if self.test_box(Vec2::new(pos.x, new_pos.y), size) {
                        if elasticity_y > R::ZERO && vel.y > R::ZERO {
                            grounded = true;
                        }
                        new_pos.y = pos.y;
                        vel.y *= -elasticity_y;
                        hits += 1;
                    }
                    if self.test_box(Vec2::new(new_pos.x, pos.y), size) {
                        new_pos.x = pos.x;
                        vel.x *= -elasticity_x;
                        hits += 1;
                    }
                    if hits == 0 {
                        if elasticity_y > R::ZERO && vel.y > R::ZERO {
                            grounded = true;
                        }
                        new_pos.y = pos.y;
                        vel.y *= -elasticity_y;
                        new_pos.x = pos.x;
                        vel.x *= -elasticity_x;
                    }
                }
                pos = new_pos;
            }
        }

        (pos, vel, grounded)
    }

    // --- Tile-type queries -------------------------------------------------------------------

    /// `CCollision::IsWallJump(int Index)`.
    pub fn is_wall_jump(&self, index: i32) -> bool {
        index >= 0 && self.game[index as usize].index == map::TILE_WALLJUMP
    }

    /// `CCollision::IsNoLaser(int x, int y)`.
    pub fn is_no_laser(&self, x: i32, y: i32) -> bool {
        self.get_tile(x, y) == map::TILE_NOLASER as i32
    }

    /// `CCollision::IsFrontNoLaser(int x, int y)`.
    pub fn is_front_no_laser(&self, x: i32, y: i32) -> bool {
        self.get_front_tile(x, y) == map::TILE_NOLASER as i32
    }

    /// `CCollision::IsTeleport(int Index)`: the tele number (`Number`, 1-based) if `Index` is a
    /// `TILE_TELEIN`, else `0`.
    pub fn is_teleport(&self, index: i32) -> i32 {
        let Some(tele) = &self.tele else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = tele[index as usize];
        if t.kind == map::TILE_TELEIN {
            i32::from(t.number)
        } else {
            0
        }
    }

    /// `CCollision::IsEvilTeleport(int Index)`: the tele number if `Index` is a
    /// `TILE_TELEINEVIL`, else `0`.
    pub fn is_evil_teleport(&self, index: i32) -> i32 {
        let Some(tele) = &self.tele else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = tele[index as usize];
        if t.kind == map::TILE_TELEINEVIL {
            i32::from(t.number)
        } else {
            0
        }
    }

    /// `CCollision::IsCheckTeleport(int Index)`.
    pub fn is_check_teleport(&self, index: i32) -> bool {
        let Some(tele) = &self.tele else { return false };
        index >= 0 && tele[index as usize].kind == map::TILE_TELECHECKIN
    }

    /// `CCollision::IsCheckEvilTeleport(int Index)`.
    pub fn is_check_evil_teleport(&self, index: i32) -> bool {
        let Some(tele) = &self.tele else { return false };
        index >= 0 && tele[index as usize].kind == map::TILE_TELECHECKINEVIL
    }

    /// `CCollision::IsTeleCheckpoint(int Index)`: the tele number if `Index` is a
    /// `TILE_TELECHECK`, else `0`.
    pub fn is_tele_checkpoint(&self, index: i32) -> i32 {
        let Some(tele) = &self.tele else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = tele[index as usize];
        if t.kind == map::TILE_TELECHECK {
            i32::from(t.number)
        } else {
            0
        }
    }

    /// `CCollision::IsTeleportWeapon(int Index)`: the tele number if `Index` is a
    /// `TILE_TELEINWEAPON`, else `0`.
    pub fn is_teleport_weapon(&self, index: i32) -> i32 {
        let Some(tele) = &self.tele else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = tele[index as usize];
        if t.kind == map::TILE_TELEINWEAPON {
            i32::from(t.number)
        } else {
            0
        }
    }

    /// `CCollision::IsTeleportHook(int Index)`: the tele number if `Index` is a
    /// `TILE_TELEINHOOK`, else `0`.
    pub fn is_teleport_hook(&self, index: i32) -> i32 {
        let Some(tele) = &self.tele else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = tele[index as usize];
        if t.kind == map::TILE_TELEINHOOK {
            i32::from(t.number)
        } else {
            0
        }
    }

    /// `CCollision::IsSpeedup(int Index)`.
    ///
    /// # Panics
    ///
    /// If `index < 0` (matches the C++ `dbg_assert(Index >= 0, ...)`).
    pub fn is_speedup(&self, index: i32) -> bool {
        assert!(index >= 0, "invalid speedup index {index}");
        self.speedup.as_ref().is_some_and(|s| s[index as usize].force > 0)
    }

    /// `CCollision::IsTune(int Index)`: the tune zone number if `Index` is in one, else `0`.
    pub fn is_tune(&self, index: i32) -> i32 {
        let Some(tune) = &self.tune else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = tune[index as usize];
        if t.kind != 0 { i32::from(t.number) } else { 0 }
    }

    /// `CCollision::GetSpeedup(int Index, vec2*, int*, int*, int*)`. `None` when the C++ version
    /// would leave every out-param untouched (`Index < 0` or no speedup layer).
    pub fn get_speedup(&self, index: i32) -> Option<SpeedupInfo<R>> {
        let speedup = self.speedup.as_ref()?;
        if index < 0 {
            return None;
        }
        let s = speedup[index as usize];
        let angle = R::from_i32(i32::from(s.angle)) * (R::PI / R::from_i32(180));
        Some(SpeedupInfo {
            dir: vmath::direction(angle),
            force: i32::from(s.force),
            max_speed: i32::from(s.max_speed),
            kind: i32::from(s.kind),
        })
    }

    /// `CCollision::GetSwitchType(int Index)`.
    pub fn get_switch_type(&self, index: i32) -> i32 {
        let Some(switch) = &self.switch else { return 0 };
        if index < 0 {
            return 0;
        }
        let t = switch[index as usize].kind;
        if t > 0 { i32::from(t) } else { 0 }
    }

    /// `CCollision::GetSwitchNumber(int Index)`.
    pub fn get_switch_number(&self, index: i32) -> i32 {
        let Some(switch) = &self.switch else { return 0 };
        if index < 0 {
            return 0;
        }
        let s = switch[index as usize];
        if s.kind > 0 && s.number > 0 && i32::from(s.number) <= self.highest_switch_number {
            i32::from(s.number)
        } else {
            0
        }
    }

    /// `CCollision::GetSwitchDelay(int Index)`.
    pub fn get_switch_delay(&self, index: i32) -> i32 {
        let Some(switch) = &self.switch else { return 0 };
        if index < 0 {
            return 0;
        }
        let s = switch[index as usize];
        if s.kind > 0 { i32::from(s.delay) } else { 0 }
    }

    /// `CCollision::MoverSpeed(int x, int y, vec2*)`: `(tile_index, target_speed)`, or `None` if
    /// `(x, y)`'s tile isn't `TILE_CP`/`TILE_CP_F` (matching the C++ `return 0` — distinguished
    /// from a genuine `TILE_CP` hit by `None` rather than an `Index` of `0`, which
    /// `TILE_CP`/`TILE_CP_F` never are).
    pub fn mover_speed(&self, x: i32, y: i32) -> Option<(i32, Vec2<R>)> {
        let nx = (x / 32).clamp(0, self.width - 1);
        let ny = (y / 32).clamp(0, self.height - 1);
        let tile = self.game[(ny * self.width + nx) as usize];
        if tile.index != map::TILE_CP && tile.index != map::TILE_CP_F {
            return None;
        }
        let mut target = match tile.flags {
            f if f == map::ROTATION_0 => Vec2::new(R::ZERO, R::from_i32(-4)),
            f if f == map::ROTATION_90 => Vec2::new(R::from_i32(4), R::ZERO),
            f if f == map::ROTATION_180 => Vec2::new(R::ZERO, R::from_i32(4)),
            f if f == map::ROTATION_270 => Vec2::new(R::from_i32(-4), R::ZERO),
            _ => Vec2::zero(),
        };
        if tile.index == map::TILE_CP_F {
            target *= R::from_i32(4);
        }
        Some((tile.index as i32, target))
    }

    // --- `TileExists`/`TileExistsNext`/`GetMapIndex`/`GetMapIndices` ------------------------

    /// `CCollision::TileExists(int Index)`.
    pub fn tile_exists(&self, index: i32) -> bool {
        if index < 0 {
            return false;
        }
        let idx = index as usize;
        let in_range = |i: u8| {
            (map::TILE_FREEZE..=map::TILE_TELE_LASER_DISABLE).contains(&i)
                || (map::TILE_LFREEZE..=map::TILE_LUNFREEZE).contains(&i)
        };
        if in_range(self.game[idx].index) {
            return true;
        }
        if let Some(front) = &self.front
            && in_range(front[idx].index)
        {
            return true;
        }
        if let Some(tele) = &self.tele {
            let t = tele[idx].kind;
            if t == map::TILE_TELEIN
                || t == map::TILE_TELEINEVIL
                || t == map::TILE_TELECHECKINEVIL
                || t == map::TILE_TELECHECK
                || t == map::TILE_TELECHECKIN
            {
                return true;
            }
        }
        if self.speedup.as_ref().is_some_and(|s| s[idx].force > 0) {
            return true;
        }
        if self.door.as_ref().is_some_and(|d| d[idx].index != 0) {
            return true;
        }
        if self.switch.as_ref().is_some_and(|s| s[idx].kind != 0) {
            return true;
        }
        if self.tune.as_ref().is_some_and(|t| t[idx].kind != 0) {
            return true;
        }
        self.tile_exists_next(index)
    }

    /// `CCollision::TileExistsNext(int Index)`. Deliberately keeps DDNet's known quirks (see
    /// `docs/research/ddnet-physics.md` §3, "TileExistsNext"):
    ///
    /// - the neighbor-index bounds checks are `> 0`, not `>= 0`/`< len` — e.g. at `Index == 1`,
    ///   `TileOnTheLeft` becomes `Index` (itself), not `0`, because `Index - 1 > 0` is `false`
    ///   for `Index - 1 == 0`;
    /// - the "below/above `TILE_STOPS`" branches (`collision.cpp:882` and its `m_pFront`/
    ///   `m_pDoor` copies) `&&` a boolean condition with `Flags | ROTATION_180 | ROTATION_0` —
    ///   `|` binds tighter than `&&` in C++, so this parses as `(bool) && (Flags | 3 | 0)`, and
    ///   since `ROTATION_180 == 3` that bitwise-OR is *always* nonzero (hence always "truthy"
    ///   when used as the right side of `&&`) regardless of `Flags` — so the `&&` never actually
    ///   gates on flags at all. Kept verbatim (as an always-nonzero computed value, not simplified
    ///   away) so this stays traceable to the C++ source; behaviorally it's exactly the same as
    ///   the `TileOnTheRight`/`TileOnTheLeft` branch two lines above it, which has no such gate.
    pub fn tile_exists_next(&self, index: i32) -> bool {
        if index < 0 {
            return false;
        }
        let n = self.width * self.height;
        let tile_on_the_left = if index - 1 > 0 { index - 1 } else { index };
        let tile_on_the_right = if index + 1 < n { index + 1 } else { index };
        let tile_below = if index + self.width < n {
            index + self.width
        } else {
            index
        };
        let tile_above = if index - self.width > 0 {
            index - self.width
        } else {
            index
        };
        let (l, r, b, a) = (
            tile_on_the_left as usize,
            tile_on_the_right as usize,
            tile_below as usize,
            tile_above as usize,
        );

        let g = &self.game;
        if (g[r].index == map::TILE_STOP && g[r].flags == map::ROTATION_270)
            || (g[l].index == map::TILE_STOP && g[l].flags == map::ROTATION_90)
        {
            return true;
        }
        if (g[b].index == map::TILE_STOP && g[b].flags == map::ROTATION_0)
            || (g[a].index == map::TILE_STOP && g[a].flags == map::ROTATION_180)
        {
            return true;
        }
        if g[r].index == map::TILE_STOPA
            || g[l].index == map::TILE_STOPA
            || (g[r].index == map::TILE_STOPS || g[l].index == map::TILE_STOPS)
        {
            return true;
        }
        let below_stops_gate = (i32::from(g[b].flags) | i32::from(map::ROTATION_180) | i32::from(map::ROTATION_0)) != 0;
        if g[b].index == map::TILE_STOPA
            || g[a].index == map::TILE_STOPA
            || ((g[b].index == map::TILE_STOPS || g[a].index == map::TILE_STOPS) && below_stops_gate)
        {
            return true;
        }
        if let Some(front) = &self.front {
            if (front[r].index == map::TILE_STOPA || front[l].index == map::TILE_STOPA)
                || (front[r].index == map::TILE_STOPS || front[l].index == map::TILE_STOPS)
            {
                return true;
            }
            let front_below_gate =
                (i32::from(front[b].flags) | i32::from(map::ROTATION_180) | i32::from(map::ROTATION_0)) != 0;
            if front[b].index == map::TILE_STOPA
                || front[a].index == map::TILE_STOPA
                || ((front[b].index == map::TILE_STOPS || front[a].index == map::TILE_STOPS) && front_below_gate)
            {
                return true;
            }
            if (front[r].index == map::TILE_STOP && front[r].flags == map::ROTATION_270)
                || (front[l].index == map::TILE_STOP && front[l].flags == map::ROTATION_90)
            {
                return true;
            }
            if (front[b].index == map::TILE_STOP && front[b].flags == map::ROTATION_0)
                || (front[a].index == map::TILE_STOP && front[a].flags == map::ROTATION_180)
            {
                return true;
            }
        }
        if let Some(door) = &self.door {
            if (door[r].index == map::TILE_STOPA || door[l].index == map::TILE_STOPA)
                || (door[r].index == map::TILE_STOPS || door[l].index == map::TILE_STOPS)
            {
                return true;
            }
            let door_below_gate =
                (i32::from(door[b].flags) | i32::from(map::ROTATION_180) | i32::from(map::ROTATION_0)) != 0;
            if door[b].index == map::TILE_STOPA
                || door[a].index == map::TILE_STOPA
                || ((door[b].index == map::TILE_STOPS || door[a].index == map::TILE_STOPS) && door_below_gate)
            {
                return true;
            }
            if (door[r].index == map::TILE_STOP && door[r].flags == map::ROTATION_270)
                || (door[l].index == map::TILE_STOP && door[l].flags == map::ROTATION_90)
            {
                return true;
            }
            if (door[b].index == map::TILE_STOP && door[b].flags == map::ROTATION_0)
                || (door[a].index == map::TILE_STOP && door[a].flags == map::ROTATION_180)
            {
                return true;
            }
        }
        false
    }

    /// `CCollision::GetMapIndex(vec2 Pos)`: `-1` if `TileExists` is false at `Pos`'s tile.
    /// Truncates (not `round_to_int`) — matches `(int)Pos.x`/`(int)Pos.y` in the C++ source.
    pub fn get_map_index(&self, pos: Vec2<R>) -> i32 {
        let nx = (pos.x.to_i32_trunc() / 32).clamp(0, self.width - 1);
        let ny = (pos.y.to_i32_trunc() / 32).clamp(0, self.height - 1);
        let index = ny * self.width + nx;
        if self.tile_exists(index) { index } else { -1 }
    }

    /// `CCollision::GetMapIndices(vec2 PrevPos, vec2 Pos, unsigned MaxIndices)`.
    pub fn get_map_indices(&self, prev_pos: Vec2<R>, pos: Vec2<R>, max_indices: usize) -> Vec<i32> {
        let d = vmath::distance(prev_pos, pos);
        let mut out = Vec::new();
        if d == R::ZERO {
            let nx = (pos.x.to_i32_trunc() / 32).clamp(0, self.width - 1);
            let ny = (pos.y.to_i32_trunc() / 32).clamp(0, self.height - 1);
            let index = ny * self.width + nx;
            if self.tile_exists(index) {
                out.push(index);
            }
            return out;
        }
        let end = (d + R::ONE).to_i32_trunc();
        let mut last_index = 0;
        for i in 0..end {
            let a = R::from_i32(i) / d;
            let tmp = vmath::mix(prev_pos, pos, a);
            let nx = (tmp.x.to_i32_trunc() / 32).clamp(0, self.width - 1);
            let ny = (tmp.y.to_i32_trunc() / 32).clamp(0, self.height - 1);
            let index = ny * self.width + nx;
            if self.tile_exists(index) && last_index != index {
                if max_indices != 0 && out.len() > max_indices {
                    return out;
                }
                out.push(index);
                last_index = index;
            }
        }
        out
    }

    /// `CCollision::GetIndex(vec2 PrevPos, vec2 Pos)`: the map index of the first tele/speedup
    /// cell along the segment, or `-1`.
    pub fn get_index_along(&self, prev_pos: Vec2<R>, pos: Vec2<R>) -> i32 {
        let distance = vmath::distance(prev_pos, pos);
        if distance == R::ZERO {
            let nx = (pos.x.to_i32_trunc() / 32).clamp(0, self.width - 1);
            let ny = (pos.y.to_i32_trunc() / 32).clamp(0, self.height - 1);
            let map_index = ny * self.width + nx;
            if self.tele.is_some() || self.speedup.as_ref().is_some_and(|s| s[map_index as usize].force > 0) {
                return map_index;
            }
        }
        let distance_rounded = <f64 as Real>::to_i32_trunc(distance.to_f64().ceil());
        for i in 0..distance_rounded {
            let a = R::from_i32(i) / distance;
            let tmp = vmath::mix(prev_pos, pos, a);
            let nx = (tmp.x.to_i32_trunc() / 32).clamp(0, self.width - 1);
            let ny = (tmp.y.to_i32_trunc() / 32).clamp(0, self.height - 1);
            let map_index = ny * self.width + nx;
            if self.tele.is_some() || self.speedup.as_ref().is_some_and(|s| s[map_index as usize].force > 0) {
                return map_index;
            }
        }
        -1
    }

    // --- Tele in/out/checkpoint tables -------------------------------------------------------

    /// `CCollision::TeleIns(int Number)`.
    pub fn tele_ins(&self, number: u8) -> &[Vec2<R>] {
        self.tele_ins.get(&number).map_or(&[], Vec::as_slice)
    }
    /// `CCollision::TeleOuts(int Number)`.
    pub fn tele_outs(&self, number: u8) -> &[Vec2<R>] {
        self.tele_outs.get(&number).map_or(&[], Vec::as_slice)
    }
    /// `CCollision::TeleCheckOuts(int Number)`.
    pub fn tele_check_outs(&self, number: u8) -> &[Vec2<R>] {
        self.tele_check_outs.get(&number).map_or(&[], Vec::as_slice)
    }
    /// `CCollision::TeleOthers(int Number)`.
    pub fn tele_others(&self, number: u8) -> &[Vec2<R>] {
        self.tele_others.get(&number).map_or(&[], Vec::as_slice)
    }

    /// `CCollision::HasHookTeleIns()`.
    pub fn has_hook_tele_ins(&self, sv_old_teleport_hook: bool) -> bool {
        self.has_hook_tele_ins || (sv_old_teleport_hook && !self.tele_ins.is_empty())
    }

    /// `CCollision::TeleAllGet(int Number, size_t Offset)`.
    pub fn tele_all_get(&self, number: u8, mut offset: usize) -> Vec2<R> {
        if let Some(v) = self.tele_ins.get(&number) {
            if offset < v.len() {
                return v[offset];
            }
            offset -= v.len();
        }
        if let Some(v) = self.tele_outs.get(&number) {
            if offset < v.len() {
                return v[offset];
            }
            offset -= v.len();
        }
        if let Some(v) = self.tele_check_outs.get(&number) {
            if offset < v.len() {
                return v[offset];
            }
            offset -= v.len();
        }
        if let Some(v) = self.tele_others.get(&number)
            && offset < v.len()
        {
            return v[offset];
        }
        Vec2::new(-R::ONE, -R::ONE)
    }

    /// `CCollision::TeleAllSize(int Number)`.
    pub fn tele_all_size(&self, number: u8) -> usize {
        self.tele_ins.get(&number).map_or(0, Vec::len)
            + self.tele_outs.get(&number).map_or(0, Vec::len)
            + self.tele_check_outs.get(&number).map_or(0, Vec::len)
            + self.tele_others.get(&number).map_or(0, Vec::len)
    }

    // --- Misc --------------------------------------------------------------------------------

    /// `CCollision::IsTimeCheckpoint(int Index)`.
    pub fn is_time_checkpoint(&self, index: i32) -> i32 {
        if index < 0 {
            return -1;
        }
        let z = self.game[index as usize].index;
        if (map::TILE_TIME_CHECKPOINT_FIRST..=map::TILE_TIME_CHECKPOINT_LAST).contains(&z) {
            i32::from(z - map::TILE_TIME_CHECKPOINT_FIRST)
        } else {
            -1
        }
    }

    /// `CCollision::IsFrontTimeCheckpoint(int Index)`.
    pub fn is_front_time_checkpoint(&self, index: i32) -> i32 {
        let Some(front) = &self.front else { return -1 };
        if index < 0 {
            return -1;
        }
        let z = front[index as usize].index;
        if (map::TILE_TIME_CHECKPOINT_FIRST..=map::TILE_TIME_CHECKPOINT_LAST).contains(&z) {
            i32::from(z - map::TILE_TIME_CHECKPOINT_FIRST)
        } else {
            -1
        }
    }

    /// `CCollision::Entity(int x, int y, int Layer)`.
    pub fn entity(&self, x: i32, y: i32, layer: MapLayer) -> i32 {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 0;
        }
        let index = (y * self.width + x) as usize;
        let raw: i32 = match layer {
            MapLayer::Game => i32::from(self.game[index].index),
            MapLayer::Front => self.front.as_ref().map_or(0, |f| i32::from(f[index].index)),
            MapLayer::Switch => self.switch.as_ref().map_or(0, |s| i32::from(s[index].kind)),
            MapLayer::Tele => self.tele.as_ref().map_or(0, |t| i32::from(t[index].kind)),
            MapLayer::Speedup => self.speedup.as_ref().map_or(0, |s| i32::from(s[index].kind)),
            MapLayer::Tune => self.tune.as_ref().map_or(0, |t| i32::from(t[index].kind)),
        };
        raw - ENTITY_OFFSET
    }
}

/// `ENTITY_OFFSET` (`mapitems.h`): `255 - 16 * 4`.
const ENTITY_OFFSET: i32 = 255 - 16 * 4;

/// Which layer [`Collision::entity`] reads (`LAYER_GAME`/`LAYER_FRONT`/... in `mapitems.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapLayer {
    /// `LAYER_GAME`.
    Game,
    /// `LAYER_FRONT`.
    Front,
    /// `LAYER_SWITCH`.
    Switch,
    /// `LAYER_TELE`.
    Tele,
    /// `LAYER_SPEEDUP`.
    Speedup,
    /// `LAYER_TUNE`.
    Tune,
}

/// `CCollision::IntersectLine`/`IntersectNoLaser`/`IntersectNoLaserNoWalls`/`IntersectAir`'s
/// output: the hit tile id (`0` = no hit; [`Collision::intersect_air`] also uses `-1`), the
/// collision point, and the point just before it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineHit<R: Real> {
    /// The hit tile id, or `0` (no hit).
    pub hit: i32,
    /// The point of collision.
    pub collision: Vec2<R>,
    /// The last point before the collision.
    pub before_collision: Vec2<R>,
}

/// [`Collision::intersect_line_tele_hook`]/[`Collision::intersect_line_tele_weapon`]'s output:
/// like [`LineHit`], plus the teleporter number hit through (`0` if none).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HookHit<R: Real> {
    /// The hit tile id, or `0` (no hit).
    pub hit: i32,
    /// The point of collision.
    pub collision: Vec2<R>,
    /// The last point before the collision.
    pub before_collision: Vec2<R>,
    /// The teleporter number hit through, or `0`.
    pub tele_nr: i32,
}

/// [`Collision::get_speedup`]'s output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedupInfo<R: Real> {
    /// Unit direction vector.
    pub dir: Vec2<R>,
    /// `m_Force`: acceleration applied per tick.
    pub force: i32,
    /// `m_MaxSpeed`: speed cap (`0` = tuning's default).
    pub max_speed: i32,
    /// `m_Type`: [`crate::map::TILE_SPEED_BOOST_OLD`] or [`crate::map::TILE_SPEED_BOOST`].
    pub kind: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{MapData, Tile};

    fn tile(index: u8) -> Tile {
        Tile {
            index,
            flags: 0,
            skip: 0,
            reserved: 0,
        }
    }

    fn tile_f(index: u8, flags: u8) -> Tile {
        Tile {
            index,
            flags,
            skip: 0,
            reserved: 0,
        }
    }

    /// A `w`x`h` map, solid border, air inside — enough for point/box queries near edges.
    fn bordered_map(w: i32, h: i32) -> MapData {
        let mut game = vec![tile(map::TILE_AIR); (w * h) as usize];
        for x in 0..w {
            game[x as usize] = tile(map::TILE_SOLID);
            game[((h - 1) * w + x) as usize] = tile(map::TILE_SOLID);
        }
        for y in 0..h {
            game[(y * w) as usize] = tile(map::TILE_SOLID);
            game[(y * w + (w - 1)) as usize] = tile(map::TILE_SOLID);
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

    fn set_tile(map: &mut MapData, x: i32, y: i32, t: Tile) {
        let w = map.width as i32;
        map.game[(y * w + x) as usize] = t;
    }

    #[test]
    fn empty_collision_is_defensively_inert() {
        let c = Collision::<f32>::empty();
        assert!(!c.is_solid(0, 0));
        assert!(!c.check_point(0.0, 0.0));
        assert_eq!(c.get_tile(0, 0), 0);
    }

    #[test]
    fn is_solid_true_for_solid_and_nohook_false_for_air() {
        let mut map = bordered_map(10, 10);
        set_tile(&mut map, 5, 5, tile(map::TILE_NOHOOK));
        let c: Collision<f32> = Collision::new(&map);
        assert!(c.is_solid(5, 0)); // border row, TILE_SOLID
        assert!(c.is_solid(5 * 32 + 10, 5 * 32 + 10));
        assert!(!c.is_solid(5 * 32 + 10, 6 * 32 + 10)); // air
    }

    #[test]
    fn get_tile_clamps_negative_and_out_of_bounds_coordinates() {
        let map = bordered_map(5, 5);
        let c: Collision<f32> = Collision::new(&map);
        // Deeply negative/huge coordinates must clamp to the border tile, never panic.
        assert_eq!(c.get_tile(-100_000, -100_000), map::TILE_SOLID as i32);
        assert_eq!(c.get_tile(1_000_000, 1_000_000), map::TILE_SOLID as i32);
    }

    #[test]
    fn check_point_uses_round_to_int() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        // Pixel (32*1 - 1)=31 is still tile 0 (border, solid); 32.4 rounds to 32 -> tile 1 (air).
        assert!(c.check_point(31.0, 32.4)); // tile x=0 (border)
        assert!(!c.check_point(48.0, 48.0)); // well inside, tile (1,1), air
    }

    #[test]
    fn test_box_detects_overlap_with_any_corner() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        let size = Vec2::new(28.0f32, 28.0f32);
        // Centered well inside an open tile: no overlap.
        assert!(!c.test_box(Vec2::new(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0), size));
        // Pushed against the left border wall: overlaps.
        assert!(c.test_box(Vec2::new(32.0 + 10.0, 5.0 * 32.0 + 16.0), size));
    }

    #[test]
    fn is_on_ground_true_just_above_a_solid_floor() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        let floor_top = 9.0 * 32.0; // first pixel row of the bottom border tile (row 9)
        // is_on_ground checks (pos.x ± size/2, pos.y + size/2 + 5); solve for pos.y so that
        // check point lands exactly on the floor's first pixel row.
        assert!(c.is_on_ground(Vec2::new(5.0 * 32.0, floor_top - 19.0), 28.0));
        assert!(!c.is_on_ground(Vec2::new(5.0 * 32.0, floor_top - 19.0 - 20.0), 28.0));
    }

    #[test]
    fn get_pure_map_index_rounds_before_dividing() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        // 31.6 rounds to 32 -> tile x=1, not x=0 (which plain truncating division would give).
        assert_eq!(c.get_pure_map_index(31.6, 0.0) % 10, 1);
    }

    #[test]
    fn get_map_index_truncates_not_rounds() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        // 31.9 truncates to 31 -> tile x=0 (border, TileExists false there since it's TILE_SOLID
        // with none of TileExists' special ranges) -> GetMapIndex returns -1.
        assert_eq!(c.get_map_index(Vec2::new(31.9, 48.0)), -1);
    }

    #[test]
    fn move_box_stops_at_a_wall_and_zeroes_velocity_component() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        let size = Vec2::new(28.0f32, 28.0f32);
        let pos = Vec2::new(48.0, 5.0 * 32.0 + 16.0); // near the left wall
        let vel = Vec2::new(-50.0, 0.0); // moving hard left into the wall
        let (new_pos, new_vel, grounded) = c.move_box(pos, vel, size, Vec2::new(0.0, 0.0));
        assert!(
            new_pos.x > 32.0 + 14.0 - 0.001,
            "must not penetrate the wall: {new_pos:?}"
        );
        assert_eq!(new_vel.x, 0.0, "elasticity 0 zeroes velocity on impact");
        assert!(!grounded, "hit a side wall, not the ground");
    }

    #[test]
    fn move_box_reports_grounded_only_with_positive_y_elasticity_and_downward_velocity() {
        let map = bordered_map(10, 10);
        let c: Collision<f32> = Collision::new(&map);
        let size = Vec2::new(28.0f32, 28.0f32);
        let pos = Vec2::new(5.0 * 32.0, 260.0); // box bottom edge 14px above the floor
        let vel = Vec2::new(0.0, 50.0); // falling, enough to reach it within this one call
        let (_, _, grounded) = c.move_box(pos, vel, size, Vec2::new(0.0, 0.5));
        assert!(grounded);
        let (_, _, grounded_zero_elasticity) = c.move_box(pos, vel, size, Vec2::new(0.0, 0.0));
        assert!(!grounded_zero_elasticity, "grounded requires ElasticityY > 0");
    }

    #[test]
    fn stoppers_block_the_expected_single_direction_in_every_rotation() {
        // TILE_STOP CANTMOVE mapping per rotation (collision.cpp GetMoveRestrictionsRaw).
        let cases = [
            (map::ROTATION_0, CANTMOVE_DOWN),
            (map::ROTATION_90, CANTMOVE_LEFT),
            (map::ROTATION_180, CANTMOVE_UP),
            (map::ROTATION_270, CANTMOVE_RIGHT),
        ];
        for (flags, expected) in cases {
            assert_eq!(move_restrictions_raw(map::TILE_STOP, flags), expected, "flags={flags}");
        }
    }

    #[test]
    fn stoppers_yflip_variants_map_to_the_mirrored_direction() {
        let cases = [
            (map::TILEFLAG_YFLIP ^ map::ROTATION_0, CANTMOVE_UP),
            (map::TILEFLAG_YFLIP ^ map::ROTATION_90, CANTMOVE_RIGHT),
            (map::TILEFLAG_YFLIP ^ map::ROTATION_180, CANTMOVE_DOWN),
            (map::TILEFLAG_YFLIP ^ map::ROTATION_270, CANTMOVE_LEFT),
        ];
        for (flags, expected) in cases {
            assert_eq!(move_restrictions_raw(map::TILE_STOP, flags), expected, "flags={flags}");
        }
    }

    #[test]
    fn stops_blocks_both_opposite_directions() {
        assert_eq!(
            move_restrictions_raw(map::TILE_STOPS, map::ROTATION_0),
            CANTMOVE_DOWN | CANTMOVE_UP
        );
        assert_eq!(
            move_restrictions_raw(map::TILE_STOPS, map::ROTATION_90),
            CANTMOVE_LEFT | CANTMOVE_RIGHT
        );
    }

    #[test]
    fn stopa_blocks_all_four_directions() {
        assert_eq!(
            move_restrictions_raw(map::TILE_STOPA, 0),
            CANTMOVE_LEFT | CANTMOVE_RIGHT | CANTMOVE_UP | CANTMOVE_DOWN
        );
    }

    #[test]
    fn get_move_restrictions_here_direction_keeps_one_way_stop_effect_on_top_of_it() {
        // "the one exception is one-way blockers, they can also block us from moving if we're on
        // top of them" (collision.cpp comment) — MR_DIR_HERE is NOT masked away for TILE_STOP.
        assert_eq!(
            move_restrictions_for(MR_DIR_HERE, map::TILE_STOP, map::ROTATION_0),
            CANTMOVE_DOWN
        );
    }

    #[test]
    fn get_move_restrictions_here_direction_masks_away_stops_and_stopa() {
        // Only TILE_STOP gets the MR_DIR_HERE exception; TILE_STOPS/TILE_STOPA are masked to 0
        // at the "here" direction (mask for MR_DIR_HERE is 0).
        assert_eq!(move_restrictions_for(MR_DIR_HERE, map::TILE_STOPS, map::ROTATION_0), 0);
        assert_eq!(move_restrictions_for(MR_DIR_HERE, map::TILE_STOPA, 0), 0);
    }

    #[test]
    fn get_move_restrictions_reads_stoppers_around_the_position() {
        let mut map = bordered_map(10, 10);
        // Put a ROTATION_0 TILE_STOP directly below tile (5,5) -> approaching from above should
        // register CANTMOVE_DOWN.
        set_tile(&mut map, 5, 6, tile_f(map::TILE_STOP, map::ROTATION_0));
        let c: Collision<f32> = Collision::new(&map);
        let pos = Vec2::new(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        let restrictions = c.get_move_restrictions_simple(pos, 18.0);
        assert_ne!(
            restrictions & CANTMOVE_DOWN,
            0,
            "expected CANTMOVE_DOWN from the stopper below"
        );
    }

    #[test]
    fn tile_exists_next_off_by_one_quirk_at_index_one() {
        // docs/research/ddnet-physics.md §3: `Index - 1 > 0` (not `>= 0`) means at Index==1,
        // "the tile on the left" resolves to Index itself (1), not 0.
        let mut map = bordered_map(10, 10);
        // A ROTATION_270 TILE_STOP at index 0 would (if TileOnTheLeft were computed as 0)
        // trigger the first TileExistsNext branch at Index==1; because of the quirk it must not.
        set_tile(&mut map, 0, 0, tile_f(map::TILE_STOP, map::ROTATION_90));
        let c: Collision<f32> = Collision::new(&map);
        assert!(
            !c.tile_exists_next(1),
            "the off-by-one quirk must make index 0 unreachable as 'left' from index 1"
        );
    }

    #[test]
    fn tile_exists_next_detects_a_stop_to_the_right() {
        let mut map = bordered_map(10, 10);
        set_tile(&mut map, 6, 5, tile_f(map::TILE_STOP, map::ROTATION_270));
        let c: Collision<f32> = Collision::new(&map);
        let index = 5 * 10 + 5;
        assert!(c.tile_exists_next(index));
    }

    #[test]
    fn tile_exists_next_below_stops_quirk_ignores_flags() {
        // The buggy `&&`/`|` precedence at collision.cpp:882 means ANY flags value on a
        // below/above TILE_STOPS still returns true — even a flags byte matching neither
        // ROTATION_180 nor ROTATION_0 (values that would sensibly gate a "correct" version).
        let mut map = bordered_map(10, 10);
        set_tile(&mut map, 5, 6, tile_f(map::TILE_STOPS, map::ROTATION_90)); // "wrong" axis flags
        let c: Collision<f32> = Collision::new(&map);
        let index = 5 * 10 + 5;
        assert!(
            c.tile_exists_next(index),
            "the quirk means flags never actually gate this branch"
        );
    }

    #[test]
    fn tele_tables_are_populated_by_number_and_type() {
        let mut map = bordered_map(10, 10);
        map.tele = Some(vec![crate::map::TeleTile::default(); 100]);
        {
            let tele = map.tele.as_mut().unwrap();
            tele[12] = crate::map::TeleTile {
                number: 1,
                kind: map::TILE_TELEIN,
            };
            tele[13] = crate::map::TeleTile {
                number: 1,
                kind: map::TILE_TELEOUT,
            };
            tele[14] = crate::map::TeleTile {
                number: 1,
                kind: map::TILE_TELEOUT,
            };
        }
        let c: Collision<f32> = Collision::new(&map);
        assert_eq!(c.tele_ins(0).len(), 1);
        assert_eq!(c.tele_outs(0).len(), 2);
        assert_eq!(c.tele_all_size(0), 3);
        assert_eq!(c.tele_all_get(0, 0), c.tele_ins(0)[0]);
        assert_eq!(c.tele_all_get(0, 1), c.tele_outs(0)[0]);
        assert_eq!(c.tele_all_get(0, 5), Vec2::new(-1.0, -1.0));
    }

    #[test]
    fn hook_through_directions_respect_travel_direction() {
        let mut map = bordered_map(10, 10);
        set_tile(&mut map, 5, 5, tile(map::TILE_SOLID));
        map.front = Some(vec![tile(map::TILE_AIR); 100]);
        map.front.as_mut().unwrap()[5 * 10 + 5] = tile_f(map::TILE_THROUGH_DIR, map::ROTATION_0);
        let c: Collision<f32> = Collision::new(&map);
        // IsThrough/IsHookBlocker take *pixel* coordinates, not tile coordinates.
        let (cx, cy) = (5 * 32 + 16, 5 * 32 + 16);
        let cell = Vec2::new(cx as f32, cy as f32);
        // `collision.cpp`: `ROTATION_0 && Pos0.y > Pos1.y` — `Pos0.y > Pos1.y` means travelling
        // from a larger y to a smaller one, i.e. *upward* (y grows downward in DDNet's coordinate
        // system) — so ROTATION_0 lets an upward-travelling hook through, not a downward one.
        let from_below = Vec2::new(cell.x, cell.y + 100.0); // pos0.y > cell.y -> travelling up
        let from_above = Vec2::new(cell.x, cell.y - 100.0); // pos0.y < cell.y -> travelling down
        assert!(
            c.is_through(cx, cy, 0, 32, from_below, cell),
            "ROTATION_0 lets an upward-travelling hook through"
        );
        assert!(
            !c.is_through(cx, cy, 0, -32, from_above, cell),
            "ROTATION_0 blocks a downward-travelling hook"
        );
    }

    #[test]
    fn is_hook_blocker_true_for_through_all_on_the_open_game_layer() {
        let mut map = bordered_map(10, 10);
        set_tile(&mut map, 5, 5, tile(map::TILE_THROUGH_ALL));
        let c: Collision<f32> = Collision::new(&map);
        let (cx, cy) = (5 * 32 + 16, 5 * 32 + 16);
        let p = Vec2::new(0.0f32, 0.0f32);
        assert!(c.is_hook_blocker(cx, cy, p, p));
    }

    #[test]
    fn intersect_line_tele_hook_finds_teleinhook_before_solid() {
        let mut map = bordered_map(10, 10);
        map.tele = Some(vec![crate::map::TeleTile::default(); 100]);
        map.tele.as_mut().unwrap()[5 * 10 + 5] = crate::map::TeleTile {
            number: 3,
            kind: map::TILE_TELEINHOOK,
        };
        let c: Collision<f32> = Collision::new(&map);
        let hit = c.intersect_line_tele_hook(
            Vec2::new(1.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0),
            Vec2::new(8.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0),
            false,
        );
        assert_eq!(hit.hit, map::TILE_TELEINHOOK as i32);
        assert_eq!(hit.tele_nr, 3);
    }

    #[test]
    fn negative_coordinates_never_panic_across_the_public_api() {
        let map = bordered_map(8, 8);
        let c: Collision<f32> = Collision::new(&map);
        for &(x, y) in &[(-1i32, -1i32), (-1000, 5), (5, -1000), (i32::MIN / 2, i32::MIN / 2)] {
            let _ = c.get_tile(x, y);
            let _ = c.is_solid(x, y);
            let _ = c.get_front_tile(x, y);
        }
        let neg = Vec2::new(-500.0f32, -500.0f32);
        let _ = c.check_point_vec(neg);
        let _ = c.get_map_index(neg);
        let _ = c.get_pure_map_index_vec(neg);
    }

    #[test]
    fn clamp_vel_zeroes_only_the_restricted_direction() {
        let v = Vec2::new(5.0f32, -5.0f32);
        assert_eq!(clamp_vel(CANTMOVE_RIGHT, v), Vec2::new(0.0, -5.0));
        assert_eq!(clamp_vel(CANTMOVE_UP, v), Vec2::new(5.0, 0.0));
        assert_eq!(clamp_vel(0, v), v);
    }
}
