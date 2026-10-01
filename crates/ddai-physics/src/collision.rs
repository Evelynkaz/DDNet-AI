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

/// Largest coordinate magnitude (px) the `intersect_line`/`move_box` early-outs will reason about
/// (2^22 px = 131072 tiles, far beyond any real map): past it they decline and the exact loop
/// runs. Keeps `to_i32_trunc` far from saturating (task 1.10b review R4) and keeps an `f32`
/// coordinate's own `ulp` (0.5 px at 2^23) well below the 1 px pads.
const EARLY_OUT_COORD_LIMIT: f64 = 4_194_304.0;

/// Builds [`Collision::solid_sat`] from `solid` — see that field's doc comment for the table's
/// exact layout. `width`/`height` `<= 0` (an [`Collision::empty`] map) yields an empty table.
fn build_solid_sat(solid: &[bool], width: i32, height: i32) -> Vec<u32> {
    if width <= 0 || height <= 0 {
        return Vec::new();
    }
    let w = width as usize;
    let h = height as usize;
    let stride = w + 1;
    let mut sat = vec![0u32; stride * (h + 1)];
    for y in 0..h {
        let mut row_sum: u32 = 0;
        for x in 0..w {
            row_sum += u32::from(solid[y * w + x]);
            sat[(y + 1) * stride + (x + 1)] = row_sum + sat[y * stride + (x + 1)];
        }
    }
    sat
}

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
    /// A summed-area table (2D prefix sum) over [`Collision::solid`] (task 1.10b, speed-up 2):
    /// `solid_sat[y * (width+1) + x]` (`0 <= x <= width`, `0 <= y <= height`, so this is
    /// `(width+1) * (height+1)` entries, one extra all-zero row/column as the base case) is the
    /// count of solid cells in the tile rectangle `[0, x) x [0, y)`. Lets
    /// [`Collision::solid_count_in_tile_rect`] answer "how many solid cells in this axis-aligned
    /// tile rectangle?" in `O(1)` (4 lookups, 3 additions) instead of `O(rectangle area)` —
    /// [`Collision::intersect_line`]'s own early-out (its doc comment) is built on this. Rebuilt
    /// in full whenever [`Collision::set_collision_at`] changes `solid` — a summed-area table
    /// doesn't support a cheap single-cell incremental update the way a per-cell cache does (every
    /// entry at or after the changed cell, in both directions, depends on it), and
    /// `set_collision_at` isn't called anywhere in this crate yet (Stage B), so this cost is
    /// currently theoretical, not a real per-tick or even per-call one.
    solid_sat: Vec<u32>,
    /// [`Collision::tile_exists`]'s answer for every cell, precomputed once in [`Collision::new`]
    /// from [`Collision::tile_exists_uncached`] instead of recomputed on every call — same
    /// reasoning as [`Collision::solid`] above, generalized to a function that reads up to 6
    /// layers (game/front/tele/speedup/switch/tune, plus `TileExistsNext`'s door/game/front
    /// neighbor checks) instead of 1. Task 1.10, acceptance criterion 2.
    tile_exists_cache: Vec<bool>,
    /// [`map::pickup_freeze_mask`]: tiles inside a heart pickup's freeze reach. Read only by
    /// [`Collision::pickup_freeze_at`] (tile-based helpers); the physics never looks at it.
    pickup_freeze: Vec<bool>,
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
            solid_sat: Vec::new(),
            tile_exists_cache: Vec::new(),
            pickup_freeze: Vec::new(),
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

        let solid: Vec<bool> = map
            .game
            .iter()
            .map(|t| t.index == map::TILE_SOLID || t.index == map::TILE_NOHOOK)
            .collect();
        let solid_sat = build_solid_sat(&solid, width, height);

        let mut result = Collision {
            width,
            height,
            solid,
            solid_sat,
            tile_exists_cache: Vec::new(),
            pickup_freeze: map::pickup_freeze_mask(map),
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
        };
        // Every layer `tile_exists_uncached` reads is already in place above, so it's safe to
        // evaluate now, once per cell, for the rest of this `Collision`'s life (see
        // `tile_exists_cache`'s own doc comment) — unless a caller goes on to place `CDoor`
        // collision afterward (`World::from_map` does, and calls
        // `recompute_tile_exists_cache` again once it's done — see that method's doc comment).
        result.recompute_tile_exists_cache();
        result
    }

    /// (Re)computes `tile_exists_cache` (the private field [`Collision::tile_exists`] reads) from
    /// every other layer, which must already
    /// be in their final state (idempotent — safe to call again after further mutation).
    /// [`Collision::new`] already calls this once; the only reason to call it again is
    /// `set_door_collision_at` (`CDoor` collision placement, `World::from_map`'s scan, task 1.6),
    /// the sole method that mutates a `Collision` after `new` returns it (see `World::collision`'s
    /// own doc comment) — `World::from_map` calls this once, after every fixture's door cells for
    /// the whole map are placed, before the result is ever read from or shared via `Arc`. A
    /// caller that builds a `Collision` directly and never touches `door` afterward (every test/
    /// bench in this crate but `World::from_map` itself) never needs to call this again.
    pub fn recompute_tile_exists_cache(&mut self) {
        let n = self.game.len();
        self.tile_exists_cache = (0..n as i32).map(|i| self.tile_exists_uncached(i)).collect();
    }

    /// The map's width, in tiles.
    pub fn width(&self) -> i32 {
        self.width
    }
    /// The map's height, in tiles.
    pub fn height(&self) -> i32 {
        self.height
    }

    /// `CCollision::m_HighestSwitchNumber` — task 1.6: [`crate::core::WorldCore::init_switchers`]
    /// (`CWorldCore::InitSwitchers`) always takes this exact value as its `HighestSwitchNumber`
    /// argument in the real server (`gameworld.cpp:47-51`, `CGameWorld::Init`).
    pub fn highest_switch_number(&self) -> i32 {
        self.highest_switch_number
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

    /// The count of solid cells in the *tile* rectangle `[x0, x1] x [y0, y1]` (both inclusive) —
    /// `O(1)` via [`Collision::solid_sat`]. `x0`/`y0`/`x1`/`y1` must already be clamped to
    /// `0..width`/`0..height` with `x0 <= x1` and `y0 <= y1` (the only caller,
    /// `pixel_box_is_solid_free`, guarantees both).
    fn solid_count_in_tile_rect(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> u32 {
        let stride = (self.width + 1) as usize;
        let at = |y: i32, x: i32| self.solid_sat[y as usize * stride + x as usize];
        at(y1 + 1, x1 + 1) + at(y0, x0) - at(y0, x1 + 1) - at(y1 + 1, x0)
    }

    /// `true` only if the pixel-space box `[min_x, max_x] x [min_y, max_y]` is *provably* free of
    /// solid cells: every tile the box touches (converted with the same truncating `/ 32` and
    /// clamp [`Collision::is_solid`] applies to a sampled integer pixel) has `solid == false`.
    /// This is the one place the `intersect_line`/`move_box` early-outs turn a float box into a
    /// tile rectangle, so it carries every "fall back to the exact path" guard (task 1.10b review
    /// R4): an empty `Collision`, a non-finite bound, a bound beyond `EARLY_OUT_COORD_LIMIT`
    /// (where `to_i32_trunc` could saturate to `i32::MIN` and *invert* the rectangle — an inverted
    /// rectangle has no cells, which would otherwise read as "free" — and where an `f32` pixel's
    /// own rounding is no longer negligible against the 1 px pads the callers use), and, as a
    /// belt-and-braces check, any inverted rectangle. All of those return `false` ("not free"),
    /// so the caller runs the exact loop.
    fn pixel_box_is_solid_free(&self, min_x: R, max_x: R, min_y: R, max_y: R) -> bool {
        let limit = R::from_f64(EARLY_OUT_COORD_LIMIT);
        let in_range = |v: R| v.is_finite() && v.abs() <= limit;
        if self.solid_sat.is_empty() || !(in_range(min_x) && in_range(max_x) && in_range(min_y) && in_range(max_y)) {
            return false;
        }
        let tx0 = (min_x.to_i32_trunc() / 32).clamp(0, self.width - 1);
        let tx1 = (max_x.to_i32_trunc() / 32).clamp(0, self.width - 1);
        let ty0 = (min_y.to_i32_trunc() / 32).clamp(0, self.height - 1);
        let ty1 = (max_y.to_i32_trunc() / 32).clamp(0, self.height - 1);
        if tx0 > tx1 || ty0 > ty1 {
            return false;
        }
        self.solid_count_in_tile_rect(tx0, ty0, tx1, ty1) == 0
    }

    /// `true` if the *tile* rectangle covering the segment `pos0..pos1`, padded by 1 px on every
    /// side, contains not a single solid cell — in which case nothing along the segment can be
    /// solid either, since every point `intersect_line`'s per-sample marching loop ever visits
    /// (`mix(pos0, pos1, a)` for `a` in `[0, 1]`) lies within `[min(pos0, pos1), max(pos0,
    /// pos1)]` on each axis (a convex combination; the single `mix` evaluation's own rounding is
    /// a few `ulp` of a coordinate below `EARLY_OUT_COORD_LIMIT`, i.e. far under a pixel), and the
    /// 1 px pad covers `round_to_int`'s up-to-0.5-px rounding on top of that. `false` if either
    /// endpoint isn't finite or is beyond the coordinate limit (see `pixel_box_is_solid_free`) —
    /// the caller must fall back to the exact loop then.
    fn segment_bbox_is_solid_free(&self, pos0: Vec2<R>, pos1: Vec2<R>) -> bool {
        let pad = R::ONE;
        self.pixel_box_is_solid_free(
            pos0.x.min(pos1.x) - pad,
            pos0.x.max(pos1.x) + pad,
            pos0.y.min(pos1.y) - pad,
            pos0.y.max(pos1.y) + pad,
        )
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
    /// [`Collision::is_hook_blocker`], which take already-(wrapping-)summed pixel coordinates.
    ///
    /// Task 1.3 review round 3, finding F7. `x`/`y` here are `ix + OffsetX`/`iy + OffsetY`
    /// computed with C++'s (wrapping, two's-complement) `int` arithmetic — see the call sites'
    /// `wrapping_add`. For every `v` with `|v| < 2^24` (`f32`'s exact-integer range),
    /// `round_to_int(R::from_i32(v))` is provably exactly `v` again (`from_i32` widens exactly,
    /// and `round_to_int` of an already-integer value adds/subtracts `0.5` then truncates,
    /// landing back on the same integer), so dividing `v` directly instead of taking the `int ->
    /// float -> round_to_int` round trip through [`Collision::get_pure_map_index`] (review round
    /// 2, finding F4) is a provable no-op — and is *not* a no-op once `|v| >= 2^24`: an `f32`
    /// cannot represent every such integer exactly, so `(float)v` can round to a different value
    /// than `v` itself, and C++ takes exactly that (potentially rounding) conversion on its way
    /// into `GetPureMapIndex(float, float)`. Concretely: `ix == i32::MIN`, `OffsetX == -32` wraps
    /// to `2147483616` (`i32`, positive) — a value the fast path's plain integer division would
    /// clamp to `width - 1`, while C++'s `(float)2147483616` rounds to exactly `2147483648.0f`
    /// (`2^31`), whose `round_to_int` is `i32::MIN` (see [`Real::to_i32_trunc`]'s doc comment),
    /// dividing (truncating toward zero) to a large *negative* number that clamps to column `0`
    /// instead. So for `|v| >= 2^24` this takes the exact same float round trip
    /// [`Collision::get_pure_map_index`] does, reproducing C++'s rounding (and its
    /// hardware-`i32::MIN`-on-overflow behavior) bit-for-bit instead of the (for this range,
    /// actually wrong) integer shortcut.
    fn pure_map_index_from_ints(&self, x: i32, y: i32) -> usize {
        const EXACT_LIMIT: i32 = 1 << 24;
        let nx = if (-EXACT_LIMIT..EXACT_LIMIT).contains(&x) {
            x / 32
        } else {
            vmath::round_to_int(R::from_i32(x)) / 32
        };
        let ny = if (-EXACT_LIMIT..EXACT_LIMIT).contains(&y) {
            y / 32
        } else {
            vmath::round_to_int(R::from_i32(y)) / 32
        };
        let nx = nx.clamp(0, self.width.max(1) - 1);
        let ny = ny.clamp(0, self.height.max(1) - 1);
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
        // Task 1.10b, speed-up 2: `solid_sat` (see its own doc comment on why a full rebuild,
        // not an incremental patch, is the right call here) must stay in sync with `solid` too.
        self.solid_sat = build_solid_sat(&self.solid, self.width, self.height);
        // Task 1.10b, finding F1: `game[i]` feeds both `tile_exists_uncached(i)` directly and
        // `tile_exists_next(j)` for every neighbor `j` of `i` — keep `tile_exists_cache` in sync
        // the same way, not just `solid`.
        self.refresh_tile_exists_cache_around(i as i32);
    }

    /// `CCollision::SetDoorCollisionAt(float x, float y, unsigned char Type, unsigned char
    /// Flags, unsigned char Number)`. A no-op if the map has no switch layer (`!m_pDoor`).
    pub fn set_door_collision_at(&mut self, x: R, y: R, kind: u8, flags: u8, number: u8) {
        let Some(door) = self.door.as_mut() else { return };
        let nx = (vmath::round_to_int(x) / 32).clamp(0, self.width - 1);
        let ny = (vmath::round_to_int(y) / 32).clamp(0, self.height - 1);
        let i = ny * self.width + nx;
        let d = &mut door[i as usize];
        d.index = kind;
        d.flags = flags;
        d.number = number;
        // Task 1.10b, finding F1: same reasoning as `set_collision_at` above, for the `door`
        // layer (`switch::place_door_collision`'s only mutation site, called from
        // `World::from_map`'s scan for every `CDoor` fixture — previously relied on that caller
        // remembering to call `recompute_tile_exists_cache()` again afterward; now this setter
        // keeps itself consistent regardless of caller).
        self.refresh_tile_exists_cache_around(i);
    }

    /// Recomputes `tile_exists_cache` at cell `index` and every cell whose own
    /// [`Collision::tile_exists_next`] neighbor set can include `index` — the minimal safe set to
    /// refresh after a single-cell `game`/`door` write ([`Collision::set_collision_at`]/
    /// [`Collision::set_door_collision_at`]), without a full `O(map size)` rebuild.
    ///
    /// Task 1.10b review R3: the `> 0` quirk in `tile_exists_next` (`index - 1 > 0`, not `>= 0`)
    /// belongs to the *reader* `j` deciding whether to look at `j - 1`, not to the cell being
    /// refreshed: reader `j = 1` does not read cell 0, but reader `j = 0` reads cells `1` and
    /// `width` (its `left`/`above` fall back to itself). So a write to cell `1` or `width` must
    /// refresh cell `0`, and the candidates `index - 1` / `index - width` are therefore admitted
    /// down to `>= 0`. Over-refreshing a cell that turns out not to depend on `index` is harmless
    /// (it recomputes to the same value); missing one is not.
    fn refresh_tile_exists_cache_around(&mut self, index: i32) {
        let n = self.game.len() as i32;
        if !(0..n).contains(&index) {
            return;
        }
        let width = self.width;
        let candidates = [
            Some(index),
            (index >= 1).then_some(index - 1),
            (index + 1 < n).then_some(index + 1),
            (index >= width).then_some(index - width),
            (index + width < n).then_some(index + width),
        ];
        for c in candidates.into_iter().flatten() {
            let v = self.tile_exists_uncached(c);
            self.tile_exists_cache[c as usize] = v;
        }
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
    ///
    /// Task 1.10b, speed-up 2: `segment_bbox_is_solid_free` (private) first — if the segment's
    /// own (padded) tile bounding box has zero solid cells, the loop below is certain to run to
    /// completion without ever finding one (see that method's doc comment for the exact argument),
    /// so this returns the same `{hit: 0, collision: pos1, before_collision: pos1}` that loop
    /// would eventually produce, without running it at all. Every input the loop *would* have
    /// produced a hit for still runs the loop exactly as before (this only ever substitutes for
    /// the "no hit" outcome, never changes which inputs hit or what they hit).
    pub fn intersect_line(&self, pos0: Vec2<R>, pos1: Vec2<R>) -> LineHit<R> {
        self.intersect_line_impl(pos0, pos1, true)
    }

    /// [`Collision::intersect_line`]'s body; `allow_early_out = false` is the pre-1.10b loop
    /// verbatim — the reference the differential tests compare the early-out against.
    fn intersect_line_impl(&self, pos0: Vec2<R>, pos1: Vec2<R>, allow_early_out: bool) -> LineHit<R> {
        if allow_early_out && self.segment_bbox_is_solid_free(pos0, pos1) {
            return LineHit {
                hit: 0,
                collision: pos1,
                before_collision: pos1,
            };
        }
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
        // `wrapping_add`, not `+`: C++'s `int` arithmetic here wraps on overflow (two's
        // complement, no UB in practice on this platform/toolchain), and `ix == i32::MIN`,
        // `OffsetX == -32` is exactly the case `pure_map_index_from_ints`'s doc comment (F7)
        // walks through — using plain `+` would panic in debug builds instead of reproducing
        // that wraparound.
        let offset_index = self.pure_map_index_from_ints(x.wrapping_add(offset_x), y.wrapping_add(offset_y));
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

    /// `true` if a `size`-sized box's *entire* swept path from `pos` to `pos + vel` (the full
    /// displacement [`Collision::move_box`]'s loop accumulates toward across its `steps`
    /// sub-steps) provably cannot touch a solid cell, so every `test_box` the loop would make
    /// returns `false` and can be skipped. Task 1.10b review R1: the pad is a *proven* bound on
    /// the loop's float drift, not "negligible at this scale" (it is not: see below).
    ///
    /// **Proof, per axis** (`p` = start, `v` = velocity, `n = steps` = `max + 1` iterations,
    /// `S = |p| + |v| + 1`, `EPS = 2^-23`, `u = EPS/2` an upper bound on the unit roundoff of
    /// `f32` *and* `f64`). The loop computes `f = fl(1/n)`, `d = fl(v*f)`, then
    /// `x_{k+1} = fl(x_k + d)`; the exact path it approximates is `P_k = p + k*v/n`, which stays
    /// between `p` and `p + v` (a convex combination), so every `test_box` corner
    /// (`x_k ± half`, then `round_to_int`, i.e. at most 0.5 px more) lies within
    /// `[min(p, p+v) - half - B - 1, max(p, p+v) + half + B + 1]` **if** `|x_k - P_k| <= B`.
    /// (a) `f` and `d` each carry a relative error `<= u`, so `|d - v/n| <= 3u|v|/n` and, summed
    ///     over the `n` steps, `<= 3u|v|`.
    /// (b) Each addition rounds by `<= u|x_{k+1}|`; by induction `|x_k| <= |P_k| + B <= 2S`
    ///     whenever `B <= S`, so all `n` additions contribute `<= n * u * 2S = n * EPS * S`.
    /// (c) `end = pos + vel` (used for the box extent) differs from the exact `p + v` by
    ///     `<= u|end| <= EPS * S / 2`, and the corner arithmetic (`x ± half`) by `<= u|x|`.
    /// Total `<= (n + 3) * EPS * S`, and we use `B = (n + 4) * EPS * S`, the extra `EPS * S`
    /// absorbing the rounding in evaluating `B` itself. The induction in (b) needs `B <= S`, i.e.
    /// `(n + 4) * EPS <= 1`; otherwise this returns `false` (the exact loop runs). At real scales
    /// `B` is tiny (`n = 31`, `|p| = 2000` gives ~0.008 px), but it is **not** negligible in
    /// general: on the non-principal axis, with `|x| >= 16384` and `n > 512` (a fast fall past a
    /// far-right wall) it exceeds 0.5 px — the review's Oracle A counterexample. Also `false` if
    /// any input is non-finite or beyond `EARLY_OUT_COORD_LIMIT`.
    ///
    /// Why one decision up front stays valid for the whole loop: `vel` is only written inside the
    /// collision branch, which this fact makes dead, so `vel` (and therefore `d` and every path
    /// above) never changes mid-loop.
    fn swept_box_is_solid_free(&self, pos: Vec2<R>, vel: Vec2<R>, size: Vec2<R>, steps: i32) -> bool {
        if !(pos.x.is_finite() && pos.y.is_finite() && vel.x.is_finite() && vel.y.is_finite()) || steps < 1 {
            return false;
        }
        let eps = R::from_f64(1.1920928955078125e-7); // 2^-23, exact in f32 and f64
        let n_plus_4 = R::from_i32(steps) + R::from_f64(4.0);
        if n_plus_4 * eps > R::ONE {
            return false;
        }
        let half = size * R::from_f64(0.5);
        let end = pos + vel;
        // Per-axis drift bound `B = (n + 4) * EPS * S`.
        let drift = |p: R, v: R| n_plus_4 * eps * (p.abs() + v.abs() + R::ONE);
        let pad_x = half.x + R::ONE + drift(pos.x, vel.x);
        let pad_y = half.y + R::ONE + drift(pos.y, vel.y);
        self.pixel_box_is_solid_free(
            pos.x.min(end.x) - pad_x,
            pos.x.max(end.x) + pad_x,
            pos.y.min(end.y) - pad_y,
            pos.y.max(end.y) + pad_y,
        )
    }

    /// `CCollision::MoveBox(vec2*, vec2*, vec2, vec2, bool*)`. Returns `(new_pos, new_vel,
    /// grounded)` — `grounded` mirrors the C++ `bool *pGrounded` output param, always computed
    /// (the C++ caller may pass `nullptr` to skip it; here it's simply ignored if unwanted).
    ///
    /// Task 1.10b, speed-up 3: `swept_box_is_solid_free` (private) computed once, before the
    /// loop — when it holds, every `test_box` call below is skipped (never even evaluated), but
    /// the float accumulation loop itself (the `pos = new_pos` steps, the `vel ==
    /// Vec2::zero()`/`new_pos == pos` early exits) runs exactly as it otherwise would; this is
    /// deliberately *not* collapsed to a single `pos + vel` — see `swept_box_is_solid_free`'s own
    /// doc comment for why skipping only the collision queries (not the loop shape) is what stays
    /// bit-exact.
    pub fn move_box(&self, pos: Vec2<R>, vel: Vec2<R>, size: Vec2<R>, elasticity: Vec2<R>) -> (Vec2<R>, Vec2<R>, bool) {
        self.move_box_impl(pos, vel, size, elasticity, true)
    }

    /// [`Collision::move_box`]'s body; `allow_skip = false` is the pre-1.10b loop verbatim (every
    /// sub-step runs its `test_box` queries) — the reference the differential tests compare the
    /// `swept_box_is_solid_free` skip against.
    fn move_box_impl(
        &self,
        pos: Vec2<R>,
        vel: Vec2<R>,
        size: Vec2<R>,
        elasticity: Vec2<R>,
        allow_skip: bool,
    ) -> (Vec2<R>, Vec2<R>, bool) {
        let mut pos = pos;
        let mut vel = vel;
        let mut grounded = false;

        let distance = vmath::length(vel);
        let max = distance.to_i32_trunc();

        if distance > R::from_f64(0.00001) {
            let fraction = R::ONE / R::from_i32(max + 1);
            let elasticity_x = elasticity.x.clamp(-R::ONE, R::ONE);
            let elasticity_y = elasticity.y.clamp(-R::ONE, R::ONE);
            let skip_test_box = allow_skip && self.swept_box_is_solid_free(pos, vel, size, max.saturating_add(1));

            for _ in 0..=max {
                if vel == Vec2::zero() {
                    break;
                }
                let mut new_pos = pos + vel * fraction;
                if new_pos == pos {
                    break;
                }
                if !skip_test_box && self.test_box(new_pos, size) {
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

    /// Is the tile inside the freeze reach of a heart pickup ([`map::pickup_freeze_mask`])?
    pub fn pickup_freeze_at(&self, tx: i32, ty: i32) -> bool {
        tx >= 0
            && ty >= 0
            && tx < self.width
            && ty < self.height
            && self
                .pickup_freeze
                .get((ty * self.width + tx) as usize)
                .copied()
                .unwrap_or(false)
    }

    /// `CCollision::TileExists(int Index)`. Reads the private `tile_exists_cache` field
    /// (precomputed once, in [`Collision::new`], from `tile_exists_uncached` — see that field's
    /// doc comment) instead of recomputing the up-to-6-layer walk on every call. Task 1.10,
    /// acceptance criterion 2's "a single combined per-cell flags array for game + front + tele +
    /// speedup + switch + tune presence" — this is exactly that, plus `tile_exists_next`'s own
    /// (game/front/door) neighbor checks folded in too, since the whole function is pure given an
    /// immutable, already-constructed `Collision` (every layer it reads is frozen after `new`).
    /// Called every tick a character is on or passes over any such tile (`get_map_indices_into`'s
    /// doc comment) — measured effect: this task's `BUILD REPORT`.
    pub fn tile_exists(&self, index: i32) -> bool {
        if index < 0 {
            return false;
        }
        self.tile_exists_cache.get(index as usize).copied().unwrap_or(false)
    }

    /// [`Collision::tile_exists`]'s actual logic, run once per cell in [`Collision::new`] to fill
    /// [`Collision::tile_exists_cache`] — moved here verbatim (not simplified, not reordered) so
    /// it stays traceable to `CCollision::TileExists` 1:1; only *when* it runs changed (`new`
    /// makes `index` range over every cell up front instead of a caller triggering one evaluation
    /// per call, review round 2's finding F4 precedent for `is_solid`).
    fn tile_exists_uncached(&self, index: i32) -> bool {
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
    /// Writes into `out` (cleared first) instead of returning a freshly allocated `Vec` — zero
    /// heap allocations in steady state when `out` is a reused scratch buffer whose capacity has
    /// already grown to fit (acceptance criterion 1; review round 2, finding F9: called from
    /// [`crate::world::ddrace_post_core_tick`] every tick a character is on or passes over any
    /// `tile_exists` tile — freeze/speedup/stopper/tele/switch/kill/etc, i.e. most of a real
    /// map's gameplay-relevant tiles — not a rare event the way `can_spawn`'s own allocation is).
    pub fn get_map_indices_into(&self, prev_pos: Vec2<R>, pos: Vec2<R>, max_indices: usize, out: &mut Vec<i32>) {
        out.clear();
        let d = vmath::distance(prev_pos, pos);
        if d == R::ZERO {
            let nx = (pos.x.to_i32_trunc() / 32).clamp(0, self.width - 1);
            let ny = (pos.y.to_i32_trunc() / 32).clamp(0, self.height - 1);
            let index = ny * self.width + nx;
            if self.tile_exists(index) {
                out.push(index);
            }
            return;
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
                    return;
                }
                out.push(index);
                last_index = index;
            }
        }
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

    /// Task 1.10b, finding F1: `set_collision_at` (`CCollision::SetCollisionAt`) must keep
    /// `tile_exists_cache` in sync — both for the cell it directly writes, and for any neighbor
    /// whose own `tile_exists_next` result depends on that cell (mirrors
    /// `tile_exists_next_below_stops_quirk_ignores_flags` above, but going through
    /// `set_collision_at` *after* construction instead of the map's own initial tile data, and
    /// checking the cached `tile_exists` wrapper instead of `tile_exists_next` directly).
    #[test]
    fn set_collision_at_keeps_tile_exists_cache_in_sync() {
        let map = bordered_map(10, 10);
        let index = 5 * 10 + 5;

        // The mutated cell's own direct check (TILE_FREEZE is in `tile_exists_uncached`'s first
        // `in_range` check, no neighbor involved at all).
        let mut c: Collision<f32> = Collision::new(&map);
        assert!(
            !c.tile_exists(index),
            "sanity: a plain air cell must start with tile_exists == false"
        );
        let pos = Vec2::new(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        c.set_collision_at(pos.x, pos.y, map::TILE_FREEZE);
        assert!(
            c.tile_exists(index),
            "set_collision_at must refresh tile_exists_cache for the cell it just wrote"
        );

        // A neighbor's cache entry: `TILE_STOPS` directly below (5,5) makes `tile_exists_next`
        // (hence `tile_exists`) true for (5,5) itself via the "below STOPS" quirk (flags-
        // independent — see `tile_exists_next_below_stops_quirk_ignores_flags`), even though
        // (5,5) itself was never written.
        let mut c2: Collision<f32> = Collision::new(&map);
        assert!(
            !c2.tile_exists(index),
            "sanity: same starting point for the neighbor case"
        );
        let pos_below = Vec2::new(5.0 * 32.0 + 16.0, 6.0 * 32.0 + 16.0);
        c2.set_collision_at(pos_below.x, pos_below.y, map::TILE_STOPS);
        assert!(
            c2.tile_exists(index),
            "set_collision_at must also refresh the neighbor cell's cache entry, not just the \
             mutated cell's own"
        );
    }

    /// Task 1.10b, finding F1: same requirement as `set_collision_at_keeps_tile_exists_cache_in_sync`,
    /// for `set_door_collision_at` (`switch::place_door_collision`'s only mutation site) and the
    /// `door` layer.
    #[test]
    fn set_door_collision_at_keeps_tile_exists_cache_in_sync() {
        let mut map = bordered_map(10, 10);
        map.switch = Some(vec![crate::map::SwitchTile::default(); 100]);
        let index = 5 * 10 + 5;

        let mut c: Collision<f32> = Collision::new(&map);
        assert!(
            !c.tile_exists(index),
            "sanity: a plain air cell must start with tile_exists == false"
        );
        let pos = Vec2::new(5.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0);
        // Any nonzero `kind` makes `tile_exists_uncached`'s own `door[idx].index != 0` check true
        // — `TILE_STOPA` matches what `switch::place_door_collision` actually writes.
        c.set_door_collision_at(pos.x, pos.y, map::TILE_STOPA, 0, 1);
        assert!(
            c.tile_exists(index),
            "set_door_collision_at must refresh tile_exists_cache for the cell it just wrote"
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

    /// Task 1.3 review round 3, finding F7 (fixed in task 1.6): `is_through`'s second
    /// `pure_map_index_from_ints` call adds `OffsetX`/`OffsetY` to `x`/`y` in `int` arithmetic
    /// (matching C++) *before* dividing by 32 — at `x == i32::MIN`, `offset_x == -32` this wraps
    /// to `2147483616` (`i32`, positive). C++'s `(float)2147483616` rounds to exactly
    /// `2147483648.0f` (`2^31`, since a plain `int -> float` conversion there loses precision
    /// past `f32`'s 24-bit mantissa), whose `round_to_int` hits `i32::MIN` (hardware "integer
    /// indefinite"), landing on column `0` after the `/32` + clamp. The pre-fix "shortcut" (plain
    /// integer division of the wrapped `2147483616`) instead landed on column `width - 1` — the
    /// wrong column. This pins the *correct* (C++-matching) column-0 answer through the public
    /// `is_through` API (`pure_map_index_from_ints` itself is private).
    #[test]
    fn is_through_matches_cpp_int_overflow_at_i32_min_offset() {
        let w = 20i32;
        // A `TILE_THROUGH` marker at column 0 (row 5) in one map, and at column `w - 1` (same
        // row) in a separate, otherwise-identical map — only the column-0 map should report
        // `is_through` true for this `x == i32::MIN`, `offset_x == -32` case.
        let mut map_col0 = bordered_map(w, 10);
        set_tile(&mut map_col0, 0, 5, tile(map::TILE_THROUGH));
        let c0: Collision<f32> = Collision::new(&map_col0);

        let mut map_last = bordered_map(w, 10);
        set_tile(&mut map_last, w - 1, 5, tile(map::TILE_THROUGH));
        let c_last: Collision<f32> = Collision::new(&map_last);

        // `y` chosen so `y / 32` lands on tile row 5 regardless of the (irrelevant here) offset;
        // `pos0`/`pos1` are dummies (no `TILE_THROUGH_DIR`/`TILE_THROUGH_ALL` involved on either
        // side of this call, so the direction check inside `is_through` never triggers).
        let y = 5 * 32 + 16;
        let dummy = Vec2::new(0.0f32, 0.0f32);
        assert!(
            c0.is_through(i32::MIN, y, -32, 0, dummy, dummy),
            "C++'s int->float overflow at i32::MIN lands on column 0, not column width-1"
        );
        assert!(
            !c_last.is_through(i32::MIN, y, -32, 0, dummy, dummy),
            "the pre-fix shortcut would have wrongly matched the width-1 column here"
        );
    }

    // --- Task 1.10b review round 1 (R1/R3/R4): early-out exactness ----------------------------

    /// A `w`x`h` map: solid border on the left/top/right and from `floor_row` down, an optional
    /// solid wall column, air elsewhere (the review's `mkmap`).
    fn wall_map(w: i32, h: i32, floor_row: i32, wall_col: Option<i32>) -> MapData {
        let mut game = vec![tile(map::TILE_AIR); (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if x == 0 || x == w - 1 || y == 0 || y >= floor_row || Some(x) == wall_col {
                    game[(y * w + x) as usize] = tile(map::TILE_SOLID);
                }
            }
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

    type MoveOut = (Vec2<f32>, Vec2<f32>, bool);

    fn move_bits(o: MoveOut) -> [u32; 5] {
        [
            o.0.x.to_bits(),
            o.0.y.to_bits(),
            o.1.x.to_bits(),
            o.1.y.to_bits(),
            u32::from(o.2),
        ]
    }

    /// R1 regression, direct form: the review's Oracle A counterexample (800x1500 map, wall at
    /// x = 22400, fast fall with a small `vel.x`). The 1 px pad alone let the skip fire although the
    /// accumulated non-principal-axis drift (`~ (n+1) * ulp(22380)/2`, `n = 963`) carries the box
    /// into the wall; `move_box` must equal the no-skip loop bit for bit — and the no-skip loop
    /// must actually hit the wall here (so this cannot pass vacuously).
    #[test]
    fn move_box_skip_is_exact_for_the_review_r1_counterexample() {
        let map = wall_map(800, 1500, 1490, Some(700));
        let c: Collision<f32> = Collision::new(&map);
        let pos = Vec2::new(22380.0f32, 34092.0);
        let vel = Vec2::new(4.9f32, 962.10);
        let size = Vec2::new(28.0f32, 28.0);
        let elasticity = Vec2::new(0.0f32, 0.0);
        let exact = c.move_box_impl(pos, vel, size, elasticity, false);
        assert!(
            exact.1.x != vel.x,
            "sanity: the exact loop must hit the wall (vel.x zeroed): {exact:?}"
        );
        let fast = c.move_box(pos, vel, size, elasticity);
        assert_eq!(move_bits(fast), move_bits(exact), "skip {fast:?} vs exact {exact:?}");
        assert!(
            !c.swept_box_is_solid_free(pos, vel, size, 963),
            "the drift-widened swept box must reach the wall column here"
        );
    }

    /// xorshift64*: deterministic, dependency-free randomness for the differential tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Differential fuzz for the `move_box` skip: random positions (incl. far-right, fractional),
    /// speeds up to a few thousand px/tick with a small non-principal component (the drift case),
    /// sizes and elasticities; skip and no-skip must be bit-identical. Returns (cases, skips taken).
    fn fuzz_move_box(cases: u32, seed: u64) -> (u32, u32) {
        let maps = [
            (wall_map(800, 1500, 1490, Some(700)), Some(700)),
            (wall_map(800, 1500, 1490, None), None),
            (wall_map(60, 60, 55, Some(30)), Some(30)),
        ];
        let collisions: Vec<Collision<f32>> = maps.iter().map(|(m, _)| Collision::new(m)).collect();
        let mut rng = Rng(seed);
        let mut skips = 0;
        for k in 0..cases {
            let mi = (rng.next() % 3) as usize;
            let (m, wall) = &maps[mi];
            let c = &collisions[mi];
            let (w, h) = (m.width as f32 * 32.0, m.height as f32 * 32.0);
            let mut pos = Vec2::new(rng.range(40.0, w - 40.0), rng.range(40.0, h - 200.0));
            let mut vel = match rng.next() % 4 {
                0 => Vec2::new(rng.range(-20.0, 20.0), rng.range(-20.0, 20.0)),
                1 => Vec2::new(rng.range(-300.0, 300.0), rng.range(-300.0, 300.0)),
                2 => Vec2::new(rng.range(-3000.0, 3000.0), rng.range(-3000.0, 3000.0)),
                // the review's drift shape: fast on one axis, small (like 4.9) on the other
                _ => {
                    let (fast, slow) = (rng.range(-3000.0, 3000.0), rng.range(-12.0, 12.0));
                    if rng.next() & 1 == 0 {
                        Vec2::new(slow, fast)
                    } else {
                        Vec2::new(fast, slow)
                    }
                }
            };
            if let Some(col) = wall
                && rng.next().is_multiple_of(2)
            {
                // adversarial: hug the wall column from either side
                let wx = *col as f32 * 32.0;
                pos.x = if rng.next() & 1 == 0 {
                    wx - rng.range(0.0, 60.0)
                } else {
                    wx + 32.0 + rng.range(0.0, 60.0)
                };
            }
            if rng.next().is_multiple_of(3) {
                pos = Vec2::new(pos.x.round(), pos.y.round());
            }
            if rng.next().is_multiple_of(5) {
                vel = Vec2::new((vel.x * 256.0).round() / 256.0, (vel.y * 256.0).round() / 256.0);
            }
            let size = Vec2::new(28.0f32, if rng.next().is_multiple_of(4) { 40.0 } else { 28.0 });
            let elasticity = match rng.next() % 3 {
                0 => Vec2::new(0.0, 0.0),
                1 => Vec2::new(rng.range(-1.5, 1.5), rng.range(-1.5, 1.5)),
                _ => Vec2::new(0.5, 0.5),
            };
            let steps = vmath::length(vel).to_i32_trunc().saturating_add(1);
            if c.swept_box_is_solid_free(pos, vel, size, steps) {
                skips += 1;
            }
            let fast = c.move_box(pos, vel, size, elasticity);
            let exact = c.move_box_impl(pos, vel, size, elasticity, false);
            assert_eq!(
                move_bits(fast),
                move_bits(exact),
                "case {k} (seed {seed}): map {mi} pos {pos:?} vel {vel:?} size {size:?} elasticity {elasticity:?}: skip {fast:?} vs exact {exact:?}"
            );
        }
        (cases, skips)
    }

    #[test]
    fn move_box_skip_matches_no_skip_on_random_cases() {
        let (cases, skips) = fuzz_move_box(40_000, 0x5EED_0001);
        assert!(
            skips * 20 > cases,
            "the fuzz must actually exercise the skip ({skips}/{cases})"
        );
    }

    /// Long-running version of the fuzz above; run with `cargo test -p ddai-physics --release --
    /// --ignored move_box_skip_matches_no_skip_heavy_fuzz`.
    #[test]
    #[ignore]
    fn move_box_skip_matches_no_skip_heavy_fuzz() {
        for seed in 1..=8u64 {
            let (cases, skips) = fuzz_move_box(2_000_000, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            eprintln!("seed {seed}: {cases} cases, {skips} skipped, 0 divergences");
        }
    }

    /// Same idea for `intersect_line`: random segments (finite, up to +-3e9 px so past-2^31 and
    /// far-outside-the-map inputs are covered) must give identical `LineHit`s with and without
    /// the early-out.
    #[test]
    fn intersect_line_early_out_matches_exact_loop_on_random_segments() {
        let map = wall_map(800, 1500, 1490, Some(700));
        let c: Collision<f32> = Collision::new(&map);
        let mut rng = Rng(0x11AE_0002);
        let (mut free, total) = (0u32, 40_000u32);
        for k in 0..total {
            let scale = [50.0f32, 2000.0, 25_600.0, 3.0e9][(rng.next() % 4) as usize];
            let (w, h) = (800.0 * 32.0, 1500.0 * 32.0);
            let p0 = Vec2::new(rng.range(-0.1, 1.1) * w, rng.range(-0.1, 1.1) * h);
            let p1 = if scale > 1.0e6 {
                Vec2::new(rng.range(-scale, scale), rng.range(-scale, scale))
            } else {
                Vec2::new(p0.x + rng.range(-scale, scale), p0.y + rng.range(-scale, scale))
            };
            if c.segment_bbox_is_solid_free(p0, p1) {
                free += 1;
            }
            let fast = c.intersect_line(p0, p1);
            let exact = c.intersect_line_impl(p0, p1, false);
            assert_eq!(fast, exact, "case {k}: {p0:?} -> {p1:?}");
        }
        assert!(
            free * 20 > total,
            "the fuzz must actually exercise the early-out ({free}/{total})"
        );
    }

    /// R4 regression: past 2^31 `to_i32_trunc` saturates to `i32::MIN`, which used to invert the
    /// tile rectangle (0 cells counted -> "free") although the exact loop hits the solid last
    /// column immediately. Non-finite and beyond-limit inputs must decline the early-out.
    #[test]
    fn early_outs_decline_out_of_range_and_non_finite_coordinates() {
        let map = bordered_map(10, 10); // solid right border
        let c: Collision<f32> = Collision::new(&map);
        let (a, b) = (Vec2::new(2.1e9f32, 150.0), Vec2::new(2.2e9f32, 150.0));
        let hit = c.intersect_line(a, b);
        assert_ne!(hit.hit, 0, "the exact loop hits the clamped right border: {hit:?}");
        assert_eq!(hit, c.intersect_line_impl(a, b, false));
        assert!(!c.segment_bbox_is_solid_free(a, b));
        let inf = f32::INFINITY;
        for bad in [
            (Vec2::new(f32::NAN, 1.0), Vec2::new(2.0, 2.0)),
            (Vec2::new(1.0, 1.0), Vec2::new(inf, 2.0)),
            (Vec2::new(-5.0e6, 1.0), Vec2::new(2.0, 2.0)),
        ] {
            assert!(!c.segment_bbox_is_solid_free(bad.0, bad.1), "{bad:?}");
        }
        let size = Vec2::new(28.0f32, 28.0);
        assert!(!c.swept_box_is_solid_free(Vec2::new(2.1e9, 150.0), Vec2::new(10.0, 0.0), size, 11));
        assert!(!c.swept_box_is_solid_free(Vec2::new(100.0, 100.0), Vec2::new(f32::NAN, 0.0), size, 1));
        assert!(!c.swept_box_is_solid_free(Vec2::new(100.0, 100.0), Vec2::new(1.0, 0.0), size, 0));
        // a huge step count would break the drift proof's `(n + 4) * EPS <= 1` premise
        assert!(!c.swept_box_is_solid_free(Vec2::new(100.0, 100.0), Vec2::new(1.0, 0.0), size, 9_000_000));
        // and an empty collision never claims "free"
        let empty = Collision::<f32>::empty();
        assert!(!empty.segment_bbox_is_solid_free(Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0)));
    }

    /// R3 regression: `tile_exists_next(0)` reads cells `1` and `width` (its `left`/`above` fall
    /// back to itself, the documented `> 0` quirk), so a write to either must refresh cell 0's
    /// cache entry — the earlier `index - 1 > 0` candidate filter skipped it.
    #[test]
    fn writes_next_to_cell_zero_refresh_its_tile_exists_cache() {
        let mut map = bordered_map(10, 10);
        map.switch = Some(vec![crate::map::SwitchTile::default(); 100]);
        for (label, x, y, index) in [("index 1", 1, 0, 1), ("index width", 0, 1, 10)] {
            let mut c: Collision<f32> = Collision::new(&map);
            assert!(!c.tile_exists(0), "{label}: sanity");
            c.set_door_collision_at(x as f32 * 32.0 + 16.0, y as f32 * 32.0 + 16.0, map::TILE_STOPA, 0, 1);
            assert!(c.tile_exists(index), "{label}: the written cell itself");
            assert_eq!(
                c.tile_exists(0),
                c.tile_exists_uncached(0),
                "{label}: cache vs recompute for cell 0"
            );
            assert!(
                c.tile_exists(0),
                "{label}: cell 0 reads the STOPA next to it via `tile_exists_next`"
            );
            // and the same through `set_collision_at`
            let mut c2: Collision<f32> = Collision::new(&map);
            c2.set_collision_at(x as f32 * 32.0 + 16.0, y as f32 * 32.0 + 16.0, map::TILE_STOPA);
            assert_eq!(
                c2.tile_exists(0),
                c2.tile_exists_uncached(0),
                "{label}: set_collision_at"
            );
            assert!(c2.tile_exists(0), "{label}: set_collision_at");
        }
    }
}
