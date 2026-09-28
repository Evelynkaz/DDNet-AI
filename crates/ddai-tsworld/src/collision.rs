//! Literal port of `src/core/collision.ts` (TS, `Wranked1/DDNet-AI`, GPL-3.0). Also a zlib-noted
//! port of the original DDNet `CCollision` this TS file itself ported from (see `NOTICE`).
//!
//! # Deliberately preserved TS quirks (see crate README "Известные причуды TS" for the full list)
//!
//! `collision.ts` addresses a tile-grid cell from a pixel `Vec2` in **three different, not
//! interchangeable, ways**:
//!
//! 1. `indexAt` (private; backs `isSolid`/`isDeath`/`isFreeze`/`isUnFreeze`/`isNoHook`/
//!    `getCollisionAt`/`testBoxAt`/the `checkPoint` used by `moveBox`/`movePoint`/
//!    `intersectLine{,Hook}`): `roundToInt`-style half-up rounding to the nearest pixel, then an
//!    **arithmetic right shift by 5** (`>> 5`) to get the tile coordinate — floor division for
//!    negative coordinates.
//! 2. `getMapIndex`/`getMapIndices`/`teleOutsFor`'s collector (via `getMapIndex`'s twin logic
//!    inlined in `getMapIndices`): `Math.trunc(Math.trunc(pixel) / 32)` — truncating (round
//!    toward zero) division, applied to an already-truncated pixel coordinate.
//! 3. `teleAt`: `Math.trunc(roundToInt(pixel) / 32)` — half-up-rounded pixel, then truncating
//!    division.
//!
//! For non-negative, tile-aligned-ish coordinates all three agree; they can disagree by one tile
//! for negative or exact-boundary coordinates (floor vs. truncate-toward-zero are different
//! functions for negative integers). This is **not unified** here — each of [`Collision::index_at`],
//! [`Collision::get_map_index`]/[`Collision::get_map_indices`] and [`Collision::tele_at`] is its
//! own literal port, on purpose, per the task's "reproduce, don't fix" rule (`docs/DECISIONS.md`
//! D-035). In practice every caller in `src/plan`/`src/env` and every tee's `pos` stays within
//! `[0, width*32) x [0, height*32)` (the outer ring is always solid, see the synthetic recipes'
//! `solid_border`), so this divergence is inert for any trace this crate's corpus can produce —
//! documented, not exercised.

use crate::tuning::{
    CANTMOVE_DOWN, CANTMOVE_LEFT, CANTMOVE_RIGHT, CANTMOVE_UP, CFLAG_DEATH, CFLAG_NOHOOK, CFLAG_SOLID, ROTATION_0,
    ROTATION_90, ROTATION_180, ROTATION_270, TILE_AIR, TILE_DEATH, TILE_FREEZE, TILE_LFREEZE, TILE_LUNFREEZE,
    TILE_NOHOOK, TILE_SOLID, TILE_STOP, TILE_STOPA, TILE_STOPS, TILE_TELE_LASER_DISABLE, TILE_TELECHECK,
    TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELECHECKOUT, TILE_TELEIN, TILE_TELEINEVIL, TILE_TELEOUT,
    TILE_THROUGH, TILE_THROUGH_ALL, TILE_THROUGH_CUT, TILE_THROUGH_DIR, TILE_UNFREEZE, TILEFLAG_ROTATE, TILEFLAG_XFLIP,
    TILEFLAG_YFLIP,
};
use crate::vmath::{Vec2, clamp, round_to_int, vdistance};
use ddai_jsmath as js;
use std::collections::HashMap;

/// `clampVel(moveRestriction, vel)` (`collision.ts:43-51`).
pub fn clamp_vel(move_restriction: i32, vel: Vec2) -> Vec2 {
    let mut x = vel.x;
    let mut y = vel.y;
    if x > 0.0 && (move_restriction & CANTMOVE_RIGHT) != 0 {
        x = 0.0;
    }
    if x < 0.0 && (move_restriction & CANTMOVE_LEFT) != 0 {
        x = 0.0;
    }
    if y > 0.0 && (move_restriction & CANTMOVE_DOWN) != 0 {
        y = 0.0;
    }
    if y < 0.0 && (move_restriction & CANTMOVE_UP) != 0 {
        y = 0.0;
    }
    Vec2 { x, y }
}

const MR_DIR_HERE: i32 = 0;
const MR_DX: [i32; 5] = [0, 1, 0, -1, 0];
const MR_DY: [i32; 5] = [0, 0, 1, 0, -1];
const MR_MASK: [i32; 5] = [0, CANTMOVE_RIGHT, CANTMOVE_DOWN, CANTMOVE_LEFT, CANTMOVE_UP];

/// `moveRestrictionsRaw(tile, flags)` (`collision.ts:58-90`).
fn move_restrictions_raw(tile: u8, flags: u8) -> i32 {
    let flags = flags & (TILEFLAG_XFLIP | TILEFLAG_YFLIP | TILEFLAG_ROTATE);
    if tile == TILE_STOP {
        return match flags {
            ROTATION_0 => CANTMOVE_DOWN,
            ROTATION_90 => CANTMOVE_LEFT,
            ROTATION_180 => CANTMOVE_UP,
            ROTATION_270 => CANTMOVE_RIGHT,
            f if f == (TILEFLAG_YFLIP ^ ROTATION_0) => CANTMOVE_UP,
            f if f == (TILEFLAG_YFLIP ^ ROTATION_90) => CANTMOVE_RIGHT,
            f if f == (TILEFLAG_YFLIP ^ ROTATION_180) => CANTMOVE_DOWN,
            f if f == (TILEFLAG_YFLIP ^ ROTATION_270) => CANTMOVE_LEFT,
            _ => 0,
        };
    }
    if tile == TILE_STOPS {
        return match flags {
            ROTATION_0 | ROTATION_180 => CANTMOVE_DOWN | CANTMOVE_UP,
            f if f == (TILEFLAG_YFLIP ^ ROTATION_0) || f == (TILEFLAG_YFLIP ^ ROTATION_180) => {
                CANTMOVE_DOWN | CANTMOVE_UP
            }
            ROTATION_90 | ROTATION_270 => CANTMOVE_LEFT | CANTMOVE_RIGHT,
            f if f == (TILEFLAG_YFLIP ^ ROTATION_90) || f == (TILEFLAG_YFLIP ^ ROTATION_270) => {
                CANTMOVE_LEFT | CANTMOVE_RIGHT
            }
            _ => 0,
        };
    }
    if tile == TILE_STOPA {
        return CANTMOVE_LEFT | CANTMOVE_RIGHT | CANTMOVE_UP | CANTMOVE_DOWN;
    }
    0
}

/// `moveRestrictionsFor(direction, tile, flags)` (`collision.ts:92-96`).
fn move_restrictions_for(direction: i32, tile: u8, flags: u8) -> i32 {
    let result = move_restrictions_raw(tile, flags);
    if direction == MR_DIR_HERE && tile == TILE_STOP {
        return result;
    }
    result & MR_MASK[direction as usize]
}

fn is_stopper(tile: u8) -> bool {
    tile == TILE_STOP || tile == TILE_STOPS || tile == TILE_STOPA
}

fn is_hook_through_tile(tile: u8) -> bool {
    tile == TILE_THROUGH_CUT || tile == TILE_THROUGH || tile == TILE_THROUGH_ALL || tile == TILE_THROUGH_DIR
}

/// A speedup tile's effect (`Collision.speedupAt`'s return shape, `collision.ts:175-180`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Speedup {
    pub force: f64,
    pub max_speed: f64,
    pub dir_x: f64,
    pub dir_y: f64,
}

/// `intersectLine`/`intersectLineHook`'s return shape (`collision.ts:484,553`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineHit {
    pub collision: i32,
    pub out_pos: Vec2,
    pub out_before_pos: Vec2,
}

/// Optional construction extras (`CollisionExtras`, `collision.ts:106-113`).
#[derive(Debug, Clone, Default)]
pub struct CollisionExtras {
    pub tile_flags: Option<Vec<u8>>,
    pub front: Option<(Vec<u8>, Vec<u8>)>,
    pub no_weak_hook: bool,
}

/// Tele layer input to [`Collision::new`] (`collision.ts` constructor's `tele` parameter).
#[derive(Debug, Clone)]
pub struct TeleLayer {
    pub types: Vec<u8>,
    pub numbers: Vec<u8>,
}

/// Speedup layer input to [`Collision::new`] (`collision.ts` constructor's `speedup` parameter).
#[derive(Debug, Clone)]
pub struct SpeedupLayer {
    pub force: Vec<u8>,
    pub max_speed: Vec<u8>,
    pub angle: Vec<i16>,
}

/// `class Collision` (`collision.ts:115-582`).
///
/// **Implementation deviation from TS (not an observable-behavior quirk):** `teleOuts`/
/// `teleCheckOuts` are lazily-memoized private fields in TS (`collision.ts:219-248`); this port
/// computes them eagerly in [`Collision::new`] instead, since neither depends on anything that
/// can change after construction (no method here ever mutates tele data) — `teleOutsFor`/
/// `teleCheckOutsFor` return byte-identical results either way, just without the interior
/// mutability a lazy cache would need. See the crate README.
#[derive(Debug, Clone)]
pub struct Collision {
    pub width: i32,
    pub height: i32,
    pub tiles: Vec<u8>,

    pub tele_type: Option<Vec<u8>>,
    pub tele_number: Option<Vec<u8>>,

    speedup_force: Option<Vec<u8>>,
    speedup_max: Option<Vec<u8>>,
    speedup_angle: Option<Vec<i16>>,

    flags: Vec<u8>,

    pub tile_flags: Vec<u8>,

    pub front_index: Option<Vec<u8>>,
    pub front_flags: Option<Vec<u8>>,

    pub no_weak_hook: bool,

    has_hook_through: bool,
    has_stoppers: bool,

    tele_outs: HashMap<i32, Vec<Vec2>>,
    tele_check_outs: HashMap<i32, Vec<Vec2>>,
}

impl Collision {
    /// `constructor(width, height, tiles, tele?, speedup?, extras?)` (`collision.ts:139-173`).
    pub fn new(
        width: i32,
        height: i32,
        tiles: Vec<u8>,
        tele: Option<TeleLayer>,
        speedup: Option<SpeedupLayer>,
        extras: Option<CollisionExtras>,
    ) -> Self {
        let n = tiles.len();
        let extras = extras.unwrap_or_default();
        let tile_flags = match extras.tile_flags {
            Some(f) if f.len() == n => f,
            _ => vec![0; n],
        };
        let (front_index, front_flags, mut has_hook_through, mut has_stoppers) = match extras.front {
            Some((index, flags)) if index.len() == n && flags.len() == n => {
                let mut hook_through = false;
                let mut stoppers = false;
                for &t in &index {
                    if is_hook_through_tile(t) {
                        hook_through = true;
                    }
                    if is_stopper(t) {
                        stoppers = true;
                    }
                }
                (Some(index), Some(flags), hook_through, stoppers)
            }
            _ => (None, None, false, false),
        };
        let no_weak_hook = extras.no_weak_hook;

        let mut col = Collision {
            width,
            height,
            tiles: vec![0; n],
            tele_type: None,
            tele_number: None,
            speedup_force: None,
            speedup_max: None,
            speedup_angle: None,
            flags: vec![0; n],
            tile_flags,
            front_index,
            front_flags,
            no_weak_hook,
            has_hook_through: false,
            has_stoppers: false,
            tele_outs: HashMap::new(),
            tele_check_outs: HashMap::new(),
        };
        // `for (let i = 0; i < tiles.length; i++) this.setTile(i, tiles[i]);` (collision.ts:163).
        for (i, &t) in tiles.iter().enumerate() {
            col.set_tile(i as i32, t);
        }
        // `setTile` above only updates `has_hook_through`/`has_stoppers` from the *game* layer;
        // the front-layer contribution computed above must be OR'd in separately, matching TS
        // (the constructor sets `this.hasHookThrough = true` from the front loop directly, then
        // `setTile` calls during the game-layer loop can only ever add more `true`s, never clear
        // them — order between the two loops in TS doesn't matter for a boolean OR).
        has_hook_through |= col.has_hook_through;
        has_stoppers |= col.has_stoppers;
        col.has_hook_through = has_hook_through;
        col.has_stoppers = has_stoppers;

        if let Some(t) = tele
            && t.types.len() == n
        {
            col.tele_type = Some(t.types);
            col.tele_number = Some(t.numbers);
        }
        if let Some(s) = speedup
            && s.force.len() == n
        {
            col.speedup_force = Some(s.force);
            col.speedup_max = Some(s.max_speed);
            col.speedup_angle = Some(s.angle);
        }

        col.tele_outs = col.collect_tele(TILE_TELEOUT);
        col.tele_check_outs = col.collect_tele(TILE_TELECHECKOUT);
        col
    }

    /// `speedupAt(index)` (`collision.ts:175-180`).
    pub fn speedup_at(&self, index: i32) -> Option<Speedup> {
        let f = self.speedup_force.as_ref()?;
        if index < 0 || index as usize >= f.len() || f[index as usize] == 0 {
            return None;
        }
        let idx = index as usize;
        let angle_deg = self.speedup_angle.as_ref().map(|a| a[idx]).unwrap_or(0);
        let a = (angle_deg as f64) * js::PI / 180.0;
        Some(Speedup {
            force: f[idx] as f64,
            max_speed: self.speedup_max.as_ref().map(|m| m[idx] as f64).unwrap_or(0.0),
            dir_x: js::cos(a),
            dir_y: js::sin(a),
        })
    }

    /// `setTile(index, tile)` (`collision.ts:182-194`).
    pub fn set_tile(&mut self, index: i32, tile: u8) {
        if index < 0 || index as usize >= self.tiles.len() {
            return;
        }
        let i = index as usize;
        self.tiles[i] = tile;
        let mut f: u8 = 0;
        if tile == TILE_SOLID {
            f |= 1;
        } else if tile == TILE_NOHOOK {
            f |= 1 | 2;
        } else if tile == TILE_DEATH {
            f |= 4;
        }
        if tile == TILE_FREEZE {
            f |= 8;
        }
        if tile == TILE_UNFREEZE {
            f |= 16;
        }
        self.flags[i] = f;
        if is_hook_through_tile(tile) {
            self.has_hook_through = true;
        }
        if is_stopper(tile) {
            self.has_stoppers = true;
        }
    }

    /// `teleAt(x, y)` (`collision.ts:196-203`). See the module doc comment: a *third* pixel-to-
    /// tile mapping, distinct from [`Self::index_at`] and [`Self::get_map_index`].
    pub fn tele_at(&self, x: f64, y: f64) -> (i32, i32) {
        let Some(t) = &self.tele_type else { return (0, 0) };
        let tx = js::min((self.width - 1) as f64, js::max(0.0, js::trunc(round_to_int(x) / 32.0)));
        let ty = js::min(
            (self.height - 1) as f64,
            js::max(0.0, js::trunc(round_to_int(y) / 32.0)),
        );
        let i = (ty as i32) * self.width + (tx as i32);
        let idx = i as usize;
        let number = self.tele_number.as_ref().map(|n| n[idx] as i32).unwrap_or(0);
        (t[idx] as i32, number)
    }

    /// `hasTele()` (`collision.ts:205-207`).
    pub fn has_tele(&self) -> bool {
        self.tele_type.is_some()
    }

    /// `teleTypeAtIndex(index)` (`collision.ts:209-212`).
    pub fn tele_type_at_index(&self, index: i32) -> i32 {
        match &self.tele_type {
            Some(t) if index >= 0 && (index as usize) < t.len() => t[index as usize] as i32,
            _ => 0,
        }
    }

    /// `teleNumberAtIndex(index)` (`collision.ts:214-217`).
    pub fn tele_number_at_index(&self, index: i32) -> i32 {
        match &self.tele_number {
            Some(n) if index >= 0 && (index as usize) < n.len() => n[index as usize] as i32,
            _ => 0,
        }
    }

    /// `teleOutsFor(number)` (`collision.ts:219-225`; see the struct doc comment on eager vs.
    /// lazy memoization).
    pub fn tele_outs_for(&self, number: i32) -> &[Vec2] {
        self.tele_outs.get(&number).map(Vec::as_slice).unwrap_or(&[])
    }

    /// `teleCheckOutsFor(number)` (`collision.ts:227-230`).
    pub fn tele_check_outs_for(&self, number: i32) -> &[Vec2] {
        self.tele_check_outs.get(&number).map(Vec::as_slice).unwrap_or(&[])
    }

    /// `collectTele(type)` (`collision.ts:232-245`).
    fn collect_tele(&self, kind: u8) -> HashMap<i32, Vec<Vec2>> {
        let mut outs: HashMap<i32, Vec<Vec2>> = HashMap::new();
        if let (Some(t), Some(n)) = (&self.tele_type, &self.tele_number) {
            for i in 0..t.len() {
                if t[i] != kind || n[i] == 0 {
                    continue;
                }
                let pos = Vec2 {
                    x: ((i % self.width as usize) * 32 + 16) as f64,
                    y: js::trunc(i as f64 / self.width as f64) * 32.0 + 16.0,
                };
                outs.entry(n[i] as i32).or_default().push(pos);
            }
        }
        outs
    }

    /// `getTileIndex(x, y)` (`collision.ts:250-252`).
    pub fn get_tile_index(&self, x: f64, y: f64) -> u8 {
        let idx = self.index_at(x, y);
        self.tiles.get(idx as usize).copied().unwrap_or(TILE_AIR)
    }

    /// `indexAt(x, y)` (`collision.ts:254-265`). See the module doc comment: the "floor via
    /// arithmetic shift" pixel-to-tile mapping.
    fn index_at(&self, x: f64, y: f64) -> i32 {
        let ix = if x > 0.0 {
            js::trunc(x + 0.5)
        } else {
            js::trunc(x - 0.5)
        };
        let iy = if y > 0.0 {
            js::trunc(y + 0.5)
        } else {
            js::trunc(y - 0.5)
        };

        let mut nx = js::to_int32(ix) >> 5;
        let mut ny = js::to_int32(iy) >> 5;
        if nx < 0 {
            nx = 0;
        } else if nx > self.width - 1 {
            nx = self.width - 1;
        }
        if ny < 0 {
            ny = 0;
        } else if ny > self.height - 1 {
            ny = self.height - 1;
        }
        ny * self.width + nx
    }

    /// `getCollisionAt(x, y)` (`collision.ts:267-270`).
    pub fn get_collision_at(&self, x: f64, y: f64) -> i32 {
        let f = self.flag_at(self.index_at(x, y));
        (if f & 1 == 0 { 0 } else { CFLAG_SOLID })
            | (if f & 2 == 0 { 0 } else { CFLAG_NOHOOK })
            | (if f & 4 == 0 { 0 } else { CFLAG_DEATH })
    }

    fn flag_at(&self, index: i32) -> u8 {
        self.flags.get(index.max(0) as usize).copied().unwrap_or(0)
    }

    /// `checkPoint(x, y)` (`collision.ts:272-274`).
    fn check_point(&self, x: f64, y: f64) -> bool {
        self.flag_at(self.index_at(x, y)) & 1 != 0
    }

    /// `isSolid(x, y)` (`collision.ts:276-278`).
    pub fn is_solid(&self, x: f64, y: f64) -> bool {
        self.flag_at(self.index_at(x, y)) & 1 != 0
    }

    /// `isDeath(x, y)` (`collision.ts:280-282`).
    pub fn is_death(&self, x: f64, y: f64) -> bool {
        self.flag_at(self.index_at(x, y)) & 4 != 0
    }

    /// `isFreeze(x, y)` (`collision.ts:284-286`).
    pub fn is_freeze(&self, x: f64, y: f64) -> bool {
        self.flag_at(self.index_at(x, y)) & 8 != 0
    }

    /// `isUnFreeze(x, y)` (`collision.ts:288-290`).
    pub fn is_un_freeze(&self, x: f64, y: f64) -> bool {
        self.flag_at(self.index_at(x, y)) & 16 != 0
    }

    /// `isNoHook(x, y)` (`collision.ts:292-294`).
    pub fn is_no_hook(&self, x: f64, y: f64) -> bool {
        self.flag_at(self.index_at(x, y)) & 2 != 0
    }

    /// `testBox(pos, size)` (`collision.ts:296-298`).
    pub fn test_box(&self, pos: Vec2, size: Vec2) -> bool {
        self.test_box_at(pos.x, pos.y, size.x * 0.5, size.y * 0.5)
    }

    /// `testBoxAt(x, y, halfX, halfY)` (`collision.ts:300-306`).
    fn test_box_at(&self, x: f64, y: f64, half_x: f64, half_y: f64) -> bool {
        self.check_point(x - half_x, y - half_y)
            || self.check_point(x + half_x, y - half_y)
            || self.check_point(x - half_x, y + half_y)
            || self.check_point(x + half_x, y + half_y)
    }

    /// `moveBox(inoutPos, inoutVel, size, elasticity)` (`collision.ts:308-369`).
    pub fn move_box(&self, inout_pos: &mut Vec2, inout_vel: &mut Vec2, size: Vec2, elasticity: Vec2) {
        let mut pos_x = inout_pos.x;
        let mut pos_y = inout_pos.y;
        let mut vel_x = inout_vel.x;
        let mut vel_y = inout_vel.y;
        let half_x = size.x * 0.5;
        let half_y = size.y * 0.5;

        let distance = js::sqrt(vel_x * vel_x + vel_y * vel_y);
        let max = js::trunc(distance);

        if distance > 0.00001 {
            let fraction = 1.0 / (max + 1.0);
            let elasticity_x = clamp(elasticity.x, -1.0, 1.0);
            let elasticity_y = clamp(elasticity.y, -1.0, 1.0);

            // `for (let i = 0; i <= max; i++)` — kept as an `f64` loop counter (not cast to an
            // integer type) so a `max` of `NaN` runs zero iterations exactly like TS's `i <=
            // NaN` (always false), instead of whatever an `f64 as iN` cast would coerce it to.
            let mut i = 0.0_f64;
            while i <= max {
                if vel_x == 0.0 && vel_y == 0.0 {
                    break;
                }

                let mut new_x = pos_x + vel_x * fraction;
                let mut new_y = pos_y + vel_y * fraction;

                if new_x == pos_x && new_y == pos_y {
                    break;
                }

                if self.test_box_at(new_x, new_y, half_x, half_y) {
                    let mut hits = 0;

                    if self.test_box_at(pos_x, new_y, half_x, half_y) {
                        new_y = pos_y;
                        vel_y *= -elasticity_y;
                        hits += 1;
                    }

                    if self.test_box_at(new_x, pos_y, half_x, half_y) {
                        new_x = pos_x;
                        vel_x *= -elasticity_x;
                        hits += 1;
                    }

                    if hits == 0 {
                        new_y = pos_y;
                        vel_y *= -elasticity_y;
                        new_x = pos_x;
                        vel_x *= -elasticity_x;
                    }
                }

                pos_x = new_x;
                pos_y = new_y;
                i += 1.0;
            }
        }

        inout_pos.x = pos_x;
        inout_pos.y = pos_y;
        inout_vel.x = vel_x;
        inout_vel.y = vel_y;
    }

    /// `movePoint(inoutPos, inoutVel, elasticity, bounces)` (`collision.ts:371-398`).
    pub fn move_point(
        &self,
        inout_pos: &mut Vec2,
        inout_vel: &mut Vec2,
        elasticity: f64,
        mut bounces: Option<&mut i32>,
    ) {
        if let Some(b) = &mut bounces {
            **b = 0;
        }

        let pos = *inout_pos;
        let vel = *inout_vel;
        if self.check_point(pos.x + vel.x, pos.y + vel.y) {
            let mut affected = 0;
            if self.check_point(pos.x + vel.x, pos.y) {
                inout_vel.x *= -elasticity;
                if let Some(b) = &mut bounces {
                    **b += 1;
                }
                affected += 1;
            }

            if self.check_point(pos.x, pos.y + vel.y) {
                inout_vel.y *= -elasticity;
                if let Some(b) = &mut bounces {
                    **b += 1;
                }
                affected += 1;
            }

            if affected == 0 {
                inout_vel.x *= -elasticity;
                inout_vel.y *= -elasticity;
            }
        } else {
            inout_pos.x = pos.x + vel.x;
            inout_pos.y = pos.y + vel.y;
        }
    }

    /// `tileExists(index)` (`collision.ts:400-416`).
    pub fn tile_exists(&self, index: i32) -> bool {
        if index < 0 {
            return false;
        }
        let idx = index as usize;
        let Some(&t) = self.tiles.get(idx) else { return false };
        if (TILE_FREEZE..=TILE_TELE_LASER_DISABLE).contains(&t) || (TILE_LFREEZE..=TILE_LUNFREEZE).contains(&t) {
            return true;
        }

        if let Some(fi) = &self.front_index
            && let Some(&ft) = fi.get(idx)
            && ((TILE_FREEZE..=TILE_TELE_LASER_DISABLE).contains(&ft) || (TILE_LFREEZE..=TILE_LUNFREEZE).contains(&ft))
        {
            return true;
        }

        let ty = self.tele_type_at_index(index) as u8;
        if ty == TILE_TELEIN
            || ty == TILE_TELEINEVIL
            || ty == TILE_TELECHECKINEVIL
            || ty == TILE_TELECHECK
            || ty == TILE_TELECHECKIN
        {
            return true;
        }

        if let Some(f) = &self.speedup_force
            && idx < f.len()
            && f[idx] != 0
        {
            return true;
        }
        self.has_stoppers && self.tile_exists_next(index)
    }

    /// `tileExistsNext(index)` (`collision.ts:418-434`).
    fn tile_exists_next(&self, index: i32) -> bool {
        let n = self.tiles.len() as i32;
        let w = self.width;
        let idx = index;
        let left = if idx - 1 > 0 { idx - 1 } else { idx };
        let right = if idx + 1 < n { idx + 1 } else { idx };
        let below = if idx + w < n { idx + w } else { idx };
        let above = if idx - w > 0 { idx - w } else { idx };
        let layer = |ix: &[u8], fl: &[u8]| -> bool {
            let at = |i: i32| -> (u8, u8) { (ix[i as usize], fl[i as usize]) };
            let (ri, rf) = at(right);
            let (li, lf) = at(left);
            let (bi, bf) = at(below);
            let (ai, af) = at(above);
            if (ri == TILE_STOP && rf == ROTATION_270) || (li == TILE_STOP && lf == ROTATION_90) {
                return true;
            }
            if (bi == TILE_STOP && bf == ROTATION_0) || (ai == TILE_STOP && af == ROTATION_180) {
                return true;
            }
            if ri == TILE_STOPA || li == TILE_STOPA || ri == TILE_STOPS || li == TILE_STOPS {
                return true;
            }
            if bi == TILE_STOPA || ai == TILE_STOPA || bi == TILE_STOPS || ai == TILE_STOPS {
                return true;
            }
            false
        };
        if layer(&self.tiles, &self.tile_flags) {
            return true;
        }
        match (&self.front_index, &self.front_flags) {
            (Some(fi), Some(ff)) => layer(fi, ff),
            _ => false,
        }
    }

    /// `getMoveRestrictions(pos, distance = 18, overrideCenterIndex = -1)` (`collision.ts:436-448`).
    pub fn get_move_restrictions(&self, pos: Vec2, distance: f64, override_center_index: i32) -> i32 {
        if !self.has_stoppers {
            return 0;
        }
        let mut restrictions = 0;
        for d in 0..5usize {
            let mut index = self.index_at(pos.x + MR_DX[d] as f64 * distance, pos.y + MR_DY[d] as f64 * distance);
            if d as i32 == MR_DIR_HERE && override_center_index >= 0 {
                index = override_center_index;
            }
            let idx = index as usize;
            restrictions |= move_restrictions_for(d as i32, self.tiles[idx], self.tile_flags[idx]);
            if let (Some(fi), Some(ff)) = (&self.front_index, &self.front_flags) {
                restrictions |= move_restrictions_for(d as i32, fi[idx], ff[idx]);
            }
        }
        restrictions
    }

    /// `getMapIndex(pos)` (`collision.ts:450-455`). See the module doc comment: the "truncate
    /// toward zero" pixel-to-tile mapping, distinct from [`Self::index_at`].
    pub fn get_map_index(&self, pos: Vec2) -> i32 {
        let nx = clamp(js::trunc(js::trunc(pos.x) / 32.0), 0.0, (self.width - 1) as f64);
        let ny = clamp(js::trunc(js::trunc(pos.y) / 32.0), 0.0, (self.height - 1) as f64);
        let index = (ny as i32) * self.width + (nx as i32);
        if self.tile_exists(index) { index } else { -1 }
    }

    /// `getMapIndices(prevPos, pos, out)` (`collision.ts:457-482`).
    pub fn get_map_indices(&self, prev_pos: Vec2, pos: Vec2, out: &mut Vec<i32>) {
        out.clear();
        let d = vdistance(prev_pos, pos);
        let end = js::trunc(d + 1.0);
        if d == 0.0 {
            let nx = clamp(js::trunc(js::trunc(pos.x) / 32.0), 0.0, (self.width - 1) as f64);
            let ny = clamp(js::trunc(js::trunc(pos.y) / 32.0), 0.0, (self.height - 1) as f64);
            let index = (ny as i32) * self.width + (nx as i32);
            if self.tile_exists(index) {
                out.push(index);
            }
            return;
        }
        let mut last_index = 0i32;
        let mut i = 0.0_f64;
        while i < end {
            let a = i / d;
            let tx = prev_pos.x + (pos.x - prev_pos.x) * a;
            let ty = prev_pos.y + (pos.y - prev_pos.y) * a;
            let nx = clamp(js::trunc(js::trunc(tx) / 32.0), 0.0, (self.width - 1) as f64);
            let ny = clamp(js::trunc(js::trunc(ty) / 32.0), 0.0, (self.height - 1) as f64);
            let index = (ny as i32) * self.width + (nx as i32);
            // TS: `if (lastIndex !== index && this.tileExists(index))` with `lastIndex` seeded to
            // `0` (`collision.ts:468`), not "the first sample always passes" — if the *very
            // first* sampled index in the sweep happens to be `0` (a real, if unusual, map
            // layout), TS's own check `0 !== 0` is false and that first sample is *not*
            // collected. Review finding F6 (a real repro: a freeze tile placed at index 0 was
            // wrongly collected — and the tee wrongly frozen — by an earlier version of this port
            // that added an `i == 0.0 ||` this TS line does not have.
            if last_index != index && self.tile_exists(index) {
                out.push(index);
                last_index = index;
            }
            i += 1.0;
        }
    }

    /// `intersectLineHook(pos0, pos1)` (`collision.ts:484-525`).
    pub fn intersect_line_hook(&self, pos0: Vec2, pos1: Vec2) -> LineHit {
        if !self.has_hook_through {
            return self.intersect_line(pos0, pos1);
        }

        let tx = pos0.x - pos1.x;
        let ty = pos0.y - pos1.y;
        let (mut dx, mut dy) = (0.0, 0.0);
        if js::abs(tx) > js::abs(ty) {
            dx = if tx < 0.0 { -32.0 } else { 32.0 };
        } else {
            dy = if ty < 0.0 { -32.0 } else { 32.0 };
        }
        let distance = vdistance(pos0, pos1);
        let end = js::trunc(distance + 1.0);

        let x0 = pos0.x;
        let y0 = pos0.y;
        let rx = pos1.x - x0;
        let ry = pos1.y - y0;
        let mut last_x = x0;
        let mut last_y = y0;
        let mut i = 0.0_f64;
        while i <= end {
            let a = i / end;
            let px = x0 + rx * a;
            let py = y0 + ry * a;
            let ix = round_to_int(px);
            let iy = round_to_int(py);

            let mut hit = 0;
            if self.check_point(ix, iy) {
                if !self.is_through(ix, iy, dx, dy, pos0, pos1) {
                    hit = self.get_collision_at(ix, iy);
                }
            } else if self.is_hook_blocker(ix, iy, pos0, pos1) {
                hit = CFLAG_NOHOOK;
            }
            if hit != 0 {
                return LineHit {
                    collision: hit,
                    out_pos: Vec2 { x: px, y: py },
                    out_before_pos: Vec2 { x: last_x, y: last_y },
                };
            }

            last_x = px;
            last_y = py;
            i += 1.0;
        }
        LineHit {
            collision: 0,
            out_pos: pos1,
            out_before_pos: pos1,
        }
    }

    /// `isThrough(x, y, offsetX, offsetY, pos0, pos1)` (`collision.ts:527-541`).
    fn is_through(&self, x: f64, y: f64, offset_x: f64, offset_y: f64, pos0: Vec2, pos1: Vec2) -> bool {
        let index = self.index_at(x, y) as usize;
        if let (Some(fi), Some(ff)) = (&self.front_index, &self.front_flags) {
            let t = fi[index];
            if t == TILE_THROUGH_ALL || t == TILE_THROUGH_CUT {
                return true;
            }
            if t == TILE_THROUGH_DIR {
                let f = ff[index];
                if (f == ROTATION_0 && pos0.y > pos1.y)
                    || (f == ROTATION_90 && pos0.x < pos1.x)
                    || (f == ROTATION_180 && pos0.y < pos1.y)
                    || (f == ROTATION_270 && pos0.x > pos1.x)
                {
                    return true;
                }
            }
        }
        let offset_index = self.index_at(x + offset_x, y + offset_y) as usize;
        self.tiles[offset_index] == TILE_THROUGH
            || matches!(&self.front_index, Some(fi) if fi[offset_index] == TILE_THROUGH)
    }

    /// `isHookBlocker(x, y, pos0, pos1)` (`collision.ts:543-551`).
    fn is_hook_blocker(&self, x: f64, y: f64, pos0: Vec2, pos1: Vec2) -> bool {
        let index = self.index_at(x, y) as usize;
        let blocks = |t: u8, f: u8| -> bool {
            if t == TILE_THROUGH_ALL {
                return true;
            }
            t == TILE_THROUGH_DIR
                && ((f == ROTATION_0 && pos0.y < pos1.y)
                    || (f == ROTATION_90 && pos0.x > pos1.x)
                    || (f == ROTATION_180 && pos0.y > pos1.y)
                    || (f == ROTATION_270 && pos0.x < pos1.x))
        };
        if blocks(self.tiles[index], self.tile_flags[index]) {
            return true;
        }
        matches!((&self.front_index, &self.front_flags), (Some(fi), Some(ff)) if blocks(fi[index], ff[index]))
    }

    /// `intersectLine(pos0, pos1)` (`collision.ts:553-581`).
    pub fn intersect_line(&self, pos0: Vec2, pos1: Vec2) -> LineHit {
        let distance = vdistance(pos0, pos1);
        let end = js::trunc(distance + 1.0);
        let x0 = pos0.x;
        let y0 = pos0.y;
        let dx = pos1.x - x0;
        let dy = pos1.y - y0;
        let mut last_x = x0;
        let mut last_y = y0;
        let mut i = 0.0_f64;
        while i <= end {
            let a = i / end;
            let px = x0 + dx * a;
            let py = y0 + dy * a;
            let ix = round_to_int(px);
            let iy = round_to_int(py);

            if self.check_point(ix, iy) {
                return LineHit {
                    collision: self.get_collision_at(ix, iy),
                    out_pos: Vec2 { x: px, y: py },
                    out_before_pos: Vec2 { x: last_x, y: last_y },
                };
            }

            last_x = px;
            last_y = py;
            i += 1.0;
        }
        LineHit {
            collision: 0,
            out_pos: pos1,
            out_before_pos: pos1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmath::vec2;

    fn flat(width: i32, height: i32, solid_border: bool) -> Collision {
        let n = (width * height) as usize;
        let mut tiles = vec![TILE_AIR; n];
        if solid_border {
            for x in 0..width {
                tiles[x as usize] = TILE_SOLID;
                tiles[((height - 1) * width + x) as usize] = TILE_SOLID;
            }
            for y in 0..height {
                tiles[(y * width) as usize] = TILE_SOLID;
                tiles[(y * width + width - 1) as usize] = TILE_SOLID;
            }
        }
        Collision::new(width, height, tiles, None, None, None)
    }

    #[test]
    fn is_solid_matches_border() {
        let c = flat(10, 10, true);
        assert!(c.is_solid(0.0, 160.0));
        assert!(!c.is_solid(160.0, 160.0));
    }

    #[test]
    fn test_box_true_at_corner_inside_wall() {
        let c = flat(10, 10, true);
        // A 28x28 box centered right at the border pixel boundary should touch the solid wall.
        assert!(c.test_box(vec2(14.0, 160.0), vec2(28.0, 28.0)));
        assert!(!c.test_box(vec2(160.0, 160.0), vec2(28.0, 28.0)));
    }

    #[test]
    fn move_box_stops_at_wall() {
        let c = flat(10, 10, true);
        let mut pos = vec2(160.0, 160.0);
        let mut vel = vec2(-500.0, 0.0);
        c.move_box(&mut pos, &mut vel, vec2(28.0, 28.0), vec2(0.0, 0.0));
        // Stopped before crossing into the solid border (x=32 is the first non-border tile edge).
        assert!(
            pos.x >= 32.0 + 14.0 - 1.0,
            "pos.x={} unexpectedly deep in the wall",
            pos.x
        );
        assert_eq!(vel.x, 0.0);
    }

    #[test]
    fn get_map_indices_single_point_when_prev_equals_pos() {
        // `tileExists` is false everywhere on this all-air map (no border, no overlay layer), so
        // both `getMapIndex` and `getMapIndices`' `d === 0` branch report "nothing interesting
        // here" — `getMapIndex` as `-1`, `getMapIndices` as an empty list (it only pushes when
        // `tileExists` is true; see `collision.ts:461-467`, ported verbatim above).
        let c = flat(10, 10, false);
        let mut out = Vec::new();
        c.get_map_indices(vec2(48.0, 48.0), vec2(48.0, 48.0), &mut out);
        assert!(out.is_empty());
        assert_eq!(c.get_map_index(vec2(48.0, 48.0)), -1);

        // Put a freeze tile (an "interesting" tile `tileExists` reports true for) at (1,1) and
        // check the same `d === 0` point there.
        let (width, x, y) = (10usize, 1usize, 1usize);
        let mut tiles = vec![TILE_AIR; width * width];
        tiles[y * width + x] = TILE_FREEZE;
        let c2 = Collision::new(10, 10, tiles, None, None, None);
        let mut out2 = Vec::new();
        let p = vec2(x as f64 * 32.0 + 16.0, y as f64 * 32.0 + 16.0);
        c2.get_map_indices(p, p, &mut out2);
        assert_eq!(out2, vec![c2.get_map_index(p)]);
        assert_eq!(out2, vec![11]);
    }

    /// Review finding F6: a freeze tile sitting exactly at map index 0 must **not** be collected
    /// by `getMapIndices` on the very first sampled point of a sweep (`lastIndex` starts at `0`
    /// in TS, `collision.ts:468`, so the first sample's own index being `0` fails `lastIndex !==
    /// index` and is skipped) — reproduces the reviewer's exact repro (5x5 map, freeze tile at
    /// index 0, sweep from (10,10) to (100,10)).
    #[test]
    fn get_map_indices_does_not_collect_index_zero_on_the_first_sample() {
        let n = 25usize;
        let mut tiles = vec![TILE_AIR; n];
        tiles[0] = TILE_FREEZE;
        let c = Collision::new(5, 5, tiles, None, None, None);
        let mut out = Vec::new();
        c.get_map_indices(vec2(10.0, 10.0), vec2(100.0, 10.0), &mut out);
        assert!(
            out.is_empty(),
            "index 0 must not be collected on the first sample: {out:?}"
        );
    }

    #[test]
    fn intersect_line_hits_solid_wall() {
        let c = flat(10, 10, true);
        let hit = c.intersect_line(vec2(160.0, 160.0), vec2(160.0, -100.0));
        assert_ne!(hit.collision, 0);
        assert!(hit.out_pos.y > 0.0);
    }

    #[test]
    fn move_restrictions_stop_rotation_0_blocks_down() {
        let width = 5;
        let height = 5;
        let n = (width * height) as usize;
        let mut tiles = vec![TILE_AIR; n];
        tiles[(2 * width + 2) as usize] = TILE_STOP;
        let mut tile_flags = vec![0u8; n];
        tile_flags[(2 * width + 2) as usize] = ROTATION_0;
        let c = Collision::new(
            width,
            height,
            tiles,
            None,
            None,
            Some(CollisionExtras {
                tile_flags: Some(tile_flags),
                front: None,
                no_weak_hook: false,
            }),
        );
        let r = c.get_move_restrictions(vec2(2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0), 18.0, -1);
        assert_eq!(r & CANTMOVE_DOWN, CANTMOVE_DOWN);
    }

    #[test]
    fn tele_outs_for_collects_all_matching_numbers() {
        let width = 3;
        let height = 3;
        let n = (width * height) as usize;
        let tiles = vec![TILE_AIR; n];
        let mut types = vec![0u8; n];
        let mut numbers = vec![0u8; n];
        types[0] = TILE_TELEOUT;
        numbers[0] = 7;
        types[4] = TILE_TELEOUT;
        numbers[4] = 7;
        let c = Collision::new(width, height, tiles, Some(TeleLayer { types, numbers }), None, None);
        let outs = c.tele_outs_for(7);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0], vec2(16.0, 16.0));
        assert_eq!(outs[1], vec2(1.0 * 32.0 + 16.0, 1.0 * 32.0 + 16.0));
        assert!(c.tele_outs_for(99).is_empty());
    }

    #[test]
    fn no_data_defaults_are_safe() {
        let c = flat(4, 4, false);
        assert_eq!(c.speedup_at(0), None);
        assert_eq!(c.tele_type_at_index(0), 0);
        assert!(!c.has_tele());
        assert_eq!(c.tele_at(0.0, 0.0), (0, 0));
    }
}
