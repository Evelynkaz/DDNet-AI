//! Tile queries on the map the bot's heuristics need (wander hazards, "standing in freeze",
//! `nearFreeze` for the seal check, reachability flood fill). Game **and** front layer are read —
//! a freeze or death tile in the front layer kills/freezes exactly like one in the game layer
//! (`character.cpp:1477-1484,1658-1673`); the TS `isFreeze`/`isDeath` looked at the game layer only
//! (a known quirk, `docs/research/orig-plan.md` §9). The planner's shield/seal keep that quirk
//! because they sit on the planner's own collision adapter; the bot's own checks do not.

use ddai_physics::map::{MapData, TILE_DEATH, TILE_DFREEZE, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID};

/// Pixels per tile.
pub const TILE_PX: i32 = 32;

#[derive(Clone)]
pub struct MapGrid {
    width: i32,
    height: i32,
    /// Bit 0 solid (incl. no-hook), bit 1 freeze, bit 2 death.
    cells: Vec<u8>,
}

const CELL_SOLID: u8 = 1;
const CELL_FREEZE: u8 = 2;
const CELL_DEATH: u8 = 4;

fn classify(index: u8) -> u8 {
    match index {
        TILE_SOLID | TILE_NOHOOK => CELL_SOLID,
        TILE_FREEZE | TILE_DFREEZE => CELL_FREEZE,
        TILE_DEATH => CELL_DEATH,
        _ => 0,
    }
}

impl MapGrid {
    pub fn new(map: &MapData) -> Self {
        let n = (map.width as usize) * (map.height as usize);
        let mut cells = vec![0u8; n];
        for (i, cell) in cells.iter_mut().enumerate() {
            if let Some(t) = map.game.get(i) {
                *cell |= classify(t.index);
            }
            if let Some(t) = map.front.as_ref().and_then(|f| f.get(i)) {
                // A solid front tile is not solid (front only adds hook-through/stoppers); freeze
                // and death in front count.
                *cell |= classify(t.index) & (CELL_FREEZE | CELL_DEATH);
            }
        }
        // Heart pickups freeze whoever comes within 48 px (task 4.2, review F2): the tile-based helpers
        // (unstick, the guard's hazard gate, the reachability flood) see that zone as freeze.
        for (cell, hazard) in cells.iter_mut().zip(ddai_physics::map::pickup_freeze_mask(map)) {
            if hazard {
                *cell |= CELL_FREEZE;
            }
        }
        MapGrid {
            width: map.width as i32,
            height: map.height as i32,
            cells,
        }
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    fn at_tile(&self, tx: i32, ty: i32) -> u8 {
        if tx < 0 || ty < 0 || tx >= self.width || ty >= self.height {
            // Outside the map: like the server's `GetCollisionAt`, the border is solid.
            return CELL_SOLID;
        }
        self.cells[(ty * self.width + tx) as usize]
    }

    fn at(&self, x: f32, y: f32) -> u8 {
        self.at_tile((x / TILE_PX as f32).floor() as i32, (y / TILE_PX as f32).floor() as i32)
    }

    pub fn is_solid(&self, x: f32, y: f32) -> bool {
        self.at(x, y) & CELL_SOLID != 0
    }

    pub fn is_freeze(&self, x: f32, y: f32) -> bool {
        self.at(x, y) & CELL_FREEZE != 0
    }

    pub fn is_death(&self, x: f32, y: f32) -> bool {
        self.at(x, y) & CELL_DEATH != 0
    }

    pub fn tile_solid(&self, tx: i32, ty: i32) -> bool {
        self.at_tile(tx, ty) & CELL_SOLID != 0
    }

    pub fn tile_hazard(&self, tx: i32, ty: i32) -> bool {
        self.at_tile(tx, ty) & (CELL_FREEZE | CELL_DEATH) != 0
    }

    /// `nearFreeze` (`bot.ts:2790-2798`): a freeze tile within `tiles` tiles of `(x, y)` on either
    /// axis.
    pub fn near_freeze(&self, x: f32, y: f32, tiles: i32) -> bool {
        for oy in -tiles..=tiles {
            for ox in -tiles..=tiles {
                if self.is_freeze(x + (ox * TILE_PX) as f32, y + (oy * TILE_PX) as f32) {
                    return true;
                }
            }
        }
        false
    }

    /// A freeze or death tile anywhere in the box `+-tiles` around `(x, y)`. The guard (shield) is
    /// pointless away from any hazard: it simulates 2 + 36 + 90 ticks, which cannot reach a tile
    /// farther than this for the small `tiles` the bot uses ([`crate::consts::GUARD_HAZARD_TILES`]).
    pub fn hazard_within(&self, x: f32, y: f32, tiles: i32) -> bool {
        let (cx, cy) = ((x / TILE_PX as f32).floor() as i32, (y / TILE_PX as f32).floor() as i32);
        for ty in cy - tiles..=cy + tiles {
            for tx in cx - tiles..=cx + tiles {
                if self.tile_hazard(tx, ty) && tx >= 0 && ty >= 0 && tx < self.width && ty < self.height {
                    return true;
                }
            }
        }
        false
    }

    /// `wanderHazardBelow` (`bot.ts:681-689`): walking down from `(x, y)` up to 6 tiles, a freeze or
    /// death tile before any solid one.
    /// `hazardWithinPx(collision, x0, y0, direction, px)` (`navigate.ts`): freeze or death within `px` ahead of
    /// `(x0, y0)` in `direction`, down to four tiles below, before a wall.
    pub fn hazard_within_px(&self, x0: f32, y0: f32, direction: i32, px: f32) -> bool {
        let mut offsets: Vec<f32> = Vec::new();
        let mut a = px.min(24.0);
        while a < px {
            offsets.push(a);
            a += 32.0;
        }
        offsets.push(px);
        for ahead in offsets {
            let x = x0 + direction as f32 * ahead;
            if self.is_solid(x, y0) {
                return false;
            }
            for dy in 0..=4 {
                let y = y0 + (dy * 32) as f32;
                if self.is_freeze(x, y) || self.is_death(x, y) {
                    return true;
                }
                if self.is_solid(x, y) {
                    break;
                }
            }
        }
        false
    }

    pub fn hazard_below(&self, x: f32, y: f32) -> bool {
        for k in 1..=6 {
            let yy = y + (k * TILE_PX) as f32;
            if self.is_solid(x, yy) {
                return false;
            }
            if self.is_freeze(x, yy) || self.is_death(x, yy) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
pub(crate) mod test_maps {
    use ddai_physics::map::{MapData, TILE_DEATH, TILE_FREEZE, TILE_SOLID, Tile};

    /// A `w x h` room: solid border, `floor` rows of solid at the bottom above the border ... kept
    /// simple: border solid; `extra` places more tiles.
    pub fn room(w: u32, h: u32, extra: &[(u32, u32, u8)]) -> MapData {
        let mut game = vec![Tile::default(); (w * h) as usize];
        for x in 0..w {
            game[x as usize].index = TILE_SOLID;
            game[((h - 1) * w + x) as usize].index = TILE_SOLID;
        }
        for y in 0..h {
            game[(y * w) as usize].index = TILE_SOLID;
            game[(y * w + w - 1) as usize].index = TILE_SOLID;
        }
        for &(x, y, index) in extra {
            game[(y * w + x) as usize].index = index;
        }
        MapData {
            width: w,
            height: h,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    pub const FREEZE: u8 = TILE_FREEZE;
    pub const DEATH: u8 = TILE_DEATH;
    pub const SOLID: u8 = TILE_SOLID;
}

#[cfg(test)]
mod tests {
    use super::test_maps::*;
    use super::*;

    #[test]
    fn classifies_tiles_and_treats_outside_as_solid() {
        let m = room(10, 10, &[(3, 3, FREEZE), (4, 4, DEATH), (5, 5, SOLID)]);
        let g = MapGrid::new(&m);
        assert!(g.is_freeze(3.0 * 32.0 + 5.0, 3.0 * 32.0 + 5.0));
        assert!(g.is_death(4.0 * 32.0 + 1.0, 4.0 * 32.0 + 1.0));
        assert!(g.is_solid(5.0 * 32.0, 5.0 * 32.0));
        assert!(!g.is_solid(2.0 * 32.0, 2.0 * 32.0));
        assert!(g.is_solid(-10.0, 50.0) && g.is_solid(50.0, 9999.0), "outside is solid");
    }

    #[test]
    fn a_heart_pickup_makes_its_3x3_neighbourhood_freeze() {
        // ENTITY_HEALTH_1 = 7 + ENTITY_OFFSET (191) = 198: it freezes whoever is within 48 px.
        let m = room(10, 10, &[(5, 5, 198)]);
        let g = MapGrid::new(&m);
        for (dx, dy) in [(0, 0), (1, 0), (-1, 1), (1, 1), (0, -1)] {
            let (x, y) = ((5 + dx) as f32 * 32.0 + 16.0, (5 + dy) as f32 * 32.0 + 16.0);
            assert!(g.is_freeze(x, y), "({dx},{dy}) around the heart");
        }
        assert!(
            !g.is_freeze(7.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0),
            "two tiles away is out of reach"
        );
        assert!(g.hazard_within(7.0 * 32.0 + 16.0, 5.0 * 32.0 + 16.0, 1));
    }

    #[test]
    fn front_layer_freeze_and_death_count_but_front_solid_does_not() {
        let mut m = room(10, 10, &[]);
        let mut front = vec![ddai_physics::map::Tile::default(); 100];
        front[2 * 10 + 2].index = FREEZE;
        front[3 * 10 + 3].index = SOLID;
        m.front = Some(front);
        let g = MapGrid::new(&m);
        assert!(g.is_freeze(2.0 * 32.0, 2.0 * 32.0));
        assert!(!g.is_solid(3.0 * 32.0, 3.0 * 32.0));
    }

    #[test]
    fn near_freeze_looks_two_tiles_around() {
        let m = room(12, 12, &[(6, 6, FREEZE)]);
        let g = MapGrid::new(&m);
        assert!(g.near_freeze(4.0 * 32.0 + 16.0, 6.0 * 32.0 + 16.0, 2), "2 tiles away");
        assert!(!g.near_freeze(2.0 * 32.0 + 16.0, 6.0 * 32.0 + 16.0, 2), "4 tiles away");
    }

    #[test]
    fn hazard_within_is_a_box_test_and_ignores_the_solid_border() {
        let m = room(40, 40, &[(20, 20, FREEZE)]);
        let g = MapGrid::new(&m);
        assert!(g.hazard_within(14.0 * 32.0, 20.0 * 32.0, 6), "6 tiles away");
        assert!(!g.hazard_within(13.0 * 32.0, 20.0 * 32.0, 6), "7 tiles away");
        assert!(
            !g.hazard_within(5.0 * 32.0, 5.0 * 32.0, 3),
            "near the map edge: outside is solid, not a hazard"
        );
    }

    #[test]
    fn hazard_below_stops_at_the_first_solid() {
        let m = room(12, 20, &[(5, 8, FREEZE), (7, 5, SOLID), (7, 8, FREEZE)]);
        let g = MapGrid::new(&m);
        let (x5, x7) = (5.0 * 32.0 + 10.0, 7.0 * 32.0 + 10.0);
        assert!(
            g.hazard_below(x5, 4.0 * 32.0),
            "freeze 4 tiles below, nothing solid between"
        );
        assert!(
            !g.hazard_below(x7, 3.0 * 32.0),
            "a solid tile shields the freeze beneath it"
        );
        assert!(!g.hazard_below(x5, 1.0 * 32.0), "beyond 6 tiles is ignored");
    }
}
