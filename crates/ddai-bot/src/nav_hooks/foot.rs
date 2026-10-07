//! Task 3.14 (D-105): the map's foot-walkable components, for one question the WB walk asks: "is the place the bot respawned in cut off, on foot, from
//! everything the wayblock needs?" (an F-DDrace `/1vs1` arena on a map that also holds a Copy Love Box hall).
//!
//! Open tiles ([`NavGrid::free`]: not solid, not freeze, not death; a tele entrance counts) are labelled once per map by a flood fill over the four
//! neighbours and the teleporters' links, so every question afterwards is a lookup. It is a *lower bound* of what the route search reaches (a rope
//! swung over a freeze strip is not followed), which is why the walk is closed only when the bot's last **respawn** was cut off too: a kill (the
//! old walk's way out of a pocket) would put it back in the same place.

use std::collections::HashSet;

use ddai_nav::grid::NavGrid;

/// The labels of the open tiles of a map, and the labels the wayblock's crossings, spots and exits lie in.
pub(super) struct FootMap {
    width: i32,
    height: i32,
    /// Component label per tile; `u32::MAX` = not open.
    comp: Vec<u32>,
    /// The components that hold a crossing start, a spot or an exit of the WB.
    wb: HashSet<u32>,
}

/// Union-find root with path halving.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

impl FootMap {
    /// Labels `grid`'s open tiles; `wb_tiles` are the tiles of the wayblock that count (crossing starts, spots, exits). Built once per map (a few
    /// milliseconds), when the map is loaded: never on the snapshot path.
    pub fn new(grid: &NavGrid, wb_tiles: impl IntoIterator<Item = (i32, i32)>) -> FootMap {
        let (w, h) = (grid.width, grid.height);
        let n = (w.max(0) as usize) * (h.max(0) as usize);
        let mut comp = vec![u32::MAX; n];
        let mut next = 0u32;
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..n {
            if grid.free[start] == 0 || comp[start] != u32::MAX {
                continue;
            }
            comp[start] = next;
            stack.push(start);
            while let Some(i) = stack.pop() {
                let (x, y) = ((i as i32) % w, (i as i32) / w);
                let mut visit = |j: usize, stack: &mut Vec<usize>| {
                    if grid.free[j] != 0 && comp[j] == u32::MAX {
                        comp[j] = next;
                        stack.push(j);
                    }
                };
                if x > 0 {
                    visit(i - 1, &mut stack);
                }
                if x + 1 < w {
                    visit(i + 1, &mut stack);
                }
                if y > 0 {
                    visit(i - w as usize, &mut stack);
                }
                if y + 1 < h {
                    visit(i + w as usize, &mut stack);
                }
            }
            next += 1;
        }
        // A teleporter joins the component of its entrance to the one of its exit (one pass over the entrances; no per-tile map lookups).
        let mut parent: Vec<u32> = (0..next).collect();
        for (i, &out) in grid.tele_out.iter().enumerate().take(n) {
            if out < 0 || (out as usize) >= n {
                continue;
            }
            let (a, b) = (comp[i], comp[out as usize]);
            if a == u32::MAX || b == u32::MAX {
                continue;
            }
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
        for c in &mut comp {
            if *c != u32::MAX {
                *c = find(&mut parent, *c);
            }
        }
        let mut map = FootMap {
            width: w,
            height: h,
            comp,
            wb: HashSet::new(),
        };
        for t in wb_tiles {
            if let Some(c) = map.component(t) {
                map.wb.insert(c);
            }
        }
        map
    }

    /// The component of tile `(tx, ty)`, `None` when it is not an open tile (or off the map).
    pub fn component(&self, (tx, ty): (i32, i32)) -> Option<u32> {
        if tx < 0 || ty < 0 || tx >= self.width || ty >= self.height {
            return None;
        }
        let c = self.comp[(ty * self.width + tx) as usize];
        (c != u32::MAX).then_some(c)
    }

    /// Whether `tile` is an open tile in a component that holds nothing of the wayblock.
    pub fn cut_off_from_wb(&self, tile: (i32, i32)) -> bool {
        self.component(tile).is_some_and(|c| !self.wb.contains(&c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_nav::grid::NavGrid;
    use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, TILE_TELEIN, TILE_TELEOUT, TeleTile, Tile};
    use ddai_planner::physics_adapter::PhysicsWorld;
    use ddai_planner::plan_world::PlanWorld;
    use std::sync::Arc;

    /// A 24 x 10 map: a solid frame, a freeze wall at x = 12 (full height), so two open rooms; a solid wall at x = 6 with a one-tile gap (row 5) inside the left one.
    /// With `tele`, a teleporter (number 1) whose entrance is at (3, 3) in the left room and whose exit is at (18, 3) in the right one.
    fn two_rooms(tele: bool) -> NavGrid {
        let (w, h) = (24usize, 10usize);
        let mut game = vec![Tile::default(); w * h];
        let mut set = |x: usize, y: usize, index: u8| {
            game[y * w + x] = Tile {
                index,
                ..Tile::default()
            }
        };
        for x in 0..w {
            set(x, 0, TILE_SOLID);
            set(x, h - 1, TILE_SOLID);
        }
        for y in 0..h {
            set(0, y, TILE_SOLID);
            set(w - 1, y, TILE_SOLID);
            set(12, y, TILE_FREEZE);
            if y != 5 {
                set(6, y, TILE_SOLID);
            }
        }
        let tele = tele.then(|| {
            let mut t = vec![TeleTile::default(); w * h];
            t[3 * w + 3] = TeleTile {
                number: 1,
                kind: TILE_TELEIN,
            };
            t[3 * w + 18] = TeleTile {
                number: 1,
                kind: TILE_TELEOUT,
            };
            t
        });
        let map = MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let world = PhysicsWorld::new(Arc::new(map), 1);
        NavGrid::new(world.collision())
    }

    #[test]
    fn rooms_are_joined_by_open_gaps_and_cut_by_freeze_and_solid() {
        let grid = two_rooms(false);
        // The wayblock's tile lies in the left room; the right room is behind the freeze wall.
        let foot = FootMap::new(&grid, [(2, 3)]);
        let a = foot.component((2, 3)).expect("open");
        assert_eq!(
            foot.component((9, 3)),
            Some(a),
            "through the gap in the solid wall (row 5) the two halves of the left room are one"
        );
        assert!(foot.component((12, 3)).is_none(), "a freeze tile is not open");
        assert!(
            foot.component((-1, 0)).is_none() && foot.component((99, 3)).is_none(),
            "off the map"
        );
        assert!(!foot.cut_off_from_wb((2, 3)), "the wayblock's own room");
        assert!(!foot.cut_off_from_wb((9, 8)), "joined to it through the gap");
        assert!(foot.cut_off_from_wb((18, 3)), "behind the freeze wall");
        assert!(
            !foot.cut_off_from_wb((12, 3)),
            "a tile that is not open is no verdict (the old walk decides)"
        );
    }

    #[test]
    fn without_a_wayblock_tile_every_room_is_cut_off() {
        let foot = FootMap::new(&two_rooms(false), std::iter::empty());
        assert!(foot.cut_off_from_wb((2, 3)) && foot.cut_off_from_wb((18, 3)));
    }

    /// Review F9: a teleporter joins the room of its entrance to the room of its exit, so a place behind the freeze wall that a teleporter leads to
    /// is no longer cut off from the wayblock (and the room beyond, with no teleporter, still is).
    #[test]
    fn a_teleporter_joins_the_rooms_it_links() {
        let foot = FootMap::new(&two_rooms(true), [(2, 3)]);
        assert_eq!(
            foot.component((18, 3)),
            foot.component((2, 3)),
            "entrance (3, 3) -> exit (18, 3)"
        );
        assert!(
            !foot.cut_off_from_wb((18, 3)),
            "the right room is reached through the teleporter"
        );
        assert!(!foot.cut_off_from_wb((20, 8)), "all of it");
        let plain = FootMap::new(&two_rooms(false), [(2, 3)]);
        assert!(plain.cut_off_from_wb((18, 3)), "without the teleporter it is cut off");
    }
}
