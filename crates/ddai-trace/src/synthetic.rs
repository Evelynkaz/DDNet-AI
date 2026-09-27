//! Deterministic synthetic map recipes.
//!
//! Each recipe builds a small, hand-designed [`MapData`] that exercises one family of tiles the
//! parity work cares about. They are deterministic pure functions (no PRNG, no time, no
//! environment) — the same recipe name always produces byte-identical `MapData`, which is what
//! lets `Scenario`'s map reference be just a recipe name plus a sha256 check.

use ddai_physics::map::{
    MapData, ROTATIONS, ROTATIONS_YFLIP, SpeedupTile, SwitchTile, TILE_DEATH, TILE_DFREEZE, TILE_DUNFREEZE,
    TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, TILE_SPEED_BOOST, TILE_SPEED_BOOST_OLD, TILE_STOP, TILE_STOPA, TILE_STOPS,
    TILE_TELECHECK, TILE_TELECHECKIN, TILE_TELECHECKINEVIL, TILE_TELECHECKOUT, TILE_TELEIN, TILE_TELEINEVIL,
    TILE_TELEINHOOK, TILE_TELEOUT, TILE_THROUGH, TILE_THROUGH_ALL, TILE_THROUGH_CUT, TILE_THROUGH_DIR, TILE_UNFREEZE,
    TeleTile, Tile, TuneTile,
};

/// The recipe names [`build`] accepts, in a fixed order (used by the CLI's `--help` and by
/// tests that iterate "every recipe").
pub const RECIPES: [&str; 4] = ["arena", "freeze", "front", "tele-speedup"];

/// Builds the named synthetic map, or `None` if `name` isn't one of [`RECIPES`].
pub fn build(name: &str) -> Option<MapData> {
    match name {
        "arena" => Some(arena()),
        "freeze" => Some(freeze()),
        "front" => Some(front()),
        "tele-speedup" => Some(tele_speedup()),
        _ => None,
    }
}

/// A tile grid under construction. Every cell starts as `TILE_AIR` (all-zero `Tile`/`TeleTile`/
/// `SpeedupTile`) — recipes only need to say what's *not* air.
struct Grid {
    width: i32,
    height: i32,
    game: Vec<Tile>,
    front: Option<Vec<Tile>>,
    tele: Option<Vec<TeleTile>>,
    speedup: Option<Vec<SpeedupTile>>,
}

impl Grid {
    fn new(width: i32, height: i32) -> Self {
        let n = (width * height) as usize;
        Grid {
            width,
            height,
            game: vec![Tile::default(); n],
            front: None,
            tele: None,
            speedup: None,
        }
    }

    fn idx(&self, x: i32, y: i32) -> usize {
        assert!(
            (0..self.width).contains(&x) && (0..self.height).contains(&y),
            "tile ({x},{y}) out of bounds"
        );
        (y * self.width + x) as usize
    }

    fn game(&mut self, x: i32, y: i32, index: u8) {
        self.game_flags(x, y, index, 0);
    }

    fn game_flags(&mut self, x: i32, y: i32, index: u8, flags: u8) {
        let i = self.idx(x, y);
        self.game[i] = Tile {
            index,
            flags,
            skip: 0,
            reserved: 0,
        };
    }

    fn front_flags(&mut self, x: i32, y: i32, index: u8, flags: u8) {
        let n = self.game.len();
        let i = self.idx(x, y);
        self.front.get_or_insert_with(|| vec![Tile::default(); n])[i] = Tile {
            index,
            flags,
            skip: 0,
            reserved: 0,
        };
    }

    fn tele(&mut self, x: i32, y: i32, number: u8, kind: u8) {
        let n = self.game.len();
        let i = self.idx(x, y);
        self.tele.get_or_insert_with(|| vec![TeleTile::default(); n])[i] = TeleTile { number, kind };
    }

    fn speedup(&mut self, x: i32, y: i32, force: u8, max_speed: u8, kind: u8, angle: i16) {
        let n = self.game.len();
        let i = self.idx(x, y);
        self.speedup.get_or_insert_with(|| vec![SpeedupTile::default(); n])[i] = SpeedupTile {
            force,
            max_speed,
            kind,
            angle,
        };
    }

    /// Fills the outermost ring with `TILE_SOLID` so a scenario's characters can never fall out
    /// of (or hook out of) the map.
    fn solid_border(&mut self) {
        for x in 0..self.width {
            self.game(x, 0, TILE_SOLID);
            self.game(x, self.height - 1, TILE_SOLID);
        }
        for y in 0..self.height {
            self.game(0, y, TILE_SOLID);
            self.game(self.width - 1, y, TILE_SOLID);
        }
    }

    fn row_solid(&mut self, y: i32, x0: i32, x1: i32) {
        for x in x0..=x1 {
            self.game(x, y, TILE_SOLID);
        }
    }

    fn into_map(self) -> MapData {
        let map = MapData {
            width: self.width as u32,
            height: self.height as u32,
            game: self.game,
            front: self.front,
            tele: self.tele,
            speedup: self.speedup,
            switch: None::<Vec<SwitchTile>>,
            tune: None::<Vec<TuneTile>>,
            settings: Vec::new(),
        };
        map.validate()
            .expect("synthetic recipe produced an internally inconsistent MapData");
        map
    }
}

/// Solid border, an internal floor with a gap, two floating platforms, a ceiling with an
/// unhookable section, and a narrow one-tile-wide vertical corridor.
fn arena() -> MapData {
    let mut g = Grid::new(40, 24);
    g.solid_border();

    // Floor with a 2-tile gap to fall through.
    g.row_solid(20, 2, 17);
    g.row_solid(20, 20, 38);

    // Floating platforms.
    g.row_solid(14, 6, 12);
    g.row_solid(10, 24, 30);

    // Ceiling, partly unhookable.
    for x in 2..=38 {
        g.game(x, 3, if (10..14).contains(&x) { TILE_NOHOOK } else { TILE_SOLID });
    }

    // Narrow (1-tile) vertical corridor, walled on both sides, clear of the floor/ceiling rows.
    for y in 4..=19 {
        g.game(34, y, TILE_SOLID);
        g.game(36, y, TILE_SOLID);
    }

    g.into_map()
}

/// Freeze / unfreeze / deep-freeze pools, a nohook internal wall, and a death-tile pit.
fn freeze() -> MapData {
    let mut g = Grid::new(36, 22);
    g.solid_border();

    // Nohook wall splitting the room, but only across the lower half (leaves a passage above).
    for y in 10..=19 {
        g.game(10, y, TILE_NOHOOK);
    }

    // Freeze pool.
    for y in 5..=9 {
        for x in 14..=19 {
            g.game(x, y, TILE_FREEZE);
        }
    }
    // Unfreeze strip right next to it.
    for y in 5..=9 {
        g.game(21, y, TILE_UNFREEZE);
    }

    // Deep-freeze pool, and its own unfreeze tile.
    for y in 13..=15 {
        for x in 23..=27 {
            g.game(x, y, TILE_DFREEZE);
        }
    }
    g.game(29, 14, TILE_DUNFREEZE);

    // Death pit.
    g.row_solid(18, 2, 8); // a ledge to walk up to the pit from
    for x in 9..=13 {
        g.game(x, 18, TILE_DEATH);
    }

    g.into_map()
}

/// A front layer exercising every hook-through tile and every stopper tile, in every rotation
/// `collision.cpp`'s `GetMoveRestrictionsRaw`/`IsThrough`/`IsHookBlocker` distinguish.
///
/// Review round 1, finding F1: the original layout put `THROUGH_CUT`/`THROUGH_ALL`/
/// `THROUGH_DIR` on the front layer over game-layer `AIR`. `CCollision::IsThrough` is only ever
/// consulted from `IntersectLineTeleHook` *after* `CheckPoint` already found something solid at
/// that cell (`collision.cpp` ~390-393) — over open air, the hook line just never registers a
/// hit there in the first place, so `IsThrough` (and therefore every one of its rotation checks)
/// was provably dead code for every scenario this recipe could ever produce. Fixed by giving
/// each hook-through cell a `TILE_SOLID` game tile underneath (row A below); a hook line that
/// would otherwise stop there instead calls `IsThrough`, which overrides the stop. Also fixed:
/// `GetMoveRestrictionsRaw`'s `TILE_STOP`/`TILE_STOPS` cases each have *8* raw flag-byte labels,
/// not 4 (see [`ddai_physics::map::ROTATIONS_YFLIP`]'s doc comment) — row B now covers both
/// halves. Row C exercises `IsThrough`'s *other* branch (a tile that is itself plain solid, with
/// a bare `TILE_THROUGH` on the front layer in the direction the hook is travelling —
/// `collision.cpp` ~619) regardless of which side the hook approaches from. Row D exercises
/// `IsHookBlocker` (a *different* function, only reachable when `CheckPoint` is false — i.e. the
/// tile is "open" — but the raw game-layer tile id is `THROUGH_ALL`/`THROUGH_DIR` anyway, which
/// `GetTile()` doesn't recognize as solid since those ids fall outside `[TILE_SOLID,
/// TILE_NOLASER]`, while `IsHookBlocker` reads the raw id directly).
///
/// Review round 2, finding F12: row D above puts `THROUGH_ALL`/`THROUGH_DIR` on the *game*
/// layer over open air, which reaches `IsHookBlocker`'s first two branches
/// (`collision.cpp:625`, `:627-631`) but never its third — the *front*-layer `THROUGH_DIR`
/// check at `collision.cpp:632-633` (`m_pFront[Index].m_Index == TILE_THROUGH_DIR && ...`),
/// which every real map that uses a front-layer `THROUGH_DIR` tile as a one-way hook blocker
/// over open ground depends on. Row E restores it: a front-layer `THROUGH_DIR` cell (game layer
/// left untouched, still plain `TILE_AIR`) for each rotation, placed in the middle of the large
/// open room above the floor — clear of every other row, and clear on all 4 sides — so a hook
/// can reach each cell from whichever direction its rotation cares about (each cell blocks a
/// hook travelling in exactly one of the 4 cardinal directions and lets every other hook,
/// including one travelling the opposite way, pass through as if it were still bare air).
fn front() -> MapData {
    let mut g = Grid::new(30, 20);
    g.solid_border();
    g.row_solid(17, 1, 28);

    // Row A: unconditional/rotational hook-through, each cell backed by a solid game tile so
    // `IsThrough` is actually reached.
    let ya = 8;
    g.game(2, ya, TILE_SOLID);
    g.front_flags(2, ya, TILE_THROUGH_CUT, 0);
    g.game(4, ya, TILE_SOLID);
    g.front_flags(4, ya, TILE_THROUGH_ALL, 0);
    for (i, flags) in ROTATIONS.iter().enumerate() {
        let x = 6 + i as i32;
        g.game(x, ya, TILE_SOLID);
        g.front_flags(x, ya, TILE_THROUGH_DIR, *flags);
    }

    // Row B: every stopper, both raw-flag-byte families (plain rotation and its YFLIP variant).
    let yb = 10;
    for (i, flags) in ROTATIONS.iter().enumerate() {
        g.front_flags(2 + i as i32, yb, TILE_STOP, *flags);
    }
    for (i, flags) in ROTATIONS_YFLIP.iter().enumerate() {
        g.front_flags(7 + i as i32, yb, TILE_STOP, *flags);
    }
    for (i, flags) in ROTATIONS.iter().enumerate() {
        g.front_flags(12 + i as i32, yb, TILE_STOPS, *flags);
    }
    for (i, flags) in ROTATIONS_YFLIP.iter().enumerate() {
        g.front_flags(17 + i as i32, yb, TILE_STOPS, *flags);
    }
    g.front_flags(22, yb, TILE_STOPA, 0);

    // Row C: a solid pillar whose front-layer tile is plain (not THROUGH_CUT/ALL/DIR), with
    // bare `TILE_THROUGH` on all 4 neighbors — whichever side a hook approaches from, the
    // matching neighbor lets it pass through the solid center tile.
    let yc = 13;
    g.game(24, yc, TILE_SOLID);
    g.front_flags(23, yc, TILE_THROUGH, 0);
    g.front_flags(25, yc, TILE_THROUGH, 0);
    g.front_flags(24, yc - 1, TILE_THROUGH, 0);
    g.front_flags(24, yc + 1, TILE_THROUGH, 0);

    // Row D: bare game-layer THROUGH_ALL/THROUGH_DIR over otherwise-open air, for
    // `IsHookBlocker` (checked when the cell is *not* solid, unlike rows A-C).
    let yd = 16;
    g.game(2, yd, TILE_THROUGH_ALL);
    for (i, flags) in ROTATIONS.iter().enumerate() {
        g.game_flags(4 + i as i32, yd, TILE_THROUGH_DIR, *flags);
    }

    // Row E (review round 2, finding F12): front-layer THROUGH_DIR, one cell per rotation, over
    // otherwise-open air — the game layer at every one of these cells stays plain TILE_AIR.
    // Placed in the open room above the floor, well clear of rows A-D and every border/pillar,
    // so hooks can reach each cell from any of the 4 cardinal directions.
    let ye = 5;
    for (i, flags) in ROTATIONS.iter().enumerate() {
        g.front_flags(10 + i as i32, ye, TILE_THROUGH_DIR, *flags);
    }

    g.into_map()
}

/// A tele layer covering every tele tile type the core-level oracle and later the DDRace tile
/// port need (tele-in/out, evil, hook, and the four checkpoint variants), plus a speedup layer
/// covering both speedup tile generations across several angles/force/max-speed combinations.
fn tele_speedup() -> MapData {
    let mut g = Grid::new(34, 20);
    g.solid_border();
    g.row_solid(17, 1, 32);

    let y = 5;
    g.tele(2, y, 1, TILE_TELEIN);
    g.tele(3, y, 1, TILE_TELEOUT);
    g.tele(5, y, 2, TILE_TELEINEVIL);
    g.tele(6, y, 2, TILE_TELEOUT);
    g.tele(8, y, 3, TILE_TELEINHOOK);
    g.tele(9, y, 3, TILE_TELEOUT);
    g.tele(11, y, 4, TILE_TELECHECK);
    g.tele(12, y, 4, TILE_TELECHECKOUT);
    g.tele(14, y, 5, TILE_TELECHECKIN);
    g.tele(15, y, 5, TILE_TELECHECKOUT);
    g.tele(17, y, 6, TILE_TELECHECKINEVIL);
    g.tele(18, y, 6, TILE_TELECHECKOUT);

    let ys = 8;
    g.speedup(2, ys, 5, 0, TILE_SPEED_BOOST_OLD, 0);
    g.speedup(4, ys, 10, 50, TILE_SPEED_BOOST_OLD, 90);
    g.speedup(6, ys, 15, 0, TILE_SPEED_BOOST_OLD, 180);
    g.speedup(8, ys, 20, 30, TILE_SPEED_BOOST_OLD, 270);
    g.speedup(10, ys, 5, 0, TILE_SPEED_BOOST, 45);
    g.speedup(12, ys, 25, 100, TILE_SPEED_BOOST, -45);
    g.speedup(14, ys, 8, 0, TILE_SPEED_BOOST, 135);
    g.speedup(16, ys, 12, 60, TILE_SPEED_BOOST, -135);

    g.into_map()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::map::TILE_AIR;

    #[test]
    fn build_returns_none_for_unknown_recipe() {
        assert!(build("no-such-recipe").is_none());
    }

    #[test]
    fn every_listed_recipe_builds_a_valid_map() {
        for name in RECIPES {
            let map = build(name).unwrap_or_else(|| panic!("recipe '{name}' failed to build"));
            map.validate()
                .unwrap_or_else(|e| panic!("recipe '{name}' built an invalid map: {e}"));
        }
    }

    #[test]
    fn recipes_are_deterministic() {
        for name in RECIPES {
            assert_eq!(build(name), build(name), "recipe '{name}' is not deterministic");
        }
    }

    /// Every recipe must leave enough open (`TILE_AIR`, both game and front) cells to spawn the
    /// maximum 4 characters the generator supports.
    #[test]
    fn every_recipe_has_at_least_four_free_cells() {
        for name in RECIPES {
            let map = build(name).unwrap();
            let free = (0..map.cell_count())
                .filter(|&i| {
                    map.game[i].index == TILE_AIR && map.front.as_ref().map(|f| f[i].index == TILE_AIR).unwrap_or(true)
                })
                .count();
            assert!(free >= 4, "recipe '{name}' has only {free} free cells, need at least 4");
        }
    }

    #[test]
    fn arena_has_solid_border() {
        let map = arena();
        let w = map.width as usize;
        for x in 0..w {
            assert_eq!(map.game[x].index, TILE_SOLID, "top border not solid at x={x}");
            assert_eq!(
                map.game[(map.height as usize - 1) * w + x].index,
                TILE_SOLID,
                "bottom border not solid at x={x}"
            );
        }
    }

    /// Collects every front-layer tile equal to `wanted_index`, as `(x, y, flags)`.
    fn find_front(map: &[Tile], width: i32, wanted_index: u8) -> Vec<(i32, i32, u8)> {
        map.iter()
            .enumerate()
            .filter(|(_, t)| t.index == wanted_index)
            .map(|(i, t)| (i as i32 % width, i as i32 / width, t.flags))
            .collect()
    }

    fn sorted_flags(v: &[(i32, i32, u8)]) -> Vec<u8> {
        let mut f: Vec<u8> = v.iter().map(|(_, _, f)| *f).collect();
        f.sort_unstable();
        f
    }

    #[test]
    fn front_recipe_covers_every_stop_and_stops_flag_byte_including_yflip_variants() {
        let map = front().front.unwrap();
        let width = 30;
        let seen_stop = find_front(&map, width, TILE_STOP);
        let seen_stops = find_front(&map, width, TILE_STOPS);
        assert_eq!(seen_stop.len(), 8, "expected 4 plain + 4 YFLIP TILE_STOP cells");
        assert_eq!(seen_stops.len(), 8, "expected 4 plain + 4 YFLIP TILE_STOPS cells");

        let mut expected: Vec<u8> = ROTATIONS.iter().chain(ROTATIONS_YFLIP.iter()).copied().collect();
        expected.sort_unstable();
        assert_eq!(sorted_flags(&seen_stop), expected);
        assert_eq!(sorted_flags(&seen_stops), expected);
    }

    #[test]
    fn front_recipe_puts_every_hook_through_rotation_over_a_solid_game_tile() {
        // Review round 1, finding F1: THROUGH_CUT/ALL/DIR on the front layer only ever reach
        // `IsThrough` when the game layer at that same cell is solid (`CheckPoint` true) —
        // otherwise the hook line never registers a hit there to begin with. Scoped to row A
        // (y=8): row E (review round 2, finding F12, see
        // `front_recipe_has_front_layer_through_dir_over_open_air`) deliberately puts
        // THROUGH_DIR on the front layer over open air instead, so it must be excluded here.
        const ROW_A_Y: i32 = 8;
        let map = front();
        let front_tiles = map.front.as_ref().unwrap();
        let width = map.width as i32;
        for index in [TILE_THROUGH_CUT, TILE_THROUGH_ALL, TILE_THROUGH_DIR] {
            let cells: Vec<(i32, i32, u8)> = find_front(front_tiles, width, index)
                .into_iter()
                .filter(|&(_, y, _)| y == ROW_A_Y)
                .collect();
            assert!(!cells.is_empty(), "expected at least one row-A front-layer {index}");
            for (x, y, _) in cells {
                let game_index = map.game[(y * width + x) as usize].index;
                assert!(
                    game_index == TILE_SOLID,
                    "front tile {index} at ({x},{y}) must sit over TILE_SOLID, found {game_index}"
                );
            }
        }
        let through_dir_cells: Vec<(i32, i32, u8)> = find_front(front_tiles, width, TILE_THROUGH_DIR)
            .into_iter()
            .filter(|&(_, y, _)| y == ROW_A_Y)
            .collect();
        assert_eq!(
            through_dir_cells.len(),
            ROTATIONS.len(),
            "expected one row-A THROUGH_DIR per rotation"
        );
        let mut expected = ROTATIONS.to_vec();
        expected.sort_unstable();
        assert_eq!(sorted_flags(&through_dir_cells), expected);
    }

    #[test]
    fn front_recipe_has_front_layer_through_dir_over_open_air() {
        // Review round 2, finding F12: `IsHookBlocker`'s front-layer THROUGH_DIR branch
        // (`collision.cpp:632-633`) is only reachable when the GAME layer at that cell is open
        // (`CheckPoint` false, unlike row A) while the FRONT layer itself carries THROUGH_DIR —
        // row D (game-layer THROUGH_ALL/DIR over air) covers `IsHookBlocker`'s other two
        // branches (`collision.cpp:625`, `:627-631`) but never this one. Row E sets this up.
        let map = front();
        let front_tiles = map.front.as_ref().unwrap();
        let width = map.width as i32;
        let through_dir: Vec<(i32, i32, u8)> = find_front(front_tiles, width, TILE_THROUGH_DIR)
            .into_iter()
            .filter(|&(x, y, _)| map.game[(y * width + x) as usize].index == TILE_AIR)
            .collect();
        assert_eq!(
            through_dir.len(),
            ROTATIONS.len(),
            "expected one front-layer THROUGH_DIR per rotation sitting over open (TILE_AIR) game"
        );
        let mut expected = ROTATIONS.to_vec();
        expected.sort_unstable();
        assert_eq!(sorted_flags(&through_dir), expected);

        // Reachable from all 4 cardinal directions: every immediate neighbor (including another
        // row-E cell) is open air on the game layer, not tucked against a wall/border/pillar.
        for &(x, y, _) in &through_dir {
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                let neighbor_index = map.game[(ny * width + nx) as usize].index;
                assert_eq!(
                    neighbor_index, TILE_AIR,
                    "cell ({x},{y})'s neighbor ({nx},{ny}) must be open air, found {neighbor_index}"
                );
            }
        }
    }

    #[test]
    fn front_recipe_has_a_through_tile_on_every_side_of_a_solid_pillar() {
        // Review round 1, finding F1: exercises `IsThrough`'s *other* branch — a solid tile that
        // isn't itself THROUGH_CUT/ALL/DIR, but has a bare TILE_THROUGH on the side the hook is
        // travelling toward (`collision.cpp`'s `IsThrough`, the `OffsetIndex` check).
        let map = front();
        let front_tiles = map.front.as_ref().unwrap();
        let width = map.width as i32;
        let pillar = find_front(front_tiles, width, TILE_SOLID); // TILE_SOLID is never placed on
        // the front layer by this recipe elsewhere, so any hit here would be a bug; assert none.
        assert!(pillar.is_empty(), "TILE_SOLID must never appear on the front layer");

        let through_cells = find_front(front_tiles, width, TILE_THROUGH);
        assert_eq!(
            through_cells.len(),
            4,
            "expected exactly 4 plain TILE_THROUGH neighbor cells"
        );
        let (cx, cy, _) = through_cells[0];
        // The 4 TILE_THROUGH cells must surround exactly one solid game-layer center tile.
        let centers: std::collections::BTreeSet<(i32, i32)> = through_cells
            .iter()
            .map(|(x, y, _)| {
                // Each TILE_THROUGH cell is adjacent (offset by exactly one tile on one axis) to
                // the shared solid center; find that center by checking all 4 neighbors.
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if map.game[(ny * width + nx) as usize].index == TILE_SOLID {
                        return (nx, ny);
                    }
                }
                (*x, *y)
            })
            .collect();
        assert_eq!(
            centers.len(),
            1,
            "all 4 TILE_THROUGH cells must surround the same solid center, got {centers:?}"
        );
        let (center_x, center_y) = *centers.iter().next().unwrap();
        assert_eq!(map.game[(center_y * width + center_x) as usize].index, TILE_SOLID);
        // Sanity: the first TILE_THROUGH cell really is adjacent to that center (not e.g. two
        // tiles away).
        assert!((cx - center_x).abs() + (cy - center_y).abs() == 1);
    }

    #[test]
    fn front_recipe_has_bare_through_all_and_through_dir_on_the_open_game_layer() {
        // Review round 1, finding F1: `IsHookBlocker` is reached when the cell is *not* solid
        // (`CheckPoint` false) but the raw game-layer tile id is THROUGH_ALL/THROUGH_DIR anyway.
        let map = front();
        let width = map.width as i32;
        let through_all: Vec<(i32, i32, u8)> = map
            .game
            .iter()
            .enumerate()
            .filter(|(_, t)| t.index == TILE_THROUGH_ALL)
            .map(|(i, t)| (i as i32 % width, i as i32 / width, t.flags))
            .collect();
        assert_eq!(
            through_all.len(),
            1,
            "expected exactly one bare game-layer TILE_THROUGH_ALL"
        );

        let through_dir: Vec<(i32, i32, u8)> = map
            .game
            .iter()
            .enumerate()
            .filter(|(_, t)| t.index == TILE_THROUGH_DIR)
            .map(|(i, t)| (i as i32 % width, i as i32 / width, t.flags))
            .collect();
        assert_eq!(
            through_dir.len(),
            ROTATIONS.len(),
            "expected one bare game-layer TILE_THROUGH_DIR per rotation"
        );
        let mut expected = ROTATIONS.to_vec();
        expected.sort_unstable();
        let mut got: Vec<u8> = through_dir.iter().map(|(_, _, f)| *f).collect();
        got.sort_unstable();
        assert_eq!(got, expected);

        // These must be genuinely "open" per `GetTile()` (i.e. NOT `IsSolid`), unlike rows A-C —
        // that's the whole point (`IsHookBlocker` is only consulted when `CheckPoint` is false).
        // `GetTile()` only recognizes indices in `[TILE_SOLID, TILE_NOLASER]` = `1..=4` as
        // solid, and THROUGH_ALL (66) / THROUGH_DIR (67) both fall outside that range.
        for (x, y, _) in through_all.iter().chain(through_dir.iter()) {
            let idx = map.game[(y * width + x) as usize].index;
            assert!(
                !(1..=4).contains(&idx),
                "expected an id outside [TILE_SOLID, TILE_NOLASER] (1..=4), got {idx}"
            );
        }
    }

    #[test]
    fn tele_speedup_recipe_has_matching_tele_numbers() {
        let map = tele_speedup();
        let tele = map.tele.unwrap();
        use std::collections::BTreeMap;
        let mut by_number: BTreeMap<u8, Vec<u8>> = BTreeMap::new();
        for t in &tele {
            if t.number != 0 {
                by_number.entry(t.number).or_default().push(t.kind);
            }
        }
        assert_eq!(by_number.len(), 6, "expected 6 distinct tele numbers");
        for (number, kinds) in &by_number {
            assert!(
                kinds.contains(&TILE_TELEOUT) || kinds.contains(&TILE_TELECHECKOUT),
                "number {number} has no out/checkout"
            );
        }
    }

    #[test]
    fn tele_speedup_recipe_covers_both_speedup_generations() {
        let map = tele_speedup();
        let speedup = map.speedup.unwrap();
        assert!(speedup.iter().any(|s| s.kind == TILE_SPEED_BOOST_OLD && s.force > 0));
        assert!(speedup.iter().any(|s| s.kind == TILE_SPEED_BOOST && s.force > 0));
        assert!(speedup.iter().any(|s| s.angle < 0));
    }
}
