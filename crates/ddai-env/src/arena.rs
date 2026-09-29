//! Arenas: a map, a spawn rule and a train/holdout tag, all defined as data (TOML) -- a literal
//! port of the phase-0 harness's `arenaPit`/`arenaPlatform`/`arenaFromMap` (`lib.mjs`) including
//! the spawn rules (`orig-run.md` §4), plus real-map arenas verified by sha256.
//!
//! The spawn RNG is the harness's: `Rng(seed * 2654435761 >>> 0)` (`ddai_jsmath::Rng`), two
//! uniform picks from the standing-slot list per attempt, rejected until their tile distance is in
//! `[min_tiles, max_tiles]`. With two players the sequence of RNG draws is therefore exactly the
//! harness's; further players (1vN) are drawn one at a time against player 0.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_jsmath as js;
use ddai_jsmath::Rng;
use ddai_physics::map::{MapData, TILE_DEATH, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID, TILE_UNFREEZE, Tile};
use ddai_physics::world::World;
use ddai_planner::plan_world::PlanCollision;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::EnvError;

/// Whether an arena may be used to tune (`Train`) or only to evaluate (`Holdout`) a brain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Holdout,
}

impl Split {
    pub fn label(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Holdout => "holdout",
        }
    }
}

/// An axis-aligned tile rectangle, inclusive on both ends (`lib.mjs` `rect`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

/// One filled rectangle of a synthetic map; later rectangles overwrite earlier ones.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    /// `solid`, `freeze`, `death`, `nohook`, `unfreeze` or `air`.
    pub tile: String,
}

/// A synthetic map built in code (never a file).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDef {
    pub width: i32,
    pub height: i32,
    /// Solid one-tile frame around the map (`lib.mjs` `border()`), drawn before `rects`.
    #[serde(default)]
    pub border: bool,
    #[serde(default)]
    pub rects: Vec<Rect>,
}

/// A real map file, verified by sha256.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileMapDef {
    /// Path relative to the map directory (`--map-dir`, default `~/aiddnet/data/maps`).
    pub file: String,
    /// Lower-case hex sha256 of the file; loading fails on a mismatch.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum MapSource {
    Synthetic(SyntheticDef),
    File(FileMapDef),
}

/// One row of explicit standing slots: tiles `x0..=x1` on row `y` (synthetic arenas).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnRow {
    pub y: i32,
    pub x0: i32,
    pub x1: i32,
}

/// Where tees may spawn and how far apart.
///
/// Either `rows` (explicit standing slots, synthetic arenas) or `boxes` (every *standing* tile
/// inside the boxes: tile centre neither solid, freeze nor death, with solid ground directly
/// below -- `arenaFromMap`'s rule) is given.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnDef {
    pub min_tiles: f64,
    pub max_tiles: f64,
    #[serde(default)]
    pub rows: Vec<SpawnRow>,
    #[serde(default)]
    pub boxes: Vec<TileBox>,
}

/// An arena definition as written in `configs/arenas/*.toml`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArenaDef {
    pub name: String,
    pub tag: Split,
    /// Human-readable notes (what the area is, why the box was chosen).
    #[serde(default)]
    pub description: String,
    pub map: MapSource,
    pub spawn: SpawnDef,
}

impl ArenaDef {
    pub fn parse(text: &str) -> Result<ArenaDef, EnvError> {
        toml::from_str(text).map_err(|e| EnvError::new(format!("arena definition: {e}")))
    }
}

/// Reads every `*.toml` in `dir` as an [`ArenaDef`], keyed by name (sorted, so listings are
/// stable). A duplicate name is an error.
pub fn load_arena_defs(dir: &Path) -> Result<BTreeMap<String, ArenaDef>, EnvError> {
    let mut out = BTreeMap::new();
    let entries =
        std::fs::read_dir(dir).map_err(|e| EnvError::new(format!("reading arena dir {}: {e}", dir.display())))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    paths.sort();
    for path in paths {
        let text = std::fs::read_to_string(&path).map_err(|e| EnvError::new(format!("{}: {e}", path.display())))?;
        let def = ArenaDef::parse(&text).map_err(|e| EnvError::new(format!("{}: {e}", path.display())))?;
        if out.insert(def.name.clone(), def).is_some() {
            return Err(EnvError::new(format!("duplicate arena name in {}", path.display())));
        }
    }
    Ok(out)
}

fn tile_id(name: &str) -> Result<u8, EnvError> {
    Ok(match name {
        "air" => 0,
        "solid" => TILE_SOLID,
        "death" => TILE_DEATH,
        "nohook" => TILE_NOHOOK,
        "freeze" => TILE_FREEZE,
        "unfreeze" => TILE_UNFREEZE,
        other => return Err(EnvError::new(format!("unknown tile name {other:?}"))),
    })
}

/// Builds the game layer of a synthetic map (`lib.mjs` `grid`/`rect`/`border`; writes outside the
/// map are ignored, exactly like `set`).
pub fn synthetic_map(def: &SyntheticDef) -> Result<MapData, EnvError> {
    if def.width <= 0 || def.height <= 0 {
        return Err(EnvError::new("synthetic map needs a positive size"));
    }
    let (w, h) = (def.width, def.height);
    let mut game = vec![Tile::default(); (w * h) as usize];
    let mut fill = |x0: i32, y0: i32, x1: i32, y1: i32, index: u8| {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if x >= 0 && y >= 0 && x < w && y < h {
                    game[(y * w + x) as usize] = Tile {
                        index,
                        ..Tile::default()
                    };
                }
            }
        }
    };
    if def.border {
        fill(0, 0, w - 1, 0, TILE_SOLID);
        fill(0, h - 1, w - 1, h - 1, TILE_SOLID);
        fill(0, 0, 0, h - 1, TILE_SOLID);
        fill(w - 1, 0, w - 1, h - 1, TILE_SOLID);
    }
    for r in &def.rects {
        fill(r.x0, r.y0, r.x1, r.y1, tile_id(&r.tile)?);
    }
    Ok(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Lower-case hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// A tile centre in world pixels.
pub fn tile_center(t: i32) -> f32 {
    t as f32 * 32.0 + 16.0
}

/// A map turned into an empty template `World<f32>`.
pub struct BuiltWorld {
    pub map: Arc<MapData>,
    pub world: World<f32>,
    /// `Some` for a real map file.
    pub sha256: Option<String>,
    /// File path of a real map, `"synthetic"` otherwise.
    pub source: String,
}

/// Builds (and, for a file, verifies by sha256) a map and its template world. `owner` names the
/// arena/scenario in error messages.
pub fn build_world(source: &MapSource, map_dir: &Path, owner: &str) -> Result<BuiltWorld, EnvError> {
    let (map, sha256, source) = match source {
        MapSource::Synthetic(s) => (synthetic_map(s)?, None, "synthetic".to_string()),
        MapSource::File(f) => {
            let path = map_dir.join(&f.file);
            let bytes = std::fs::read(&path).map_err(|e| {
                EnvError::new(format!(
                    "{owner}: reading map {}: {e} (maps are local data, see --map-dir)",
                    path.display()
                ))
            })?;
            let actual = sha256_hex(&bytes);
            if !actual.eq_ignore_ascii_case(&f.sha256) {
                return Err(EnvError::new(format!(
                    "{owner}: map {} has sha256 {actual}, expected {}",
                    path.display(),
                    f.sha256
                )));
            }
            let loaded = ddai_map::load_map(&bytes)
                .map_err(|e| EnvError::new(format!("{owner}: parsing {}: {e}", path.display())))?;
            (loaded.data, Some(actual), path.display().to_string())
        }
    };
    let map = Arc::new(map);
    let mut world = World::<f32>::from_map(&map, 1);
    // Map settings may contain commands this port does not model; the phase-3 planner tests
    // ignore that result the same way.
    let _ = world.init(std::iter::empty::<&str>());
    Ok(BuiltWorld {
        map,
        world,
        sha256,
        source,
    })
}

/// An arena ready to play on: the map, one template `World<f32>` (cloned per game -- building a
/// world scans the whole map) and the standing slots.
#[derive(Clone)]
pub struct Arena {
    pub name: String,
    pub tag: Split,
    pub description: String,
    pub map: Arc<MapData>,
    /// `Some` for a real map; `None` for a synthetic one.
    pub map_sha256: Option<String>,
    /// File path of a real map (for the run record), `"synthetic"` otherwise.
    pub map_source: String,
    template: World<f32>,
    slots: Vec<(i32, i32)>,
    min_tiles: f64,
    max_tiles: f64,
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena")
            .field("name", &self.name)
            .field("tag", &self.tag)
            .field("slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl Arena {
    /// Builds the arena; a real map is read from `map_dir` and must match its declared sha256.
    pub fn build(def: &ArenaDef, map_dir: &Path) -> Result<Arena, EnvError> {
        let built = build_world(&def.map, map_dir, &def.name)?;
        let (map, template, map_sha256, map_source) = (built.map, built.world, built.sha256, built.source);
        let slots = if def.spawn.rows.is_empty() {
            standing_tiles(
                &*template.collision,
                map.width as i32,
                map.height as i32,
                &def.spawn.boxes,
            )
        } else {
            def.spawn
                .rows
                .iter()
                .flat_map(|r| (r.x0..=r.x1).map(move |x| (x, r.y)))
                .collect()
        };
        if slots.len() < 2 {
            return Err(EnvError::new(format!(
                "arena {}: fewer than two standing slots",
                def.name
            )));
        }
        Ok(Arena {
            name: def.name.clone(),
            tag: def.tag,
            description: def.description.clone(),
            map,
            map_sha256,
            map_source,
            template,
            slots,
            min_tiles: def.spawn.min_tiles,
            max_tiles: def.spawn.max_tiles,
        })
    }

    /// A fresh copy of the arena's empty world (no tees).
    pub fn new_world(&self) -> World<f32> {
        self.template.clone()
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn slots(&self) -> &[(i32, i32)] {
        &self.slots
    }

    fn within(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        let d = js::hypot2(f64::from(a.0 - b.0), f64::from(a.1 - b.1));
        d >= self.min_tiles && d <= self.max_tiles
    }

    /// Spawn tiles for `players` tees, deterministic in `seed` (see the module docs).
    ///
    /// The first two tees are placed exactly like the harness. Every further tee (1vN) must be
    /// within `[min_tiles, max_tiles]` of player 0 and at least `min_tiles` from every tee already
    /// placed, so no two players start on (or next to) the same tile. If some tee cannot be
    /// placed given the first two, the whole placement is redrawn (for two players that never
    /// happens, so their stream is the harness's).
    pub fn spawn_tiles(&self, seed: u64, players: usize) -> Result<Vec<(i32, i32)>, EnvError> {
        let mut rng = Rng::new(seed.wrapping_mul(2_654_435_761) as u32);
        let n = self.slots.len();
        let pick = |rng: &mut Rng| self.slots[js::floor(rng.next_float() * n as f64) as usize];
        let dist = |a: (i32, i32), b: (i32, i32)| js::hypot2(f64::from(a.0 - b.0), f64::from(a.1 - b.1));
        if players == 0 {
            return Ok(Vec::new());
        }
        if players == 1 {
            return Ok(vec![pick(&mut rng)]);
        }
        'redraw: for _ in 0..1_000 {
            let mut out = Vec::with_capacity(players);
            let mut placed = false;
            for _ in 0..100_000 {
                let a = pick(&mut rng);
                let b = pick(&mut rng);
                if self.within(a, b) {
                    out.push(a);
                    out.push(b);
                    placed = true;
                    break;
                }
            }
            if !placed {
                return Err(EnvError::new(format!("arena {}: could not place two tees", self.name)));
            }
            while out.len() < players {
                let fits =
                    |c: (i32, i32)| self.within(out[0], c) && out[1..].iter().all(|&t| dist(t, c) >= self.min_tiles);
                let mut next = None;
                for _ in 0..10_000 {
                    let c = pick(&mut rng);
                    if fits(c) {
                        next = Some(c);
                        break;
                    }
                }
                match next {
                    Some(c) => out.push(c),
                    None => continue 'redraw,
                }
            }
            return Ok(out);
        }
        Err(EnvError::new(format!(
            "arena {}: could not place {players} tees at least {} tiles apart",
            self.name, self.min_tiles
        )))
    }
}

/// `arenaFromMap`'s standing-tile scan: inside each box (rows clamped to `height - 2`, columns to
/// `width - 1`), a tile counts when its centre is neither solid, freeze nor death and the point
/// 32 px below the centre is solid. Boxes are scanned in order, row-major, first occurrence wins.
fn standing_tiles(col: &impl PlanCollision, width: i32, height: i32, boxes: &[TileBox]) -> Vec<(i32, i32)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for b in boxes {
        for ty in b.y0.max(0)..=b.y1.min(height - 2) {
            for tx in b.x0.max(0)..=b.x1.min(width - 1) {
                let px = f64::from(tile_center(tx));
                let py = f64::from(tile_center(ty));
                if col.is_solid(px, py) || col.is_freeze(px, py) || col.is_death(px, py) {
                    continue;
                }
                if !col.is_solid(px, py + 32.0) {
                    continue;
                }
                if seen.insert((tx, ty)) {
                    out.push((tx, ty));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIT: &str = include_str!("../../../configs/arenas/pit.toml");

    #[test]
    fn synthetic_pit_matches_the_harness_layout() {
        let def = ArenaDef::parse(PIT).unwrap();
        let MapSource::Synthetic(s) = &def.map else {
            panic!("pit is synthetic")
        };
        let m = synthetic_map(s).unwrap();
        assert_eq!((m.width, m.height), (48, 24));
        let at = |x: i32, y: i32| m.game[(y * 48 + x) as usize].index;
        assert_eq!(at(0, 10), TILE_SOLID, "left wall");
        assert_eq!(at(10, 3), TILE_SOLID, "thick ceiling");
        assert_eq!(at(10, 6), 0, "air below the ceiling underside (row 6)");
        assert_eq!(at(10, 17), TILE_SOLID, "floor");
        assert_eq!(at(21, 17), TILE_FREEZE, "pit top row is flush with the floor");
        assert_eq!(at(26, 21), TILE_FREEZE);
        assert_eq!(at(27, 17), TILE_SOLID);
        assert_eq!(at(23, 22), TILE_SOLID, "pit bottom");
    }

    #[test]
    fn spawn_is_deterministic_and_respects_the_distance_rule() {
        let def = ArenaDef::parse(PIT).unwrap();
        let arena = Arena::build(&def, Path::new("/nonexistent")).unwrap();
        // pit slots: x in 2..=19 and 28..=45 on row 16.
        assert_eq!(arena.slot_count(), 18 + 18);
        for seed in 0..200u64 {
            let a = arena.spawn_tiles(seed, 2).unwrap();
            assert_eq!(a, arena.spawn_tiles(seed, 2).unwrap());
            assert_eq!(a[0].1, 16);
            let d = (a[0].0 - a[1].0).abs();
            assert!((4..=20).contains(&d), "seed {seed}: distance {d}");
            assert!(a.iter().all(|t| arena.slots().contains(t)));
        }
        let all: HashSet<_> = (0..50u64).map(|s| arena.spawn_tiles(s, 2).unwrap()).collect();
        assert!(all.len() > 20, "different seeds must give different spawns");
    }

    #[test]
    fn extra_players_are_spread_out_and_near_player_zero() {
        let def = ArenaDef::parse(PIT).unwrap();
        let arena = Arena::build(&def, Path::new("/nonexistent")).unwrap();
        for players in [3usize, 4] {
            for seed in 0..2_000u64 {
                let s = arena.spawn_tiles(seed, players).unwrap();
                assert_eq!(s.len(), players);
                assert_eq!(s, arena.spawn_tiles(seed, players).unwrap(), "deterministic");
                for (i, &t) in s.iter().enumerate() {
                    for &u in &s[..i] {
                        let d = f64::from(t.0 - u.0).hypot(f64::from(t.1 - u.1));
                        assert!(
                            d >= 4.0,
                            "seed {seed}: two of {players} tees only {d} tiles apart: {s:?}"
                        );
                    }
                    if i > 0 {
                        assert!(
                            (s[0].0 - t.0).abs() <= 20,
                            "seed {seed}: tee {i} too far from player 0: {s:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unknown_fields_and_tiles_are_rejected() {
        assert!(ArenaDef::parse("name = \"x\"\ntag = \"train\"\nbogus = 1\n").is_err());
        let bad = SyntheticDef {
            width: 4,
            height: 4,
            border: false,
            rects: vec![Rect {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
                tile: "lava".into(),
            }],
        };
        assert!(synthetic_map(&bad).is_err());
    }

    #[test]
    fn sha256_mismatch_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m.map"), b"not a map").unwrap();
        let def = ArenaDef {
            name: "bad".into(),
            tag: Split::Holdout,
            description: String::new(),
            map: MapSource::File(FileMapDef {
                file: "m.map".into(),
                sha256: "00".repeat(32),
            }),
            spawn: SpawnDef {
                min_tiles: 3.0,
                max_tiles: 12.0,
                rows: vec![],
                boxes: vec![TileBox {
                    x0: 0,
                    y0: 0,
                    x1: 5,
                    y1: 5,
                }],
            },
        };
        let err = Arena::build(&def, dir.path()).unwrap_err().to_string();
        assert!(err.contains("sha256"), "{err}");
    }
}
