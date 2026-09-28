//! Dev tool (not part of the published crate, only built with `cargo run --example`): writes the
//! synthetic recipes `ddai_trace::synthetic` defines (`arena`, `freeze`, `front`, `tele-speedup`)
//! out as REAL DDNet `.map` datafiles, so both the real TS `src/map/loadMap.ts` and this crate's
//! own `map_load` can load the exact same bytes through their real, file-based loaders — see
//! `crate::map_load`'s doc comment and the crate README's "Синтетические карты как настоящие
//! .map-файлы" for why this crate does not special-case synthetic maps at all.
//!
//! Usage: `cargo run --example gen_fixture_maps -- <output-dir>` (default:
//! `~/aiddnet/data/maps/synthetic`, matching where `docs/STATUS.md`/the task spec says generated
//! map data lives — never committed to the repository).

use ddai_map::testutil::{MapWriter, TileLayerSpec, TilemapShape, encode_tile_skip};
use ddai_physics::map::MapData;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Encodes a `Vec<Tile>` as plain (non-tile-skip) `CTile` bytes: `(index, flags, 0, 0)` per
/// cell — required for the front layer (`ddai-map`'s loader rejects tile-skip-encoded front
/// layers, matching real DDNet — see `crate::map_load`'s doc comment) and used here for the game
/// layer too when we want the plain-encoding path exercised instead of tile-skip.
fn plain_tiles(tiles: &[ddai_physics::map::Tile]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tiles.len() * 4);
    for t in tiles {
        out.push(t.index);
        out.push(t.flags);
        out.push(0);
        out.push(0);
    }
    out
}

fn skip_tiles(tiles: &[ddai_physics::map::Tile]) -> Vec<u8> {
    let pairs: Vec<(u8, u8)> = tiles.iter().map(|t| (t.index, t.flags)).collect();
    encode_tile_skip(&pairs)
}

fn write_map(data: &MapData, out_path: &Path) {
    let mut w = MapWriter::new(4);
    w.add_version_item(1);
    w.add_info_item(Some("ddai-tsworld"), None, None, None, &[]);

    let width = data.width as i32;
    let height = data.height as i32;

    // Game layer: tile-skip-encoded, item_version 4 — exercises the same RLE-unpack path a real
    // map's game layer does (`VERSION_TEEWORLDS_TILESKIP` in TS, `allow_skip` in `ddai-map`).
    let game_bytes = skip_tiles(&data.game);
    w.add_tile_layer(&TileLayerSpec {
        shape: TilemapShape::Full,
        item_version: 4,
        width,
        height,
        flags: ddai_map::testutil::TILESLAYERFLAG_GAME,
        data: &game_bytes,
    });

    if let Some(front) = &data.front {
        // Front layers are never tile-skip-encoded (`ddai-map` rejects that combination,
        // matching real DDNet — see `crate::map_load`), so item_version 3, plain per-cell bytes.
        let front_bytes = plain_tiles(front);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width,
            height,
            flags: ddai_map::testutil::TILESLAYERFLAG_FRONT,
            data: &front_bytes,
        });
    }

    if let Some(tele) = &data.tele {
        let mut bytes = Vec::with_capacity(tele.len() * 2);
        for t in tele {
            bytes.push(t.number);
            bytes.push(t.kind);
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width,
            height,
            flags: ddai_map::testutil::TILESLAYERFLAG_TELE,
            data: &bytes,
        });
    }

    if let Some(speedup) = &data.speedup {
        let mut bytes = Vec::with_capacity(speedup.len() * 6);
        for s in speedup {
            bytes.push(s.force);
            bytes.push(s.max_speed);
            bytes.push(s.kind);
            bytes.push(0);
            bytes.extend_from_slice(&s.angle.to_le_bytes());
        }
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width,
            height,
            flags: ddai_map::testutil::TILESLAYERFLAG_SPEEDUP,
            data: &bytes,
        });
    }

    w.add_single_group_with_all_layers();
    let bytes = w.finish();

    std::fs::create_dir_all(out_path.parent().unwrap()).expect("mkdir");
    let mut f = std::fs::File::create(out_path).expect("create map file");
    f.write_all(&bytes).expect("write map file");
    println!("wrote {} ({} bytes)", out_path.display(), bytes.len());
}

fn main() {
    let out_dir: PathBuf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        PathBuf::from(home).join("aiddnet/data/maps/synthetic")
    });

    for name in ddai_trace::synthetic::RECIPES {
        let data = ddai_trace::synthetic::build(name).expect("recipe exists");
        let out_path = out_dir.join(format!("{name}.map"));
        write_map(&data, &out_path);
    }
}
