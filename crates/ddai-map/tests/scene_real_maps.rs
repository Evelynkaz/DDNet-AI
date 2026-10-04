//! The visual scene reader against the real maps of this project (Copy Love Box in two versions, including Swarfey's
//! 468x255 copy, BlmapChill, ChillBlock5). The maps are not in the repository (`CLAUDE.md`: never commit maps), so each
//! test skips with a message when the map is not on this machine; point `DDAI_MAPS_DIR` at the directory that holds
//! `cache/` (default `~/aiddnet/data/maps`) to run them.
//!
//! The key cross-check: the game layer the scene reader returns must equal the one `load_map` returns, which is proven
//! byte for byte against DDNet's own loader (`tools/ddnet-oracle/map-corpus-check.sh`).

use ddai_map::load_map;
use ddai_map::scene::{Layer, TileRole, VisualScene, extract_visual_scene};
use std::path::PathBuf;

fn maps_dir() -> PathBuf {
    std::env::var_os("DDAI_MAPS_DIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
            .join("aiddnet")
            .join("data")
            .join("maps")
    })
}

/// The bytes of the cached map whose file name starts with `prefix` (names are `<map>_<sha256>.map`), or `None`.
fn find(prefix: &str) -> Option<Vec<u8>> {
    let dir = maps_dir().join("cache");
    let entry = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with(prefix))?;
    std::fs::read(entry.path()).ok()
}

fn with_map(prefix: &str, check: impl FnOnce(&[u8], VisualScene)) {
    let Some(bytes) = find(prefix) else {
        eprintln!(
            "skipped: no cached map starting with {prefix:?} under {}",
            maps_dir().display()
        );
        return;
    };
    let scene = extract_visual_scene(&bytes).expect("a real map yields a scene");
    assert_eq!(scene.skipped, 0, "nothing in a real map may be dropped");
    // The game layer is the one the physics loader sees.
    let loaded = load_map(&bytes).expect("load_map");
    let game = scene.game_layer().expect("a real map has a game layer");
    assert_eq!((game.width, game.height), (loaded.data.width, loaded.data.height));
    let physics: Vec<u8> = loaded.data.game.iter().flat_map(|t| [t.index, t.flags]).collect();
    assert_eq!(game.tiles, physics, "scene game layer == physics game layer");
    check(&bytes, scene);
}

fn layers(scene: &VisualScene) -> Vec<&Layer> {
    scene.groups.iter().flat_map(|g| g.layers.iter()).collect()
}

fn external_names(scene: &VisualScene) -> Vec<&str> {
    scene
        .images
        .iter()
        .filter(|i| i.external)
        .map(|i| i.name.as_str())
        .collect()
}

#[test]
fn copy_love_box_current_version() {
    with_map("Copy Love Box_6e79ef", |_, scene| {
        assert_eq!(
            (scene.groups.len(), scene.images.len(), scene.envelopes.len()),
            (7, 15, 13)
        );
        let game = scene.game_layer().unwrap();
        assert_eq!((game.width, game.height), (387, 250));
        assert_eq!(external_names(&scene), ["grass_main", "moon"]);
        // Embedded images carry their pixels, external ones do not.
        assert!(scene.images.iter().all(|i| i.external == i.rgba.is_none()));
        assert!(scene.images.iter().filter(|i| !i.external).all(|i| {
            i.rgba
                .as_ref()
                .is_some_and(|p| p.len() == (i.width * i.height * 4) as usize)
        }));
        // Draw order: the parallax-0 sky group first, the entity layers in the last group, after every design layer of it.
        assert_eq!((scene.groups[0].parallax_x, scene.groups[0].parallax_y), (0, 0));
        let last = &scene.groups[6];
        let roles: Vec<TileRole> = last
            .layers
            .iter()
            .filter_map(|l| match l {
                Layer::Tiles(t) => Some(t.role),
                Layer::Quads(_) => None,
            })
            .collect();
        assert_eq!(roles[0], TileRole::Game);
        assert!(roles.contains(&TileRole::Tele) && roles.contains(&TileRole::Speedup));
        // Quads: the heart layer has 583 of them.
        assert!(
            layers(&scene)
                .iter()
                .any(|l| matches!(l, Layer::Quads(q) if q.quads.len() == 583))
        );
    });
}

#[test]
fn copy_love_box_swarfeys_468x255_copy() {
    with_map("Copy Love Box_23f188", |_, scene| {
        assert_eq!(
            (scene.groups.len(), scene.images.len(), scene.envelopes.len()),
            (9, 13, 3)
        );
        let game = scene.game_layer().unwrap();
        assert_eq!((game.width, game.height), (468, 255));
        assert_eq!(external_names(&scene), ["moon"]);
        // This copy has a front, switch and tune layer next to the game layer.
        let roles: Vec<TileRole> = layers(&scene)
            .iter()
            .filter_map(|l| match l {
                Layer::Tiles(t) => Some(t.role),
                Layer::Quads(_) => None,
            })
            .collect();
        for role in [
            TileRole::Front,
            TileRole::Switch,
            TileRole::Tune,
            TileRole::Tele,
            TileRole::Speedup,
        ] {
            assert!(roles.contains(&role), "{role:?} layer missing");
        }
        // Eight design layers share one tileset (image 5, "clbtiles") with different colours: draw order is the file's.
        let tinted: Vec<[u8; 4]> = layers(&scene)
            .iter()
            .filter_map(|l| match l {
                Layer::Tiles(t) if t.image == 5 => Some(t.color),
                _ => None,
            })
            .collect();
        assert_eq!(tinted.len(), 7);
        assert_eq!(tinted[0], [0, 0, 0, 102]);
        assert_eq!(tinted[3], [4, 6, 26, 163]);
        // Envelope 2 has 11 points and is a three-channel (position) envelope.
        assert_eq!((scene.envelopes[2].channels, scene.envelopes[2].points.len()), (3, 11));
    });
}

#[test]
fn blmap_chill() {
    with_map("BlmapChill_", |_, scene| {
        assert_eq!(
            (scene.groups.len(), scene.images.len(), scene.envelopes.len()),
            (19, 17, 16)
        );
        for name in ["bg_cloud1", "jungle_main", "jungle_deathtiles", "sun"] {
            assert!(external_names(&scene).contains(&name), "{name}");
        }
        // Every layer's image index is -1 or points at an image.
        for layer in layers(&scene) {
            let image = match layer {
                Layer::Tiles(t) => t.image,
                Layer::Quads(q) => q.image,
            };
            assert!(image == -1 || (image as usize) < scene.images.len());
        }
    });
}

#[test]
fn chill_block_5() {
    with_map("ChillBlock5_", |_, scene| {
        assert_eq!(
            (scene.groups.len(), scene.images.len(), scene.envelopes.len()),
            (7, 12, 4)
        );
        for name in [
            "bg_cloud1",
            "bg_cloud2",
            "generic_unhookable",
            "grass_main",
            "jungle_doodads",
        ] {
            assert!(external_names(&scene).contains(&name), "{name}");
        }
    });
}

#[test]
fn a_map_with_a_path_in_an_image_name_keeps_it_for_the_server_to_refuse() {
    // blmapV3ROYAL names external images "../skins/greyfox" and "../skins/saddo".
    with_map("blmapV3ROYAL_", |_, scene| {
        let names = external_names(&scene);
        assert!(names.contains(&"../skins/greyfox"), "{names:?}");
    });
}

/// Every cached map: parse, and check the structure is sane (image refs in range, envelope refs in range, tile data sized).
#[test]
fn every_cached_map_parses_and_is_internally_consistent() {
    let dir = maps_dir().join("cache");
    let Ok(read) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: {} is not there", dir.display());
        return;
    };
    let mut count = 0;
    for entry in read.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("map") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let scene = extract_visual_scene(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for layer in layers(&scene) {
            match layer {
                Layer::Tiles(t) => {
                    assert_eq!(t.tiles.len(), (t.width * t.height * 2) as usize);
                    assert_eq!(t.aux.len(), (t.width * t.height) as usize * t.role.aux_stride());
                    assert!(t.image == -1 || (t.image as usize) < scene.images.len());
                    assert!(t.color_env == -1 || (t.color_env as usize) < scene.envelopes.len());
                }
                Layer::Quads(q) => {
                    assert!(q.image == -1 || (q.image as usize) < scene.images.len());
                    for quad in &q.quads {
                        assert!(quad.pos_env == -1 || (quad.pos_env as usize) < scene.envelopes.len());
                        assert!(quad.color_env == -1 || (quad.color_env as usize) < scene.envelopes.len());
                    }
                }
            }
        }
        count += 1;
    }
    eprintln!("checked {count} cached maps");
}
