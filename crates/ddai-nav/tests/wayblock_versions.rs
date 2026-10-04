//! Task 4.8: which maps get the Copy Love Box wayblock. Upstream af49dfb finds the hall by its tiles in
//! any map (`findHallOffset`), so the Swarfey version (468x255) and the copies with the hall elsewhere get
//! it too. Runs on the live adapter (`PhysicsWorld`, the world of the bot); the bit-exact comparison with
//! the TS is `parity_nav.rs` (`hall`, `wbdef`, `wbguard` lines).
//!
//! The maps are real (`~/aiddnet/data/maps`, never in the repository): a test whose map is missing prints
//! why and passes, as the other map tests of the project do. The JoniTee version is not among them, so it
//! is made: the 387x250 map copied into 600x600 shifted by `(182, 212)`, which is what the definition says.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_nav::wayblock::{WbSide, find_hall_offset, wayblock_for, wayblocks};
use ddai_physics::map::{MapData, Tile};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;

fn maps() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data/maps")
}

fn load(rel: &str) -> Option<MapData> {
    let path = maps().join(rel);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipped: no map {}", path.display());
        return None;
    };
    Some(ddai_map::load_map(&bytes).expect("map").data)
}

const ORIGINAL: &str =
    "copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map";
/// The version of Swarfey (468x255), the trigger of the task.
const SWARFEY: &str = "cache/Copy Love Box_23f188bba2a0a98f6005767481912f2b24dd55ea4980735fe31f81a090af94d7.map";
/// A 387x250 version whose hall sits 152 tiles to the right.
const HALL_RIGHT: &str =
    "copy-love-box/Copy Love Box_100590d06e3888aebb481245e89cae98461b50ffe998920b50021a0b9f1d2ed0.map";
const OTHER_MAP: &str = "chillblock5/ChillBlock5.map";

fn world(map: MapData) -> PhysicsWorld {
    PhysicsWorld::new(Arc::new(map), 1)
}

/// `map` placed into a `w` x `h` map of air at `(dx, dy)`.
fn shifted(map: &MapData, w: u32, h: u32, dx: u32, dy: u32) -> MapData {
    let mut game = vec![Tile::default(); (w * h) as usize];
    for y in 0..map.height {
        for x in 0..map.width {
            game[((y + dy) * w + x + dx) as usize] = map.game[(y * map.width + x) as usize];
        }
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

#[test]
fn the_original_map_is_the_named_definition_and_the_swarfey_version_is_found_by_its_hall() {
    let Some(orig) = load(ORIGINAL) else { return };
    let w = world(orig);
    let def = wayblock_for("Copy Love Box", Some(w.collision())).expect("the original has the WB");
    assert_eq!(
        def.name, "Copy Love Box",
        "the 387x250 map is the named definition, not a found one"
    );
    assert!(
        def.left.crossing.wall.is_some() && def.right.crossing.wall.is_some(),
        "route 2 fits it"
    );

    let Some(swarfey) = load(SWARFEY) else { return };
    assert_eq!((swarfey.width, swarfey.height), (468, 255));
    let w = world(swarfey);
    let hall = find_hall_offset(w.collision()).expect("the hall is in the Swarfey map");
    assert_eq!((hall.dx, hall.dy), (0, 0));
    assert!(
        hall.matched > 0.95 && hall.matched < 1.0,
        "a version with small differences: {}",
        hall.matched
    );
    let def = wayblock_for("Copy Love Box", Some(w.collision())).expect("the Swarfey version has the WB (the trigger)");
    assert_eq!(def.name, "Copy Love Box hall at +0,+0");
    assert_eq!(def.size, (468, 255));
    for s in [WbSide::Left, WbSide::Right] {
        assert!(
            def.side(s).crossing.wall.is_some(),
            "route 2 fits the Swarfey tubes ({s:?})"
        );
    }
    // Whatever the map is called: it is the tiles that count.
    assert!(wayblock_for("Some Other Name", Some(w.collision())).is_some());
    // Without a collision only the name can say: `Swarfey` has none for the plain name lookup either.
    assert!(wayblock_for("Some Other Name", None::<&<PhysicsWorld as PlanWorld>::Collision>).is_none());
}

#[test]
fn a_version_with_the_hall_elsewhere_gets_the_definition_shifted_there() {
    let Some(map) = load(HALL_RIGHT) else { return };
    let w = world(map);
    let hall = find_hall_offset(w.collision()).expect("hall");
    assert_eq!((hall.dx, hall.dy), (152, 0));
    let def = wayblock_for("Copy Love Box", Some(w.collision())).expect("WB");
    assert_eq!(def.name, "Copy Love Box hall at +152,+0");
    let orig = wayblocks().remove(0);
    assert_eq!(def.left.spots[0], (orig.left.spots[0].0 + 152, orig.left.spots[0].1));
    assert_eq!(
        def.right.crossing.start,
        (orig.right.crossing.start.0 + 152, orig.right.crossing.start.1)
    );
}

#[test]
fn a_shifted_copy_in_a_bigger_map_is_found_and_the_joni_tee_size_is_the_named_definition() {
    let Some(orig) = load(ORIGINAL) else { return };
    // JoniTee: 600x600, shifted by (182, 212): by name and size the definition of the table.
    let joni = shifted(&orig, 600, 600, 182, 212);
    let w = world(joni);
    let def = wayblock_for("Copy Love Box JoniTee", Some(w.collision())).expect("JoniTee WB");
    assert_eq!(def.name, "Copy Love Box JoniTee");
    // The same map under another name: the hall is found at the same place; the definition is the same
    // one with the found offset in its name.
    let found = wayblock_for("Copy Love Box", Some(w.collision())).expect("found by the hall");
    assert_eq!(found.name, "Copy Love Box hall at +182,+212");
    assert_eq!(found.left, def.left);
    assert_eq!(found.right, def.right);
    assert_eq!(found.crossings, def.crossings);
    // An odd offset in a map of another size.
    let odd = shifted(&orig, 420, 300, 17, 31);
    let w = world(odd);
    let found = wayblock_for("whatever", Some(w.collision())).expect("found");
    assert_eq!(found.name, "Copy Love Box hall at +17,+31");
    assert_eq!(found.size, (420, 300));
}

#[test]
fn a_damaged_hall_or_another_map_has_no_wayblock() {
    let Some(orig) = load(ORIGINAL) else { return };
    // The left third of the hall wiped out (solid and freeze alike): far below the 95% match.
    let mut broken = shifted(&orig, 400, 260, 3, 4);
    for y in 68..100u32 {
        for x in 79..107u32 {
            broken.game[(y * 400 + x) as usize] = Tile::default();
        }
    }
    let w = world(broken);
    assert!(
        wayblock_for("Copy Love Box", Some(w.collision())).is_none(),
        "a hall that matches less than 95%"
    );
    if let Some(other) = load(OTHER_MAP) {
        let w = world(other);
        assert!(find_hall_offset(w.collision()).is_none());
        assert!(
            wayblock_for("Copy Love Box", Some(w.collision())).is_none(),
            "the name alone is not enough"
        );
    }
    // A map smaller than the hall.
    let tiny = shifted(&orig, 387, 250, 0, 0);
    let mut small = tiny;
    small.width = 50;
    small.height = 20;
    small.game.truncate(50 * 20);
    let w = world(small);
    assert!(find_hall_offset(w.collision()).is_none());
}

#[test]
fn the_guard_geometry_is_measured_from_the_watch_point_and_mirrored_for_the_right_side() {
    let def = wayblocks().remove(0);
    let l = def.guard_geom(WbSide::Left);
    let r = def.guard_geom(WbSide::Right);
    // The watch point (89, 79) is the origin of the left hall's numbers.
    assert_eq!(l.job, (91, 79));
    assert_eq!(l.step_off, (86, 79));
    assert_eq!((l.shelf.x0, l.shelf.y0, l.shelf.x1, l.shelf.y1), (92, 81, 104, 84));
    assert_eq!(
        (l.corridor.x0, l.corridor.y0, l.corridor.x1, l.corridor.y1),
        (73, 70, 77, 88)
    );
    // The right hall is the left one mirrored in x (234 - x).
    assert_eq!(r.job, (234 - 91, 79));
    assert_eq!(r.step_off, (234 - 86, 79));
    assert_eq!((r.shelf.x0, r.shelf.x1), (234 - 104, 234 - 92));
    assert_eq!((r.shelf.y0, r.shelf.y1), (81, 84));
}

#[test]
fn with_the_guard_the_first_spot_is_the_left_end_of_the_upper_shelf() {
    let def = wayblocks().remove(0);
    assert!(
        ddai_nav::wayblock::wb_guard(),
        "the guard is on by default (DDAI_WB_GUARD=0 turns it off)"
    );
    assert_eq!(def.left.spots, vec![(83, 79), (94, 84), (101, 84)]);
    assert_eq!(def.right.spots, vec![(234 - 83, 79), (234 - 94, 84), (234 - 101, 84)]);
}

#[test]
#[ignore = "a timing measurement, prints its result: cargo test -p ddai-nav --release --test wayblock_versions -- --ignored --nocapture"]
fn the_hall_search_on_the_big_block_maps_takes_milliseconds_once_per_map() {
    for (rel, name) in [
        (OTHER_MAP, "ChillBlock5 (943x1075)"),
        (
            "cache/BlmapChill_c902b2da07291266ab201054e6b6c28abd31e10b099fa5b1066b2f5a88f98240.map",
            "BlmapChill (1244x667)",
        ),
        (SWARFEY, "Copy Love Box Swarfey (468x255)"),
    ] {
        let Some(map) = load(rel) else { continue };
        let w = world(map);
        let t0 = std::time::Instant::now();
        let found = find_hall_offset(w.collision());
        let one = t0.elapsed();
        let t1 = std::time::Instant::now();
        let def = wayblock_for("whatever", Some(w.collision()));
        eprintln!(
            "[hall search] {name}: find_hall_offset {:.1} ms ({}), wayblock_for {:.1} ms ({})",
            one.as_secs_f64() * 1e3,
            found.map_or("no hall".to_string(), |h| format!("hall at {:+},{:+}", h.dx, h.dy)),
            t1.elapsed().as_secs_f64() * 1e3,
            def.map_or("none".to_string(), |d| d.name),
        );
    }
}
