//! Task 4.2: the wayblock hold arenas (`clb-wb-left` / `clb-wb-right`): the focal player spawns on the
//! hall's first WB spot, intruders come from the hall boxes, and a game reports the time the focal player
//! spent in the hall and in `wbBand`. Skipped (with a note) when the Copy Love Box map is not present.

use std::path::PathBuf;

use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{PlayerSpec, Rules, builtin_brain};
use ddai_env::run::play_indexed;
use ddai_nav::wayblock::wayblocks;

fn repo(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

fn map_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join("aiddnet/data/maps"))
        .unwrap_or_default()
}

fn clb_present() -> bool {
    let p = map_dir()
        .join("copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map");
    if !p.exists() {
        eprintln!("skipping: {} not present", p.display());
    }
    p.exists()
}

#[test]
fn the_focal_player_starts_on_the_wb_spot_and_the_intruders_in_the_hall() {
    if !clb_present() {
        return;
    }
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    let clb = wayblocks().into_iter().next().unwrap();
    for (name, side_spots) in [("clb-wb-left", clb.left.spots[0]), ("clb-wb-right", clb.right.spots[0])] {
        let arena = Arena::build(&defs[name], &map_dir()).unwrap();
        assert!(arena.wb.is_some());
        for seed in 0..40 {
            for players in [2, 3] {
                let s = arena.spawn_tiles(seed, players).unwrap();
                assert_eq!(s[0], side_spots, "{name}: slot 0 holds the first WB spot");
                for t in &s[1..] {
                    assert!(
                        arena.slots().contains(t),
                        "{name}: intruders start on standing tiles of the hall"
                    );
                    let d = f64::from(t.0 - s[0].0).hypot(f64::from(t.1 - s[0].1));
                    assert!((3.0..=14.0).contains(&d), "{name}: {d}");
                }
                if players == 3 {
                    let sep = f64::from(s[1].0 - s[2].0).hypot(f64::from(s[1].1 - s[2].1));
                    assert!(sep >= 3.0, "intruders do not share a tile");
                }
                assert_eq!(
                    s,
                    arena.spawn_tiles(seed, players).unwrap(),
                    "deterministic in the seed"
                );
            }
        }
    }
    // Arenas without a [wayblock] are untouched.
    let plain = Arena::build(&defs["clb-left"], &map_dir()).unwrap();
    assert!(plain.wb.is_none());
}

#[test]
fn a_wb_game_reports_time_in_the_hall_and_in_the_band_and_wb_hints_need_a_wb_arena() {
    if !clb_present() {
        return;
    }
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    let arena = Arena::build(&defs["clb-wb-left"], &map_dir()).unwrap();
    let rules = Rules::default();
    let factory = |spec: &PlayerSpec| builtin_brain(spec);
    let slots = [PlayerSpec::simple("scripted"), PlayerSpec::simple("scripted")];
    let r = play_indexed(&arena, &rules, &slots, &factory, 7, 0).unwrap();
    assert!(
        r.a_ticks > 0 && r.a_ticks as i32 <= r.end_tick + 1,
        "ticks counted until the decision: {}",
        r.a_ticks
    );
    assert!(r.a_band_ticks <= r.a_ticks && r.a_hall_ticks <= r.a_ticks);
    assert!(r.a_hall_ticks > 0, "the focal player spends the game in the hall");
    // The same game on an arena with no [wayblock] counts nothing, and `wb = true` is refused there.
    let plain = Arena::build(&defs["clb-left"], &map_dir()).unwrap();
    let q = play_indexed(&plain, &rules, &slots, &factory, 7, 0).unwrap();
    assert_eq!((q.a_ticks, q.a_band_ticks, q.a_hall_ticks), (0, 0, 0));
    let mut wb = PlayerSpec::simple("idle");
    wb.wb = true;
    let err = play_indexed(&plain, &rules, &[wb, PlayerSpec::simple("scripted")], &factory, 7, 0).err();
    assert!(err.is_some_and(|e| e.to_string().contains("no [wayblock]")));
}

#[test]
fn a_wb_arena_never_swaps_the_holder_and_the_intruder() {
    if !clb_present() {
        return;
    }
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    let arena = Arena::build(&defs["clb-wb-left"], &map_dir()).unwrap();
    let spot = wayblocks().into_iter().next().unwrap().left.spots[0];
    let factory = |spec: &PlayerSpec| builtin_brain(spec);
    let slots = [PlayerSpec::simple("idle"), PlayerSpec::simple("idle")];
    let mut orders = [0, 0];
    for g in 0..8 {
        let r = play_indexed(&arena, &Rules::default(), &slots, &factory, 100, g).unwrap();
        assert!(!r.swap, "game {g}: no swap on a wayblock arena");
        assert_eq!(
            r.spawns[0],
            [
                f32::from(spot.0 as i16) * 32.0 + 16.0,
                f32::from(spot.1 as i16) * 32.0 + 16.0
            ],
            "game {g}: slot 0 holds the spot"
        );
        orders[usize::from(r.reverse_order)] += 1;
    }
    assert_eq!(orders, [4, 4], "the spawn order alternates");
}
