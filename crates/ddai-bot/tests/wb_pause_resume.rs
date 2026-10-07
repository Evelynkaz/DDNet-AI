//! Task 3.14 (D-105, review F7): the walk to the wayblock after a server pause, through the whole [`ddai_bot::Bot`] with the real navigation hooks.
//!
//! On the joniTee map (a copy of the Copy Love Box hall plus a sealed `/1vs1` arena) a bot that respawned in the arena is closed in: the hall cannot be walked to
//! and a kill would put it back. A resume from `/pause` is reported to the navigation as a respawn, but the tee did not move to a spawn: where its life began is
//! unknown, so the old walk must run again. Needs the joniTee map in `~/aiddnet/data/maps/cache` (the test skips without it).

mod support;

use std::path::PathBuf;
use std::sync::Arc;

use ddai_bot::hooks::MapIdent;
use ddai_bot::nav_hooks::{NavConfig, NavHandle, nav_hooks};
use ddai_bot::{Bot, BrainKind, Relations};
use ddai_brain::Action;
use ddai_net::generated::enums::explayerflagflag;
use support::*;

fn joni_map() -> Option<Arc<ddai_physics::map::MapData>> {
    let path = PathBuf::from(std::env::var("HOME").ok()?).join(
        "aiddnet/data/maps/cache/Copy Love Box JoniTee_d45815470abe2f832f6eb0a94a40bd186806198adcd0087948d1b5a3ade3d1e7.map",
    );
    Some(Arc::new(ddai_map::load_map(&std::fs::read(path).ok()?).ok()?.data))
}

#[test]
fn a_resume_from_a_server_pause_is_not_a_respawn_for_the_walled_off_walk() {
    big_stack(|| {
        let Some(map) = joni_map() else {
            eprintln!("skipping: the joniTee map is not present");
            return;
        };
        let handle = NavHandle::new();
        let hooks = nav_hooks(
            NavConfig {
                memory_dir: None,
                ..NavConfig::default()
            },
            handle.clone(),
        );
        let (probe, ..) = Probe::new(Action::neutral());
        let mut bot = Bot::new(cfg(BrainKind::Planner), Box::new(probe), hooks, Relations::new());
        bot.set_map_ident(MapIdent {
            name: "Copy Love Box JoniTee".to_string(),
            sha256: [9; 32],
        });
        bot.on_map_loaded(Arc::clone(&map));
        // One tee, alone in the sealed arena at its spawn tile (174, 54): nobody to fight, so it wants to go back to its wayblock.
        let mut me = tee(0, 174 * 32 + 16);
        me.y = 1745;
        let mut sc = Scenario::new(map, vec![me]);
        let walked = |bot: &mut Bot, sc: &mut Scenario, snapshots: usize| {
            let mut walking = false;
            for _ in 0..snapshots {
                sc.tee_mut(0).angle += 37; // never AFK
                run(bot, sc, 1);
                walking |= handle.status().walking;
            }
            walking
        };
        // The life began in the arena: closed in, no walk (a kill would respawn it there again).
        assert!(
            !walked(&mut bot, &mut sc, 400),
            "respawned in the sealed arena: the walk is closed"
        );
        // Paused for longer than the closing time (10 s = 250 snapshots), then resumed on the same tile.
        sc.player_mut(0).ex_flags = explayerflagflag::PAUSED;
        assert!(!walked(&mut bot, &mut sc, 400));
        assert!(bot.paused());
        sc.player_mut(0).ex_flags = 0;
        assert!(
            walked(&mut bot, &mut sc, 200),
            "after the resume where its life began is unknown: the old walk runs again"
        );
        assert!(!bot.paused());
    });
}
