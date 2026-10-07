//! Task 3.20 (D-115): the server's pre-inputs inside the whole bot pipeline, offline.
//!
//! Synthetic snapshots through [`ddai_bot::Bot`] with a probing brain that records the exact world it is handed. The opponent stands still in every
//! snapshot; the "server" tells the bot ahead of time that it walks left. With the pre-inputs played the brain's predicted opponent walks left, without
//! them (the default) it stands -- and the messages are counted either way.

mod support;

use ddai_bot::{Bot, BrainKind, PreInputMode, Relations};
use ddai_brain::Action;
use ddai_net::generated::messages::SvPreInput;
use support::*;

struct Rig {
    bot: Bot,
    sc: Scenario,
    log: std::rc::Rc<std::cell::RefCell<Vec<Seen>>>,
}

fn rig(mode: PreInputMode) -> Rig {
    let (probe, log, _r, _a) = Probe::new(Action::neutral());
    let mut bot = bot_with(Box::new(probe), cfg(BrainKind::Hybrid), Relations::new());
    let map = room(&[]);
    bot.on_map_loaded(std::sync::Arc::clone(&map));
    bot.set_preinput(mode);
    let mut a = tee(0, 1000);
    let mut b = tee(1, 1200);
    a.angle = 100;
    b.angle = 200;
    Rig {
        bot,
        sc: Scenario::new(map, vec![a, b]),
        log,
    }
}

fn message(owner: i32, tick: i32, direction: i32) -> SvPreInput {
    SvPreInput {
        direction,
        target_x: 100,
        target_y: -1,
        jump: 0,
        fire: 0,
        hook: 0,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
        owner,
        intended_tick: tick,
    }
}

/// One snapshot. The opponent stands in the snapshots up to tick 1006 and walks left from tick 1007 on (its snapshots from 1008 show it); the "server" has
/// told the bot about the change from the first snapshot on (`announce`): a message for tick 1007, and a later one to keep the horizon open.
fn step(r: &mut Rig, announce: bool) {
    {
        let tick = r.sc.tick;
        let t = r.sc.tee_mut(1);
        t.angle = (t.angle + 37) % 1000;
        if tick >= 1008 {
            t.direction = -1;
            t.x -= 6;
        }
    }
    if announce && r.sc.tick < 1008 {
        r.bot.on_pre_input(&message(1, 1007, -1));
        r.bot.on_pre_input(&message(1, 1012, -1));
    }
    run(&mut r.bot, &mut r.sc, 1);
}

/// `(world tick, predicted opponent vx)` of every decision.
fn target_vx(r: &Rig) -> Vec<(i32, f32)> {
    r.log
        .borrow()
        .iter()
        .filter_map(|s| s.target_vel.map(|v| (s.world_tick, v.0)))
        .collect()
}

#[test]
fn the_pre_inputs_are_counted_always_and_played_only_when_asked_for() {
    big_stack(|| {
        let (mut off, mut on, mut killed) = (rig(PreInputMode::Off), rig(PreInputMode::On), rig(PreInputMode::Killed));
        for _ in 0..4 {
            // The first snapshot creates the live world: messages before it have nowhere to go.
            let announce = off.bot.stats().snapshots > 0;
            step(&mut off, announce);
            step(&mut on, announce);
            step(&mut killed, announce);
        }
        // Decisions at snapshots 1004 and 1006 roll the world to 1007 and 1009: the opponent still stands in the snapshots, the server says it walks from 1007.
        let at = |r: &Rig, tick| target_vx(r).into_iter().find(|(t, _)| *t == tick).map(|(_, v)| v);
        for tick in [1007, 1009] {
            assert_eq!(
                at(&off, tick),
                Some(0.0),
                "off: a standing opponent stays standing ({tick})"
            );
            assert_eq!(at(&killed, tick), Some(0.0), "killed by the marker: the same ({tick})");
            assert!(
                at(&on, tick).is_some_and(|v| v < -0.5),
                "on: it walks left in the brain's world ({tick}): {:?}",
                at(&on, tick)
            );
        }
        // Before the change nothing differs.
        assert_eq!(at(&on, 1005), Some(0.0));
        let (mode, c) = on.bot.preinput_status();
        assert_eq!(mode, PreInputMode::On);
        assert!(c.received >= 4 && c.stored >= 2 && c.used > 0, "{c:?}");
        let (mode, c) = off.bot.preinput_status();
        assert_eq!(mode, PreInputMode::Off);
        assert!(c.stored >= 2 && c.used == 0, "counted and stored, never played: {c:?}");
        // Everything but the opponent is the same in all three: our own tee and the ticks.
        let (a, b, k) = (off.log.borrow(), on.log.borrow(), killed.log.borrow());
        for ((x, y), z) in a.iter().zip(b.iter()).zip(k.iter()) {
            assert_eq!((x.world_tick, x.self_x), (y.world_tick, y.self_x));
            assert_eq!(
                (x.world_tick, x.self_x, x.target_pos, x.target_vel),
                (z.world_tick, z.self_x, z.target_pos, z.target_vel)
            );
        }
    });
}

#[test]
fn off_decides_exactly_like_a_bot_that_never_heard_of_pre_inputs() {
    big_stack(|| {
        let (mut heard, mut deaf) = (rig(PreInputMode::Off), rig(PreInputMode::Off));
        for _ in 0..12 {
            let announce = heard.bot.stats().snapshots > 0;
            step(&mut heard, announce);
            step(&mut deaf, false);
        }
        let (a, b) = (heard.log.borrow(), deaf.log.borrow());
        assert!(a.len() > 8);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                (x.world_tick, x.self_x, x.target_pos, x.target_vel),
                (y.world_tick, y.self_x, y.target_pos, y.target_vel)
            );
        }
    });
}

#[test]
fn a_message_for_our_own_id_or_a_bad_owner_moves_nothing() {
    big_stack(|| {
        let mut r = rig(PreInputMode::On);
        step(&mut r, false);
        for _ in 0..3 {
            let t = r.sc.tick;
            r.bot.on_pre_input(&message(0, t + 1, -1));
            r.bot.on_pre_input(&message(77, t + 1, -1));
            step(&mut r, false);
        }
        assert!(target_vx(&r).iter().all(|v| v.1 == 0.0));
        let (_, c) = r.bot.preinput_status();
        assert_eq!(c.used, 0);
        assert!(c.invalid >= 3, "{c:?}");
    });
}

#[test]
fn steady_state_snapshots_with_pre_inputs_allocate_nothing_in_the_bots_own_code() {
    big_stack(|| {
        let mut bot = bot_with(
            Box::new(ddai_brain::IdleBrain),
            cfg(BrainKind::Hybrid),
            Relations::new(),
        );
        let map = room(&[]);
        bot.on_map_loaded(map.clone());
        bot.set_preinput(PreInputMode::On);
        let mut tees = vec![tee(0, 1000)];
        for i in 1..=6 {
            tees.push(tee(i, 1000 + 120 * i));
        }
        let mut sc = Scenario::new(map, tees);
        for _ in 0..80 {
            for i in 1..=6 {
                let t = sc.tee_mut(i);
                t.angle = (t.angle + 37) % 1000;
            }
            run(&mut bot, &mut sc, 1);
        }
        assert_eq!(bot.target_id(), 1);
        let mut snaps = Vec::new();
        let mut msgs = Vec::new();
        for _ in 0..200 {
            for i in 1..=6 {
                let t = sc.tee_mut(i);
                t.angle = (t.angle + 37) % 1000;
            }
            // Changes of the fire counter only: consistent with an opponent that stands (direction 0), and every message really plays.
            let mut m = [message(1, sc.tick + 1, 0), message(1, sc.tick + 4, 0)];
            m[0].fire = sc.tick;
            m[1].fire = sc.tick + 1;
            msgs.push(m);
            snaps.push(sc.snapshot());
            sc.tick += 2;
        }
        let info = allocation_counter::measure(|| {
            for (snap, m) in snaps.iter().zip(&msgs) {
                bot.on_pre_input(&m[0]);
                bot.on_pre_input(&m[1]);
                let out = bot.on_snapshot(snap);
                std::hint::black_box(out);
                bot.on_input_sent(snap.pred_tick + 1, &out.input.expect("decided"));
            }
        });
        assert_eq!(info.count_total, 0, "{info:?}");
        assert!(
            bot.preinput_status().1.used > 100,
            "the pre-inputs really played inside the loop"
        );
    });
}
