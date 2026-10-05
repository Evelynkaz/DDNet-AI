//! Offline scenario tests: synthetic snapshot sequences through the whole [`ddai_bot::Bot`] pipeline,
//! asserting target choice, kill decisions, what the brain is given, the post-filters and the
//! protocol actions. No network; every nickname is a `p<id>` test string.

mod support;

use ddai_bot::relations::ListKind;
use ddai_bot::{Bot, BotEvent, BrainKind, Hooks, Mode, Relations};
use ddai_brain::{Action, IVec2};
use ddai_net::generated::enums::explayerflagflag;
use support::*;

fn neutral() -> Action {
    Action::neutral()
}

/// A bot with the probe brain (shield-less `Scripted` kind so the guard path is exercised too) on
/// the plain room.
fn setup(tees: Vec<TeeSpec>, rel: Relations) -> (Bot, Scenario, Handles) {
    setup_on(room(&[]), tees, rel, BrainKind::Planner)
}

type Handles = (
    std::rc::Rc<std::cell::RefCell<Vec<Seen>>>,
    std::rc::Rc<std::cell::RefCell<Vec<ddai_brain::ResetContext>>>,
    std::rc::Rc<std::cell::RefCell<Action>>,
);

fn setup_on(
    map: std::sync::Arc<ddai_physics::map::MapData>,
    tees: Vec<TeeSpec>,
    rel: Relations,
    kind: BrainKind,
) -> (Bot, Scenario, Handles) {
    let (probe, log, resets, action) = Probe::new(neutral());
    let mut bot = bot_with(Box::new(probe), cfg(kind), rel);
    bot.on_map_loaded(std::sync::Arc::clone(&map));
    (bot, Scenario::new(map, tees), (log, resets, action))
}

/// A tee whose aim keeps changing, so it never looks AFK.
fn wiggle(sc: &mut Scenario, id: i32) {
    let t = sc.tee_mut(id);
    t.angle = (t.angle + 37) % 1000;
}

fn run_active(bot: &mut Bot, sc: &mut Scenario, ids: &[i32], n: usize) {
    for _ in 0..n {
        for &id in ids {
            wiggle(sc, id);
        }
        run(bot, sc, 1);
    }
}

// --- target selection -----------------------------------------------------------------------------

#[test]
fn the_nearest_active_player_is_the_target() {
    support::big_stack(|| {
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1200), tee(2, 1600)], Relations::new());
        run_active(&mut bot, &mut sc, &[1, 2], 20);
        assert_eq!(bot.target_id(), 1, "200 px beats 600 px");
        let seen = log.borrow();
        assert!(!seen.is_empty());
        assert_eq!(seen.last().unwrap().target, Some(1), "and the brain is told");
    });
}

#[test]
fn friends_ignored_and_clan_friends_are_never_targets_and_war_outranks_distance() {
    support::big_stack(|| {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p1");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], rel);
        run_active(&mut bot, &mut sc, &[1, 2], 10);
        assert_eq!(bot.target_id(), 2, "p1 is a friend: skipped although nearer");

        let mut rel = Relations::new();
        rel.add(ListKind::Ignore, "p1");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], rel);
        run_active(&mut bot, &mut sc, &[1, 2], 10);
        assert_eq!(bot.target_id(), 2, "ignored");

        let mut rel = Relations::new();
        rel.add(ListKind::ClanFriend, "buddies");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], rel);
        sc.player_mut(1).clan = "Buddies".to_string();
        run_active(&mut bot, &mut sc, &[1, 2], 10);
        assert_eq!(bot.target_id(), 2, "clan friend");

        // War: p2 (500 px, +900) beats p1 (100 px, no bonus).
        let mut rel = Relations::new();
        rel.add(ListKind::War, "p2");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], rel);
        run_active(&mut bot, &mut sc, &[1, 2], 10);
        assert_eq!(bot.target_id(), 2, "at war");
        let mut rel = Relations::new();
        rel.add(ListKind::ClanWar, "foes");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], rel);
        sc.player_mut(2).clan = "FOES".to_string();
        run_active(&mut bot, &mut sc, &[1, 2], 10);
        assert_eq!(bot.target_id(), 2, "clan war");
    });
}

#[test]
fn list_entries_match_exactly_not_as_substrings() {
    support::big_stack(|| {
        // The TS bug: a short list entry matched every name containing it.
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p1");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(10, 1500)], rel);
        run_active(&mut bot, &mut sc, &[1, 10], 10);
        // "p10" contains "p1" but is not the friend.
        assert_eq!(bot.target_id(), 10, "p1 is skipped, p10 is fair game");
    });
}

#[test]
fn an_afk_player_is_skipped_until_it_moves_and_war_ignores_afk() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], Relations::new());
        // p1 never changes; p2 wiggles. After > 500 ticks p1 is AFK.
        run_active(&mut bot, &mut sc, &[2], 10);
        assert_eq!(bot.target_id(), 1, "still fresh: nearest");
        run_active(&mut bot, &mut sc, &[2], 260);
        assert_eq!(bot.target_id(), 2, "p1 has not moved for > 500 ticks: skipped");
        sc.tee_mut(1).attack_tick = 99;
        sc.player_mut(2).ex_flags = explayerflagflag::AFK; // take p2 out so only p1's state decides
        run_active(&mut bot, &mut sc, &[], 2);
        assert_eq!(bot.target_id(), 1, "p1 swung: active again");

        // The server's AFK flag counts at once.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], Relations::new());
        sc.player_mut(1).ex_flags = explayerflagflag::AFK;
        run_active(&mut bot, &mut sc, &[1, 2], 4);
        assert_eq!(bot.target_id(), 2, "server AFK flag");

        // ... except for a player at war.
        let mut rel = Relations::new();
        rel.add(ListKind::War, "p1");
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], rel);
        run_active(&mut bot, &mut sc, &[2], 300);
        assert_eq!(bot.target_id(), 1, "war ignores AFK");
    });
}

#[test]
fn paused_and_spectating_players_with_a_tee_on_the_map_are_fought_like_anybody() {
    // Upstream af49dfb (2026-10-01): a player in /pause or /spec whose tee is still on the map is a target (it was "out
    // of the game" before); only a player AFK while it plays is skipped.
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(
            vec![tee(0, 1000), tee(1, 1100), tee(2, 1500), tee(3, 1300)],
            Relations::new(),
        );
        sc.player_mut(1).ex_flags = explayerflagflag::PAUSED;
        sc.player_mut(3).ex_flags = explayerflagflag::SPEC;
        run_active(&mut bot, &mut sc, &[1, 2, 3], 4);
        assert_eq!(bot.target_id(), 1, "the paused tee is the nearest foe");
        // The server's AFK flag still says "away": that one is skipped.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], Relations::new());
        sc.player_mut(1).ex_flags = explayerflagflag::AFK;
        run_active(&mut bot, &mut sc, &[1, 2], 4);
        assert_eq!(bot.target_id(), 2, "AFK while playing: not a target");
    });
}

#[test]
fn players_beyond_1600_px_are_ignored() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(
            vec![tee(0, 500), tee(1, 500 + 1601), tee(2, 500 + 1599)],
            Relations::new(),
        );
        run_active(&mut bot, &mut sc, &[1, 2], 4);
        assert_eq!(bot.target_id(), 2);
        let (mut bot, mut sc, _) = setup(vec![tee(0, 500), tee(1, 500 + 1601)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 4);
        assert_eq!(bot.target_id(), -1, "nobody in range");
    });
}

#[test]
fn frozen_players_are_settled_after_their_first_tick_and_a_kept_target_survives() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1500)], Relations::new());
        run_active(&mut bot, &mut sc, &[1, 2], 6);
        assert_eq!(bot.target_id(), 1);
        sc.tee_mut(1).frozen = true;
        run_active(&mut bot, &mut sc, &[2], 1);
        assert_eq!(
            bot.target_id(),
            1,
            "on its first frozen snapshot it is still the target to finish"
        );
        run_active(&mut bot, &mut sc, &[2], 6);
        assert_eq!(bot.target_id(), 2, "frozen a while: settled, switch to the free player");

        // Only a frozen player is left and it was the target: keepSettled keeps it.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        assert_eq!(bot.target_id(), 1);
        sc.tee_mut(1).frozen = true;
        run_active(&mut bot, &mut sc, &[], 12);
        assert_eq!(bot.target_id(), 1, "nobody else qualifies: the settled target is kept");
        // A second, free player takes over from a frozen, settled target at once.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        sc.tee_mut(1).frozen = true;
        run_active(&mut bot, &mut sc, &[], 6);
        sc.tees.push(tee(2, 1800));
        sc.players.push(player(2, "p2"));
        run_active(&mut bot, &mut sc, &[2], 3);
        assert_eq!(bot.target_id(), 2);
    });
}

#[test]
fn a_frozen_player_sealed_in_freeze_is_settled_even_on_its_first_tick() {
    support::big_stack(|| {
        // A freeze pit under a floor: tee 1 sits in it, frozen with plenty of time left.
        let pit: Vec<(u32, u32, u8)> = (40..50).flat_map(|x| (30..38).map(move |y| (x, y, FREEZE))).collect();
        let (mut bot, mut sc, _) = setup_on(
            room(&pit),
            vec![tee(0, 1000), tee(1, 45 * 32), tee(2, 1500)],
            Relations::new(),
            BrainKind::Planner,
        );
        sc.tee_mut(1).y = 33 * 32;
        run_active(&mut bot, &mut sc, &[1, 2], 3);
        sc.tee_mut(1).frozen = true;
        run_active(&mut bot, &mut sc, &[1, 2], 2);
        assert_eq!(bot.target_id(), 2, "p1 cannot get out of the pit: not worth hitting");
    });
}

#[test]
fn with_the_async_seal_worker_the_verdict_arrives_a_snapshot_or_two_later() {
    support::big_stack(|| {
        // The same pit as the synchronous test, but the search runs on the worker thread: the sealed tee
        // stays targetable until its answer arrives, then is dropped.
        let pit: Vec<(u32, u32, u8)> = (40..50).flat_map(|x| (30..38).map(move |y| (x, y, FREEZE))).collect();
        let (probe, _, _, _) = Probe::new(neutral());
        let mut c = cfg(BrainKind::Planner);
        c.async_seal = true;
        let mut bot = bot_with(Box::new(probe), c, Relations::new());
        let map = room(&pit);
        bot.on_map_loaded(map.clone());
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 45 * 32), tee(2, 1500)]);
        sc.tee_mut(1).y = 33 * 32;
        run_active(&mut bot, &mut sc, &[1, 2], 3);
        sc.tee_mut(1).frozen = true;
        let mut dropped_at = None;
        for k in 0..60 {
            run_active(&mut bot, &mut sc, &[1, 2], 1);
            std::thread::sleep(std::time::Duration::from_millis(10));
            if bot.target_id() == 2 && dropped_at.is_none() {
                dropped_at = Some(k);
            }
        }
        assert!(
            dropped_at.is_some(),
            "the worker's 'sealed' verdict must eventually drop p1"
        );
        assert_eq!(bot.target_id(), 2);
        assert!(bot.seal_times().summary().count >= 1, "the search ran on the worker");
    });
}

#[test]
fn score_terms_hold_aggressor_approach_and_distance_work_as_in_the_table() {
    support::big_stack(|| {
        // Hold: the current target at 450 px keeps the job against a challenger at 440 px
        // (hold bonus up to 420 px, fading: 400 * (1 - 30/400) = 370).
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1450)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        assert_eq!(bot.target_id(), 1);
        sc.tees.push(tee(2, 1000 - 440));
        sc.players.push(player(2, "p2"));
        run_active(&mut bot, &mut sc, &[1, 2], 4);
        assert_eq!(bot.target_id(), 1, "sticky: the hold bonus outweighs 10 px");
        // The hold bonus fades to nothing 400 px beyond 420 px: a target that drifted to 900 px
        // loses the job to the challenger at 440 px (-225 against -110).
        sc.tee_mut(1).x = 1000 + 900;
        run_active(&mut bot, &mut sc, &[1, 2], 4);
        assert_eq!(bot.target_id(), 2, "the drifted target lost its hold bonus");

        // Aggressor: a far tee that recently attacked (within 500 px) outscores a nearer idle one.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1150), tee(2, 1400)], Relations::new());
        run_active(&mut bot, &mut sc, &[1, 2], 4);
        let t = sc.tick;
        sc.tee_mut(2).attack_tick = t - 10;
        run_active(&mut bot, &mut sc, &[1], 1);
        assert_eq!(
            bot.target_id(),
            2,
            "+500 for the aggressor beats 250 px of extra distance (62 points)"
        );

        // Hooking us is +1000 at any distance.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100), tee(2, 1800)], Relations::new());
        sc.tee_mut(2).hooked_player = 0;
        run_active(&mut bot, &mut sc, &[1, 2], 6);
        assert_eq!(bot.target_id(), 2);
    });
}

#[test]
fn a_tie_goes_to_the_lowest_client_id() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(
            vec![tee(0, 1000), tee(3, 1300), tee(2, 700), tee(5, 1300)],
            Relations::new(),
        );
        run_active(&mut bot, &mut sc, &[2, 3, 5], 4);
        assert_eq!(bot.target_id(), 2, "300 px both sides of us, equal scores: lowest id");
    });
}

#[test]
fn unreachable_far_targets_are_penalised_and_reachable_ones_are_not() {
    support::big_stack(|| {
        // A wall splits the room: p1 is 500 px away on the far side, p2 is 700 px away on our side.
        let wall: Vec<(u32, u32, u8)> = (1..39).map(|y| (40, y, SOLID)).collect();
        let (mut bot, mut sc, _) = setup_on(
            room(&wall),
            vec![tee(0, 38 * 32), tee(1, 38 * 32 + 500), tee(2, 38 * 32 - 700)],
            Relations::new(),
            BrainKind::Planner,
        );
        run_active(&mut bot, &mut sc, &[1, 2], 12);
        assert_eq!(
            bot.target_id(),
            2,
            "p1 is behind the wall: -700 makes the farther p2 win"
        );
        assert!(bot.reach_searches() >= 1);
    });
}

#[test]
fn a_fixed_target_ignores_every_other_filter() {
    support::big_stack(|| {
        let mut c = cfg(BrainKind::Planner);
        c.fixed_target = Some("  P2 ".to_string());
        let (probe, _, _, _) = Probe::new(neutral());
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let mut bot = bot_with(Box::new(probe), c, rel);
        let map = room(&[]);
        bot.on_map_loaded(map.clone());
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100), tee(2, 1900)]);
        sc.tee_mut(2).frozen = true;
        run_active(&mut bot, &mut sc, &[1], 10);
        assert_eq!(bot.target_id(), 2, "a friend, frozen, far: still the chosen target");
        sc.player_mut(2).name = "someone else".to_string();
        run_active(&mut bot, &mut sc, &[1], 2);
        assert_eq!(bot.target_id(), -1, "no player by that name any more");
    });
}

// --- modes ----------------------------------------------------------------------------------------

#[test]
fn hold_idles_and_passive_wanders_without_targets() {
    support::big_stack(|| {
        for mode in [Mode::Hold, Mode::Passive] {
            let mut c = cfg(BrainKind::Planner);
            c.mode = mode;
            let (probe, log, _, _) = Probe::new(neutral());
            let mut bot = bot_with(Box::new(probe), c, Relations::new());
            let map = room(&[]);
            bot.on_map_loaded(map.clone());
            let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100)]);
            run_active(&mut bot, &mut sc, &[1], 60);
            assert_eq!(bot.target_id(), -1, "{mode:?}: no target");
            assert!(log.borrow().is_empty(), "{mode:?}: the brain is never asked");
            let stats = bot.stats();
            if mode == Mode::Hold {
                assert_eq!(stats.wander_decisions, 0);
                assert_eq!(stats.idle_decisions, 60);
            } else {
                assert_eq!(stats.wander_decisions, 60, "passive wanders");
            }
        }
    });
}

#[test]
fn an_idle_brain_stands_still_even_without_targets() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup_on(room(&[]), vec![tee(0, 1000)], Relations::new(), BrainKind::Idle);
        let outs = run(&mut bot, &mut sc, 50);
        for o in outs {
            let i = o.input.expect("an input every snapshot");
            assert_eq!((i.direction, i.jump, i.hook, i.fire), (0, 0, 0, 0));
        }
    });
}

#[test]
fn set_mode_hold_drops_the_target() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        assert_eq!(bot.target_id(), 1);
        bot.set_mode(Mode::Hold);
        assert_eq!(bot.target_id(), -1);
    });
}

// --- unstick: Cl_Kill decisions -------------------------------------------------------------------

#[test]
fn frozen_in_a_freeze_tile_asks_for_a_kill_after_200_ticks_and_respects_the_cooldown() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup_on(
            room(&[(35, 38, FREEZE), (35, 37, FREEZE)]),
            vec![tee(0, 35 * 32 + 16), tee(1, 2000)],
            Relations::new(),
            BrainKind::Planner,
        );
        sc.tee_mut(0).y = 37 * 32 + 16;
        run_active(&mut bot, &mut sc, &[1], 3);
        let mut kills = Vec::new();
        for _ in 0..800 {
            sc.tee_mut(0).frozen = true;
            wiggle(&mut sc, 1);
            let t = sc.tick;
            let out = run(&mut bot, &mut sc, 1).pop().unwrap();
            if out.kill {
                kills.push(t);
            }
        }
        assert!(
            !kills.is_empty(),
            "frozen in the tile for 1600 ticks must ask for a kill"
        );
        assert!(
            kills[0] - 1006 >= 200 && kills[0] - 1006 <= 204,
            "first kill {}",
            kills[0]
        );
        for w in kills.windows(2) {
            assert!(w[1] - w[0] >= 500, "KILL_COOLDOWN_TICKS: {kills:?}");
        }
        assert_eq!(bot.stats().self_kills as usize, kills.len());
        let events: Vec<_> = bot.drain_events().collect();
        assert!(events.iter().any(|e| matches!(e, BotEvent::Killed { .. })));
    });
}

// --- the /kill fallback (task 4.6, D-078) -------------------------------------------------------------

/// A bot frozen in a freeze tile that the scenario never respawns (the server drops its `Cl_Kill`, like kill protection does).
fn frozen_bot_forever(notice_after_first_kill: bool) -> (Vec<(i32, bool, bool)>, Vec<BotEvent>) {
    let (mut bot, mut sc, _) = setup_on(
        room(&[(35, 38, FREEZE), (35, 37, FREEZE)]),
        vec![tee(0, 35 * 32 + 16), tee(1, 2000)],
        Relations::new(),
        BrainKind::Planner,
    );
    sc.tee_mut(0).y = 37 * 32 + 16;
    run_active(&mut bot, &mut sc, &[1], 3);
    let mut log = Vec::new();
    let mut noticed = false;
    for _ in 0..1700 {
        sc.tee_mut(0).frozen = true;
        wiggle(&mut sc, 1);
        let t = sc.tick;
        let out = run(&mut bot, &mut sc, 1).pop().unwrap();
        if out.kill || out.kill_command {
            log.push((t, out.kill, out.kill_command));
        }
        if notice_after_first_kill && out.kill && !noticed {
            noticed = true;
            bot.on_kill_protection_notice();
        }
    }
    (log, bot.drain_events().collect())
}

#[test]
fn a_cl_kill_without_effect_is_followed_by_exactly_one_slash_kill_per_decision() {
    support::big_stack(|| {
        let (log, events) = frozen_bot_forever(false);
        let kills: Vec<i32> = log.iter().filter(|e| e.1).map(|e| e.0).collect();
        let commands: Vec<i32> = log.iter().filter(|e| e.2).map(|e| e.0).collect();
        assert!(kills.len() >= 2, "{log:?}");
        // one /kill per decision, and no more than 3 in a life that never ends (the bot then stops asking: no spam)
        assert_eq!(commands.len(), kills.len().min(3), "{log:?}");
        assert!(kills.len() > 3, "the scenario outlasts the limit: {log:?}");
        for (k, c) in kills.iter().zip(&commands) {
            assert!(
                c - k >= 50 && c - k <= 54,
                "the /kill comes 50 ticks after its Cl_Kill: {log:?}"
            );
        }
        for w in commands.windows(2) {
            assert!(w[1] - w[0] >= 500, "KILL_COOLDOWN_TICKS between /kill: {commands:?}");
        }
        // never in the same output as a protocol kill (that would be a notice-driven case)
        assert!(log.iter().all(|e| !(e.1 && e.2)), "{log:?}");
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, BotEvent::KillFallback { noticed: false, .. }))
                .count(),
            commands.len()
        );
        assert!(
            events.iter().any(|e| matches!(e, BotEvent::KillFallbackGaveUp { .. })),
            "{events:?}"
        );
    });
}

#[test]
fn the_servers_notice_sends_the_slash_kill_at_once() {
    support::big_stack(|| {
        let (log, events) = frozen_bot_forever(true);
        let kills: Vec<i32> = log.iter().filter(|e| e.1).map(|e| e.0).collect();
        let commands: Vec<i32> = log.iter().filter(|e| e.2).map(|e| e.0).collect();
        assert!(kills.len() >= 2 && commands.len() >= 2, "{log:?}");
        // the first decision: the notice came a snapshot later, the /kill follows at once (well before 50 ticks)
        assert!(commands[0] - kills[0] <= 6, "{log:?}");
        // the next decisions of this life: nothing to wait for, only the 500-tick cooldown since the last /kill (2 ticks here)
        assert!(commands[1] - kills[1] <= 4, "{log:?}");
        for w in commands.windows(2) {
            assert!(w[1] - w[0] >= 500, "{commands:?}");
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, BotEvent::KillFallback { noticed: true, .. }))
        );
    });
}

/// Task 4.9b: the owner's `/pause` or `/spec` sets our own `DDNetPlayer` flag `PAUSED` / `SPEC` (the team and the tee stay in the snapshot).
/// While it is set the bot idles: neutral input, no `Cl_Kill`, no `/kill`, however long the unstick and fallback conditions have been met;
/// when it clears the bot plays again, with no stale timer making it kill at once.
#[test]
fn a_bot_paused_by_the_server_idles_and_never_kills_and_resumes_when_the_flag_clears() {
    use ddai_net::generated::enums::explayerflagflag::{PAUSED, SPEC};
    for flag in [PAUSED, SPEC] {
        support::big_stack(move || {
            let (mut bot, mut sc, _) = setup_on(
                room(&[(35, 38, FREEZE), (35, 37, FREEZE)]),
                vec![tee(0, 35 * 32 + 16), tee(1, 2000)],
                Relations::new(),
                BrainKind::Planner,
            );
            sc.tee_mut(0).y = 37 * 32 + 16;
            run_active(&mut bot, &mut sc, &[1], 3);
            // The conditions are met: a frozen bot that the server never respawns decides to kill (and, with the server's notice, to
            // send `/kill`) when it is not paused.
            let mut free_kills = 0;
            let mut free_commands = 0;
            for _ in 0..700 {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                let out = run(&mut bot, &mut sc, 1).pop().unwrap();
                free_kills += usize::from(out.kill);
                free_commands += usize::from(out.kill_command);
                if out.kill {
                    bot.on_kill_protection_notice();
                }
            }
            assert!(
                free_kills >= 1 && free_commands >= 1,
                "kills {free_kills}, commands {free_commands}"
            );
            let _ = bot.drain_events().count();
            // Paused: the same frozen bot, a long time (more than the unstick and the fallback ever need).
            sc.player_mut(0).ex_flags = flag;
            let mut checked = 0;
            for _ in 0..1500 {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                let out = run(&mut bot, &mut sc, 1).pop().unwrap();
                assert!(
                    !out.kill && !out.kill_command && out.set_team.is_none(),
                    "flag {flag}: {out:?}"
                );
                let input = out.input.expect("an idle input, every snapshot");
                assert_eq!(
                    (input.direction, input.jump, input.hook, input.fire & 1),
                    (0, 0, 0, 0),
                    "flag {flag}: neutral"
                );
                checked += 1;
            }
            assert_eq!(checked, 1500);
            assert!(bot.paused());
            let evs: Vec<_> = bot.drain_events().collect();
            assert_eq!(
                evs.iter()
                    .filter(|e| matches!(e, BotEvent::PausedByServer { .. }))
                    .count(),
                1,
                "reported once: {evs:?}"
            );
            assert!(!evs.iter().any(|e| matches!(
                e,
                BotEvent::Killed { .. } | BotEvent::KillFallback { .. } | BotEvent::MovedToSpectators { .. }
            )));
            assert!(
                bot.stop_reason().is_none(),
                "a pause is not the move to the spectators of D-058"
            );
            // Resumed: the flag clears, the bot plays on, and the pause did not leave a timer that kills it at once.
            sc.player_mut(0).ex_flags = 0;
            let mut first_kill = None;
            for i in 0..700 {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                let out = run(&mut bot, &mut sc, 1).pop().unwrap();
                if first_kill.is_none() && out.kill {
                    first_kill = Some(i);
                }
            }
            assert!(!bot.paused());
            assert!(
                first_kill.is_some_and(|i| i >= 50),
                "it plays again, and a kill needs its conditions anew after the pause: {first_kill:?}"
            );
            let evs: Vec<_> = bot.drain_events().collect();
            assert_eq!(
                evs.iter()
                    .filter(|e| matches!(e, BotEvent::ResumedByServer { .. }))
                    .count(),
                1
            );
        });
    }
}

/// 4.9b review F1: with `sv_pauseable 1`, a practice team or an admin `force_pause` the server removes a still, grounded tee in the very
/// snapshot that sets the `SPEC` flag. The bot must see the pause without a tee: `paused`, no kill (not even the operator's `!kill`, which
/// is told so), no death counted, no respawn reported, and on resume no stale timer kills it at once.
#[test]
fn a_pause_that_removes_the_tee_in_the_same_snapshot_is_still_a_pause() {
    use ddai_bot::BotCommand;
    use ddai_net::generated::enums::explayerflagflag::SPEC;
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup_on(
            room(&[(35, 38, FREEZE), (35, 37, FREEZE)]),
            vec![tee(0, 35 * 32 + 16), tee(1, 2000)],
            Relations::new(),
            BrainKind::Planner,
        );
        sc.tee_mut(0).y = 37 * 32 + 16;
        run_active(&mut bot, &mut sc, &[1], 3);
        // frozen and not respawned for a while: the unstick's clock has been running
        for _ in 0..120 {
            sc.tee_mut(0).frozen = true;
            wiggle(&mut sc, 1);
            run(&mut bot, &mut sc, 1);
        }
        let deaths_before = bot.stats().deaths;
        let _ = bot.drain_events().count();
        // the same snapshot: the flag is set and our tee is gone
        let own = sc.tees.remove(0);
        sc.player_mut(0).ex_flags = SPEC;
        let mut told = None;
        for i in 0..400 {
            if i == 10 {
                told = Some(bot.command(BotCommand::Kill));
            }
            wiggle(&mut sc, 1);
            let out = run(&mut bot, &mut sc, 1).pop().unwrap();
            assert!(
                !out.kill && !out.kill_command && out.set_team.is_none(),
                "snapshot {i}: {out:?}"
            );
            assert!(bot.paused(), "snapshot {i}: paused without a tee");
        }
        let told = told.unwrap();
        assert!(!told.ok && told.text.contains("paused by the server"), "{told:?}");
        assert_eq!(bot.stats().deaths, deaths_before, "a paused tee is not a death");
        assert!(bot.stop_reason().is_none());
        let evs: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            evs.iter()
                .filter(|e| matches!(e, BotEvent::PausedByServer { .. }))
                .count(),
            1,
            "{evs:?}"
        );
        assert!(
            !evs.iter().any(|e| matches!(
                e,
                BotEvent::Respawned { .. } | BotEvent::Killed { .. } | BotEvent::MovedToSpectators { .. }
            )),
            "{evs:?}"
        );
        // resume: the tee is back (still frozen), the flag clears
        sc.tees.insert(0, own);
        sc.player_mut(0).ex_flags = 0;
        let mut first_kill = None;
        for i in 0..400 {
            sc.tee_mut(0).frozen = true;
            wiggle(&mut sc, 1);
            let out = run(&mut bot, &mut sc, 1).pop().unwrap();
            if first_kill.is_none() && out.kill {
                first_kill = Some(i);
            }
        }
        assert!(!bot.paused());
        assert!(
            first_kill.is_some_and(|i| i >= 50),
            "the time of the pause is not frozen time: {first_kill:?}"
        );
        let evs: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            evs.iter()
                .filter(|e| matches!(e, BotEvent::ResumedByServer { .. }))
                .count(),
            1,
            "{evs:?}"
        );
        assert_eq!(
            bot.stats().deaths,
            deaths_before,
            "no death was ever counted for the pause"
        );
    });
}

/// 4.9b review F2: the pause is forgotten with the world (map change) and with the connection, not left to a tee that may never come.
#[test]
fn the_pause_does_not_outlive_a_map_change_or_a_disconnect() {
    use ddai_net::generated::enums::explayerflagflag::SPEC;
    for disconnect in [false, true] {
        support::big_stack(move || {
            let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
            run_active(&mut bot, &mut sc, &[1], 6);
            sc.player_mut(0).ex_flags = SPEC;
            run_active(&mut bot, &mut sc, &[1], 3);
            assert!(bot.paused());
            if disconnect {
                bot.on_disconnected();
            } else {
                let map = sc.map.clone();
                bot.on_map_changing();
                bot.on_map_loaded(map);
            }
            assert!(!bot.paused(), "disconnect {disconnect}: cleared at once");
            // joining again, in the spectators with no tee and no flag: not "paused"
            sc.tees.remove(0);
            sc.player_mut(0).ex_flags = 0;
            sc.player_mut(0).team = -1;
            sc.tick = 10;
            run(&mut bot, &mut sc, 30);
            assert!(!bot.paused(), "disconnect {disconnect}");
        });
    }
}

/// A real move to the spectators (team -1) is not a pause: D-058 still stops the bot, even with a pause flag around.
#[test]
fn a_pause_flag_does_not_hide_a_move_to_the_spectators() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        sc.tees.remove(0);
        sc.player_mut(0).team = -1;
        sc.player_mut(0).ex_flags = ddai_net::generated::enums::explayerflagflag::SPEC;
        for _ in 0..10 {
            run(&mut bot, &mut sc, 1);
        }
        assert_eq!(bot.stop_reason(), Some(ddai_bot::bot::StopReason::MovedToSpectators));
        assert!(!bot.paused());
    });
}

#[test]
fn a_free_bot_standing_still_with_a_target_gets_unstuck_but_not_without_one() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 2400)], Relations::new());
        let mut killed = false;
        for _ in 0..150 {
            wiggle(&mut sc, 1);
            killed |= run(&mut bot, &mut sc, 1)[0].kill;
        }
        assert!(
            killed,
            "a target 1400 px away, we never move: wedged for 200 ticks -> kill"
        );

        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000)], Relations::new());
        let killed = run(&mut bot, &mut sc, 400).iter().any(|o| o.kill);
        assert!(!killed, "no target: standing still is not being stuck");
    });
}

#[test]
fn hold_mode_never_requests_a_kill() {
    support::big_stack(|| {
        let mut c = cfg(BrainKind::Planner);
        c.mode = Mode::Hold;
        let (probe, ..) = Probe::new(neutral());
        let mut bot = bot_with(Box::new(probe), c, Relations::new());
        let map = room(&[(35, 38, FREEZE)]);
        bot.on_map_loaded(map.clone());
        let mut sc = Scenario::new(map, vec![tee(0, 35 * 32 + 16), tee(1, 2000)]);
        sc.tee_mut(0).y = 37 * 32 + 16;
        sc.tee_mut(0).frozen = true;
        assert!(!run(&mut bot, &mut sc, 600).iter().any(|o| o.kill));
    });
}

// --- what the brain is given ----------------------------------------------------------------------

#[test]
fn the_world_is_predicted_to_the_drivers_pred_tick_with_no_lag_guess() {
    support::big_stack(|| {
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        sc.pred_ahead = 4;
        run_active(&mut bot, &mut sc, &[1], 6);
        let seen = log.borrow();
        let last = seen.last().unwrap();
        assert_eq!(
            last.obs_tick,
            sc.tick - 2 + 4,
            "observation at the snapshot tick + pred_ahead"
        );
        assert_eq!(last.world_tick, last.obs_tick);
        assert_eq!(
            last.lag_ticks, 0,
            "in-flight inputs are inside the world, not a lag guess"
        );
        assert_eq!(last.in_flight, 0);
        assert_eq!(last.self_id, 0);
    });
}

#[test]
fn a_decision_that_misses_the_next_input_is_predicted_one_tick_further() {
    support::big_stack(|| {
        let obs_tick_with = |next_in_ms: u64| {
            let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
            sc.pred_ahead = 4;
            sc.next_input_in = Some(std::time::Duration::from_millis(next_in_ms));
            run_active(&mut bot, &mut sc, &[1], 6);
            let last_snapshot_tick = sc.tick - 2;
            log.borrow().last().unwrap().obs_tick - last_snapshot_tick
        };
        // The bot's estimate of its own decision (1 ms to start, + 2 ms driver pickup) still makes an
        // input due in 15 ms: predict to the last sent tick (snapshot + 4).
        assert_eq!(obs_tick_with(15), 4);
        // Due in 1 ms: the decision is too late for it and goes out with the next one.
        assert_eq!(obs_tick_with(1), 5);
    });
}

/// A brain that takes its time: the host's real decision time must not move the expected tick of the
/// scenarios above (3.5b review F8: a ~10 ms host pause flipped it on a loaded machine).
struct Slow {
    inner: Probe,
    delay: std::time::Duration,
}

impl ddai_brain::Brain for Slow {
    fn reset(&mut self, ctx: &ddai_brain::ResetContext) {
        self.inner.reset(ctx);
    }
    fn decide(&mut self, obs: &ddai_brain::Observation) -> Action {
        self.inner.decide(obs)
    }
    fn decide_in(&mut self, obs: &ddai_brain::Observation, view: Option<&ddai_brain::WorldView<'_>>) -> Action {
        std::thread::sleep(self.delay);
        self.inner.decide_in(obs, view)
    }
    fn set_live_context(&mut self, ctx: &ddai_brain::LiveContext<'_>) {
        self.inner.set_live_context(ctx);
    }
    fn name(&self) -> &str {
        "slow"
    }
}

#[test]
fn the_slot_choice_does_not_depend_on_how_long_the_host_takes_to_decide() {
    support::big_stack(|| {
        let obs_tick_with = |delay_ms: u64, fixed: bool| {
            let (probe, log, ..) = Probe::new(neutral());
            let mut c = cfg(BrainKind::Planner);
            if !fixed {
                c.decision_time_override = None;
            }
            let slow = Slow {
                inner: probe,
                delay: std::time::Duration::from_millis(delay_ms),
            };
            let mut bot = bot_with(Box::new(slow), c, Relations::new());
            let map = room(&[]);
            bot.on_map_loaded(map.clone());
            let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100)]);
            sc.pred_ahead = 4;
            sc.next_input_in = Some(std::time::Duration::from_millis(15));
            run_active(&mut bot, &mut sc, &[1], 6);
            let last_snapshot_tick = sc.tick - 2;
            log.borrow().last().unwrap().obs_tick - last_snapshot_tick
        };
        // With the fixed estimate a 40 ms brain changes nothing: the input due in 15 ms is predicted
        // to the last sent tick, as for an instant brain.
        assert_eq!(obs_tick_with(0, true), 4);
        assert_eq!(obs_tick_with(40, true), 4, "real time is ignored under the override");
        // Without it (production) the measured time does count: a 40 ms decision misses the input due
        // in 15 ms and is aimed further out.
        assert!(obs_tick_with(40, false) > 4, "production still measures");
    });
}

#[test]
fn before_the_timing_bootstrap_a_two_tick_guess_is_used_and_the_horizon_is_capped() {
    support::big_stack(|| {
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        sc.pred_tick_fixed = Some(0); // the timing bootstrap has not happened yet
        run_active(&mut bot, &mut sc, &[1], 3);
        assert_eq!(log.borrow().last().unwrap().obs_tick, sc.tick - 2 + 2);
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        // A long RTT (the driver 20 ticks ahead): the horizon follows it instead of stopping at 12.
        sc.pred_ahead = 20;
        run_active(&mut bot, &mut sc, &[1], 3);
        assert_eq!(log.borrow().last().unwrap().obs_tick, sc.tick - 2 + 20, "RTT-aware cap");
        assert_eq!(bot.stats().predict_clamped, 0);
        // An absurd one hits the absolute cap, and the bot says so (once, rate-limited).
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        sc.pred_ahead = 500;
        run_active(&mut bot, &mut sc, &[1], 30);
        assert_eq!(
            log.borrow().last().unwrap().obs_tick,
            sc.tick - 2 + ddai_bot::bot::MAX_PREDICT_TICKS_ABSOLUTE,
            "MAX_PREDICT_TICKS_ABSOLUTE"
        );
        assert!(bot.stats().predict_clamped >= 20);
        let evs: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            evs.iter()
                .filter(|e| matches!(e, BotEvent::PredictionClamped { .. }))
                .count(),
            1,
            "reported once per 10 s: {evs:?}"
        );
    });
}

#[test]
fn our_inputs_in_flight_move_the_predicted_own_tee() {
    support::big_stack(|| {
        // The same snapshot twice: once with nothing in flight, once with "run right" already sent
        // for every tick up to the pred tick. The brain's observation must show the difference.
        let predicted_x = |in_flight_dir: Option<i32>| {
            let (probe, log, _, _) = Probe::new(neutral());
            let mut bot = bot_with(Box::new(probe), cfg(BrainKind::Planner), Relations::new());
            let map = room(&[]);
            bot.on_map_loaded(map.clone());
            let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100)]);
            sc.pred_ahead = 4;
            // A few snapshots so the tee is settled on the floor and the bot is warm.
            for _ in 0..4 {
                wiggle(&mut sc, 1);
                let snap = sc.snapshot();
                bot.on_snapshot(&snap);
                sc.tick += 2;
            }
            wiggle(&mut sc, 1);
            let snap = sc.snapshot();
            if let Some(dir) = in_flight_dir {
                for t in (snap.tick - 6)..=snap.pred_tick {
                    let mut input = ddai_bot::input::neutral_input(0);
                    input.direction = dir;
                    bot.on_input_sent(t, &input);
                }
            }
            bot.on_snapshot(&snap);
            log.borrow().last().unwrap().self_x
        };
        let still = predicted_x(None);
        let right = predicted_x(Some(1));
        let left = predicted_x(Some(-1));
        assert!((still - 1000.0).abs() < 1.0, "standing: {still}");
        assert!(
            right > still + 10.0,
            "run right in flight moved the predicted tee: {right} vs {still}"
        );
        assert!(left < still - 10.0, "and left: {left} vs {still}");
    });
}

#[test]
fn the_brain_only_gets_the_target_roped_tees_and_the_nearest_few() {
    support::big_stack(|| {
        // 12 other tees: the target (#1, 300 px), and a crowd. The brain must see at most the target plus
        // 5 more, never the tee 4000 px away that is not the target.
        let mut tees = vec![tee(0, 3000)];
        for i in 1..=12 {
            tees.push(tee(i, 3000 + 100 * i));
        }
        tees.push(tee(20, 3000 + 4000));
        let (mut bot, mut sc, (log, ..)) = setup(tees, Relations::new());
        let ids: Vec<i32> = (1..=12).chain([20]).collect();
        run_active(&mut bot, &mut sc, &ids, 6);
        let seen = log.borrow();
        let last = seen.last().unwrap();
        assert_eq!(last.target, Some(1));
        assert!(last.others.len() <= 6, "target + 5 nearest: {:?}", last.others);
        assert!(last.others.contains(&1), "the target is always there");
        assert!(!last.others.contains(&20), "far tees are not handed to the brain");
        let nearest: Vec<i32> = (1..=6).collect();
        assert_eq!(
            last.others.iter().copied().filter(|i| nearest.contains(i)).count(),
            6,
            "the nearest five and the target"
        );
        assert!(
            last.world_ids.len() <= 7,
            "the exact world is cut too: {:?}",
            last.world_ids
        );
        assert!(last.world_ids.contains(&0), "and still has us");
    });
}

#[test]
fn a_far_target_is_kept_in_the_brains_world_and_roped_tees_are_always_kept() {
    support::big_stack(|| {
        // p1 is 1200 px away and hooks us (+1000): the target. p9 is 450 px away, behind five nearer
        // tees, and *we* hook it: it is not the target but must still be in the brain's world.
        let mut tees = vec![tee(0, 500), tee(1, 500 + 1200)];
        for i in 2..=8 {
            tees.push(tee(i, 500 + 40 * i));
        }
        tees.push(tee(9, 500 + 450));
        let (mut bot, mut sc, (log, ..)) = setup(tees, Relations::new());
        sc.tee_mut(1).hooked_player = 0;
        sc.tee_mut(0).hooked_player = 9;
        sc.tee_mut(0).hook_state = 5;
        let ids: Vec<i32> = (1..=9).collect();
        run_active(&mut bot, &mut sc, &ids, 6);
        let seen = log.borrow();
        let last = seen.last().unwrap();
        assert_eq!(last.target, Some(1), "the far tee that hooks us");
        assert!(last.others.contains(&1), "the far target is in the brain's world");
        assert!(last.others.contains(&9), "a tee we hook is always kept");
        assert!(last.others.len() <= 7, "{:?}", last.others);
    });
}

#[test]
fn the_brain_is_reset_on_the_first_life_and_on_every_respawn() {
    support::big_stack(|| {
        let (mut bot, mut sc, (_, resets, _)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 4);
        assert_eq!(resets.borrow().len(), 1);
        assert_eq!(resets.borrow()[0].self_id, 0);
        // We die: the tee disappears from the snapshot for a while, then comes back.
        let us = sc.tees.remove(0);
        run_active(&mut bot, &mut sc, &[1], 4);
        sc.tees.insert(0, us);
        run_active(&mut bot, &mut sc, &[1], 4);
        assert_eq!(resets.borrow().len(), 2, "a new life: fresh brain state");
        assert_ne!(resets.borrow()[0].seed, resets.borrow()[1].seed);
        assert_eq!(bot.stats().deaths, 1);
        let evs: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            evs.iter().filter(|e| matches!(e, BotEvent::Respawned { .. })).count(),
            2
        );
    });
}

/// Opponents p6..p11 at 160..610 px (the nearest five are kept), the target p1 at 150 px, and spared
/// tees: friends p2 (50 px), p5 (140 px) and p12 (500 px, out of contact range), ignored p3 (80 px),
/// AFK p4 (110 px).
fn crowd_with_spared() -> (Vec<TeeSpec>, Relations) {
    let mut rel = Relations::new();
    rel.add(ListKind::Friend, "p2");
    rel.add(ListKind::Ignore, "p3");
    rel.add(ListKind::Friend, "p5");
    rel.add(ListKind::Friend, "p12");
    let mut tees = vec![tee(0, 1000), tee(1, 1150)];
    for (id, dx) in [(2, 50), (3, 80), (4, 110), (5, 140), (12, 500)] {
        tees.push(tee(id, 1000 + dx));
    }
    for (i, id) in (6..=11).enumerate() {
        tees.push(tee(id, 1000 + 160 + 90 * i as i32));
    }
    (tees, rel)
}

#[test]
fn spared_tees_within_contact_range_are_bodies_counted_apart_from_the_local_others() {
    for kind in [BrainKind::Planner, BrainKind::Hybrid] {
        support::big_stack(move || {
            // Review F8 (task 4.1b): the planner and (task 3.5b) the hybrid honour `spare_ids`, so it gets the nearest three spared
            // tees within 200 px as physical bodies, in addition to the target and its five nearest others.
            let (tees, rel) = crowd_with_spared();
            let (mut bot, mut sc, (log, ..)) = setup_on(room(&[]), tees, rel, kind);
            sc.player_mut(4).ex_flags = explayerflagflag::AFK;
            let active: Vec<i32> = (1..=12).filter(|&i| i != 4).collect();
            run_active(&mut bot, &mut sc, &active, 8);
            let seen = log.borrow();
            let last = seen.last().unwrap();
            assert_eq!(last.target, Some(1));
            let world: Vec<i32> = last.world_ids.iter().copied().filter(|&i| i != 0).collect();
            // Target + the 5 nearest opponents (p6..p10) + the 3 nearest spared (p2, p3, p4).
            for id in [1, 6, 7, 8, 9, 10] {
                assert!(world.contains(&id), "{id} in {world:?}");
            }
            assert!(
                !world.contains(&11),
                "only five opponents besides the target: {world:?}"
            );
            for id in [2, 3, 4] {
                assert!(world.contains(&id), "spared body {id} in {world:?}");
            }
            assert!(!world.contains(&5), "at most three spared bodies: {world:?}");
            assert!(!world.contains(&12), "beyond contact range: {world:?}");
            assert_eq!(world.len(), 1 + 5 + 3);
            // The brain is told who is spared (every spared tee in hook reach + 64 px, not just the bodies).
            let mut ids = last.spare_ids.clone();
            ids.sort_unstable();
            assert_eq!(ids, vec![2, 3, 4, 5], "ids; p12 is 500 px away");
            assert_eq!(last.spares.len(), 4, "and their positions stay for the geometric gates");
        });
    }
}

#[test]
fn a_brain_that_does_not_honour_spare_ids_gets_no_spared_bodies_unless_roped_to_us() {
    support::big_stack(|| {
        // A brain that ignores `spare_ids` (the idle brain here) would read a body as an opponent: round-1 behaviour stays.
        let (tees, rel) = crowd_with_spared();
        let (mut bot, mut sc, (log, ..)) = setup_on(room(&[]), tees, rel, BrainKind::Idle);
        sc.player_mut(4).ex_flags = explayerflagflag::AFK;
        sc.tee_mut(5).hooked_player = 0; // p5 (a friend) is roped to us: kept
        let active: Vec<i32> = (1..=12).filter(|&i| i != 4).collect();
        run_active(&mut bot, &mut sc, &active, 8);
        let seen = log.borrow();
        let last = seen.last().unwrap();
        for id in [2, 3, 4, 12] {
            assert!(
                !last.world_ids.contains(&id),
                "{id} must not be in {:?}",
                last.world_ids
            );
            assert!(!last.others.contains(&id));
        }
        assert!(
            last.world_ids.contains(&5),
            "a tee roped to us stays: {:?}",
            last.world_ids
        );
        assert!(last.world_ids.contains(&1), "and the target");
        assert!(
            last.spare_ids.contains(&2) && last.spare_ids.len() >= 3,
            "{:?}",
            last.spare_ids
        );
    });
}

#[test]
fn a_spared_body_between_us_and_a_freeze_edge_is_simulated_so_the_brain_does_not_plan_through_it() {
    support::big_stack(|| {
        // A friend stands 30 px to our right, a freeze strip starts 24 px behind it, and we run right.
        // Tees collide, so the predicted tee is held back by the body and stays out of the freeze;
        // for a brain that gets no body (the round-1 behaviour) the same inputs run into the strip.
        let predicted = |kind: BrainKind| {
            let mut rel = Relations::new();
            rel.add(ListKind::Friend, "p2");
            let map = room(&[(32, 38, FREEZE), (33, 38, FREEZE), (34, 38, FREEZE)]);
            let (mut bot, mut sc, (log, _, act)) =
                setup_on(map, vec![tee(0, 1000), tee(1, 1400), tee(2, 1030)], rel, kind);
            *act.borrow_mut() = Action {
                direction: 1,
                ..neutral()
            };
            sc.pred_ahead = 10;
            run_active(&mut bot, &mut sc, &[1, 2], 6);
            let seen = log.borrow();
            let last = seen.last().unwrap();
            (last.self_x, last.self_frozen, last.world_ids.contains(&2))
        };
        let (with_x, with_frozen, has_body) = predicted(BrainKind::Planner);
        let (without_x, without_frozen, no_body) = predicted(BrainKind::Idle);
        println!(
            "predicted: with the body x={with_x} frozen={with_frozen}; without x={without_x} frozen={without_frozen}"
        );
        assert!(has_body && !no_body);
        assert!(!with_frozen, "the body keeps us out of the freeze");
        assert!(with_x < without_x - 1.0, "{with_x} vs {without_x}");
        assert!(without_frozen, "control: with no body the same run ends in the freeze");
    });
}

// --- post-filters ---------------------------------------------------------------------------------

#[test]
fn a_hook_that_would_catch_a_spared_tee_is_vetoed_and_one_that_does_not_is_kept() {
    support::big_stack(|| {
        // Us at x=1000, target p1 to the right at 340 px, a friend p2 between us (200 px): hooking
        // toward the target would catch the friend first.
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let (mut bot, mut sc, (log, _, act)) = setup(vec![tee(0, 1000), tee(1, 1340), tee(2, 1200)], rel);
        *act.borrow_mut() = Action {
            hook: true,
            target: IVec2::new(300, 0),
            ..neutral()
        };
        run_active(&mut bot, &mut sc, &[1, 2], 8);
        let outs = run(&mut bot, &mut sc, 1);
        let input = outs[0].input.unwrap();
        assert_eq!(input.hook, 0, "the rope would catch the friend first");
        assert!(bot.stats().vetoed_hooks >= 1);
        assert_eq!(
            log.borrow().last().unwrap().spares.len(),
            1,
            "the brain is told who is spared"
        );

        // Hooking away from the friend is fine.
        *act.borrow_mut() = Action {
            hook: true,
            target: IVec2::new(-300, 0),
            ..neutral()
        };
        let outs = run(&mut bot, &mut sc, 1);
        assert_eq!(outs[0].input.unwrap().hook, 1, "a clear line keeps the hook");
    });
}

#[test]
fn a_spared_afk_player_is_spared_but_a_war_target_is_not_and_holding_a_friend_is_dropped() {
    support::big_stack(|| {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let (mut bot, mut sc, (_, _, act)) = setup(vec![tee(0, 1000), tee(1, 1340), tee(2, 1200)], rel);
        *act.borrow_mut() = Action {
            hook: true,
            target: IVec2::new(300, 0),
            ..neutral()
        };
        // We are already hooking the friend: the hook must be released.
        sc.tee_mut(0).hooked_player = 2;
        sc.tee_mut(0).hook_state = 5;
        run_active(&mut bot, &mut sc, &[1, 2], 8);
        let out = run(&mut bot, &mut sc, 1).pop().unwrap();
        assert_eq!(out.input.unwrap().hook, 0, "never keep holding a friend");
    });
}

#[test]
fn a_brain_without_its_own_shield_is_guarded_and_one_with_it_is_not() {
    support::big_stack(|| {
        // A freeze tile at (36, 38) on the floor: we stand 1 px short of it (freezing needs the tee's centre inside the tile) and the probe brain says
        // "run right", which freezes us within two ticks. The scripted-kind bot has no shield of its
        // own, so the guard must replace that input; the planner kind trusts the brain's own
        // shield and passes it through.
        for (kind, expect_guard) in [(BrainKind::Scripted, true), (BrainKind::Planner, false)] {
            let (probe, _, _, act) = Probe::new(neutral());
            *act.borrow_mut() = Action {
                direction: 1,
                ..neutral()
            };
            let mut bot = bot_with(Box::new(probe), cfg(kind), Relations::new());
            let map = room(&[(36, 38, FREEZE)]);
            bot.on_map_loaded(map.clone());
            let mut sc = Scenario::new(map, vec![tee(0, 36 * 32 - 1), tee(1, 600)]);
            let outs = {
                let mut outs = Vec::new();
                for _ in 0..6 {
                    wiggle(&mut sc, 1);
                    outs.extend(run(&mut bot, &mut sc, 1));
                }
                outs
            };
            assert!(bot.stats().brain_decisions > 0, "{kind:?}: a target exists");
            let guarded = bot.stats().guarded_inputs;
            if expect_guard {
                assert!(
                    guarded > 0,
                    "{kind:?}: the guard must have stepped in (stats {:?})",
                    bot.stats()
                );
                assert!(
                    outs.iter().any(|o| o.input.is_some_and(|i| i.direction != 1)),
                    "{kind:?}: and the sent input is not 'run right' any more"
                );
            } else {
                assert_eq!(guarded, 0, "{kind:?}: the planner family guards itself");
                assert!(
                    outs.iter().all(|o| o.input.is_some_and(|i| i.direction == 1)),
                    "{kind:?}: passed through"
                );
            }
        }
    });
}

// --- encoding through the bot ---------------------------------------------------------------------

#[test]
fn fire_levels_become_press_counters_and_hammer_is_the_default_weapon() {
    support::big_stack(|| {
        let (mut bot, mut sc, (_, _, act)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        *act.borrow_mut() = Action {
            fire: true,
            ..neutral()
        };
        let outs = run(&mut bot, &mut sc, 1); // first snapshot decides a target already
        let _ = outs;
        let fires: Vec<i32> = run(&mut bot, &mut sc, 5)
            .iter()
            .map(|o| o.input.unwrap().fire)
            .collect();
        // Every decision is a fresh press: held -> +2 each time.
        for w in fires.windows(2) {
            assert_eq!(w[1] - w[0], 2, "{fires:?}");
        }
        assert!(fires.iter().all(|f| f & 1 == 1), "held after each press: {fires:?}");
        *act.borrow_mut() = neutral();
        let f = run(&mut bot, &mut sc, 1)[0].input.unwrap();
        assert_eq!(f.fire & 1, 0, "released");
        assert_eq!(f.wanted_weapon, 1, "hammer by default");
        assert_eq!(f.player_flags, ddai_net::generated::enums::playerflagflag::PLAYING);
        assert!(bot.stats().hammer_fires >= 5, "the tee holds the hammer (weapon 0)");
    });
}

// --- lifecycle ------------------------------------------------------------------------------------

#[test]
fn nothing_is_decided_before_a_map_is_loaded_or_own_id_is_known() {
    support::big_stack(|| {
        let (probe, ..) = Probe::new(neutral());
        let mut bot = bot_with(Box::new(probe), cfg(BrainKind::Planner), Relations::new());
        let map = room(&[]);
        let mut sc = Scenario::new(map.clone(), vec![tee(0, 1000), tee(1, 1100)]);
        let out = bot.on_snapshot(&sc.snapshot());
        assert!(out.input.is_none() && !out.kill && out.set_team.is_none(), "no map yet");
        bot.on_map_loaded(map);
        sc.own_id = 0;
        let mut snap = sc.snapshot();
        snap.own_id = None;
        assert!(bot.on_snapshot(&snap).input.is_none(), "own id unknown");
        assert!(bot.on_snapshot(&sc.snapshot()).input.is_some());
    });
}

#[test]
fn a_backwards_game_tick_resets_the_tick_dependent_state() {
    support::big_stack(|| {
        let (mut bot, mut sc, (_, resets, _)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        sc.tick = 100; // the map restarted
        run_active(&mut bot, &mut sc, &[1], 4);
        let evs: Vec<_> = bot.drain_events().collect();
        assert!(evs.iter().any(|e| matches!(e, BotEvent::TickReset { .. })), "{evs:?}");
        assert_eq!(resets.borrow().len(), 2, "a new episode for the brain");
        assert_eq!(bot.target_id(), 1, "and the fight goes on");
    });
}

#[test]
fn a_bot_in_the_spectators_asks_to_join_every_three_seconds_and_gives_up_after_ten_tries() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        sc.tees.remove(0); // no character: we are not playing
        sc.player_mut(0).team = -1;
        let mut joins = Vec::new();
        for _ in 0..1000 {
            let t = sc.tick;
            if run(&mut bot, &mut sc, 1)[0].set_team == Some(0) {
                joins.push(t);
            }
        }
        assert_eq!(joins.len(), 10, "capped at max_join_attempts: {joins:?}");
        for w in joins.windows(2) {
            assert!(w[1] - w[0] >= 150, "JOIN_RETRY_TICKS: {joins:?}");
        }
        let evs: Vec<_> = bot.drain_events().collect();
        assert!(evs.iter().any(|e| matches!(e, BotEvent::JoinGaveUp { .. })));
    });
}

#[test]
fn a_player_table_change_reports_the_roster_and_blocks_are_attributed() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1200)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 4);
        // We hook p1, then it freezes within the credit window: a block for us.
        sc.tee_mut(0).hooked_player = 1;
        sc.tee_mut(0).hook_state = 5;
        run_active(&mut bot, &mut sc, &[1], 3);
        sc.tee_mut(0).hooked_player = -1;
        sc.tee_mut(0).hook_state = 0;
        run_active(&mut bot, &mut sc, &[1], 2);
        sc.tee_mut(1).frozen = true;
        run_active(&mut bot, &mut sc, &[], 2);
        assert_eq!(bot.block_stats().blocks, 1, "attributed to us");
        let evs: Vec<_> = bot.drain_events().collect();
        let blocked: Vec<_> = evs
            .iter()
            .filter_map(|e| {
                if let BotEvent::Block { victim, .. } = e {
                    Some(victim.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(blocked.len(), 1);
        assert!(
            blocked[0].starts_with("c1-") && !blocked[0].contains("p1"),
            "a tag, not a nickname: {}",
            blocked[0]
        );
    });
}

// --- the no-chat / no-nickname guarantees --------------------------------------------------------

#[test]
fn the_output_type_has_no_chat_and_events_carry_only_tags() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        let mut rel_names = bot.relations().len(ListKind::Friend);
        rel_names += 0;
        let _ = rel_names;
        run_active(&mut bot, &mut sc, &[1], 6);
        for e in bot.drain_events() {
            let s = format!("{e:?}");
            assert!(!s.contains("\"p1\"") && !s.contains("name"), "{s}");
        }
        // `Output` is the whole surface the shell can act on: input, kill, kill_command (a bool: the typed `/kill`, D-078), set_team.
        // There is no field that could carry text (asserted structurally by destructuring it exhaustively).
        let ddai_bot::Output {
            input: _,
            kill: _,
            kill_command: _,
            set_team: _,
            tag: _,
        } = ddai_bot::Output::default();
    });
}

// --- allocation -----------------------------------------------------------------------------------

/// D-042 / acceptance 5: no allocation per snapshot in steady state for the bot's own code. The
/// idle brain and a quiet scene (nobody frozen, no list hits) keep the planner's allocating helpers
/// (seal check, guard, rope veto) out of the picture — those belong to `ddai-planner`, and the
/// brain itself is excluded by construction (`Brain::decide_in` of `IdleBrain` allocates nothing).
#[test]
fn a_steady_state_snapshot_allocates_nothing_in_the_bots_own_code() {
    support::big_stack(|| {
        let mut bot = bot_with(Box::new(ddai_brain::IdleBrain), cfg(BrainKind::Idle), Relations::new());
        // Fight mode with a target so the whole path (keep mask, prediction, observation, brain call,
        // filters, encoding) runs; the idle kind still wanders nothing.
        let map = room(&[]);
        bot.on_map_loaded(map.clone());
        let mut tees = vec![tee(0, 1000)];
        for i in 1..=6 {
            tees.push(tee(i, 1000 + 120 * i));
        }
        let mut sc = Scenario::new(map, tees);
        // Warm-up: first snapshots create the world, grow the scratch buffers, settle the tees.
        for _ in 0..60 {
            for i in 1..=6 {
                wiggle(&mut sc, i);
            }
            run(&mut bot, &mut sc, 1);
        }
        assert_eq!(bot.target_id(), 1, "a target, so the brain path runs");
        // Pre-build the snapshots: constructing them allocates (it is the driver's job, not the bot's).
        let mut snaps = Vec::new();
        for _ in 0..200 {
            for i in 1..=6 {
                wiggle(&mut sc, i);
            }
            snaps.push(sc.snapshot());
            sc.tick += 2;
        }
        let info = allocation_counter::measure(|| {
            for snap in &snaps {
                let out = bot.on_snapshot(snap);
                std::hint::black_box(out);
                bot.on_input_sent(snap.pred_tick + 1, &out.input.expect("decided"));
            }
        });
        assert_eq!(
            info.count_total, 0,
            "allocations per snapshot in steady state: {info:?}"
        );
        assert!(bot.stats().brain_decisions >= 200, "{:?}", bot.stats());
    });
}

// --- review round 1: F1 hammer veto, F2 respawn after Cl_Kill, F3 spectator stop, F6 ---------------

/// Whether the press `prev -> now` of the fire counter is a new press.
fn pressed(prev: i32, now: i32) -> bool {
    now != prev && now & 1 == 1
}

#[test]
fn a_hammer_swing_that_would_hit_a_spared_tee_is_withheld_and_a_clear_one_is_not() {
    support::big_stack(|| {
        // Us at 1000, target p1 at 1060, a friend p2 at 1030 between us: a swing to the right is
        // centred 21 px in front of us (x = 1021), 9 px from the friend. Weapon 0 is the hammer.
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let (mut bot, mut sc, (_, _, act)) = setup(vec![tee(0, 1000), tee(1, 1060), tee(2, 1030)], rel);
        *act.borrow_mut() = Action {
            fire: true,
            target: IVec2::new(300, 0),
            ..neutral()
        };
        run_active(&mut bot, &mut sc, &[1, 2], 6);
        let mut prev = 0;
        for _ in 0..10 {
            wiggle(&mut sc, 1);
            wiggle(&mut sc, 2);
            let input = run(&mut bot, &mut sc, 1)[0].input.unwrap();
            assert!(
                !pressed(prev, input.fire),
                "the swing would hit the friend: fire {}",
                input.fire
            );
            prev = input.fire;
        }
        assert!(bot.stats().vetoed_fires >= 10, "{}", bot.stats().vetoed_fires);

        // Swinging the other way (the friend is 51 px from the swing centre at x = 979... clear).
        *act.borrow_mut() = Action {
            fire: true,
            target: IVec2::new(-300, 0),
            ..neutral()
        };
        let vetoed = bot.stats().vetoed_fires;
        let mut presses = 0;
        for _ in 0..4 {
            wiggle(&mut sc, 1);
            wiggle(&mut sc, 2);
            let input = run(&mut bot, &mut sc, 1)[0].input.unwrap();
            presses += i32::from(pressed(prev, input.fire));
            prev = input.fire;
        }
        assert!(presses >= 3, "a clear swing keeps its fire presses: {presses}");
        assert_eq!(bot.stats().vetoed_fires, vetoed, "nothing more was vetoed");
    });
}

/// Review F9 (task 4.1b): for 1-3 ticks after a spawn the snapshot still says the gun is in hand, yet
/// the press asks for the hammer (the default) and `FireWeapon` switches before it fires.
#[test]
fn a_swing_in_the_first_ticks_after_a_spawn_is_vetoed_although_the_snapshot_says_gun() {
    support::big_stack(|| {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let (mut bot, mut sc, (_, _, act)) = setup(vec![tee(0, 1000), tee(1, 1060), tee(2, 1030)], rel);
        *act.borrow_mut() = Action {
            fire: true,
            target: IVec2::new(300, 0),
            ..neutral()
        };
        sc.tee_mut(0).weapon = 1; // just spawned: the gun is the active weapon
        run_active(&mut bot, &mut sc, &[1, 2], 6);
        let mut prev = 0;
        for _ in 0..4 {
            wiggle(&mut sc, 1);
            wiggle(&mut sc, 2);
            let input = run(&mut bot, &mut sc, 1)[0].input.unwrap();
            assert_eq!(input.wanted_weapon, 1, "the encoder asks for the hammer");
            assert!(!pressed(prev, input.fire), "the swing would hit the friend: {input:?}");
            prev = input.fire;
        }
        assert!(bot.stats().vetoed_fires >= 4, "{}", bot.stats().vetoed_fires);
        // An action that names the gun is a bullet, not a swing: not vetoed.
        *act.borrow_mut() = Action {
            fire: true,
            target: IVec2::new(300, 0),
            wanted_weapon: Some(1),
            ..neutral()
        };
        let vetoed = bot.stats().vetoed_fires;
        run(&mut bot, &mut sc, 3);
        assert_eq!(bot.stats().vetoed_fires, vetoed);
    });
}

/// The real hybrid with its fire key forced down and its aim forced at the target: static tees give
/// it no reason to swing, so this makes it the worst case the veto must hold against. Everything
/// else (reset, live context, the world it plans in) is the hybrid's own.
struct ForceFire(Box<dyn ddai_brain::Brain>);

impl ddai_brain::Brain for ForceFire {
    fn reset(&mut self, ctx: &ddai_brain::ResetContext) {
        self.0.reset(ctx);
    }
    fn decide(&mut self, obs: &ddai_brain::Observation) -> Action {
        self.0.decide(obs)
    }
    fn decide_in(&mut self, obs: &ddai_brain::Observation, view: Option<&ddai_brain::WorldView<'_>>) -> Action {
        let mut a = self.0.decide_in(obs, view);
        a.fire = true;
        a.target = IVec2::new(300, 0);
        a
    }
    fn set_live_context(&mut self, ctx: &ddai_brain::LiveContext<'_>) {
        self.0.set_live_context(ctx);
    }
    fn name(&self) -> &str {
        "force-fire(hybrid)"
    }
}

/// `--brain hybrid` end to end: a friend stands in the swing's path; whatever the hybrid decides, no
/// fire press may leave the bot while the hammer would hit the friend. Today the hybrid ignores the
/// live context (task 3.5b adds it), so this is the bot's own second line of defence at work.
#[test]
fn hybrid_with_a_spared_tee_in_hammer_reach_never_presses_fire_into_it() {
    support::big_stack(|| {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let map = room(&[]);
        let hybrid = ddai_bot::make_brain(BrainKind::Hybrid, &ddai_bot::BrainOptions::default()).expect("hybrid");
        let mut bot = bot_with(Box::new(ForceFire(hybrid)), cfg(BrainKind::Hybrid), rel);
        bot.on_map_loaded(std::sync::Arc::clone(&map));
        // The target stands 60 px to the right (in hammer reach), the friend 30 px (also in reach, and
        // in front of the target on the line).
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1060), tee(2, 1030)]);
        let mut prev = 0;
        for _ in 0..60 {
            wiggle(&mut sc, 1);
            wiggle(&mut sc, 2);
            let input = run(&mut bot, &mut sc, 1)[0].input.unwrap();
            assert!(
                !pressed(prev, input.fire),
                "a fire press toward the friend at 1030: {input:?}"
            );
            prev = input.fire;
        }
        assert!(
            bot.stats().brain_decisions >= 50,
            "the hybrid decided: {:?}",
            bot.stats()
        );
        assert!(
            bot.stats().vetoed_fires >= 50,
            "every swing was withheld: {:?}",
            bot.stats()
        );

        // Control: the same hybrid and the same forced swing with no friend in the way presses fire.
        let map = room(&[]);
        let hybrid = ddai_bot::make_brain(BrainKind::Hybrid, &ddai_bot::BrainOptions::default()).expect("hybrid");
        let mut bot = bot_with(Box::new(ForceFire(hybrid)), cfg(BrainKind::Hybrid), Relations::new());
        bot.on_map_loaded(std::sync::Arc::clone(&map));
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1060)]);
        let mut presses = 0;
        let mut prev = 0;
        for _ in 0..20 {
            wiggle(&mut sc, 1);
            let input = run(&mut bot, &mut sc, 1)[0].input.unwrap();
            presses += i32::from(pressed(prev, input.fire));
            prev = input.fire;
        }
        assert!(presses >= 15, "without a spared tee the swing goes out: {presses}");
        assert_eq!(bot.stats().vetoed_fires, 0);
    });
}

#[test]
fn our_own_kill_message_starts_a_new_life_even_though_no_snapshot_is_ever_without_us() {
    support::big_stack(|| {
        // F2 (review round 1): DDNet respawns right after Cl_Kill, so the tee is never absent. The
        // kill message must end the life; the next snapshot with us in it starts the next.
        let (mut bot, mut sc, (_, resets, _)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 4);
        assert_eq!(resets.borrow().len(), 1);
        bot.on_kill_message(-1, 0, -1);
        assert_eq!(bot.stats().deaths, 1, "counted once");
        sc.tee_mut(0).x = 1500; // the spawn point
        run_active(&mut bot, &mut sc, &[1], 3);
        assert_eq!(resets.borrow().len(), 2, "the brain is reset for the new life");
        assert_ne!(resets.borrow()[0].seed, resets.borrow()[1].seed);
        let evs: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            evs.iter().filter(|e| matches!(e, BotEvent::Respawned { .. })).count(),
            2,
            "first spawn and the respawn: {evs:?}"
        );
        assert_eq!(bot.stats().deaths, 1, "and still one death");
        // The same message again (a late duplicate) does not count a second death.
        bot.on_kill_message(-1, 0, -1);
        bot.on_kill_message(-1, 0, -1);
        assert_eq!(bot.stats().deaths, 2, "one per life");
        // Somebody else's death never touches us.
        bot.on_kill_message(-1, 1, -1);
        assert_eq!(bot.stats().deaths, 2);
    });
}

#[test]
fn a_death_seen_both_as_a_message_and_as_an_absent_snapshot_is_counted_once() {
    support::big_stack(|| {
        let (mut bot, mut sc, (_, resets, _)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 4);
        bot.on_kill_message(-1, 0, -1);
        let us = sc.tees.remove(0);
        run_active(&mut bot, &mut sc, &[1], 4);
        sc.tees.insert(0, us);
        run_active(&mut bot, &mut sc, &[1], 3);
        assert_eq!(bot.stats().deaths, 1);
        assert_eq!(resets.borrow().len(), 2);
    });
}

#[test]
fn moved_to_the_spectators_after_playing_stops_the_bot_without_a_rejoin() {
    support::big_stack(|| {
        // F3: a moderation signal (D-016): stay a spectator, stop, never ask to join again.
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        assert!(bot.stop_reason().is_none());
        sc.tees.remove(0);
        sc.player_mut(0).team = -1;
        let mut joins = 0;
        for _ in 0..1000 {
            joins += usize::from(run(&mut bot, &mut sc, 1)[0].set_team.is_some());
        }
        assert_eq!(joins, 0, "no rejoin after having played");
        assert_eq!(bot.stop_reason(), Some(ddai_bot::bot::StopReason::MovedToSpectators));
        let evs: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            evs.iter()
                .filter(|e| matches!(e, BotEvent::MovedToSpectators { .. }))
                .count(),
            1,
            "reported once: {evs:?}"
        );
    });
}

#[test]
fn an_ordinary_death_is_not_a_move_to_the_spectators() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 6);
        sc.tees.remove(0); // dead, team 0
        run_active(&mut bot, &mut sc, &[1], 30);
        assert!(bot.stop_reason().is_none());
    });
}

#[test]
fn the_join_cap_is_per_run_and_does_not_reset_when_a_join_succeeds() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        let map = std::sync::Arc::clone(&sc.map);
        let us = sc.tees.remove(0);
        sc.player_mut(0).team = -1;
        let count_joins = |bot: &mut Bot, sc: &mut Scenario, n: usize| {
            (0..n).filter(|_| run(bot, sc, 1)[0].set_team == Some(0)).count()
        };
        let first = count_joins(&mut bot, &mut sc, 400); // ~4 tries in 800 ticks
        assert!((3..=6).contains(&first), "{first}");
        // The server lets us in; we play; the map changes; we are in the spectators again.
        sc.tees.insert(0, us.clone());
        sc.player_mut(0).team = 0;
        run_active(&mut bot, &mut sc, &[1], 4);
        bot.on_map_changing();
        bot.on_map_loaded(map);
        sc.tees.remove(0);
        sc.player_mut(0).team = -1;
        let second = count_joins(&mut bot, &mut sc, 2000);
        assert_eq!(first + second, 10, "the cap counts the whole run: {first} + {second}");
    });
}

#[test]
fn the_wayblock_kill_hook_gets_the_real_frozen_duration() {
    use ddai_bot::hooks::{HookContext, WayBlock};
    use std::cell::RefCell;
    use std::rc::Rc;
    struct Recorder(Rc<RefCell<Vec<i32>>>);
    impl WayBlock for Recorder {
        fn holding(&self) -> bool {
            true
        }
        fn wants_kill(&mut self, _ctx: &HookContext<'_>, frozen_for: i32) -> bool {
            self.0.borrow_mut().push(frozen_for);
            false
        }
    }
    support::big_stack(|| {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let (probe, ..) = Probe::new(neutral());
        let hooks = Hooks {
            wayblock: Box::new(Recorder(Rc::clone(&seen))),
            ..Hooks::default()
        };
        let mut bot = Bot::new(cfg(BrainKind::Planner), Box::new(probe), hooks, Relations::new());
        let map = room(&[]);
        bot.on_map_loaded(std::sync::Arc::clone(&map));
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100)]);
        sc.tee_mut(0).frozen = true;
        run_active(&mut bot, &mut sc, &[1], 30);
        let seen = seen.borrow();
        assert_eq!(seen[0], 0, "first frozen snapshot");
        assert!(*seen.last().unwrap() >= 50, "grows with the frozen time: {seen:?}");
    });
}

// --- review round 2, F6: re-basing and late adoption through the driver's real adoption rules -----

/// How many presses `CountInput` sees between two inputs: every odd counter value in `(prev, cur]`.
fn wire_presses(prev: i32, cur: i32) -> i32 {
    (prev + 1..=cur).filter(|v| v & 1 == 1).count() as i32
}

/// The bot's decisions go through `ddai_client::InputState`, exactly as the driver adopts them. Every
/// decision presses (fire level true). The script: decision 0 adopted on time (one press); decision 1
/// replaced by decision 2 before its tick (its press never goes out); decision 2 adopted 2 ticks late
/// (still pressed); decision 3 adopted 3 ticks late (no press).
#[test]
fn the_wire_carries_exactly_the_presses_of_decisions_adopted_in_time() {
    support::big_stack(|| {
        let (mut bot, mut sc, (_, _, act)) = setup(vec![tee(0, 1000), tee(1, 1300)], Relations::new());
        *act.borrow_mut() = Action {
            fire: true,
            target: IVec2::new(300, 0),
            ..neutral()
        };
        run_active(&mut bot, &mut sc, &[1], 4); // warm up (the encoder counter is nonzero by now)
        let mut driver = ddai_client::InputState::new();
        let decide = |bot: &mut Bot, sc: &mut Scenario, driver: &mut ddai_client::InputState| {
            wiggle(sc, 1);
            let snap = sc.snapshot();
            let out = bot.on_snapshot(&snap);
            sc.tick += 2;
            let tag = out.tag.expect("a brain decision is tagged");
            driver.decide(out.input.expect("a decision"), snap.arrived, Some(tag));
            tag.expected_tick
        };
        let mut wire = driver.current().fire;
        let take = |driver: &ddai_client::InputState, wire: &mut i32| {
            let now = driver.current().fire;
            let n = wire_presses(*wire, now);
            *wire = now;
            n
        };
        // Decision 0: adopted on time.
        let e0 = decide(&mut bot, &mut sc, &mut driver);
        driver.adopt_if_due(Some(e0));
        assert_eq!(take(&driver, &mut wire), 1, "one press on time");
        // Decision 1 is replaced by decision 2 before its tick comes: its press never goes out.
        let e1 = decide(&mut bot, &mut sc, &mut driver);
        driver.adopt_if_due(Some(e1 - 1));
        assert_eq!(take(&driver, &mut wire), 0, "held, nothing sent");
        let e2 = decide(&mut bot, &mut sc, &mut driver);
        assert_eq!(driver.superseded(), 1);
        // Decision 2 is adopted 2 ticks late: still one press (the veto looked 2 ticks on).
        driver.adopt_if_due(Some(e2 + ddai_client::MAX_LATE_PRESS_TICKS));
        // The bot's own counter assumed decision 1 went out; the wire still shows exactly ONE press for it.
        assert_eq!(
            take(&driver, &mut wire),
            1,
            "decision 1's press did not leak into decision 2's"
        );
        // Decision 3 is adopted 3 ticks late: its press is dropped, the hammer cannot swing at a tick the
        // veto did not check.
        let e3 = decide(&mut bot, &mut sc, &mut driver);
        driver.adopt_if_due(Some(e3 + ddai_client::MAX_LATE_PRESS_TICKS + 1));
        let released = take(&driver, &mut wire);
        assert_eq!(released, 0, "no press 3 ticks late");
        assert_eq!(driver.current().fire & 1, 0, "even parity: nothing held");
        assert_eq!(driver.late_presses_dropped(), 1);
        // The next decision on time presses again.
        let e4 = decide(&mut bot, &mut sc, &mut driver);
        driver.adopt_if_due(Some(e4));
        assert_eq!(take(&driver, &mut wire), 1);
    });
}

#[test]
fn no_swing_at_a_spared_tee_reaches_the_wire_whatever_the_adoption_pattern() {
    support::big_stack(|| {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        let (mut bot, mut sc, (_, _, act)) = setup(vec![tee(0, 1000), tee(1, 1060), tee(2, 1030)], rel);
        *act.borrow_mut() = Action {
            fire: true,
            target: IVec2::new(300, 0),
            ..neutral()
        };
        let mut driver = ddai_client::InputState::new();
        let mut wire = 0;
        let mut presses = 0;
        for i in 0..40 {
            wiggle(&mut sc, 1);
            wiggle(&mut sc, 2);
            let snap = sc.snapshot();
            let out = bot.on_snapshot(&snap);
            sc.tick += 2;
            let tag = out.tag.expect("tagged");
            driver.decide(out.input.unwrap(), snap.arrived, Some(tag));
            // On time, late, or replaced by the next one (no adoption this round).
            match i % 3 {
                0 => driver.adopt_if_due(Some(tag.expected_tick)),
                1 => driver.adopt_if_due(Some(tag.expected_tick + 4)),
                _ => {}
            }
            let now = driver.current().fire;
            presses += wire_presses(wire, now);
            wire = now;
        }
        assert_eq!(presses, 0, "the friend stands in the hammer's reach: no press ever");
        assert!(bot.stats().vetoed_fires >= 30);
    });
}

// --- task 4.2 (review F6, F9): the navigator hooks through the whole bot, with a stub navigator -------

mod nav_stub {
    use super::*;
    use ddai_bot::hooks::{HookContext, NavStep, Navigator, Poll, WayBlock};
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    #[derive(Default)]
    pub struct Log {
        pub vetoed: u32,
        pub kills: Vec<(i32, bool)>,
        pub polls: u32,
    }

    /// Answers `drive` with the scripted steps (the last one repeats), `poll` with the scripted mode
    /// requests, and records what the bot tells it back.
    pub struct Stub {
        pub log: Rc<RefCell<Log>>,
        pub steps: VecDeque<Option<NavStep>>,
        pub modes: VecDeque<Option<Mode>>,
    }

    impl Navigator for Stub {
        fn poll(&mut self, _ctx: &HookContext<'_>) -> Poll {
            self.log.borrow_mut().polls += 1;
            Poll {
                mode: self.modes.pop_front().flatten(),
                knowledge: None,
            }
        }
        fn drive(&mut self, _ctx: &HookContext<'_>) -> Option<NavStep> {
            if self.steps.len() > 1 {
                self.steps.pop_front().flatten()
            } else {
                self.steps.front().copied().flatten()
            }
        }
        fn vetoed(&mut self) {
            self.log.borrow_mut().vetoed += 1;
        }
        fn kill_sent(&mut self, tick: i32, by_route: bool) {
            self.log.borrow_mut().kills.push((tick, by_route));
        }
    }

    pub struct WantsKill;
    impl WayBlock for WantsKill {
        fn holding(&self) -> bool {
            true
        }
        fn wants_kill(&mut self, _ctx: &HookContext<'_>, _frozen_for: i32) -> bool {
            true
        }
    }

    pub fn bot_with_nav(
        steps: Vec<Option<NavStep>>,
        modes: Vec<Option<Mode>>,
        map: std::sync::Arc<ddai_physics::map::MapData>,
        wb: bool,
    ) -> (Bot, Rc<RefCell<Log>>) {
        let log = Rc::new(RefCell::new(Log::default()));
        let mut hooks = Hooks {
            navigator: Box::new(Stub {
                log: Rc::clone(&log),
                steps: steps.into(),
                modes: modes.into(),
            }),
            ..Hooks::default()
        };
        if wb {
            hooks.wayblock = Box::new(WantsKill);
        }
        let (probe, ..) = Probe::new(neutral());
        let mut bot = Bot::new(cfg(BrainKind::Planner), Box::new(probe), hooks, Relations::new());
        bot.on_map_loaded(map);
        (bot, log)
    }
}

/// `n` snapshots with tee 1 wiggling (so it never looks AFK), returning the outputs.
fn run_wiggling(bot: &mut Bot, sc: &mut Scenario, n: usize) -> Vec<ddai_bot::Output> {
    let mut outs = Vec::new();
    for _ in 0..n {
        wiggle(sc, 1);
        outs.extend(run(bot, sc, 1));
    }
    outs
}

fn run_right() -> Action {
    Action {
        direction: 1,
        ..neutral()
    }
}

#[test]
fn a_guarded_nav_step_that_runs_into_freeze_is_vetoed_and_the_navigator_is_told() {
    use ddai_bot::hooks::NavStep;
    support::big_stack(|| {
        for (guard, expect_veto) in [(true, true), (false, false)] {
            let map = room(&[(36, 38, FREEZE)]);
            let (mut bot, log) = nav_stub::bot_with_nav(
                vec![Some(NavStep::Input {
                    action: run_right(),
                    guard,
                })],
                vec![],
                map.clone(),
                false,
            );
            let mut sc = Scenario::new(map, vec![tee(0, 36 * 32 - 1), tee(1, 600)]);
            let outs = run_wiggling(&mut bot, &mut sc, 6);
            let vetoed = log.borrow().vetoed;
            assert_eq!(vetoed > 0, expect_veto, "guard={guard}: vetoes {vetoed}");
            if expect_veto {
                assert!(
                    vetoed as u64 <= bot.stats().guarded_inputs,
                    "every veto is a guarded input"
                );
                assert!(
                    outs.iter().any(|o| o.input.is_some_and(|i| i.direction != 1)),
                    "the sent input changed"
                );
            } else {
                assert_eq!(
                    bot.stats().guarded_inputs,
                    0,
                    "crossing / planned-freeze steps skip the guard"
                );
                assert!(
                    outs.iter().all(|o| o.input.is_some_and(|i| i.direction == 1)),
                    "passed through"
                );
            }
        }
    });
}

#[test]
fn the_nav_guard_checks_the_world_our_inflight_inputs_lead_to_not_the_snapshot() {
    use ddai_bot::hooks::NavStep;
    support::big_stack(|| {
        // The tee stands 36 px short of a freeze tile. Run right from standing still: two more ticks of
        // it do not reach the freeze, so the guard lets the step through. With "run right" already in
        // flight for the 4 ticks up to the driver's pred tick, the tee will be at the edge when this
        // input acts: the same step must now be vetoed (TS `guard` rolls `lagTicks` ticks first).
        let vetoes_with = |in_flight_right: bool| {
            let map = room(&[(36, 38, FREEZE)]);
            let (mut bot, log) = nav_stub::bot_with_nav(
                vec![Some(NavStep::Input {
                    action: run_right(),
                    guard: true,
                })],
                vec![],
                map.clone(),
                false,
            );
            let mut sc = Scenario::new(map, vec![tee(0, 36 * 32 - 36), tee(1, 600)]);
            sc.pred_ahead = 4;
            for _ in 0..3 {
                wiggle(&mut sc, 1);
                let snap = sc.snapshot();
                if in_flight_right {
                    for t in (snap.tick - 6)..=snap.pred_tick {
                        let mut input = ddai_bot::input::neutral_input(0);
                        input.direction = 1;
                        bot.on_input_sent(t, &input);
                    }
                }
                bot.on_snapshot(&snap);
                sc.tick += 2;
            }
            log.borrow().vetoed
        };
        assert_eq!(
            vetoes_with(false),
            0,
            "from standing still the step is safe for two ticks"
        );
        assert!(
            vetoes_with(true) > 0,
            "with the lag in flight the same step walks into the freeze"
        );
    });
}

#[test]
fn a_nav_kill_goes_out_only_when_the_cooldown_allows_and_the_navigator_hears_of_it() {
    use ddai_bot::consts::KILL_COOLDOWN_TICKS;
    use ddai_bot::hooks::NavStep;
    support::big_stack(|| {
        let map = room(&[]);
        let (mut bot, log) = nav_stub::bot_with_nav(
            vec![Some(NavStep::Kill { action: neutral() })],
            vec![],
            map.clone(),
            false,
        );
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100)]);
        let mut sent = Vec::new();
        for _ in 0..700 {
            wiggle(&mut sc, 1);
            let t = sc.tick;
            if run(&mut bot, &mut sc, 1)[0].kill {
                sent.push(t);
            }
        }
        assert!(sent.len() >= 2, "a kill every 500 ticks over 1400 ticks: {sent:?}");
        for w in sent.windows(2) {
            assert!(w[1] - w[0] >= KILL_COOLDOWN_TICKS, "cooldown kept: {sent:?}");
        }
        let told = log.borrow().kills.clone();
        assert_eq!(
            told.len(),
            sent.len(),
            "the navigator hears of every kill that went out and no other"
        );
        assert!(told.iter().all(|&(_, by_route)| by_route), "and that they were its own");
        assert_eq!(bot.stats().self_kills as usize, sent.len());
    });
}

#[test]
fn the_wayblock_lying_kill_shares_the_cooldown_with_every_other_kill() {
    use ddai_bot::consts::{KILL_COOLDOWN_TICKS, WB_KILL_COOLDOWN_TICKS};
    use ddai_bot::hooks::NavStep;
    support::big_stack(|| {
        // WB alone: a frozen tee lying in the freeze tiles is killed every WB_KILL_COOLDOWN_TICKS.
        let map = room(&[(10, 38, FREEZE)]);
        let (mut bot, log) = nav_stub::bot_with_nav(vec![None], vec![], map.clone(), true);
        let mut sc = Scenario::new(map.clone(), vec![tee(0, 10 * 32 + 16), tee(1, 1100)]);
        sc.tee_mut(0).frozen = true;
        let mut kills = Vec::new();
        for _ in 0..200 {
            wiggle(&mut sc, 1);
            let t = sc.tick;
            if run(&mut bot, &mut sc, 1)[0].kill {
                kills.push(t);
            }
        }
        assert!(kills.len() >= 3, "{kills:?}");
        assert!(
            kills.windows(2).all(|w| w[1] - w[0] >= WB_KILL_COOLDOWN_TICKS),
            "never closer than the WB cooldown: {kills:?}"
        );
        assert!(
            kills.windows(2).any(|w| w[1] - w[0] < KILL_COOLDOWN_TICKS),
            "the WB rule is the short one: {kills:?}"
        );
        assert_eq!(
            log.borrow().kills.len(),
            kills.len(),
            "the navigator hears of the unstick's kills too"
        );
        assert!(log.borrow().kills.iter().all(|&(_, by_route)| !by_route));

        // A navigator kill first: the WB rule must wait its 100 ticks after it (one shared clock).
        let (mut bot, _log) = nav_stub::bot_with_nav(
            vec![Some(NavStep::Kill { action: neutral() }), None],
            vec![],
            map.clone(),
            true,
        );
        let mut sc = Scenario::new(map, vec![tee(0, 10 * 32 + 16), tee(1, 1100)]);
        sc.tee_mut(0).frozen = true;
        let mut kills = Vec::new();
        for _ in 0..120 {
            wiggle(&mut sc, 1);
            let t = sc.tick;
            if run(&mut bot, &mut sc, 1)[0].kill {
                kills.push(t);
            }
        }
        assert!(kills.len() >= 2, "{kills:?}");
        assert!(kills[1] - kills[0] >= WB_KILL_COOLDOWN_TICKS, "shared clock: {kills:?}");
    });
}

#[test]
fn the_navigators_mode_requests_are_applied_before_the_unstick_and_the_pipeline() {
    support::big_stack(|| {
        let map = room(&[]);
        let (mut bot, log) = nav_stub::bot_with_nav(
            vec![None],
            vec![None, Some(Mode::Goto), None, Some(Mode::Hold), Some(Mode::Fight)],
            map.clone(),
            false,
        );
        let mut sc = Scenario::new(map, vec![tee(0, 1000), tee(1, 1100)]);
        let mut seen = Vec::new();
        for _ in 0..5 {
            wiggle(&mut sc, 1);
            run(&mut bot, &mut sc, 1);
            seen.push(bot.mode());
        }
        assert_eq!(seen, vec![Mode::Fight, Mode::Goto, Mode::Goto, Mode::Hold, Mode::Fight]);
        assert_eq!(log.borrow().polls, 5, "polled every snapshot with a live tee");
        // Hold really holds: the request took effect on the snapshot that carried it.
        let (mut bot, _) = nav_stub::bot_with_nav(vec![None], vec![Some(Mode::Hold)], room(&[]), false);
        let mut sc = Scenario::new(room(&[]), vec![tee(0, 1000), tee(1, 1100)]);
        let out = run_wiggling(&mut bot, &mut sc, 1).pop().unwrap();
        assert_eq!(bot.mode(), Mode::Hold);
        assert_eq!(
            bot.stats().brain_decisions,
            0,
            "no brain decision in the snapshot that said hold: {out:?}"
        );
    });
}
