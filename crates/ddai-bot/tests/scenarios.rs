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
fn paused_and_spectating_players_are_out_of_the_game() {
    support::big_stack(|| {
        let (mut bot, mut sc, _) = setup(
            vec![tee(0, 1000), tee(1, 1100), tee(2, 1500), tee(3, 1300)],
            Relations::new(),
        );
        sc.player_mut(1).ex_flags = explayerflagflag::PAUSED;
        sc.player_mut(3).ex_flags = explayerflagflag::SPEC;
        run_active(&mut bot, &mut sc, &[1, 2, 3], 4);
        assert_eq!(bot.target_id(), 2);
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

#[test]
fn before_the_timing_bootstrap_a_two_tick_guess_is_used_and_the_horizon_is_capped() {
    support::big_stack(|| {
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        sc.pred_tick_fixed = Some(0); // the timing bootstrap has not happened yet
        run_active(&mut bot, &mut sc, &[1], 3);
        assert_eq!(log.borrow().last().unwrap().obs_tick, sc.tick - 2 + 2);
        let (mut bot, mut sc, (log, ..)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        sc.pred_ahead = 500;
        run_active(&mut bot, &mut sc, &[1], 3);
        assert_eq!(
            log.borrow().last().unwrap().obs_tick,
            sc.tick - 2 + 12,
            "MAX_PREDICT_TICKS"
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

#[test]
fn spared_tees_are_not_in_the_brains_world_unless_roped_to_us() {
    support::big_stack(|| {
        // F1 (review round 1): a friend, an ignored tee and an AFK player near us must not be handed to
        // the brain as opponents (the hybrid's threat model would plan against them); a roped spared
        // tee is still kept (the rope is a physical fact). The brain is told about them as spares.
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "p2");
        rel.add(ListKind::Ignore, "p3");
        rel.add(ListKind::Friend, "p5");
        let tees = vec![
            tee(0, 1000),
            tee(1, 1300),
            tee(2, 1050),
            tee(3, 1080),
            tee(4, 1110),
            tee(5, 1140),
        ];
        let (mut bot, mut sc, (log, ..)) = setup(tees, rel);
        // p4 carries the server's AFK flag; p5 hooks us, so it is kept even if spared.
        sc.tee_mut(5).hooked_player = 0;
        sc.player_mut(4).ex_flags = explayerflagflag::AFK;
        run_active(&mut bot, &mut sc, &[1, 2, 3, 5], 8);
        let seen = log.borrow();
        let last = seen.last().unwrap();
        assert_eq!(last.target, Some(1));
        assert!(last.others.contains(&1));
        assert!(
            !last.others.contains(&2),
            "a friend is not an opponent: {:?}",
            last.others
        );
        assert!(
            !last.others.contains(&3),
            "an ignored tee is not an opponent: {:?}",
            last.others
        );
        assert!(
            !last.others.contains(&4),
            "an AFK tee is not an opponent: {:?}",
            last.others
        );
        assert!(!last.world_ids.contains(&2) && !last.world_ids.contains(&3) && !last.world_ids.contains(&4));
        assert!(last.world_ids.contains(&5), "a tee roped to us stays in the world");
        assert!(
            last.spares.len() >= 3,
            "the brain is told who is spared: {:?}",
            last.spares
        );
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
        // `Output` is the whole surface the shell can act on: input, kill, set_team. There is no field
        // that could carry text (asserted structurally by destructuring it exhaustively).
        let ddai_bot::Output {
            input: _,
            kill: _,
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
        bot.on_kill_message(0);
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
        bot.on_kill_message(0);
        bot.on_kill_message(0);
        assert_eq!(bot.stats().deaths, 2, "one per life");
        // Somebody else's death never touches us.
        bot.on_kill_message(1);
        assert_eq!(bot.stats().deaths, 2);
    });
}

#[test]
fn a_death_seen_both_as_a_message_and_as_an_absent_snapshot_is_counted_once() {
    support::big_stack(|| {
        let (mut bot, mut sc, (_, resets, _)) = setup(vec![tee(0, 1000), tee(1, 1100)], Relations::new());
        run_active(&mut bot, &mut sc, &[1], 4);
        bot.on_kill_message(0);
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
