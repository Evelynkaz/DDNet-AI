//! Task 4.12 (D-108): the smart self-kill policy and the automatic duel detection, as scenarios on synthetic snapshots.
//!
//! Every frozen-bot situation is played under both policies and the tick of the first kill (relative to the start of the freeze) is
//! compared: the legacy timers against the smart policy. The scenarios are the suite the task asks for: thaw soon, deep freeze, a pit
//! (resting on a freeze tile), a friend near / hooking, and a duel.

mod support;

use std::sync::Arc;

use ddai_bot::duel::DuelWhy;
use ddai_bot::smartkill::{SelfKillPolicy, SmartWhy};
use ddai_bot::{Bot, BotEvent, BrainKind, Relations};
use ddai_brain::Action;
use ddai_net::tuning::TeamsState;
use ddai_physics::map::MapData;
use support::*;

const PIT: [(u32, u32, u8); 2] = [(35, 38, FREEZE), (35, 37, FREEZE)];

fn neutral() -> Action {
    Action::neutral()
}

fn setup(policy: SelfKillPolicy, map: Arc<MapData>, tees: Vec<TeeSpec>, rel: Relations) -> (Bot, Scenario) {
    let (probe, _, _, _) = Probe::new(neutral());
    let mut c = cfg(BrainKind::Planner);
    c.selfkill_policy = policy;
    let mut bot = bot_with(Box::new(probe), c, rel);
    bot.on_map_loaded(Arc::clone(&map));
    (bot, Scenario::new(map, tees))
}

fn wiggle(sc: &mut Scenario, id: i32) {
    let t = sc.tee_mut(id);
    t.angle = (t.angle + 37) % 1000;
}

/// What a played scenario shows.
struct Play {
    /// Ticks since the freeze began at which a `Cl_Kill` went out.
    kills: Vec<i32>,
    events: Vec<BotEvent>,
}

impl Play {
    fn first(&self) -> Option<i32> {
        self.kills.first().copied()
    }
    fn skipped(&self) -> Vec<SmartWhy> {
        self.events
            .iter()
            .filter_map(|e| match e {
                BotEvent::SelfKillSkipped { why, .. } => Some(*why),
                _ => None,
            })
            .collect()
    }
    fn smart_whys(&self) -> Vec<SmartWhy> {
        self.events
            .iter()
            .filter_map(|e| match e {
                BotEvent::Killed { why: Some(w), .. } => Some(*w),
                _ => None,
            })
            .collect()
    }
}

/// Plays `steps` snapshots (2 ticks each) after 3 unfrozen warm-up ones. `each` runs before every snapshot with the number of ticks since
/// the freeze began (it puts the tee into the state the scenario wants).
fn play(
    policy: SelfKillPolicy,
    map: Arc<MapData>,
    tees: Vec<TeeSpec>,
    rel: Relations,
    steps: usize,
    mut each: impl FnMut(&mut Scenario, i32),
) -> (Play, Bot) {
    let (mut bot, mut sc) = setup(policy, map, tees, rel);
    for _ in 0..3 {
        wiggle(&mut sc, 1);
        run(&mut bot, &mut sc, 1);
    }
    let t0 = sc.tick;
    let mut kills = Vec::new();
    for _ in 0..steps {
        wiggle(&mut sc, 1);
        let since = sc.tick - t0;
        each(&mut sc, since);
        let out = run(&mut bot, &mut sc, 1).pop().unwrap();
        if out.kill {
            kills.push(since);
        }
    }
    let events = bot.drain_events().collect();
    (Play { kills, events }, bot)
}

fn pit_tee() -> TeeSpec {
    let mut t = tee(0, 35 * 32 + 16);
    t.y = 37 * 32 + 16;
    t
}

fn friend_rel() -> Relations {
    let mut rel = Relations::new();
    rel.add(ddai_bot::relations::ListKind::Friend, "p2");
    rel
}

fn freeze_pit(sc: &mut Scenario, _since: i32) {
    sc.tee_mut(0).frozen = true;
}

fn in_range(v: Option<i32>, lo: i32, hi: i32) -> bool {
    v.is_some_and(|v| (lo..=hi).contains(&v))
}

// ---- thaw soon -------------------------------------------------------------------------------------------------------------------

/// A freeze on open ground that runs down by itself (150 ticks, then the tee is free): nobody kills it, under either policy; the smart
/// policy says why it held back (it would otherwise have been `too early`, then `thaw soon`).
#[test]
fn a_freeze_that_ends_by_itself_is_never_killed() {
    support::big_stack(|| {
        for policy in [SelfKillPolicy::Legacy, SelfKillPolicy::Smart] {
            let (p, _) = play(
                policy,
                room(&[]),
                vec![tee(0, 1000), tee(1, 2000)],
                Relations::new(),
                120,
                |sc, since| {
                    let left = 130 - since;
                    let t = sc.tee_mut(0);
                    t.frozen = left > 0;
                    t.freeze_left = left.max(0);
                },
            );
            assert!(p.kills.is_empty(), "{policy:?}: {:?}", p.kills);
        }
    });
}

/// The same, but frozen for longer than the smart floor with a thaw 70 ticks away: the forecast is what keeps the kill back.
#[test]
fn the_forecast_holds_back_a_kill_the_floor_alone_would_allow() {
    support::big_stack(|| {
        let (p, bot) = play(
            SelfKillPolicy::Smart,
            room(&[]),
            vec![tee(0, 1000), tee(1, 2000)],
            Relations::new(),
            100,
            |sc, since| {
                let left = 130 - since;
                let t = sc.tee_mut(0);
                t.frozen = left > 0;
                t.freeze_left = left.max(0);
            },
        );
        assert!(p.kills.is_empty(), "{:?}", p.kills);
        assert_eq!(bot.stats().self_kills, 0);
    });
}

// ---- deep freeze -----------------------------------------------------------------------------------------------------------------

#[test]
fn a_deep_frozen_bot_is_killed_at_once_by_the_smart_policy_and_after_400_ticks_by_the_legacy_one() {
    support::big_stack(|| {
        let deep = |sc: &mut Scenario, _since: i32| {
            let t = sc.tee_mut(0);
            t.frozen = true;
            t.deep = true;
        };
        let (legacy, _) = play(
            SelfKillPolicy::Legacy,
            room(&[]),
            vec![tee(0, 1000), tee(1, 2000)],
            Relations::new(),
            260,
            deep,
        );
        assert!(in_range(legacy.first(), 400, 406), "{:?}", legacy.kills);
        let (smart, _) = play(
            SelfKillPolicy::Smart,
            room(&[]),
            vec![tee(0, 1000), tee(1, 2000)],
            Relations::new(),
            260,
            deep,
        );
        assert!(in_range(smart.first(), 50, 56), "{:?}", smart.kills);
        assert_eq!(smart.smart_whys().first(), Some(&SmartWhy::DeepFreeze));
        // The cooldown still holds between two kills.
        for w in smart.kills.windows(2) {
            assert!(w[1] - w[0] >= 500, "{:?}", smart.kills);
        }
    });
}

// ---- a pit (resting on a freeze tile: no exit) -----------------------------------------------------------------------------------

#[test]
fn a_bot_in_a_pit_is_killed_after_50_ticks_by_the_smart_policy_and_200_by_the_legacy_one() {
    support::big_stack(|| {
        let (legacy, _) = play(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
            200,
            freeze_pit,
        );
        assert!(in_range(legacy.first(), 200, 206), "{:?}", legacy.kills);
        let (smart, _) = play(
            SelfKillPolicy::Smart,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
            200,
            freeze_pit,
        );
        assert!(in_range(smart.first(), 50, 56), "{:?}", smart.kills);
        assert_eq!(smart.smart_whys().first(), Some(&SmartWhy::NoExit));
    });
}

/// An enemy near the pit is no rescuer: only friends count.
#[test]
fn a_hostile_tee_next_to_the_pit_does_not_hold_the_smart_policy_back() {
    support::big_stack(|| {
        let (p, _) = play(
            SelfKillPolicy::Smart,
            room(&PIT),
            vec![pit_tee(), tee(1, 35 * 32 + 100)],
            Relations::new(),
            100,
            freeze_pit,
        );
        assert!(in_range(p.first(), 50, 56), "{:?}", p.kills);
    });
}

// ---- a friend nearby -------------------------------------------------------------------------------------------------------------

/// A friend 300 px away (outside the legacy 140 px, inside the hook's reach) who does not hook: a short grace of 200 ticks (the legacy
/// in-tile timer), then the hopeless freeze is killed, like the legacy bot does (review 4.12, F3: not 1500).
#[test]
fn a_friend_merely_within_hook_reach_is_a_short_grace_not_a_long_wait() {
    support::big_stack(|| {
        let tees = || vec![pit_tee(), tee(1, 2000), tee(2, 35 * 32 + 16 + 300)];
        let (legacy, _) = play(
            SelfKillPolicy::Legacy,
            room(&PIT),
            tees(),
            friend_rel(),
            400,
            freeze_pit,
        );
        assert!(in_range(legacy.first(), 200, 206), "{:?}", legacy.kills);
        let (smart, _) = play(SelfKillPolicy::Smart, room(&PIT), tees(), friend_rel(), 400, freeze_pit);
        assert!(
            in_range(smart.first(), 200, 206),
            "{:?}: FRIEND_GRACE_TICKS",
            smart.kills
        );
        assert_eq!(smart.smart_whys().first(), Some(&SmartWhy::NoExit));
        assert!(
            smart.skipped().is_empty(),
            "the grace ended before the legacy timer was due: nothing was held back"
        );
    });
}

/// A friend who actually hooks us holds the kill back up to the legacy bound, 1500 ticks.
#[test]
fn a_friend_hooking_us_is_waited_for() {
    support::big_stack(|| {
        let mut tees = vec![pit_tee(), tee(1, 2000), tee(2, 35 * 32 + 16 + 340)];
        tees[2].hooked_player = 0;
        tees[2].hook_state = ddai_bot::tees::HOOK_GRABBED;
        let (legacy, _) = play(
            SelfKillPolicy::Legacy,
            room(&PIT),
            tees.clone(),
            friend_rel(),
            400,
            freeze_pit,
        );
        assert!(
            in_range(legacy.first(), 400, 406),
            "legacy: a hook does not stop the 400-tick limit: {:?}",
            legacy.kills
        );
        let (smart, _) = play(SelfKillPolicy::Smart, room(&PIT), tees, friend_rel(), 800, freeze_pit);
        assert!(in_range(smart.first(), 1500, 1506), "{:?}", smart.kills);
        assert!(
            smart.skipped().contains(&SmartWhy::FriendHooking),
            "{:?}",
            smart.skipped()
        );
    });
}

/// A friend who is frozen himself cannot help.
#[test]
fn a_frozen_friend_is_no_rescuer() {
    support::big_stack(|| {
        let mut tees = vec![pit_tee(), tee(1, 2000), tee(2, 35 * 32 + 16 + 100)];
        tees[2].frozen = true;
        let (smart, _) = play(SelfKillPolicy::Smart, room(&PIT), tees, friend_rel(), 100, freeze_pit);
        assert!(in_range(smart.first(), 50, 56), "{:?}", smart.kills);
    });
}

// ---- the default is the old behaviour --------------------------------------------------------------------------------------------

#[test]
fn the_policy_is_legacy_unless_asked() {
    assert_eq!(ddai_bot::BotConfig::default().selfkill_policy, SelfKillPolicy::Legacy);
    let (bot, _) = setup(SelfKillPolicy::Smart, room(&[]), vec![tee(0, 1000)], Relations::new());
    assert_eq!(bot.selfkill_policy(), SelfKillPolicy::Smart);
}

// ---- a duel ----------------------------------------------------------------------------------------------------------------------

fn team_of(assign: &[(usize, i32)]) -> Option<TeamsState> {
    let mut t = TeamsState {
        teams: [0; 128],
        received: 128,
    };
    for &(i, team) in assign {
        t.teams[i] = team;
    }
    Some(t)
}

/// The server's invitation line of the `/1vs1` minigame: the F-DDrace evidence the team signal needs (review 4.12, F1).
const INVITE: &str = "You have been invited to a fight by 'p1', type '/1vs1 p0' to join";

/// Frozen in the pit for 1000 ticks while our DDRace team holds exactly one other player: no `Cl_Kill`, no `/kill`, under either policy
/// and with the owner's switch off (the default): the duel is found by itself.
#[test]
fn in_a_two_player_team_the_bot_never_kills_itself_under_either_policy() {
    support::big_stack(|| {
        for policy in [SelfKillPolicy::Legacy, SelfKillPolicy::Smart] {
            let (mut bot, mut sc) = setup(policy, room(&PIT), vec![pit_tee(), tee(1, 2000)], Relations::new());
            sc.teams = team_of(&[(0, 7), (1, 7)]);
            bot.on_chat_line(-1, INVITE);
            let mut sent = 0;
            for _ in 0..500 {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                let out = run(&mut bot, &mut sc, 1).pop().unwrap();
                sent += usize::from(out.kill || out.kill_command);
            }
            assert_eq!(sent, 0, "{policy:?}");
            assert_eq!(bot.stats().self_kills, 0);
            assert!(bot.no_selfkill(), "the effective switch is on");
            assert_eq!(bot.duel(), Some(DuelWhy::Team));
            let events: Vec<_> = bot.drain_events().collect();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, BotEvent::DuelStarted { why: DuelWhy::Team, .. }))
                    .count(),
                1,
                "one line: {events:?}"
            );
        }
    });
}

/// Review 4.12 F1: a two-player DDRace team with no F-DDrace evidence (vanilla DDNet `/team`, an admin's `set_team_ddr`, F-DDrace's
/// Durak game) is NOT a duel: a deep-frozen bot is killed as usual, under either policy.
#[test]
fn a_two_player_team_without_f_ddrace_evidence_is_not_a_duel() {
    support::big_stack(|| {
        for policy in [SelfKillPolicy::Legacy, SelfKillPolicy::Smart] {
            let (mut bot, mut sc) = setup(policy, room(&PIT), vec![pit_tee(), tee(1, 2000)], Relations::new());
            sc.teams = team_of(&[(0, 7), (1, 7)]);
            let mut kills = 0;
            for _ in 0..300 {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                kills += usize::from(run(&mut bot, &mut sc, 1).pop().unwrap().kill);
            }
            assert!(kills >= 1, "{policy:?}");
            assert_eq!(bot.duel(), None);
            assert!(!bot.no_selfkill());
        }
    });
}

/// F-DDrace evidence and the team make a duel; when the fight is over (back in team 0) and a Durak-like two-player team follows with no
/// `/1vs1` chat, it is not one.
#[test]
fn a_durak_like_team_after_a_duel_is_not_a_duel() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(-1, INVITE);
        run(&mut bot, &mut sc, 3);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        sc.teams = team_of(&[]);
        run(&mut bot, &mut sc, 80); // 160 ticks: past the release time
        assert_eq!(bot.duel(), None);
        sc.teams = team_of(&[(0, 9), (1, 9)]);
        run(&mut bot, &mut sc, 20);
        assert_eq!(bot.duel(), None, "the evidence went with the fight");
    });
}

/// Control: alone in a team, or three in it, the same pit kills as usual.
#[test]
fn a_team_that_is_not_a_duel_changes_nothing() {
    support::big_stack(|| {
        for teams in [
            team_of(&[]),
            team_of(&[(0, 7)]),
            team_of(&[(0, 7), (1, 7), (2, 7)]),
            team_of(&[(0, 64), (1, 64)]),
        ] {
            let (mut bot, mut sc) = setup(
                SelfKillPolicy::Legacy,
                room(&PIT),
                vec![pit_tee(), tee(1, 2000), tee(2, 2200)],
                Relations::new(),
            );
            sc.teams = teams;
            bot.on_chat_line(-1, INVITE);
            let mut kills = 0;
            for _ in 0..200 {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                kills += usize::from(run(&mut bot, &mut sc, 1).pop().unwrap().kill);
            }
            assert!(kills >= 1, "{teams:?}");
            assert_eq!(bot.duel(), None);
            assert!(!bot.no_selfkill());
        }
    });
}

/// The duel starts and ends in the middle of a run: kills stop with the team and resume (after the release time) when it is gone; the
/// kill that was sent before is not followed by a `/kill`.
#[test]
fn the_duel_switches_the_bot_off_and_on_again_as_the_team_comes_and_goes() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        let mut log: Vec<(&str, usize)> = Vec::new();
        for (phase, teams, steps) in [
            ("before", None, 130),
            ("duel", team_of(&[(0, 9), (1, 9)]), 400),
            ("after", team_of(&[]), 400),
        ] {
            sc.teams = teams;
            if phase == "duel" {
                bot.on_chat_line(-1, INVITE);
            }
            let mut kills = 0;
            for _ in 0..steps {
                sc.tee_mut(0).frozen = true;
                wiggle(&mut sc, 1);
                let out = run(&mut bot, &mut sc, 1).pop().unwrap();
                kills += usize::from(out.kill);
                assert!(!(phase == "duel" && (out.kill || out.kill_command)), "{out:?}");
            }
            log.push((phase, kills));
        }
        assert_eq!(log[0].1, 1, "{log:?}: the legacy kill at 200 ticks of 260");
        assert_eq!(log[1].1, 0, "{log:?}");
        assert!(log[2].1 >= 1, "{log:?}: kills resume once the team is gone");
        let events: Vec<_> = bot.drain_events().collect();
        assert!(
            events.iter().any(|e| matches!(e, BotEvent::DuelStarted { .. })),
            "{events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, BotEvent::DuelEnded { .. })),
            "{events:?}"
        );
    });
}

/// The chat is the second signal: "you have accepted the invite" arms it before any team message, and the end line (naming us) ends it.
#[test]
fn the_f_ddrace_chat_lines_start_and_end_the_duel_without_a_teams_message() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        let step = |bot: &mut Bot, sc: &mut Scenario, n: usize| -> usize {
            let mut kills = 0;
            for _ in 0..n {
                sc.tee_mut(0).frozen = true;
                wiggle(sc, 1);
                kills += usize::from(run(bot, sc, 1).pop().unwrap().kill);
            }
            kills
        };
        step(&mut bot, &mut sc, 3);
        // A player typing the words is not the server.
        bot.on_chat_line(5, "You have accepted the invite by 'p1'");
        assert_eq!(bot.duel(), None);
        bot.on_chat_line(-1, "You have accepted the invite by 'p1'");
        assert_eq!(bot.duel(), Some(DuelWhy::Chat));
        assert!(bot.no_selfkill());
        assert_eq!(step(&mut bot, &mut sc, 400), 0, "no kill during the duel");
        // Somebody else's fight ends: not ours.
        bot.on_chat_line(-1, "'a' won a 1vs1 round against 'b'! Final scores: 10 - 7");
        assert_eq!(bot.duel(), Some(DuelWhy::Chat));
        // Ours ends (our name in the tests is `p0`).
        bot.on_chat_line(-1, "'p1' won a 1vs1 round against 'p0'! Final scores: 10 - 7");
        assert_eq!(bot.duel(), None);
        assert!(step(&mut bot, &mut sc, 300) >= 1, "kills resume after the round");
    });
}

/// The owner's own console `!kill` is not the bot's decision: it still goes out during a duel (D-102).
#[test]
fn the_owners_console_kill_is_still_allowed_during_a_duel() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Smart,
            room(&[]),
            vec![tee(0, 1000), tee(1, 1300)],
            Relations::new(),
        );
        sc.teams = team_of(&[(0, 3), (1, 3)]);
        bot.on_chat_line(-1, INVITE);
        for _ in 0..3 {
            wiggle(&mut sc, 1);
            run(&mut bot, &mut sc, 1);
        }
        assert!(bot.no_selfkill());
        assert!(bot.command(ddai_bot::BotCommand::Kill).ok);
        wiggle(&mut sc, 1);
        let out = run(&mut bot, &mut sc, 1).pop().unwrap();
        assert!(out.kill, "{out:?}");
    });
}

/// A map change forgets the duel (the teams do not outlive it).
#[test]
fn a_map_change_forgets_the_duel() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        sc.teams = team_of(&[(0, 3), (1, 3)]);
        bot.on_chat_line(-1, INVITE);
        run(&mut bot, &mut sc, 3);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        bot.on_map_loaded(Arc::clone(&sc.map));
        assert_eq!(bot.duel(), None);
        assert!(!bot.no_selfkill());
    });
}

/// `--no-duel-detect` (the escape hatch): the same two-player team, and the bot kills as usual; the chat lines do nothing either.
#[test]
fn with_the_detection_off_a_two_player_team_changes_nothing() {
    support::big_stack(|| {
        let (probe, _, _, _) = Probe::new(neutral());
        let mut c = cfg(BrainKind::Planner);
        c.duel_detect = false;
        let mut bot = bot_with(Box::new(probe), c, Relations::new());
        let map = room(&PIT);
        bot.on_map_loaded(Arc::clone(&map));
        let mut sc = Scenario::new(map, vec![pit_tee(), tee(1, 2000)]);
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(-1, "You have accepted the invite by 'p1'");
        let mut kills = 0;
        for _ in 0..200 {
            sc.tee_mut(0).frozen = true;
            wiggle(&mut sc, 1);
            kills += usize::from(run(&mut bot, &mut sc, 1).pop().unwrap().kill);
        }
        assert!(kills >= 1);
        assert_eq!(bot.duel(), None);
        assert!(!bot.no_selfkill());
    });
}

/// The marker file `duel-detect.off` (the runner calls `set_duel_detect` when it appears): a duel that was detected is forgotten at once and
/// the stuck bot is killed as usual; when the marker is gone the detector looks again.
#[test]
fn turning_the_detection_off_at_run_time_frees_a_bot_that_the_duel_held() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(-1, INVITE);
        let kills = |bot: &mut Bot, sc: &mut Scenario, n: usize| -> usize {
            let mut k = 0;
            for _ in 0..n {
                sc.tee_mut(0).frozen = true;
                wiggle(sc, 1);
                k += usize::from(run(bot, sc, 1).pop().unwrap().kill);
            }
            k
        };
        assert_eq!(kills(&mut bot, &mut sc, 300), 0);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        bot.set_duel_detect(false);
        assert_eq!(bot.duel(), None);
        assert!(!bot.no_selfkill());
        assert!(kills(&mut bot, &mut sc, 300) >= 1, "freed");
        bot.on_chat_line(-1, INVITE);
        assert_eq!(bot.duel(), None, "while off nothing is read");
        bot.set_duel_detect(true);
        bot.on_chat_line(-1, INVITE);
        kills(&mut bot, &mut sc, 3);
        assert_eq!(bot.duel(), Some(DuelWhy::Team), "on again");
    });
}

// ---- evidence from the owner's duel command (joniTee starts duels with `/duel`, not `/1vs1`) ---------------------------------------------

fn step_frozen(bot: &mut Bot, sc: &mut Scenario, n: usize) -> usize {
    let mut kills = 0;
    for _ in 0..n {
        sc.tee_mut(0).frozen = true;
        wiggle(sc, 1);
        kills += usize::from(run(bot, sc, 1).pop().unwrap().kill);
    }
    kills
}

/// The owner sent `/duel x` from the website (the bot itself sent the line) and our team holds one other player: a duel.
#[test]
fn an_owner_duel_command_and_a_two_player_team_make_a_duel() {
    support::big_stack(|| {
        for line in ["/duel x", "/DUEL", "/1vs1 Rival", "  /Duel   Rival  "] {
            let (mut bot, mut sc) = setup(
                SelfKillPolicy::Legacy,
                room(&PIT),
                vec![pit_tee(), tee(1, 2000)],
                Relations::new(),
            );
            sc.teams = team_of(&[(0, 7), (1, 7)]);
            bot.on_owner_command(line);
            assert_eq!(step_frozen(&mut bot, &mut sc, 400), 0, "{line:?}: no kill in the duel");
            assert_eq!(bot.duel(), Some(DuelWhy::Team), "{line:?}");
            let events: Vec<_> = bot.drain_events().collect();
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, BotEvent::DuelEvidence { len, .. } if *len == line.len())),
                "{events:?}: logged by length only"
            );
        }
    });
}

/// The command alone is no duel: with no two-player team the bot is killed as usual.
#[test]
fn an_owner_duel_command_without_a_team_is_no_duel() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        sc.teams = team_of(&[]);
        bot.on_owner_command("/duel x");
        assert!(step_frozen(&mut bot, &mut sc, 300) >= 1);
        assert_eq!(bot.duel(), None);
        assert!(!bot.no_selfkill());
    });
}

/// Only a line the bot sent counts: a player saying `/duel` in chat is no evidence, nor is an unrelated owner command.
#[test]
fn a_player_saying_duel_in_chat_and_other_owner_commands_are_no_evidence() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(3, "/duel x");
        bot.on_chat_line(-1, "/duel x");
        bot.on_chat_line(3, "You have been invited to a fight by 'p3', type '/1vs1 p3' to join");
        bot.on_owner_command("/team 1");
        bot.on_owner_command("/duelx");
        bot.on_owner_command("duel");
        assert!(step_frozen(&mut bot, &mut sc, 300) >= 1, "no evidence: killed as usual");
        assert_eq!(bot.duel(), None);
    });
}

/// `duel_commands` of the settings file replaces the default list.
#[test]
fn the_duel_commands_are_configurable() {
    support::big_stack(|| {
        let (probe, _, _, _) = Probe::new(neutral());
        let mut c = cfg(BrainKind::Planner);
        c.duel_commands = vec!["/fight".into()];
        let mut bot = bot_with(Box::new(probe), c, Relations::new());
        let map = room(&PIT);
        bot.on_map_loaded(Arc::clone(&map));
        let mut sc = Scenario::new(map, vec![pit_tee(), tee(1, 2000)]);
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_owner_command("/duel x");
        assert!(
            step_frozen(&mut bot, &mut sc, 300) >= 1,
            "the default words are replaced"
        );
        bot.on_owner_command("/FIGHT x");
        step_frozen(&mut bot, &mut sc, 3);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
    });
}

/// The detection off: the owner's duel command does nothing.
#[test]
fn with_the_detection_off_the_owner_duel_command_does_nothing() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.set_duel_detect(false);
        bot.on_owner_command("/duel x");
        assert!(step_frozen(&mut bot, &mut sc, 300) >= 1);
        assert_eq!(bot.duel(), None);
    });
}

// ---- review 4.12 round 2 (F6, F7) ----------------------------------------------------------------------------------------------------------

/// F6: one snapshot that does not show the fight team does not end the protection for the rest of the fight.
#[test]
fn a_one_snapshot_glitch_in_the_fight_team_does_not_end_the_protection() {
    support::big_stack(|| {
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000), tee(2, 2200)],
            Relations::new(),
        );
        bot.on_chat_line(-1, INVITE);
        let mut kills = 0;
        for i in 0..400 {
            // The 200th snapshot (tick ~1400) shows a third player with our number, as a rainbow-named player's cycling number can.
            sc.teams = if i == 200 {
                team_of(&[(0, 7), (1, 7), (2, 7)])
            } else {
                team_of(&[(0, 7), (1, 7)])
            };
            kills += step_frozen(&mut bot, &mut sc, 1);
        }
        assert_eq!(kills, 0);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        let events: Vec<_> = bot.drain_events().collect();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, BotEvent::DuelEnded { .. }))
                .count(),
            0,
            "{events:?}"
        );
    });
}

/// F7: a reconnect (the timeout code puts the bot back into its fight team) or a reload of the same map keeps the evidence; another map does not.
#[test]
fn a_reconnect_or_a_reload_of_the_same_map_keeps_the_duel_evidence() {
    use ddai_bot::hooks::MapIdent;
    support::big_stack(|| {
        let ident = |n: &str| MapIdent {
            name: n.to_string(),
            sha256: [4; 32],
        };
        let (mut bot, mut sc) = setup(
            SelfKillPolicy::Legacy,
            room(&PIT),
            vec![pit_tee(), tee(1, 2000)],
            Relations::new(),
        );
        bot.set_map_ident(ident("joni"));
        bot.on_map_loaded(Arc::clone(&sc.map));
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(-1, INVITE);
        step_frozen(&mut bot, &mut sc, 3);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        // The connection drops and comes back: the team is gone until the snapshots show it again, the evidence stays.
        bot.on_disconnected();
        assert_eq!(bot.duel(), None);
        assert_eq!(
            step_frozen(&mut bot, &mut sc, 300),
            0,
            "back in the fight team: protected again with no new chat line"
        );
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        // The same map reloaded (MapChanging, MapLoaded): the same.
        bot.on_map_changing();
        bot.set_map_ident(ident("joni"));
        bot.on_map_loaded(Arc::clone(&sc.map));
        assert_eq!(step_frozen(&mut bot, &mut sc, 300), 0);
        assert_eq!(bot.duel(), Some(DuelWhy::Team));
        // Another map: the fight is not ours any more.
        bot.on_map_changing();
        bot.set_map_ident(ident("other"));
        bot.on_map_loaded(Arc::clone(&sc.map));
        assert!(step_frozen(&mut bot, &mut sc, 300) >= 1);
        assert_eq!(bot.duel(), None);
    });
}

/// F11: the evidence is carried over a reconnect only for a short outage; after a long one the fight is over.
#[test]
fn duel_evidence_is_not_carried_over_a_long_outage() {
    support::big_stack(|| {
        for (outage_ms, kept) in [(0u64, true), (600, false)] {
            let (probe, _, _, _) = Probe::new(neutral());
            let mut c = cfg(BrainKind::Planner);
            c.duel_outage_max = std::time::Duration::from_millis(300);
            let mut bot = bot_with(Box::new(probe), c, Relations::new());
            let map = room(&PIT);
            bot.on_map_loaded(Arc::clone(&map));
            let mut sc = Scenario::new(map, vec![pit_tee(), tee(1, 2000)]);
            sc.teams = team_of(&[(0, 7), (1, 7)]);
            bot.on_chat_line(-1, INVITE);
            step_frozen(&mut bot, &mut sc, 3);
            assert_eq!(bot.duel(), Some(DuelWhy::Team));
            bot.on_disconnected();
            std::thread::sleep(std::time::Duration::from_millis(outage_ms));
            let k = step_frozen(&mut bot, &mut sc, 300);
            if kept {
                assert_eq!(k, 0, "a short outage: still protected");
                assert_eq!(bot.duel(), Some(DuelWhy::Team));
            } else {
                assert!(
                    k >= 1,
                    "a long outage: the evidence is gone, a bare two-player team is no duel"
                );
                assert_eq!(bot.duel(), None);
            }
        }
    });
}
