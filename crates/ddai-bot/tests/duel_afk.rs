//! Task 3.23 (D-121, `--duel-fixes static`): in a detected F-DDrace duel the opponent who stands idle is still the target, however far he is.
//! The post-mortem of the 2026-10-08 duel against a human: the AFK filter of the target selection drops an idle tee that is farther than 96 px (it only
//! keeps one it is already fighting that is next to us), so a human who walked away from his keyboard in the box would have left the bot with no target.

mod support;

use std::sync::Arc;

use ddai_bot::{Bot, BotConfig, BrainKind, Relations};
use ddai_brain::Action;
use ddai_net::tuning::TeamsState;
use support::*;

const INVITE: &str = "You have been invited to a fight by 'p1', type '/1vs1 p0' to join";

fn in_team(a: usize, b: usize) -> Option<TeamsState> {
    let mut t = TeamsState {
        teams: [0; 128],
        received: 128,
    };
    t.teams[a] = 7;
    t.teams[b] = 7;
    Some(t)
}

/// Plays `snapshots` snapshots of an idle opponent 500 px away; returns the bot's target at the end.
fn target_after(duel: bool, duel_afk: bool, snapshots: usize) -> i32 {
    let map = room(&[]);
    let (probe, _, _, _) = Probe::new(Action::neutral());
    let cfg = BotConfig {
        duel_afk,
        ..cfg(BrainKind::Planner)
    };
    let mut bot = Bot::new(
        cfg,
        Box::new(probe),
        ddai_bot::hooks::Hooks::default(),
        Relations::new(),
    );
    bot.on_map_loaded(Arc::clone(&map));
    let mut sc = Scenario::new(map, vec![tee(0, 600), tee(1, 1100)]);
    if duel {
        sc.teams = in_team(0, 1);
        bot.on_chat_line(-1, INVITE);
    }
    run(&mut bot, &mut sc, snapshots);
    assert_eq!(bot.duel().is_some(), duel, "the duel detector");
    bot.target_id()
}

#[test]
fn the_duel_opponent_who_stands_idle_stays_the_target_only_with_the_fix() {
    big_stack(|| {
        // 5 s in: not idle long enough for the filter, he is the target either way.
        assert_eq!(target_after(true, false, 50), 1);
        // 30 s idle: the filter has dropped him ...
        assert_eq!(
            target_after(true, false, 800),
            -1,
            "without the fix the idle opponent is dropped"
        );
        // ... unless the fix is on and a duel is detected.
        assert_eq!(target_after(true, true, 800), 1, "with the fix he stays the target");
        // Outside a duel the fix does nothing: an idle tee is skipped as before.
        assert_eq!(target_after(false, true, 800), -1, "no duel, no exemption");
    });
}
