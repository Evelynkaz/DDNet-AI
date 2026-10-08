//! Task 3.19 (D-116, acceptance 5a): a detected F-DDrace duel reaches the navigator (`Navigator::set_duel`), which stops holding the wayblock
//! (`WayBlock::holding(.., duel)`, hard-coded `false` before); the end of the duel, and another map, take it back.

mod support;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use ddai_bot::hooks::{Hooks, Navigator};
use ddai_bot::{Bot, BrainKind, Relations};
use ddai_brain::Action;
use ddai_net::tuning::TeamsState;
use support::*;

/// A navigator that only records what it is told about the duel.
struct Spy(Rc<RefCell<Vec<bool>>>);

impl Navigator for Spy {
    fn set_duel(&mut self, on: bool) {
        self.0.borrow_mut().push(on);
    }
}

const INVITE: &str = "You have been invited to a fight by 'p1', type '/1vs1 p0' to join";

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

#[test]
fn a_duel_reaches_the_navigator_and_so_does_its_end() {
    big_stack(|| {
        let map = room(&[]);
        let (probe, _, _, _) = Probe::new(Action::neutral());
        let seen = Rc::new(RefCell::new(Vec::new()));
        let hooks = Hooks {
            navigator: Box::new(Spy(Rc::clone(&seen))),
            ..Hooks::default()
        };
        let mut bot = Bot::new(cfg(BrainKind::Planner), Box::new(probe), hooks, Relations::new());
        bot.on_map_loaded(Arc::clone(&map));
        let mut sc = Scenario::new(map, vec![tee(0, 600), tee(1, 700)]);
        // Before: no team, no duel.
        run(&mut bot, &mut sc, 4);
        assert!(seen.borrow().iter().all(|&on| !on), "{:?}", seen.borrow());
        // The duel: an invitation and a two-player team.
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(-1, INVITE);
        run(&mut bot, &mut sc, 6);
        assert!(bot.duel().is_some());
        assert_eq!(
            seen.borrow().last(),
            Some(&true),
            "the navigator was told: {:?}",
            seen.borrow()
        );
        // The team is gone: the duel is over.
        sc.teams = team_of(&[]);
        run(&mut bot, &mut sc, 400);
        assert!(bot.duel().is_none());
        assert_eq!(
            seen.borrow().last(),
            Some(&false),
            "and told it is over: {:?}",
            seen.borrow()
        );
    });
}

/// Acceptance 7 (the live protocol's journal): a detected duel logs a window every 30 s with the hammer and jump presses.
#[test]
fn a_duel_logs_a_window_every_thirty_seconds() {
    big_stack(|| {
        let map = room(&[]);
        let (probe, _, _, _) = Probe::new(Action {
            jump: true,
            fire: true,
            ..Action::neutral()
        });
        let mut bot = Bot::new(
            cfg(BrainKind::Planner),
            Box::new(probe),
            Hooks::default(),
            Relations::new(),
        );
        bot.on_map_loaded(Arc::clone(&map));
        let mut sc = Scenario::new(map, vec![tee(0, 600), tee(1, 700)]);
        sc.teams = team_of(&[(0, 7), (1, 7)]);
        bot.on_chat_line(-1, INVITE);
        run(&mut bot, &mut sc, 1700); // 1 700 snapshots = 3 400 ticks
        let windows: Vec<_> = bot
            .drain_events()
            .filter_map(|e| match e {
                ddai_bot::BotEvent::DuelWindow {
                    ticks, hammer_presses, ..
                } => Some((ticks, hammer_presses)),
                _ => None,
            })
            .collect();
        assert!(windows.len() >= 2, "{windows:?}");
        assert!(windows.iter().all(|&(t, _)| t >= 1500), "{windows:?}");
        assert!(windows[0].1 > 0, "the first window counts the presses: {windows:?}");
    });
}
