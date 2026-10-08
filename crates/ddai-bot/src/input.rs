//! Turning a brain's [`Action`] into the wire `PlayerInput` — the port of `applyInput`
//! (`bot.ts:4828-4864`, `docs/research/orig-bot.md` §4.4).
//!
//! - **Fire is a counter.** `m_Fire` on the wire is a press counter whose low bit is "held"
//!   (`controls.cpp:78-89`); the server counts presses and releases between two inputs
//!   (`CountInput`). The brain's `fire` is a level where `true` means *a fresh press*: released
//!   before, it adds 1 (one press); already held, it adds 2 (release then press — the TS
//!   `if m_Fire & 1 { Fire() } Fire()`); `false` while held adds 1 (release); `false` while released
//!   changes nothing. The counter then stays constant in every repeated `NETMSG_INPUT` until the next
//!   decision, so a press is counted exactly once.
//! - **Wanted weapon** is the hammer unless the action names another (`WantedWeapon(input.wantedWeapon
//!   === 0 ? WEAPON_HAMMER + 1 : input.wantedWeapon)`); the wire value is 1-based.
//! - **No `FlagScoreboard` trick.** The TS switched the "scoreboard open" player flag on for one
//!   snapshot every second (`FlagScoreboard(tick % 50 < 2)`, `bot.ts:4843`). The code does not say
//!   why; it reads as faking a human's activity to the server's AFK detection, so it is deliberately
//!   **not** ported (task spec 4; `player_flags` is always just `PLAYING`).

use ddai_brain::Action;
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects::PlayerInput;
use ddai_physics::core::WEAPON_HAMMER;
use ddai_world::player_input_to_net;

/// What encoding one action did, for the statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodeInfo {
    /// The hook went from released to pressed.
    pub hook_rising: bool,
    /// A fresh fire press was sent.
    pub fire_pressed: bool,
    /// The jump key went from released to pressed (task 3.19: the duel journal counts jump presses).
    pub jump_rising: bool,
}

/// The encoder's memory: what we last sent.
#[derive(Debug, Clone, Copy)]
pub struct InputEncoder {
    prev_fire: i32,
    prev_hook: bool,
    prev_jump: bool,
}

impl Default for InputEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// The neutral input: standing, aim straight up (never `(0, 0)`), nothing held, the hammer wanted.
pub fn neutral_input(fire: i32) -> PlayerInput {
    PlayerInput {
        direction: 0,
        target_x: 0,
        target_y: -1,
        jump: 0,
        fire,
        hook: 0,
        player_flags: playerflagflag::PLAYING,
        wanted_weapon: WEAPON_HAMMER + 1,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// The fire counter after `action_fire` given the counter we last sent (see the module docs). The
/// driver applies the same rule to the counter actually on the wire when it adopts a decision
/// (`ddai_client::next_fire_counter`), so a decision that was replaced before it went out cannot leak
/// its press into the next one (task 4.1b, review F2c).
pub fn next_fire_counter(prev: i32, action_fire: bool) -> i32 {
    ddai_client::next_fire_counter(prev, action_fire)
}

impl InputEncoder {
    pub fn new() -> Self {
        InputEncoder {
            prev_fire: 0,
            prev_hook: false,
            prev_jump: false,
        }
    }

    /// Forgets the hook edge (the fire counter must continue: the server remembers it).
    pub fn reset_edges(&mut self) {
        self.prev_hook = false;
        self.prev_jump = false;
    }

    /// The counter last sent.
    pub fn fire_counter(&self) -> i32 {
        self.prev_fire
    }

    /// Encodes `action`, advancing the counter.
    pub fn encode(&mut self, action: &Action) -> (PlayerInput, EncodeInfo) {
        let mut input = player_input_to_net(action.to_player_input());
        input.player_flags = playerflagflag::PLAYING;
        input.wanted_weapon = action.wanted_weapon.unwrap_or(WEAPON_HAMMER) + 1;
        let fire = next_fire_counter(self.prev_fire, action.fire);
        input.fire = fire;
        let info = EncodeInfo {
            hook_rising: action.hook && !self.prev_hook,
            fire_pressed: action.fire,
            jump_rising: action.jump && !self.prev_jump,
        };
        self.prev_fire = fire;
        self.prev_hook = action.hook;
        self.prev_jump = action.jump;
        (input, info)
    }

    /// `idle()` (`bot.ts:4973-4986`): nothing held, fire released if it was held.
    pub fn idle(&mut self) -> PlayerInput {
        let fire = next_fire_counter(self.prev_fire, false);
        self.prev_fire = fire;
        self.prev_hook = false;
        self.prev_jump = false;
        neutral_input(fire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_brain::IVec2;

    fn fire(f: bool) -> Action {
        Action {
            fire: f,
            ..Action::neutral()
        }
    }

    #[test]
    fn a_fire_level_becomes_the_press_counter_with_press_and_release_semantics() {
        let mut e = InputEncoder::new();
        // released -> press: +1 (now held)
        assert_eq!(e.encode(&fire(true)).0.fire, 1);
        // held, wants a fresh press: release + press = +2 (still held)
        assert_eq!(e.encode(&fire(true)).0.fire, 3);
        assert_eq!(e.encode(&fire(true)).0.fire, 5);
        // held, wants released: +1 (released, even)
        assert_eq!(e.encode(&fire(false)).0.fire, 6);
        // released, stays released: unchanged
        assert_eq!(e.encode(&fire(false)).0.fire, 6);
        assert_eq!(e.encode(&fire(false)).0.fire, 6);
        // a press after a release: +1
        assert_eq!(e.encode(&fire(true)).0.fire, 7);
        assert_eq!(e.fire_counter(), 7);
    }

    #[test]
    fn the_counter_matches_the_servers_count_input_walk() {
        // `CountInput(prev, cur)`: walking prev -> cur, odd steps are presses, even releases.
        let presses = |prev: i32, cur: i32| (prev + 1..=cur).filter(|n| n & 1 == 1).count();
        let mut e = InputEncoder::new();
        let mut prev = 0;
        for want in [true, true, false, true, false, false, true] {
            let cur = e.encode(&fire(want)).0.fire;
            let pressed = presses(prev, cur);
            assert_eq!(pressed, usize::from(want), "want={want} prev={prev} cur={cur}");
            prev = cur;
        }
    }

    #[test]
    fn idle_releases_a_held_trigger_and_is_otherwise_neutral() {
        let mut e = InputEncoder::new();
        e.encode(&fire(true));
        let idle = e.idle();
        assert_eq!(idle.fire, 2, "released");
        assert_eq!((idle.direction, idle.jump, idle.hook), (0, 0, 0));
        assert_eq!((idle.target_x, idle.target_y), (0, -1));
        assert_eq!(idle.wanted_weapon, 1);
        assert_eq!(e.idle().fire, 2, "already released: unchanged");
    }

    #[test]
    fn levels_aim_weapon_and_flags_are_encoded() {
        let mut e = InputEncoder::new();
        let a = Action {
            direction: -1,
            jump: true,
            hook: true,
            fire: false,
            target: IVec2::new(120, -45),
            wanted_weapon: None,
        };
        let (i, info) = e.encode(&a);
        assert_eq!(
            (i.direction, i.jump, i.hook, i.target_x, i.target_y, i.fire),
            (-1, 1, 1, 120, -45, 0)
        );
        assert_eq!(i.wanted_weapon, WEAPON_HAMMER + 1, "hammer by default");
        assert_eq!(i.player_flags, playerflagflag::PLAYING, "no scoreboard flag, ever");
        assert!(info.hook_rising && !info.fire_pressed);
        let (_, info2) = e.encode(&a);
        assert!(!info2.hook_rising, "still held: not a new press");
        let gun = Action {
            wanted_weapon: Some(1),
            target: IVec2::new(0, 0),
            ..Action::neutral()
        };
        let (g, _) = e.encode(&gun);
        assert_eq!(g.wanted_weapon, 2, "gun = slot 1, 1-based on the wire");
        assert_eq!((g.target_x, g.target_y), (0, -1), "never a zero aim");
    }

    #[test]
    fn the_scoreboard_flag_is_never_sent_on_any_tick() {
        let mut e = InputEncoder::new();
        for tick in 0..200 {
            let (i, _) = e.encode(&Action::neutral());
            assert_eq!(i.player_flags, playerflagflag::PLAYING, "tick {tick}");
            assert_eq!(i.player_flags & playerflagflag::SCOREBOARD, 0);
        }
    }

    /// The arena's `wire_from_action` (`ddai-env`) is the same convention; pinned here so the two
    /// cannot drift (the planner's own counter and ours must agree on what a press is).
    #[test]
    fn matches_the_arenas_convention() {
        for prev in 0..6 {
            for want in [false, true] {
                let arena = {
                    let held = (prev & 1) != 0;
                    match (want, held) {
                        (true, true) => prev + 2,
                        (true, false) => prev + 1,
                        (false, true) => prev + 1,
                        (false, false) => prev,
                    }
                };
                assert_eq!(next_fire_counter(prev, want), arena);
            }
        }
    }
}
