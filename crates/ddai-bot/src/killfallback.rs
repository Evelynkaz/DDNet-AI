//! The `/kill` fallback (task 4.6, D-078): when DDNet drops the protocol `Cl_Kill`, the bot sends the one chat-channel message
//! it is allowed, the server command `/kill`.
//!
//! **Why.** DDNet 20.1 (`gamecontext.cpp:2977`) refuses a `Cl_Kill` silently when `sv_kill_protection` is not 0, the race has run
//! that many minutes (default 20) and the life is started; the only trace is a system chat line "Kill Protection enabled. ...".
//! In the 4.5 rehearsal the bot sat frozen for 93 minutes, asking every 10 s. `/kill` (`ConProtectedKill`) is not subject to it.
//!
//! **When.** Only after the bot itself decided to kill (unstick, navigation, wayblock, console or web `!kill`: all of them end in
//! `Output::kill`, the protocol `Cl_Kill`) and
//! - no death and no new life followed within [`EFFECT_WAIT_TICKS`], or
//! - the server's notice "Kill Protection enabled" was seen for this life (it comes at once: no waiting).
//!
//! **How often.** One `/kill` per decision, and at least [`KILL_COOLDOWN_TICKS`] between two `/kill`s; at most
//! [`MAX_COMMANDS_PER_LIFE`] without a death, then the bot stops asking for this life and says so (a server that ignores `/kill`
//! too must not be spammed). A death or a new life resets everything; so do a map change and a reconnect.
//!
//! **What it never does.** It does not build text: [`ddai_net::server_command::ServerCommand::Kill`] is the only thing that can be
//! sent, and the outgoing allow-list checks the bytes. The notice is read, never stored or logged (a system line, not a player's);
//! nothing else in chat is looked at.

use crate::consts::KILL_COOLDOWN_TICKS;

/// Ticks after a protocol `Cl_Kill` without a death or a new life before it counts as dropped (4.5's liveness gate uses the same).
pub const EFFECT_WAIT_TICKS: i32 = 50;
/// `/kill`s sent in one life without a death, then the bot stops asking for that life.
pub const MAX_COMMANDS_PER_LIFE: u32 = 3;
/// A life tick: DDNet runs 50 ticks a second.
const TICKS_PER_MINUTE: f32 = 3000.0;
const NEVER: i32 = i32::MIN / 2;

/// The beginning of the server's system line sent when it drops a `Cl_Kill` ("Kill Protection enabled. If you really want to kill,
/// type /kill").
const NOTICE_PREFIX: &str = "Kill Protection enabled";

/// Whether a `Sv_Chat` is the server's kill-protection notice: a system line (`client_id` -1), never a player's.
pub fn is_kill_protection_notice(client_id: i32, text: &str) -> bool {
    client_id == -1 && text.starts_with(NOTICE_PREFIX)
}

/// What a new life taught about the server's threshold (only after a `/kill` was needed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Learned {
    /// How long the life had lasted when the `/kill` was sent, in minutes.
    pub life_minutes: f32,
}

#[derive(Debug)]
pub struct KillFallback {
    /// The tick of a protocol `Cl_Kill` that has shown no effect yet.
    awaiting: Option<i32>,
    /// A `/kill` was already sent for the awaited decision (one per decision).
    sent_for: bool,
    /// The server's notice was seen in this life.
    noticed: bool,
    last_command: i32,
    commands_this_life: u32,
    gave_up: bool,
    life_start: Option<i32>,
    /// The tick of a `/kill` whose effect (a new life) has not been seen yet: for the threshold.
    command_tick: Option<i32>,
    /// The first tick our tee was seen gone after that `/kill` (a death): the effect has to follow the `/kill` closely to count.
    dead_tick: Option<i32>,
}

impl Default for KillFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl KillFallback {
    pub fn new() -> Self {
        KillFallback {
            awaiting: None,
            sent_for: false,
            noticed: false,
            last_command: NEVER,
            commands_this_life: 0,
            gave_up: false,
            life_start: None,
            command_tick: None,
            dead_tick: None,
        }
    }

    /// A map change, a reconnect: nothing carries over (ticks restart, the life is gone).
    pub fn reset(&mut self) {
        *self = KillFallback::new();
    }

    /// The server's notice was seen.
    pub fn on_notice(&mut self) {
        self.noticed = true;
    }

    /// Whether the notice was seen in this life.
    pub fn noticed(&self) -> bool {
        self.noticed
    }

    /// The bot sent a protocol `Cl_Kill` at `tick`: its effect is awaited, and one `/kill` may follow for this decision.
    pub fn on_protocol_kill(&mut self, tick: i32) {
        self.awaiting = Some(tick);
        self.sent_for = false;
    }

    /// Our tee is gone (dead) at `tick`: whatever kill was awaited has taken effect. The first such tick after a `/kill` is
    /// kept: only a death that follows the `/kill` closely says the `/kill` did it.
    pub fn on_dead(&mut self, tick: i32) {
        self.awaiting = None;
        self.sent_for = false;
        if self.command_tick.is_some() && self.dead_tick.is_none() {
            self.dead_tick = Some(tick);
        }
    }

    /// A new life started at `tick`. Returns what a `/kill` taught, when one ended the previous life: the death (or this new
    /// life) came within [`EFFECT_WAIT_TICKS`] of the `/kill`. `/kill` acts only under kill protection, so one that did nothing
    /// (the life ended minutes later some other way) says the opposite and teaches nothing.
    pub fn on_life_started(&mut self, tick: i32) -> Option<Learned> {
        let effect = self.dead_tick.take().unwrap_or(tick);
        let learned = match (self.command_tick.take(), self.life_start) {
            (Some(cmd), Some(start)) if cmd >= start && (0..=EFFECT_WAIT_TICKS).contains(&(effect - cmd)) => {
                Some(Learned {
                    life_minutes: (cmd - start) as f32 / TICKS_PER_MINUTE,
                })
            }
            _ => None,
        };
        self.awaiting = None;
        self.sent_for = false;
        self.noticed = false;
        self.commands_this_life = 0;
        self.gave_up = false;
        self.life_start = Some(tick);
        learned
    }

    /// Asked once per snapshot: `true` means send `/kill` now.
    pub fn poll(&mut self, tick: i32) -> bool {
        let Some(asked) = self.awaiting else { return false };
        if self.sent_for || self.gave_up {
            return false;
        }
        if !(self.noticed || tick - asked >= EFFECT_WAIT_TICKS) {
            return false;
        }
        if tick - self.last_command < KILL_COOLDOWN_TICKS {
            return false;
        }
        if self.commands_this_life >= MAX_COMMANDS_PER_LIFE {
            self.gave_up = true;
            return false;
        }
        self.last_command = tick;
        self.sent_for = true;
        self.commands_this_life += 1;
        self.command_tick = Some(tick);
        self.dead_tick = None;
        true
    }

    /// The owner's own line starting with `/` was sent (task 4.9b: a server command typed on the website, `/kill` among them). The bot
    /// did not decide it, so whatever it does to the life must not be credited to the bot's `/kill` that may still be in flight: that
    /// `/kill` is forgotten as a candidate for the threshold ([`KillFallback::on_life_started`] then learns nothing). Nothing else
    /// changes: the owner's lines never count as a `/kill` of the fallback (no cooldown, no per-life count, no new awaited kill).
    pub fn on_owner_command(&mut self) {
        self.command_tick = None;
        self.dead_tick = None;
    }

    /// The server paused the bot (task 4.9b): whatever kill was awaited, and any `/kill` in flight, is forgotten. Cooldown and the
    /// per-life count stay (a pause is no reason to lift them).
    pub fn cancel_pending(&mut self) {
        self.awaiting = None;
        self.sent_for = false;
        self.command_tick = None;
        self.dead_tick = None;
    }

    /// The bot stopped asking for this life (too many `/kill`s without a death).
    pub fn gave_up(&self) -> bool {
        self.gave_up
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polled(f: &mut KillFallback, from: i32, to: i32) -> Vec<i32> {
        (from..=to).filter(|&t| f.poll(t)).collect()
    }

    #[test]
    fn nothing_is_sent_without_a_protocol_kill() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        assert!(polled(&mut f, 0, 5000).is_empty());
        f.on_notice();
        assert!(polled(&mut f, 0, 5000).is_empty(), "the notice alone sends nothing");
    }

    #[test]
    fn a_kill_without_effect_gets_exactly_one_slash_kill_after_the_wait() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(1000);
        assert_eq!(
            polled(&mut f, 1000, 2000),
            vec![1000 + EFFECT_WAIT_TICKS],
            "one /kill, on the 50th tick"
        );
    }

    #[test]
    fn a_death_or_a_new_life_within_the_wait_cancels_it() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(1000);
        assert!(polled(&mut f, 1000, 1030).is_empty());
        f.on_dead(1031);
        assert!(polled(&mut f, 1031, 3000).is_empty());
        f.on_protocol_kill(2000);
        f.on_life_started(2020);
        assert!(polled(&mut f, 2020, 4000).is_empty());
    }

    #[test]
    fn the_notice_sends_at_once_without_waiting() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(1000);
        assert!(!f.poll(1000));
        f.on_notice(); // the server says so right after dropping the kill
        assert!(f.poll(1001));
        assert!(!f.poll(1002), "one per decision");
        // a notice seen earlier in this life: the next decision's /kill is immediate
        f.on_protocol_kill(1500);
        assert!(f.poll(1501));
    }

    #[test]
    fn the_cooldown_between_two_slash_kills_holds() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_notice();
        f.on_protocol_kill(1000);
        assert!(f.poll(1000));
        // a second decision soon after: no /kill inside the cooldown
        f.on_protocol_kill(1100);
        assert!(polled(&mut f, 1100, 1000 + KILL_COOLDOWN_TICKS - 1).is_empty());
        assert!(f.poll(1000 + KILL_COOLDOWN_TICKS));
    }

    #[test]
    fn it_gives_up_for_the_life_after_three_without_a_death() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_notice();
        let mut sent = 0;
        let mut t = 1000;
        for _ in 0..10 {
            f.on_protocol_kill(t);
            if f.poll(t) {
                sent += 1;
            }
            t += KILL_COOLDOWN_TICKS;
        }
        assert_eq!(sent, MAX_COMMANDS_PER_LIFE);
        assert!(f.gave_up());
        // a new life starts afresh
        f.on_life_started(t);
        f.on_notice();
        f.on_protocol_kill(t + 1);
        assert!(f.poll(t + 1));
        assert!(!f.gave_up());
    }

    #[test]
    fn a_reset_forgets_everything() {
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_notice();
        f.on_protocol_kill(1000);
        assert!(f.poll(1000));
        f.reset();
        assert!(!f.noticed());
        f.on_protocol_kill(5); // ticks restart after a reconnect: the cooldown does not carry over
        assert_eq!(polled(&mut f, 5, 100), vec![5 + EFFECT_WAIT_TICKS]);
    }

    #[test]
    fn a_slash_kill_that_ended_the_life_teaches_the_threshold() {
        let mut f = KillFallback::new();
        f.on_life_started(1000);
        f.on_protocol_kill(1000 + 3000 * 21);
        assert!(f.poll(1000 + 3000 * 21 + EFFECT_WAIT_TICKS));
        let learned = f.on_life_started(1000 + 3000 * 21 + 60).expect("a /kill ended it");
        assert!((learned.life_minutes - 21.0).abs() < 0.05, "{learned:?}");
        // an ordinary death teaches nothing
        f.on_protocol_kill(5000);
        f.on_dead(5002);
        assert_eq!(f.on_life_started(5100), None);
    }

    #[test]
    fn a_slash_kill_that_did_nothing_teaches_nothing_even_if_the_life_ends_later() {
        // The reviewer's sequence: a /kill at 1050 that did nothing, the life ends minutes later some other way.
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(1000);
        assert!(f.poll(1050));
        f.on_dead(13_040);
        assert_eq!(f.on_life_started(13_050), None);
        // The successful path passes through `on_dead` before the new life and still teaches.
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(3000 * 21);
        assert!(f.poll(3000 * 21 + EFFECT_WAIT_TICKS));
        f.on_dead(3000 * 21 + EFFECT_WAIT_TICKS + 2);
        let learned = f
            .on_life_started(3000 * 21 + EFFECT_WAIT_TICKS + 4)
            .expect("the /kill did it");
        assert!((learned.life_minutes - 21.0).abs() < 0.05, "{learned:?}");
        // A new life without a death seen in between, right after the /kill, counts as well.
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(500);
        assert!(f.poll(550));
        assert!(f.on_life_started(552).is_some());
    }

    /// 4.9b: the owner may type `/kill` on the site. That line is the owner's (counted as `Cl_Say(owner)`, never the fallback), so a life it
    /// ends is not something the bot learned the kill-protection threshold from, even while a fallback `/kill` of the bot's own is in
    /// flight.
    #[test]
    fn an_owner_typed_command_is_never_learned_as_a_threshold() {
        let life_start = 1000;
        let dropped_kill = life_start + 3000 * 21;
        // (a) the owner's /kill alone: no fallback `/kill` was sent, so nothing can be learned
        let mut f = KillFallback::new();
        f.on_life_started(life_start);
        f.on_owner_command();
        f.on_dead(dropped_kill);
        assert_eq!(f.on_life_started(dropped_kill + 2), None);
        assert!(!f.poll(dropped_kill + 100), "and it does not make the bot send one");
        // (b) the bot's own `/kill` is still in flight (no effect yet) when the owner's `/kill` ends the life within the effect window:
        // without the owner's line this would be credited to the fallback, with it nothing is learned
        for owner_types in [false, true] {
            let mut f = KillFallback::new();
            f.on_life_started(life_start);
            f.on_protocol_kill(dropped_kill);
            assert!(f.poll(dropped_kill + EFFECT_WAIT_TICKS), "the fallback's /kill");
            if owner_types {
                f.on_owner_command();
            }
            f.on_dead(dropped_kill + EFFECT_WAIT_TICKS + 10);
            let learned = f.on_life_started(dropped_kill + EFFECT_WAIT_TICKS + 12);
            assert_eq!(
                learned.is_some(),
                !owner_types,
                "owner_types {owner_types}: {learned:?}"
            );
        }
        // (c) the owner's line does not disturb the fallback's own accounting: its cooldown, its count and its awaited kill stay
        let mut f = KillFallback::new();
        f.on_life_started(0);
        f.on_protocol_kill(1000);
        f.on_owner_command();
        assert_eq!(
            polled(&mut f, 1000, 2000),
            vec![1000 + EFFECT_WAIT_TICKS],
            "still exactly one /kill"
        );
        f.on_owner_command();
        f.on_protocol_kill(1600);
        assert_eq!(
            polled(&mut f, 1600, 3000),
            vec![1600 + EFFECT_WAIT_TICKS],
            "the cooldown held"
        );
    }

    #[test]
    fn only_the_systems_notice_counts() {
        assert!(is_kill_protection_notice(
            -1,
            "Kill Protection enabled. If you really want to kill, type /kill"
        ));
        assert!(!is_kill_protection_notice(
            3,
            "Kill Protection enabled. If you really want to kill, type /kill"
        ));
        assert!(!is_kill_protection_notice(-1, "kill protection enabled"));
        assert!(!is_kill_protection_notice(-1, "hello"));
        assert!(!is_kill_protection_notice(-1, ""));
        // 4.9b: the server's answers to the owner's commands are system lines too, and none of them is the notice
        for reply in [
            "Emote commands are: /emote surprise /emote blink",
            "DDraceNetwork Mod. Version: 20.1",
            "Unknown command: nosuch",
            "You are force-paused for 30 seconds.",
            "Kicked (spam)",
            "You have been banned",
        ] {
            assert!(!is_kill_protection_notice(-1, reply), "{reply}");
        }
    }
}
