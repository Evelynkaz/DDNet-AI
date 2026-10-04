//! The owner's chat lines (task 4.9, D-094): pacing, the queue and the answers for a line the owner typed on the website.
//!
//! **What this is not.** Nothing here makes text. A line arrives only as an [`OwnerSay`] (a validated [`ddai_net::owner_chat::OwnerText`])
//! that the web control channel put on the command bus, and leaves only through `Client::owner_say`. No auto-reply, no timer, no
//! reading of the game chat. The text is held in the queue and nowhere else: the answers, the log lines and the counters carry
//! lengths and reasons, never the line.
//!
//! **Limits (the bot's own, well below the server's).**
//! - at least [`MIN_GAP`] (3 s) between two lines, and longer after a long one: DDNet drops a chat line when less than
//!   `(31 + length) / 32` seconds passed since the last one (`gamecontext.cpp`, `OnSayNetMessage`: "more than 32 characters per second"),
//!   so the gap is at least that plus a second;
//! - at most [`MAX_PER_MINUTE`] (10) lines in any 60 s;
//! - a model of the server's own spam score (`sv_chat_penalty` 250 per line, one point less per tick, mute above
//!   `sv_chat_threshold` 1000 for a minute, `ProcessSpamProtection`): the bot sends only while the score after the line stays at
//!   [`SCORE_CEILING`] or below, one penalty under the mute threshold (a server with other settings is not modelled);
//! - a queue of at most [`QUEUE_CAP`] (3) lines waiting for their turn; one more is refused, and so is a line that would have to wait
//!   longer than [`MAX_WAIT`] (the owner is told when to try again instead of finding a stale line said later);
//! - nothing while the bot is not in the game: such a line is refused, and lines still queued when the bot leaves the game are dropped
//!   (never said after a reconnect).
//!
//! Time is a [`Duration`] on a monotonic clock the caller owns (the runner's `Instant`), so every rule is testable without sleeping.

use std::collections::VecDeque;
use std::time::Duration;

use ddai_net::owner_chat::OwnerSay;

use crate::command::CommandReply;

/// The least time between two lines.
pub const MIN_GAP: Duration = Duration::from_secs(3);
/// Lines in any [`WINDOW`].
pub const MAX_PER_MINUTE: usize = 10;
/// The window of [`MAX_PER_MINUTE`].
pub const WINDOW: Duration = Duration::from_secs(60);
/// Lines that may wait for their turn.
pub const QUEUE_CAP: usize = 3;
/// A line that would wait longer than this is refused when it arrives.
pub const MAX_WAIT: Duration = Duration::from_secs(30);
/// What the server adds to a player's chat score per line (`sv_chat_penalty`).
const SERVER_CHAT_PENALTY: f64 = 250.0;
/// What the server takes off per second (one per tick, 50 ticks).
const SERVER_DECAY_PER_SEC: f64 = 50.0;
/// The score the server mutes above (`sv_chat_threshold`).
const SERVER_MUTE_THRESHOLD: f64 = 1000.0;
/// The score the bot allows itself to reach by sending: one penalty below the mute threshold.
pub const SCORE_CEILING: f64 = SERVER_MUTE_THRESHOLD - SERVER_CHAT_PENALTY;

/// The shortest wait after the previous line before a line of `chars` code points: the bot's own gap, or the server's
/// "32 characters a second" rule plus a second.
fn gap_for(chars: usize) -> Duration {
    let server_secs = u64::try_from(chars.div_ceil(32)).unwrap_or(u64::MAX / 2);
    MIN_GAP.max(Duration::from_secs(server_secs.saturating_add(1)))
}

/// What the bot knows of its own recent lines, enough to say when the next may go.
#[derive(Debug, Clone)]
struct Pace {
    last_at: Option<Duration>,
    /// The send times of the last [`MAX_PER_MINUTE`] lines, oldest first.
    sent: VecDeque<Duration>,
    /// The model of the server's chat score as of `score_at`.
    score: f64,
    score_at: Duration,
}

impl Pace {
    fn new() -> Pace {
        Pace {
            last_at: None,
            sent: VecDeque::with_capacity(MAX_PER_MINUTE),
            score: 0.0,
            score_at: Duration::ZERO,
        }
    }

    fn score_at(&self, t: Duration) -> f64 {
        let dt = t.saturating_sub(self.score_at).as_secs_f64();
        (self.score - SERVER_DECAY_PER_SEC * dt).max(0.0)
    }

    /// The first moment at or after `now` when a line of `chars` code points may be sent.
    fn earliest(&self, now: Duration, chars: usize) -> Duration {
        let mut t = now;
        if let Some(last) = self.last_at {
            t = t.max(last + gap_for(chars));
        }
        if self.sent.len() >= MAX_PER_MINUTE {
            t = t.max(self.sent[self.sent.len() - MAX_PER_MINUTE] + WINDOW);
        }
        let over = self.score_at(t) + SERVER_CHAT_PENALTY - SCORE_CEILING;
        if over > 1e-6 {
            // Rounded up to the millisecond so that asking again at the returned time agrees.
            let wait_ms = (over / SERVER_DECAY_PER_SEC * 1000.0).ceil() + 1.0;
            t += Duration::from_secs_f64(wait_ms / 1000.0);
        }
        t
    }

    fn record(&mut self, at: Duration) {
        self.score = self.score_at(at) + SERVER_CHAT_PENALTY;
        self.score_at = at;
        self.last_at = Some(at);
        if self.sent.len() >= MAX_PER_MINUTE {
            self.sent.pop_front();
        }
        self.sent.push_back(at);
    }
}

/// Why a line was refused when it arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The bot is not in the game (still joining, disconnected): nothing is queued for later.
    NotInGame,
    /// [`QUEUE_CAP`] lines are already waiting.
    QueueFull,
    /// The limits ([`MIN_GAP`], [`MAX_PER_MINUTE`], the server's spam score) would make it wait longer than [`MAX_WAIT`].
    TooSoon { retry_in: Duration },
}

impl Refusal {
    /// A short fixed code for the answer's `data.reason` (the web maps it to a status).
    pub fn code(self) -> &'static str {
        match self {
            Refusal::NotInGame => "not_in_game",
            Refusal::QueueFull => "queue_full",
            Refusal::TooSoon { .. } => "rate_limited",
        }
    }
}

/// A line that was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accepted {
    /// Lines that will be said before it.
    pub ahead: usize,
    /// About how long until it is said (zero: on the next poll).
    pub wait: Duration,
}

/// What the owner chat did, for the run report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OwnerChatStats {
    pub accepted: u64,
    pub sent: u64,
    pub refused: u64,
    /// Lines taken but dropped before they were said: the bot left the game first.
    pub dropped: u64,
}

/// The queue and the pacing of the owner's lines.
#[derive(Debug)]
pub struct OwnerChat {
    queue: VecDeque<OwnerSay>,
    pace: Pace,
    stats: OwnerChatStats,
}

impl Default for OwnerChat {
    fn default() -> Self {
        Self::new()
    }
}

impl OwnerChat {
    pub fn new() -> OwnerChat {
        OwnerChat {
            queue: VecDeque::with_capacity(QUEUE_CAP),
            pace: Pace::new(),
            stats: OwnerChatStats::default(),
        }
    }

    pub fn stats(&self) -> OwnerChatStats {
        self.stats
    }

    /// Lines waiting for their turn.
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Takes a line, or says why not. Does not send anything: [`OwnerChat::poll`] does.
    pub fn submit(&mut self, say: OwnerSay, now: Duration, in_game: bool) -> Result<Accepted, Refusal> {
        let verdict = self.verdict(&say, now, in_game);
        match verdict {
            Ok(accepted) => {
                self.stats.accepted += 1;
                self.queue.push_back(say);
                Ok(accepted)
            }
            Err(refusal) => {
                self.stats.refused += 1;
                Err(refusal)
            }
        }
    }

    fn verdict(&mut self, say: &OwnerSay, now: Duration, in_game: bool) -> Result<Accepted, Refusal> {
        if !in_game {
            self.drop_queue();
            return Err(Refusal::NotInGame);
        }
        if self.queue.len() >= QUEUE_CAP {
            return Err(Refusal::QueueFull);
        }
        // When would it go, behind what is already waiting?
        let mut pace = self.pace.clone();
        for waiting in &self.queue {
            let chars = waiting.text.char_count();
            let at = pace.earliest(now, chars);
            pace.record(at);
        }
        let at = pace.earliest(now, say.text.char_count());
        let wait = at.saturating_sub(now);
        if wait > MAX_WAIT {
            return Err(Refusal::TooSoon {
                retry_in: wait - MAX_WAIT,
            });
        }
        Ok(Accepted {
            ahead: self.queue.len(),
            wait,
        })
    }

    /// The next line that may be said now, if any: at most one per call, and only in the game. Out of the game the queue is dropped
    /// (a line is never said after a reconnect).
    pub fn poll(&mut self, now: Duration, in_game: bool) -> Option<OwnerSay> {
        if !in_game {
            self.drop_queue();
            return None;
        }
        let chars = self.queue.front()?.text.char_count();
        if self.pace.earliest(now, chars) > now {
            return None;
        }
        let say = self.queue.pop_front()?;
        self.pace.record(now);
        self.stats.sent += 1;
        Some(say)
    }

    fn drop_queue(&mut self) {
        self.stats.dropped += u64::try_from(self.queue.len()).unwrap_or(u64::MAX);
        self.queue.clear();
    }
}

/// The control channel's answer for a submitted line: `ok` when taken, with the reason in `data.reason` when refused. Never contains
/// the text.
pub fn reply_for(result: &Result<Accepted, Refusal>) -> CommandReply {
    match result {
        Ok(a) if a.ahead == 0 && a.wait.is_zero() => CommandReply::ok("accepted: it is being said now"),
        Ok(a) if a.ahead == 0 => CommandReply::ok(format!(
            "accepted: it will be said in about {} s",
            a.wait.as_secs_f64().ceil()
        )),
        Ok(a) => CommandReply::ok(format!(
            "accepted: {} line(s) ahead, it will be said in about {} s",
            a.ahead,
            a.wait.as_secs_f64().ceil()
        )),
        Err(refusal) => {
            let text = match refusal {
                Refusal::NotInGame => "refused: the bot is not in the game".to_string(),
                Refusal::QueueFull => {
                    format!("refused: {QUEUE_CAP} lines are already waiting, wait for them to be said")
                }
                Refusal::TooSoon { retry_in } => format!(
                    "refused: too many chat lines lately (3 s apart, at most 10 a minute); try again in about {} s",
                    retry_in.as_secs_f64().ceil().max(1.0)
                ),
            };
            let mut reply = CommandReply::err(text);
            reply.data = Some(serde_json::json!({ "reason": refusal.code() }));
            reply
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_net::owner_chat::{OwnerChannel, OwnerText};

    fn say(text: &str) -> OwnerSay {
        OwnerSay::new(false, OwnerText::new(&OwnerChannel::mint_for_tests(), text).unwrap())
    }

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    #[test]
    fn the_first_line_goes_at_once() {
        let mut c = OwnerChat::new();
        let a = c.submit(say("hello"), secs(10.0), true).unwrap();
        assert_eq!(
            a,
            Accepted {
                ahead: 0,
                wait: Duration::ZERO
            }
        );
        let said = c.poll(secs(10.0), true).expect("due");
        assert_eq!(said.text.as_str(), "hello");
        assert!(
            c.poll(secs(10.0), true).is_none(),
            "one line per poll, and the queue is empty"
        );
        assert_eq!(
            c.stats(),
            OwnerChatStats {
                accepted: 1,
                sent: 1,
                refused: 0,
                dropped: 0
            }
        );
    }

    #[test]
    fn two_lines_are_three_seconds_apart() {
        let mut c = OwnerChat::new();
        c.submit(say("one"), secs(0.0), true).unwrap();
        assert!(c.poll(secs(0.0), true).is_some());
        let a = c.submit(say("two"), secs(0.1), true).unwrap();
        assert_eq!(a.ahead, 0);
        assert_eq!(a.wait, secs(2.9), "the wait is told");
        assert!(c.poll(secs(0.1), true).is_none());
        assert!(c.poll(secs(2.99), true).is_none(), "not before 3 s");
        let said = c.poll(secs(3.0), true).expect("at 3 s");
        assert_eq!(said.text.as_str(), "two");
    }

    #[test]
    fn at_most_three_wait_and_the_fourth_is_refused_with_a_reason() {
        let mut c = OwnerChat::new();
        c.submit(say("a"), secs(0.0), true).unwrap();
        assert!(c.poll(secs(0.0), true).is_some());
        for (i, t) in ["b", "c", "d"].into_iter().enumerate() {
            let a = c.submit(say(t), secs(0.2), true).unwrap();
            assert_eq!(a.ahead, i);
        }
        assert_eq!(c.queued(), QUEUE_CAP);
        let refused = c.submit(say("e"), secs(0.2), true);
        assert_eq!(refused, Err(Refusal::QueueFull));
        let reply = reply_for(&refused);
        assert!(!reply.ok);
        assert!(reply.text.contains("already waiting"), "{}", reply.text);
        assert_eq!(reply.data.as_ref().unwrap()["reason"], "queue_full");
        // they come out in order, paced
        let mut out = Vec::new();
        let mut t = 0.2;
        while out.len() < 3 && t < 60.0 {
            if let Some(s) = c.poll(secs(t), true) {
                out.push((s.text.as_str().to_string(), t));
            }
            t += 0.1;
        }
        assert_eq!(out.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(), ["b", "c", "d"]);
        assert!(out[1].1 - out[0].1 >= 2.99 && out[2].1 - out[1].1 >= 2.99, "{out:?}");
        assert_eq!(c.stats().refused, 1);
    }

    #[test]
    fn nothing_is_taken_or_said_outside_the_game_and_waiting_lines_are_dropped() {
        let mut c = OwnerChat::new();
        let refused = c.submit(say("hi"), secs(0.0), false);
        assert_eq!(refused, Err(Refusal::NotInGame));
        assert_eq!(reply_for(&refused).data.unwrap()["reason"], "not_in_game");
        assert_eq!(c.queued(), 0);
        // a line waiting when the bot leaves the game is not said after it comes back
        c.submit(say("one"), secs(1.0), true).unwrap();
        assert!(c.poll(secs(1.0), true).is_some());
        c.submit(say("two"), secs(1.5), true).unwrap();
        assert_eq!(c.queued(), 1);
        assert!(c.poll(secs(5.0), false).is_none(), "not in the game: nothing");
        assert_eq!(c.queued(), 0, "dropped");
        assert!(c.poll(secs(60.0), true).is_none(), "and it does not come back");
        assert_eq!(c.stats().dropped, 1);
    }

    /// DDNet drops a line sent less than `(31 + length) / 32` seconds after the last one: a 255-character line needs 8 s.
    #[test]
    fn a_long_line_waits_for_the_servers_per_character_rule() {
        let long = "x".repeat(255);
        let mut c = OwnerChat::new();
        c.submit(say("hi"), secs(0.0), true).unwrap();
        assert!(c.poll(secs(0.0), true).is_some());
        let a = c.submit(say(&long), secs(3.0), true).unwrap();
        assert_eq!(a.wait, secs(6.0), "9 s after the first line, not 3");
        assert!(c.poll(secs(8.9), true).is_none());
        assert!(c.poll(secs(9.0), true).is_some());
        assert_eq!(gap_for(1), MIN_GAP);
        assert_eq!(gap_for(32), MIN_GAP);
        assert_eq!(gap_for(255), Duration::from_secs(9));
        assert_eq!(gap_for(0), MIN_GAP);
    }

    /// Whatever the owner does, driven at the fastest the bot allows for two minutes: never closer than 3 s, never more than 10 in a
    /// minute, and the modelled server score never reaches the mute threshold.
    #[test]
    fn flooding_for_two_minutes_stays_inside_every_limit() {
        let mut c = OwnerChat::new();
        let mut sent_at: Vec<f64> = Vec::new();
        let (mut score, mut score_at) = (0.0f64, 0.0f64);
        let mut worst = 0.0f64;
        let mut t = 0.0;
        while t < 120.0 {
            // the owner hammers the button every 200 ms
            let _ = c.submit(say("spam"), secs(t), true);
            if c.poll(secs(t), true).is_some() {
                score = (score - 50.0 * (t - score_at)).max(0.0) + 250.0;
                score_at = t;
                worst = worst.max(score);
                sent_at.push(t);
            }
            t += 0.2;
        }
        assert!(sent_at.len() >= 10, "it does say things: {}", sent_at.len());
        for w in sent_at.windows(2) {
            assert!(w[1] - w[0] >= 2.99, "{w:?}");
        }
        for (i, &s) in sent_at.iter().enumerate() {
            let in_minute = sent_at[i..].iter().take_while(|&&u| u < s + 60.0).count();
            assert!(in_minute <= MAX_PER_MINUTE, "{in_minute} lines in the minute from {s}");
        }
        assert!(worst <= SCORE_CEILING + 1e-6, "modelled server score {worst}");
        let st = c.stats();
        assert!(st.refused > 0 && st.sent as usize == sent_at.len(), "{st:?}");
    }

    #[test]
    fn a_line_that_would_wait_too_long_is_refused_and_says_when_to_retry() {
        let mut c = OwnerChat::new();
        // ten lines as fast as the bot lets them
        let mut t = 0.0;
        let mut sent = 0;
        while sent < MAX_PER_MINUTE {
            let _ = c.submit(say("x"), secs(t), true);
            if c.poll(secs(t), true).is_some() {
                sent += 1;
            }
            t += 0.1;
        }
        // now the minute window is full: more lines queue up for it, until one would wait over 30 s
        let mut outcomes = Vec::new();
        for _ in 0..QUEUE_CAP {
            outcomes.push(c.submit(say("y"), secs(t), true));
        }
        let first_refusal = outcomes.iter().find_map(|r| r.as_ref().err().copied());
        assert!(
            matches!(first_refusal, Some(Refusal::TooSoon { .. })),
            "a line behind a full minute is refused as too soon: {outcomes:?}"
        );
        let reply = reply_for(&Err(first_refusal.unwrap()));
        assert!(reply.text.contains("try again in about"), "{}", reply.text);
        assert_eq!(reply.data.unwrap()["reason"], "rate_limited");
        // whatever was taken waited at most MAX_WAIT
        for r in outcomes.iter().flatten() {
            assert!(r.wait <= MAX_WAIT, "{r:?}");
        }
    }

    #[test]
    fn answers_never_contain_the_text() {
        let secret = "SECRET-LINE-12345";
        let mut c = OwnerChat::new();
        let results = vec![
            c.submit(say(secret), secs(0.0), false),
            c.submit(say(secret), secs(0.0), true),
            c.submit(say(secret), secs(0.0), true),
        ];
        for r in &results {
            let reply = reply_for(r);
            assert!(!reply.text.contains("SECRET"), "{}", reply.text);
            assert!(!format!("{:?}", reply.data).contains("SECRET"));
        }
        assert!(
            !format!("{c:?}").contains("SECRET"),
            "Debug of the queue shows lengths only"
        );
    }

    #[test]
    fn the_queue_never_grows_past_the_cap_whatever_is_thrown_at_it() {
        let mut c = OwnerChat::new();
        let mut t = 0.0;
        for i in 0..2000 {
            let _ = c.submit(say("z"), secs(t), i % 7 != 0);
            assert!(c.queued() <= QUEUE_CAP);
            if i % 3 == 0 {
                let _ = c.poll(secs(t), i % 11 != 0);
            }
            t += 0.37;
        }
    }
}
