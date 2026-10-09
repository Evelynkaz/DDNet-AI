//! Automatic detection of an F-DDrace `/1vs1` duel (task 4.12, D-108): the bot never kills itself in one.
//!
//! **Why.** In a duel any death of ours (`Cl_Kill`, `/kill`, a kill tile) is a point for the opponent when he touched us, and a tee frozen
//! on the ground for a second loses the round anyway (`arenas.cpp`: `CanSelfkill`, `IncreaseScore`, `Tick`), so a kill the bot decides
//! never helps there. Task 4.11 (D-102) made that a switch the owner sets by hand; this finds it from what the client sees.
//!
//! **The structural signal.** `CArenas::StartFight` puts both players into a free DDRace team 1..63 (`SetForceCharacterTeam`) and locks
//! it (`SetTeamLock`). The server sends the teams to every client (`Sv_TeamsState`, the snapshot's `teams`). Our team is not 0 (nor the
//! super team 64) and **exactly one other present player** is in it. `EndFight` puts both back to team 0, which ends it
//! ([`RELEASE_TICKS`] later, to ride out a transition). Without a teams message the signal is unknown, not false.
//!
//! **But a two-player team alone proves nothing** (review 4.12, F1): vanilla DDNet has them too (`/team`, an admin's `set_team_ddr`),
//! and F-DDrace's two-player Durak card game makes one (`durak.cpp`). So the team counts **only with F-DDrace evidence**: a system chat
//! line of the `/1vs1` minigame seen for the current fight ([`EVIDENCE_TICKS`]), i.e. `You have been invited to a fight by '..'` or the
//! accept lines. The evidence is used up when the duel really ends (no counting team for [`RELEASE_TICKS`], or the end line comes; one snapshot
//! that does not show the team does not end it: team numbers are partly cosmetic on F-DDrace), and it lapses
//! [`EVIDENCE_TICKS`] after the last such line, so a team that stays without any `/1vs1` chat is released after that long.
//!
//! **A second kind of evidence, for servers whose duel command we do not know** (joniTee starts duels with `/duel`, which is not in the
//! F-DDrace source): the owner typed a duel command (`/duel`, `/1vs1`, or those of `duel_commands` in the settings file) on the website, and
//! the bot sent that line ([`crate::Bot::on_owner_command`]). Only a line the bot itself sent counts, so no player can fake it. It lasts as
//! long as the chat evidence and the team decides from then on, with the same release and end rules.
//!
//! **Chat as a signal of its own.** `You have accepted the invite by '<name>'` / `'<name>' has accepted your invite` also arm the
//! detector for [`CHAT_ARM_TICKS`] (the first seconds, before the team has arrived); `'<a>' won a 1vs1 round against '<b>'! Final scores:
//! n - m` and `'<a>' left a 1vs1 round against '<b>'! Current scores: n - m` with our own name in the right place end the fight. Only
//! `client_id` -1 lines are looked at; nothing is stored or logged. The lines are English (`Localize` for the bot's language); another
//! language simply never gives evidence, and then the detector says nothing (the owner's `--no-selfkill` stays).
//!
//! The detector owns no kill decision: [`crate::Bot`] ORs it with the owner's switch (`--no-selfkill`, the marker) and behaves as if the
//! switch were on, with the same exceptions (the owner's console `!kill` and typed lines stay). The owner turns the detector itself off
//! with `--no-duel-detect` or the marker file `<data-dir>/bot/duel-detect.off` (re-read once a second, like `selfkill.off`).

use ddai_net::tuning::TeamsState;

use crate::players::PlayerTable;

/// The structural signal is held this long after it stops (2 s).
pub const RELEASE_TICKS: i32 = 100;
/// A chat arm lasts this long (1 min) unless the team confirms it (then the team decides) or an end line arrives.
pub const CHAT_ARM_TICKS: i32 = 3000;
/// F-DDrace evidence (an invite or accept line) lets the team signal count for this long (20 min) after the last such line.
pub const EVIDENCE_TICKS: i32 = 60_000;
/// The highest ordinary DDRace team (`NUM_DDRACE_TEAMS` 65 = `TEAM_SUPER` 64 + 1).
const TEAM_SUPER: i32 = 64;

/// What made the detector say "duel".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuelWhy {
    /// Our DDRace team holds exactly one other player.
    Team,
    /// The server's chat said we accepted a fight.
    Chat,
}

impl DuelWhy {
    pub fn name(self) -> &'static str {
        match self {
            DuelWhy::Team => "team",
            DuelWhy::Chat => "chat",
        }
    }
}

/// What a system chat line means for the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatSignal {
    /// "You have been invited to a fight by '..', type '/1vs1 ..' to join": the server is F-DDrace and a fight is being made. Evidence only.
    Invited,
    /// "You have accepted the invite by '..'" / "'..' has accepted your invite": a fight starts for us.
    Accepted,
    /// "'..' won a 1vs1 round against '..'! Final scores: n - m" / "'..' left a 1vs1 round against '..'! Current scores: n - m" naming us
    /// in the winner's/leaver's or the other's place: the fight is over.
    Over,
}

/// `Final scores: n - m` / `Current scores: n - m` and nothing after it: the digits only (so no `(` and no vote text can follow).
fn ends_with_scores(rest: &str) -> bool {
    let mut parts = rest.splitn(2, " - ");
    let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.len() <= 6 && s.bytes().all(|c| c.is_ascii_digit());
    digits(a) && digits(b)
}

/// Whether `text` is exactly an F-DDrace end line naming `own` as the first or the second player (`arenas.cpp`:
/// `'%s' won a 1vs1 round against '%s'! Final scores: %d - %d`, `'%s' left a 1vs1 round against '%s'! Current scores: %d - %d`).
fn is_over_line(text: &str, own: &str) -> bool {
    for (verb, tail) in [
        (" won a 1vs1 round against '", "'! Final scores: "),
        (" left a 1vs1 round against '", "'! Current scores: "),
    ] {
        // The score tail is the last thing in the line: find its last occurrence and require only `n - m` after it.
        let Some(t) = text.rfind(tail) else { continue };
        if !ends_with_scores(&text[t + tail.len()..]) {
            continue;
        }
        let body = &text[..t + 1]; // up to and including the closing quote of the second name
        // We are the first name: the line starts with `'<own>' won a 1vs1 round against '`.
        let first = format!("'{own}'{verb}");
        if body.starts_with(&first) && body.len() > first.len() {
            return true;
        }
        // We are the second name: the body ends with `' won a 1vs1 round against '<own>'`.
        let second = format!("'{verb}{own}'");
        if body.starts_with('\'') && body.ends_with(&second) && body.len() > second.len() {
            return true;
        }
    }
    false
}

/// Reads one chat line. `own_name` is our nickname (empty: unknown, then only the invite and accept lines can be read). Nothing is kept.
pub fn read_chat(client_id: i32, text: &str, own_name: &str) -> Option<ChatSignal> {
    if client_id != -1 {
        return None;
    }
    if text.starts_with("You have been invited to a fight by '") {
        return Some(ChatSignal::Invited);
    }
    if text.starts_with("You have accepted the invite by '") {
        return Some(ChatSignal::Accepted);
    }
    if text.starts_with('\'') && text.ends_with("' has accepted your invite") {
        return Some(ChatSignal::Accepted);
    }
    if !own_name.is_empty() && is_over_line(text, own_name) {
        return Some(ChatSignal::Over);
    }
    None
}

/// The structural signal for `own_id`: `Some(true)` in a two-player DDRace team 1..63, `Some(false)` outside one, `None` when the teams
/// message does not cover us.
pub fn in_duel_team(own_id: i32, teams: &TeamsState, players: &PlayerTable) -> Option<bool> {
    let own = usize::try_from(own_id)
        .ok()
        .filter(|&i| i < teams.received.min(teams.teams.len()))?;
    let team = teams.teams[own];
    if team <= 0 || team >= TEAM_SUPER {
        return Some(false);
    }
    let others = teams.teams[..teams.received.min(teams.teams.len())]
        .iter()
        .enumerate()
        .filter(|&(i, &t)| i != own && t == team && players.get(i as i32).is_some_and(|s| s.present))
        .count();
    Some(others == 1)
}

/// The one other present player in our two-player DDRace team `1..64`, if there is exactly one.
fn duel_opponent(own_id: i32, teams: &TeamsState, players: &PlayerTable) -> Option<usize> {
    let own = usize::try_from(own_id)
        .ok()
        .filter(|&i| i < teams.received.min(teams.teams.len()))?;
    let team = teams.teams[own];
    if team <= 0 || team >= TEAM_SUPER {
        return None;
    }
    let mut others = teams.teams[..teams.received.min(teams.teams.len())]
        .iter()
        .enumerate()
        .filter(|&(i, &t)| i != own && t == team && players.get(i as i32).is_some_and(|s| s.present))
        .map(|(i, _)| i);
    let first = others.next()?;
    others.next().is_none().then_some(first)
}

/// Our own DDRace team number, `None` when the teams message does not cover us.
fn own_team(own_id: i32, teams: &TeamsState) -> Option<i32> {
    let own = usize::try_from(own_id)
        .ok()
        .filter(|&i| i < teams.received.min(teams.teams.len()))?;
    Some(teams.teams[own])
}

/// A change of the detector's answer, for the log and the clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuelChange {
    Started(DuelWhy),
    Ended,
}

#[derive(Debug, Default)]
pub struct DuelDetector {
    /// The structural signal's last known raw value (a two-player team), evidence or not.
    team_now: Option<bool>,
    /// The last tick the team signal was true **with evidence** (what counts).
    team_true_at: Option<i32>,
    /// The other player of the confirmed duel team: while our number shows as 64 (a rainbow-named player in view), his number says whether the
    /// fight is still on (review 4.12, F9: a rainbow number cycles 1..63, never 0).
    opponent: Option<usize>,
    /// A chat arm: until this tick.
    chat_until: Option<i32>,
    /// F-DDrace evidence for the current fight: until this tick.
    evidence_until: Option<i32>,
    /// Evidence kept across a reset (a reconnect, a reload of the same map): ticks it still has, re-anchored on the next call (the game tick
    /// may have restarted).
    evidence_left: Option<i32>,
    active: Option<DuelWhy>,
}

impl DuelDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget everything (another map, the detector switched off).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// A reconnect or a reload of the same map (review 4.12, F7): the teams and the chat arm are gone, but the evidence stays. The
    /// timeout code (D-100) puts the bot straight back into its fight team, and the fight is still the same one.
    pub fn reset_team(&mut self, tick: i32) {
        let left = self
            .evidence_until
            .map(|u| u - tick)
            .filter(|&l| l > 0)
            .or(self.evidence_left);
        *self = Self::default();
        self.evidence_left = left;
    }

    /// Forget only the evidence (the bot was offline longer than it may carry one over, review 4.12, F11).
    pub fn drop_evidence(&mut self) {
        self.evidence_until = None;
        self.evidence_left = None;
    }

    fn rebase(&mut self, tick: i32) {
        if let Some(left) = self.evidence_left.take() {
            self.evidence_until = Some(tick + left);
        }
    }

    /// The answer: `Some(why)` while a duel is on.
    pub fn active(&self) -> Option<DuelWhy> {
        self.active
    }

    /// Task 3.23 (D-121): the client id of the one other player of the duel team, while a duel is on and the team named him (a duel that only the chat
    /// armed has none yet).
    pub fn opponent(&self) -> Option<i32> {
        self.active.and(self.opponent).and_then(|o| i32::try_from(o).ok())
    }

    fn evidence(&self, tick: i32) -> bool {
        self.evidence_until.is_some_and(|until| tick < until)
    }

    /// A chat signal at `tick`.
    pub fn on_chat(&mut self, tick: i32, signal: ChatSignal) -> Option<DuelChange> {
        self.rebase(tick);
        match signal {
            ChatSignal::Invited => self.evidence_until = Some(tick + EVIDENCE_TICKS),
            ChatSignal::Accepted => {
                self.evidence_until = Some(tick + EVIDENCE_TICKS);
                self.chat_until = Some(tick + CHAT_ARM_TICKS);
            }
            ChatSignal::Over => {
                // The fight is over by the server's own word: nothing counts until the next invite (a stale snapshot that still shows the
                // team is no evidence of a new fight).
                self.chat_until = None;
                self.evidence_until = None;
                self.team_true_at = None;
                self.opponent = None;
            }
        }
        self.settle(tick)
    }

    /// The owner sent a duel command (`/duel x`, `/1vs1 x`) through the website chat: evidence that this server has such a minigame,
    /// the same as the server's own invite line (nothing a player can say counts: only a line the bot itself sent).
    pub fn on_owner_command(&mut self, tick: i32) -> Option<DuelChange> {
        self.on_chat(tick, ChatSignal::Invited)
    }

    /// One snapshot: `teams` is the snapshot's teams state (`None`: never received).
    pub fn update(
        &mut self,
        tick: i32,
        own_id: i32,
        teams: Option<&TeamsState>,
        players: &PlayerTable,
    ) -> Option<DuelChange> {
        self.rebase(tick);
        if let Some(t) = teams {
            // F-DDrace sends our own number as 64 (its "super team" marker) while a rainbow-named player of our team is in view, and the
            // opponent's number then cycles (review 4.12, F8): with evidence alive that is "unknown", and a duel that is on stays on.
            let super_team = self.evidence(tick) && own_team(own_id, t) == Some(TEAM_SUPER);
            // A finished duel with our number still 64 (the rainbow-named player is in view in team 0): the opponent shows 0 or is gone.
            let opponent_left = self.opponent.is_some_and(|o| {
                (o < t.received.min(t.teams.len()) && t.teams[o] == 0)
                    || !players.get(o as i32).is_some_and(|s| s.present)
            });
            if super_team && !opponent_left {
                if self.active == Some(DuelWhy::Team) {
                    self.team_true_at = Some(tick);
                }
                return self.settle(tick);
            }
            let known = if super_team {
                Some(false)
            } else {
                in_duel_team(own_id, t, players)
            };
            match known {
                Some(true) => {
                    self.team_now = Some(true);
                    if self.evidence(tick) {
                        self.opponent = duel_opponent(own_id, t, players);
                    }
                    if self.evidence(tick) {
                        self.team_true_at = Some(tick);
                        // The team confirms what the chat armed: from now on the team decides.
                        self.chat_until = None;
                    }
                }
                Some(false) => {
                    // One snapshot that does not show the team is not the end of the fight (team numbers are partly cosmetic on
                    // F-DDrace, review 4.12, F6): the hold of RELEASE_TICKS rides it out, and the evidence is used up only when the duel
                    // really ends (`settle`). A chat arm that the team has not confirmed yet stays too: the team message may trail it.
                    self.team_now = Some(false);
                }
                None => {}
            }
        }
        self.settle(tick)
    }

    fn settle(&mut self, tick: i32) -> Option<DuelChange> {
        let team_on = (self.team_now == Some(true) && self.evidence(tick))
            || self.team_true_at.is_some_and(|t| tick - t < RELEASE_TICKS);
        let chat_on = self.chat_until.is_some_and(|until| tick < until);
        let now = if team_on {
            Some(DuelWhy::Team)
        } else if chat_on {
            Some(DuelWhy::Chat)
        } else {
            None
        };
        if now == self.active {
            return None;
        }
        let was = self.active;
        self.active = now;
        if was == Some(DuelWhy::Team) && now.is_none() {
            // The team-based duel really ended: its evidence is used up (a Durak team that follows is no duel).
            self.evidence_until = None;
            self.team_true_at = None;
            self.opponent = None;
        }
        match (was, now) {
            (None, Some(w)) => Some(DuelChange::Started(w)),
            (Some(_), None) => Some(DuelChange::Ended),
            // Chat -> team: the same duel, now confirmed; not a change worth a line.
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::players::test_support::player;
    use crate::relations::Relations;

    fn table(ids: &[i32]) -> PlayerTable {
        let mut t = PlayerTable::new([0; 16]);
        let list: Vec<_> = ids
            .iter()
            .map(|&i| player(i, &format!("p{i}"), "", i == 0, 0, Some(0)))
            .collect();
        t.update(&list, &Relations::new());
        t
    }

    fn teams(assign: &[(usize, i32)]) -> TeamsState {
        let mut t = TeamsState {
            teams: [0; 128],
            received: 128,
        };
        for &(i, team) in assign {
            t.teams[i] = team;
        }
        t
    }

    #[test]
    fn a_two_player_team_is_a_duel_and_nothing_else_is() {
        let players = table(&[0, 1, 2, 3]);
        assert_eq!(in_duel_team(0, &teams(&[]), &players), Some(false), "team 0");
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 5)]), &players),
            Some(false),
            "alone in a team"
        );
        assert_eq!(in_duel_team(0, &teams(&[(0, 5), (1, 5)]), &players), Some(true));
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 63), (3, 63)]), &players),
            Some(true),
            "team 63"
        );
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 5), (1, 5), (2, 5)]), &players),
            Some(false),
            "three in a team"
        );
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 5), (1, 6)]), &players),
            Some(false),
            "others in another team"
        );
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 64), (1, 64)]), &players),
            Some(false),
            "the super team"
        );
        assert_eq!(
            in_duel_team(0, &teams(&[(0, -1), (1, -1)]), &players),
            Some(false),
            "flock"
        );
    }

    #[test]
    fn a_stale_team_entry_of_a_player_who_left_does_not_count() {
        let players = table(&[0, 1]);
        // Client 9 is not in the table: its slot in the teams message is stale.
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 5), (1, 5), (9, 5)]), &players),
            Some(true),
            "the ghost is not counted"
        );
        let players = table(&[0]);
        assert_eq!(
            in_duel_team(0, &teams(&[(0, 5), (9, 5)]), &players),
            Some(false),
            "the only other is a ghost"
        );
    }

    #[test]
    fn a_teams_message_that_does_not_cover_us_says_nothing() {
        let players = table(&[0, 1]);
        let mut t = teams(&[(0, 5), (1, 5)]);
        t.received = 0;
        assert_eq!(in_duel_team(0, &t, &players), None);
        t.received = 1;
        assert_eq!(in_duel_team(1, &t, &players), None, "client 1 is past `received`");
        assert_eq!(in_duel_team(-1, &t, &players), None);
    }

    #[test]
    fn the_team_starts_the_duel_at_once_and_ends_it_after_the_release_time() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(5, ChatSignal::Invited); // F-DDrace evidence
        assert_eq!(d.update(10, 0, Some(&teams(&[])), &players), None);
        assert_eq!(
            d.update(20, 0, Some(&teams(&[(0, 3), (1, 3)])), &players),
            Some(DuelChange::Started(DuelWhy::Team))
        );
        assert_eq!(d.active(), Some(DuelWhy::Team));
        assert_eq!(d.update(22, 0, Some(&teams(&[(0, 3), (1, 3)])), &players), None);
        // Back in team 0: held for RELEASE_TICKS after the last snapshot that showed the team (22).
        assert_eq!(d.update(30, 0, Some(&teams(&[])), &players), None);
        assert_eq!(d.update(22 + RELEASE_TICKS - 1, 0, Some(&teams(&[])), &players), None);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        assert_eq!(
            d.update(22 + RELEASE_TICKS, 0, Some(&teams(&[])), &players),
            Some(DuelChange::Ended)
        );
        assert_eq!(d.active(), None);
    }

    #[test]
    fn no_teams_message_changes_nothing() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(1, ChatSignal::Invited);
        assert_eq!(d.update(5, 0, None, &players), None);
        assert_eq!(d.active(), None);
        d.update(6, 0, Some(&teams(&[(0, 3), (1, 3)])), &players);
        assert_eq!(
            d.update(7, 0, None, &players),
            None,
            "unknown is not false: the answer stays"
        );
        assert_eq!(d.active(), Some(DuelWhy::Team));
    }

    #[test]
    fn the_chat_arms_before_the_team_arrives_and_the_team_then_takes_over() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.update(100, 0, Some(&teams(&[])), &players);
        assert_eq!(
            d.on_chat(101, ChatSignal::Accepted),
            Some(DuelChange::Started(DuelWhy::Chat))
        );
        // The teams message still shows team 0 a snapshot later: the chat arm stays.
        assert_eq!(d.update(102, 0, Some(&teams(&[])), &players), None);
        assert_eq!(d.active(), Some(DuelWhy::Chat));
        // The team arrives: same duel, no new change.
        assert_eq!(d.update(103, 0, Some(&teams(&[(0, 7), (1, 7)])), &players), None);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        // And now the team decides: back to team 0 ends it (after the release time), the old chat arm does not revive it.
        d.update(110, 0, Some(&teams(&[])), &players);
        assert_eq!(
            d.update(103 + RELEASE_TICKS, 0, Some(&teams(&[])), &players),
            Some(DuelChange::Ended)
        );
        assert_eq!(d.update(300, 0, Some(&teams(&[])), &players), None);
    }

    #[test]
    fn an_unconfirmed_chat_arm_expires() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(100, ChatSignal::Accepted);
        assert_eq!(d.update(100 + CHAT_ARM_TICKS - 1, 0, None, &players), None);
        assert_eq!(d.active(), Some(DuelWhy::Chat));
        assert_eq!(
            d.update(100 + CHAT_ARM_TICKS, 0, None, &players),
            Some(DuelChange::Ended)
        );
    }

    #[test]
    fn the_over_line_ends_the_duel_at_once_even_while_the_team_still_shows() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(5, ChatSignal::Invited);
        d.update(10, 0, Some(&teams(&[(0, 3), (1, 3)])), &players);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        assert_eq!(d.on_chat(12, ChatSignal::Over), Some(DuelChange::Ended));
        // A stale snapshot with the team still set is no evidence of a new fight: the evidence went with the end line.
        assert_eq!(d.update(13, 0, Some(&teams(&[(0, 3), (1, 3)])), &players), None);
        assert_eq!(d.active(), None);
    }

    /// Review 4.12 F1: vanilla DDNet (`/team`, an admin's `set_team_ddr`) and F-DDrace's Durak game make two-player teams too.
    #[test]
    fn a_two_player_team_without_f_ddrace_evidence_is_not_a_duel() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        for tick in (10..2000).step_by(2) {
            assert_eq!(
                d.update(tick, 0, Some(&teams(&[(0, 5), (1, 5)])), &players),
                None,
                "tick {tick}"
            );
        }
        assert_eq!(d.active(), None);
    }

    #[test]
    fn f_ddrace_evidence_and_the_team_make_a_duel_with_either_order() {
        let players = table(&[0, 1]);
        // Evidence first (the invite), then the team.
        let mut d = DuelDetector::new();
        assert_eq!(d.on_chat(10, ChatSignal::Invited), None, "an invite alone is no duel");
        assert_eq!(
            d.update(40, 0, Some(&teams(&[(0, 5), (1, 5)])), &players),
            Some(DuelChange::Started(DuelWhy::Team))
        );
        // The team first, the evidence a snapshot later.
        let mut d = DuelDetector::new();
        assert_eq!(d.update(10, 0, Some(&teams(&[(0, 5), (1, 5)])), &players), None);
        assert_eq!(
            d.on_chat(12, ChatSignal::Invited),
            Some(DuelChange::Started(DuelWhy::Team)),
            "the team was already there: the evidence completes it"
        );
        assert_eq!(d.update(14, 0, Some(&teams(&[(0, 5), (1, 5)])), &players), None);
    }

    /// A Durak-like team (two in it, no `/1vs1` chat) after a duel is over is not a duel: the evidence was used up with the fight.
    #[test]
    fn the_evidence_is_used_up_when_the_fight_ends() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(10, ChatSignal::Accepted);
        d.update(20, 0, Some(&teams(&[(0, 5), (1, 5)])), &players);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        d.update(300, 0, Some(&teams(&[])), &players); // back in team 0: the fight is over
        d.update(300 + RELEASE_TICKS, 0, Some(&teams(&[])), &players);
        assert_eq!(d.active(), None);
        // A new two-player team with no chat: not a duel, however long it stays.
        assert_eq!(d.update(500, 0, Some(&teams(&[(0, 9), (1, 9)])), &players), None);
        assert_eq!(d.update(5000, 0, Some(&teams(&[(0, 9), (1, 9)])), &players), None);
        assert_eq!(d.active(), None);
    }

    /// The upper bound: a team that stays with no `/1vs1` chat for EVIDENCE_TICKS releases.
    #[test]
    fn a_team_with_no_f_ddrace_chat_for_twenty_minutes_is_released() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(10, ChatSignal::Invited);
        let team = teams(&[(0, 5), (1, 5)]);
        d.update(20, 0, Some(&team), &players);
        let mut ended = None;
        for tick in (22..(10 + EVIDENCE_TICKS + 400)).step_by(2) {
            if d.update(tick, 0, Some(&team), &players) == Some(DuelChange::Ended) {
                ended = Some(tick);
                break;
            }
        }
        let t = ended.expect("released");
        assert!(
            (10 + EVIDENCE_TICKS..10 + EVIDENCE_TICKS + RELEASE_TICKS + 4).contains(&t),
            "{t}"
        );
        // Another invite line starts the evidence again.
        assert_eq!(
            d.on_chat(t + 2, ChatSignal::Invited),
            Some(DuelChange::Started(DuelWhy::Team))
        );
        assert_eq!(d.update(t + 4, 0, Some(&team), &players), None);
    }

    /// Review 4.12 F6: one snapshot that does not show the fight team (a rainbow-named player's number, Durak, a flag marker) must not end
    /// the protection for the rest of the fight.
    #[test]
    fn one_bad_snapshot_in_the_middle_of_a_fight_does_not_end_the_protection() {
        let players = table(&[0, 1, 2]);
        let mut d = DuelDetector::new();
        let fight = teams(&[(0, 1), (1, 1)]);
        let glitch = teams(&[(0, 1), (1, 1), (2, 1)]); // a third number equal to ours, for one snapshot
        let mut changes = Vec::new();
        changes.extend(d.on_chat(10, ChatSignal::Accepted).map(|c| (10, c)));
        for tick in (20..3000).step_by(2) {
            let t = if tick == 400 { &glitch } else { &fight };
            if let Some(c) = d.update(tick, 0, Some(t), &players) {
                changes.push((tick, c));
            }
        }
        // Started by the chat arm at 10, confirmed by the team at 20; never ended, not even by the glitch at 400.
        assert_eq!(changes, vec![(10, DuelChange::Started(DuelWhy::Chat))], "{changes:?}");
        assert_eq!(d.active(), Some(DuelWhy::Team));
        // A glitch that lasts: past RELEASE_TICKS it is the end, and the evidence goes with it.
        let mut e = DuelDetector::new();
        e.on_chat(10, ChatSignal::Accepted);
        e.update(20, 0, Some(&fight), &players);
        for tick in (22..400).step_by(2) {
            e.update(tick, 0, Some(&glitch), &players);
        }
        assert_eq!(e.active(), None);
        assert_eq!(
            e.update(500, 0, Some(&fight), &players),
            None,
            "the evidence was used up with the duel"
        );
    }

    /// Review 4.12 F8: F-DDrace sends our own number as 64 while a rainbow-named player of our team is in view and the opponent's number
    /// cycles: with evidence alive that is unknown, and a duel that is on stays on.
    #[test]
    fn our_own_super_team_number_with_a_cycling_opponent_keeps_a_duel_on() {
        let players = table(&[0, 1]);
        let mut d = DuelDetector::new();
        d.on_chat(10, ChatSignal::Accepted);
        d.update(20, 0, Some(&teams(&[(0, 1), (1, 1)])), &players);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        for (i, tick) in (22..4000).step_by(2).enumerate() {
            let cycling = 1 + (i as i32 % 63);
            assert_eq!(
                d.update(tick, 0, Some(&teams(&[(0, 64), (1, cycling)])), &players),
                None,
                "tick {tick}"
            );
        }
        assert_eq!(
            d.active(),
            Some(DuelWhy::Team),
            "after 4000 ticks of 64 and a cycling opponent"
        );
        // Back to the real number: still on; a real end (team 0) ends it as usual.
        assert_eq!(d.update(4002, 0, Some(&teams(&[(0, 1), (1, 1)])), &players), None);
        d.update(4004, 0, Some(&teams(&[])), &players);
        assert_eq!(
            d.update(4004 + RELEASE_TICKS, 0, Some(&teams(&[])), &players),
            Some(DuelChange::Ended)
        );
        // Without evidence the super team is just "not a duel", and a duel is not started from it.
        let mut e = DuelDetector::new();
        assert_eq!(e.update(10, 0, Some(&teams(&[(0, 64), (1, 64)])), &players), None);
        e.on_chat(12, ChatSignal::Invited);
        assert_eq!(
            e.update(14, 0, Some(&teams(&[(0, 64), (1, 5)])), &players),
            None,
            "64 from the start: unknown, nothing starts"
        );
    }

    /// Review 4.12 F9: a finished duel (the server sends no end line) must not be held on by our number 64: after the fight we stand in team 0
    /// next to a rainbow-named player (we show 64), and the opponent shows 0, which a rainbow number (1..63) never does.
    #[test]
    fn our_number_64_does_not_hold_a_finished_duel_on_when_the_opponent_is_back_in_team_0() {
        let players = table(&[0, 1, 2]);
        let mut d = DuelDetector::new();
        d.on_owner_command(0);
        d.update(2, 0, Some(&teams(&[(0, 5), (1, 5)])), &players);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        // The fight is over: we show 64, the opponent 0, a third player cycles 1..63.
        let mut ended = None;
        for (i, tick) in (4..3000).step_by(2).enumerate() {
            let cycling = 1 + (i as i32 % 63);
            let t = teams(&[(0, 64), (2, cycling)]);
            if d.update(tick, 0, Some(&t), &players) == Some(DuelChange::Ended) {
                ended = Some(tick);
                break;
            }
        }
        let t = ended.expect("the finished duel ends");
        assert!(
            (4 + RELEASE_TICKS - 4..4 + RELEASE_TICKS + 4).contains(&t),
            "released after the hold: {t}"
        );
        assert_eq!(d.active(), None);
        // The evidence went with the duel: a Durak-like pair afterwards is no duel.
        assert_eq!(d.update(3000, 0, Some(&teams(&[(0, 9), (1, 9)])), &players), None);
    }

    /// An opponent who left the game counts like one in team 0.
    #[test]
    fn our_number_64_with_the_opponent_gone_ends_the_duel() {
        let mut d = DuelDetector::new();
        d.on_owner_command(0);
        let both = table(&[0, 1]);
        d.update(2, 0, Some(&teams(&[(0, 5), (1, 5)])), &both);
        let alone = table(&[0]);
        let mut ended = false;
        for tick in (4..400).step_by(2) {
            ended |= d.update(tick, 0, Some(&teams(&[(0, 64), (1, 17)])), &alone) == Some(DuelChange::Ended);
        }
        assert!(ended);
    }

    /// Review 4.12 F7: a reconnect or a reload of the same map keeps the evidence (the timeout code puts the bot back into its fight team).
    #[test]
    fn a_reset_of_the_team_keeps_the_evidence_and_a_full_reset_does_not() {
        let players = table(&[0, 1]);
        let fight = teams(&[(0, 3), (1, 3)]);
        let mut d = DuelDetector::new();
        d.on_chat(100, ChatSignal::Accepted);
        d.update(110, 0, Some(&fight), &players);
        assert_eq!(d.active(), Some(DuelWhy::Team));
        d.reset_team(200);
        assert_eq!(d.active(), None, "the team state is gone");
        // The game tick restarted (a map reload): the evidence is re-anchored on the next call.
        assert_eq!(
            d.update(10, 0, Some(&fight), &players),
            Some(DuelChange::Started(DuelWhy::Team))
        );
        // Still bounded: it lapses EVIDENCE_TICKS after the original line, i.e. what was left at the reset.
        let left = 100 + EVIDENCE_TICKS - 200;
        let mut ended = None;
        for tick in (12..(10 + left + 400)).step_by(2) {
            if d.update(tick, 0, Some(&fight), &players) == Some(DuelChange::Ended) {
                ended = Some(tick);
                break;
            }
        }
        let t = ended.expect("lapses");
        assert!((10 + left..10 + left + RELEASE_TICKS + 4).contains(&t), "{t}");
        // A full reset forgets the evidence.
        let mut e = DuelDetector::new();
        e.on_chat(100, ChatSignal::Accepted);
        e.reset();
        assert_eq!(e.update(110, 0, Some(&fight), &players), None);
    }

    #[test]
    fn the_chat_lines_are_read_by_their_f_ddrace_wording_and_only_from_the_system() {
        let own = "Muha";
        for (id, text, want) in [
            (
                -1,
                "You have accepted the invite by 'Rival'",
                Some(ChatSignal::Accepted),
            ),
            (-1, "'Rival' has accepted your invite", Some(ChatSignal::Accepted)),
            (
                -1,
                "'Rival' won a 1vs1 round against 'Muha'! Final scores: 10 - 7",
                Some(ChatSignal::Over),
            ),
            (
                -1,
                "'Muha' won a 1vs1 round against 'Rival'! Final scores: 10 - 7",
                Some(ChatSignal::Over),
            ),
            (
                -1,
                "'Muha' left a 1vs1 round against 'Rival'! Current scores: 3 - 4",
                Some(ChatSignal::Over),
            ),
            // Somebody else's fight: not ours.
            (-1, "'A' won a 1vs1 round against 'B'! Final scores: 10 - 7", None),
            (-1, "'Rival' left a 1vs1 round against 'B'! Current scores: 0 - 0", None),
            // An invitation is evidence of F-DDrace, not a duel.
            (
                -1,
                "You have been invited to a fight by 'Rival', type '/1vs1 Rival' to join",
                Some(ChatSignal::Invited),
            ),
            // Review 4.12 F2: lines that only look like the end line.
            (
                -1,
                "'Evil' called for vote to move 'Muha' to spectators (' won a 1vs1 round against ')",
                None,
            ),
            (
                -1,
                "'Bob' won a 1vs1 round against 'Muha' x'! Final scores: 10 - 7",
                None,
            ),
            (
                -1,
                "'Muha' x' won a 1vs1 round against 'Bob'! Final scores: 10 - 7",
                None,
            ),
            (
                -1,
                "'Bob' won a 1vs1 round against 'Alice'! Final scores: 10 - 7 ('Muha')",
                None,
            ),
            (
                -1,
                "'Muha' won a 1vs1 round against 'Bob'! Final scores: 10 - 7 (x)",
                None,
            ),
            (-1, "'Muha' won a 1vs1 round against 'Bob'! Final scores: ten - 7", None),
            (
                -1,
                "'Muha' won a 1vs1 round against 'Bob'! Current scores: 10 - 7",
                None,
            ),
            (-1, "'Muha' left a 1vs1 round against 'Bob'! Final scores: 10 - 7", None),
            (-1, "'Muha' won a 1vs1 round against 'Bob'", None),
            (
                -1,
                "x 'Muha' won a 1vs1 round against 'Bob'! Final scores: 10 - 7",
                None,
            ),
            // A player typing the words is not the server.
            (3, "You have accepted the invite by 'Rival'", None),
            (3, "'Rival' won a 1vs1 round against 'Muha'! Final scores: 10 - 7", None),
            (
                -1,
                "Kill Protection enabled. If you really want to kill, type /kill",
                None,
            ),
        ] {
            assert_eq!(read_chat(id, text, own), want, "{id}: {text}");
        }
        assert_eq!(
            read_chat(-1, "'A' won a 1vs1 round against 'B'! Final scores: 1 - 0", ""),
            None,
            "own name unknown"
        );
        assert_eq!(
            read_chat(-1, "'Rival' has accepted your invite", ""),
            Some(ChatSignal::Accepted)
        );
    }
}
