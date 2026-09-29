//! Task 8.4a acceptance criterion 2's `--anonymize` export mode: nicknames are stored as-is in a
//! local recording (D-038: "ники не публикуются", nicknames never leave the local machine as
//! themselves), but anything that might be shared or committed — an anonymized export, or a
//! synthetic committed test fixture — replaces `ClientInfo::name`/`clan` with a **stable
//! per-recording id**: the same real nickname always maps to the same id *within one recording*
//! (so a reconstruction/analysis that groups frames by player still works), but that mapping is
//! never written anywhere and is not derived from the name in any reversible way (a fresh
//! first-seen-order counter, not a hash of the name — a hash could in principle be reversed by
//! brute force over a small nickname space, `MAX_NAME_LENGTH = 16` bytes; a counter cannot be
//! reversed at all).
//!
//! Review round 2, finding F19: ids are assigned per `(client_id, stint)`
//! ([`crate::reconstruct::stint_for`]), not per bare `client_id` — a `client_id` slot reused by a
//! different real person within one recording (finding F5) would otherwise silently collapse both
//! people onto the same anonymized label.

use crate::format::{Frame, PlayerRecord};
use crate::reconstruct::{self, Identity};
use std::collections::HashMap;

/// Maps each distinct `(client_id, stint)` pair seen so far to a stable `"player_<n>"` id,
/// assigned in the order each pair was first observed — built fresh per export (never persisted,
/// never shared across recordings, so the same real person's nickname does not map to the same
/// anonymized id across two different recordings either, which would itself be a re-
/// identification channel).
///
/// Review round 2, finding F19: keyed by `(client_id, stint)`, not bare `client_id` — round 1's
/// fix keyed by slot alone, so a slot reused by a different real person within the *same*
/// recording (Alice disconnects from client id 3, Bob later connects and is assigned that same,
/// now-free, id) collapsed both people onto one anonymized label (`player_0` for both), silently
/// merging two different real people's data under one export identity — exactly the kind of bug
/// [`crate::reconstruct`]'s own `PlayerReconstruction::stint` (finding F5) already exists to
/// prevent for reconstruction; the anonymizer just hadn't been updated to use it yet. Reuses
/// [`reconstruct::stint_for`]/[`Identity`] directly (rather than a second, independent
/// implementation) so this can never disagree with how `reconstruct`'s own output is keyed for the
/// exact same recording.
#[derive(Debug, Default)]
pub struct Anonymizer {
    next_index: u32,
    assigned: HashMap<(i32, u32), String>,
    /// Stint-tracking state, one entry per `client_id` ever seen — see [`reconstruct::stint_for`].
    identities: HashMap<i32, Identity>,
    /// This export's own count of `Snapshot` frames processed so far (0-based) — the
    /// `frame_index` [`reconstruct::stint_for`] needs, kept in exact lockstep with how
    /// [`reconstruct::reconstruct`] counts the same thing over the same frame sequence (only
    /// `Snapshot` frames advance it; `GameEvent` frames do not).
    frame_index: usize,
}

impl Anonymizer {
    pub fn new() -> Self {
        Anonymizer::default()
    }

    fn id_for(&mut self, key: (i32, u32)) -> String {
        self.assigned
            .entry(key)
            .or_insert_with(|| {
                let id = format!("player_{}", self.next_index);
                self.next_index += 1;
                id
            })
            .clone()
    }

    /// Replaces `player.client_info.name`/`.clan` with this player's stable anonymized id (clan
    /// is blanked entirely — a clan tag is exactly as identifying as a name, and carries no
    /// information this crate's reconstruction needs); every other field (team, score, latency,
    /// skin, colors, the `DDNetPlayer` extension) is left untouched, since none of those on their
    /// own identify a real person the way a nickname does.
    ///
    /// Review round 2, finding F19: `stint_for` is called here with this player's *real* name
    /// (before it gets overwritten below) — calling it any later, after `ci.name` already holds
    /// the anonymized label, would compare the anonymized label against itself on every
    /// subsequent frame and could never detect a real name change at all. Called *unconditionally*
    /// for every player in `players` — including one with no `ClientInfo` at all (`name: None`) —
    /// exactly matching how `reconstruct::reconstruct`'s own loop calls it, so `frame_index`/
    /// `last_seen_frame` bookkeeping can never silently drift out of step between the two modules
    /// for the same recording (a player lacking `ClientInfo` simply has nothing left to
    /// anonymize afterwards, so the rest of this function is skipped for it, but the stint
    /// tracking itself must still see it).
    fn anonymize_player(&mut self, player: &mut PlayerRecord) {
        let name = player.client_info.as_ref().map(|ci| ci.name.as_str());
        let stint = reconstruct::stint_for(&mut self.identities, player.id, self.frame_index, name);
        if let Some(ci) = &mut player.client_info {
            let anon = self.id_for((player.id, stint));
            ci.name = anon;
            ci.clan = String::new();
        }
    }

    /// Anonymizes one [`Frame`] in place — a `Snapshot`'s player records, or any free-text game
    /// message body that a real player or the server's own text (which can itself echo a vote
    /// caller's or subject's name, e.g. `SvVoteSet`) might contain a nickname in. `Kill`'s
    /// `killer`/`victim` and `Tuning`'s params are left untouched — bare client ids/numbers,
    /// already exactly as anonymous as a `Snapshot`'s own `id` fields, carrying no free text.
    ///
    /// Review round 1, finding F2: round 1 only blanket-redacted `Chat`, leaving two other
    /// string-bearing kinds untouched:
    /// - `Broadcast.message` — a server-side broadcast can itself embed a nickname (e.g. a mod's
    ///   "well played, <name>!" broadcast) exactly as freely as a chat message can; redacted the
    ///   same way, wholesale, for the same reason (no reliable way to redact names within free
    ///   text short of replacing it outright).
    /// - `Other.debug` — the `{msg:?}` fallback text for every `GameMsg`/`ExGameMsg` variant this
    ///   crate does not curate a typed shape for (see [`crate::format::RecordedGameMessage`]'s own
    ///   doc comment). Several of DDNet's own message kinds embed a nickname directly in their
    ///   `Debug` output — `SvVoteSet`'s `description`/`reason` fields foremost (a vote-kick's
    ///   reason routinely reads like `"Kick 'RealNick'"`), but this crate does not maintain an
    ///   exhaustive list of which of the ~30 kinds do; instead of trying to enumerate them, this
    ///   keeps only the leading variant name (`"SvVoteSet"`, `"SvChatTime"`, ...) — the part before
    ///   the first `(` — and drops every field's value entirely, since none of this crate's own
    ///   code reads anything out of `Other.debug` besides knowing which kind of message it was.
    pub fn anonymize_frame(&mut self, frame: &mut Frame) {
        match frame {
            Frame::Snapshot { players, .. } => {
                for p in players {
                    self.anonymize_player(p);
                }
                // Review round 2, finding F19: advances in exact lockstep with
                // `reconstruct::reconstruct`'s own `frame_index` (only `Snapshot` frames count,
                // and only after this frame's own players have already been attributed to their
                // pre-advance stint, matching that function's own ordering) — required for
                // `stint_for`'s presence-gap detection to agree between the two modules.
                self.frame_index += 1;
            }
            Frame::GameEvent { message, .. } => match message {
                crate::format::RecordedGameMessage::Chat { message, .. } => {
                    *message = "<redacted>".to_string();
                }
                crate::format::RecordedGameMessage::Broadcast { message } => {
                    *message = "<redacted>".to_string();
                }
                crate::format::RecordedGameMessage::Other { debug } => {
                    *debug = match debug.find('(') {
                        Some(idx) if !debug[..idx].trim().is_empty() => debug[..idx].trim().to_string(),
                        _ => "<redacted>".to_string(),
                    };
                }
                crate::format::RecordedGameMessage::Kill { .. } | crate::format::RecordedGameMessage::Tuning(_) => {}
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{CharacterRecord, RecordedGameMessage};
    use ddai_net::generated::objects;

    fn player_with_name(id: i32, name: &str) -> PlayerRecord {
        PlayerRecord {
            id,
            info: objects::PlayerInfo {
                local: 0,
                client_id: id,
                team: 0,
                score: 0,
                latency: 0,
            },
            client_info: Some(objects::ClientInfo {
                name: name.to_string(),
                clan: "TheClan".to_string(),
                country: -1,
                skin: "default".to_string(),
                use_custom_color: 0,
                color_body: 0,
                color_feet: 0,
            }),
            ddnet: None,
        }
    }

    #[test]
    fn same_client_id_gets_the_same_stable_id_within_one_export() {
        let mut anon = Anonymizer::new();
        let mut frame1 = Frame::Snapshot {
            tick: 0,
            characters: vec![],
            players: vec![player_with_name(3, "RealNick")],
        };
        let mut frame2 = Frame::Snapshot {
            tick: 1,
            characters: vec![],
            players: vec![player_with_name(3, "RealNick")],
        };
        anon.anonymize_frame(&mut frame1);
        anon.anonymize_frame(&mut frame2);
        let name = |f: &Frame| match f {
            Frame::Snapshot { players, .. } => players[0].client_info.as_ref().unwrap().name.clone(),
            _ => unreachable!(),
        };
        assert_eq!(name(&frame1), name(&frame2));
        assert!(!name(&frame1).contains("RealNick"));
    }

    #[test]
    fn different_client_ids_get_different_stable_ids() {
        let mut anon = Anonymizer::new();
        let mut frame = Frame::Snapshot {
            tick: 0,
            characters: vec![CharacterRecord {
                id: 0,
                character: dummy_character(),
                ddnet: None,
            }],
            players: vec![player_with_name(0, "Alice"), player_with_name(1, "Bob")],
        };
        anon.anonymize_frame(&mut frame);
        let Frame::Snapshot { players, .. } = &frame else {
            unreachable!()
        };
        assert_ne!(
            players[0].client_info.as_ref().unwrap().name,
            players[1].client_info.as_ref().unwrap().name
        );
    }

    /// Review round 2, finding F19: a `client_id` slot reused by a different real person within
    /// one recording (Alice disconnects from id 3; a presence gap of at least one whole frame
    /// later, Bob connects and is assigned that same, now-free, id) must get a *different*
    /// anonymized label for each — round 1's fix keyed by bare `client_id` alone, so both
    /// collapsed onto the same `player_0`, silently merging two different real people's data
    /// under one export identity.
    #[test]
    fn a_slot_reused_by_a_different_person_gets_a_different_label() {
        let mut anon = Anonymizer::new();
        let mut alice_frame = Frame::Snapshot {
            tick: 0,
            characters: vec![],
            players: vec![player_with_name(3, "Alice")],
        };
        let mut gap_frame = Frame::Snapshot {
            tick: 1,
            characters: vec![],
            players: vec![], // id 3 absent for a whole frame — a real disconnect (finding F5).
        };
        let mut bob_frame = Frame::Snapshot {
            tick: 2,
            characters: vec![],
            players: vec![player_with_name(3, "Bob")],
        };
        anon.anonymize_frame(&mut alice_frame);
        anon.anonymize_frame(&mut gap_frame);
        anon.anonymize_frame(&mut bob_frame);

        let name = |f: &Frame| match f {
            Frame::Snapshot { players, .. } => players[0].client_info.as_ref().unwrap().name.clone(),
            _ => unreachable!(),
        };
        let alice_label = name(&alice_frame);
        let bob_label = name(&bob_frame);
        assert_ne!(
            alice_label, bob_label,
            "same slot, different people, must get different labels"
        );
        assert!(!alice_label.contains("Alice"));
        assert!(!bob_label.contains("Bob"));
    }

    /// Companion to the above: a slot reused by the same nick, but across a genuine presence gap,
    /// must *still* get a new stint (finding F5's own "presence, not just name, is the identity
    /// signal" rule) — and therefore a new anonymized label too, since the anonymizer keys by
    /// `(client_id, stint)`, not name.
    #[test]
    fn a_presence_gap_under_the_same_name_still_gets_a_different_label() {
        let mut anon = Anonymizer::new();
        let mut first = Frame::Snapshot {
            tick: 0,
            characters: vec![],
            players: vec![player_with_name(3, "Alice")],
        };
        let mut gap = Frame::Snapshot {
            tick: 1,
            characters: vec![],
            players: vec![],
        };
        let mut second = Frame::Snapshot {
            tick: 2,
            characters: vec![],
            players: vec![player_with_name(3, "Alice")],
        };
        anon.anonymize_frame(&mut first);
        anon.anonymize_frame(&mut gap);
        anon.anonymize_frame(&mut second);

        let name = |f: &Frame| match f {
            Frame::Snapshot { players, .. } => players[0].client_info.as_ref().unwrap().name.clone(),
            _ => unreachable!(),
        };
        assert_ne!(name(&first), name(&second));
    }

    #[test]
    fn clan_is_blanked_too() {
        let mut anon = Anonymizer::new();
        let mut frame = Frame::Snapshot {
            tick: 0,
            characters: vec![],
            players: vec![player_with_name(0, "Alice")],
        };
        anon.anonymize_frame(&mut frame);
        let Frame::Snapshot { players, .. } = &frame else {
            unreachable!()
        };
        assert_eq!(players[0].client_info.as_ref().unwrap().clan, "");
    }

    #[test]
    fn chat_message_text_is_redacted() {
        let mut anon = Anonymizer::new();
        let mut frame = Frame::GameEvent {
            tick_hint: 0,
            message: RecordedGameMessage::Chat {
                team: 0,
                client_id: 2,
                message: "hi, this is RealNick speaking".to_string(),
            },
        };
        anon.anonymize_frame(&mut frame);
        let Frame::GameEvent {
            message: RecordedGameMessage::Chat { message, .. },
            ..
        } = &frame
        else {
            unreachable!()
        };
        assert_eq!(message, "<redacted>");
    }

    /// Review round 1, finding F2: `Broadcast` carries free text exactly like `Chat` does, and
    /// must be blanket-redacted the same way.
    #[test]
    fn broadcast_message_text_is_redacted() {
        let mut anon = Anonymizer::new();
        let mut frame = Frame::GameEvent {
            tick_hint: 0,
            message: RecordedGameMessage::Broadcast {
                message: "well played, RealNick!".to_string(),
            },
        };
        anon.anonymize_frame(&mut frame);
        let Frame::GameEvent {
            message: RecordedGameMessage::Broadcast { message },
            ..
        } = &frame
        else {
            unreachable!()
        };
        assert_eq!(message, "<redacted>");
    }

    /// Review round 1, finding F2: `Other.debug` is the `{msg:?}` fallback for every message kind
    /// this crate does not curate — `game_msg_to_recorded`/`ex_game_msg_to_recorded` build it from
    /// `format!("{msg:?}")` on a *tuple-variant* enum (`GameMsg`/`ExGameMsg`, see
    /// `crates/ddai-net/src/generated/messages.rs`), so the real shape is always
    /// `VariantName(InnerStruct { field: value, ... })` — this must have every field's value
    /// dropped, keeping only the leading variant name, so a vote-kick's embedded reason/nickname
    /// never survives.
    #[test]
    fn other_debug_keeps_only_the_variant_name_dropping_every_field_value() {
        let mut anon = Anonymizer::new();
        let mut frame = Frame::GameEvent {
            tick_hint: 0,
            message: RecordedGameMessage::Other {
                debug: "SvVoteSet(SvVoteSet { timeout: 30, description: \"Kick 'RealNick'\", reason: \"griefing\" })"
                    .to_string(),
            },
        };
        anon.anonymize_frame(&mut frame);
        let Frame::GameEvent {
            message: RecordedGameMessage::Other { debug },
            ..
        } = &frame
        else {
            unreachable!()
        };
        assert_eq!(debug, "SvVoteSet");
        assert!(!debug.contains("RealNick"));
    }

    /// A degenerate `Other.debug` with no `(` at all (should not happen for a real `{:?}` of a
    /// struct/enum variant, but this crate never assumes input it did not itself validate) must
    /// still redact to a safe placeholder rather than passing the whole string through unchanged.
    #[test]
    fn other_debug_with_no_parenthesis_falls_back_to_a_placeholder() {
        let mut anon = Anonymizer::new();
        let mut frame = Frame::GameEvent {
            tick_hint: 0,
            message: RecordedGameMessage::Other {
                debug: "RealNick".to_string(),
            },
        };
        anon.anonymize_frame(&mut frame);
        let Frame::GameEvent {
            message: RecordedGameMessage::Other { debug },
            ..
        } = &frame
        else {
            unreachable!()
        };
        assert_eq!(debug, "<redacted>");
    }

    /// Review round 1, finding F2: a sweep across *every* string-bearing
    /// [`RecordedGameMessage`] kind — the exact class of finding the reviewer raised (some kinds
    /// were checked, others silently were not) — asserts none of them can carry the literal
    /// nickname through to the anonymized output. `Kill`/`Tuning` carry no free text at all (bare
    /// ids/numbers) and are intentionally excluded, matching `non_chat_game_events_are_left_untouched`.
    #[test]
    fn every_string_bearing_message_kind_has_the_nickname_redacted() {
        const NICK: &str = "RealNick";
        let cases: Vec<RecordedGameMessage> = vec![
            RecordedGameMessage::Chat {
                team: 0,
                client_id: 0,
                message: format!("hi, {NICK} here"),
            },
            RecordedGameMessage::Broadcast {
                message: format!("welcome {NICK}"),
            },
            RecordedGameMessage::Other {
                debug: format!("SvVoteSet(SvVoteSet {{ description: \"Kick '{NICK}'\" }})"),
            },
        ];
        for original in cases {
            let mut anon = Anonymizer::new();
            let mut frame = Frame::GameEvent {
                tick_hint: 0,
                message: original.clone(),
            };
            anon.anonymize_frame(&mut frame);
            let Frame::GameEvent { message, .. } = &frame else {
                unreachable!()
            };
            let rendered = format!("{message:?}");
            assert!(
                !rendered.contains(NICK),
                "{original:?} still leaked the nickname after anonymization: {rendered:?}"
            );
        }
    }

    #[test]
    fn non_chat_game_events_are_left_untouched() {
        let mut anon = Anonymizer::new();
        let original = RecordedGameMessage::Kill {
            killer: 0,
            victim: 1,
            weapon: 1,
            mode_special: 0,
        };
        let mut frame = Frame::GameEvent {
            tick_hint: 0,
            message: original.clone(),
        };
        anon.anonymize_frame(&mut frame);
        let Frame::GameEvent { message, .. } = &frame else {
            unreachable!()
        };
        assert_eq!(message, &original);
    }

    fn dummy_character() -> objects::Character {
        objects::Character {
            tick: 0,
            x: 0,
            y: 0,
            vel_x: 0,
            vel_y: 0,
            angle: 0,
            direction: 0,
            jumped: 0,
            hooked_player: -1,
            hook_state: 0,
            hook_tick: 0,
            hook_x: 0,
            hook_y: 0,
            hook_dx: 0,
            hook_dy: 0,
            player_flags: 1,
            health: 10,
            armor: 0,
            ammo_count: -1,
            weapon: 1,
            emote: 0,
            attack_tick: 0,
        }
    }
}
