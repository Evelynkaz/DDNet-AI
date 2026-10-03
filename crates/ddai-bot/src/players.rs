//! The player table: per client id the name, clan, team, AFK/paused/spectator flags and the list
//! flags, updated from each snapshot's `PlayerInfo`/`ClientInfo`/`DDNetPlayer` items.
//!
//! Real names live here only to be matched against the lists and, with `--debug-names`, in the
//! local-only debug log. Everything else — logs, telemetry, the web bridge — goes through
//! [`PlayerTable::tag`]: the client id plus a short **salted hash** of the folded name
//! (`c12-9f3a01bc`), so a shared log neither shows nicknames nor lets one be looked up by hash
//! (the salt is random per process unless configured). Nicknames must never reach a committed file
//! (`CLAUDE.md`, D-040).
//!
//! The update is allocation-free while nobody's name changes: strings are compared in place and
//! only copied (into the slot's own buffers) when they differ.

use std::fmt;

use ddai_net::generated::enums::explayerflagflag;
use ddai_net::view::PlayerView;
use sha2::{Digest, Sha256};

use crate::names::fold_name_into;
use crate::relations::{RelationFlags, Relations};

/// `ddai_physics::core::MAX_CLIENTS`.
pub const MAX_CLIENTS: usize = 128;

/// Bytes of the per-process log salt.
pub type Salt = [u8; 16];

/// A random salt from the OS-seeded `RandomState` keys (not a secret, just unguessable enough that
/// hashes in two shared logs cannot be joined).
pub fn random_salt() -> Salt {
    use std::hash::{BuildHasher, Hasher};
    let mut out = [0u8; 16];
    for chunk in out.chunks_mut(8) {
        let v = std::collections::hash_map::RandomState::new().build_hasher().finish();
        chunk.copy_from_slice(&v.to_le_bytes()[..chunk.len()]);
    }
    out
}

/// One client slot.
#[derive(Debug, Clone, Default)]
pub struct PlayerSlot {
    /// Seen in the latest snapshot.
    pub present: bool,
    /// Display name — **local use only** (see the module docs).
    pub name: String,
    pub clan: String,
    /// [`crate::names::fold_name`] of `name` / `clan`.
    pub name_key: String,
    pub clan_key: String,
    pub team: i32,
    pub latency_ms: i32,
    pub score: i32,
    /// `DDNetPlayer::flags` (`AFK`/`PAUSED`/`SPEC`); 0 when the server sent none.
    pub ex_flags: i32,
    pub flags: RelationFlags,
    /// Bumped whenever the slot starts describing someone else (joined, renamed, left) — the
    /// activity clock restarts its record then (the TS kept records per `name`).
    pub generation: u32,
    tag: [u8; 8],
    rel_version: u64,
}

impl PlayerSlot {
    /// `serverAfk` (`liveWorld.ts`, flags `AFK`).
    pub fn server_afk(&self) -> bool {
        self.ex_flags & explayerflagflag::AFK != 0
    }

    /// `spectating`: the DDNet `SPEC` flag, or the classic spectator team `-1`.
    pub fn spectating(&self) -> bool {
        self.ex_flags & explayerflagflag::SPEC != 0 || self.team == -1
    }

    /// `notPlaying`: paused or spectating (`liveWorld.ts`, `PAUSED|SPEC`).
    pub fn not_playing(&self) -> bool {
        self.ex_flags & explayerflagflag::PAUSED != 0 || self.spectating()
    }
}

/// `c<id>-<8 hex>` — how a client is named in logs and on the wire to the web unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tag {
    id: u8,
    hash: [u8; 8],
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The bytes are ASCII hex by construction.
        write!(
            f,
            "c{}-{}",
            self.id,
            std::str::from_utf8(&self.hash).unwrap_or("????????")
        )
    }
}

/// All 128 slots.
pub struct PlayerTable {
    slots: Vec<PlayerSlot>,
    salt: Salt,
    /// Our own client id, from the last snapshot (`PlayerInfo::local`).
    own_id: Option<i32>,
    scratch_key: String,
}

impl PlayerTable {
    pub fn new(salt: Salt) -> Self {
        PlayerTable {
            slots: vec![PlayerSlot::default(); MAX_CLIENTS],
            salt,
            own_id: None,
            scratch_key: String::new(),
        }
    }

    /// The slot of `id` (`None` for an id outside `0..128`).
    pub fn get(&self, id: i32) -> Option<&PlayerSlot> {
        usize::try_from(id).ok().and_then(|i| self.slots.get(i))
    }

    pub fn own_id(&self) -> Option<i32> {
        self.own_id
    }

    /// The loggable identity of `id`.
    pub fn tag(&self, id: i32) -> Tag {
        let hash = self.get(id).map_or([b'?'; 8], |s| s.tag);
        Tag {
            id: u8::try_from(id.clamp(0, 255)).unwrap_or(0),
            hash,
        }
    }

    /// Re-reads the lists into the flags of every present slot (after `relations` changed).
    pub fn refresh_flags(&mut self, relations: &Relations) {
        for slot in &mut self.slots {
            if slot.present {
                slot.flags = relations.flags_for(&slot.name_key, &slot.clan_key);
                slot.rel_version = relations.version();
            }
        }
    }

    /// Applies one snapshot's player list. Returns true when the roster changed (someone joined,
    /// left or was renamed) — the web bridge re-sends its `players` message then.
    pub fn update(&mut self, players: &[PlayerView], relations: &Relations) -> bool {
        let mut changed = false;
        let mut seen = [false; MAX_CLIENTS];
        self.own_id = None;
        for p in players {
            let Some(idx) = usize::try_from(p.id).ok().filter(|&i| i < MAX_CLIENTS) else {
                continue;
            };
            seen[idx] = true;
            if p.info.local == 1 {
                self.own_id = Some(p.id);
            }
            let (name, clan) = p
                .client_info
                .as_ref()
                .map_or(("", ""), |c| (c.name.as_str(), c.clan.as_str()));
            let slot = &mut self.slots[idx];
            let new_person = !slot.present || slot.name != name || slot.clan != clan;
            if new_person {
                slot.name.clear();
                slot.name.push_str(name);
                slot.clan.clear();
                slot.clan.push_str(clan);
                fold_name_into(name, &mut slot.name_key);
                fold_name_into(clan, &mut slot.clan_key);
                slot.generation = slot.generation.wrapping_add(1);
                slot.tag = hash_tag(&self.salt, &slot.name_key, &mut self.scratch_key);
                slot.flags = relations.flags_for(&slot.name_key, &slot.clan_key);
                slot.rel_version = relations.version();
                changed = true;
            } else if slot.rel_version != relations.version() {
                slot.flags = relations.flags_for(&slot.name_key, &slot.clan_key);
                slot.rel_version = relations.version();
            }
            slot.present = true;
            slot.team = p.info.team;
            slot.latency_ms = p.info.latency;
            slot.score = p.info.score;
            slot.ex_flags = p.ddnet.map_or(0, |d| d.flags);
        }
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            if slot.present && !seen[idx] {
                slot.present = false;
                slot.generation = slot.generation.wrapping_add(1);
                slot.flags = RelationFlags::default();
                changed = true;
            }
        }
        changed
    }

    /// Forgets everyone (map change, reconnect).
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            if slot.present {
                slot.present = false;
                slot.generation = slot.generation.wrapping_add(1);
                slot.flags = RelationFlags::default();
            }
        }
        self.own_id = None;
    }

    /// `text` with every present player's name and clan replaced by that player's tag (`c<id>-<hash>`), matched case-insensitively
    /// (a 1-2 character name only as a whole word, so `a` does not eat every letter). For free-form text from the server (a kick or
    /// disconnect reason: a moderator may type a nickname into it) before it reaches a log.
    pub fn redact(&self, text: &str) -> String {
        let fold = |c: char| c.to_lowercase().next().unwrap_or(c);
        let mut needles: Vec<(Vec<char>, String)> = Vec::new();
        for (id, slot) in self.present() {
            let tag = self.tag(id).to_string();
            for s in [slot.name.trim(), slot.clan.trim()] {
                if !s.is_empty() {
                    needles.push((s.chars().map(fold).collect(), tag.clone()));
                }
            }
        }
        needles.sort_by_key(|(n, _)| std::cmp::Reverse(n.len()));
        let chars: Vec<char> = text.chars().collect();
        let folded: Vec<char> = chars.iter().map(|&c| fold(c)).collect();
        let word = |c: Option<&char>| c.is_some_and(|c| c.is_alphanumeric());
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        'outer: while i < chars.len() {
            for (needle, tag) in &needles {
                let end = i + needle.len();
                if end <= chars.len()
                    && folded[i..end] == needle[..]
                    && (needle.len() > 2
                        || (!word(i.checked_sub(1).and_then(|j| chars.get(j))) && !word(chars.get(end))))
                {
                    out.push_str(tag);
                    i = end;
                    continue 'outer;
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }

    /// Present slots, ascending id.
    pub fn present(&self) -> impl Iterator<Item = (i32, &PlayerSlot)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.present)
            .map(|(i, s)| (i as i32, s))
    }
}

fn hash_tag(salt: &Salt, name_key: &str, _scratch: &mut String) -> [u8; 8] {
    let mut h = Sha256::new();
    h.update(salt);
    h.update(name_key.as_bytes());
    let digest = h.finalize();
    let mut out = [0u8; 8];
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, b) in digest.iter().take(4).enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 15) as usize];
    }
    out
}

#[cfg(test)]
pub(crate) mod test_support {
    use ddai_net::generated::objects::{ClientInfo, DDNetPlayer, PlayerInfo};
    use ddai_net::view::PlayerView;

    /// A `PlayerView` for tests.
    pub fn player(id: i32, name: &str, clan: &str, local: bool, team: i32, ex_flags: Option<i32>) -> PlayerView {
        PlayerView {
            id,
            info: PlayerInfo {
                local: i32::from(local),
                client_id: id,
                team,
                score: 0,
                latency: 20,
            },
            client_info: Some(ClientInfo {
                name: name.to_string(),
                clan: clan.to_string(),
                country: -1,
                skin: "default".to_string(),
                use_custom_color: 0,
                color_body: 0,
                color_feet: 0,
            }),
            ddnet: ex_flags.map(|flags| DDNetPlayer {
                flags,
                auth_level: 0,
                finish_time_seconds: 0,
                finish_time_millis: 0,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::player;
    use super::*;
    use crate::relations::ListKind;

    #[test]
    fn free_text_loses_every_known_name_and_clan_to_the_tag() {
        let mut t = PlayerTable::new([7; 16]);
        t.update(
            &[
                player(0, "Muha", "Neuroset", true, 0, None),
                player(3, "Bob", "ХАОС", false, 0, None),
                player(5, "a", "", false, 0, None),
            ],
            &Relations::new(),
        );
        let (bob, own) = (t.tag(3).to_string(), t.tag(0).to_string());
        let out = t.redact("banned BOB (clan хаос) by muha: a bad day, not Bobby-free");
        assert!(
            !out.to_lowercase().contains("bob ") && !out.contains("хаос") && !out.to_lowercase().contains("muha"),
            "{out}"
        );
        assert!(
            out.starts_with(&format!("banned {bob} (clan {bob}) by {own}: ")),
            "{out}"
        );
        assert!(
            out.contains(&format!("{} bad day", t.tag(5))),
            "the one-letter name as a whole word: {out}"
        );
        assert!(
            out.contains("bad day, not"),
            "the 'a' inside words is left alone: {out}"
        );
        assert_eq!(
            t.redact("Server shutdown"),
            "Server shutdown",
            "ordinary reasons are untouched"
        );
        assert_eq!(t.redact(""), "");
        assert_eq!(
            PlayerTable::new([7; 16]).redact("Bob"),
            "Bob",
            "nobody known: nothing to replace"
        );
    }

    #[test]
    fn flags_follow_the_lists_and_names_fold() {
        let mut rel = Relations::new();
        rel.add(ListKind::Friend, "pal");
        rel.add(ListKind::ClanWar, "bad");
        let mut t = PlayerTable::new([1; 16]);
        assert!(t.update(
            &[
                player(0, "Me", "", true, 0, None),
                player(1, " Pal ", "", false, 0, None),
                player(2, "(1)pal", "", false, 0, None),
                player(3, "palace", "Bad", false, 0, None),
            ],
            &rel
        ));
        assert_eq!(t.own_id(), Some(0));
        assert!(t.get(1).unwrap().flags.friend);
        assert!(
            t.get(2).unwrap().flags.friend,
            "duplicate-prefixed copy of a listed name"
        );
        assert!(!t.get(3).unwrap().flags.friend, "exact, not substring");
        assert!(t.get(3).unwrap().flags.clan_war);
    }

    #[test]
    fn a_list_change_reaches_existing_players() {
        let mut rel = Relations::new();
        let mut t = PlayerTable::new([1; 16]);
        let ps = [player(0, "Me", "", true, 0, None), player(1, "Foe", "", false, 0, None)];
        t.update(&ps, &rel);
        assert!(!t.get(1).unwrap().flags.war);
        rel.add(ListKind::War, "foe");
        assert!(!t.update(&ps, &rel), "no roster change");
        assert!(t.get(1).unwrap().flags.war, "but the flag followed the list");
    }

    #[test]
    fn ex_flags_and_spectator_team_decide_out_of_game() {
        let rel = Relations::new();
        let mut t = PlayerTable::new([1; 16]);
        t.update(
            &[
                player(0, "a", "", true, 0, Some(explayerflagflag::AFK)),
                player(1, "b", "", false, 0, Some(explayerflagflag::PAUSED)),
                player(2, "c", "", false, 0, Some(explayerflagflag::SPEC)),
                player(3, "d", "", false, -1, None),
                player(4, "e", "", false, 0, None),
            ],
            &rel,
        );
        let s = |i| t.get(i).unwrap();
        assert!(s(0).server_afk() && !s(0).not_playing());
        assert!(s(1).not_playing() && !s(1).spectating());
        assert!(s(2).spectating() && s(2).not_playing());
        assert!(s(3).spectating(), "classic spectator team");
        assert!(!s(4).not_playing() && !s(4).server_afk());
    }

    #[test]
    fn renames_leaves_and_joins_bump_the_generation_and_report_a_change() {
        let rel = Relations::new();
        let mut t = PlayerTable::new([1; 16]);
        assert!(t.update(&[player(1, "a", "", false, 0, None)], &rel));
        let g = t.get(1).unwrap().generation;
        assert!(!t.update(&[player(1, "a", "", false, 0, None)], &rel), "same roster");
        assert_eq!(t.get(1).unwrap().generation, g);
        assert!(t.update(&[player(1, "b", "", false, 0, None)], &rel), "renamed");
        assert_ne!(t.get(1).unwrap().generation, g);
        assert!(t.update(&[], &rel), "left");
        assert!(!t.get(1).unwrap().present);
    }

    #[test]
    fn tags_hide_names_and_depend_on_the_salt() {
        let rel = Relations::new();
        let ps = [player(7, "Secret Nick", "", false, 0, None)];
        let mut a = PlayerTable::new([1; 16]);
        let mut b = PlayerTable::new([2; 16]);
        a.update(&ps, &rel);
        b.update(&ps, &rel);
        let (ta, tb) = (a.tag(7).to_string(), b.tag(7).to_string());
        assert!(ta.starts_with("c7-") && ta.len() == 3 + 8, "{ta}");
        assert!(!ta.to_lowercase().contains("secret") && !ta.contains("nick"));
        assert_ne!(ta, tb, "a different salt gives a different hash");
        let again = PlayerTable::new([1; 16]);
        let mut again = again;
        again.update(&ps, &rel);
        assert_eq!(again.tag(7).to_string(), ta, "stable for a salt");
    }

    #[test]
    fn steady_state_update_allocates_nothing() {
        let rel = Relations::new();
        let mut t = PlayerTable::new([1; 16]);
        let ps: Vec<_> = (0..8)
            .map(|i| player(i, &format!("player{i}"), "clan", i == 0, 0, Some(0)))
            .collect();
        t.update(&ps, &rel);
        let info = allocation_counter::measure(|| {
            for _ in 0..100 {
                std::hint::black_box(t.update(&ps, &rel));
            }
        });
        assert_eq!(info.count_total, 0, "{info:?}");
    }
}
