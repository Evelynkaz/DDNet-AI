//! A typed view over a decoded [`crate::snapshot::Snapshot`] — task 2.2b acceptance criterion 3's
//! "typed view API (iterate characters with DDNetCharacter merged, players, projectiles, lasers,
//! game info, etc.)".
//!
//! This is entirely our own code (no direct DDNet/libtw2 port): DDNet's own client does the
//! equivalent merging ad hoc, scattered across `gameclient.cpp`'s many `GameClient()->m_*`
//! components (`m_Snap.m_paPlayerInfos`, `m_Snap.m_aCharacters[i].m_HasExtendedData`, ...) rather
//! than as one reusable API — there is nothing to port here, only DDNet's *semantics* (which
//! item(s) make up "a player"/"a character") to reproduce.
//!
//! Ex (UUID) object resolution needs a name registry; this module keeps its own, built once from
//! [`crate::generated::objects::EX_NAMES`] — independent of `crate::message`'s registry (which
//! covers *messages*, a disjoint set of UUID names) since nothing here needs the two combined.

use crate::generated::objects;
use crate::snapshot::{Snapshot, SnapshotItem};
use crate::uuid::UuidRegistry;
use std::sync::OnceLock;

fn ex_object_registry() -> &'static UuidRegistry {
    static REGISTRY: OnceLock<UuidRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| UuidRegistry::from_names(objects::EX_NAMES))
}

/// The UUID name of `item`'s type, or `None` if `item` is a type-0 descriptor item itself
/// (`internal_type() == 0`, not a "real" ex item), its type is not an ex type at all
/// (`internal_type() < OFFSET_UUID_TYPE`), or the snapshot has no matching descriptor / the UUID
/// is not one this crate knows.
fn ex_name(snap: &Snapshot, item: &SnapshotItem) -> Option<&'static str> {
    if item.internal_type() == 0 {
        return None;
    }
    let uuid = snap.ex_type_uuid(item.internal_type())?;
    let registry = ex_object_registry();
    let id = registry.lookup(uuid)?;
    registry.name(id)
}

/// Finds the (at most one) ex item of type `name` with snapshot id `id` — used to find e.g. "the
/// `DDNetCharacter` for client id 3", whose *internal* type is assigned dynamically per snapshot
/// (see `crate::snapshot::Snapshot::ex_type_uuid`), so it cannot be looked up by a fixed key.
fn find_ex_item<'a>(snap: &'a Snapshot, name: &str, id: i32) -> Option<&'a SnapshotItem> {
    snap.items
        .iter()
        .find(|it| it.id() == id && ex_name(snap, it) == Some(name))
}

fn decode_ex_item<T>(snap: &Snapshot, name: &str, id: i32, decode: impl Fn(&[i32]) -> Option<(T, u32)>) -> Option<T> {
    let item = find_ex_item(snap, name, id)?;
    decode(&item.data).map(|(v, _corrections)| v)
}

/// A `Character` (task 2.2a/b's core per-tick physics state), with its `DDNetCharacter`
/// extension merged in when the server sent one (it always does for a real DDNet 20.x server;
/// `None` only against an older/non-DDNet 0.6 peer, out of this crate's scope but still handled
/// tolerantly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterView {
    pub id: i32,
    pub character: objects::Character,
    pub ddnet: Option<objects::DDNetCharacter>,
}

/// A player slot: `PlayerInfo` (always present for a connected client id) plus `ClientInfo`
/// (name/clan/skin) and the DDNet `DDNetPlayer` extension, when present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerView {
    pub id: i32,
    pub info: objects::PlayerInfo,
    pub client_info: Option<objects::ClientInfo>,
    pub ddnet: Option<objects::DDNetPlayer>,
}

/// A projectile: the server sends exactly one of these three shapes per projectile (never more
/// than one for the same id — which shape depends on what the server/mod supports), so this is an
/// alternative, not a merge like [`CharacterView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectileView {
    Legacy(objects::Projectile),
    DDRace(objects::DDRaceProjectile),
    DDNet(objects::DDNetProjectile),
}

/// A laser: `DDNetLaser` (used by every real DDNet 20.x server) or the legacy `Laser` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaserView {
    Legacy(objects::Laser),
    DDNet(objects::DDNetLaser),
}

/// A pickup: `DDNetPickup` or the legacy `Pickup` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickupView {
    Legacy(objects::Pickup),
    DDNet(objects::DDNetPickup),
}

/// A read-only typed view over one decoded [`Snapshot`]. Cheap to construct (borrows the
/// snapshot; does no work up front) — every accessor scans `snap.items` on demand, which is
/// perfectly fine for a snapshot's realistic size (at most [`crate::snapshot::MAX_ITEMS`] = 1024
/// items).
pub struct View<'a> {
    snap: &'a Snapshot,
}

impl<'a> View<'a> {
    pub fn new(snap: &'a Snapshot) -> Self {
        View { snap }
    }

    /// The underlying snapshot this view was built from.
    pub fn snapshot(&self) -> &'a Snapshot {
        self.snap
    }

    /// `NETOBJTYPE_GAMEINFO`, id 0 — global round/warmup/score-limit state. At most one per
    /// snapshot; `None` if absent (e.g. a snapshot taken before the game info was first sent).
    pub fn game_info(&self) -> Option<objects::GameInfo> {
        self.snap
            .find(objects::GameInfo::ID, 0)
            .and_then(|it| objects::GameInfo::decode(&it.data))
            .map(|(v, _)| v)
    }

    /// `NETOBJTYPE_GAMEDATA`, id 0 — flag carriers/team scores (CTF-only fields; still sent, just
    /// zeroed, in non-CTF modes).
    pub fn game_data(&self) -> Option<objects::GameData> {
        self.snap
            .find(objects::GameData::ID, 0)
            .and_then(|it| objects::GameData::decode(&it.data))
            .map(|(v, _)| v)
    }

    /// The DDNet `gameinfo@netobj.ddnet.tw` extension (mode flags, team-size limits), id 0.
    pub fn game_info_ex(&self) -> Option<objects::GameInfoEx> {
        decode_ex_item(self.snap, "gameinfo@netobj.ddnet.tw", 0, objects::GameInfoEx::decode)
    }

    /// Every `Character` in this snapshot, merged with its `DDNetCharacter` extension when
    /// present, ordered by ascending client id (deterministic iteration, matching
    /// [`Self::players`]).
    pub fn characters(&self) -> Vec<CharacterView> {
        let mut out: Vec<CharacterView> = self
            .snap
            .items
            .iter()
            .filter(|it| it.internal_type() == objects::Character::ID)
            .filter_map(|it| {
                let (character, _) = objects::Character::decode(&it.data)?;
                let id = it.id();
                let ddnet = decode_ex_item(
                    self.snap,
                    "character@netobj.ddnet.tw",
                    id,
                    objects::DDNetCharacter::decode,
                );
                Some(CharacterView { id, character, ddnet })
            })
            .collect();
        out.sort_by_key(|c| c.id);
        out
    }

    /// The one `Character` (merged with `DDNetCharacter`, if present) at client id `id`, if any.
    pub fn character(&self, id: i32) -> Option<CharacterView> {
        let item = self.snap.find(objects::Character::ID, id)?;
        let (character, _) = objects::Character::decode(&item.data)?;
        let ddnet = decode_ex_item(
            self.snap,
            "character@netobj.ddnet.tw",
            id,
            objects::DDNetCharacter::decode,
        );
        Some(CharacterView { id, character, ddnet })
    }

    /// Every player slot (`PlayerInfo`, with `ClientInfo`/`DDNetPlayer` merged in when present),
    /// ordered by ascending client id.
    pub fn players(&self) -> Vec<PlayerView> {
        let mut out: Vec<PlayerView> = self
            .snap
            .items
            .iter()
            .filter(|it| it.internal_type() == objects::PlayerInfo::ID)
            .filter_map(|it| {
                let (info, _) = objects::PlayerInfo::decode(&it.data)?;
                let id = it.id();
                let client_info = self
                    .snap
                    .find(objects::ClientInfo::ID, id)
                    .and_then(|ci| objects::ClientInfo::decode(&ci.data))
                    .map(|(v, _)| v);
                let ddnet = decode_ex_item(self.snap, "player@netobj.ddnet.tw", id, objects::DDNetPlayer::decode);
                Some(PlayerView {
                    id,
                    info,
                    client_info,
                    ddnet,
                })
            })
            .collect();
        out.sort_by_key(|p| p.id);
        out
    }

    /// Every projectile in this snapshot (whichever of the three wire shapes the server used —
    /// see [`ProjectileView`]), as `(id, view)` pairs.
    pub fn projectiles(&self) -> Vec<(i32, ProjectileView)> {
        let mut out = Vec::new();
        for it in &self.snap.items {
            if it.internal_type() == objects::Projectile::ID {
                if let Some((v, _)) = objects::Projectile::decode(&it.data) {
                    out.push((it.id(), ProjectileView::Legacy(v)));
                }
            } else if let Some(name) = ex_name(self.snap, it) {
                match name {
                    "projectile@netobj.ddnet.tw" => {
                        if let Some((v, _)) = objects::DDRaceProjectile::decode(&it.data) {
                            out.push((it.id(), ProjectileView::DDRace(v)));
                        }
                    }
                    "ddnet-projectile@netobj.ddnet.tw" => {
                        if let Some((v, _)) = objects::DDNetProjectile::decode(&it.data) {
                            out.push((it.id(), ProjectileView::DDNet(v)));
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Every laser in this snapshot — see [`LaserView`].
    pub fn lasers(&self) -> Vec<(i32, LaserView)> {
        let mut out = Vec::new();
        for it in &self.snap.items {
            if it.internal_type() == objects::Laser::ID {
                if let Some((v, _)) = objects::Laser::decode(&it.data) {
                    out.push((it.id(), LaserView::Legacy(v)));
                }
            } else if ex_name(self.snap, it) == Some("laser@netobj.ddnet.tw")
                && let Some((v, _)) = objects::DDNetLaser::decode(&it.data)
            {
                out.push((it.id(), LaserView::DDNet(v)));
            }
        }
        out
    }

    /// Every pickup in this snapshot — see [`PickupView`].
    pub fn pickups(&self) -> Vec<(i32, PickupView)> {
        let mut out = Vec::new();
        for it in &self.snap.items {
            if it.internal_type() == objects::Pickup::ID {
                if let Some((v, _)) = objects::Pickup::decode(&it.data) {
                    out.push((it.id(), PickupView::Legacy(v)));
                }
            } else if ex_name(self.snap, it) == Some("pickup@netobj.ddnet.tw")
                && let Some((v, _)) = objects::DDNetPickup::decode(&it.data)
            {
                out.push((it.id(), PickupView::DDNet(v)));
            }
        }
        out
    }

    /// Every item this view has no specific accessor for (an unknown numbered type, an
    /// unresolved/unregistered UUID, or a known ex type with no wrapper method above) — task
    /// acceptance criterion 3's "unknown object types are kept as raw items": nothing in
    /// [`Snapshot`] ever drops them, and this lets a caller enumerate them for diagnostics
    /// (e.g. the real-traffic report's per-type counts, `tests/real_traffic.rs`).
    pub fn describe_item(&self, item: &SnapshotItem) -> ItemKind {
        if item.internal_type() == 0 {
            return ItemKind::UuidTypeDescriptor;
        }
        if item.internal_type() < objects::PlayerInput::ID {
            return ItemKind::Unknown; // internal_type 0 handled above; nothing is type < 1 here otherwise
        }
        if let Some(name) = ex_name(self.snap, item) {
            ItemKind::Ex(name)
        } else if item.internal_type() >= crate::snapshot::OFFSET_UUID_TYPE {
            ItemKind::UnresolvedEx
        } else {
            ItemKind::Numbered(item.internal_type())
        }
    }
}

/// What [`View::describe_item`] classified a raw item as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    /// A `type == 0` (`NETOBJTYPE_EX`) descriptor item itself, not a "real" item.
    UuidTypeDescriptor,
    /// A known non-ex type (its numeric id — may or may not be one of the 20 this crate
    /// specifically decodes; any numbered type outside that set is still "known" in the sense
    /// that it is a real, assigned, non-ex `NETOBJTYPE_*`, just not one this task's generator
    /// emitted a struct for, which cannot happen for DDNet 20.1 itself but is meaningful for a
    /// future protocol version's server sending a type only *it* knows).
    Numbered(i32),
    /// A UUID (ex) type this crate's registry resolved to a known name.
    Ex(&'static str),
    /// A UUID (ex) type whose descriptor item is missing/malformed, or whose UUID this crate does
    /// not have registered — genuinely unknown, kept raw.
    UnresolvedEx,
    /// Anything else (should not occur for a well-formed snapshot; defensive fallback).
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::SnapshotItem;
    use crate::uuid::calculate_uuid;

    fn ex_type_descriptor(internal_type: i32, name: &str) -> SnapshotItem {
        let uuid = calculate_uuid(name);
        let mut ints = [0i32; 4];
        for (i, chunk) in uuid.0.chunks(4).enumerate() {
            ints[i] = u32::from_be_bytes(chunk.try_into().unwrap()) as i32;
        }
        SnapshotItem {
            key: internal_type,
            data: ints.to_vec(),
        }
    }

    #[test]
    fn character_merges_ddnet_extension_by_id() {
        let character = objects::Character {
            tick: 100,
            x: 10,
            y: 20,
            vel_x: 0,
            vel_y: 0,
            angle: 0,
            direction: 0,
            jumped: 0,
            hooked_player: -1,
            hook_state: -1,
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
        };
        let mut char_data = vec![
            character.tick,
            character.x,
            character.y,
            character.vel_x,
            character.vel_y,
            character.angle,
            character.direction,
            character.jumped,
            character.hooked_player,
            character.hook_state,
            character.hook_tick,
            character.hook_x,
            character.hook_y,
            character.hook_dx,
            character.hook_dy,
        ];
        char_data.extend_from_slice(&[
            character.player_flags,
            character.health,
            character.armor,
            character.ammo_count,
            character.weapon,
            character.emote,
            character.attack_tick,
        ]);

        let ddnet_data = vec![0, 0, 2, -1, 0, -1, -1, -1, 5, 6, -1];

        let snap = Snapshot {
            items: vec![
                ex_type_descriptor(0x7fff, "character@netobj.ddnet.tw"),
                SnapshotItem {
                    key: (objects::Character::ID << 16) | 3,
                    data: char_data,
                },
                SnapshotItem {
                    key: (0x7fff << 16) | 3,
                    data: ddnet_data,
                },
            ],
        };
        let view = View::new(&snap);
        let got = view.character(3).expect("character 3 present");
        assert_eq!(got.id, 3);
        assert_eq!(got.character, character);
        assert!(got.ddnet.is_some());
        assert_eq!(got.ddnet.unwrap().target_x, 5);

        assert_eq!(view.character(4), None);
        assert_eq!(view.characters().len(), 1);
    }

    #[test]
    fn character_without_ddnet_extension_still_decodes() {
        let snap = Snapshot {
            items: vec![SnapshotItem {
                key: objects::Character::ID << 16,
                data: vec![0; 22],
            }],
        };
        let view = View::new(&snap);
        let got = view.character(0).expect("present");
        assert_eq!(got.ddnet, None);
    }

    #[test]
    fn describe_item_classifies_correctly() {
        let snap = Snapshot {
            items: vec![
                ex_type_descriptor(0x7fff, "character@netobj.ddnet.tw"),
                SnapshotItem {
                    key: objects::Flag::ID << 16,
                    data: vec![0, 0, 0],
                },
                SnapshotItem {
                    key: (0x7fff << 16) | 1,
                    data: vec![0; 11],
                },
                SnapshotItem {
                    key: (0x7ffe << 16) | 2, // no descriptor for 0x7ffe at all
                    data: vec![1, 2, 3],
                },
            ],
        };
        let view = View::new(&snap);
        assert_eq!(view.describe_item(&snap.items[0]), ItemKind::UuidTypeDescriptor);
        assert_eq!(
            view.describe_item(&snap.items[1]),
            ItemKind::Numbered(objects::Flag::ID)
        );
        assert_eq!(
            view.describe_item(&snap.items[2]),
            ItemKind::Ex("character@netobj.ddnet.tw")
        );
        assert_eq!(view.describe_item(&snap.items[3]), ItemKind::UnresolvedEx);
    }

    #[test]
    fn empty_snapshot_view_never_panics() {
        let snap = Snapshot::empty();
        let view = View::new(&snap);
        assert!(view.characters().is_empty());
        assert!(view.players().is_empty());
        assert!(view.projectiles().is_empty());
        assert!(view.lasers().is_empty());
        assert!(view.pickups().is_empty());
        assert_eq!(view.game_info(), None);
        assert_eq!(view.game_data(), None);
        assert_eq!(view.game_info_ex(), None);
    }

    #[test]
    fn player_merges_client_info_and_ddnet_extension() {
        let name_ints = crate::intstr::str_to_ints("Muha", 4);
        let clan_ints = crate::intstr::str_to_ints("", 3);
        let skin_ints = crate::intstr::str_to_ints("default", 6);
        let mut client_info_data = name_ints;
        client_info_data.extend_from_slice(&clan_ints);
        client_info_data.push(0); // country
        client_info_data.extend_from_slice(&skin_ints);
        client_info_data.extend_from_slice(&[0, 0, 0]); // use_custom_color, color_body, color_feet

        let snap = Snapshot {
            items: vec![
                SnapshotItem {
                    key: objects::PlayerInfo::ID << 16,
                    data: vec![1, 0, 0, 0, 0], // local=1, client_id=0, team=0
                },
                SnapshotItem {
                    key: objects::ClientInfo::ID << 16,
                    data: client_info_data,
                },
            ],
        };
        let view = View::new(&snap);
        let players = view.players();
        assert_eq!(players.len(), 1);
        assert_eq!(players[0].client_info.as_ref().unwrap().name, "Muha");
        assert_eq!(players[0].ddnet, None);
    }
}
