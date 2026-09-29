//! rec v1: the observer recorder's own binary format (task 8.4a acceptance criterion 2;
//! documented in Russian in `docs/formats.md` §16). This module owns the in-memory shape and the
//! encode/decode of one [`Header`] and one [`Frame`] at a time — chunking, zstd compression, and
//! sha256 checksums are [`crate::writer`]/[`crate::reader`]'s job, one layer up (exactly the same
//! split `ddai-trace`'s `rawmap`/`scenario`/`trace` formats use between "what a record means" and
//! "how records are framed on disk").
//!
//! Every record here is a *decoded, typed* snapshot of state (`ddai_net::view::CharacterView`/
//! `PlayerView`'s own fields), not raw wire bytes — the recorder decodes once, at record time
//! (`ddai_net`/`ddai_client` already did the hard part), and stores the result; reconstruction
//! (`crate::reconstruct`) reads exactly what was stored. A player unchanged between two snapshots
//! is still written out in full on both — no delta-encoding at this layer (unlike DDNet's own
//! wire protocol) — because zstd already collapses that redundancy extremely well in practice
//! (see the crate's BUILD REPORT for the measured bytes/hour), and a self-contained
//! "one frame, one full state" record is far simpler to decode offline than a differencing scheme
//! would be.

use crate::binio::{DecodeError, Reader, Writer};
use ddai_net::generated::objects;
use ddai_net::tuning::TuneParams;

/// `"DAR1"` (DDai Auto Recorder, v1) — the first 4 bytes of every rec v1 file.
pub const MAGIC: [u8; 4] = *b"DAR1";
pub const VERSION: u32 = 1;

/// The recording's header — task acceptance criterion 2's "server address, map name + sha256,
/// client version, start time". [`Header::observer_nick`] is this crate's own addition (not
/// explicitly asked for, but free given everything else here and useful for telling recordings
/// apart without also opening the body) — noted here, not hidden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub server_address: String,
    pub map_name: String,
    pub map_sha256: [u8; 32],
    pub client_version: String,
    /// Milliseconds since the Unix epoch (UTC) — `SystemTime::now()` at the moment recording
    /// started, not the server's own clock (the recorder has no access to that).
    pub start_time_unix_ms: u64,
    pub observer_nick: String,
}

impl Header {
    pub fn encode(&self, w: &mut Writer) {
        w.push_raw(&MAGIC);
        w.push_u32(VERSION);
        w.push_string(&self.server_address);
        w.push_string(&self.map_name);
        w.push_bytes32(&self.map_sha256);
        w.push_string(&self.client_version);
        w.push_u64(self.start_time_unix_ms);
        w.push_string(&self.observer_nick);
    }

    pub fn decode(r: &mut Reader) -> Result<Self, FormatError> {
        let magic = r.read_raw(4)?;
        if magic != MAGIC {
            return Err(FormatError::BadMagic(magic.to_vec()));
        }
        let version = r.read_u32()?;
        if version != VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }
        Ok(Header {
            server_address: r.read_string()?,
            map_name: r.read_string()?,
            map_sha256: r.read_bytes32()?,
            client_version: r.read_string()?,
            start_time_unix_ms: r.read_u64()?,
            observer_nick: r.read_string()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FormatError {
    #[error("not a rec v1 file: bad magic {0:02x?}")]
    BadMagic(Vec<u8>),
    #[error("unsupported rec version {0} (this reader only understands version {VERSION})")]
    UnsupportedVersion(u32),
    #[error("unknown frame tag {0}")]
    UnknownFrameTag(u8),
    #[error("unknown game-message kind tag {0}")]
    UnknownGameMessageKind(u8),
    #[error(transparent)]
    Decode(#[from] DecodeError),
}

/// One player's character state in one snapshot — task acceptance criterion 2's "all
/// character/DDNetCharacter ... objects".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterRecord {
    pub id: i32,
    pub character: objects::Character,
    pub ddnet: Option<objects::DDNetCharacter>,
}

/// One player's info/identity state in one snapshot — task acceptance criterion 2's "... player
/// info objects". Nicknames (`client_info.name`/`.clan`) are stored as-is here (local-only
/// storage per the task/D-038); [`crate::anonymize`] is the only place that ever strips them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerRecord {
    pub id: i32,
    pub info: objects::PlayerInfo,
    pub client_info: Option<objects::ClientInfo>,
    pub ddnet: Option<objects::DDNetPlayer>,
}

/// A curated, still-typed subset of `ddai_net::generated::messages::GameMsg` — task acceptance
/// criterion 2's "kill messages, tuning, broadcasts, chat as read-only text". Every other game
/// message this crate's own `SessionEvent::GameMessage`/`ExGameMessage` ever delivers is not
/// dropped (this format never silently discards data) but stored as [`RecordedGameMessage::Other`]
/// — its `Debug` text, not a further-typed shape: DDNet's message set is large (~30 `GameMsg`
/// variants alone) and this task's acceptance criterion only calls out these four kinds by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordedGameMessage {
    Kill {
        killer: i32,
        victim: i32,
        weapon: i32,
        mode_special: i32,
    },
    Broadcast {
        message: String,
    },
    /// Read-only: this crate never *sends* chat (D-007) — this variant only ever holds chat
    /// *received* from the server, exactly like `ddai_client::allowlist`'s own doc comment notes
    /// decoding is fine, only sending is forbidden.
    Chat {
        team: i32,
        client_id: i32,
        message: String,
    },
    Tuning(TuneParams),
    /// `format!("{msg:?}")` of whatever `GameMsg`/`ExGameMsg` this was — see this enum's own docs
    /// for why anything not curated above lands here instead of being dropped.
    Other {
        debug: String,
    },
}

/// One record in a recording's body — task acceptance criterion 2's per-snapshot state plus game
/// messages. [`crate::writer::RecordingWriter`] packs a sequence of these into zstd-compressed
/// chunks; [`crate::reader::RecordingReader`] hands them back out in the same order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// One assembled snapshot — task acceptance criterion 2's "per snapshot: the tick, all
    /// character/DDNetCharacter/player info objects". `tick` is the *snapshot's* tick
    /// (`SessionEvent::Snapshot`'s own field), not `character.tick` (the dead-reckoning tick
    /// embedded in each `Character` — task 2.2b's F4 finding, §13.8): both are kept, the former
    /// as this frame's own field (when this data arrived), the latter inside each
    /// [`CharacterRecord`] (each character's own physics tick) — `crate::reconstruct` uses the
    /// latter for trajectories, exactly like `tests/real_traffic.rs` already does.
    Snapshot {
        tick: i32,
        characters: Vec<CharacterRecord>,
        players: Vec<PlayerRecord>,
    },
    /// A game message — task acceptance criterion 2's second bullet. `tick_hint` is the most
    /// recently observed snapshot tick *at the moment this message arrived* (game messages are
    /// not themselves tick-stamped on the wire), for rough time-ordering against the `Snapshot`
    /// frames around it; never treated as exact.
    GameEvent {
        tick_hint: i32,
        message: RecordedGameMessage,
    },
}

const FRAME_TAG_SNAPSHOT: u8 = 1;
const FRAME_TAG_GAME_EVENT: u8 = 2;

const KIND_KILL: u8 = 1;
const KIND_BROADCAST: u8 = 2;
const KIND_CHAT: u8 = 3;
const KIND_TUNING: u8 = 4;
const KIND_OTHER: u8 = 5;

fn encode_character(w: &mut Writer, c: &objects::Character) {
    w.push_i32(c.tick);
    w.push_i32(c.x);
    w.push_i32(c.y);
    w.push_i32(c.vel_x);
    w.push_i32(c.vel_y);
    w.push_i32(c.angle);
    w.push_i32(c.direction);
    w.push_i32(c.jumped);
    w.push_i32(c.hooked_player);
    w.push_i32(c.hook_state);
    w.push_i32(c.hook_tick);
    w.push_i32(c.hook_x);
    w.push_i32(c.hook_y);
    w.push_i32(c.hook_dx);
    w.push_i32(c.hook_dy);
    w.push_i32(c.player_flags);
    w.push_i32(c.health);
    w.push_i32(c.armor);
    w.push_i32(c.ammo_count);
    w.push_i32(c.weapon);
    w.push_i32(c.emote);
    w.push_i32(c.attack_tick);
}

fn decode_character(r: &mut Reader) -> Result<objects::Character, DecodeError> {
    Ok(objects::Character {
        tick: r.read_i32()?,
        x: r.read_i32()?,
        y: r.read_i32()?,
        vel_x: r.read_i32()?,
        vel_y: r.read_i32()?,
        angle: r.read_i32()?,
        direction: r.read_i32()?,
        jumped: r.read_i32()?,
        hooked_player: r.read_i32()?,
        hook_state: r.read_i32()?,
        hook_tick: r.read_i32()?,
        hook_x: r.read_i32()?,
        hook_y: r.read_i32()?,
        hook_dx: r.read_i32()?,
        hook_dy: r.read_i32()?,
        player_flags: r.read_i32()?,
        health: r.read_i32()?,
        armor: r.read_i32()?,
        ammo_count: r.read_i32()?,
        weapon: r.read_i32()?,
        emote: r.read_i32()?,
        attack_tick: r.read_i32()?,
    })
}

fn encode_ddnet_character(w: &mut Writer, c: &objects::DDNetCharacter) {
    w.push_i32(c.flags);
    w.push_i32(c.freeze_end);
    w.push_i32(c.jumps);
    w.push_i32(c.tele_checkpoint);
    w.push_i32(c.strong_weak_id);
    w.push_i32(c.jumped_total);
    w.push_i32(c.ninja_activation_tick);
    w.push_i32(c.freeze_start);
    w.push_i32(c.target_x);
    w.push_i32(c.target_y);
    w.push_i32(c.tune_zone_override);
}

fn decode_ddnet_character(r: &mut Reader) -> Result<objects::DDNetCharacter, DecodeError> {
    Ok(objects::DDNetCharacter {
        flags: r.read_i32()?,
        freeze_end: r.read_i32()?,
        jumps: r.read_i32()?,
        tele_checkpoint: r.read_i32()?,
        strong_weak_id: r.read_i32()?,
        jumped_total: r.read_i32()?,
        ninja_activation_tick: r.read_i32()?,
        freeze_start: r.read_i32()?,
        target_x: r.read_i32()?,
        target_y: r.read_i32()?,
        tune_zone_override: r.read_i32()?,
    })
}

fn encode_player_info(w: &mut Writer, p: &objects::PlayerInfo) {
    w.push_i32(p.local);
    w.push_i32(p.client_id);
    w.push_i32(p.team);
    w.push_i32(p.score);
    w.push_i32(p.latency);
}

fn decode_player_info(r: &mut Reader) -> Result<objects::PlayerInfo, DecodeError> {
    Ok(objects::PlayerInfo {
        local: r.read_i32()?,
        client_id: r.read_i32()?,
        team: r.read_i32()?,
        score: r.read_i32()?,
        latency: r.read_i32()?,
    })
}

fn encode_client_info(w: &mut Writer, c: &objects::ClientInfo) {
    w.push_string(&c.name);
    w.push_string(&c.clan);
    w.push_i32(c.country);
    w.push_string(&c.skin);
    w.push_i32(c.use_custom_color);
    w.push_i32(c.color_body);
    w.push_i32(c.color_feet);
}

fn decode_client_info(r: &mut Reader) -> Result<objects::ClientInfo, DecodeError> {
    Ok(objects::ClientInfo {
        name: r.read_string()?,
        clan: r.read_string()?,
        country: r.read_i32()?,
        skin: r.read_string()?,
        use_custom_color: r.read_i32()?,
        color_body: r.read_i32()?,
        color_feet: r.read_i32()?,
    })
}

fn encode_ddnet_player(w: &mut Writer, p: &objects::DDNetPlayer) {
    w.push_i32(p.flags);
    w.push_i32(p.auth_level);
    w.push_i32(p.finish_time_seconds);
    w.push_i32(p.finish_time_millis);
}

fn decode_ddnet_player(r: &mut Reader) -> Result<objects::DDNetPlayer, DecodeError> {
    Ok(objects::DDNetPlayer {
        flags: r.read_i32()?,
        auth_level: r.read_i32()?,
        finish_time_seconds: r.read_i32()?,
        finish_time_millis: r.read_i32()?,
    })
}

/// `TuneParams`'s 47 fields, in exactly `tuning.h`'s declaration order (the same order
/// `ddai_net::tuning` itself documents as part of the wire format) — `received` first, as a
/// `u32` (it is a `usize` in memory, but never exceeds 47, so `u32` is exact and portable).
fn encode_tuning(w: &mut Writer, t: &TuneParams) {
    w.push_u32(t.received as u32);
    for v in [
        t.ground_control_speed,
        t.ground_control_accel,
        t.ground_friction,
        t.ground_jump_impulse,
        t.air_jump_impulse,
        t.air_control_speed,
        t.air_control_accel,
        t.air_friction,
        t.hook_length,
        t.hook_fire_speed,
        t.hook_drag_accel,
        t.hook_drag_speed,
        t.gravity,
        t.velramp_start,
        t.velramp_range,
        t.velramp_curvature,
        t.gun_curvature,
        t.gun_speed,
        t.gun_lifetime,
        t.shotgun_curvature,
        t.shotgun_speed,
        t.shotgun_speeddiff,
        t.shotgun_lifetime,
        t.grenade_curvature,
        t.grenade_speed,
        t.grenade_lifetime,
        t.laser_reach,
        t.laser_bounce_delay,
        t.laser_bounce_num,
        t.laser_bounce_cost,
        t.laser_damage,
        t.player_collision,
        t.player_hooking,
        t.jetpack_strength,
        t.shotgun_strength,
        t.explosion_strength,
        t.hammer_strength,
        t.hook_duration,
        t.hammer_fire_delay,
        t.gun_fire_delay,
        t.shotgun_fire_delay,
        t.grenade_fire_delay,
        t.laser_fire_delay,
        t.ninja_fire_delay,
        t.hammer_hit_fire_delay,
        t.ground_elasticity_x,
        t.ground_elasticity_y,
    ] {
        w.push_i32(v);
    }
}

fn decode_tuning(r: &mut Reader) -> Result<TuneParams, DecodeError> {
    let received = r.read_u32()? as usize;
    let mut v = [0i32; ddai_net::tuning::NUM_TUNE_PARAMS];
    for slot in &mut v {
        *slot = r.read_i32()?;
    }
    Ok(TuneParams {
        received,
        ground_control_speed: v[0],
        ground_control_accel: v[1],
        ground_friction: v[2],
        ground_jump_impulse: v[3],
        air_jump_impulse: v[4],
        air_control_speed: v[5],
        air_control_accel: v[6],
        air_friction: v[7],
        hook_length: v[8],
        hook_fire_speed: v[9],
        hook_drag_accel: v[10],
        hook_drag_speed: v[11],
        gravity: v[12],
        velramp_start: v[13],
        velramp_range: v[14],
        velramp_curvature: v[15],
        gun_curvature: v[16],
        gun_speed: v[17],
        gun_lifetime: v[18],
        shotgun_curvature: v[19],
        shotgun_speed: v[20],
        shotgun_speeddiff: v[21],
        shotgun_lifetime: v[22],
        grenade_curvature: v[23],
        grenade_speed: v[24],
        grenade_lifetime: v[25],
        laser_reach: v[26],
        laser_bounce_delay: v[27],
        laser_bounce_num: v[28],
        laser_bounce_cost: v[29],
        laser_damage: v[30],
        player_collision: v[31],
        player_hooking: v[32],
        jetpack_strength: v[33],
        shotgun_strength: v[34],
        explosion_strength: v[35],
        hammer_strength: v[36],
        hook_duration: v[37],
        hammer_fire_delay: v[38],
        gun_fire_delay: v[39],
        shotgun_fire_delay: v[40],
        grenade_fire_delay: v[41],
        laser_fire_delay: v[42],
        ninja_fire_delay: v[43],
        hammer_hit_fire_delay: v[44],
        ground_elasticity_x: v[45],
        ground_elasticity_y: v[46],
    })
}

fn encode_game_message(w: &mut Writer, m: &RecordedGameMessage) {
    match m {
        RecordedGameMessage::Kill {
            killer,
            victim,
            weapon,
            mode_special,
        } => {
            w.push_u8(KIND_KILL);
            w.push_i32(*killer);
            w.push_i32(*victim);
            w.push_i32(*weapon);
            w.push_i32(*mode_special);
        }
        RecordedGameMessage::Broadcast { message } => {
            w.push_u8(KIND_BROADCAST);
            w.push_string(message);
        }
        RecordedGameMessage::Chat {
            team,
            client_id,
            message,
        } => {
            w.push_u8(KIND_CHAT);
            w.push_i32(*team);
            w.push_i32(*client_id);
            w.push_string(message);
        }
        RecordedGameMessage::Tuning(t) => {
            w.push_u8(KIND_TUNING);
            encode_tuning(w, t);
        }
        RecordedGameMessage::Other { debug } => {
            w.push_u8(KIND_OTHER);
            w.push_string(debug);
        }
    }
}

fn decode_game_message(r: &mut Reader) -> Result<RecordedGameMessage, FormatError> {
    let kind = r.read_u8()?;
    Ok(match kind {
        KIND_KILL => RecordedGameMessage::Kill {
            killer: r.read_i32()?,
            victim: r.read_i32()?,
            weapon: r.read_i32()?,
            mode_special: r.read_i32()?,
        },
        KIND_BROADCAST => RecordedGameMessage::Broadcast {
            message: r.read_string()?,
        },
        KIND_CHAT => RecordedGameMessage::Chat {
            team: r.read_i32()?,
            client_id: r.read_i32()?,
            message: r.read_string()?,
        },
        KIND_TUNING => RecordedGameMessage::Tuning(decode_tuning(r)?),
        KIND_OTHER => RecordedGameMessage::Other {
            debug: r.read_string()?,
        },
        other => return Err(FormatError::UnknownGameMessageKind(other)),
    })
}

impl Frame {
    /// Encodes this frame's body (no leading tag/length — [`crate::writer::RecordingWriter`]
    /// wraps that around whatever this returns, so a corrupt/unknown frame in the middle of a
    /// chunk can be skipped by length rather than desynchronising the rest of the chunk).
    pub fn encode_body(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Frame::Snapshot {
                tick,
                characters,
                players,
            } => {
                w.push_i32(*tick);
                w.push_u32(characters.len() as u32);
                for c in characters {
                    w.push_i32(c.id);
                    encode_character(&mut w, &c.character);
                    match &c.ddnet {
                        Some(d) => {
                            w.push_u8(1);
                            encode_ddnet_character(&mut w, d);
                        }
                        None => w.push_u8(0),
                    }
                }
                w.push_u32(players.len() as u32);
                for p in players {
                    w.push_i32(p.id);
                    encode_player_info(&mut w, &p.info);
                    match &p.client_info {
                        Some(ci) => {
                            w.push_u8(1);
                            encode_client_info(&mut w, ci);
                        }
                        None => w.push_u8(0),
                    }
                    match &p.ddnet {
                        Some(d) => {
                            w.push_u8(1);
                            encode_ddnet_player(&mut w, d);
                        }
                        None => w.push_u8(0),
                    }
                }
            }
            Frame::GameEvent { tick_hint, message } => {
                w.push_i32(*tick_hint);
                encode_game_message(&mut w, message);
            }
        }
        w.into_bytes()
    }

    pub fn tag(&self) -> u8 {
        match self {
            Frame::Snapshot { .. } => FRAME_TAG_SNAPSHOT,
            Frame::GameEvent { .. } => FRAME_TAG_GAME_EVENT,
        }
    }

    /// Decodes one frame's body given its `tag` (as written by [`Frame::tag`]) — the exact
    /// inverse of [`Frame::encode_body`]. Never panics on truncated/corrupt `body`: every field
    /// read can fail with [`FormatError::Decode`].
    pub fn decode_body(tag: u8, body: &[u8]) -> Result<Frame, FormatError> {
        let mut r = Reader::new(body);
        let frame = match tag {
            FRAME_TAG_SNAPSHOT => {
                let tick = r.read_i32()?;
                let num_characters = r.read_u32()? as usize;
                let mut characters = Vec::with_capacity(num_characters.min(1024));
                for _ in 0..num_characters {
                    let id = r.read_i32()?;
                    let character = decode_character(&mut r)?;
                    let ddnet = if r.read_u8()? != 0 {
                        Some(decode_ddnet_character(&mut r)?)
                    } else {
                        None
                    };
                    characters.push(CharacterRecord { id, character, ddnet });
                }
                let num_players = r.read_u32()? as usize;
                let mut players = Vec::with_capacity(num_players.min(1024));
                for _ in 0..num_players {
                    let id = r.read_i32()?;
                    let info = decode_player_info(&mut r)?;
                    let client_info = if r.read_u8()? != 0 {
                        Some(decode_client_info(&mut r)?)
                    } else {
                        None
                    };
                    let ddnet = if r.read_u8()? != 0 {
                        Some(decode_ddnet_player(&mut r)?)
                    } else {
                        None
                    };
                    players.push(PlayerRecord {
                        id,
                        info,
                        client_info,
                        ddnet,
                    });
                }
                Frame::Snapshot {
                    tick,
                    characters,
                    players,
                }
            }
            FRAME_TAG_GAME_EVENT => {
                let tick_hint = r.read_i32()?;
                let message = decode_game_message(&mut r)?;
                Frame::GameEvent { tick_hint, message }
            }
            other => return Err(FormatError::UnknownFrameTag(other)),
        };
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_character(id: i32) -> CharacterRecord {
        CharacterRecord {
            id,
            character: objects::Character {
                tick: 100,
                x: 320,
                y: -640,
                vel_x: 12,
                vel_y: -34,
                angle: 5000,
                direction: 1,
                jumped: 1,
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
                attack_tick: 42,
            },
            ddnet: Some(objects::DDNetCharacter {
                flags: 0,
                freeze_end: 0,
                jumps: 2,
                tele_checkpoint: -1,
                strong_weak_id: 0,
                jumped_total: -1,
                ninja_activation_tick: -1,
                freeze_start: -1,
                target_x: 5,
                target_y: -5,
                tune_zone_override: -1,
            }),
        }
    }

    fn sample_player(id: i32, name: &str) -> PlayerRecord {
        PlayerRecord {
            id,
            info: objects::PlayerInfo {
                local: 0,
                client_id: id,
                team: 0,
                score: 3,
                latency: 20,
            },
            client_info: Some(objects::ClientInfo {
                name: name.to_string(),
                clan: "".to_string(),
                country: -1,
                skin: "default".to_string(),
                use_custom_color: 0,
                color_body: 0,
                color_feet: 0,
            }),
            ddnet: Some(objects::DDNetPlayer {
                flags: 0,
                auth_level: 0,
                finish_time_seconds: -2,
                finish_time_millis: 0,
            }),
        }
    }

    #[test]
    fn header_round_trips() {
        let header = Header {
            server_address: "127.0.0.1:8303".to_string(),
            map_name: "Copy Love Box".to_string(),
            map_sha256: [9u8; 32],
            client_version: "DDNet 20.1".to_string(),
            start_time_unix_ms: 1_800_000_000_000,
            observer_nick: "Muha".to_string(),
        };
        let mut w = Writer::new();
        header.encode(&mut w);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(Header::decode(&mut r).unwrap(), header);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut w = Writer::new();
        w.push_raw(b"NOPE");
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert!(matches!(Header::decode(&mut r), Err(FormatError::BadMagic(_))));
    }

    #[test]
    fn header_rejects_unsupported_version() {
        let mut w = Writer::new();
        w.push_raw(&MAGIC);
        w.push_u32(999);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(
            Header::decode(&mut r).unwrap_err(),
            FormatError::UnsupportedVersion(999)
        );
    }

    #[test]
    fn snapshot_frame_with_no_ddnet_extension_round_trips() {
        let frame = Frame::Snapshot {
            tick: 1234,
            characters: vec![CharacterRecord {
                ddnet: None,
                ..sample_character(0)
            }],
            players: vec![PlayerRecord {
                client_info: None,
                ddnet: None,
                ..sample_player(0, "unused")
            }],
        };
        let body = frame.encode_body();
        let decoded = Frame::decode_body(frame.tag(), &body).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn snapshot_frame_with_full_ddnet_extension_round_trips() {
        let frame = Frame::Snapshot {
            tick: -5, // negative ticks must round-trip too (signed field)
            characters: vec![sample_character(0), sample_character(3)],
            players: vec![sample_player(0, "Müha"), sample_player(3, "")],
        };
        let body = frame.encode_body();
        let decoded = Frame::decode_body(frame.tag(), &body).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn empty_snapshot_frame_round_trips() {
        let frame = Frame::Snapshot {
            tick: 0,
            characters: vec![],
            players: vec![],
        };
        let body = frame.encode_body();
        assert_eq!(Frame::decode_body(frame.tag(), &body).unwrap(), frame);
    }

    #[test]
    fn every_game_message_kind_round_trips() {
        let messages = vec![
            RecordedGameMessage::Kill {
                killer: 1,
                victim: 2,
                weapon: 3,
                mode_special: 0,
            },
            RecordedGameMessage::Broadcast {
                message: "gg".to_string(),
            },
            RecordedGameMessage::Chat {
                team: 0,
                client_id: 5,
                message: "hello, 世界".to_string(),
            },
            RecordedGameMessage::Tuning(ddai_net::tuning::DEFAULT_TUNE_PARAMS),
            RecordedGameMessage::Other {
                debug: "SvMotd(SvMotd { message: \"welcome\" })".to_string(),
            },
        ];
        for message in messages {
            let frame = Frame::GameEvent {
                tick_hint: 42,
                message: message.clone(),
            };
            let body = frame.encode_body();
            let decoded = Frame::decode_body(frame.tag(), &body).unwrap();
            assert_eq!(decoded, frame, "round trip failed for {message:?}");
        }
    }

    #[test]
    fn unknown_frame_tag_is_an_error_not_a_panic() {
        assert!(matches!(
            Frame::decode_body(99, &[]),
            Err(FormatError::UnknownFrameTag(99))
        ));
    }

    #[test]
    fn truncated_frame_body_is_an_error_not_a_panic() {
        let frame = Frame::Snapshot {
            tick: 1,
            characters: vec![sample_character(0)],
            players: vec![],
        };
        let body = frame.encode_body();
        for cut in 0..body.len() {
            let truncated = &body[..cut];
            // Must never panic; an error is fine (and expected for most cut points).
            let _ = Frame::decode_body(frame.tag(), truncated);
        }
    }

    #[test]
    fn tuning_with_jetpack_and_extreme_values_round_trips() {
        let t = TuneParams {
            received: 47,
            jetpack_strength: i32::MIN,
            gravity: i32::MAX,
            ..ddai_net::tuning::DEFAULT_TUNE_PARAMS
        };
        let msg = RecordedGameMessage::Tuning(t);
        let frame = Frame::GameEvent {
            tick_hint: 0,
            message: msg.clone(),
        };
        let body = frame.encode_body();
        assert_eq!(Frame::decode_body(frame.tag(), &body).unwrap(), frame);
        let _ = msg;
    }
}
