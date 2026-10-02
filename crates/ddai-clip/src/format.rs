//! The clip format, version 1 (`docs/formats.md` §24): what a clip holds and how it is stored.
//!
//! A clip is a window of the bot's own life: for every snapshot frame the wire state of the tees that
//! matter to our physics (ours, the nearest few, everyone hooked to or by us), the projectiles, the
//! inputs of ours that were actually **sent** (as tagged on the wire, the ones the server applies),
//! what the bot knew and decided, and the events that really happened. With the map identity and the
//! tuning that is enough to replay the bot's own physics offline bit for bit
//! ([`crate::replay`]).
//!
//! Everything here is plain data (`serde`). **No nicknames**: players are the 4.1 tags `c<id>-<hash>`.
//! The file is `magic + zstd(postcard(Clip))`; the magic carries the version, so a reader refuses a
//! version it does not know before decoding anything.

use std::io::{Read, Write};
use std::path::Path;

use ddai_net::generated::objects;
use ddai_net::view::ProjectileView;
use serde::{Deserialize, Serialize};

/// First bytes of every clip file: `DDCLIP` + format version 1.
pub const MAGIC: [u8; 8] = *b"DDCLIP\0\x01";
/// The current format version (the last byte of [`MAGIC`]).
pub const FORMAT_VERSION: u8 = 1;
/// Frames of the ring: 30 s of snapshots at 25 Hz.
pub const RING_FRAMES: usize = 750;
/// Tees recorded per frame: ours, the nearest 7 and everyone hooked to or by us, at most.
pub const MAX_TEES: usize = 12;
/// Projectiles recorded per frame (the rest of a crowded snapshot is counted, not stored).
pub const MAX_PROJECTILES: usize = 24;
/// Sent inputs recorded per frame (the ticks since the previous frame; 2 normally).
pub const MAX_SENT: usize = 8;
/// Events recorded per frame.
pub const MAX_EVENTS: usize = 16;
/// Switch states recorded per frame.
pub const MAX_SWITCHES: usize = 4;

/// A `CNetObj_Character` (the core of a tee as the server sends it), 22 ints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CharRec {
    pub tick: i32,
    pub x: i32,
    pub y: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub angle: i32,
    pub direction: i32,
    pub jumped: i32,
    pub hooked_player: i32,
    pub hook_state: i32,
    pub hook_tick: i32,
    pub hook_x: i32,
    pub hook_y: i32,
    pub hook_dx: i32,
    pub hook_dy: i32,
    pub player_flags: i32,
    pub health: i32,
    pub armor: i32,
    pub ammo_count: i32,
    pub weapon: i32,
    pub emote: i32,
    pub attack_tick: i32,
}

impl CharRec {
    pub fn from_net(c: &objects::Character) -> CharRec {
        CharRec {
            tick: c.tick,
            x: c.x,
            y: c.y,
            vel_x: c.vel_x,
            vel_y: c.vel_y,
            angle: c.angle,
            direction: c.direction,
            jumped: c.jumped,
            hooked_player: c.hooked_player,
            hook_state: c.hook_state,
            hook_tick: c.hook_tick,
            hook_x: c.hook_x,
            hook_y: c.hook_y,
            hook_dx: c.hook_dx,
            hook_dy: c.hook_dy,
            player_flags: c.player_flags,
            health: c.health,
            armor: c.armor,
            ammo_count: c.ammo_count,
            weapon: c.weapon,
            emote: c.emote,
            attack_tick: c.attack_tick,
        }
    }

    pub fn to_net(&self) -> objects::Character {
        objects::Character {
            tick: self.tick,
            x: self.x,
            y: self.y,
            vel_x: self.vel_x,
            vel_y: self.vel_y,
            angle: self.angle,
            direction: self.direction,
            jumped: self.jumped,
            hooked_player: self.hooked_player,
            hook_state: self.hook_state,
            hook_tick: self.hook_tick,
            hook_x: self.hook_x,
            hook_y: self.hook_y,
            hook_dx: self.hook_dx,
            hook_dy: self.hook_dy,
            player_flags: self.player_flags,
            health: self.health,
            armor: self.armor,
            ammo_count: self.ammo_count,
            weapon: self.weapon,
            emote: self.emote,
            attack_tick: self.attack_tick,
        }
    }
}

/// A `CNetObj_DDNetCharacter` (the DDNet extension of a tee), 11 ints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DdRec {
    pub flags: i32,
    pub freeze_end: i32,
    pub jumps: i32,
    pub tele_checkpoint: i32,
    pub strong_weak_id: i32,
    pub jumped_total: i32,
    pub ninja_activation_tick: i32,
    pub freeze_start: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub tune_zone_override: i32,
}

impl DdRec {
    pub fn from_net(d: &objects::DDNetCharacter) -> DdRec {
        DdRec {
            flags: d.flags,
            freeze_end: d.freeze_end,
            jumps: d.jumps,
            tele_checkpoint: d.tele_checkpoint,
            strong_weak_id: d.strong_weak_id,
            jumped_total: d.jumped_total,
            ninja_activation_tick: d.ninja_activation_tick,
            freeze_start: d.freeze_start,
            target_x: d.target_x,
            target_y: d.target_y,
            tune_zone_override: d.tune_zone_override,
        }
    }

    pub fn to_net(&self) -> objects::DDNetCharacter {
        objects::DDNetCharacter {
            flags: self.flags,
            freeze_end: self.freeze_end,
            jumps: self.jumps,
            tele_checkpoint: self.tele_checkpoint,
            strong_weak_id: self.strong_weak_id,
            jumped_total: self.jumped_total,
            ninja_activation_tick: self.ninja_activation_tick,
            freeze_start: self.freeze_start,
            target_x: self.target_x,
            target_y: self.target_y,
            tune_zone_override: self.tune_zone_override,
        }
    }
}

/// One recorded tee: the wire state plus what the bot derived from it (`frozen` and friends are what the
/// incidents read; the replay reads only the wire part).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TeeRec {
    pub id: i32,
    pub ch: CharRec,
    pub dd: Option<DdRec>,
    /// Frozen (normal or deep) at this frame.
    pub frozen: bool,
    pub deep_frozen: bool,
    /// Ticks until it thaws (150 for a deep freeze).
    pub freeze_left: i32,
}

impl TeeRec {
    /// Position in pixels.
    pub fn pos(&self) -> (f64, f64) {
        (f64::from(self.ch.x), f64::from(self.ch.y))
    }

    /// Velocity in pixels per tick (the wire carries 1/256 px).
    pub fn vel(&self) -> (f64, f64) {
        (f64::from(self.ch.vel_x) / 256.0, f64::from(self.ch.vel_y) / 256.0)
    }
}

/// A projectile item of the snapshot, kept as its raw ints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProjRec {
    /// The snapshot item id.
    pub id: i32,
    /// 0 = legacy (`Projectile`), 1 = DDRace, 2 = DDNet.
    pub kind: u8,
    pub v: [i32; 10],
}

impl ProjRec {
    pub fn from_view(id: i32, p: &ProjectileView) -> ProjRec {
        let mut v = [0i32; 10];
        let kind = match p {
            ProjectileView::Legacy(o) => {
                v[..6].copy_from_slice(&[o.x, o.y, o.vel_x, o.vel_y, o.type_, o.start_tick]);
                0
            }
            ProjectileView::DDRace(o) => {
                v[..6].copy_from_slice(&[o.x, o.y, o.angle, o.data, o.type_, o.start_tick]);
                1
            }
            ProjectileView::DDNet(o) => {
                v.copy_from_slice(&[
                    o.x,
                    o.y,
                    o.vel_x,
                    o.vel_y,
                    o.type_,
                    o.start_tick,
                    o.owner,
                    o.switch_number,
                    o.tune_zone,
                    o.flags,
                ]);
                2
            }
        };
        ProjRec { id, kind, v }
    }

    pub fn to_view(&self) -> (i32, ProjectileView) {
        let v = &self.v;
        let view = match self.kind {
            0 => ProjectileView::Legacy(objects::Projectile {
                x: v[0],
                y: v[1],
                vel_x: v[2],
                vel_y: v[3],
                type_: v[4],
                start_tick: v[5],
            }),
            1 => ProjectileView::DDRace(objects::DDRaceProjectile {
                x: v[0],
                y: v[1],
                angle: v[2],
                data: v[3],
                type_: v[4],
                start_tick: v[5],
            }),
            _ => ProjectileView::DDNet(objects::DDNetProjectile {
                x: v[0],
                y: v[1],
                vel_x: v[2],
                vel_y: v[3],
                type_: v[4],
                start_tick: v[5],
                owner: v[6],
                switch_number: v[7],
                tune_zone: v[8],
                flags: v[9],
            }),
        };
        (self.id, view)
    }
}

/// A `PlayerInput` as the wire carries it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InputRec {
    pub direction: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
}

impl InputRec {
    pub fn from_net(i: &objects::PlayerInput) -> InputRec {
        InputRec {
            direction: i.direction,
            target_x: i.target_x,
            target_y: i.target_y,
            jump: i.jump,
            fire: i.fire,
            hook: i.hook,
            player_flags: i.player_flags,
            wanted_weapon: i.wanted_weapon,
            next_weapon: i.next_weapon,
            prev_weapon: i.prev_weapon,
        }
    }

    pub fn to_net(&self) -> objects::PlayerInput {
        objects::PlayerInput {
            direction: self.direction,
            target_x: self.target_x,
            target_y: self.target_y,
            jump: self.jump,
            fire: self.fire,
            hook: self.hook,
            player_flags: self.player_flags,
            wanted_weapon: self.wanted_weapon,
            next_weapon: self.next_weapon,
            prev_weapon: self.prev_weapon,
        }
    }
}

/// An input in force at a tick: the one the server applies there (a late-corrected claim of the sent log).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SentRec {
    pub tick: i32,
    pub input: InputRec,
    /// The server's timing report for this input had arrived when the frame was recorded (a late
    /// input is re-targeted forward; without the report the tick is the one it was sent for).
    pub timing_known: bool,
}

/// A switch state item (`CNetObj_SwitchState`), per team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SwitchRec {
    pub team: i32,
    pub highest_switch_number: i32,
    pub status: [i32; 8],
    pub switch_numbers: [i32; 4],
    pub end_ticks: [i32; 4],
}

impl SwitchRec {
    pub fn from_net(team: i32, s: &objects::SwitchState) -> SwitchRec {
        SwitchRec {
            team,
            highest_switch_number: s.highest_switch_number,
            status: s.status,
            switch_numbers: s.switch_numbers,
            end_ticks: s.end_ticks,
        }
    }

    pub fn to_net(&self) -> (i32, objects::SwitchState) {
        (
            self.team,
            objects::SwitchState {
                highest_switch_number: self.highest_switch_number,
                status: self.status,
                switch_numbers: self.switch_numbers,
                end_ticks: self.end_ticks,
            },
        )
    }
}

/// Things that really happened, as the bot saw them (task 4.3: the TS recorded none of these live).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClipEvent {
    /// A hammer swing of `from` landed on `to` (the activity clock's attribution: the swing point within 56 px).
    HammerHit { from: i32, to: i32 },
    /// `from` swung its hammer (its attack tick moved while it held the hammer); `hits` of the 56 px reach.
    HammerFire { from: i32, hits: u8 },
    /// `id`'s hook grabbed `target` (`-1`: a wall).
    HookAttach { id: i32, target: i32 },
    /// `id`'s hook let go of `target` after `held` ticks.
    HookRelease { id: i32, target: i32, held: i32 },
    /// `id` went from free to frozen.
    FreezeOnset { id: i32 },
    /// A death announced by `SV_KILLMSG`: `killer` (`-1` none), `victim`, the weapon (`-1` world, `-2` /kill, `-3` game end).
    Kill { killer: i32, victim: i32, weapon: i32 },
    /// Our own tee came back after a death (or `Cl_Kill`).
    Respawn { id: i32 },
    /// We asked the server to kill us (`Cl_Kill`), and why (`KillWhy`).
    KillSent { why: u8 },
}

/// Why a `Cl_Kill` was sent (for [`ClipEvent::KillSent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KillWhy {
    Unstick = 0,
    WayBlockLying = 1,
    Navigation = 2,
    Trek = 3,
    Console = 4,
}

/// What the bot knew and did at one frame (the `lastPlan` of the TS, plus the nav state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BotRec {
    /// The picked target, `-1` none.
    pub target: i32,
    /// The brain kind (`ddai_bot::BrainKind` as a number: 0 hybrid, 1 planner, 2 scripted, 3 idle, 4 fly), 255 unknown.
    pub brain: u8,
    /// `BIT_*` flags.
    pub flags: u16,
    /// Index into [`ClipHeader::labels`] of the walk label (`0`: not walking).
    pub walk: u16,
    /// Decision time (the whole pipeline) and the brain's share, in microseconds.
    pub total_us: u32,
    pub brain_us: u32,
    /// The brain's last plan: candidates scored, the search ran, ran out of time, the shield stepped in.
    pub candidates: u32,
    /// The frame ticks the bot's decision was aimed at (the input slot), `0` when none.
    pub aimed_tick: i32,
    /// D-059 attribution counters so far: blocks by us / blocks of us by others.
    pub blocks: u16,
    pub blocked_by: u16,
}

impl BotRec {
    pub const BIT_SEARCHED: u16 = 1;
    pub const BIT_OUT_OF_TIME: u16 = 2;
    pub const BIT_SHIELDED: u16 = 4;
    pub const BIT_SHIELD_INCOMPLETE: u16 = 8;
    pub const BIT_GUARDED: u16 = 16;
    pub const BIT_VETOED_HOOK: u16 = 32;
    pub const BIT_VETOED_FIRE: u16 = 64;
    pub const BIT_WANDER: u16 = 128;
    pub const BIT_CROSSING: u16 = 256;
    pub const BIT_PLANNED_FREEZE: u16 = 512;
    pub const BIT_WB_HOLDING: u16 = 1024;
    pub const BIT_BRAIN_DECIDED: u16 = 2048;

    pub fn has(&self, bit: u16) -> bool {
        self.flags & bit != 0
    }
}

/// One frame, the serialised form.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Frame {
    /// The snapshot tick.
    pub tick: i32,
    /// Our own tee is in the snapshot (a frame without it is a death or a spectator frame).
    pub own_alive: bool,
    pub tees: Vec<TeeRec>,
    pub projectiles: Vec<ProjRec>,
    /// Projectiles of the snapshot that did not fit (the cap is [`MAX_PROJECTILES`]).
    pub projectiles_dropped: u8,
    /// Tees of the snapshot that were not recorded (only the nearest matter; the replay cannot see them).
    pub tees_dropped: u8,
    pub switches: Vec<SwitchRec>,
    /// Our inputs in force at the ticks since the previous frame (the last is the frame's own tick).
    pub sent: Vec<SentRec>,
    pub events: Vec<ClipEvent>,
    pub bot: BotRec,
}

impl Frame {
    pub fn tee(&self, id: i32) -> Option<&TeeRec> {
        self.tees.iter().find(|t| t.id == id)
    }

    /// The input of ours in force at the frame's own tick.
    pub fn own_input(&self) -> Option<&SentRec> {
        self.sent.last()
    }
}

/// A change of the tuning (`Sv_TuneParams`), recorded when it changes: `values` are the 47 ints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneChange {
    /// First frame tick it holds from (the clip's first tick for the initial one).
    pub from_tick: i32,
    pub received: u32,
    pub values: Vec<i32>,
}

/// A change of the DDRace teams state: team of each client below `received`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamsChange {
    pub from_tick: i32,
    pub received: u32,
    pub teams: Vec<i32>,
}

/// A player of the clip: the 4.1 tag, never a nickname.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerTag {
    pub id: i32,
    pub tag: String,
}

/// Why the clip was saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClipReason {
    /// `manual`, an incident kind, or `cross-fail`.
    pub kind: String,
    pub severity: i32,
    /// The incident's tick (the manual clip's last tick).
    pub tick: i32,
    pub note: String,
}

/// The clip header: the map identity (never the map), the world constants, who was who.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClipHeader {
    pub map_name: String,
    pub map_sha256: [u8; 32],
    pub own_id: i32,
    /// The brain's name when the clip was saved.
    pub brain: String,
    pub reason: ClipReason,
    /// Walk labels by index (index 0 is the empty label).
    pub labels: Vec<String>,
    pub players: Vec<PlayerTag>,
    pub tuning: Vec<TuneChange>,
    pub teams: Vec<TeamsChange>,
    /// The seed of the `LiveWorld` that recorded it (`LiveWorld::new`'s; the physics does not use it).
    pub world_seed: u64,
}

/// A clip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clip {
    pub header: ClipHeader,
    pub frames: Vec<Frame>,
}

#[derive(Debug, thiserror::Error)]
pub enum ClipError {
    #[error("not a clip file (bad magic)")]
    BadMagic,
    #[error("clip format version {0} is not supported (this build reads {FORMAT_VERSION})")]
    Version(u8),
    #[error("reading or writing the clip: {0}")]
    Io(#[from] std::io::Error),
    #[error("decoding the clip: {0}")]
    Decode(String),
}

impl Clip {
    /// `magic + zstd(postcard(self))`.
    pub fn encode(&self) -> Result<Vec<u8>, ClipError> {
        let raw = postcard::to_stdvec(self).map_err(|e| ClipError::Decode(e.to_string()))?;
        let mut out = Vec::with_capacity(raw.len() / 4 + 16);
        out.extend_from_slice(&MAGIC);
        let mut z = zstd::Encoder::new(&mut out, 6)?;
        z.write_all(&raw)?;
        z.finish()?;
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Clip, ClipError> {
        if bytes.len() < MAGIC.len() || bytes[..MAGIC.len() - 1] != MAGIC[..MAGIC.len() - 1] {
            return Err(ClipError::BadMagic);
        }
        let version = bytes[MAGIC.len() - 1];
        if version != FORMAT_VERSION {
            return Err(ClipError::Version(version));
        }
        let mut raw = Vec::new();
        zstd::Decoder::new(&bytes[MAGIC.len()..])?.read_to_end(&mut raw)?;
        postcard::from_bytes(&raw).map_err(|e| ClipError::Decode(e.to_string()))
    }

    /// Writes the clip atomically (`<path>.tmp` then rename), creating the directory.
    pub fn write(&self, path: &Path) -> Result<(), ClipError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("clip.tmp");
        std::fs::write(&tmp, self.encode()?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Clip, ClipError> {
        Clip::decode(&std::fs::read(path)?)
    }

    /// The first and last tick of the frames, if any.
    pub fn tick_range(&self) -> Option<(i32, i32)> {
        Some((self.frames.first()?.tick, self.frames.last()?.tick))
    }
}
