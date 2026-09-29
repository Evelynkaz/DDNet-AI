//! On-disk record types of the human-play dataset (`docs/formats.md` §20) and the in-memory
//! timeline the detectors run over. Everything here is plain data (`serde`), lossless with respect
//! to `ddai_brain::CharacterObservation` (positions/velocities stay `f32`), and contains **no
//! nickname and no free text**: players are only ever addressed by the anonymous per-demo label
//! number (`player`), which is not derived from a name (see [`crate::ingest`]).

use ddai_brain::{Action, CharacterObservation, IVec2};
use ddai_physics::vmath::Vec2;
use serde::{Deserialize, Serialize};

/// Bumped on any change of the serialized shape of [`Chunk`] or the manifest. Version 2: `CharRec::flags`
/// widened from `u8` to `u16` (new `JUMP_HELD` bit).
pub const FORMAT_VERSION: u32 = 2;

/// [`CharRec::flags`] bits.
pub mod char_flags {
    /// Inputs are forced to zero by a freeze (timed or deep) right now.
    pub const FROZEN: u16 = 1 << 0;
    pub const DEEP_FROZEN: u16 = 1 << 1;
    pub const LIVE_FROZEN: u16 = 1 << 2;
    pub const GROUNDED: u16 = 1 << 3;
    /// The freeze state was *inferred* (no `DDNetCharacter` extension in this demo: the ninja-
    /// weapon convention of old servers was used) rather than read from the wire.
    pub const FREEZE_INFERRED: u16 = 1 << 4;
    /// A fire event (`attack_tick` advanced) happened in the two ticks before this snapshot.
    pub const FIRED: u16 = 1 << 5;
    /// The wire core of this character was fresh (age <= 1 tick) at this snapshot: its position,
    /// velocity, direction and hook fields were sent by the server for this very tick.
    pub const FRESH: u16 = 1 << 6;
    /// The air jump has been used (`jumped & 2`): no jump executes until the next landing.
    pub const AIR_JUMP_USED: u16 = 1 << 7;
    /// The jump button is held (`jumped & 1`): the character pressed jump.
    pub const JUMP_HELD: u16 = 1 << 8;
}

/// One character in one snapshot: the [`CharacterObservation`] fields plus the derived extras the
/// detectors need (aim direction, fire event, wire freshness).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CharRec {
    /// Client id inside the demo (`0..MAX_CLIENTS`); not an identity (ids are reused).
    pub id: u8,
    /// Anonymous per-demo player label number (one per `(client id, stint)`, first-appearance
    /// order, see [`crate::ingest::anonymize`]).
    pub player: u16,
    pub team: i16,
    /// World pixels (unquantized physics value of the reconstructed tick).
    pub pos: [f32; 2],
    /// Pixels per tick (raw physics velocity).
    pub vel: [f32; 2],
    pub hook_state: i8,
    pub hook_pos: [f32; 2],
    /// Client id of the hooked character, `-1` for none (terrain or nothing).
    pub hooked_player: i16,
    pub flags: u16,
    pub freeze_ticks: i16,
    pub jumps_left: i8,
    pub jumps_used: i8,
    pub weapon: i8,
    pub direction: i8,
    /// Aim direction, integer vector (magnitude is nominal when only the wire angle is known).
    pub aim: [i32; 2],
}

impl CharRec {
    pub fn has(&self, flag: u16) -> bool {
        self.flags & flag != 0
    }
    pub fn frozen(&self) -> bool {
        self.has(char_flags::FROZEN)
    }
    pub fn grounded(&self) -> bool {
        self.has(char_flags::GROUNDED)
    }
    pub fn pos_v(&self) -> Vec2<f32> {
        Vec2::new(self.pos[0], self.pos[1])
    }

    /// The `ddai-brain` view of this record (7.3 units: pixels, pixels per tick).
    pub fn to_observation(&self) -> CharacterObservation {
        CharacterObservation {
            id: i32::from(self.id),
            team: i32::from(self.team),
            pos: Vec2::new(self.pos[0], self.pos[1]),
            vel: Vec2::new(self.vel[0], self.vel[1]),
            hook_state: i32::from(self.hook_state),
            hook_pos: Vec2::new(self.hook_pos[0], self.hook_pos[1]),
            hooked_player: i32::from(self.hooked_player),
            is_frozen: self.has(char_flags::FROZEN),
            is_deep_frozen: self.has(char_flags::DEEP_FROZEN),
            is_live_frozen: self.has(char_flags::LIVE_FROZEN),
            freeze_ticks_remaining: i32::from(self.freeze_ticks),
            jumps_left: i32::from(self.jumps_left),
            jumps_used: i32::from(self.jumps_used),
            grounded: self.has(char_flags::GROUNDED),
            weapon: i32::from(self.weapon),
            direction: i32::from(self.direction),
        }
    }
}

/// One fresh snapshot: every character that was present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRec {
    pub tick: i32,
    pub chars: Vec<CharRec>,
}

/// The reconstructed human input for one decision interval: the two steps `tick -> tick + 1 -> tick + 2`, i.e. `[tick, tick + 2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRec {
    pub direction: i8,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    pub aim: [i32; 2],
}

impl ActionRec {
    pub fn to_action(&self) -> Action {
        Action {
            direction: i32::from(self.direction),
            jump: self.jump,
            hook: self.hook,
            fire: self.fire,
            target: IVec2::new(self.aim[0], self.aim[1]),
            wanted_weapon: None,
        }
    }
}

/// How well the physics replay of the reconstructed inputs reproduced the demo's next state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ReplayClass {
    /// No next state to compare with (player absent 2 ticks later, gap in the snapshots, ...).
    Unavailable = 0,
    /// Position differs by more than 1 px.
    Off = 1,
    /// Position within 1 px on both axes, quantized state not identical.
    Within1px = 2,
    /// Quantized position and velocity identical to the demo's next state.
    Exact = 3,
}

impl ReplayClass {
    pub const ALL: [ReplayClass; 4] = [
        ReplayClass::Unavailable,
        ReplayClass::Off,
        ReplayClass::Within1px,
        ReplayClass::Exact,
    ];
    pub fn name(self) -> &'static str {
        match self {
            ReplayClass::Unavailable => "unavailable",
            ReplayClass::Off => "off",
            ReplayClass::Within1px => "within_1px",
            ReplayClass::Exact => "exact",
        }
    }
}

/// [`SampleRec::q`] bits: reconstruction-quality flags.
pub mod quality {
    /// The wire core at the *next* snapshot was fresh, i.e. the inputs were actually observable.
    pub const NEXT_FRESH: u8 = 1 << 2;
    /// A non-neutral action, or the character moved (speed >= 1 px/tick), or it was hooking.
    pub const ACTIVE: u8 = 1 << 3;
    /// Mask of the two low bits: the [`super::ReplayClass`].
    pub const REPLAY_MASK: u8 = 0b11;
}

/// Skill bucket of the player (documented rule in [`crate::skill`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SkillBucket {
    /// Present for less than the minimum time: not ranked.
    Unranked = 0,
    Low = 1,
    Mid = 2,
    Top = 3,
}

impl SkillBucket {
    pub const ALL: [SkillBucket; 4] = [
        SkillBucket::Unranked,
        SkillBucket::Low,
        SkillBucket::Mid,
        SkillBucket::Top,
    ];
    pub fn name(self) -> &'static str {
        match self {
            SkillBucket::Unranked => "unranked",
            SkillBucket::Low => "low",
            SkillBucket::Mid => "mid",
            SkillBucket::Top => "top",
        }
    }
    pub fn from_u8(v: u8) -> SkillBucket {
        match v {
            1 => SkillBucket::Low,
            2 => SkillBucket::Mid,
            3 => SkillBucket::Top,
            _ => SkillBucket::Unranked,
        }
    }
}

/// One `(Observation, Action)` sample, stored relative to its chunk.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SampleRec {
    /// Index of the frame inside the chunk.
    pub frame: u32,
    /// Index of the acting character inside `FrameRec::chars`.
    pub slot: u8,
    pub action: ActionRec,
    /// `target_id` per the documented rule ([`crate::pipeline::choose_target`]); `-1` = none.
    pub target: i16,
    /// Technique / skill tag bits ([`crate::tags`]).
    pub tags: u32,
    /// [`quality`] flags + [`ReplayClass`] in the two low bits.
    pub q: u8,
    /// Skill bucket of the acting player in this demo.
    pub skill: u8,
}

impl SampleRec {
    pub fn replay(&self) -> ReplayClass {
        match self.q & quality::REPLAY_MASK {
            1 => ReplayClass::Off,
            2 => ReplayClass::Within1px,
            3 => ReplayClass::Exact,
            _ => ReplayClass::Unavailable,
        }
    }
    pub fn next_fresh(&self) -> bool {
        self.q & quality::NEXT_FRESH != 0
    }
    pub fn active(&self) -> bool {
        self.q & quality::ACTIVE != 0
    }
    /// A sample gets training weight only when the physics replay reproduced the next state.
    pub fn confident(&self) -> bool {
        self.replay() >= ReplayClass::Within1px
    }
}

/// The unit of storage: one demo's frames and samples (a long demo is split over several chunks
/// at frame boundaries; a sample never refers to a frame of another chunk).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub format_version: u32,
    /// Index into the manifest's demo table.
    pub demo: u32,
    pub frames: Vec<FrameRec>,
    pub samples: Vec<SampleRec>,
}
