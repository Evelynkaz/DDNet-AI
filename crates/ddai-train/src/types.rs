//! Records of the teacher dataset (`docs/formats.md`-style, format `ddai-teacher` v1): one
//! [`Episode`] per arena game of the labelled player, one [`TeacherStep`] per decision.
//!
//! Characters are stored as `ddai_dataset::types::CharRec`, the record type of the human dataset,
//! so both kinds of data turn into observations the same way. A step keeps the teacher's hard
//! action, its soft target (the elite set's first-step statistics), and the action that was
//! actually *played* (different from the label whenever a student acted: DAgger), plus flags.
//!
//! Aim angles in this crate are **ring angles** (`atan2(-dy, dx)`, `0` = right, `pi/2` = up), the
//! convention of the fly's decoder; the planner's own aim is in screen coordinates and is
//! converted here ([`soft_from_elite`]).

use ddai_brain::{Action, CharacterObservation, IVec2};
use ddai_dataset::types::{ActionRec, CharRec, char_flags};
use ddai_planner::elite::EliteFirstStep;
use serde::{Deserialize, Serialize};

pub const TEACHER_FORMAT: &str = "ddai-teacher";
pub const TEACHER_FORMAT_VERSION: u32 = 1;

/// `TeacherStep::flags` bits.
pub mod step_flags {
    /// The planner ran a CEM search for this decision (a soft target may exist).
    pub const SEARCHED: u8 = 1 << 0;
    /// The teacher's own action was played (DAgger mixing, or the teacher is the actor).
    pub const TEACHER_ACTED: u8 = 1 << 1;
    /// The played action is a random perturbation (DART-style exploration noise).
    pub const NOISE: u8 = 1 << 2;
}

/// The elite set's first-step statistics, with the aim already in ring convention.
///
/// The direction/jump/hook/fire frequencies are the soft targets of the loss (mixed with the hard
/// label by `soft_mix`). `aim_mean` (circular mean of the elite's absolute aim) and `aim_spread`
/// (circular standard deviation) are **recorded only**: they are stored for analysis of how
/// uncertain the teacher's aim was, but the aim term of the loss is a von Mises likelihood of the
/// hard label's aim (`ddai_fly::bc`), and `SoftTargets` has no aim fields. Changing that would be a
/// loss change and needs a retrain (E-005 review, nit 2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SoftRec {
    pub left: f32,
    pub stop: f32,
    pub right: f32,
    pub jump: f32,
    pub hook: f32,
    pub fire: f32,
    pub aim_mean: f32,
    pub aim_spread: f32,
}

/// Converts the planner's screen-convention elite summary into the stored soft record.
pub fn soft_from_elite(e: &EliteFirstStep) -> SoftRec {
    SoftRec {
        left: e.left,
        stop: e.stop,
        right: e.right,
        jump: e.jump,
        hook: e.hook,
        fire: e.fire,
        aim_mean: -e.aim_mean,
        aim_spread: e.aim_spread,
    }
}

/// Ring angle of an aim target vector (`(0, 0)` is treated as straight up, as the wire format does).
pub fn ring_angle_of_target(target: [i32; 2]) -> f32 {
    let (x, y) = if target == [0, 0] {
        (0, -1)
    } else {
        (target[0], target[1])
    };
    (-(y as f32)).atan2(x as f32)
}

pub fn action_rec(a: &Action) -> ActionRec {
    ActionRec {
        direction: a.direction.clamp(-1, 1) as i8,
        jump: a.jump,
        hook: a.hook,
        fire: a.fire,
        aim: [a.target.x, a.target.y],
    }
}

pub fn action_of(rec: &ActionRec) -> Action {
    Action {
        direction: i32::from(rec.direction),
        jump: rec.jump,
        hook: rec.hook,
        fire: rec.fire,
        target: IVec2::new(rec.aim[0], rec.aim[1]),
        wanted_weapon: None,
    }
}

/// A character observation as a stored record (no aim, anonymous player label `0`).
pub fn char_rec(c: &CharacterObservation) -> CharRec {
    let mut flags = 0u16;
    if c.is_frozen {
        flags |= char_flags::FROZEN;
    }
    if c.is_deep_frozen {
        flags |= char_flags::DEEP_FROZEN;
    }
    if c.is_live_frozen {
        flags |= char_flags::LIVE_FROZEN;
    }
    if c.grounded {
        flags |= char_flags::GROUNDED;
    }
    CharRec {
        id: c.id.clamp(0, 255) as u8,
        player: 0,
        team: c.team as i16,
        pos: [c.pos.x, c.pos.y],
        vel: [c.vel.x, c.vel.y],
        hook_state: c.hook_state as i8,
        hook_pos: [c.hook_pos.x, c.hook_pos.y],
        hooked_player: c.hooked_player as i16,
        flags,
        freeze_ticks: c.freeze_ticks_remaining.clamp(0, i32::from(i16::MAX)) as i16,
        jumps_left: c.jumps_left.clamp(0, 127) as i8,
        jumps_used: c.jumps_used.clamp(0, 127) as i8,
        weapon: c.weapon.clamp(0, 127) as i8,
        direction: c.direction.clamp(-1, 1) as i8,
        aim: [0, 0],
    }
}

/// One decision of the labelled player.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherStep {
    pub tick: i32,
    pub me: CharRec,
    /// Every other live character of the game (the arena observation).
    pub others: Vec<CharRec>,
    /// `Observation::target_id`, `-1` for none.
    pub target: i16,
    /// The teacher's decision.
    pub label: ActionRec,
    pub soft: Option<SoftRec>,
    /// What was played (equals `label` when the teacher acted).
    pub played: ActionRec,
    pub flags: u8,
}

impl TeacherStep {
    pub fn searched(&self) -> bool {
        self.flags & step_flags::SEARCHED != 0
    }
    pub fn teacher_acted(&self) -> bool {
        self.flags & step_flags::TEACHER_ACTED != 0
    }
    pub fn noise(&self) -> bool {
        self.flags & step_flags::NOISE != 0
    }
}

/// How a game ended for the labelled player (slot 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Outcome {
    Win = 0,
    Loss = 1,
    Draw = 2,
    Timeout = 3,
}

/// The labelled player's decisions of one game.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    /// Index into the manifest's arena list.
    pub arena: u16,
    pub seed: u64,
    /// Number of players in the game (2 = 1v1).
    pub players: u8,
    pub outcome: Outcome,
    pub end_tick: i32,
    pub steps: Vec<TeacherStep>,
}

/// The unit of storage: a handful of episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherChunk {
    pub format_version: u32,
    pub episodes: Vec<Episode>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::vmath::Vec2;

    #[test]
    fn ring_angle_convention_matches_the_decoder() {
        assert!((ring_angle_of_target([100, 0]) - 0.0).abs() < 1e-6);
        assert!(
            (ring_angle_of_target([0, -100]) - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "up"
        );
        assert!(
            (ring_angle_of_target([0, 100]) + std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "down"
        );
        assert_eq!(ring_angle_of_target([0, 0]), ring_angle_of_target([0, -1]));
        // Round trip through the fly's own angle -> target conversion.
        for a in [-3.0f32, -1.0, 0.0, 0.7, 2.5] {
            let t = ddai_fly::brain::aim_angle_to_target(a);
            assert!((ring_angle_of_target([t.x, t.y]) - a).abs() < 2e-3, "{a}");
        }
    }

    #[test]
    fn the_planners_screen_aim_becomes_a_ring_angle() {
        // Screen angle +pi/4 points right and *down*; the ring angle of that is -pi/4.
        let e = EliteFirstStep {
            elites: 6,
            left: 0.0,
            stop: 1.0,
            right: 0.0,
            jump: 0.0,
            hook: 1.0,
            fire: 0.0,
            aim_mean: std::f32::consts::FRAC_PI_4,
            aim_spread: 0.1,
        };
        let s = soft_from_elite(&e);
        assert!((s.aim_mean + std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        let target = [(100.0 * e.aim_mean.cos()) as i32, (100.0 * e.aim_mean.sin()) as i32];
        assert!((ring_angle_of_target(target) - s.aim_mean).abs() < 2e-2);
    }

    #[test]
    fn character_records_keep_what_the_encoder_reads() {
        let mut c = CharacterObservation::at_rest(3);
        c.pos = Vec2::new(10.5, -3.25);
        c.vel = Vec2::new(-1.5, 2.0);
        c.grounded = true;
        c.is_frozen = true;
        c.hook_state = 4;
        c.hooked_player = 2;
        c.jumps_left = 1;
        c.freeze_ticks_remaining = 77;
        let back = char_rec(&c).to_observation();
        assert_eq!(back.pos, c.pos);
        assert_eq!(back.vel, c.vel);
        assert!(back.grounded && back.is_frozen && !back.is_deep_frozen);
        assert_eq!((back.hook_state, back.hooked_player, back.jumps_left), (4, 2, 1));
        assert_eq!(back.freeze_ticks_remaining, 77);
        assert_eq!(back.id, 3);
    }

    #[test]
    fn actions_round_trip_through_records() {
        let a = Action {
            direction: -1,
            jump: true,
            hook: false,
            fire: true,
            target: IVec2::new(-40, 25),
            wanted_weapon: None,
        };
        assert_eq!(action_of(&action_rec(&a)), a);
    }
}
