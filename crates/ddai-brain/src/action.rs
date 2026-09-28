//! [`Action`]: the shared action space every [`crate::Brain`] decides in (task 7.3, acceptance
//! criterion 1) — direction/jump/hook/fire/aim/weapon, converted to and from the wire-format
//! [`ddai_physics::core::PlayerInput`] (`CNetObj_PlayerInput`, DDNet 20.1 semantics).

use ddai_physics::core::PlayerInput;

/// An integer 2D point (DDNet's aim-target convention: whole pixels, relative to the tee).
/// `ddai_physics::vmath::Vec2<R>` is generic only over [`ddai_physics::real::Real`] (`f32`/`f64`),
/// not `i32`, so `Action::target` gets its own tiny integer vector instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IVec2 {
    pub x: i32,
    pub y: i32,
}

impl IVec2 {
    pub const fn new(x: i32, y: i32) -> Self {
        IVec2 { x, y }
    }
}

/// The bot's action for one decision — the same space the planner, a scripted bot, the fly, and
/// (later) human-replay all target, so nothing downstream (an arena, a live connector) needs to
/// know which of them produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    /// `-1`/`0`/`1`: desired horizontal movement.
    pub direction: i32,
    /// Jump key held this decision (a *level*, not an edge — see the module doc comment on
    /// [`Action::to_player_input`]).
    pub jump: bool,
    /// Hook key held this decision.
    pub hook: bool,
    /// Fire key "wants to fire this decision" — a level, even though the wire format
    /// (`CNetObj_PlayerInput::m_Fire`) is a press-*counter* (odd = held); see
    /// [`Action::to_player_input`]'s doc comment for why turning this into that counter is left to
    /// the caller.
    pub fire: bool,
    /// Aim target, in DDNet's own convention: an integer vector *relative to the tee*
    /// (`CNetObj_PlayerInput::(m_TargetX, m_TargetY)`), not an absolute position or an angle.
    pub target: IVec2,
    /// `Some(weapon_index)` to request switching to that weapon slot this decision, `None` to
    /// leave the current weapon alone. Matches `character.cpp`'s `HandleWeaponSwitch`, which reads
    /// the wire's `m_WantedWeapon` as `1`-based-or-`0`-for-none (`WantedWeapon = m_WantedWeapon -
    /// 1` only when `m_WantedWeapon > 0`) — [`Action::to_player_input`]/[`Action::from_player_input`]
    /// do that `+1`/`-1` translation so this field itself stays a plain, `0`-based weapon index.
    pub wanted_weapon: Option<i32>,
}

impl Action {
    /// A neutral action: no movement, nothing held, aiming straight up (review round 1, F16: an
    /// earlier revision's doc comment said "down" — DDNet/Teeworlds screen `y` grows *downward*,
    /// so `target: (0, -1)` is up, not down; an arbitrary but valid nonzero default either way —
    /// see [`Action::to_player_input`]'s doc comment on why `target` is never `(0, 0)`), no
    /// weapon switch requested.
    pub fn neutral() -> Self {
        Action {
            direction: 0,
            jump: false,
            hook: false,
            fire: false,
            target: IVec2::new(0, -1),
            wanted_weapon: None,
        }
    }

    /// Converts to the wire format (`CNetObj_PlayerInput`, DDNet 20.1 semantics).
    ///
    /// **Edge semantics are the caller's concern, documented here rather than guessed at:**
    /// `jump`/`hook` are read by core-level physics as plain "held this tick" booleans (`gamecore.cpp`
    /// checks `Input().m_Jump != 0` / `Input().m_Hook != 0`, no parity) — a level, exactly what this
    /// method produces from `self.jump`/`self.hook`. `fire`, on the other hand, is read by
    /// `character.cpp`'s weapon-firing logic as a press-*counter*: only the low bit matters
    /// (`m_Fire & 1` = "held right now"), but a semi-automatic weapon fires once per *transition*
    /// into that bit being set, not once per tick it stays set — so a caller sustaining `fire =
    /// true` for several decisions in a row must itself track the previous counter and increment it
    /// only on true 0->1 transitions (this method has no memory of the previous call, so it always
    /// emits either `0` or `1` for `m_Fire`, i.e. "just pressed" every single time `fire` is
    /// `true` — correct for one shot, and it is the caller's job to turn a sustained `fire` signal
    /// into the right increasing-counter sequence if it wants held-fire semantics for an automatic
    /// weapon).
    ///
    /// `target` is substituted with `(0, -1)` when it is exactly `(0, 0)`: DDNet's own wire format
    /// never stores an all-zero aim vector (`docs/formats.md` §2's `resolve_input` note — the
    /// client itself falls back to an explicit nonzero target rather than ever sending `(0, 0)`).
    pub fn to_player_input(&self) -> PlayerInput {
        let target = if self.target == IVec2::new(0, 0) {
            IVec2::new(0, -1)
        } else {
            self.target
        };
        PlayerInput {
            direction: self.direction,
            target_x: target.x,
            target_y: target.y,
            jump: i32::from(self.jump),
            fire: i32::from(self.fire),
            hook: i32::from(self.hook),
            player_flags: 0,
            wanted_weapon: self.wanted_weapon.map_or(0, |w| w + 1),
            next_weapon: 0,
            prev_weapon: 0,
        }
    }

    /// Converts from the wire format. `fire`/`jump`/`hook` are read as "nonzero" (matching
    /// [`Action::to_player_input`]'s own level convention — see its doc comment for `fire`'s
    /// caveat: a wire value with the low bit set decodes to `fire: true` regardless of the
    /// counter's higher bits, since those only matter for edge detection across *several* inputs,
    /// which a single [`PlayerInput`] can't carry).
    pub fn from_player_input(input: &PlayerInput) -> Self {
        Action {
            direction: input.direction,
            jump: input.jump != 0,
            hook: input.hook != 0,
            fire: (input.fire & 1) != 0,
            target: IVec2::new(input.target_x, input.target_y),
            wanted_weapon: (input.wanted_weapon > 0).then_some(input.wanted_weapon - 1),
        }
    }

    /// X-mirrors this action (task 7.3, acceptance criterion 1's "mirror helpers"; acceptance
    /// criterion 8's mirror-symmetry test uses this): flips `direction` and `target.x`; `jump`/
    /// `hook`/`fire`/`wanted_weapon`/`target.y` are unaffected by a horizontal flip.
    pub fn mirror_x(&self) -> Self {
        Action {
            direction: -self.direction,
            target: IVec2::new(-self.target.x, self.target.y),
            ..*self
        }
    }
}

impl Default for Action {
    fn default() -> Self {
        Action::neutral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_player_input_round_trips_through_from_player_input() {
        let a = Action {
            direction: 1,
            jump: true,
            hook: false,
            fire: true,
            target: IVec2::new(50, -30),
            wanted_weapon: Some(3),
        };
        let pi = a.to_player_input();
        assert_eq!(pi.direction, 1);
        assert_eq!(pi.jump, 1);
        assert_eq!(pi.hook, 0);
        assert_eq!(pi.fire, 1);
        assert_eq!(pi.target_x, 50);
        assert_eq!(pi.target_y, -30);
        assert_eq!(pi.wanted_weapon, 4);

        let back = Action::from_player_input(&pi);
        assert_eq!(back.direction, a.direction);
        assert_eq!(back.jump, a.jump);
        assert_eq!(back.hook, a.hook);
        assert_eq!(back.fire, a.fire);
        assert_eq!(back.target, a.target);
        assert_eq!(back.wanted_weapon, a.wanted_weapon);
    }

    #[test]
    fn no_wanted_weapon_round_trips_as_wire_zero() {
        let a = Action {
            wanted_weapon: None,
            ..Action::neutral()
        };
        let pi = a.to_player_input();
        assert_eq!(pi.wanted_weapon, 0);
        assert_eq!(Action::from_player_input(&pi).wanted_weapon, None);
    }

    #[test]
    fn exact_zero_target_is_never_sent_on_the_wire() {
        let a = Action {
            target: IVec2::new(0, 0),
            ..Action::neutral()
        };
        let pi = a.to_player_input();
        assert_ne!((pi.target_x, pi.target_y), (0, 0));
    }

    #[test]
    fn fire_wire_value_only_checks_the_low_bit() {
        let pi = PlayerInput {
            fire: 5, // odd, higher bits set -- still "held" per the low-bit convention
            ..Action::neutral().to_player_input()
        };
        assert!(Action::from_player_input(&pi).fire);
        let pi2 = PlayerInput { fire: 4, ..pi };
        assert!(!Action::from_player_input(&pi2).fire);
    }

    #[test]
    fn mirror_x_flips_direction_and_target_x_only() {
        let a = Action {
            direction: 1,
            jump: true,
            hook: true,
            fire: true,
            target: IVec2::new(40, -20),
            wanted_weapon: Some(2),
        };
        let m = a.mirror_x();
        assert_eq!(m.direction, -1);
        assert_eq!(m.target, IVec2::new(-40, -20));
        assert_eq!(m.jump, a.jump);
        assert_eq!(m.hook, a.hook);
        assert_eq!(m.fire, a.fire);
        assert_eq!(m.wanted_weapon, a.wanted_weapon);
        // Mirroring twice returns to the original.
        assert_eq!(m.mirror_x(), a);
    }
}
