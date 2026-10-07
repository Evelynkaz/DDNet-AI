//! Task 3.15 (E-028, opt-in): the **lag-window model**, a learned prediction of the victim's inputs
//! for the ticks the input lag hides.
//!
//! The planning world is rolled forward through the lag window (the ticks between the snapshot and the
//! first tick our decision can act on) with our own in-flight inputs, which are known, and the
//! victim's inputs, which are not: until now the victim "keeps doing what the snapshot shows"
//! ([`crate::brains::enemy_input_from_tee`]). A [`WindowModel`] predicts those inputs instead. The trait
//! lives here so the planner stays light; the network that implements it (`ddai-oppnet`) is built
//! by the arena's brain factory and handed to [`crate::hybrid::HybridBrain::set_window_model`].
//!
//! The model is a *predictor of decisions*: it sees what a live snapshot shows (the tees' states, a
//! short history of them) plus our own in-flight inputs, never the victim's raw inputs.

use ddai_physics::core::PlayerInput as WireInput;
use ddai_physics::world::World;

use crate::types::PlayerInput;

/// What the model predicts for one tick of the window: the victim's input as levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictedInput {
    /// `-1`, `0` or `1`.
    pub direction: i32,
    pub jump: bool,
    pub hook: bool,
    /// A fresh fire press (a hammer swing) at this tick.
    pub press: bool,
    /// Absolute aim angle (radians, `atan2(target_y, target_x)`).
    pub aim: f64,
}

/// What a model may read at a decision.
pub struct WindowCtx<'a> {
    /// The exact world at the snapshot tick (before the lag window is rolled).
    pub world: &'a World<f32>,
    pub self_id: i32,
    pub victim_id: i32,
    /// Our inputs already sent and not yet applied, one per tick of the window (the ticks
    /// `world.tick ..`, `lag_ticks` of them).
    pub in_flight: &'a [WireInput],
}

/// A predictor of the victim's inputs over the lag window.
pub trait WindowModel {
    fn name(&self) -> &str;
    /// New episode: forgets the observation history.
    fn reset(&mut self) {}
    /// Called at every decision made from an exact world (also with an empty window, so the model's
    /// history stays complete). Fills `out[k]` with the predicted input of window tick `k` (`k <
    /// ctx.in_flight.len()`), or leaves `None` where it has no prediction (the victim then keeps its
    /// input, as without a model). `out` has `ctx.in_flight.len()` entries, all `None` on entry.
    fn predict(&mut self, ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]);
    /// What one call costs the work clock, in tee-tick equivalents (a network does not simulate physics).
    fn work_units(&self) -> u64 {
        0
    }
}

/// The planner input a [`PredictedInput`] stands for: `hold` (what the snapshot shows) with the predicted direction, jump, hook and aim, and the fire
/// counter `fire` moved on by a press when one is predicted (`fire` is the victim's own counter, carried from tick to tick: a press is a move
/// to the next odd value, never a phantom press from a stale counter and never a release).
pub fn input_from_prediction(p: &PredictedInput, hold: &PlayerInput, fire: &mut i32) -> PlayerInput {
    let mut input = *hold;
    input.direction = p.direction.clamp(-1, 1);
    input.jump = i32::from(p.jump);
    input.hook = i32::from(p.hook);
    input.target_x = ddai_jsmath::round(ddai_jsmath::cos(p.aim) * 300.0);
    input.target_y = ddai_jsmath::round(ddai_jsmath::sin(p.aim) * 300.0);
    if p.press {
        *fire += if *fire & 1 != 0 { 2 } else { 1 };
    }
    input.fire = *fire;
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::empty_input;

    #[test]
    fn a_predicted_press_moves_the_counter_to_the_next_odd_value_and_nothing_else_moves_it() {
        let hold = empty_input();
        let mut fire = 4;
        let pred = |press| PredictedInput {
            direction: -1,
            jump: true,
            hook: true,
            press,
            aim: 0.0,
        };
        let a = input_from_prediction(&pred(false), &hold, &mut fire);
        assert_eq!(
            (a.fire, a.direction, a.jump, a.hook, a.target_x, a.target_y),
            (4, -1, 1, 1, 300.0, 0.0)
        );
        let b = input_from_prediction(&pred(true), &hold, &mut fire);
        assert_eq!(b.fire, 5, "even -> odd: a press");
        let c = input_from_prediction(&pred(true), &hold, &mut fire);
        assert_eq!(c.fire, 7, "odd -> next odd: release and press in one step");
        let d = input_from_prediction(&pred(false), &hold, &mut fire);
        assert_eq!(d.fire, 7, "held");
    }
}
