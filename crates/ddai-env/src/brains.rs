//! Small scripted [`Brain`]s the arena itself needs: a fixed timeline of actions. Used for the
//! non-subject tees of technique scenarios, for the reference solutions that prove a scenario is
//! solvable, and by tests to force known situations.

use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, IVec2, Observation, ResetContext, WorldView};
use serde::Deserialize;

fn d_dir() -> i32 {
    0
}

/// One entry of a timeline as written in scenario TOML: from world tick `from` on (until the
/// next entry) the player holds this action.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptStep {
    #[serde(default)]
    pub from: i32,
    /// `-1`, `0` or `1`.
    #[serde(default = "d_dir")]
    pub direction: i32,
    #[serde(default)]
    pub jump: bool,
    #[serde(default)]
    pub hook: bool,
    /// A `true` level is a fresh press on every decision (see the crate docs on fire).
    #[serde(default)]
    pub fire: bool,
    /// Aim vector relative to the tee, in pixels. `[0, 0]` is sent as `[0, -1]`.
    #[serde(default)]
    pub aim: [i32; 2],
    /// Weapon slot to switch to: 0 = hammer, 1 = gun.
    #[serde(default)]
    pub weapon: Option<i32>,
}

impl ScriptStep {
    pub fn action(&self) -> Action {
        Action {
            direction: self.direction,
            jump: self.jump,
            hook: self.hook,
            fire: self.fire,
            target: IVec2::new(self.aim[0], self.aim[1]),
            wanted_weapon: self.weapon,
        }
    }
}

/// Plays back `(from_tick, action)` pairs: the action of the last entry with `from_tick <= tick`,
/// neutral before the first.
#[derive(Debug, Clone)]
pub struct TimelineBrain {
    steps: Vec<(i32, Action)>,
    name: String,
}

impl TimelineBrain {
    pub fn new(name: &str, steps: Vec<(i32, Action)>) -> Self {
        let mut steps = steps;
        steps.sort_by_key(|s| s.0);
        TimelineBrain {
            steps,
            name: name.to_string(),
        }
    }

    pub fn from_script(name: &str, script: &[ScriptStep]) -> Self {
        Self::new(name, script.iter().map(|s| (s.from, s.action())).collect())
    }
}

impl Brain for TimelineBrain {
    fn reset(&mut self, _ctx: &ResetContext) {}

    fn decide(&mut self, obs: &Observation) -> Action {
        self.steps
            .iter()
            .rev()
            .find(|(t, _)| *t <= obs.tick)
            .map_or(Action::neutral(), |(_, a)| *a)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A shared log of `(tick, action)` decisions.
pub type ActionLog = Arc<Mutex<Vec<(i32, Action)>>>;

/// A shared log of `(tick, telemetry JSON)` taken right after each decision.
pub type TelemetryLog = Arc<Mutex<Vec<(i32, String)>>>;

/// Wraps a brain and records every action it returns, with the tick it was decided on -- to turn a
/// closed-loop solution (e.g. the planner's) into an open-loop `[[reference]]` timeline. With
/// [`RecordingBrain::with_telemetry`] it also keeps the inner brain's telemetry after each
/// decision (which candidate source and technique the hybrid picked, and why).
pub struct RecordingBrain {
    inner: Box<dyn Brain>,
    log: ActionLog,
    telemetry: Option<TelemetryLog>,
}

impl RecordingBrain {
    pub fn new(inner: Box<dyn Brain>) -> (Self, ActionLog) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (
            RecordingBrain {
                inner,
                log: log.clone(),
                telemetry: None,
            },
            log,
        )
    }

    pub fn with_telemetry(inner: Box<dyn Brain>) -> (Self, ActionLog, TelemetryLog) {
        let (mut b, log) = Self::new(inner);
        let t: TelemetryLog = Arc::new(Mutex::new(Vec::new()));
        b.telemetry = Some(t.clone());
        (b, log, t)
    }

    fn record(&self, tick: i32, a: Action) {
        self.log.lock().expect("log lock").push((tick, a));
        if let (Some(t), Some(json)) = (&self.telemetry, self.inner.telemetry()) {
            t.lock().expect("telemetry lock").push((tick, json));
        }
    }
}

impl Brain for RecordingBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let a = self.inner.decide(obs);
        self.record(obs.tick, a);
        a
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.inner.decide_in(obs, view);
        self.record(obs.tick, a);
        a
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn telemetry(&self) -> Option<String> {
        self.inner.telemetry()
    }
}

/// `[[reference]]` TOML for a recorded action list: consecutive decisions with identical actions
/// are merged into one step.
pub fn actions_to_toml(actions: &[(i32, Action)]) -> String {
    let mut out = String::new();
    let mut last: Option<Action> = None;
    for (tick, a) in actions {
        if last == Some(*a) {
            continue;
        }
        last = Some(*a);
        out.push_str(&format!("[[reference]]\nfrom = {tick}\n"));
        if a.direction != 0 {
            out.push_str(&format!("direction = {}\n", a.direction));
        }
        if a.jump {
            out.push_str("jump = true\n");
        }
        if a.hook {
            out.push_str("hook = true\n");
        }
        if a.fire {
            out.push_str("fire = true\n");
        }
        out.push_str(&format!("aim = [{}, {}]\n", a.target.x, a.target.y));
        if let Some(w) = a.wanted_weapon {
            out.push_str(&format!("weapon = {w}\n"));
        }
        out.push('\n');
    }
    out
}

/// The live bot's wayblock hints for an arena player (task 4.2): wraps a brain so that, while the tee
/// stands in the hall of the held side, it is told what `ddai-bot`'s wayblock hook tells the planner
/// (`WB_PLAN_OVERRIDES` and the band, `LiveContext::wb`). Everything else passes through.
pub struct WbHintBrain {
    inner: Box<dyn Brain>,
    def: ddai_nav::wayblock::WbDef,
    side: ddai_nav::wayblock::WbSide,
    strong: bool,
    name: String,
}

impl WbHintBrain {
    pub fn new(
        inner: Box<dyn Brain>,
        def: ddai_nav::wayblock::WbDef,
        side: ddai_nav::wayblock::WbSide,
        strong: bool,
    ) -> Self {
        let name = format!("{}+wb", inner.name());
        WbHintBrain {
            inner,
            def,
            side,
            strong,
            name,
        }
    }

    fn hints(&self, pos: ddai_physics::vmath::Vec2<f32>) -> ddai_brain::WbHints {
        let (tx, ty) = ((pos.x / 32.0).trunc() as i32, (pos.y / 32.0).trunc() as i32);
        if !self.def.in_hall(self.side, tx, ty) {
            return ddai_brain::WbHints::default();
        }
        ddai_brain::WbHints {
            in_hall: true,
            strong: self.strong,
            band: ddai_nav::wayblock::wb_band(&self.def, self.side),
        }
    }
}

impl Brain for WbHintBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let wb = self.hints(obs.self_state.pos);
        self.inner.set_live_context(&ddai_brain::LiveContext {
            wb,
            ..Default::default()
        });
        self.inner.decide(obs)
    }

    fn decide_in(&mut self, obs: &Observation, world: Option<&WorldView<'_>>) -> Action {
        let wb = self.hints(obs.self_state.pos);
        self.inner.set_live_context(&ddai_brain::LiveContext {
            wb,
            ..Default::default()
        });
        self.inner.decide_in(obs, world)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn telemetry(&self) -> Option<String> {
        self.inner.telemetry()
    }
}

#[cfg(test)]
mod wb_hint_tests {
    use super::*;
    use ddai_nav::wayblock::{WbSide, wayblocks};
    use ddai_physics::vmath::Vec2;

    struct Nothing;
    impl Brain for Nothing {
        fn reset(&mut self, _ctx: &ResetContext) {}
        fn decide(&mut self, _obs: &Observation) -> Action {
            Action::neutral()
        }
        fn name(&self) -> &str {
            "nothing"
        }
    }

    #[test]
    fn hints_exist_only_while_the_tee_stands_in_the_hall() {
        let def = wayblocks().into_iter().next().unwrap();
        let zone = def.left.zone[0];
        let b = WbHintBrain::new(Box::new(Nothing), def, WbSide::Left, true);
        assert_eq!(b.name(), "nothing+wb");
        let inside = b.hints(Vec2::new((zone.x0 * 32 + 40) as f32, (zone.y0 * 32 + 40) as f32));
        assert!(inside.in_hall && inside.strong && inside.band.is_some(), "{inside:?}");
        let outside = b.hints(Vec2::new(100.0, 100.0));
        assert_eq!(outside, ddai_brain::WbHints::default());
    }
}
