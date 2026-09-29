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

/// Wraps a brain and records every action it returns, with the tick it was decided on -- to turn a
/// closed-loop solution (e.g. the planner's) into an open-loop `[[reference]]` timeline.
pub struct RecordingBrain {
    inner: Box<dyn Brain>,
    log: ActionLog,
}

impl RecordingBrain {
    pub fn new(inner: Box<dyn Brain>) -> (Self, ActionLog) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (
            RecordingBrain {
                inner,
                log: log.clone(),
            },
            log,
        )
    }
}

impl Brain for RecordingBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.inner.reset(ctx);
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let a = self.inner.decide(obs);
        self.log.lock().expect("log lock").push((obs.tick, a));
        a
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.inner.decide_in(obs, view);
        self.log.lock().expect("log lock").push((obs.tick, a));
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
