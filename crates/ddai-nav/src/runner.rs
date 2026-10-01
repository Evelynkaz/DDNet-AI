//! `RouteRunner` (`route.ts:615-865`): walks a [`RouteStep`] list, one tee decision at a time.

use ddai_planner::types::{PlayerInput, TeeState, empty_input};

use crate::TILE_PX;
use crate::grid::NavGrid;
use crate::route::{MoveKind, RouteStep};

const REACHED_PX: f64 = 48.0;
const CENTRE_PX: f64 = 10.0;
const HOOK_MAX_TICKS: i32 = 100;
const STALL_TICKS: i64 = 100;
const FROZEN_GIVE_UP_TICKS: i32 = 25;
const FREEZE_MOVE_TICKS: i32 = 3 * 50 + 50;
const VETO_LIMIT: i32 = 12;
const THAW_LOOKAHEAD_STEPS: usize = 8;
const IN_THE_WAY_PX: f64 = 44.0;

/// `RunnerState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerState {
    Running,
    Arrived,
    Stuck,
    Replan,
}

/// `class RouteRunner`.
#[derive(Debug, Clone)]
pub struct RouteRunner {
    steps: Vec<RouteStep>,
    at: usize,
    hook_ticks: i32,
    best_dist: f64,
    best_tick: i64,
    frozen_ticks: i32,
    pub state: RunnerState,
    pub reason: String,
    kill_wanted: bool,
    kill_tick: i64,
    kill_died: bool,
    decisions: i64,
    vetoes: i32,
    grid: Option<NavGrid>,
    early_freeze: bool,
}

impl RouteRunner {
    /// `new RouteRunner(route, collision?, {earlyFreeze})`. `grid` is the route search's grid (TS
    /// recomputes it from the collision; here it is shared by the caller).
    pub fn new(route: Vec<RouteStep>, grid: Option<&NavGrid>, early_freeze: bool) -> RouteRunner {
        let empty = route.is_empty();
        RouteRunner {
            steps: route,
            at: 0,
            hook_ticks: 0,
            best_dist: f64::INFINITY,
            best_tick: -1,
            frozen_ticks: 0,
            state: if empty {
                RunnerState::Arrived
            } else {
                RunnerState::Running
            },
            reason: String::new(),
            kill_wanted: false,
            kill_tick: -1,
            kill_died: false,
            decisions: 0,
            vetoes: 0,
            grid: grid.cloned(),
            early_freeze,
        }
    }

    pub fn remaining(&self) -> usize {
        self.steps.len().saturating_sub(self.at)
    }

    pub fn current(&self) -> Option<&RouteStep> {
        self.steps.get(self.at)
    }

    pub fn steps(&self) -> &[RouteStep] {
        &self.steps
    }

    /// `failedMove`: the move key of the step the runner got stuck on.
    pub fn failed_move(&self) -> Option<i32> {
        if self.state == RunnerState::Stuck {
            self.steps.get(self.at).map(|s| s.move_key)
        } else {
            None
        }
    }

    pub fn awaiting_kill(&self) -> bool {
        self.state == RunnerState::Running && self.kill_tick >= 0 && !self.kill_died
    }

    pub fn respawned(&mut self) {
        if self.kill_tick >= 0 {
            self.kill_died = true;
        }
    }

    /// The guard refused a step: give up after [`VETO_LIMIT`] in a row.
    pub fn vetoed(&mut self) {
        if self.state != RunnerState::Running {
            return;
        }
        self.vetoes += 1;
        if self.vetoes <= VETO_LIMIT {
            return;
        }
        self.state = RunnerState::Stuck;
        self.reason = format!(
            "the guard refused step {}/{} ({}) {} times: no way back from the freeze",
            self.at + 1,
            self.steps.len(),
            self.steps.get(self.at).map_or("?", |s| kind_name(s.kind)),
            self.vetoes
        );
    }

    /// `freezeAhead(n)`: a planned freeze among the next `n` steps.
    pub fn freeze_ahead(&self, n: usize) -> bool {
        if self.state != RunnerState::Running {
            return false;
        }
        let from = self.at.saturating_sub(1);
        let to = self.steps.len().min(self.at + n);
        (from..to).any(|i| self.steps[i].freeze)
    }

    pub fn take_kill(&mut self) -> bool {
        std::mem::take(&mut self.kill_wanted)
    }

    fn freeze_planned(&self) -> bool {
        let step = |i: isize| -> bool { i >= 0 && self.steps.get(i as usize).is_some_and(|s| s.freeze) };
        let at = self.at as isize;
        step(at) || step(at - 1) || (self.early_freeze && step(at + 1))
    }

    fn advance(&mut self, tick: i64) {
        self.at += 1;
        self.vetoes = 0;
        self.kill_tick = -1;
        self.hook_ticks = 0;
        self.best_dist = f64::INFINITY;
        self.best_tick = tick;
    }

    fn reached(&self, me: &TeeState, i: usize) -> bool {
        let s = &self.steps[i];
        ddai_jsmath::hypot2(
            me.pos.x - (f64::from(s.x * TILE_PX) + 16.0),
            me.pos.y - (f64::from(s.y * TILE_PX) + 16.0),
        ) < REACHED_PX
    }

    /// `step(self, tick, others?)`.
    pub fn step(&mut self, me: &TeeState, tick: i64, others: Option<&[TeeState]>) -> PlayerInput {
        let mut out = empty_input();
        self.decisions += 1;
        if self.state == RunnerState::Running && !me.alive && self.kill_tick >= 0 {
            self.kill_died = true;
        }
        if self.state != RunnerState::Running || !me.alive {
            return out;
        }
        if me.frozen {
            self.frozen_ticks += 1;
            let limit = if self.freeze_planned() {
                FREEZE_MOVE_TICKS
            } else {
                FROZEN_GIVE_UP_TICKS
            };
            if self.frozen_ticks > limit {
                self.state = RunnerState::Stuck;
                self.reason = format!("froze on the way at step {}/{}", self.at + 1, self.steps.len());
            }
            self.best_tick = tick;
            return out;
        }
        let thawed = self.frozen_ticks > 0;
        self.frozen_ticks = 0;
        if thawed && self.freeze_planned() {
            let mut on: isize = -1;
            for k in self.at..self.steps.len().min(self.at + THAW_LOOKAHEAD_STEPS) {
                if self.reached(me, k) {
                    on = k as isize;
                }
            }
            if on < 0 {
                self.state = RunnerState::Replan;
                self.reason = format!(
                    "came out of the planned freeze at step {}/{} off the route",
                    self.at + 1,
                    self.steps.len()
                );
                return out;
            }
            while (self.at as isize) < on {
                self.advance(tick);
            }
        }
        let mut step = self.steps.get(self.at).cloned();
        while let Some(s) = &step {
            if s.tele {
                if self.at + 1 < self.steps.len() && self.reached(me, self.at + 1) {
                    self.advance(tick);
                    step = self.steps.get(self.at).cloned();
                    continue;
                }
                break;
            }
            if !self.reached(me, self.at) {
                break;
            }
            self.advance(tick);
            step = self.steps.get(self.at).cloned();
        }
        let Some(step) = step else {
            self.state = RunnerState::Arrived;
            return out;
        };
        if step.kind == MoveKind::Kill {
            if self.kill_tick < 0 {
                self.kill_wanted = true;
                self.kill_tick = tick;
                self.kill_died = false;
                return out;
            }
            if self.kill_died {
                self.state = RunnerState::Replan;
                self.reason = format!(
                    "respawned away from the spawn step {}/{} planned",
                    self.at + 1,
                    self.steps.len()
                );
                return out;
            }
            if tick - self.kill_tick > STALL_TICKS {
                self.state = RunnerState::Stuck;
                self.reason = format!("the /kill at step {}/{} never came", self.at + 1, self.steps.len());
            }
            return out;
        }
        let tx = f64::from(step.x * TILE_PX) + 16.0;
        let ty = f64::from(step.y * TILE_PX) + 16.0;
        let dx = tx - me.pos.x;
        let dy = ty - me.pos.y;
        let d = ddai_jsmath::hypot2(dx, dy);
        if self.best_tick < 0 {
            self.best_tick = tick;
        }
        if d < self.best_dist - 4.0 {
            self.best_dist = d;
            self.best_tick = tick;
        } else if tick - self.best_tick > STALL_TICKS {
            self.state = RunnerState::Stuck;
            self.reason = format!(
                "stuck {}px from step {}/{} ({})",
                ddai_jsmath::round(d),
                self.at + 1,
                self.steps.len(),
                kind_name(step.kind)
            );
            return out;
        }
        out.direction = if dx > CENTRE_PX {
            1
        } else if dx < -CENTRE_PX {
            -1
        } else {
            0
        };
        if step.kind == MoveKind::Hook
            && let Some((ax_t, ay_t)) = step.anchor
        {
            let ax = f64::from(ax_t * TILE_PX) + 16.0 - me.pos.x;
            let ay = f64::from(ay_t * TILE_PX) + 16.0 - me.pos.y;
            let n = {
                let h = ddai_jsmath::hypot2(ax, ay);
                if h == 0.0 { 1.0 } else { h }
            };
            out.target_x = ddai_jsmath::round(ax / n * 300.0);
            out.target_y = ddai_jsmath::round(ay / n * 300.0);
            out.hook = 1;
            self.hook_ticks += 1;
            if self.hook_ticks > HOOK_MAX_TICKS {
                self.state = RunnerState::Stuck;
                self.reason = format!(
                    "the rope at step {}/{} did not get us there",
                    self.at + 1,
                    self.steps.len()
                );
            }
            out.direction = if ax > CENTRE_PX {
                1
            } else if ax < -CENTRE_PX {
                -1
            } else {
                0
            };
            return out;
        }
        let dn = if d == 0.0 { 1.0 } else { d };
        out.target_x = ddai_jsmath::round(dx / dn * 300.0);
        out.target_y = ddai_jsmath::round(dy / dn * 300.0);
        let blocked = out.direction != 0
            && others.is_some_and(|o| tee_in_the_way(me, out.direction, o))
            && self.hop_clear(out.direction);
        if step.kind == MoveKind::Jump || step.leap || dy < -f64::from(TILE_PX) / 2.0 || blocked {
            let rising = me.vel.y < -0.5;
            out.jump = if rising || self.decisions % 2 == 0 { 1 } else { 0 };
        }
        out
    }

    /// `hopClear(direction)`: the tee ahead can be hopped over (three more steps on the same row, with
    /// ground at the end).
    fn hop_clear(&self, direction: i32) -> bool {
        let Some(step) = self.steps.get(self.at) else {
            return false;
        };
        let mut x = step.x;
        for k in 1..=2usize {
            let Some(s) = self.steps.get(self.at + k) else {
                return false;
            };
            if s.y != step.y || (s.x - x) * direction <= 0 {
                return false;
            }
            x = s.x;
        }
        let bx = x + direction;
        if let Some(next) = self.steps.get(self.at + 3)
            && next.y == step.y
            && next.x == bx
        {
            return true;
        }
        let Some(g) = &self.grid else { return false };
        if bx < 0 || bx >= g.width {
            return false;
        }
        let hazard = |i: usize| g.free[i] == 0 && g.solid[i] == 0;
        let i = g.idx(bx, step.y);
        if hazard(i) {
            return false;
        }
        !(step.y + 1 < g.height && g.free[i] == 1 && hazard(i + g.width as usize))
    }
}

fn kind_name(k: MoveKind) -> &'static str {
    match k {
        MoveKind::Walk => "walk",
        MoveKind::Fall => "fall",
        MoveKind::Jump => "jump",
        MoveKind::Hook => "hook",
        MoveKind::Kill => "kill",
    }
}

/// `teeInTheWay`: another tee closer than 44 px ahead on the same row.
fn tee_in_the_way(me: &TeeState, direction: i32, others: &[TeeState]) -> bool {
    for o in others {
        if o.id == me.id || !o.alive {
            continue;
        }
        let ahead = (o.pos.x - me.pos.x) * f64::from(direction);
        if ahead > 0.0 && ahead < IN_THE_WAY_PX && (o.pos.y - me.pos.y).abs() < f64::from(TILE_PX) / 2.0 {
            return true;
        }
    }
    false
}
