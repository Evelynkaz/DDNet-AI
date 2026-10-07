//! Critical-decision analysis (task 8.6): which decisions of the fly after the freeze decide the held block, and what differs there.
//!
//! The arena is deterministic, so a "fork" is a replay: the episode of a post-freeze start is played again from its recipe
//! ([`crate::bank::play_from_start`]) with a [`SwapBrain`] in the focal seat. The brain plays its **main** brain (the fly) and, at
//! chosen decisions after the freeze, plays the **alternative's** action instead (the planner's), wholly or one component at a
//! time (direction, jump, hook key, fire, aim). The main brain still decides every decision, so its recurrent state is its own.
//!
//! * The alternative's action at a decision is either **stored** (the planner's label for the state the fly was in, from a *shadow*
//!   run in which the planner is asked at every decision and the fly plays: one planner call per decision, the DAgger labelling) or
//!   **live** (the planner is asked at every decision and its action is played inside a window of decisions: the planner really
//!   takes over for a while).
//! * The reverse direction plays the planner as the main brain and swaps in the fly's stored actions.
//!
//! The record of every decision keeps both actions, so the analysis can say which component differs where the swap flips the outcome.

use std::sync::{Arc, Mutex};

use ddai_brain::{Action, Brain, Observation, ResetContext, WorldView};
use ddai_env::EnvError;
use ddai_env::config::{PlayerSpec, Rules};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::bank::{BankStart, play_from_start};
use crate::es::eval::BrainMaker;
use crate::experiment::Env;
use crate::types::ring_angle_of_target;

/// A part of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Component {
    Direction,
    Jump,
    Hook,
    Fire,
    Aim,
}

pub const COMPONENTS: [Component; 5] = [
    Component::Direction,
    Component::Jump,
    Component::Hook,
    Component::Fire,
    Component::Aim,
];

/// `base` with the listed components of `alt`.
pub fn combine(base: &Action, alt: &Action, components: &[Component]) -> Action {
    let mut a = *base;
    for c in components {
        match c {
            Component::Direction => a.direction = alt.direction,
            Component::Jump => a.jump = alt.jump,
            Component::Hook => a.hook = alt.hook,
            Component::Fire => a.fire = alt.fire,
            Component::Aim => a.target = alt.target,
        }
    }
    a
}

/// Which decisions (counted from the freeze, 0 = the first decision at or after it) play the alternative's action, and which parts of it.
#[derive(Debug, Clone, PartialEq)]
pub struct SwapPlan {
    pub from: usize,
    pub len: usize,
    pub components: Vec<Component>,
}

impl SwapPlan {
    pub fn covers(&self, k: usize) -> bool {
        k >= self.from && k < self.from + self.len
    }
}

/// One decision after the freeze.
#[derive(Debug, Clone, Copy)]
pub struct DecisionLog {
    pub k: usize,
    pub state: i32,
    pub main: Action,
    pub alt: Option<Action>,
    pub played: Action,
}

pub type Sink = Arc<Mutex<Vec<DecisionLog>>>;

/// Plays `main`, with the alternative's action swapped in as the plan says (module docs).
pub struct SwapBrain {
    main: Box<dyn Brain>,
    alt_live: Option<Box<dyn Brain>>,
    alt_stored: Option<Arc<Vec<Action>>>,
    handover: i32,
    plan: Option<SwapPlan>,
    sink: Sink,
    k: usize,
}

impl SwapBrain {
    pub fn new(
        main: Box<dyn Brain>,
        alt_live: Option<Box<dyn Brain>>,
        alt_stored: Option<Arc<Vec<Action>>>,
        handover: i32,
        plan: Option<SwapPlan>,
        sink: Sink,
    ) -> Self {
        SwapBrain {
            main,
            alt_live,
            alt_stored,
            handover,
            plan,
            sink,
            k: 0,
        }
    }
}

impl Brain for SwapBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        self.main.reset(ctx);
        if let Some(a) = &mut self.alt_live {
            a.reset(ctx);
        }
        self.k = 0;
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        self.decide_in(obs, None)
    }

    fn decide_in(&mut self, obs: &Observation, view: Option<&WorldView<'_>>) -> Action {
        let a = self.main.decide_in(obs, view);
        let live = self.alt_live.as_mut().map(|b| b.decide_in(obs, view));
        if obs.tick < self.handover {
            return a;
        }
        let k = self.k;
        self.k += 1;
        let alt = live.or_else(|| self.alt_stored.as_ref().and_then(|s| s.get(k).copied()));
        let played = match (&self.plan, alt) {
            (Some(p), Some(alt)) if p.covers(k) => combine(&a, &alt, &p.components),
            _ => a,
        };
        self.sink.lock().expect("sink lock").push(DecisionLog {
            k,
            state: obs.self_state.hook_state,
            main: a,
            alt,
            played,
        });
        played
    }

    fn name(&self) -> &str {
        self.main.name()
    }
}

/// What one run gave.
#[derive(Debug, Clone)]
pub struct Run {
    pub held: bool,
    pub self_out: bool,
    pub log: Vec<DecisionLog>,
}

/// Plays `start` with `main` in the focal seat and the plan of swaps.
#[allow(clippy::too_many_arguments)]
pub fn run_swap(
    env: &Env,
    rules: &Rules,
    start: &BankStart,
    main: &BrainMaker<'_>,
    alt_live: Option<&BrainMaker<'_>>,
    alt_stored: Option<Arc<Vec<Action>>>,
    plan: Option<SwapPlan>,
    window: i32,
    burn_in: i32,
) -> Result<Run, EnvError> {
    let arena = env
        .arenas
        .get(&start.arena)
        .ok_or_else(|| EnvError::new(format!("unknown arena {:?}", start.arena)))?;
    let sink: Sink = Arc::new(Mutex::new(Vec::new()));
    let live = match alt_live {
        Some(m) => Some(m()?),
        None => None,
    };
    let focal = Box::new(SwapBrain::new(
        main()?,
        live,
        alt_stored,
        start.end_tick,
        plan,
        sink.clone(),
    ));
    let opponent = env.models.factory()(&PlayerSpec::simple("scripted"))?;
    let o = play_from_start(arena, rules, start, focal, opponent, window, burn_in)?;
    let log = std::mem::take(&mut *sink.lock().expect("sink lock"));
    Ok(Run {
        held: o.held_block,
        self_out: o.focal_out_in_window,
        log,
    })
}

/// A compact action for the analysis files.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActRec {
    pub dir: i8,
    pub jump: bool,
    pub hook: bool,
    pub fire: bool,
    /// Ring angle of the aim, radians.
    pub aim: f32,
}

impl From<&Action> for ActRec {
    fn from(a: &Action) -> Self {
        ActRec {
            dir: a.direction.clamp(-1, 1) as i8,
            jump: a.jump,
            hook: a.hook,
            fire: a.fire,
            aim: ring_angle_of_target([a.target.x, a.target.y]),
        }
    }
}

/// One decision of the shadow run: the main brain's action and the alternative's at the same state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShadowStep {
    pub k: usize,
    pub state: i32,
    pub main: ActRec,
    pub alt: Option<ActRec>,
}

/// The outcome of one swap experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwapResult {
    pub from: usize,
    pub len: usize,
    pub components: Vec<Component>,
    pub held: bool,
    pub self_out: bool,
}

/// Everything measured for one start in one direction (`forward`: the fly is the main brain; reverse: the planner).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartAnalysis {
    pub arena: String,
    pub seed: u64,
    /// `V`, `B` or `H`.
    pub class: char,
    pub base_held: bool,
    pub base_self_out: bool,
    pub shadow: Vec<ShadowStep>,
    /// Single-decision swaps of the whole action, by decision.
    pub singles: Vec<SwapResult>,
    /// Single-component swaps at decisions where the whole-action swap flipped the outcome.
    pub components: Vec<SwapResult>,
    /// Windows during which the alternative really plays (live).
    pub windows: Vec<SwapResult>,
}

/// What to measure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalSpec {
    /// Single-decision swaps at every `single_stride`-th decision of the first `single_first` decisions and, beyond, every `single_late_stride`-th.
    pub single_first: usize,
    pub single_stride: usize,
    pub single_late_stride: usize,
    /// The first this many critical decisions get single-component swaps.
    pub component_decisions: usize,
    /// `(from, len)` of live windows.
    pub windows: Vec<(usize, usize)>,
}

impl Default for CriticalSpec {
    fn default() -> Self {
        CriticalSpec {
            single_first: 40,
            single_stride: 1,
            single_late_stride: 4,
            component_decisions: 6,
            windows: vec![
                (0, 1),
                (0, 4),
                (0, 12),
                (0, 40),
                (0, 125),
                (2, 4),
                (4, 4),
                (8, 4),
                (16, 4),
                (32, 4),
                (8, 12),
                (32, 12),
            ],
        }
    }
}

fn shadow_of(log: &[DecisionLog]) -> Vec<ShadowStep> {
    log.iter()
        .map(|d| ShadowStep {
            k: d.k,
            state: d.state,
            main: ActRec::from(&d.main),
            alt: d.alt.as_ref().map(ActRec::from),
        })
        .collect()
}

fn class_of(s: &BankStart) -> char {
    crate::hook_eval::start_class(s)
}

/// The analysis of one start with `main` as the brain that plays and `alt` as the one swapped in. `main` and `alt` are the brains of
/// the forward direction (fly, planner) or of the reverse one (planner, fly).
#[allow(clippy::too_many_arguments)]
pub fn analyse_start(
    env: &Env,
    rules: &Rules,
    start: &BankStart,
    main: &BrainMaker<'_>,
    alt: &BrainMaker<'_>,
    spec: &CriticalSpec,
    window: i32,
    burn_in: i32,
) -> Result<StartAnalysis, EnvError> {
    // The shadow run: main plays, alt is asked at every decision (live), nothing is swapped.
    let base = run_swap(env, rules, start, main, Some(alt), None, None, window, burn_in)?;
    let stored: Arc<Vec<Action>> = Arc::new(base.log.iter().map(|d| d.alt.unwrap_or_else(Action::neutral)).collect());
    let n = base.log.len();
    let mut out = StartAnalysis {
        arena: start.arena.clone(),
        seed: start.seed,
        class: class_of(start),
        base_held: base.held,
        base_self_out: base.self_out,
        shadow: shadow_of(&base.log),
        singles: Vec::new(),
        components: Vec::new(),
        windows: Vec::new(),
    };
    let all = COMPONENTS.to_vec();
    let result = |plan: &SwapPlan, r: &Run| SwapResult {
        from: plan.from,
        len: plan.len,
        components: plan.components.clone(),
        held: r.held,
        self_out: r.self_out,
    };
    for k in 0..n {
        let early = k < spec.single_first;
        let stride = if early {
            spec.single_stride
        } else {
            spec.single_late_stride
        };
        if !k.is_multiple_of(stride.max(1)) {
            continue;
        }
        let plan = SwapPlan {
            from: k,
            len: 1,
            components: all.clone(),
        };
        let r = run_swap(
            env,
            rules,
            start,
            main,
            None,
            Some(stored.clone()),
            Some(plan.clone()),
            window,
            burn_in,
        )?;
        out.singles.push(result(&plan, &r));
    }
    let critical: Vec<usize> = out
        .singles
        .iter()
        .filter(|s| s.held && !base.held)
        .map(|s| s.from)
        .take(spec.component_decisions)
        .collect();
    for &k in &critical {
        for c in COMPONENTS {
            let plan = SwapPlan {
                from: k,
                len: 1,
                components: vec![c],
            };
            let r = run_swap(
                env,
                rules,
                start,
                main,
                None,
                Some(stored.clone()),
                Some(plan.clone()),
                window,
                burn_in,
            )?;
            out.components.push(result(&plan, &r));
        }
    }
    for &(from, len) in &spec.windows {
        let plan = SwapPlan {
            from,
            len,
            components: all.clone(),
        };
        let r = run_swap(
            env,
            rules,
            start,
            main,
            Some(alt),
            None,
            Some(plan.clone()),
            window,
            burn_in,
        )?;
        out.windows.push(result(&plan, &r));
    }
    Ok(out)
}

/// [`analyse_start`] over `starts`, in parallel; results in the order of the starts.
#[allow(clippy::too_many_arguments)]
pub fn analyse_starts(
    env: &Env,
    pool: &rayon::ThreadPool,
    rules: &Rules,
    starts: &[&BankStart],
    main: &BrainMaker<'_>,
    alt: &BrainMaker<'_>,
    spec: &CriticalSpec,
    window: i32,
    burn_in: i32,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<Vec<StartAnalysis>, String> {
    let done = std::sync::atomic::AtomicUsize::new(0);
    let r: Vec<Result<StartAnalysis, EnvError>> = pool.install(|| {
        starts
            .par_iter()
            .map(|s| {
                let a = analyse_start(env, rules, s, main, alt, spec, window, burn_in);
                progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
                a
            })
            .collect()
    });
    r.into_iter().collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_brain::IVec2;

    fn act(dir: i32, jump: bool, hook: bool, fire: bool, x: i32) -> Action {
        Action {
            direction: dir,
            jump,
            hook,
            fire,
            target: IVec2::new(x, 0),
            wanted_weapon: None,
        }
    }

    #[test]
    fn combine_takes_only_the_listed_components_of_the_alternative() {
        let (a, b) = (act(1, false, false, false, 100), act(-1, true, true, true, -100));
        assert_eq!(combine(&a, &b, &[]), a);
        assert_eq!(combine(&a, &b, &COMPONENTS), b);
        let h = combine(&a, &b, &[Component::Hook]);
        assert!(h.hook && h.direction == 1 && !h.jump && !h.fire && h.target.x == 100);
        let m = combine(&a, &b, &[Component::Direction, Component::Aim]);
        assert!(m.direction == -1 && m.target.x == -100 && !m.hook && !m.jump);
    }

    #[test]
    fn a_plan_covers_exactly_its_window() {
        let p = SwapPlan {
            from: 3,
            len: 2,
            components: vec![Component::Hook],
        };
        assert_eq!(
            (2..7).map(|k| p.covers(k)).collect::<Vec<_>>(),
            [false, true, true, false, false]
        );
    }

    struct Fixed(Action, &'static str);
    impl Brain for Fixed {
        fn reset(&mut self, _: &ResetContext) {}
        fn decide(&mut self, _: &Observation) -> Action {
            self.0
        }
        fn name(&self) -> &str {
            self.1
        }
    }

    fn obs(tick: i32) -> Observation {
        Observation {
            map: Arc::new(ddai_physics::map::MapData {
                width: 2,
                height: 2,
                game: vec![Default::default(); 4],
                front: None,
                tele: None,
                speedup: None,
                switch: None,
                tune: None,
                settings: Vec::new(),
            }),
            tick,
            self_state: ddai_brain::CharacterObservation::at_rest(0),
            others: vec![],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        }
    }

    /// Before the freeze the main brain's action passes through and nothing is logged; from the freeze on the decisions are counted from
    /// zero, a stored alternative is played inside the plan only, and a live alternative is asked at every decision.
    #[test]
    fn the_swap_brain_counts_from_the_freeze_and_swaps_only_inside_the_plan() {
        let main = act(1, false, false, false, 100);
        let alt = act(-1, true, true, false, -100);
        let sink: Sink = Arc::new(Mutex::new(Vec::new()));
        let stored = Arc::new(vec![alt; 10]);
        let mut b = SwapBrain::new(
            Box::new(Fixed(main, "m")),
            None,
            Some(stored),
            10,
            Some(SwapPlan {
                from: 1,
                len: 2,
                components: vec![Component::Hook, Component::Direction],
            }),
            sink.clone(),
        );
        assert_eq!(
            b.decide(&obs(6)),
            main,
            "before the freeze: the main action, not logged"
        );
        let played: Vec<Action> = [10, 12, 14, 16].iter().map(|&t| b.decide(&obs(t))).collect();
        assert_eq!(played[0], main);
        assert!(played[1].hook && played[1].direction == -1 && !played[1].jump && played[1].target.x == 100);
        assert_eq!(played[2], played[1]);
        assert_eq!(played[3], main, "after the plan");
        let log = sink.lock().unwrap();
        assert_eq!(log.iter().map(|d| d.k).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert!(log.iter().all(|d| d.alt == Some(alt) && d.main == main));
        drop(log);

        // A live alternative is asked at every decision, also before the freeze (it warms up), and its action is the logged alternative.
        let sink: Sink = Arc::new(Mutex::new(Vec::new()));
        let mut b = SwapBrain::new(
            Box::new(Fixed(main, "m")),
            Some(Box::new(Fixed(alt, "a"))),
            None,
            0,
            Some(SwapPlan {
                from: 0,
                len: 1,
                components: COMPONENTS.to_vec(),
            }),
            sink.clone(),
        );
        assert_eq!(b.decide(&obs(0)), alt);
        assert_eq!(b.decide(&obs(2)), main);
        assert_eq!(sink.lock().unwrap().len(), 2);
    }
}
