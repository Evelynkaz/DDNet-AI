//! The online guard of the live window model (task 3.17, D-111): the model is judged against "hold" on what the opponent really did.
//!
//! Every resolved window (a decision whose prediction could be compared with the snapshots that followed it) adds two costs, the model's and
//! the hold baseline's, over the same `n` observed samples. The guard keeps the last [`GuardConfig::windows`] windows. It starts **active**
//! (the model drives the window); once [`GuardConfig::min_windows`] are in, it falls back to hold when the model's summed cost exceeds the
//! baseline's by more than [`GuardConfig::margin`] (relative). While fallen back the model keeps predicting in the shadow and keeps being
//! scored, so the guard judges the same evidence in both states; after [`GuardConfig::retry_after`] windows it tries the model again, but
//! only if the shadow cost is not worse than the baseline's by more than [`GuardConfig::retry_margin`] (a model that is still losing is not
//! given the window back just because time has passed).

/// The guard's knobs. [`GuardConfig::default`] is what the live bot runs with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardConfig {
    /// Windows kept (the judging span). One window per brain decision: about 25 a second.
    pub windows: usize,
    /// Windows needed before the first verdict (the model drives the window until then).
    pub min_windows: usize,
    /// Fall back when `model > hold * (1 + margin)`.
    pub margin: f32,
    /// Windows spent in fallback before the model may come back.
    pub retry_after: u32,
    /// The model may come back when `model <= hold * (1 + retry_margin)`.
    pub retry_margin: f32,
}

impl Default for GuardConfig {
    fn default() -> Self {
        GuardConfig {
            windows: 250,
            min_windows: 100,
            margin: 0.05,
            retry_after: 500,
            retry_margin: 0.0,
        }
    }
}

impl GuardConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.windows == 0 || self.min_windows == 0 || self.min_windows > self.windows {
            return Err("guard: 0 < min_windows <= windows".into());
        }
        if !(self.margin.is_finite() && self.margin >= 0.0 && self.retry_margin.is_finite() && self.retry_margin >= 0.0)
        {
            return Err("guard: margins must be finite and >= 0".into());
        }
        Ok(())
    }
}

/// Whose predictions drive the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardState {
    /// The model drives the window.
    Active,
    /// Hold drives it; the model is scored in the shadow.
    Fallback,
}

impl GuardState {
    pub fn name(self) -> &'static str {
        match self {
            GuardState::Active => "model",
            GuardState::Fallback => "hold",
        }
    }
}

/// A change of state, reported once by [`Guard::push`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition {
    pub to: GuardState,
    pub model_cost: f32,
    pub hold_cost: f32,
    pub samples: u32,
    pub windows: usize,
}

/// A look at the guard, for STATUS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardStatus {
    pub state: GuardState,
    /// Windows in the judging span now.
    pub windows: usize,
    /// Mean cost per sample over the span (0 when empty), model and hold.
    pub model_cost: f32,
    pub hold_cost: f32,
    pub fallbacks: u32,
    pub retries: u32,
    /// Windows resolved since the start.
    pub resolved: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    model: f32,
    hold: f32,
    n: u16,
}

pub struct Guard {
    cfg: GuardConfig,
    ring: Vec<Entry>,
    head: usize,
    len: usize,
    state: GuardState,
    /// Windows since the last change of state.
    since: u32,
    fallbacks: u32,
    retries: u32,
    resolved: u64,
}

impl Guard {
    pub fn new(cfg: GuardConfig) -> Result<Guard, String> {
        cfg.validate()?;
        Ok(Guard {
            cfg,
            ring: vec![Entry::default(); cfg.windows],
            head: 0,
            len: 0,
            state: GuardState::Active,
            since: 0,
            fallbacks: 0,
            retries: 0,
            resolved: 0,
        })
    }

    pub fn config(&self) -> &GuardConfig {
        &self.cfg
    }

    pub fn state(&self) -> GuardState {
        self.state
    }

    pub fn using_model(&self) -> bool {
        self.state == GuardState::Active
    }

    /// The sums over the span: `(model, hold, samples)`. Recomputed from the ring each time (250 additions), so no drift.
    fn sums(&self) -> (f64, f64, u32) {
        let (mut m, mut h, mut n) = (0.0f64, 0.0f64, 0u32);
        for e in &self.ring[..self.len] {
            m += f64::from(e.model);
            h += f64::from(e.hold);
            n += u32::from(e.n);
        }
        (m, h, n)
    }

    pub fn status(&self) -> GuardStatus {
        let (m, h, n) = self.sums();
        let per = |x: f64| if n == 0 { 0.0 } else { (x / f64::from(n)) as f32 };
        GuardStatus {
            state: self.state,
            windows: self.len,
            model_cost: per(m),
            hold_cost: per(h),
            fallbacks: self.fallbacks,
            retries: self.retries,
            resolved: self.resolved,
        }
    }

    /// Adds one resolved window (`n` samples; the model's and the baseline's summed costs). A window with no samples is ignored.
    pub fn push(&mut self, model: f32, hold: f32, n: u16) -> Option<Transition> {
        if n == 0 || !model.is_finite() || !hold.is_finite() {
            return None;
        }
        self.resolved += 1;
        // The ring is a plain circular buffer; `head` is the next slot to write.
        self.ring[self.head] = Entry { model, hold, n };
        self.head = (self.head + 1) % self.ring.len();
        self.len = (self.len + 1).min(self.ring.len());
        self.since = self.since.saturating_add(1);
        if self.len < self.cfg.min_windows {
            return None;
        }
        let (m, h, samples) = self.sums();
        let change = match self.state {
            GuardState::Active if m > h * (1.0 + f64::from(self.cfg.margin)) => {
                self.fallbacks += 1;
                Some(GuardState::Fallback)
            }
            GuardState::Fallback
                if self.since >= self.cfg.retry_after && m <= h * (1.0 + f64::from(self.cfg.retry_margin)) =>
            {
                self.retries += 1;
                Some(GuardState::Active)
            }
            _ => None,
        }?;
        self.state = change;
        self.since = 0;
        let per = |x: f64| (x / f64::from(samples.max(1))) as f32;
        Some(Transition {
            to: change,
            model_cost: per(m),
            hold_cost: per(h),
            samples,
            windows: self.len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> GuardConfig {
        GuardConfig {
            windows: 20,
            min_windows: 10,
            margin: 0.05,
            retry_after: 15,
            retry_margin: 0.0,
        }
    }

    fn feed(g: &mut Guard, n: usize, model: f32, hold: f32) -> Vec<Transition> {
        (0..n).filter_map(|_| g.push(model, hold, 4)).collect()
    }

    #[test]
    fn it_starts_active_and_waits_for_evidence() {
        let mut g = Guard::new(small()).unwrap();
        assert!(g.using_model());
        // A terrible model, but fewer windows than `min_windows`: no verdict yet.
        assert!(feed(&mut g, 9, 8.0, 1.0).is_empty());
        assert!(g.using_model());
        let t = g.push(8.0, 1.0, 4).expect("the tenth window decides");
        assert_eq!(t.to, GuardState::Fallback);
        assert!(!g.using_model());
        assert_eq!((t.windows, t.samples), (10, 40));
        assert!(t.model_cost > t.hold_cost);
    }

    #[test]
    fn a_model_better_than_or_close_to_hold_is_kept() {
        let mut g = Guard::new(small()).unwrap();
        assert!(feed(&mut g, 60, 2.0, 2.3).is_empty(), "better");
        let mut g = Guard::new(small()).unwrap();
        assert!(
            feed(&mut g, 60, 2.09, 2.0).is_empty(),
            "worse by 4.5% is inside the 5% margin"
        );
        assert!(g.using_model());
        let t = feed(&mut g, 60, 2.2, 2.0);
        assert_eq!(t.len(), 1, "worse by 10% is not");
    }

    #[test]
    fn after_a_fallback_the_model_returns_only_when_time_has_passed_and_the_shadow_is_not_worse() {
        let mut g = Guard::new(small()).unwrap();
        feed(&mut g, 10, 3.0, 2.0);
        assert!(!g.using_model());
        // Still losing in the shadow: stays benched however long it takes.
        assert!(feed(&mut g, 100, 3.0, 2.0).is_empty());
        assert!(!g.using_model());
        // The shadow improves; the span (20 windows) turns over, and after `retry_after` windows the model is back.
        let back = feed(&mut g, 40, 1.8, 2.0);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].to, GuardState::Active);
        assert!(g.using_model());
        let s = g.status();
        assert_eq!((s.fallbacks, s.retries), (1, 1));
        assert!(s.model_cost < s.hold_cost);
    }

    #[test]
    fn the_retry_waits_for_the_configured_number_of_windows() {
        let mut g = Guard::new(small()).unwrap();
        feed(&mut g, 10, 3.0, 2.0);
        assert!(!g.using_model());
        // The shadow is excellent at once, but the retry is `retry_after` = 15 windows after the fallback.
        let mut windows_until_back = 0;
        for i in 1..=40 {
            if g.push(0.1, 2.0, 4).is_some() {
                windows_until_back = i;
                break;
            }
        }
        // The span has to turn over to the good windows too (3.0 vs 2.0 entries must stop dominating): at least 15, and it is back soon after.
        assert!(windows_until_back >= 15, "{windows_until_back}");
        assert!(windows_until_back <= 25, "{windows_until_back}");
    }

    #[test]
    fn empty_and_non_finite_windows_are_ignored() {
        let mut g = Guard::new(small()).unwrap();
        assert!(g.push(5.0, 1.0, 0).is_none());
        assert!(g.push(f32::NAN, 1.0, 3).is_none());
        assert!(g.push(1.0, f32::INFINITY, 3).is_none());
        assert_eq!(g.status().windows, 0);
        assert_eq!(g.status().resolved, 0);
    }

    #[test]
    fn the_span_is_bounded() {
        let mut g = Guard::new(small()).unwrap();
        feed(&mut g, 100, 2.0, 2.0);
        assert_eq!(g.status().windows, 20);
        feed(&mut g, 10, 9.0, 1.0);
        assert!(!g.using_model());
        assert_eq!(
            g.status().windows,
            20,
            "old windows are overwritten, the span never grows"
        );
        assert_eq!(g.status().fallbacks, 1);
    }

    #[test]
    fn bad_configs_are_refused() {
        for bad in [
            GuardConfig { windows: 0, ..small() },
            GuardConfig {
                min_windows: 30,
                ..small()
            },
            GuardConfig {
                margin: -0.1,
                ..small()
            },
            GuardConfig {
                retry_margin: f32::NAN,
                ..small()
            },
        ] {
            assert!(Guard::new(bad).is_err(), "{bad:?}");
        }
        assert!(Guard::new(GuardConfig::default()).is_ok());
    }
}
