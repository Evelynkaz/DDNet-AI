//! The two input-lag knobs of task 3.16 (D-115), resolved from the command line and the settings file: `hybrid_budget_ms` and
//! `prediction_margin_ms`. Both are **off by default** (nothing set = the behaviour of the build before the knobs: the hybrid at 4 ms
//! under a 5 ms cap, the adaptive prediction margin), so a settings file that does not name them changes nothing.
//!
//! * `hybrid_budget_ms` (whole ms, 1 to 8): the hybrid's search budget; the decision cap moves with it
//!   ([`ddai_planner::hybrid::HybridConfig::with_budget_ms`]). Fewer ms of decision time is a smaller share of the input lag that the 20 ms
//!   tick rounds up (`docs/research/lag-shave.md`).
//! * `prediction_margin_ms` (whole ms, 0 to 30): a **fixed** `cl_prediction_margin` (the existing `--prediction-margin-ms`, now also a
//!   settings key): the adaptive controller of D-063 is **off** for the run, the margin stays at this value. It is an *override*, not a
//!   floor: a floor could only add lag, and the controller's own range is 3 to 20 ms.
//!
//! The command line wins over the file. A value in the settings file that is out of range, negative, fractional or not a number at all is **ignored
//! with a warning** (the default applies; the bot must not stay down, nor the file be moved aside, over a typo in a file the owner edits by hand), a flag out of range is refused by the argument parser.

use std::ops::RangeInclusive;

use crate::settings::{RawKnob, Settings};

/// The range of `hybrid_budget_ms`, whole ms (the hybrid's own limits: [`ddai_planner::hybrid::BUDGET_MS_RANGE`]).
pub const HYBRID_BUDGET_MS_RANGE: RangeInclusive<u32> = ddai_planner::hybrid::BUDGET_MS_RANGE;
/// The range of `prediction_margin_ms`, whole ms.
pub const PREDICTION_MARGIN_MS_RANGE: RangeInclusive<i32> = 0..=30;

/// Where a knob's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Flag,
    Settings,
    Default,
}

impl Source {
    pub fn describe(self, flag: &str, key: &str) -> String {
        match self {
            Source::Flag => flag.to_string(),
            Source::Settings => format!("settings file, {key}"),
            Source::Default => "default".to_string(),
        }
    }
}

/// What the two knobs resolve to, and the warnings for the settings values that were refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimingKnobs {
    /// `None`: the library default (4 ms under a 5 ms cap).
    pub hybrid_budget_ms: Option<u32>,
    pub hybrid_budget_source: Source,
    /// `None`: the adaptive controller.
    pub prediction_margin_ms: Option<i32>,
    pub prediction_margin_source: Source,
    /// One line per settings value that was out of range and ignored.
    pub warnings: Vec<String>,
}

impl TimingKnobs {
    /// The effective hybrid search budget, ms.
    pub fn effective_budget_ms(&self) -> u32 {
        self.hybrid_budget_ms
            .unwrap_or(ddai_planner::hybrid::DEFAULT_BUDGET_MS as u32)
    }

    /// The lines the bot prints at start (the live protocol of task 3.16 reads them from the journal).
    pub fn start_lines(&self, hybrid: bool) -> Vec<String> {
        let budget = self.effective_budget_ms();
        let cap = f64::from(budget) + ddai_planner::hybrid::CAP_HEADROOM_MS;
        let mut lines = vec![format!(
            "hybrid budget: {budget} ms search, {cap} ms decision cap ({}){}",
            self.hybrid_budget_source
                .describe("--hybrid-budget-ms", "hybrid_budget_ms"),
            if hybrid { "" } else { "; not used by this brain" }
        )];
        lines.push(match self.prediction_margin_ms {
            Some(m) => format!(
                "prediction margin: fixed {m} ms, adaptive controller off ({})",
                self.prediction_margin_source
                    .describe("--prediction-margin-ms", "prediction_margin_ms")
            ),
            None => "prediction margin: adaptive (starts at 10 ms, 3..20; D-063)".to_string(),
        });
        lines
    }
}

/// Resolves the two knobs: the flag, else the settings file, else the default. Pure.
pub fn resolve(flag_budget: Option<u32>, flag_margin: Option<i32>, settings: &Settings) -> TimingKnobs {
    let mut warnings = Vec::new();
    // A settings value is judged here, not by serde: it must be a whole number in range, anything else (negative, fractional, quoted, ...) is
    // ignored with a warning and the default applies. Never an error that would cost the owner the file (review F1).
    let mut from_settings = |name: &str, raw: &RawKnob, range: RangeInclusive<i64>| -> Option<i64> {
        match raw {
            RawKnob::Int(v) if range.contains(v) => Some(*v),
            RawKnob::Int(v) => {
                warnings.push(format!(
                    "settings: {name} = {v} is out of range {}..={}, ignored (the default applies)",
                    range.start(),
                    range.end()
                ));
                None
            }
            RawKnob::Other(text) => {
                warnings.push(format!(
                    "settings: {name} = {text} is not a whole number, ignored (the default applies)"
                ));
                None
            }
        }
    };
    let budget_range = i64::from(*HYBRID_BUDGET_MS_RANGE.start())..=i64::from(*HYBRID_BUDGET_MS_RANGE.end());
    let margin_range = i64::from(*PREDICTION_MARGIN_MS_RANGE.start())..=i64::from(*PREDICTION_MARGIN_MS_RANGE.end());
    let (hybrid_budget_ms, hybrid_budget_source) = match (flag_budget, &settings.hybrid_budget_ms) {
        (Some(v), _) => (Some(v), Source::Flag),
        (None, Some(raw)) => match from_settings("hybrid_budget_ms", raw, budget_range) {
            Some(v) => (u32::try_from(v).ok(), Source::Settings),
            None => (None, Source::Default),
        },
        _ => (None, Source::Default),
    };
    let (prediction_margin_ms, prediction_margin_source) = match (flag_margin, &settings.prediction_margin_ms) {
        (Some(v), _) => (Some(v), Source::Flag),
        (None, Some(raw)) => match from_settings("prediction_margin_ms", raw, margin_range) {
            Some(v) => (i32::try_from(v).ok(), Source::Settings),
            None => (None, Source::Default),
        },
        _ => (None, Source::Default),
    };
    TimingKnobs {
        hybrid_budget_ms,
        hybrid_budget_source,
        prediction_margin_ms,
        prediction_margin_source,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(budget: Option<u32>, margin: Option<i32>) -> Settings {
        Settings {
            hybrid_budget_ms: budget.map(|v| RawKnob::Int(i64::from(v))),
            prediction_margin_ms: margin.map(|v| RawKnob::Int(i64::from(v))),
            ..Settings::default()
        }
    }

    #[test]
    fn nothing_set_is_the_old_behaviour_and_says_so() {
        let k = resolve(None, None, &Settings::default());
        assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (None, None));
        assert_eq!(
            (k.hybrid_budget_source, k.prediction_margin_source),
            (Source::Default, Source::Default)
        );
        assert!(k.warnings.is_empty());
        assert_eq!(k.effective_budget_ms(), 4);
        let lines = k.start_lines(true);
        assert!(
            lines[0].contains("4 ms search, 5 ms decision cap (default)"),
            "{lines:?}"
        );
        assert!(lines[1].contains("adaptive"), "{lines:?}");
    }

    #[test]
    fn the_flag_wins_over_the_file_and_the_file_over_the_default() {
        let k = resolve(Some(2), Some(6), &s(Some(7), Some(12)));
        assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (Some(2), Some(6)));
        assert_eq!(
            (k.hybrid_budget_source, k.prediction_margin_source),
            (Source::Flag, Source::Flag)
        );
        let k = resolve(None, None, &s(Some(7), Some(12)));
        assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (Some(7), Some(12)));
        assert_eq!(
            (k.hybrid_budget_source, k.prediction_margin_source),
            (Source::Settings, Source::Settings)
        );
        assert!(k.start_lines(true)[0].contains("7 ms search, 8 ms decision cap (settings file, hybrid_budget_ms)"));
        assert!(k.start_lines(true)[1].contains("fixed 12 ms"));
        // One knob from each source.
        let k = resolve(None, Some(0), &s(Some(3), None));
        assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (Some(3), Some(0)));
    }

    #[test]
    fn the_range_edges_are_accepted_and_out_of_range_file_values_are_ignored_with_a_warning() {
        for (b, m) in [(1, 0), (8, 30)] {
            let k = resolve(None, None, &s(Some(b), Some(m)));
            assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (Some(b), Some(m)));
            assert!(k.warnings.is_empty());
        }
        for (b, m) in [(0, -1), (9, 31), (4000, i32::MAX)] {
            let k = resolve(None, None, &s(Some(b), Some(m)));
            assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (None, None), "{b} {m}");
            assert_eq!(k.warnings.len(), 2, "{:?}", k.warnings);
            assert_eq!(
                (k.hybrid_budget_source, k.prediction_margin_source),
                (Source::Default, Source::Default)
            );
        }
        // A flag is trusted here (the argument parser enforces the range), and a good flag silences a bad file value.
        let k = resolve(Some(3), Some(5), &s(Some(99), Some(99)));
        assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (Some(3), Some(5)));
        assert!(k.warnings.is_empty());
    }

    /// Review F1: negative, fractional, quoted and other-typed values read from a file are ignored with a warning, never an error.
    #[test]
    fn values_of_the_wrong_kind_are_ignored_with_a_warning_like_out_of_range_ones() {
        let dir = tempfile::tempdir().unwrap();
        for value in ["-1", "4.5", "\"5\"", "true", "[3]", "99999999999999999"] {
            let path = dir.path().join("settings.toml");
            std::fs::write(
                &path,
                format!("clan = \"Keep\"\nhybrid_budget_ms = {value}\nprediction_margin_ms = {value}\n"),
            )
            .unwrap();
            let crate::settings::Loaded::Ok(settings) = crate::settings::load(&path) else {
                panic!("{value}: the file was treated as corrupt");
            };
            assert_eq!(settings.clan.as_deref(), Some("Keep"), "{value}");
            let k = resolve(None, None, &settings);
            assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (None, None), "{value}");
            assert_eq!(
                (k.hybrid_budget_source, k.prediction_margin_source),
                (Source::Default, Source::Default)
            );
            assert_eq!(k.warnings.len(), 2, "{value}: {:?}", k.warnings);
            assert!(
                k.warnings
                    .iter()
                    .all(|w| w.starts_with("settings: ") && w.contains("ignored")),
                "{:?}",
                k.warnings
            );
            assert!(path.exists());
        }
        // A good flag still wins over (and silences nothing about) a bad file value; the file value does not matter then.
        let settings = Settings {
            hybrid_budget_ms: Some(RawKnob::Other(toml::Value::Float(3.5))),
            ..Settings::default()
        };
        let k = resolve(Some(2), None, &settings);
        assert_eq!(k.hybrid_budget_ms, Some(2));
    }

    #[test]
    fn the_start_line_for_another_brain_says_the_budget_is_not_used() {
        let k = resolve(Some(2), None, &Settings::default());
        assert!(k.start_lines(false)[0].contains("not used by this brain"));
        assert!(!k.start_lines(true)[0].contains("not used"));
    }

    #[test]
    fn the_keys_read_from_a_settings_file_of_the_documented_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(
            &path,
            "hybrid_budget_ms = 3\nprediction_margin_ms = 8\nbrain = \"hybrid\"\n",
        )
        .unwrap();
        assert!(
            crate::settings::unknown_keys(&path).is_empty(),
            "the keys are known (an unknown key would switch the owner chat off)"
        );
        let settings = crate::settings::load(&path).settings();
        let k = resolve(None, None, &settings);
        assert_eq!((k.hybrid_budget_ms, k.prediction_margin_ms), (Some(3), Some(8)));
    }
}
