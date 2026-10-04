//! Brain selection — `--brain hybrid|planner|scripted|idle|fly`.
//!
//! [`make_brain`] is the single place a [`BrainKind`] becomes a `Box<dyn Brain>`, so a new brain is
//! one match arm. `hybrid` is `ddai_planner::hybrid::HybridBrain` (task 3.5, D-041/D-055) with its
//! default production configuration: 4 ms search budget, 5 ms decision cap (the proposals' time comes off it,
//! D-080), adaptive extension up to 15 ms in confirmed danger (D-042), the 1vN threat model, the technique
//! library, `--search-threads` scoring threads (task 3.7a: one by default, `auto` = the free cores, at most 4, opt-in),
//! and no proposer (`NoProposer`: the fly is untrained, so nothing proposes yet; with `--fly-bundle` (7.4) a trained fly proposes, `hybrid:fly`). The bot talks to
//! `dyn Brain` only (plus [`BrainKind::has_own_shield`], which says whether the bot must guard the
//! brain's output); the hybrid sees the same local tees and exact predicted world as the planner.
//!
//! **Shields.** The planner (and the hybrid) run the shield inside their own decision
//! (`PlannerConfig::shield`); `scripted` and `fly` do not, so the bot guards their output
//! ([`crate::planning::PlanScratch::guard`], the TS `guard` for the non-planner brains, `bot.ts:2598`).
//! `idle` does nothing that needs guarding.

use std::path::PathBuf;

use ddai_brain::{Brain, IdleBrain};
use ddai_fly::proposer::FlyProposer;
use ddai_planner::brains::{ClockKind, PlannerBrain, PlannerBrainConfig, PlannerMode, PlannerPreset, ScriptedBrain};
use ddai_planner::hybrid::{HybridBrain, HybridConfig, NoProposer, Proposer};

/// Which brain plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainKind {
    /// The D-041 brain: the fly proposes, the exact search decides (task 3.5, no proposer until the fly is trained).
    Hybrid,
    /// The CEM planner alone (debug mode and the arena baseline).
    Planner,
    /// The scripted bot of the phase-0 harness.
    Scripted,
    /// Neutral input only.
    Idle,
    /// The fly alone (untrained until phase 8 delivers a checkpoint).
    Fly,
}

impl BrainKind {
    pub const ALL: [BrainKind; 5] = [
        BrainKind::Hybrid,
        BrainKind::Planner,
        BrainKind::Scripted,
        BrainKind::Idle,
        BrainKind::Fly,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BrainKind::Hybrid => "hybrid",
            BrainKind::Planner => "planner",
            BrainKind::Scripted => "scripted",
            BrainKind::Idle => "idle",
            BrainKind::Fly => "fly",
        }
    }

    pub fn parse(s: &str) -> Option<BrainKind> {
        BrainKind::ALL.into_iter().find(|k| k.name() == s)
    }

    /// Whether the brain shields its own output; if not, the bot applies the guard.
    pub fn has_own_shield(self) -> bool {
        matches!(self, BrainKind::Hybrid | BrainKind::Planner | BrainKind::Idle)
    }

    /// Whether the brain honours `LiveContext::spare_ids` (never treats those ids as target, threat,
    /// victim or hook target), so spared tees may sit in its world as physical bodies (task 4.1b,
    /// review F8). A brain that does not would read a body as an opponent, which is worse than not
    /// simulating it, so for those the bot keeps the round-1 behaviour (spared tees out of the world).
    ///
    /// The hybrid does (task 3.5b: `HybridBrain::set_live_context` keeps the ids out of its threat,
    /// victim, target and hook-target sets).
    pub fn honours_spare_ids(self) -> bool {
        matches!(self, BrainKind::Planner | BrainKind::Hybrid)
    }
}

/// Knobs for building brains.
#[derive(Debug, Clone)]
pub struct BrainOptions {
    /// The planner's per-decision wall budget (D-042: 5 ms usually, up to 15 in danger).
    pub planner_budget_ms: f64,
    pub planner_preset: PlannerPreset,
    /// `--brain fly`: the compiled graph, the brain config and the seed.
    pub fly_flyg: PathBuf,
    pub fly_config: PathBuf,
    /// Task 7.4: a trained fly bundle (`--fly-bundle`). With it `--brain fly` plays with the trained weights and
    /// `--brain hybrid` gets the fly as its proposer (`hybrid:fly`); without it the fly is untrained and the hybrid
    /// has no proposer. The bundle names its own brain config; `.flyg` is `fly_flyg` (the hash must match).
    pub fly_bundle: Option<PathBuf>,
    /// Task 3.7a (D-080): threads that score the hybrid's candidates, the deciding thread included; `None` = auto
    /// ([`auto_search_threads`] at the time the brain is built). One thread by default, here and in the CLI (D-080:
    /// no measured gain from more on a loaded machine; `--search-threads auto` is opt-in).
    pub search_threads: Option<usize>,
    /// Task 3.7a (D-080): the proposer's time comes off the hybrid's decision cap (`HybridConfig::proposal_in_cap`);
    /// `false` only for the before/after comparison of E-012.
    pub proposal_in_cap: bool,
    /// Task 3.7b (D-090): the hybrid's opponent model (`HybridConfig::mirror`: a small search from the victim's seat predicts its
    /// plan). On by default; `--hybrid-mirror off` is the way back to "the victim holds its input" (it was built and measured against
    /// planners, scripted bots and idle/wandering/hook-spamming tees, not yet against people).
    pub hybrid_mirror: bool,
    pub seed: u64,
}

/// Most threads `--search-threads auto` ever picks (the cores beyond the fourth buy little: E-012).
pub const AUTO_SEARCH_THREADS_CAP: usize = 4;

/// The number of search threads for a machine with `cores` cores and a 1-minute load average of `load1` (`None` =
/// unknown): the cores nobody is using, at least 1 and at most [`AUTO_SEARCH_THREADS_CAP`]. The load average
/// already contains the bot itself, so a quiet machine with four cores gets 4 only if the bot is the only user,
/// which is what "free" means for a latency-bound job: helpers that would only fight a neighbour for a core
/// make the decision slower, not faster.
pub fn auto_search_threads(cores: usize, load1: Option<f64>) -> usize {
    let busy = load1
        .filter(|l| l.is_finite() && *l >= 0.0)
        .unwrap_or(cores as f64 / 2.0);
    let free = (cores as f64 - busy.round()).max(0.0) as usize;
    free.clamp(1, AUTO_SEARCH_THREADS_CAP)
}

/// [`auto_search_threads`] for this machine (`/proc/loadavg` where it exists).
pub fn auto_search_threads_here() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let load1 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|t| t.split_whitespace().next().and_then(|v| v.parse::<f64>().ok()));
    auto_search_threads(cores, load1)
}

impl Default for BrainOptions {
    fn default() -> Self {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        BrainOptions {
            planner_budget_ms: 5.0,
            planner_preset: PlannerPreset::Normal,
            fly_flyg: home.join("aiddnet/data/connectome/compiled/fly-S-v1.flyg"),
            fly_config: PathBuf::from("configs/fly/S-brain.toml"),
            fly_bundle: None,
            search_threads: Some(1),
            proposal_in_cap: true,
            hybrid_mirror: true,
            seed: 1,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BrainError {
    #[error("hybrid brain: {0}")]
    Hybrid(String),
    #[error("fly brain: {0}")]
    Fly(String),
}

/// The live hybrid's configuration: the library defaults plus what the options set.
pub fn hybrid_config(opts: &BrainOptions) -> HybridConfig {
    HybridConfig {
        workers: opts.search_threads.unwrap_or_else(auto_search_threads_here).max(1),
        proposal_in_cap: opts.proposal_in_cap,
        mirror: opts.hybrid_mirror,
        ..HybridConfig::default()
    }
}

/// Builds the brain. Not `Send`: the planner holds `Rc`s, so build it on the thread that plays.
pub fn make_brain(kind: BrainKind, opts: &BrainOptions) -> Result<Box<dyn Brain>, BrainError> {
    Ok(match kind {
        BrainKind::Hybrid => {
            let proposer: Box<dyn Proposer> = match &opts.fly_bundle {
                Some(_) => Box::new(FlyProposer::new(make_bundle_fly(opts)?, opts.seed)),
                None => Box::new(NoProposer),
            };
            Box::new(HybridBrain::new(hybrid_config(opts), ClockKind::Wall, proposer).map_err(BrainError::Hybrid)?)
        }
        BrainKind::Planner => Box::new(PlannerBrain::new(PlannerBrainConfig {
            preset: opts.planner_preset,
            mode: PlannerMode::Deadline {
                budget_ms: opts.planner_budget_ms,
            },
            clock: ClockKind::Wall,
        })),
        BrainKind::Scripted => Box::new(ScriptedBrain::new()),
        BrainKind::Idle => Box::new(IdleBrain),
        BrainKind::Fly => Box::new(make_fly(opts)?),
    })
}

/// A fly with the weights of `opts.fly_bundle` (which must be set); remembers the bundle's name and hash for the web panel.
fn make_bundle_fly(opts: &BrainOptions) -> Result<ddai_fly::brain::FlyBrain, BrainError> {
    use ddai_fly::brain::{ActionSelection, FlyBrainConfig};
    let bundle = opts.fly_bundle.as_deref().expect("the caller checked");
    let template = ddai_fly::bundle::FlyBrainTemplate::load(bundle, Some(&opts.fly_flyg))
        .map_err(|e| BrainError::Fly(format!("bundle {}: {e}", bundle.display())))?;
    Ok(template.instantiate(FlyBrainConfig {
        action_selection: ActionSelection::Argmax,
        seed: opts.seed,
    }))
}

fn make_fly(opts: &BrainOptions) -> Result<ddai_fly::brain::FlyBrain, BrainError> {
    use ddai_fly::brain::{ActionSelection, FlyBrain, FlyBrainConfig};
    if opts.fly_bundle.is_some() {
        return make_bundle_fly(opts);
    }
    use ddai_fly::decoder::{DecoderModel, DnCalibration};
    use ddai_fly::encoder::{EncoderModel, EncoderParams};
    use ddai_fly::{FlyConfig, FlyModel, FlyParams};

    let flyg =
        ddai_flyg::load(&opts.fly_flyg).map_err(|e| BrainError::Fly(format!("{}: {e}", opts.fly_flyg.display())))?;
    let cfg = ddai_fly::brain_config::load_brain_config(&opts.fly_config)
        .map_err(|e| BrainError::Fly(format!("{}: {e:?}", opts.fly_config.display())))?;
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, opts.seed);
    let model = FlyModel::new(flyg, config, params).map_err(|e| BrainError::Fly(format!("{e:?}")))?;
    let encoder =
        EncoderModel::new(&model, cfg.ray_grid, &cfg.proprioception).map_err(|e| BrainError::Fly(format!("{e:?}")))?;
    let encoder_params = EncoderParams::init_default(encoder.num_params());
    let decoder = DecoderModel::new(&model, cfg.decoder).map_err(|e| BrainError::Fly(format!("{e:?}")))?;
    let decoder_params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };
    Ok(FlyBrain::new(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        FlyBrainConfig {
            action_selection: ActionSelection::Argmax,
            seed: opts.seed,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_only_planner_family_shields_itself() {
        for k in BrainKind::ALL {
            assert_eq!(BrainKind::parse(k.name()), Some(k));
        }
        assert_eq!(BrainKind::parse("nope"), None);
        assert!(BrainKind::Planner.has_own_shield() && BrainKind::Hybrid.has_own_shield());
        assert!(!BrainKind::Scripted.has_own_shield() && !BrainKind::Fly.has_own_shield());
    }

    #[test]
    fn auto_search_threads_follow_the_free_cores_within_one_and_the_cap() {
        // 8 cores, nobody else: 4 (the cap); a busy neighbour takes cores away; a saturated or unreadable machine: 1.
        assert_eq!(auto_search_threads(8, Some(0.2)), 4);
        assert_eq!(auto_search_threads(8, Some(5.2)), 3);
        assert_eq!(auto_search_threads(8, Some(6.0)), 2);
        assert_eq!(auto_search_threads(8, Some(4.4)), 4);
        assert_eq!(auto_search_threads(8, Some(7.5)), 1);
        assert_eq!(
            auto_search_threads(8, Some(28.8)),
            1,
            "overloaded: one thread, never zero"
        );
        assert_eq!(auto_search_threads(2, Some(0.1)), 2);
        assert_eq!(auto_search_threads(1, Some(0.0)), 1);
        assert_eq!(auto_search_threads(16, None), 4, "unknown load: half the cores, capped");
        assert_eq!(
            auto_search_threads(4, Some(f64::NAN)),
            2,
            "a garbage load reads as half busy"
        );
        assert_eq!(auto_search_threads(4, Some(-1.0)), 2);
        let here = auto_search_threads_here();
        assert!((1..=AUTO_SEARCH_THREADS_CAP).contains(&here));
    }

    #[test]
    fn the_hybrid_gets_the_requested_number_of_search_threads() {
        for (asked, want) in [(Some(1), 1usize), (Some(3), 3), (Some(0), 1)] {
            let opts = BrainOptions {
                search_threads: asked,
                ..BrainOptions::default()
            };
            let b = make_brain(BrainKind::Hybrid, &opts).expect("hybrid");
            let t = b.telemetry().expect("telemetry");
            let v: serde_json::Value = serde_json::from_str(&t).expect("json");
            assert_eq!(v["workers"], want, "{asked:?}");
        }
        assert_eq!(
            BrainOptions::default().search_threads,
            Some(1),
            "the library default is one thread"
        );
    }

    #[test]
    fn the_hybrid_opponent_model_follows_the_option() {
        assert!(BrainOptions::default().hybrid_mirror, "on by default");
        assert!(hybrid_config(&BrainOptions::default()).mirror);
        for on in [true, false] {
            let opts = BrainOptions {
                hybrid_mirror: on,
                ..BrainOptions::default()
            };
            assert_eq!(hybrid_config(&opts).mirror, on);
            make_brain(BrainKind::Hybrid, &opts).expect("hybrid builds either way");
        }
    }

    #[test]
    fn the_cheap_brains_and_the_hybrid_build() {
        let opts = BrainOptions::default();
        for k in [
            BrainKind::Hybrid,
            BrainKind::Planner,
            BrainKind::Scripted,
            BrainKind::Idle,
        ] {
            let b = make_brain(k, &opts).unwrap_or_else(|e| panic!("{k:?}: {e}"));
            assert!(!b.name().is_empty());
        }
        let bad = BrainOptions {
            fly_flyg: PathBuf::from("/nonexistent/fly.flyg"),
            ..BrainOptions::default()
        };
        assert!(matches!(make_brain(BrainKind::Fly, &bad), Err(BrainError::Fly(_))));
    }
}
