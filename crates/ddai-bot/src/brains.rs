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
    /// Task 3.10 (opt-in, `--finish full`): the hybrid's finishing switches ([`HybridConfig::with_finish`]).
    pub hybrid_finish: bool,
    /// Task 3.18 (opt-in, `--finish wb`): the hybrid's wayblock hold ([`HybridConfig::wb_hold`]).
    pub hybrid_wb_hold: bool,
    /// Task 3.16 (D-115): the hybrid's search budget in whole milliseconds, 1 to 8 (`--hybrid-budget-ms`, `hybrid_budget_ms` in the settings
    /// file); the decision cap moves with it ([`HybridConfig::with_budget_ms`]). `None` = the library default (4 ms under a 5 ms cap), which is
    /// also what `Some(4)` builds. Not read by the planner brain (`planner_budget_ms` is its own knob).
    pub hybrid_budget_ms: Option<u32>,
    /// Task 3.19 (D-116, opt-in, `--duel-hammer`): the hybrid's reflex hammer and hammer-safe envelope, which act only in a detected duel.
    pub reflex: ddai_planner::hybrid::ReflexConfig,
    /// Task 3.23 (D-121, opt-in, `--duel-fixes`): the hybrid's fixes for the weaknesses of the 2026-10-08 duel against a human; they act only in a detected duel.
    pub duel_fixes: ddai_planner::hybrid::DuelFixConfig,
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

/// [`auto_search_threads`] for this machine (the load average where the OS has one: `/proc/loadavg` on Linux, nothing on Windows).
pub fn auto_search_threads_here() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let load1 = ddai_os::host::load_average().map(|l| l[0]);
    auto_search_threads(cores, load1)
}

impl Default for BrainOptions {
    fn default() -> Self {
        BrainOptions {
            planner_budget_ms: 5.0,
            planner_preset: PlannerPreset::Normal,
            fly_flyg: ddai_os::dirs::data_root()
                .unwrap_or_default()
                .join("connectome/compiled/fly-S-v1.flyg"),
            fly_config: PathBuf::from("configs/fly/S-brain.toml"),
            fly_bundle: None,
            search_threads: Some(1),
            proposal_in_cap: true,
            hybrid_mirror: true,
            hybrid_finish: false,
            hybrid_wb_hold: false,
            hybrid_budget_ms: None,
            reflex: ddai_planner::hybrid::ReflexConfig::default(),
            duel_fixes: ddai_planner::hybrid::DuelFixConfig::default(),
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

/// Task 3.19 (D-116): what `--duel-hammer` / `duel_hammer` in `settings.toml` mean: `off` (the default), `reflex` (the reflex hammer, only when the hit throws the
/// other tee into a freeze: the variant that did no harm in the arena), `reflex-all` (swing at every chance: **worse** than nothing in the arena, kept for
/// diagnostics), `envelope` (the hammer-safe envelope) or `both` (`reflex` + `envelope`). All act only while the bot has detected a duel
/// ([`ddai_planner::hybrid::ReflexConfig::duel_only`]); the reflex waits 16 ticks after our last swing because a live world does not know the reload timer
/// (`lockout_ticks`). None of them passed the pre-registered go bars (docs/research/duel-3.19.md): they are tools for a live measurement, not a recommendation.
pub fn duel_hammer(mode: &str) -> Option<ddai_planner::hybrid::ReflexConfig> {
    use ddai_planner::hybrid::ReflexConfig;
    let live = ReflexConfig {
        duel_only: true,
        lockout_ticks: 16,
        ..ReflexConfig::default()
    };
    let reflex = ReflexConfig {
        hammer: true,
        hazard_only: true,
        ..live
    };
    match mode.to_ascii_lowercase().as_str() {
        "off" => Some(ReflexConfig::default()),
        "reflex" => Some(reflex),
        "reflex-all" => Some(ReflexConfig {
            hazard_only: false,
            ..reflex
        }),
        "envelope" => Some(ReflexConfig { envelope: true, ..live }),
        "both" => Some(ReflexConfig {
            envelope: true,
            ..reflex
        }),
        _ => None,
    }
}

/// What `--duel-fixes` means (task 3.23, D-121): `off` (the default), `all`, or a comma list of `static` (fix 1: the duel opponent is never dropped as AFK -- the
/// bot's [`BotConfig::duel_afk`] -- and the hybrid answers a standing opponent by a plan that acts), `counter` (fix 2: the reacting opponent of the robust stage
/// lets go of us once he is below us while we rise, believed whenever his hook holds us, with the defensive techniques re-scored) and `finish` (fix 3: a frozen
/// victim lying off the freeze is answered by a plan that acts, approach plans join the pool, no swing at a frozen tee). They act only in a detected duel
/// ([`ddai_planner::hybrid::DuelFixConfig::duel_only`]). `None` for anything else. The result is the hybrid's configuration and whether the picker's AFK
/// exemption is on.
pub fn duel_fixes(list: &str) -> Option<(ddai_planner::hybrid::DuelFixConfig, bool)> {
    use ddai_planner::hybrid::DuelFixConfig;
    let mut c = DuelFixConfig::default();
    let mut afk = false;
    for part in list.split(',').map(|p| p.trim().to_ascii_lowercase()) {
        match part.as_str() {
            "off" | "" => {}
            "static" => {
                c.static_push = true;
                afk = true;
            }
            "counter" => {
                c.counter_release = true;
                c.hooked_belief = COUNTER_HOOKED_BELIEF;
                c.protect_defence = true;
            }
            "finish" => {
                c.finish_push = true;
                c.finish_approach = FINISH_APPROACH_PLANS;
                c.no_hammer_frozen = true;
            }
            "all" => {
                return Some((
                    DuelFixConfig {
                        static_push: true,
                        counter_release: true,
                        hooked_belief: COUNTER_HOOKED_BELIEF,
                        protect_defence: true,
                        finish_push: true,
                        finish_approach: FINISH_APPROACH_PLANS,
                        no_hammer_frozen: true,
                        ..c
                    },
                    true,
                ));
            }
            _ => return None,
        }
    }
    Some((c, afk))
}

/// The `--duel-fixes` word of a configuration (task 5.18, D-129; STATUS `duel_fixes`): `off`, or the fixes that are on as a comma list in the order
/// `static,counter,finish`. The inverse of [`duel_fixes`] for the three fixes (`all` reads as the full list).
pub fn duel_fixes_name(c: &ddai_planner::hybrid::DuelFixConfig) -> String {
    let on: Vec<&str> = [
        (c.static_push, "static"),
        (c.counter_release, "counter"),
        (c.finish_push, "finish"),
    ]
    .into_iter()
    .filter_map(|(is_on, name)| is_on.then_some(name))
    .collect();
    if on.is_empty() { "off".to_string() } else { on.join(",") }
}

/// The belief that the opponent reacts while his hook holds us, with `--duel-fixes counter` (E-038).
pub const COUNTER_HOOKED_BELIEF: f64 = 0.8;
/// Approach plans (technique T30) per decision against a frozen victim lying off the freeze, with `--duel-fixes finish` (E-038).
pub const FINISH_APPROACH_PLANS: usize = 6;

/// The live hybrid's configuration: the library defaults plus what the options set.
pub fn hybrid_config(opts: &BrainOptions) -> HybridConfig {
    let cfg = HybridConfig {
        workers: opts.search_threads.unwrap_or_else(auto_search_threads_here).max(1),
        proposal_in_cap: opts.proposal_in_cap,
        mirror: opts.hybrid_mirror,
        wb_hold: opts.hybrid_wb_hold,
        reflex: opts.reflex,
        duel_fixes: opts.duel_fixes,
        ..HybridConfig::default()
    };
    let cfg = if opts.hybrid_finish { cfg.with_finish() } else { cfg };
    match opts.hybrid_budget_ms {
        Some(ms) => cfg.with_budget_ms(f64::from(ms)),
        None => cfg,
    }
}

/// The hybrid the bot plays: wall clock, with the fly of `opts.fly_bundle` as its proposer when there is one.
pub fn make_hybrid(opts: &BrainOptions) -> Result<HybridBrain, BrainError> {
    let proposer: Box<dyn Proposer> = match &opts.fly_bundle {
        Some(_) => Box::new(make_bundle_proposer(opts)?),
        None => Box::new(NoProposer),
    };
    HybridBrain::new(hybrid_config(opts), ClockKind::Wall, proposer).map_err(BrainError::Hybrid)
}

/// Builds the brain. Not `Send`: the planner holds `Rc`s, so build it on the thread that plays.
pub fn make_brain(kind: BrainKind, opts: &BrainOptions) -> Result<Box<dyn Brain>, BrainError> {
    Ok(match kind {
        BrainKind::Hybrid => Box::new(make_hybrid(opts)?),
        BrainKind::Planner => Box::new(PlannerBrain::new(PlannerBrainConfig {
            preset: opts.planner_preset,
            mode: PlannerMode::Deadline {
                budget_ms: opts.planner_budget_ms,
            },
            clock: ClockKind::Wall,
        })),
        BrainKind::Scripted => Box::new(ScriptedBrain::new()),
        BrainKind::Idle => Box::new(IdleBrain),
        BrainKind::Fly => make_fly(opts)?,
    })
}

/// The template of `opts.fly_bundle` (which must be set); remembers the bundle's name and hash for the web panel.
fn load_bundle_template(opts: &BrainOptions) -> Result<ddai_fly::bundle::FlyBrainTemplate, BrainError> {
    let bundle = opts.fly_bundle.as_deref().expect("the caller checked");
    let template = ddai_fly::bundle::FlyBrainTemplate::load(bundle, Some(&opts.fly_flyg))
        .map_err(|e| BrainError::Fly(format!("bundle {}: {e}", bundle.display())))?;
    // A latched / intent hook head is not played live yet: the server pause and its resume, the guard and the hook veto change what is sent.
    template
        .require_unlatched("live bot")
        .map_err(|e| BrainError::Fly(format!("bundle {}: {e}", bundle.display())))?;
    // The encoder-input control readout (task 8.7) is a measurement control, not a fly: never played live.
    template
        .require_fly_readout("live bot")
        .map_err(|e| BrainError::Fly(format!("bundle {}: {e}", bundle.display())))?;
    // The Gm neuron model (task 8.8) is a pilot measured in the arena only: not played live until the owner confirms it.
    template
        .require_rate("live bot")
        .map_err(|e| BrainError::Fly(format!("bundle {}: {e}", bundle.display())))?;
    Ok(template)
}

fn bundle_fly_config(opts: &BrainOptions) -> ddai_fly::brain::FlyBrainConfig {
    ddai_fly::brain::FlyBrainConfig {
        action_selection: ddai_fly::brain::ActionSelection::Argmax,
        seed: opts.seed,
    }
}

/// The hybrid's proposer over `opts.fly_bundle` (which must be set): a model trained with the hook head masked takes
/// the hook probability of its proposals from its second view (`FlyProposer::from_template`).
fn make_bundle_proposer(opts: &BrainOptions) -> Result<FlyProposer, BrainError> {
    FlyProposer::from_template(&load_bundle_template(opts)?, bundle_fly_config(opts), opts.seed)
        .map_err(|e| BrainError::Fly(e.to_string()))
}

/// The brain that plays the weights of `opts.fly_bundle`, the way the bundle was trained: a model trained with the
/// hook head masked (8.2b, `HookView::MaskedForHookHead`) is played in two views, exactly as in the arena
/// (`FlyBrainTemplate::instantiate_played`); playing it single-view would be a policy nobody evaluated.
fn make_bundle_fly(opts: &BrainOptions) -> Result<Box<dyn Brain>, BrainError> {
    Ok(load_bundle_template(opts)?.instantiate_played(bundle_fly_config(opts)))
}

fn make_fly(opts: &BrainOptions) -> Result<Box<dyn Brain>, BrainError> {
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
    Ok(Box::new(FlyBrain::new(
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
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task 3.23 (D-121): `--duel-fixes` is off by default and then changes nothing; each name switches its own fix on, `all` the three; all of them duel-only.
    #[test]
    fn duel_fixes_are_off_by_default_and_each_name_switches_its_own_fix_on() {
        use ddai_planner::hybrid::DuelFixConfig;
        assert_eq!(BrainOptions::default().duel_fixes, DuelFixConfig::default());
        assert_eq!(duel_fixes("off"), Some((DuelFixConfig::default(), false)));
        assert_eq!(
            hybrid_config(&BrainOptions::default()).duel_fixes,
            DuelFixConfig::default()
        );
        let (st, afk) = duel_fixes("static").unwrap();
        assert!(afk && st.static_push && !st.counter_release && !st.finish_push && st.duel_only);
        let (co, afk) = duel_fixes("COUNTER").unwrap();
        assert!(
            !afk && !co.static_push
                && co.counter_release
                && co.protect_defence
                && co.hooked_belief == COUNTER_HOOKED_BELIEF
        );
        assert!(!co.finish_push && co.duel_only);
        let (fi, afk) = duel_fixes("finish").unwrap();
        assert!(
            !afk && fi.finish_push
                && fi.no_hammer_frozen
                && fi.finish_approach == FINISH_APPROACH_PLANS
                && !fi.counter_release
        );
        let (all, afk) = duel_fixes("all").unwrap();
        assert!(afk && all.static_push && all.counter_release && all.finish_push && all.duel_only);
        let (two, _) = duel_fixes("static, finish").unwrap();
        assert!(two.static_push && two.finish_push && !two.counter_release);
        assert_eq!(duel_fixes("sometimes"), None);
        assert_eq!(duel_fixes("static,sometimes"), None);
        let opts = BrainOptions {
            duel_fixes: all,
            ..BrainOptions::default()
        };
        assert_eq!(hybrid_config(&opts).duel_fixes, all, "the live hybrid carries them");
    }

    /// Task 5.18 (D-129): STATUS names the fixes the hybrid runs with, and the name parses back to the same configuration.
    #[test]
    fn the_duel_fixes_name_lists_the_fixes_that_are_on() {
        use ddai_planner::hybrid::DuelFixConfig;
        assert_eq!(duel_fixes_name(&DuelFixConfig::default()), "off");
        for (list, name) in [
            ("off", "off"),
            ("finish", "finish"),
            ("static", "static"),
            ("counter", "counter"),
            ("finish,static", "static,finish"),
            ("static,finish", "static,finish"),
            ("all", "static,counter,finish"),
        ] {
            let (c, _) = duel_fixes(list).unwrap();
            assert_eq!(duel_fixes_name(&c), name, "{list}");
            assert_eq!(duel_fixes(name).unwrap().0, c, "{name} parses back");
        }
    }

    #[test]
    fn duel_hammer_modes_are_off_by_default_and_duel_only() {
        use ddai_planner::hybrid::ReflexConfig;
        assert_eq!(BrainOptions::default().reflex, ReflexConfig::default());
        assert_eq!(duel_hammer("off"), Some(ReflexConfig::default()));
        let both = duel_hammer("BOTH").unwrap();
        assert!(both.hammer && both.hazard_only && both.envelope && both.duel_only && both.lockout_ticks == 16);
        let reflex = duel_hammer("reflex").unwrap();
        assert!(reflex.hammer && reflex.hazard_only && !reflex.envelope && reflex.duel_only);
        let all = duel_hammer("reflex-all").unwrap();
        assert!(all.hammer && !all.hazard_only && !all.envelope && all.duel_only);
        let env = duel_hammer("envelope").unwrap();
        assert!(!env.hammer && env.envelope && env.duel_only);
        assert_eq!(duel_hammer("sometimes"), None);
        // The hybrid gets it from the options and nothing else changes.
        let opts = BrainOptions {
            reflex: both,
            ..BrainOptions::default()
        };
        assert_eq!(hybrid_config(&opts).reflex, both);
        assert_eq!(hybrid_config(&BrainOptions::default()).reflex, ReflexConfig::default());
    }

    #[test]
    fn names_round_trip_and_only_planner_family_shields_itself() {
        for k in BrainKind::ALL {
            assert_eq!(BrainKind::parse(k.name()), Some(k));
        }
        assert_eq!(BrainKind::parse("nope"), None);
        assert!(BrainKind::Planner.has_own_shield() && BrainKind::Hybrid.has_own_shield());
        assert!(!BrainKind::Scripted.has_own_shield() && !BrainKind::Fly.has_own_shield());
    }

    /// 8.7 review F1: the encoder-input control readout (an MLP with no connectome) is a measurement control and is never played live, in the
    /// plain fly or in the hybrid; the DN readouts (`linear-dn`, `mlp-dn-<H>`) are fine.
    #[test]
    fn the_encoder_input_control_readout_is_refused_by_the_live_loader_and_the_hybrid() {
        use ddai_fly::bc::HookView;
        use ddai_fly::bundle::{load_bundle, save_bundle, upgrade_hook_readout};
        use ddai_fly::hook_wide::HookReadout;
        let dir = tempfile::tempdir().unwrap();
        let (path, flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle(dir.path(), HookView::Shared);
        let base = load_bundle(&path).unwrap();
        let opts_for = |kind: HookReadout| {
            let b = upgrade_hook_readout(&base, ddai_flyg::load(&flyg).unwrap(), kind, 1).unwrap();
            let p = dir.path().join(format!("{}.bundle", kind.label()));
            save_bundle(&p, &b).unwrap();
            BrainOptions {
                fly_bundle: Some(p),
                fly_flyg: flyg.clone(),
                ..BrainOptions::default()
            }
        };
        let control = opts_for(HookReadout::EncoderMlp { hidden: 3 });
        for kind in [BrainKind::Fly, BrainKind::Hybrid] {
            let e = make_brain(kind, &control)
                .err()
                .expect("the control must be refused")
                .to_string();
            assert!(e.contains("encoder-input control"), "{kind:?}: {e}");
        }
        for readout in [HookReadout::LinearDn, HookReadout::MlpDn { hidden: 3 }] {
            let o = opts_for(readout);
            assert!(make_brain(BrainKind::Fly, &o).is_ok(), "{readout:?} fly");
            assert!(make_brain(BrainKind::Hybrid, &o).is_ok(), "{readout:?} hybrid");
        }
    }

    /// 8.8 review F3: a `Gm` neuron-model checkpoint (a pilot that is measured in the arena only) is refused by the live loader, in the plain fly and
    /// in the hybrid; the same bundle with the rate model is fine.
    #[test]
    fn a_gm_fly_is_refused_by_the_live_loader_and_the_hybrid() {
        use ddai_fly::bc::HookView;
        use ddai_fly::bundle::{NeuronModel, load_bundle, save_bundle};
        use ddai_fly::gm::GmConfig;
        let dir = tempfile::tempdir().unwrap();
        let (path, flyg_path) = ddai_fly::brain_fixtures::write_tiny_fly_bundle(dir.path(), HookView::Shared);
        let rate = load_bundle(&path).unwrap();
        let model = rate
            .build_model(ddai_flyg::load(&flyg_path).unwrap())
            .unwrap()
            .with_gm(GmConfig::default(), None, 3)
            .unwrap();
        let mut gm = rate.clone();
        gm.neuron_model = NeuronModel::Gm {
            config: GmConfig::default(),
            params: model.gm().unwrap().params().clone(),
        };
        let gm_path = dir.path().join("gm.bundle");
        save_bundle(&gm_path, &gm).unwrap();
        let opts_of = |p: &std::path::Path| BrainOptions {
            fly_bundle: Some(p.to_path_buf()),
            fly_flyg: flyg_path.clone(),
            ..BrainOptions::default()
        };
        for kind in [BrainKind::Fly, BrainKind::Hybrid] {
            let e = make_brain(kind, &opts_of(&gm_path))
                .err()
                .expect("a Gm fly must be refused")
                .to_string();
            assert!(e.contains("Gm neuron model"), "{kind:?}: {e}");
            assert!(
                make_brain(kind, &opts_of(&path)).is_ok(),
                "{kind:?}: the rate fly plays"
            );
        }
    }

    /// 8.2b F1: the live bot plays a mask-hook bundle in two views, as the arena does (one code path,
    /// `FlyBrainTemplate::instantiate_played`); the hybrid gets the second view for its proposer's hook too.
    #[test]
    fn the_bot_plays_a_masked_fly_bundle_in_two_views_like_the_arena() {
        use ddai_fly::bc::{HookView, mask_own_hook};
        use ddai_fly::brain::{ActionSelection, FlyBrainConfig};
        use ddai_fly::bundle::FlyBrainTemplate;
        let dir = tempfile::tempdir().unwrap();
        let opts_for = |view: HookView| {
            let sub = dir.path().join(format!("{view:?}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (bundle, flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle(&sub, view);
            BrainOptions {
                fly_bundle: Some(bundle),
                fly_flyg: flyg,
                ..BrainOptions::default()
            }
        };
        let (shared, masked) = (opts_for(HookView::Shared), opts_for(HookView::MaskedForHookHead));
        assert!(
            !make_brain(BrainKind::Fly, &shared)
                .unwrap()
                .name()
                .ends_with("+hookview")
        );
        let mut bot = make_brain(BrainKind::Fly, &masked).unwrap();
        assert!(bot.name().ends_with("+hookview"), "{}", bot.name());
        // The bot's decisions are the arena's: the same template through the same entry point.
        let template = FlyBrainTemplate::load(masked.fly_bundle.as_deref().unwrap(), Some(&masked.fly_flyg)).unwrap();
        let cfg = FlyBrainConfig {
            action_selection: ActionSelection::Argmax,
            seed: masked.seed,
        };
        let mut arena = template.instantiate_played(cfg.clone());
        let (mut full, mut hook_view) = (template.instantiate(cfg.clone()), template.instantiate(cfg));
        let mut me = ddai_brain::CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let map = std::sync::Arc::new(ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![Default::default(); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        });
        let reset = ddai_brain::ResetContext {
            map: map.clone(),
            self_id: 0,
            seed: 1,
        };
        bot.reset(&reset);
        arena.reset(&reset);
        full.reset(&reset);
        hook_view.reset(&reset);
        let states: Vec<i32> = (0..60)
            .map(|i| [ddai_brain::HOOK_IDLE, ddai_brain::HOOK_FLYING, ddai_brain::HOOK_GRABBED][i % 3])
            .collect();
        let mut disagreements = 0;
        for (i, &state) in states.iter().enumerate() {
            let mut other = ddai_brain::CharacterObservation::at_rest(1);
            other.pos = ddai_physics::vmath::Vec2::new(340.0 + 4.0 * (i / 3) as f32, 300.0);
            let mut self_state = me;
            self_state.hook_state = state;
            let obs = ddai_brain::Observation {
                map: map.clone(),
                tick: i as i32,
                self_state,
                others: vec![other],
                target_id: None,
                tuning: ddai_physics::tuning::TuningParams::default(),
            };
            let (a_bot, a_arena) = (bot.decide(&obs), arena.decide(&obs));
            let (f, h) = (full.decide(&obs), hook_view.decide(&mask_own_hook(&obs)));
            assert_eq!(a_bot, a_arena, "decision {i}: the bot plays what the arena plays");
            assert_eq!(a_bot, ddai_brain::Action { hook: h.hook, ..f }, "decision {i}");
            disagreements += usize::from(f.hook != h.hook);
        }
        // The fixture's hook output follows the own hook input, so the views really disagree about the hook: a bot that
        // fed the second view the unmasked observation (or played one view) would fail the equalities above.
        assert!(
            disagreements > 5,
            "masking must change the hook decision ({disagreements}/{})",
            states.len()
        );
        // The hybrid's fly proposer takes the hook probability from the second view as well ...
        assert!(make_bundle_proposer(&masked).unwrap().has_hook_view());
        assert!(!make_bundle_proposer(&shared).unwrap().has_hook_view());
        // ... and the hybrid the bot really plays (`make_brain(Hybrid)` is `make_hybrid`) charges the work clock for both
        // views: twice the price of a one-view proposer (many substeps, so that the price does not round to 0 on the
        // tiny graph).
        let priced = |view: HookView| {
            let sub = dir.path().join(format!("priced-{view:?}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (bundle, flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle_with(&sub, view, 4000);
            let mut hybrid = make_hybrid(&BrainOptions {
                fly_bundle: Some(bundle),
                fly_flyg: flyg,
                search_threads: Some(1),
                ..BrainOptions::default()
            })
            .unwrap();
            let before = hybrid.proposer_work_units();
            // After a reset the proposer lives inside the search (the other arm of `proposer_work_units`).
            hybrid.reset(&ddai_brain::ResetContext {
                map: map.clone(),
                self_id: 0,
                seed: 1,
            });
            assert_eq!(hybrid.proposer_work_units(), before);
            before
        };
        let one_view = priced(HookView::Shared);
        assert!(one_view > 100, "{one_view}");
        assert_eq!(priced(HookView::MaskedForHookHead), 2 * one_view);
        assert_eq!(make_hybrid(&BrainOptions::default()).unwrap().proposer_work_units(), 0);
        let opts = BrainOptions {
            search_threads: Some(1),
            ..masked
        };
        assert!(make_brain(BrainKind::Hybrid, &opts).is_ok());
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

    /// Task 3.16 (D-115): no option is the library default byte for byte, `Some(4)` is the same point, and a budget moves the cap with it.
    #[test]
    fn the_hybrid_budget_option_defaults_to_the_library_config_and_moves_the_cap() {
        let default_cfg = hybrid_config(&BrainOptions::default());
        assert_eq!(
            default_cfg,
            hybrid_config(&BrainOptions {
                hybrid_budget_ms: Some(4),
                ..BrainOptions::default()
            })
        );
        assert_eq!(
            default_cfg.mode,
            ddai_planner::hybrid::HybridMode::Deadline { budget_ms: 4.0 }
        );
        assert_eq!(default_cfg.decision_cap_ms, Some(5.0));
        let two = hybrid_config(&BrainOptions {
            hybrid_budget_ms: Some(2),
            ..BrainOptions::default()
        });
        assert_eq!(two.mode, ddai_planner::hybrid::HybridMode::Deadline { budget_ms: 2.0 });
        assert_eq!(two.decision_cap_ms, Some(3.0));
        // Nothing else of the config moves.
        assert_eq!(
            HybridConfig {
                mode: default_cfg.mode,
                decision_cap_ms: default_cfg.decision_cap_ms,
                ..two
            },
            default_cfg
        );
        // The brain name carries the budget (the web page and the journal show it).
        let b = make_brain(
            BrainKind::Hybrid,
            &BrainOptions {
                hybrid_budget_ms: Some(2),
                ..BrainOptions::default()
            },
        )
        .unwrap();
        assert!(b.name().contains("2ms"), "{}", b.name());
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
            // The hybrid the bot really builds follows the option, with no proposer and with a fly proposer.
            assert_eq!(make_hybrid(&opts).unwrap().config().mirror, on);
            let dir = tempfile::tempdir().unwrap();
            let (bundle, flyg) =
                ddai_fly::brain_fixtures::write_tiny_fly_bundle(dir.path(), ddai_fly::bc::HookView::MaskedForHookHead);
            let with_fly = BrainOptions {
                fly_bundle: Some(bundle),
                fly_flyg: flyg,
                ..opts
            };
            assert_eq!(make_hybrid(&with_fly).unwrap().config().mirror, on);
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
