//! [`FlyProposer`] (task 3.5, D-041): the fly as the *proposing* half of the hybrid brain. One
//! `decide()` of the wrapped [`FlyBrain`] (~1 ms) yields the action distribution of the decision;
//! [`ddai_planner::hybrid::plans_from_distribution`] turns it into `K` plans in the planner's plan
//! encoding (the argmax plan plus sampled ones), and the exact search scores them next to its own
//! candidates. A proposal is only a candidate: the fly never decides.
//!
//! Also here: [`untrained_fly_brain`], a fly with default parameters (the state before phase-8
//! training), used to prove the plumbing and measure the cost of proposing; it says nothing about
//! strength.

use std::path::Path;

use ddai_brain::{Brain, ResetContext};
use ddai_planner::hybrid::{ActionDistribution, ProposalOutcome, ProposeCtx, Proposer, plans_from_distribution};
use ddai_planner::planner::PlanStep;

use crate::brain::{FlyBrain, FlyBrainConfig};
use crate::brain_config::load_brain_config;
use crate::decoder::DecoderModel;
use crate::encoder::EncoderParams;
use crate::rng::SplitMix64;
use crate::{FlyConfig, FlyModel, FlyParams};

/// What one fly proposal costs on the arena's **work clock**, per mega-synapse-step (task 3.7a, D-080), in tee-tick
/// equivalents (one tee-tick = `WORK_US_PER_TEE_TICK`, 1.25 us, of a whole search decision). A proposal runs
/// `substeps_per_decision` exponential-Euler steps over the graph's synapses (`nnz`), plus the encoder and the
/// plan sampling, whose cost is folded into the constant: `units = nnz x substeps x this / 10^6`. Calibrated on the
/// S graph (49 741 synapse rows, 4 substeps, one proposal = 500 tee-ticks = 0.63 ms on an unloaded core), by
/// `tests/fly_proposal_cost.rs` (E-012 §1): the ratio of the fly's wall time to the wall time per tee-tick of a whole
/// decision, measured in the same process so that the load of a shared machine scales both alike. Proportional by
/// construction; for a graph of another size it is an extrapolation.
pub const FLY_TEE_TICKS_PER_MEGA_SYNAPSE_STEP: f64 = 2513.0;

/// The work-clock price of one proposal of `brain` (see [`FLY_TEE_TICKS_PER_MEGA_SYNAPSE_STEP`]).
pub fn proposal_tee_ticks(brain: &FlyBrain) -> u64 {
    let model = brain.model();
    let nnz = model.flyg().edges.row_start.last().copied().unwrap_or(0) as f64;
    let substeps = f64::from(model.config().substeps_per_decision);
    (nnz * substeps * FLY_TEE_TICKS_PER_MEGA_SYNAPSE_STEP / 1e6).round() as u64
}

/// The fly as a [`Proposer`].
pub struct FlyProposer {
    brain: FlyBrain,
    /// A second fly for the hook head (`HookView::MaskedForHookHead`): it sees the observation with the own hook
    /// state hidden and supplies the hook probability; everything else comes from `brain`.
    hook_brain: Option<FlyBrain>,
    rng: SplitMix64,
    name: String,
    /// The work-clock price of one `propose` call, in tee-tick equivalents.
    units: u64,
    /// Wall time of the last `decide()` (µs) and the decisions made (cost reporting).
    pub last_decide_us: u64,
    pub decisions: u64,
}

impl FlyProposer {
    pub fn new(brain: FlyBrain, seed: u64) -> FlyProposer {
        FlyProposer {
            units: proposal_tee_ticks(&brain),
            brain,
            hook_brain: None,
            rng: SplitMix64::new(seed),
            name: "fly".to_string(),
            last_decide_us: 0,
            decisions: 0,
        }
    }

    /// A proposer over a trained bundle, played the way the bundle was trained: a model with the hook head masked
    /// (`HookView::MaskedForHookHead`) gets its second view for the hook probability.
    ///
    /// Refuses a checkpoint with an intent hook head or a latched decode (`FlyBrainTemplate::require_unlatched`): the proposer's latch would
    /// follow the fly's own argmax, not the action the hybrid plays. Also refuses the encoder-input control readout
    /// (`FlyBrainTemplate::require_fly_readout`).
    pub fn from_template(
        template: &crate::bundle::FlyBrainTemplate,
        config: FlyBrainConfig,
        seed: u64,
    ) -> Result<FlyProposer, crate::bundle::BundleError> {
        template.require_unlatched("hybrid fly proposer")?;
        template.require_fly_readout("hybrid fly proposer")?;
        let masked = template.hook_view() == crate::bc::HookView::MaskedForHookHead;
        let hook_brain = masked.then(|| template.instantiate(config.clone()));
        let p = FlyProposer::new(template.instantiate(config), seed);
        Ok(match hook_brain {
            Some(h) => p.with_hook_brain(h),
            None => p,
        })
    }

    pub fn brain(&self) -> &FlyBrain {
        &self.brain
    }

    /// Whether the hook probability comes from a second, masked view.
    pub fn has_hook_view(&self) -> bool {
        self.hook_brain.is_some()
    }

    /// The action distribution of one decision: the fly's heads, with the hook probability from the second, masked view
    /// when there is one (and then the viewer's frame shows that played hook and both views' time). `None` before the
    /// fly has decoded anything.
    pub fn distribution(&mut self, obs: &ddai_brain::Observation) -> Option<ActionDistribution> {
        let t0 = std::time::Instant::now();
        let _ = self.brain.decide(obs);
        let hook_view = self.hook_brain.as_mut().map(|h| {
            let masked = crate::bc::mask_own_hook(obs);
            let played = h.decide(&masked);
            (played.hook, h.last_decoded().map(|d| d.hook_prob))
        });
        self.last_decide_us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.decisions += 1;
        let d = *self.brain.last_decoded()?;
        let mut hook_prob = d.hook_prob;
        if let (Some(h), Some((hook, Some(p)))) = (&self.hook_brain, hook_view) {
            hook_prob = p;
            // The viewer's frame shows the hook the proposal is built from (the masked view's own decision) and both
            // views' time.
            self.brain.set_played_override(Some(crate::brain::PlayedOverride {
                hook,
                hook_prob: p,
                latency: self.brain.last_latency() + h.last_latency(),
            }));
        }
        // The fly's aim angle uses the ring convention `(cos a, -sin a)`; the planner's angles are
        // `atan2(dy, dx)` with y down, hence the sign.
        Some(ActionDistribution {
            direction: d.direction_probs.map(f64::from),
            jump: f64::from(d.jump_prob),
            hook: f64::from(hook_prob),
            fire: f64::from(d.fire_prob),
            aim_angle: -f64::from(d.aim_angle),
        })
    }

    /// A proposer for a model played in two views: `hook_brain` decides the hook probability.
    ///
    /// The second view is a second network run per proposal, so its price (`proposal_tee_ticks`) is added to the
    /// work-clock price: a masked proposer costs twice a shared one (3.7a's `proposal_in_cap` takes that price off
    /// the search budget; one view's price would hand a two-view proposer free search time).
    pub fn with_hook_brain(mut self, hook_brain: FlyBrain) -> FlyProposer {
        self.units += proposal_tee_ticks(&hook_brain);
        self.hook_brain = Some(hook_brain);
        self
    }
}

impl Proposer for FlyProposer {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self, ctx: &ResetContext) {
        self.brain.reset(ctx);
        if let Some(h) = &mut self.hook_brain {
            h.reset(ctx);
        }
        self.rng = SplitMix64::new(ctx.seed ^ 0xF1F1_F1F1);
    }

    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        let Some(dist) = self.distribution(ctx.obs) else {
            return;
        };
        let rng = &mut self.rng;
        plans_from_distribution(&dist, ctx.steps, ctx.k, &mut || f64::from(rng.next_f32_unit()), out);
    }

    fn work_units(&self) -> u64 {
        self.units
    }

    fn viz_meta(&self) -> Option<String> {
        Some(self.brain.viz_meta_json("proposer"))
    }

    fn viz_frame(&mut self, tick: u32, outcome: Option<ProposalOutcome>) -> Option<&[u8]> {
        self.brain.viz_frame_with(tick, outcome)
    }
}

/// A fly with default (untrained) parameters on the compiled graph `flyg_path` and the brain
/// config `brain_config_path`: the network as it is before phase-8 training. Warm-up happens at
/// the first `reset`.
pub fn untrained_fly_brain(flyg_path: &Path, brain_config_path: &Path, seed: u64) -> Result<FlyBrain, String> {
    let flyg = ddai_flyg::load(flyg_path).map_err(|e| format!("{}: {e}", flyg_path.display()))?;
    let brain_cfg =
        load_brain_config(brain_config_path).map_err(|e| format!("{}: {e}", brain_config_path.display()))?;
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, seed);
    let model = FlyModel::new(flyg, config, params).map_err(|e| format!("fly model: {e}"))?;
    let encoder = brain_cfg.encoder_model(&model).map_err(|e| format!("encoder: {e}"))?;
    let encoder_params = EncoderParams::init_default(encoder.num_params());
    let decoder = DecoderModel::new(&model, brain_cfg.decoder).map_err(|e| format!("decoder: {e}"))?;
    let decoder_params = decoder.init_default_params();
    let calib = crate::calibrate_from_rest(&model, seed, decoder.config().min_sigma)
        .map_err(|e| format!("calibration: {e}"))?;
    Ok(FlyBrain::new(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        FlyBrainConfig {
            seed,
            ..FlyBrainConfig::default()
        },
    ))
}
