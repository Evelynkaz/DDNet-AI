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
use ddai_planner::hybrid::{ActionDistribution, ProposeCtx, Proposer, plans_from_distribution};
use ddai_planner::planner::PlanStep;

use crate::brain::{FlyBrain, FlyBrainConfig};
use crate::brain_config::load_brain_config;
use crate::decoder::DecoderModel;
use crate::encoder::{EncoderModel, EncoderParams};
use crate::rng::SplitMix64;
use crate::{FlyConfig, FlyModel, FlyParams};

/// The fly as a [`Proposer`].
pub struct FlyProposer {
    brain: FlyBrain,
    rng: SplitMix64,
    name: String,
    /// Wall time of the last `decide()` (µs) and the decisions made (cost reporting).
    pub last_decide_us: u64,
    pub decisions: u64,
}

impl FlyProposer {
    pub fn new(brain: FlyBrain, seed: u64) -> FlyProposer {
        FlyProposer {
            brain,
            rng: SplitMix64::new(seed),
            name: "fly".to_string(),
            last_decide_us: 0,
            decisions: 0,
        }
    }

    pub fn brain(&self) -> &FlyBrain {
        &self.brain
    }
}

impl Proposer for FlyProposer {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self, ctx: &ResetContext) {
        self.brain.reset(ctx);
        self.rng = SplitMix64::new(ctx.seed ^ 0xF1F1_F1F1);
    }

    fn propose(&mut self, ctx: &ProposeCtx<'_>, out: &mut Vec<Vec<PlanStep>>) {
        let t0 = std::time::Instant::now();
        let _ = self.brain.decide(ctx.obs);
        self.last_decide_us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.decisions += 1;
        let Some(d) = self.brain.last_decoded() else {
            return;
        };
        // The fly's aim angle uses the ring convention `(cos a, -sin a)`; the planner's angles are
        // `atan2(dy, dx)` with y down, hence the sign.
        let dist = ActionDistribution {
            direction: d.direction_probs.map(f64::from),
            jump: f64::from(d.jump_prob),
            hook: f64::from(d.hook_prob),
            fire: f64::from(d.fire_prob),
            aim_angle: -f64::from(d.aim_angle),
        };
        let rng = &mut self.rng;
        plans_from_distribution(&dist, ctx.steps, ctx.k, &mut || f64::from(rng.next_f32_unit()), out);
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
    let encoder = EncoderModel::new(&model, brain_cfg.ray_grid, &brain_cfg.proprioception)
        .map_err(|e| format!("encoder: {e}"))?;
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
