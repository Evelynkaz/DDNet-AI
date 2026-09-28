//! [`FlyBrain`]: the `ddai_brain::Brain` implementation that plays through the fly (task 7.3,
//! acceptance criterion 7) — glues [`crate::encoder::EncoderModel`], [`crate::model::FlyModel`]/
//! [`crate::state::FlyState`] (task 7.1/7.2), and [`crate::decoder::DecoderModel`] together behind
//! one `decide()` call. Every buffer [`FlyBrain::decide`] touches is sized once in
//! [`FlyBrain::new`] — see that method's doc comment and `tests/no_alloc_brain.rs` for the
//! allocation-free proof (acceptance criterion 7).

use std::time::{Duration, Instant};

use ddai_brain::{Action, IVec2, Observation, ResetContext};
use ddai_flyg::NeuronRole;
use serde::{Deserialize, Serialize};

use crate::decoder::{DecodedAction, DecoderModel, DecoderParams, DecoderScratch, DnCalibration, decoder_forward_into};
use crate::encoder::{EncoderModel, EncoderParams, RayGridFeatures, compute_proprioception_values};
use crate::model::FlyModel;
use crate::rng::SplitMix64;
use crate::state::FlyState;

/// How [`FlyBrain::decide`] turns head probabilities into a discrete [`Action`] (task spec:
/// "argmax or sampled, configurable").
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActionSelection {
    Argmax,
    /// Samples every head independently and reproducibly from [`FlyBrainConfig::seed`] (never
    /// from an unseeded global RNG — matches `ddai_brain::ResetContext::seed`'s own convention).
    Sampled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlyBrainConfig {
    pub action_selection: ActionSelection,
    pub seed: u64,
}

impl Default for FlyBrainConfig {
    fn default() -> Self {
        FlyBrainConfig {
            action_selection: ActionSelection::Argmax,
            seed: 0,
        }
    }
}

/// A compact per-decision telemetry snapshot (task spec, FLY.md §10): per-group activity (VPN by
/// type, central by superclass, DN z-scores by action group), the decoded action, and the last
/// `decide()` call's wall-clock latency. Deliberately **excludes** the full ray-grid "eye" (task
/// spec: "compact, copyable" — a 48-direction x 4-bin x 7-channel grid every decision is neither);
/// a caller that wants that too calls [`FlyBrain::last_ray_features`] directly rather than paying
/// for it in every `telemetry()` string.
#[derive(Debug, Clone)]
pub struct FlyTelemetry {
    /// `(VPN type name, mean rate)`, one entry per `InputVisual` type present on this graph.
    pub vpn_by_type: Vec<(String, f32)>,
    /// `(superclass, mean rate)`, aggregated over every `Hidden`-role type sharing that
    /// superclass (FLY.md §10: "central by neuropil/superclass" — this graph's `types` table
    /// carries `superclass`, not a separate neuropil field, see `docs/formats.md` §8).
    pub central_by_superclass: Vec<(String, f32)>,
    /// `(output_groups action name, mean DN z-score over that action's members)`.
    pub dn_z_by_action: Vec<(String, f32)>,
    pub decoded_action: DecodedAction,
    pub latency: Duration,
}

impl FlyTelemetry {
    /// A compact, hand-written JSON encoding (no `serde_json` dependency added to this crate for
    /// one telemetry string — every value here is a plain string/number, so a hand-rolled encoder
    /// is simpler than pulling in a new dependency; type/superclass/action names are MaleCNS
    /// type names or this crate's own fixed action-name strings, never arbitrary user input, so
    /// no escaping beyond a defensive `"`-strip is needed).
    pub fn to_json(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('"', "")
        }
        fn pairs(name: &str, items: &[(String, f32)], out: &mut String) {
            out.push_str(&format!("\"{name}\":{{"));
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("\"{}\":{v}", esc(k)));
            }
            out.push('}');
        }
        let mut out = String::from("{");
        pairs("vpn_by_type", &self.vpn_by_type, &mut out);
        out.push(',');
        pairs("central_by_superclass", &self.central_by_superclass, &mut out);
        out.push(',');
        pairs("dn_z_by_action", &self.dn_z_by_action, &mut out);
        out.push_str(&format!(
            ",\"decoded_action\":{{\"direction_probs\":[{},{},{}],\"jump_prob\":{},\"hook_prob\":{},\"fire_prob\":{},\"aim_angle\":{}}}",
            self.decoded_action.direction_probs[0],
            self.decoded_action.direction_probs[1],
            self.decoded_action.direction_probs[2],
            self.decoded_action.jump_prob,
            self.decoded_action.hook_prob,
            self.decoded_action.fire_prob,
            self.decoded_action.aim_angle,
        ));
        out.push_str(&format!(",\"latency_us\":{}}}", self.latency.as_micros()));
        out
    }
}

/// Aim-angle -> DDNet target vector: `(round(cos(angle)*R), round(-sin(angle)*R))` — the ring
/// convention `crate::encoder`'s module doc comment defines (`ddx = cos`, `ddy = -sin`), scaled to
/// an explicit-target vector of length `R` (`docs/formats.md` §2's own convention for an explicit,
/// non-`aim_slot` target: "vector of length 1000" — reused here rather than invented fresh).
const AIM_TARGET_LENGTH: f32 = 1000.0;

fn aim_angle_to_target(angle: f32) -> IVec2 {
    let x = (angle.cos() * AIM_TARGET_LENGTH).round() as i32;
    let y = (-angle.sin() * AIM_TARGET_LENGTH).round() as i32;
    IVec2::new(x, y)
}

/// Each type's role, indexed like `flyg().types` — precomputed once (mirrors
/// `crate::params::type_roles`'s private helper, which this module cannot reach from outside its
/// own crate file) so [`FlyBrain::build_telemetry`] never re-scans every neuron per call.
fn role_of_type(flyg: &ddai_flyg::Flyg) -> Vec<Option<NeuronRole>> {
    let mut roles: Vec<Option<NeuronRole>> = vec![None; flyg.types.len()];
    for n in &flyg.neurons {
        let slot = &mut roles[n.type_index as usize];
        if slot.is_none() {
            *slot = Some(n.role);
        }
    }
    roles
}

/// The fly, playing through [`ddai_brain::Brain`]. Every field [`FlyBrain::decide`] touches is
/// either read-only (`model`/`encoder`/`decoder`/the three parameter sets/`calib`/`role_of_type`)
/// or a pre-sized scratch buffer (`state`/`ray_features`/`input_buf`/`decoder_scratch`/
/// `last_dn_rates`/`last_per_type_mean_rate`) — nothing is ever pushed to or reallocated inside
/// `decide` itself.
pub struct FlyBrain {
    model: FlyModel,
    state: FlyState,
    encoder: EncoderModel,
    encoder_params: EncoderParams,
    decoder: DecoderModel,
    decoder_params: DecoderParams,
    calib: DnCalibration,
    config: FlyBrainConfig,

    role_of_type: Vec<Option<NeuronRole>>,
    ray_features: RayGridFeatures,
    input_buf: Vec<f32>,
    decoder_scratch: DecoderScratch,
    last_dn_rates: Vec<f32>,
    last_per_type_mean_rate: Vec<f32>,
    last_decoded: Option<DecodedAction>,
    rng: SplitMix64,
    last_latency: Duration,
    last_warmup_converged: bool,
    name: String,
}

impl FlyBrain {
    /// Builds a `FlyBrain` and allocates every scratch buffer `decide()` will ever need, once.
    /// The fly's dynamical state (`V`) starts at all-zeros ([`crate::state::FlyState::new`]'s own
    /// default) — call [`ddai_brain::Brain::reset`] (which warms it up) before the first real
    /// decision, matching every other controller in this crate's convention that a cold, never-
    /// warmed state is a documented default, not an error.
    pub fn new(
        model: FlyModel,
        encoder: EncoderModel,
        encoder_params: EncoderParams,
        decoder: DecoderModel,
        decoder_params: DecoderParams,
        calib: DnCalibration,
        config: FlyBrainConfig,
    ) -> Self {
        let state = FlyState::new(&model);
        let num_outputs = model.num_outputs();
        let num_types = model.num_types();
        let seed = config.seed;
        let ray_features = RayGridFeatures::new(encoder.ray_grid_config());
        let input_buf = vec![0.0; encoder.num_inputs()];
        let decoder_scratch = DecoderScratch::new(&decoder);
        let role_of_type = role_of_type(model.flyg());
        let name = format!("fly-{}n-{}e", model.num_neurons(), model.flyg().edges.num_edges());
        FlyBrain {
            model,
            state,
            encoder,
            encoder_params,
            decoder,
            decoder_params,
            calib,
            config,
            role_of_type,
            ray_features,
            input_buf,
            decoder_scratch,
            last_dn_rates: vec![0.0; num_outputs],
            last_per_type_mean_rate: vec![0.0; num_types],
            last_decoded: None,
            rng: SplitMix64::new(seed),
            last_latency: Duration::ZERO,
            last_warmup_converged: false,
            name,
        }
    }

    pub fn model(&self) -> &FlyModel {
        &self.model
    }

    pub fn last_ray_features(&self) -> &RayGridFeatures {
        &self.ray_features
    }

    pub fn last_latency(&self) -> Duration {
        self.last_latency
    }

    /// Whether the most recent `reset()`'s warm-up (`crate::state::FlyState::warm_up`) actually
    /// converged before its cap — surfaced here (rather than silently discarded) per that
    /// method's own `#[must_use]`: a caller that cares (the demo, `ddnet-ai`) can log/assert on
    /// this; `Brain::reset`'s `()` return type has no other way to report it.
    pub fn last_warmup_converged(&self) -> bool {
        self.last_warmup_converged
    }

    fn select_direction(&mut self, probs: [f32; 3]) -> i32 {
        let idx = match self.config.action_selection {
            ActionSelection::Argmax => (0..3)
                .max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap())
                .unwrap(),
            ActionSelection::Sampled => {
                let u = self.rng.next_f32_unit();
                let mut cum = 0.0f32;
                let mut chosen = 2;
                for (i, &p) in probs.iter().enumerate() {
                    cum += p;
                    if u < cum {
                        chosen = i;
                        break;
                    }
                }
                chosen
            }
        };
        // `decoder.config().direction_actions` is `[left, stop, right]` -> `[-1, 0, 1]`.
        [-1, 0, 1][idx]
    }

    fn select_bool(&mut self, prob: f32) -> bool {
        match self.config.action_selection {
            ActionSelection::Argmax => prob >= 0.5,
            ActionSelection::Sampled => self.rng.next_f32_unit() < prob,
        }
    }

    fn build_telemetry(&self, decoded: DecodedAction) -> FlyTelemetry {
        let flyg = self.model.flyg();
        let mut vpn_by_type = Vec::new();
        let mut central_sum: std::collections::BTreeMap<String, (f32, u32)> = std::collections::BTreeMap::new();
        for (ti, ty) in flyg.types.iter().enumerate() {
            match self.role_of_type[ti] {
                Some(NeuronRole::InputVisual) => vpn_by_type.push((ty.name.clone(), self.last_per_type_mean_rate[ti])),
                Some(NeuronRole::Hidden) => {
                    let entry = central_sum.entry(ty.superclass.clone()).or_insert((0.0, 0));
                    entry.0 += self.last_per_type_mean_rate[ti];
                    entry.1 += 1;
                }
                _ => {}
            }
        }
        let central_by_superclass = central_sum
            .into_iter()
            .map(|(k, (sum, n))| (k, sum / n.max(1) as f32))
            .collect();

        let clip_at = self.decoder.config().z_clip;
        let z = self.calib.z(&self.last_dn_rates, clip_at);
        let mut dn_z_by_action = Vec::new();
        for group in &flyg.output_groups {
            if group.members.is_empty() {
                continue;
            }
            let mean: f32 = group
                .members
                .iter()
                .filter_map(|m| self.model.output_slot_for_neuron(m.neuron_index))
                .map(|slot| z[slot])
                .sum::<f32>()
                / group.members.len() as f32;
            dn_z_by_action.push((group.action.clone(), mean));
        }

        FlyTelemetry {
            vpn_by_type,
            central_by_superclass,
            dn_z_by_action,
            decoded_action: decoded,
            latency: self.last_latency,
        }
    }
}

impl ddai_brain::Brain for FlyBrain {
    fn reset(&mut self, ctx: &ResetContext) {
        // Review round 1, F13 (CONFIRMED): re-seeds from `ctx.seed`, not `self.config.seed` — an
        // earlier revision ignored `ctx` entirely, so every episode's `Sampled` action-selection
        // noise replayed byte-for-byte identically regardless of what the caller's `ResetContext`
        // asked for, contradicting that field's own doc comment ("a brain that samples must do so
        // reproducibly from *this*"). `self.config.seed` remains what `FlyBrain::new` seeds from
        // before any `reset` call.
        self.rng = SplitMix64::new(ctx.seed);
        self.last_warmup_converged = self.state.warm_up(&self.model).converged;
        self.last_decoded = None;
    }

    fn decide(&mut self, obs: &Observation) -> Action {
        let start = Instant::now();

        let an_values = compute_proprioception_values(&obs.self_state, self.encoder.ray_grid_config());
        self.ray_features.compute(obs, self.encoder.ray_grid_config());
        self.encoder.forward(
            &self.ray_features,
            &an_values,
            &self.encoder_params,
            &mut self.input_buf,
        );

        let output = self.state.step_decision(&self.model, &self.input_buf);
        self.last_dn_rates.copy_from_slice(output.dn_rates);
        self.last_per_type_mean_rate.copy_from_slice(output.per_type_mean_rate);

        let decoded = decoder_forward_into(
            &self.decoder,
            &self.last_dn_rates,
            &self.calib,
            &self.decoder_params,
            &mut self.decoder_scratch,
        );

        let direction = self.select_direction(decoded.direction_probs);
        let jump = self.select_bool(decoded.jump_prob);
        let hook = self.select_bool(decoded.hook_prob);
        let fire = self.select_bool(decoded.fire_prob);
        let target = aim_angle_to_target(decoded.aim_angle);

        self.last_decoded = Some(decoded);
        self.last_latency = start.elapsed();

        Action {
            direction,
            jump,
            hook,
            fire,
            target,
            wanted_weapon: None,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn telemetry(&self) -> Option<String> {
        let decoded = self.last_decoded?;
        Some(self.build_telemetry(decoded).to_json())
    }
}

#[cfg(test)]
mod tests;
