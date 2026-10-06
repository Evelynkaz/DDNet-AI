//! The ray-grid "eye" plus proprioception encoder (task 7.3, acceptance criteria 2/3): turns a
//! [`ddai_brain::Observation`] into per-input-neuron currents for [`crate::state::FlyState::
//! step_decision`], through the real VPN/AN neurons' receptive fields (`.flyg`'s `input_channels`
//! for VPN, [`ProprioceptionConfig`] here for AN — AN types carry no functional channel mapping in
//! the `.flyg` format itself, see `docs/formats.md` §8).
//!
//! ## The mapping (FLY.md §5)
//! `I_i = Σ_channel∈channels(type(i)) [ Σ_rays G(RF_i, φ_ray) · feature_{channel,ray} ·
//! g_{type(i),channel} + c_{type(i),channel} ]` for a **spatial** channel (a value per ray
//! direction/distance bin — walls, hazards, opponents, ...); for a **scalar** channel (own
//! velocity, or a proprioceptive quantity) there is no ray grid to weight over, so the term is
//! just `feature_channel · direction_factor(i) · g + c` (see [`Channel::is_spatial`] and the
//! "Own-motion channels" section below for `direction_factor`). `g`/`c` are the only learnable
//! part of this module, one pair per **(type, channel)** — shared between a type's L and R
//! copies, per FLY.md §4's "left/right copies share parameters" (this is also what makes the
//! encoder itself mirror-symmetric — see `tests/brain_mirror.rs`).
//!
//! ## Ring angle convention (screen -> receptive-field angle)
//! The game's screen coordinates have `x` rightward, `y` **downward** (DDNet/Teeworlds
//! convention). Every angle this module computes — a ray's direction, a neuron's receptive-field
//! center, an opponent's bearing — is expressed as one **ring angle** `θ`, `0` = pointing along
//! `+x` (right), `π/2` = pointing "up"/dorsal (against gravity, screen `-y`), via the single
//! conversion `ring_angle_from_screen_delta(dx, dy) = atan2(-dy, dx)`. A `.flyg` receptive
//! field's `(azimuth_deg, elevation_deg)` (FLY.md §5: azimuth negative = left eye, positive =
//! right; elevation positive = dorsal) is folded into this same ring via
//! `ring_angle_from_screen_delta(sin(azimuth), -elevation_gain * sin(elevation))` — an engineering
//! bridge (label **П**, not biological) between the compound eye's 2D receptive field and the
//! game's single ring of rays, chosen because it is the simplest composition under which the
//! whole pipeline stays exactly mirror-symmetric (see the module-level property this buys,
//! spelled out in `tests/brain_mirror.rs`'s doc comment): negating `azimuth` (the L/R-homolog
//! relationship the `.flyg` format already gives every VPN pair) maps `θ -> π - θ`, exactly the
//! ring angle an X-flipped bearing/ray direction also maps to, for *any* `elevation_gain`
//! ([`RayGridConfig::elevation_gain`], review round 1, F9 — the gain only rescales the untouched-
//! by-mirroring elevation term).

use std::f32::consts::TAU;

use ddai_brain::{CharacterObservation, HOOK_FLYING, HOOK_GRABBED, Observation};
use ddai_flyg::NeuronRole;
use serde::{Deserialize, Serialize};

use crate::model::FlyModel;

// --- Channel names (FLY.md §5's table; exact strings used by `configs/fly/{S,M}.toml`'s
// `[[input_channels]]` entries and this module's own `ProprioceptionConfig`). -------------------

pub const CH_OPPONENT_POSITION: &str = "opponent_position";
pub const CH_OPPONENT_APPROACH: &str = "opponent_approach";
pub const CH_OPPONENT_HOOK: &str = "opponent_hook";
pub const CH_OTHER_PLAYERS: &str = "other_players";
pub const CH_WALLS: &str = "walls";
pub const CH_FREEZE_DEATH_TILES: &str = "freeze_death_tiles";
pub const CH_NO_HOOK_TILES: &str = "no_hook_tiles";
pub const CH_SELF_VELOCITY_X: &str = "self_velocity_x";
pub const CH_SELF_VELOCITY_Y: &str = "self_velocity_y";
pub const CH_SELF_VELOCITY_FLOW: &str = "self_velocity_flow";

/// Every visual channel name FLY.md §5's table names, in table order — used to report which ones
/// end up with zero neurons on a given graph (acceptance criterion 2: "record ... which channels
/// end up with no neurons").
pub const ALL_VISUAL_CHANNELS: [&str; 10] = [
    CH_OPPONENT_POSITION,
    CH_OPPONENT_APPROACH,
    CH_OPPONENT_HOOK,
    CH_OTHER_PLAYERS,
    CH_WALLS,
    CH_FREEZE_DEATH_TILES,
    CH_NO_HOOK_TILES,
    CH_SELF_VELOCITY_X,
    CH_SELF_VELOCITY_Y,
    CH_SELF_VELOCITY_FLOW,
];

/// One visual (VPN) input channel — a spatial one carries a value per ray direction/distance bin
/// in [`RayGridFeatures`]; a scalar one (own motion) carries a single value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    OpponentPosition,
    OpponentApproach,
    OpponentHook,
    OtherPlayers,
    Walls,
    FreezeDeathTiles,
    NoHookTiles,
    SelfVelocityX,
    SelfVelocityY,
    SelfVelocityFlow,
}

impl Channel {
    pub fn parse(name: &str) -> Option<Channel> {
        Some(match name {
            CH_OPPONENT_POSITION => Channel::OpponentPosition,
            CH_OPPONENT_APPROACH => Channel::OpponentApproach,
            CH_OPPONENT_HOOK => Channel::OpponentHook,
            CH_OTHER_PLAYERS => Channel::OtherPlayers,
            CH_WALLS => Channel::Walls,
            CH_FREEZE_DEATH_TILES => Channel::FreezeDeathTiles,
            CH_NO_HOOK_TILES => Channel::NoHookTiles,
            CH_SELF_VELOCITY_X => Channel::SelfVelocityX,
            CH_SELF_VELOCITY_Y => Channel::SelfVelocityY,
            CH_SELF_VELOCITY_FLOW => Channel::SelfVelocityFlow,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Channel::OpponentPosition => CH_OPPONENT_POSITION,
            Channel::OpponentApproach => CH_OPPONENT_APPROACH,
            Channel::OpponentHook => CH_OPPONENT_HOOK,
            Channel::OtherPlayers => CH_OTHER_PLAYERS,
            Channel::Walls => CH_WALLS,
            Channel::FreezeDeathTiles => CH_FREEZE_DEATH_TILES,
            Channel::NoHookTiles => CH_NO_HOOK_TILES,
            Channel::SelfVelocityX => CH_SELF_VELOCITY_X,
            Channel::SelfVelocityY => CH_SELF_VELOCITY_Y,
            Channel::SelfVelocityFlow => CH_SELF_VELOCITY_FLOW,
        }
    }

    /// `true` for a channel that carries one value per ray direction/distance bin
    /// ([`RayGridFeatures::spatial`]); `false` for a single-scalar own-motion channel
    /// ([`RayGridFeatures::scalar`]).
    pub fn is_spatial(self) -> bool {
        !matches!(
            self,
            Channel::SelfVelocityX | Channel::SelfVelocityY | Channel::SelfVelocityFlow
        )
    }
}

/// One proprioceptive (AN) input channel (FLY.md §5's "Тело (проприоцепция)"). AN types in
/// MaleCNS v1.0 carry no functional name (`docs/formats.md` §8), so — unlike [`Channel`] — the
/// channel -> AN-type assignment is not baked into the `.flyg` at all; it lives in
/// [`ProprioceptionConfig`] instead, and every mapping here is labeled **П** (arbitrary) in the
/// crate README, not **Б**/**С** (contrast FLY.md §5's own text, which names specific functional
/// AN classes like "Walk/Rest AN" that MaleCNS v1.0's actual per-type names don't let this
/// codebase identify).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProprioceptionChannel {
    Grounded,
    Airborne,
    OwnHook,
    JumpsLeft,
    FreezeTimer,
    Speed,
}

pub const PROPRIOCEPTION_CHANNELS: [ProprioceptionChannel; 6] = [
    ProprioceptionChannel::Grounded,
    ProprioceptionChannel::Airborne,
    ProprioceptionChannel::OwnHook,
    ProprioceptionChannel::JumpsLeft,
    ProprioceptionChannel::FreezeTimer,
    ProprioceptionChannel::Speed,
];

impl ProprioceptionChannel {
    pub fn name(self) -> &'static str {
        match self {
            ProprioceptionChannel::Grounded => "grounded",
            ProprioceptionChannel::Airborne => "airborne",
            ProprioceptionChannel::OwnHook => "own_hook",
            ProprioceptionChannel::JumpsLeft => "jumps_left",
            ProprioceptionChannel::FreezeTimer => "freeze_timer",
            ProprioceptionChannel::Speed => "speed",
        }
    }
}

/// One input channel about the **target opponent's own state** (task 8.5a): the observation always carried it
/// (`CharacterObservation::{is_frozen, freeze_ticks_remaining, vel, hook_state}`) but the fly was never shown it, so it
/// could not tell a frozen opponent from a free one. All five are scalars (no ray grid): mirror-symmetric ones feed the
/// neurons of a type as they are, the signed velocity ones are weighted by each neuron's own direction factor like the
/// own-velocity channels. They are wired by [`OpponentStateConfig`] (the `[opponent_state]` section of a brain config),
/// not by the `.flyg`, with parameters appended **after** every existing `(type, channel)` pair and zero-initialised by
/// the bundle upgrade, so a bundle trained without them plays bit for bit the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpponentChannel {
    Frozen,
    FreezeLeft,
    VelocityX,
    VelocityY,
    HookState,
}

pub const OPPONENT_CHANNELS: [OpponentChannel; 5] = [
    OpponentChannel::Frozen,
    OpponentChannel::FreezeLeft,
    OpponentChannel::VelocityX,
    OpponentChannel::VelocityY,
    OpponentChannel::HookState,
];

impl OpponentChannel {
    pub fn name(self) -> &'static str {
        match self {
            OpponentChannel::Frozen => "opponent_frozen",
            OpponentChannel::FreezeLeft => "opponent_freeze_left",
            OpponentChannel::VelocityX => "opponent_velocity_x",
            OpponentChannel::VelocityY => "opponent_velocity_y",
            OpponentChannel::HookState => "opponent_hook_state",
        }
    }

    fn slot(self) -> usize {
        self as usize
    }

    /// The signed channels need a neuron's direction factor, which only visual neurons have.
    fn needs_visual(self) -> bool {
        matches!(self, OpponentChannel::VelocityX | OpponentChannel::VelocityY)
    }
}

/// `[opponent_state]` of a brain config: per channel, the names of the input types (visual or ascending) it feeds.
/// Empty everywhere (the default, and every config written before task 8.5a) = no such channels.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpponentStateConfig {
    #[serde(default)]
    pub frozen: Vec<String>,
    #[serde(default)]
    pub freeze_left: Vec<String>,
    #[serde(default)]
    pub velocity_x: Vec<String>,
    #[serde(default)]
    pub velocity_y: Vec<String>,
    #[serde(default)]
    pub hook_state: Vec<String>,
}

impl OpponentStateConfig {
    fn types_for(&self, channel: OpponentChannel) -> &[String] {
        match channel {
            OpponentChannel::Frozen => &self.frozen,
            OpponentChannel::FreezeLeft => &self.freeze_left,
            OpponentChannel::VelocityX => &self.velocity_x,
            OpponentChannel::VelocityY => &self.velocity_y,
            OpponentChannel::HookState => &self.hook_state,
        }
    }

    pub fn is_empty(&self) -> bool {
        OPPONENT_CHANNELS.iter().all(|&c| self.types_for(c).is_empty())
    }
}

/// A visual or proprioceptive input channel, unified for [`EncoderModel`]'s single flat parameter
/// table (one shared `(type, channel)` -> `param_id` numbering across VPN and AN alike).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputChannel {
    Visual(Channel),
    Ascending(ProprioceptionChannel),
    Opponent(OpponentChannel),
}

impl InputChannel {
    pub fn name(self) -> &'static str {
        match self {
            InputChannel::Visual(c) => c.name(),
            InputChannel::Ascending(c) => c.name(),
            InputChannel::Opponent(c) => c.name(),
        }
    }
}

// --- Config (task spec: "configurable in a TOML under configs/fly/") ---------------------------

/// The ray grid's geometry plus a handful of shaping constants (RF/object angular width, looming
/// saturation, own-velocity normalization). Defaults match FLY.md §5 ("48-64 directions x 3-4
/// distance bins, ~20 tiles range"); see each field's doc comment for the rest.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RayGridConfig {
    /// Number of ray directions around the tee, evenly spaced over the full circle. **Must be
    /// even** ([`RayGridConfig::validate`]) — mirror symmetry (`tests/brain_mirror.rs`) needs
    /// `π` radians to land exactly on another grid direction (see the module doc comment's ring
    /// convention), which only holds when `num_directions` is even.
    pub num_directions: usize,
    /// Distance bins per ray, from the tee out to `max_range_tiles`.
    pub num_distance_bins: usize,
    pub max_range_tiles: f32,
    /// Angular half-width (degrees, standard deviation of the Gaussian) of a receptive field's
    /// weighting over ray directions (FLY.md §5: "`G` is a fixed Gaussian over angular
    /// distance"). Fixed (not learnable) — only `g`/`c` are.
    pub rf_sigma_deg: f32,
    /// Angular half-width (degrees) of the Gaussian bump a point object (an opponent, another
    /// player, an opponent's hook) is spread over when it doesn't exactly land on one ray
    /// direction — label **П** (an engineering choice, not derived from any specific LC
    /// physiology).
    pub object_sigma_deg: f32,
    /// Saturation constant for the looming feature (`opponent_approach`, label **Б**: "угловой
    /// рост ~ 2rv/d²" per FLY.md §5): `feature = closing_rate / (closing_rate + looming_k)` for
    /// `closing_rate > 0`, `0` otherwise — bounds the raw `2rv/d²` quantity into `[0, 1)` without
    /// an exponential. **Units** (review round 1, F3/F4, CONFIRMED): `r`/`d` are pixels, `v` is
    /// `ddai_brain::CharacterObservation::vel`'s own unit, **pixels per server tick** (never
    /// pixels/second — an earlier revision's doc comment said "pixels/second-ish" and picked
    /// `looming_k = 400.0` accordingly, which made the feature saturate to essentially `0` for
    /// every realistic approach: at `v = 10` px/tick (`tuning.ground_control_speed()`'s own
    /// value) closing head-on from 3 tiles, `2rv/d² ≈ 0.03`, giving `0.03/400 ≈ 7.5e-5`). `0.03`
    /// (this field's new default) makes that same realistic approach give `≈ 0.5` (see
    /// `tests::looming_feature_is_meaningfully_nonzero_for_a_realistic_approach`) while a
    /// receding/near-stationary opponent still saturates to `≈ 0`.
    pub looming_k: f32,
    /// Normalization scale for own-velocity scalar channels (`self_velocity_x/y/flow`): raw
    /// **pixels per server tick** (matches `CharacterObservation::vel`'s unit — review round 1,
    /// F4) divided by this, then clamped to `[-1, 1]` (`[0, 1]` for the unsigned `flow` channel)
    /// — label **П**. `30.0` is `3x` DDNet's default ground-running speed
    /// (`tuning.ground_control_speed()` = `10.0` px/tick), giving headroom for hook-boosted
    /// movement without making ordinary running saturate the channel.
    pub velocity_norm_scale: f32,
    /// Review round 1, F9 (CONFIRMED, minor): a real MaleCNS VPN's receptive field never comes
    /// anywhere near straight up/down — on the real S graph, `max(|elevation_deg|) == 47.06°`
    /// across all 1290 visual receptive fields, `p99 == 43.59°` (see
    /// `tests::dump_real_s_elevation_and_ring_angle_distribution`, the diagnostic this default
    /// was picked from) — so folding `(azimuth, elevation)` into one ring angle via plain
    /// `atan2(sin(elevation), sin(azimuth))` (gain `1.0`) put **every single one** of them within
    /// 60° of the horizontal axis, leaving the fly's vision of directly-above/below essentially
    /// unrepresented. This field multiplies `sin(elevation_deg)` before the `atan2` (label **П**:
    /// an engineering compensation, not a biological quantity) — `2.0` (this default) spreads that
    /// same real distribution's *median* distance-from-horizontal from `23.5°` to `41.1°` and
    /// drops the "everyone within 60° of horizontal" fraction from `100%` to `87.7%`, a meaningful
    /// improvement without the more extreme (and, per the same diagnostic, less azimuth-
    /// preserving) `~50%` a gain of `4.0` would give. Exactly mirror-symmetric for any value
    /// (`ring_angle_from_rf`'s own doc comment): elevation is untouched by an X-mirror, so scaling
    /// it by a constant before the `atan2` cannot affect the `azimuth -> -azimuth` mirror
    /// argument, which only depends on `atan2`'s `x`-argument sign.
    pub elevation_gain: f32,
    /// Task 8.2: learn a gain per **distance bin** for every `(type, channel)` (see
    /// [`EncoderParams::bin_gain`]). Off (the default, and the 7.3 behaviour) the current of a
    /// spatial term sums a ray's distance bins with equal weight, and because the Gaussian bins
    /// partition distance (their sum is ~constant), a neuron cannot tell a wall, hazard or
    /// opponent that is one tile away from one that is twenty tiles away.
    #[serde(default)]
    pub learn_distance_gains: bool,
}

impl Default for RayGridConfig {
    fn default() -> Self {
        RayGridConfig {
            num_directions: 48,
            num_distance_bins: 4,
            max_range_tiles: 20.0,
            rf_sigma_deg: 20.0,
            object_sigma_deg: 12.0,
            looming_k: 0.03,
            velocity_norm_scale: 30.0,
            elevation_gain: 2.0,
            learn_distance_gains: false,
        }
    }
}

impl RayGridConfig {
    pub fn validate(&self) -> Result<(), EncoderError> {
        if self.num_directions == 0 || !self.num_directions.is_multiple_of(2) {
            return Err(EncoderError::InvalidConfig(format!(
                "num_directions must be even and > 0, got {}",
                self.num_directions
            )));
        }
        if self.num_distance_bins == 0 {
            return Err(EncoderError::InvalidConfig(
                "num_distance_bins must be >= 1".to_string(),
            ));
        }
        if !(self.max_range_tiles > 0.0 && self.max_range_tiles.is_finite()) {
            return Err(EncoderError::InvalidConfig("max_range_tiles must be > 0".to_string()));
        }
        if !(self.rf_sigma_deg > 0.0 && self.rf_sigma_deg.is_finite()) {
            return Err(EncoderError::InvalidConfig("rf_sigma_deg must be > 0".to_string()));
        }
        if !(self.object_sigma_deg > 0.0 && self.object_sigma_deg.is_finite()) {
            return Err(EncoderError::InvalidConfig("object_sigma_deg must be > 0".to_string()));
        }
        if !(self.looming_k > 0.0 && self.looming_k.is_finite()) {
            return Err(EncoderError::InvalidConfig("looming_k must be > 0".to_string()));
        }
        if !(self.velocity_norm_scale > 0.0 && self.velocity_norm_scale.is_finite()) {
            return Err(EncoderError::InvalidConfig(
                "velocity_norm_scale must be > 0".to_string(),
            ));
        }
        if !(self.elevation_gain > 0.0 && self.elevation_gain.is_finite()) {
            return Err(EncoderError::InvalidConfig("elevation_gain must be > 0".to_string()));
        }
        Ok(())
    }

    #[inline]
    pub fn max_range_px(&self) -> f32 {
        self.max_range_tiles * 32.0
    }
}

/// Channel -> AN type-name assignment (FLY.md §5's proprioception table, adapted per the module
/// doc comment: since real AN type names carry no function, this partition is arbitrary — see
/// `configs/fly/{S,M}-brain.toml`, where it is a deterministic round-robin over the graph's real
/// AN types, sorted by name).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProprioceptionConfig {
    #[serde(default)]
    pub grounded: Vec<String>,
    #[serde(default)]
    pub airborne: Vec<String>,
    #[serde(default)]
    pub own_hook: Vec<String>,
    #[serde(default)]
    pub jumps_left: Vec<String>,
    #[serde(default)]
    pub freeze_timer: Vec<String>,
    #[serde(default)]
    pub speed: Vec<String>,
}

impl ProprioceptionConfig {
    fn types_for(&self, channel: ProprioceptionChannel) -> &[String] {
        match channel {
            ProprioceptionChannel::Grounded => &self.grounded,
            ProprioceptionChannel::Airborne => &self.airborne,
            ProprioceptionChannel::OwnHook => &self.own_hook,
            ProprioceptionChannel::JumpsLeft => &self.jumps_left,
            ProprioceptionChannel::FreezeTimer => &self.freeze_timer,
            ProprioceptionChannel::Speed => &self.speed,
        }
    }
}

// --- Ring angle helpers --------------------------------------------------------------------------

/// `atan2(-dy, dx)` — see the module doc comment's "ring angle convention". Screen `y` grows
/// downward, so negating it before the `atan2` is what makes `π/2` mean "up".
#[inline]
pub fn ring_angle_from_screen_delta(dx: f32, dy: f32) -> f32 {
    (-dy).atan2(dx)
}

/// A `.flyg` receptive field's `(azimuth_deg, elevation_deg)` folded into one ring angle — see the
/// module doc comment for the exact composition and why it was chosen. `elevation_gain`:
/// [`RayGridConfig::elevation_gain`] (review round 1, F9) — `1.0` reproduces the original,
/// ungained fold.
fn ring_angle_from_rf(azimuth_deg: f32, elevation_deg: f32, elevation_gain: f32) -> f32 {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    // Review round 1, F1 (CONFIRMED, blocker): elevation was inverted. Positive elevation is
    // dorsal/up (`docs/formats.md` §8, FLY.md §5), and "up" in this module's ring convention is
    // *negative* screen `dy` (screen `y` grows downward) — so the screen-`dy` argument fed to
    // `ring_angle_from_screen_delta` must be `-el.sin()`, not `el.sin()`. The bug was itself
    // mirror-symmetric (negating azimuth alone still maps `θ -> π - θ` either way), which is
    // exactly why `tests::mirroring_azimuth_sign_gives_pi_minus_theta` never caught it — see
    // `tests::opponent_above_drives_dorsal_cells_more_than_ventral_ones` for the test that does.
    //
    // Review round 1, F9 (CONFIRMED, minor): `elevation_gain` scales `el.sin()` before the
    // `atan2` — see [`RayGridConfig::elevation_gain`]'s own doc comment for why and the real-S
    // numbers behind its default. Exactly mirror-symmetric for any gain: this term is untouched
    // by `azimuth -> -azimuth`, so it cannot affect the `θ -> π - θ` argument, which depends only
    // on `atan2`'s `x`-argument (`az.sin()`) changing sign.
    ring_angle_from_screen_delta(az.sin(), -elevation_gain * el.sin())
}

/// The ring angle of ray direction `d` (`0..num_directions`).
#[inline]
fn ray_angle(d: usize, num_directions: usize) -> f32 {
    TAU * (d as f32) / (num_directions as f32)
}

/// Signed angular distance `a - b`, wrapped to `(-π, π]` (never used to pick a sign convention –
/// only ever squared or fed to a symmetric Gaussian – so which of the two representable forms at
/// the exact `π` boundary is returned does not matter here).
#[inline]
fn angular_diff(a: f32, b: f32) -> f32 {
    let d = a - b;
    d.sin().atan2(d.cos())
}

/// `G(θ, φ) = exp(-Δ(θ,φ)² / (2σ²))`, `σ` in radians.
#[inline]
fn angular_gaussian(theta: f32, phi: f32, sigma_rad: f32) -> f32 {
    let d = angular_diff(theta, phi);
    (-(d * d) / (2.0 * sigma_rad * sigma_rad)).exp()
}

/// The bin index `b`'s Gaussian weight at normalized distance `distance_norm` (`0` = at the tee,
/// `1` = `max_range`) — "Gaussian bins", acceptance criterion 2's feature encoding choice: a
/// smooth handoff between adjacent bins instead of one-hot hard binning, with `σ` set to half a
/// bin's width so a hit exactly on a bin boundary splits close to evenly between its two
/// neighbors.
#[inline]
fn distance_bin_weight(distance_norm: f32, bin: usize, num_bins: usize) -> f32 {
    let center = (bin as f32 + 0.5) / num_bins as f32;
    let sigma = 0.5 / num_bins as f32;
    let d = distance_norm - center;
    (-(d * d) / (2.0 * sigma * sigma)).exp()
}

/// One axis's Amanatides & Woo (1987) grid-traversal setup for [`RayGridFeatures::cast_ray`]'s
/// DDA (review round 1, F7): given the ray's origin coordinate on this axis, its direction
/// component, the tile index the origin already sits in, and the tile size, returns `(step,
/// t_delta, t_max)` — `step` is `-1`/`0`/`+1` (which way the tile index moves along this axis),
/// `t_delta` is the parametric distance (pixels along the *whole* ray) to cross one tile on this
/// axis, and `t_max` is the distance to the next grid line the ray will cross on this axis. A
/// zero direction component gives `step = 0` and both distances `+inf`, so this axis never
/// "wins" the `t_max_x < t_max_y` comparison in the traversal loop.
#[inline]
fn axis_traversal(origin: f32, d: f32, tile_index: i64, tile_size: f32) -> (i64, f32, f32) {
    if d > 0.0 {
        let next_boundary = (tile_index + 1) as f32 * tile_size;
        (1, tile_size / d, (next_boundary - origin) / d)
    } else if d < 0.0 {
        let next_boundary = tile_index as f32 * tile_size;
        (-1, tile_size / -d, (next_boundary - origin) / d)
    } else {
        (0, f32::INFINITY, f32::INFINITY)
    }
}

// --- Features -------------------------------------------------------------------------------------

/// Ray-grid + own-motion features for one decision (task spec: "features in `[0, 1]`" for every
/// spatial/geometric channel; own-velocity channels are signed, `[-1, 1]` — see
/// [`RayGridConfig::velocity_norm_scale`]'s doc comment for why). Allocated once
/// ([`RayGridFeatures::new`]) and refilled in place by [`RayGridFeatures::compute`] every
/// decision — no per-decision allocation (acceptance criterion 2).
#[derive(Debug, Clone)]
pub struct RayGridFeatures {
    num_directions: usize,
    num_bins: usize,
    /// One `num_directions * num_bins` grid per spatial channel, indexed `[Channel as
    /// usize][d * num_bins + b]` via [`RayGridFeatures::spatial`] — `SPATIAL_CHANNELS.len()` grids.
    grids: Vec<Vec<f32>>,
    /// One scalar per own-motion channel, indexed via [`RayGridFeatures::scalar`].
    scalars: [f32; 3],
    /// One scalar per [`OpponentChannel`] (task 8.5a), indexed via [`RayGridFeatures::opponent`].
    opponent: [f32; OPPONENT_CHANNELS.len()],
}

/// The spatial channels, in the fixed order [`RayGridFeatures::grids`] stores them.
const SPATIAL_CHANNELS: [Channel; 7] = [
    Channel::OpponentPosition,
    Channel::OpponentApproach,
    Channel::OpponentHook,
    Channel::OtherPlayers,
    Channel::Walls,
    Channel::FreezeDeathTiles,
    Channel::NoHookTiles,
];

fn spatial_slot(channel: Channel) -> usize {
    SPATIAL_CHANNELS
        .iter()
        .position(|&c| c == channel)
        .expect("spatial_slot called with a non-spatial channel")
}

fn scalar_slot(channel: Channel) -> usize {
    match channel {
        Channel::SelfVelocityX => 0,
        Channel::SelfVelocityY => 1,
        Channel::SelfVelocityFlow => 2,
        _ => panic!("scalar_slot called with a non-scalar channel"),
    }
}

impl RayGridFeatures {
    pub fn new(cfg: &RayGridConfig) -> Self {
        RayGridFeatures {
            num_directions: cfg.num_directions,
            num_bins: cfg.num_distance_bins,
            grids: (0..SPATIAL_CHANNELS.len())
                .map(|_| vec![0.0f32; cfg.num_directions * cfg.num_distance_bins])
                .collect(),
            scalars: [0.0; 3],
            opponent: [0.0; OPPONENT_CHANNELS.len()],
        }
    }

    /// A spatial channel's full `num_directions * num_bins` grid, flattened direction-major: ray
    /// direction `d`'s `num_bins` values sit at `[d * num_bins .. d * num_bins + num_bins]`
    /// (review round 1, F13: this exact layout, needed by any caller reading
    /// [`crate::brain::FlyBrain::last_ray_features`] directly, was previously only implicit in
    /// this doc comment's own indexing example rather than stated outright).
    pub fn spatial(&self, channel: Channel) -> &[f32] {
        &self.grids[spatial_slot(channel)]
    }

    pub fn scalar(&self, channel: Channel) -> f32 {
        self.scalars[scalar_slot(channel)]
    }

    /// The target opponent's state channel (all zero when there is no opponent in the observation).
    pub fn opponent(&self, channel: OpponentChannel) -> f32 {
        self.opponent[channel.slot()]
    }

    pub fn num_directions(&self) -> usize {
        self.num_directions
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn spatial_mut(&mut self, channel: Channel) -> &mut [f32] {
        &mut self.grids[spatial_slot(channel)]
    }

    fn clear(&mut self) {
        for g in &mut self.grids {
            g.fill(0.0);
        }
        self.scalars = [0.0; 3];
        self.opponent = [0.0; OPPONENT_CHANNELS.len()];
    }

    /// Places a Gaussian-angle x Gaussian-bin bump for one point object (an opponent, another
    /// player, an opponent's hook) at `bearing`/`distance_px`, max-combined into whatever is
    /// already in that channel's grid (so multiple objects in `other_players` don't need to be
    /// renormalized to stay in `[0, 1]`). No-op if `distance_px` is beyond `max_range`.
    fn place_point_feature(
        &mut self,
        channel: Channel,
        bearing: f32,
        distance_px: f32,
        amplitude: f32,
        cfg: &RayGridConfig,
    ) {
        let max_range = cfg.max_range_px();
        if !(distance_px.is_finite() && distance_px <= max_range) {
            return;
        }
        let distance_norm = (distance_px / max_range).clamp(0.0, 1.0);
        let sigma_rad = cfg.object_sigma_deg.to_radians();
        let num_bins = self.num_bins;
        let num_directions = self.num_directions;
        let grid = self.spatial_mut(channel);
        for d in 0..num_directions {
            let angular_w = angular_gaussian(ray_angle(d, num_directions), bearing, sigma_rad);
            if angular_w < 1e-6 {
                continue;
            }
            for b in 0..num_bins {
                let v = amplitude * angular_w * distance_bin_weight(distance_norm, b, num_bins);
                let idx = d * num_bins + b;
                grid[idx] = grid[idx].max(v);
            }
        }
    }

    /// Casts one ray via tile-grid DDA (task spec: "exact on the tile grid"), writing `walls`/
    /// `no_hook_tiles`/`freeze_death_tiles` bins from what it finds. Occlusion: a solid tile stops
    /// the ray (nothing behind a wall contributes); a hazard tile (freeze/deep-freeze/death, game
    /// or front layer) does not stop it — only the *nearest* hazard along the ray is recorded,
    /// matching "no-hook incl. front layer, deep" (FLY.md §5).
    ///
    /// Real Amanatides-Woo tile-grid DDA (review round 1, F7, CONFIRMED — an earlier revision
    /// marched in fixed 32px steps along the ray, which the reviewer's own brute-force reference
    /// found let 0.19% of rays see straight through a wall and 0.23% overshoot by more than a
    /// full tile on the real Copy Love Box map). This visits the *exact* sequence of tiles the
    /// ray crosses, in order, entering each one at its exact continuous parametric distance
    /// (`t`, in tiles) — no tile is ever skipped regardless of ray angle, and the reported hit
    /// distance is the true crossing distance, not a step-quantized multiple of the step size.
    fn cast_ray(&mut self, d: usize, origin: (f32, f32), map: &ddai_physics::map::MapData, cfg: &RayGridConfig) {
        let theta = ray_angle(d, self.num_directions);
        let (ddx, ddy) = (theta.cos(), -theta.sin());
        let max_range_tiles = cfg.max_range_tiles;
        let num_bins = self.num_bins;
        const TILE: f32 = 32.0;

        let mut hazard_dist: Option<f32> = None;
        let mut wall_dist: Option<f32> = None;
        let mut wall_is_no_hook = false;

        let mut tx = (origin.0 / TILE).floor() as i64;
        let mut ty = (origin.1 / TILE).floor() as i64;

        // Standard Amanatides & Woo (1987) setup: `step_*` is which way the tile index moves,
        // `t_delta_*` is the parametric distance (in pixels along the ray) to cross one whole
        // tile on that axis, `t_max_*` is the distance to the *next* grid line on that axis.
        let (step_x, t_delta_x, mut t_max_x) = axis_traversal(origin.0, ddx, tx, TILE);
        let (step_y, t_delta_y, mut t_max_y) = axis_traversal(origin.1, ddy, ty, TILE);

        // Hard iteration cap (never an infinite loop even for a degenerate direction): at most a
        // few tiles more than the number a ray could possibly cross within `max_range_tiles`
        // (a straight axis-aligned ray crosses exactly that many; a diagonal one crosses close to
        // double that, one extra step per axis at each grid crossing near the diagonal).
        let max_iters = (max_range_tiles.ceil() as i64) * 2 + 4;

        let mut t_px = 0.0f32; // current parametric distance along the ray, in pixels.
        for _ in 0..max_iters {
            let class = tile_class(map, (tx as f32 + 0.5) * TILE, (ty as f32 + 0.5) * TILE);
            let t_tiles = t_px / TILE;
            if hazard_dist.is_none() && class.is_hazard() {
                hazard_dist = Some(t_tiles);
            }
            if class.is_solid() {
                wall_dist = Some(t_tiles);
                wall_is_no_hook = class.is_no_hook();
                break;
            }
            if t_tiles > max_range_tiles {
                break;
            }
            if t_max_x < t_max_y {
                t_px = t_max_x;
                tx += step_x;
                t_max_x += t_delta_x;
            } else {
                t_px = t_max_y;
                ty += step_y;
                t_max_y += t_delta_y;
            }
        }

        if let Some(t) = wall_dist {
            let norm = (t / max_range_tiles).clamp(0.0, 1.0);
            let grid = self.spatial_mut(Channel::Walls);
            for b in 0..num_bins {
                grid[d * num_bins + b] = distance_bin_weight(norm, b, num_bins);
            }
            if wall_is_no_hook {
                let grid = self.spatial_mut(Channel::NoHookTiles);
                for b in 0..num_bins {
                    grid[d * num_bins + b] = distance_bin_weight(norm, b, num_bins);
                }
            }
        }
        if let Some(t) = hazard_dist {
            let norm = (t / max_range_tiles).clamp(0.0, 1.0);
            let grid = self.spatial_mut(Channel::FreezeDeathTiles);
            for b in 0..num_bins {
                grid[d * num_bins + b] = distance_bin_weight(norm, b, num_bins);
            }
        }
    }

    /// Refills every channel from `obs` (allocation-free: reuses this struct's own buffers).
    /// Which of `obs.others` drives `opponent_position`/`opponent_approach`/`opponent_hook` is
    /// [`Observation::target_or_nearest`] (review round 1, F13, CONFIRMED: an earlier revision
    /// hardcoded `obs.others.first()` — always index `0` — while this same doc comment already
    /// named a `primary_opponent_id` that didn't actually exist as a parameter anywhere, and
    /// `Observation::others`'s own doc comment says its order carries no meaning; a producer that
    /// returned `others` in, say, distance order would then have silently fed a *different*
    /// opponent into every one of these three channels depending on happenstance ordering).
    /// `None` if `obs.others` is empty. Every *other* character in `obs.others` (i.e. everyone
    /// except whichever one was selected as the target) contributes to `other_players` instead.
    pub fn compute(&mut self, obs: &Observation, cfg: &RayGridConfig) {
        self.clear();
        let origin = (obs.self_state.pos.x, obs.self_state.pos.y);

        for d in 0..self.num_directions {
            self.cast_ray(d, origin, &obs.map, cfg);
        }

        let target = obs.target_or_nearest();
        if let Some(opp) = target {
            self.place_opponent_features(&obs.self_state, opp, cfg);
            // Deep freeze does not tick down: the live world reports it with no freeze time at all, the arena's physics re-freezes the
            // tee every tick. Either way the opponent is out and stays out, so it reads as frozen with a full timer. Live freeze is
            // **not** the same: that tee can still hook and is not "out" (`observe::is_out` is `freeze_time > 0`), so it reads as before.
            let held_frozen = opp.is_deep_frozen;
            self.opponent[OpponentChannel::Frozen.slot()] =
                f32::from(opp.is_frozen || held_frozen || opp.freeze_ticks_remaining > 0);
            self.opponent[OpponentChannel::FreezeLeft.slot()] = if held_frozen {
                1.0
            } else {
                (opp.freeze_ticks_remaining as f32 / FREEZE_TIMER_NORMALIZATION_TICKS).clamp(0.0, 1.0)
            };
            self.opponent[OpponentChannel::VelocityX.slot()] = (opp.vel.x / cfg.velocity_norm_scale).clamp(-1.0, 1.0);
            self.opponent[OpponentChannel::VelocityY.slot()] = (opp.vel.y / cfg.velocity_norm_scale).clamp(-1.0, 1.0);
            self.opponent[OpponentChannel::HookState.slot()] = match opp.hook_state {
                HOOK_FLYING => 0.5,
                HOOK_GRABBED => 1.0,
                _ => 0.0,
            };
        }
        let target_id = target.map(|opp| opp.id);
        for other in obs.others.iter().filter(|o| Some(o.id) != target_id) {
            let dx = other.pos.x - obs.self_state.pos.x;
            let dy = other.pos.y - obs.self_state.pos.y;
            let dist = (dx * dx + dy * dy).sqrt();
            let bearing = ring_angle_from_screen_delta(dx, dy);
            self.place_point_feature(Channel::OtherPlayers, bearing, dist, 1.0, cfg);
        }

        let vx = obs.self_state.vel.x / cfg.velocity_norm_scale;
        let vy = obs.self_state.vel.y / cfg.velocity_norm_scale;
        let speed = (obs.self_state.vel.x.powi(2) + obs.self_state.vel.y.powi(2)).sqrt() / cfg.velocity_norm_scale;
        self.scalars[scalar_slot(Channel::SelfVelocityX)] = vx.clamp(-1.0, 1.0);
        self.scalars[scalar_slot(Channel::SelfVelocityY)] = vy.clamp(-1.0, 1.0);
        self.scalars[scalar_slot(Channel::SelfVelocityFlow)] = speed.clamp(0.0, 1.0);
    }

    fn place_opponent_features(&mut self, me: &CharacterObservation, opp: &CharacterObservation, cfg: &RayGridConfig) {
        let dx = opp.pos.x - me.pos.x;
        let dy = opp.pos.y - me.pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let bearing = ring_angle_from_screen_delta(dx, dy);
        self.place_point_feature(Channel::OpponentPosition, bearing, dist, 1.0, cfg);

        let looming = compute_looming_feature(me, opp, dx, dy, dist, cfg.looming_k);
        self.place_point_feature(Channel::OpponentApproach, bearing, dist, looming, cfg);

        if matches!(opp.hook_state, HOOK_FLYING | HOOK_GRABBED) {
            let hdx = opp.hook_pos.x - me.pos.x;
            let hdy = opp.hook_pos.y - me.pos.y;
            let hdist = (hdx * hdx + hdy * hdy).sqrt();
            let hbearing = ring_angle_from_screen_delta(hdx, hdy);
            self.place_point_feature(Channel::OpponentHook, hbearing, hdist, 1.0, cfg);
        }
    }
}

/// Character physical radius (half of `ddai_physics::core::physical_size()`) used by the looming
/// feature's `2rv/d²` formula (FLY.md §5). A plain constant (not read from `tuning`, which has no
/// such field): DDNet's tee hitbox size is fixed by the game, not tunable.
const CHARACTER_RADIUS_PX: f32 = 14.0;

/// `opponent_approach` (FLY.md §5, label **Б**): "угловой рост ≈ 2rv/d²", `v` the *closing* speed
/// (positive when the gap is shrinking). Saturated into `[0, 1)` via
/// [`RayGridConfig::looming_k`] rather than left as a raw, unbounded rate — a receding or
/// stationary opponent gives exactly `0.0` (not a negative feature: this channel represents
/// "how fast is it looming", which has no meaning below zero).
fn compute_looming_feature(
    me: &CharacterObservation,
    opp: &CharacterObservation,
    dx: f32,
    dy: f32,
    dist: f32,
    looming_k: f32,
) -> f32 {
    if dist < 1e-3 {
        return 0.0;
    }
    let (rx, ry) = (dx / dist, dy / dist); // unit vector self -> opponent
    let rel_vx = opp.vel.x - me.vel.x;
    let rel_vy = opp.vel.y - me.vel.y;
    let radial_speed = rel_vx * rx + rel_vy * ry; // > 0 => separating
    let closing_speed = (-radial_speed).max(0.0);
    let raw = 2.0 * CHARACTER_RADIUS_PX * closing_speed / (dist * dist);
    raw / (raw + looming_k)
}

// --- Tile classification (only what the encoder needs: solid/no-hook/hazard by index) ------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TileClass {
    solid: bool,
    no_hook: bool,
    hazard: bool,
}

impl TileClass {
    fn is_solid(self) -> bool {
        self.solid
    }
    fn is_no_hook(self) -> bool {
        self.no_hook
    }
    fn is_hazard(self) -> bool {
        self.hazard
    }
}

/// `CCollision::round_to_int` (`base/math.h`): `f > 0 ? (int)(f + 0.5) : (int)(f - 0.5)` — the
/// exact rounding rule DDNet's own `CheckPoint`/`GetCollisionAt` apply to a continuous position
/// before dividing by the tile size (review round 1, F18, CONFIRMED: an earlier revision of
/// [`tile_class`] truncated `x/32.0` directly instead, which is `GetTile(int, int)`'s own
/// behavior for an *already-integer* pixel coordinate, not `GetCollisionAt(float, float)`'s —
/// the two disagree for roughly half of every tile's width, wherever rounding would push into
/// the *next* cell but truncation would not). A local copy of the formula (not
/// `ddai_physics::vmath::round_to_int`, which is generic over `Real` for both `f32`/`f64` parity
/// work this module has no need of) rather than a dependency on `ddai_physics::vmath` for one
/// three-line function.
#[inline]
pub(crate) fn round_to_int_f32(f: f32) -> i32 {
    if f > 0.0 { (f + 0.5) as i32 } else { (f - 0.5) as i32 }
}

/// Classifies the tile at pixel `(x, y)` for the ray-casting channels — deliberately a small,
/// standalone reimplementation of `ddai_physics::collision::Collision`'s tile lookup (same
/// round-then-divide-then-clamp convention as `CCollision::GetCollisionAt`/`CheckPoint`, see
/// [`round_to_int_f32`]), not a dependency on `Collision` itself: the encoder only ever needs "is
/// this tile solid/no-hook/a hazard", and `Collision` would pull in tele/switch hashmap
/// bookkeeping (and the `Real` generic) this module has no use for. Front-layer freeze/death
/// counts as a hazard too (FLY.md §5: "incl. front layer, deep").
fn tile_class(map: &ddai_physics::map::MapData, x: f32, y: f32) -> TileClass {
    use ddai_physics::map::{TILE_DEATH, TILE_DFREEZE, TILE_FREEZE, TILE_NOHOOK, TILE_SOLID};

    if map.width == 0 || map.height == 0 {
        return TileClass {
            solid: false,
            no_hook: false,
            hazard: false,
        };
    }
    let nx = ((round_to_int_f32(x) / 32) as i64).clamp(0, map.width as i64 - 1) as usize;
    let ny = ((round_to_int_f32(y) / 32) as i64).clamp(0, map.height as i64 - 1) as usize;
    let idx = ny * map.width as usize + nx;

    let game_index = map.game.get(idx).map(|t| t.index).unwrap_or(0);
    let front_index = map
        .front
        .as_ref()
        .and_then(|f| f.get(idx))
        .map(|t| t.index)
        .unwrap_or(0);

    let is_hazard = |i: u8| i == TILE_FREEZE || i == TILE_DFREEZE || i == TILE_DEATH;

    TileClass {
        solid: game_index == TILE_SOLID || game_index == TILE_NOHOOK,
        no_hook: game_index == TILE_NOHOOK,
        hazard: is_hazard(game_index) || is_hazard(front_index),
    }
}

// --- Params / model -------------------------------------------------------------------------------

/// The encoder's only learnable state: `g`/`c` per `(type, channel)` assignment, in
/// [`EncoderModel`]'s own dense `param_id` order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderParams {
    pub g: Vec<f32>,
    pub c: Vec<f32>,
    /// Per-distance-bin gain of each `(type, channel)` assignment, `param * num_bins + bin`
    /// (task 8.2, [`RayGridConfig::learn_distance_gains`]); it multiplies the ray-weighted sum of
    /// that bin. **Empty** = every bin has gain `1` (the 7.3 encoder, bit for bit).
    pub bin_gain: Vec<f32>,
}

impl EncoderParams {
    /// `g = 1.0`, `c = 0.0` for every assignment — a deterministic, seed-free default (there is no
    /// connectome-derived quantity to draw these from, unlike `FlyParams`'s type-pair scales).
    pub fn init_default(num_params: usize) -> Self {
        EncoderParams {
            g: vec![1.0; num_params],
            c: vec![0.0; num_params],
            bin_gain: Vec::new(),
        }
    }

    /// `num_bins` is the ray grid's distance-bin count: `bin_gain` is empty (no learned gains) or
    /// has exactly `num_params * num_bins` entries (a count that is merely a multiple would index
    /// the wrong bin of the wrong assignment, or panic on a slice).
    pub fn validate_shape(&self, num_params: usize, num_bins: usize) -> Result<(), EncoderError> {
        if !self.bin_gain.is_empty() && self.bin_gain.len() != num_params * num_bins {
            return Err(EncoderError::ParamShapeMismatch(format!(
                "bin_gain.len()={} but {num_params} assignments x {num_bins} bins = {}",
                self.bin_gain.len(),
                num_params * num_bins
            )));
        }
        if self.bin_gain.iter().any(|x| !x.is_finite()) {
            return Err(EncoderError::ParamShapeMismatch("bin_gain must be finite".to_string()));
        }
        if self.g.len() != num_params || self.c.len() != num_params {
            return Err(EncoderError::ParamShapeMismatch(format!(
                "g.len()={}, c.len()={}, expected {num_params}",
                self.g.len(),
                self.c.len()
            )));
        }
        if self.g.iter().chain(&self.c).any(|x| !x.is_finite()) {
            return Err(EncoderError::ParamShapeMismatch(
                "g/c must not contain NaN or infinity".to_string(),
            ));
        }
        Ok(())
    }
}

/// `dL/dg`, `dL/dc`, same shape/order as [`EncoderParams`].
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderGradients {
    pub g: Vec<f32>,
    pub c: Vec<f32>,
    /// Same layout as [`EncoderParams::bin_gain`]; empty when that is empty.
    pub bin_gain: Vec<f32>,
}

impl EncoderGradients {
    pub fn zeros(num_params: usize) -> Self {
        EncoderGradients {
            g: vec![0.0; num_params],
            c: vec![0.0; num_params],
            bin_gain: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum EncoderError {
    InvalidConfig(String),
    ParamShapeMismatch(String),
    /// A [`ProprioceptionConfig`] entry names a type that either doesn't exist in this `.flyg` or
    /// isn't `InputAscending` (acceptance criterion "validate all index inputs").
    UnknownAnType(String),
    /// A `.flyg` `input_channels` entry names a channel string this module doesn't recognize
    /// (acceptance criterion "validate all index inputs" — a typo in `configs/fly/*.toml` must
    /// fail loudly at load time, not silently drop the channel).
    UnknownChannel(String),
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderError::InvalidConfig(m) => write!(f, "invalid encoder config: {m}"),
            EncoderError::ParamShapeMismatch(m) => write!(f, "encoder params shape mismatch: {m}"),
            EncoderError::UnknownAnType(name) => write!(f, "proprioception config names unknown/non-AN type '{name}'"),
            EncoderError::UnknownChannel(name) => write!(f, "unknown input channel name '{name}'"),
        }
    }
}

impl std::error::Error for EncoderError {}

/// Largest number of distance bins the per-bin gains support.
const MAX_BINS: usize = 8;

/// One term of one input neuron's current formula — a `(channel, param_id)` pair, plus (for a
/// spatial VPN channel) the neuron's precomputed per-ray-direction Gaussian weight, shared across
/// every spatial channel that neuron's type carries (the weight depends only on the neuron's own
/// receptive field, never on which channel is being read).
#[derive(Debug, Clone)]
struct Term {
    channel: InputChannel,
    param_id: u32,
}

#[derive(Debug, Clone, Default)]
struct InputNeuronInfo {
    /// `Some` (len `num_directions`) for a VPN neuron with at least one spatial channel term;
    /// `None` for an AN neuron or a VPN neuron with only scalar channel terms.
    ray_weight: Option<Vec<f32>>,
    /// `sin(azimuth)`/`sin(elevation)` — the direction factors a `SelfVelocityX`/`SelfVelocityY`
    /// term (if this neuron has one) is multiplied by; see the module doc comment's "Own-motion
    /// channels" note. `0.0` (inert) when this neuron carries neither term.
    vx_dir_factor: f32,
    vy_dir_factor: f32,
    terms: Vec<Term>,
}

/// One (type, channel) assignment and the report data a caller (`ddnet-ai fly brain-demo`, this
/// module's own tests) wants about it.
#[derive(Debug, Clone)]
pub struct ChannelAssignmentInfo {
    pub type_name: String,
    pub type_index: u32,
    pub channel: &'static str,
    pub param_id: u32,
}

/// Built once from a [`FlyModel`]'s graph plus config (mirrors [`FlyModel`]'s own "build once,
/// query many times" shape): every input neuron's precomputed receptive-field weights/direction
/// factors and channel terms, plus the flat `(type, channel) -> param_id` table
/// [`EncoderParams`]/[`EncoderGradients`] are indexed by.
#[derive(Debug, Clone)]
pub struct EncoderModel {
    ray_cfg: RayGridConfig,
    /// Parallel to `model.input_neuron_indices()` — `per_input[k]` is input slot `k`'s info, the
    /// same order [`crate::state::FlyState::step_decision`]'s `inputs` slice uses.
    per_input: Vec<InputNeuronInfo>,
    assignments: Vec<ChannelAssignmentInfo>,
}

impl EncoderModel {
    /// Validates `ray_cfg` and every `proprio_cfg` type name, precomputes every input neuron's
    /// receptive-field data, and builds the flat parameter table. Returns
    /// [`EncoderError::UnknownAnType`]/[`EncoderError::UnknownChannel`] for a config that names a
    /// type/channel this graph/module doesn't have — never silently ignored (acceptance
    /// criterion: "validate all index inputs").
    pub fn new(
        model: &FlyModel,
        ray_cfg: RayGridConfig,
        proprio_cfg: &ProprioceptionConfig,
    ) -> Result<Self, EncoderError> {
        Self::with_opponent_state(model, ray_cfg, proprio_cfg, &OpponentStateConfig::default())
    }

    /// [`EncoderModel::new`] plus the target-opponent state channels of `opp_cfg` (task 8.5a). Their `(type, channel)`
    /// parameters are numbered **after** every parameter [`EncoderModel::new`] would have made, so the first
    /// `new(..).num_params()` entries of `g`/`c`/`bin_gain` keep their meaning and a bundle can be upgraded by
    /// appending to them (see [`crate::bundle::upgrade_with_opponent_state`]).
    pub fn with_opponent_state(
        model: &FlyModel,
        ray_cfg: RayGridConfig,
        proprio_cfg: &ProprioceptionConfig,
        opp_cfg: &OpponentStateConfig,
    ) -> Result<Self, EncoderError> {
        ray_cfg.validate()?;
        let flyg = model.flyg();

        // `(type_index, channel) -> param_id`, first-seen order (deterministic: `flyg.
        // input_channels` and `proprio_cfg`'s fields are both already in a fixed, non-hashmap
        // order).
        let mut assignments: Vec<ChannelAssignmentInfo> = Vec::new();
        let lookup = |type_index: u32, channel: InputChannel, assignments: &mut Vec<ChannelAssignmentInfo>| -> u32 {
            if let Some(pos) = assignments
                .iter()
                .position(|a| a.type_index == type_index && a.channel == channel.name())
            {
                return pos as u32;
            }
            let id = assignments.len() as u32;
            assignments.push(ChannelAssignmentInfo {
                type_name: flyg.types[type_index as usize].name.clone(),
                type_index,
                channel: channel.name(),
                param_id: id,
            });
            id
        };

        // Every input neuron's role, indexed by type (used below to validate both
        // `flyg.input_channels` and `proprio_cfg` against the role the `.flyg` format actually
        // requires for each — acceptance criterion "validate all index inputs").
        let mut role_of_type: Vec<Option<NeuronRole>> = vec![None; flyg.types.len()];
        for n in &flyg.neurons {
            let slot = &mut role_of_type[n.type_index as usize];
            if slot.is_none() {
                *slot = Some(n.role);
            }
        }

        // `flyg.input_channels` (task 6.3's output) only ever names `InputVisual` types in
        // practice (`docs/formats.md` §8: AN types carry no functional channel mapping at all),
        // but the format's own struct doc comment allows `InputAscending` there too — reject that
        // rather than silently ignoring it later (this module's per-neuron loop below only reads
        // `flyg.input_channels` for `InputVisual` neurons, so a future `.flyg` that *did* use it
        // for an AN type would otherwise have that channel term quietly vanish, not error).
        for mapping in &flyg.input_channels {
            if role_of_type[mapping.type_index as usize] != Some(NeuronRole::InputVisual) {
                return Err(EncoderError::UnknownChannel(format!(
                    "input_channels entry for type_index {} is not InputVisual (unsupported here)",
                    mapping.type_index
                )));
            }
            for name in &mapping.channels {
                Channel::parse(name).ok_or_else(|| EncoderError::UnknownChannel(name.clone()))?;
            }
        }

        // Proprioceptive (AN) channel assignments come from `proprio_cfg` — validate every named
        // type actually exists and is `InputAscending` before trusting it.
        let mut an_type_index_of = std::collections::HashMap::new();
        for (i, ty) in flyg.types.iter().enumerate() {
            an_type_index_of.insert(ty.name.as_str(), i as u32);
        }
        let mut proprio_by_type: std::collections::HashMap<u32, ProprioceptionChannel> =
            std::collections::HashMap::new();
        for channel in PROPRIOCEPTION_CHANNELS {
            for name in proprio_cfg.types_for(channel) {
                let type_index = *an_type_index_of
                    .get(name.as_str())
                    .ok_or_else(|| EncoderError::UnknownAnType(name.clone()))?;
                if role_of_type[type_index as usize] != Some(NeuronRole::InputAscending) {
                    return Err(EncoderError::UnknownAnType(name.clone()));
                }
                if let Some(existing) = proprio_by_type.insert(type_index, channel)
                    && existing != channel
                {
                    return Err(EncoderError::UnknownAnType(format!(
                        "{name} is assigned to both '{}' and '{}' in ProprioceptionConfig",
                        existing.name(),
                        channel.name()
                    )));
                }
            }
        }

        // Per-input-neuron precompute, in `model.input_neuron_indices()`'s canonical order.
        let mut per_input = Vec::with_capacity(model.num_inputs());
        for &dense in model.input_neuron_indices() {
            let neuron = &flyg.neurons[dense as usize];
            let mut info = InputNeuronInfo::default();

            match neuron.role {
                NeuronRole::InputVisual => {
                    let rf = neuron
                        .rf
                        .expect("InputVisual neuron must have an rf per .flyg's own invariant");
                    let theta = ring_angle_from_rf(rf.azimuth_deg, rf.elevation_deg, ray_cfg.elevation_gain);
                    info.vx_dir_factor = rf.azimuth_deg.to_radians().sin();
                    info.vy_dir_factor = rf.elevation_deg.to_radians().sin();

                    let mut any_spatial = false;
                    if let Some(mapping) = flyg.input_channels.iter().find(|m| m.type_index == neuron.type_index) {
                        for name in &mapping.channels {
                            let ch = Channel::parse(name).expect("validated above");
                            let param_id = lookup(neuron.type_index, InputChannel::Visual(ch), &mut assignments);
                            info.terms.push(Term {
                                channel: InputChannel::Visual(ch),
                                param_id,
                            });
                            any_spatial |= ch.is_spatial();
                        }
                    }
                    if any_spatial {
                        let num_directions = ray_cfg.num_directions;
                        let sigma_rad = ray_cfg.rf_sigma_deg.to_radians();
                        info.ray_weight = Some(
                            (0..num_directions)
                                .map(|d| angular_gaussian(theta, ray_angle(d, num_directions), sigma_rad))
                                .collect(),
                        );
                    }
                }
                NeuronRole::InputAscending => {
                    if let Some(&channel) = proprio_by_type.get(&neuron.type_index) {
                        let param_id = lookup(neuron.type_index, InputChannel::Ascending(channel), &mut assignments);
                        info.terms.push(Term {
                            channel: InputChannel::Ascending(channel),
                            param_id,
                        });
                    }
                }
                NeuronRole::Hidden | NeuronRole::Output => {
                    unreachable!("model.input_neuron_indices() only yields InputVisual/InputAscending neurons")
                }
            }
            per_input.push(info);
        }

        // Target-opponent state channels: a second pass, so that their parameters come after all the others.
        let mut opp_by_type: Vec<(u32, OpponentChannel)> = Vec::new();
        for channel in OPPONENT_CHANNELS {
            for name in opp_cfg.types_for(channel) {
                let type_index = *an_type_index_of
                    .get(name.as_str())
                    .ok_or_else(|| EncoderError::UnknownAnType(name.clone()))?;
                let role = role_of_type[type_index as usize];
                let ok = match role {
                    Some(NeuronRole::InputVisual) => true,
                    Some(NeuronRole::InputAscending) => !channel.needs_visual(),
                    _ => false,
                };
                if !ok {
                    return Err(EncoderError::UnknownAnType(format!(
                        "{name} cannot carry '{}' (an input type is needed{})",
                        channel.name(),
                        if channel.needs_visual() {
                            ", a visual one for a velocity"
                        } else {
                            ""
                        }
                    )));
                }
                if opp_by_type.contains(&(type_index, channel)) {
                    return Err(EncoderError::UnknownAnType(format!(
                        "{name} is listed twice for '{}' in [opponent_state]",
                        channel.name()
                    )));
                }
                opp_by_type.push((type_index, channel));
            }
        }
        for (k, &dense) in model.input_neuron_indices().iter().enumerate() {
            let type_index = flyg.neurons[dense as usize].type_index;
            for &(_, channel) in opp_by_type.iter().filter(|(t, _)| *t == type_index) {
                let ch = InputChannel::Opponent(channel);
                let param_id = lookup(type_index, ch, &mut assignments);
                per_input[k].terms.push(Term { channel: ch, param_id });
            }
        }

        Ok(EncoderModel {
            ray_cfg,
            per_input,
            assignments,
        })
    }

    pub fn num_params(&self) -> usize {
        self.assignments.len()
    }

    pub fn ray_grid_config(&self) -> &RayGridConfig {
        &self.ray_cfg
    }

    pub fn assignments(&self) -> &[ChannelAssignmentInfo] {
        &self.assignments
    }

    /// Which of [`ALL_VISUAL_CHANNELS`] have zero VPN neurons feeding them on this graph
    /// (acceptance criterion 2's "report channels with no neurons").
    pub fn visual_channels_with_no_neurons(&self) -> Vec<&'static str> {
        ALL_VISUAL_CHANNELS
            .iter()
            .copied()
            .filter(|&name| !self.assignments.iter().any(|a| a.channel == name))
            .collect()
    }

    /// Fresh parameters for this model: unit gains, zero offsets, and (when
    /// [`RayGridConfig::learn_distance_gains`] is set) unit per-bin gains.
    pub fn init_params(&self) -> EncoderParams {
        let mut p = EncoderParams::init_default(self.num_params());
        if self.ray_cfg.learn_distance_gains {
            p.bin_gain = vec![1.0; self.num_params() * self.ray_cfg.num_distance_bins];
        }
        p
    }

    /// Zero gradients shaped like [`EncoderModel::init_params`].
    pub fn zero_grads(&self) -> EncoderGradients {
        let mut g = EncoderGradients::zeros(self.num_params());
        if self.ray_cfg.learn_distance_gains {
            g.bin_gain = vec![0.0; self.num_params() * self.ray_cfg.num_distance_bins];
        }
        g
    }

    /// Ray-weighted sum of each distance bin of a spatial channel, `sums[b] = sum_d w_d *
    /// feat[d, b]`; `sums[..num_bins]` is meaningful. The uniform-gain contribution is their sum.
    fn spatial_bin_sums(
        &self,
        info: &InputNeuronInfo,
        channel: Channel,
        features: &RayGridFeatures,
    ) -> [f32; MAX_BINS] {
        let w = info.ray_weight.as_deref().expect("spatial term implies ray_weight");
        let feat = features.spatial(channel);
        let nb = features.num_bins();
        assert!(nb <= MAX_BINS, "at most {MAX_BINS} distance bins are supported");
        let mut sums = [0.0f32; MAX_BINS];
        for (d, &wd) in w.iter().enumerate() {
            if wd == 0.0 {
                continue;
            }
            for b in 0..nb {
                sums[b] += wd * feat[d * nb + b];
            }
        }
        sums
    }

    /// The raw (pre-`g`/`c`) contribution of one **visual** channel term — never called for an
    /// [`InputChannel::Ascending`] term, which reads straight from `an_values` at the call site
    /// instead (there is no ray grid or direction factor for a proprioceptive channel).
    fn visual_term_contribution(&self, info: &InputNeuronInfo, channel: Channel, features: &RayGridFeatures) -> f32 {
        if channel.is_spatial() {
            let w = info.ray_weight.as_deref().expect("spatial term implies ray_weight");
            let feat = features.spatial(channel);
            let num_bins = features.num_bins();
            let mut acc = 0.0f32;
            for (d, &wd) in w.iter().enumerate() {
                if wd == 0.0 {
                    continue;
                }
                let base = d * num_bins;
                for b in 0..num_bins {
                    acc += wd * feat[base + b];
                }
            }
            return acc;
        }
        match channel {
            Channel::SelfVelocityX => features.scalar(Channel::SelfVelocityX) * info.vx_dir_factor,
            Channel::SelfVelocityY => features.scalar(Channel::SelfVelocityY) * info.vy_dir_factor,
            Channel::SelfVelocityFlow => features.scalar(Channel::SelfVelocityFlow),
            _ => unreachable!("is_spatial() is false only for the three arms above"),
        }
    }

    fn term_contribution(
        &self,
        info: &InputNeuronInfo,
        term: &Term,
        features: &RayGridFeatures,
        an_values: &ProprioceptionValues,
    ) -> f32 {
        match term.channel {
            InputChannel::Visual(ch) => self.visual_term_contribution(info, ch, features),
            InputChannel::Ascending(ch) => an_values.get(ch),
            InputChannel::Opponent(ch) => {
                let v = features.opponent(ch);
                match ch {
                    OpponentChannel::VelocityX => v * info.vx_dir_factor,
                    OpponentChannel::VelocityY => v * info.vy_dir_factor,
                    _ => v,
                }
            }
        }
    }

    /// Writes `out[k] = I_k` for every input neuron, in `model.input_neuron_indices()` order —
    /// ready to feed straight into [`crate::state::FlyState::step_decision`]. Allocation-free
    /// (acceptance criterion 2): `out.len()` must already equal [`EncoderModel::num_inputs`].
    /// `an_values` supplies each [`ProprioceptionChannel`]'s scalar for this decision (see
    /// [`compute_proprioception_values`]).
    pub fn forward(
        &self,
        features: &RayGridFeatures,
        an_values: &ProprioceptionValues,
        params: &EncoderParams,
        out: &mut [f32],
    ) {
        assert_eq!(
            out.len(),
            self.per_input.len(),
            "forward: out.len() must equal num_inputs"
        );
        for (k, info) in self.per_input.iter().enumerate() {
            let mut i_k = 0.0f32;
            for term in &info.terms {
                let g = params.g[term.param_id as usize];
                let c = params.c[term.param_id as usize];
                let contribution = match (&term.channel, params.bin_gain.is_empty()) {
                    (InputChannel::Visual(ch), false) if ch.is_spatial() => {
                        let nb = features.num_bins();
                        let gains = &params.bin_gain[term.param_id as usize * nb..(term.param_id as usize + 1) * nb];
                        let sums = self.spatial_bin_sums(info, *ch, features);
                        gains.iter().zip(&sums[..nb]).map(|(&gb, &sb)| gb * sb).sum()
                    }
                    _ => self.term_contribution(info, term, features, an_values),
                };
                i_k += g * contribution + c;
            }
            out[k] = i_k;
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.per_input.len()
    }

    /// Accumulates `dL/dg`/`dL/dc` into `grads` from `grad_input_currents[k] = dL/dI_k` (exactly
    /// [`crate::backward::BpttGradients::grad_inputs`]'s per-decision entry, in the same
    /// `model.input_neuron_indices()` order this whole module already uses throughout).
    pub fn backward(
        &self,
        features: &RayGridFeatures,
        an_values: &ProprioceptionValues,
        grad_input_currents: &[f32],
        grads: &mut EncoderGradients,
    ) {
        let unit = EncoderParams::init_default(self.num_params());
        self.backward_with_params(features, an_values, &unit, grad_input_currents, grads);
    }

    /// [`EncoderModel::backward`] for parameters that may carry per-bin gains: `params` are the
    /// values the forward pass used (only its `g` and `bin_gain` matter here, and only when
    /// `bin_gain` is non-empty; with it empty this is exactly [`EncoderModel::backward`]).
    pub fn backward_with_params(
        &self,
        features: &RayGridFeatures,
        an_values: &ProprioceptionValues,
        params: &EncoderParams,
        grad_input_currents: &[f32],
        grads: &mut EncoderGradients,
    ) {
        assert_eq!(
            grad_input_currents.len(),
            self.per_input.len(),
            "backward: grad_input_currents.len() must equal num_inputs"
        );
        for (k, info) in self.per_input.iter().enumerate() {
            let gi = grad_input_currents[k];
            if gi == 0.0 {
                continue;
            }
            for term in &info.terms {
                let pid = term.param_id as usize;
                if let (InputChannel::Visual(ch), false) = (&term.channel, params.bin_gain.is_empty())
                    && ch.is_spatial()
                {
                    let nb = features.num_bins();
                    let sums = self.spatial_bin_sums(info, *ch, features);
                    let gains = &params.bin_gain[pid * nb..(pid + 1) * nb];
                    let g = params.g[pid];
                    let contribution: f32 = gains.iter().zip(&sums[..nb]).map(|(&gb, &sb)| gb * sb).sum();
                    grads.g[pid] += gi * contribution;
                    for (b, &sb) in sums.iter().enumerate().take(nb) {
                        grads.bin_gain[pid * nb + b] += gi * g * sb;
                    }
                } else {
                    let contribution = self.term_contribution(info, term, features, an_values);
                    grads.g[pid] += gi * contribution;
                }
                grads.c[pid] += gi;
            }
        }
    }
}

/// One decision's proprioceptive scalar values (task spec, acceptance criterion 3), each in
/// `[0, 1]`. Computed by [`compute_proprioception_values`]; passed alongside [`RayGridFeatures`]
/// to [`EncoderModel::forward`]/[`EncoderModel::backward`] rather than folded into that struct,
/// since it has nothing to do with the ray grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProprioceptionValues {
    pub grounded: f32,
    pub airborne: f32,
    pub own_hook: f32,
    pub jumps_left: f32,
    pub freeze_timer: f32,
    pub speed: f32,
}

impl ProprioceptionValues {
    fn get(self, channel: ProprioceptionChannel) -> f32 {
        match channel {
            ProprioceptionChannel::Grounded => self.grounded,
            ProprioceptionChannel::Airborne => self.airborne,
            ProprioceptionChannel::OwnHook => self.own_hook,
            ProprioceptionChannel::JumpsLeft => self.jumps_left,
            ProprioceptionChannel::FreezeTimer => self.freeze_timer,
            ProprioceptionChannel::Speed => self.speed,
        }
    }
}

/// Maximum jumps DDNet allows a config to grant (`sv_max_jumps`-ish upper bound used only to
/// normalize [`ProprioceptionValues::jumps_left`] into `[0, 1]` — not a gameplay limit this crate
/// enforces anywhere else).
const MAX_JUMPS_FOR_NORMALIZATION: f32 = 4.0;
/// Normalization cap (ticks) for [`ProprioceptionValues::freeze_timer`] — 300 ticks = 6s at the
/// server's 50Hz tick rate (`ddai_physics::core::SERVER_TICK_SPEED`), comfortably covering a
/// normal freeze duration without saturating every-but-the-longest freeze to `1.0`.
const FREEZE_TIMER_NORMALIZATION_TICKS: f32 = 300.0;

/// FLY.md §5's proprioception table (grounded/airborne/own-hook/jumps/freeze-timer/speed),
/// computed from a character's own observed state. Label **С** throughout (semantic analogy: the
/// *feature* names match FLY.md, but which AN type receives which is **П** per
/// [`ProprioceptionChannel`]'s doc comment).
pub fn compute_proprioception_values(me: &CharacterObservation, cfg: &RayGridConfig) -> ProprioceptionValues {
    let own_hook = match me.hook_state {
        HOOK_FLYING => 0.5,
        HOOK_GRABBED => 1.0,
        _ => 0.0,
    };
    let speed = ((me.vel.x.powi(2) + me.vel.y.powi(2)).sqrt() / cfg.velocity_norm_scale).clamp(0.0, 1.0);
    ProprioceptionValues {
        grounded: f32::from(me.grounded),
        airborne: f32::from(!me.grounded),
        own_hook,
        jumps_left: (me.jumps_left as f32 / MAX_JUMPS_FOR_NORMALIZATION).clamp(0.0, 1.0),
        freeze_timer: if me.is_deep_frozen {
            1.0
        } else {
            (me.freeze_ticks_remaining as f32 / FREEZE_TIMER_NORMALIZATION_TICKS).clamp(0.0, 1.0)
        },
        speed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain_fixtures::{FxInputChannel, FxNeuron, FxType, build_brain_flyg};
    use crate::config::FlyConfig;
    use crate::params::FlyParams;
    use ddai_brain::CharacterObservation;
    use ddai_flyg::{Side, Sign};
    use ddai_physics::map::{TILE_AIR, TILE_FREEZE, TILE_SOLID, Tile};
    use std::f32::consts::PI;
    use std::sync::Arc;

    #[test]
    fn bin_gain_must_have_exactly_params_times_bins_entries() {
        let mut p = EncoderParams::init_default(5);
        p.validate_shape(5, 4).unwrap();
        p.bin_gain = vec![1.0; 20];
        p.validate_shape(5, 4).unwrap();
        // A multiple of the parameter count that is not params x bins used to pass.
        p.bin_gain = vec![1.0; 10];
        assert!(p.validate_shape(5, 4).is_err());
        p.bin_gain = vec![1.0; 40];
        assert!(p.validate_shape(5, 4).is_err());
        // Another bin count makes the same vector wrong.
        p.bin_gain = vec![1.0; 20];
        assert!(p.validate_shape(5, 3).is_err());
        p.bin_gain[3] = f32::NAN;
        assert!(p.validate_shape(5, 4).is_err());
    }

    // --- Config validation ------------------------------------------------------------------

    #[test]
    fn odd_num_directions_is_rejected() {
        let cfg = RayGridConfig {
            num_directions: 47,
            ..RayGridConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(EncoderError::InvalidConfig(_))));
    }

    #[test]
    fn default_config_validates() {
        RayGridConfig::default().validate().unwrap();
    }

    #[test]
    fn zero_bins_is_rejected() {
        let cfg = RayGridConfig {
            num_distance_bins: 0,
            ..RayGridConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- Channel name round-trip ---------------------------------------------------------------

    #[test]
    fn every_channel_name_round_trips() {
        for name in ALL_VISUAL_CHANNELS {
            let ch = Channel::parse(name).unwrap_or_else(|| panic!("{name} should parse"));
            assert_eq!(ch.name(), name);
        }
    }

    #[test]
    fn unknown_channel_name_does_not_parse() {
        assert!(Channel::parse("bogus_channel").is_none());
    }

    #[test]
    fn scalar_channels_are_exactly_the_velocity_ones() {
        for ch in [
            Channel::SelfVelocityX,
            Channel::SelfVelocityY,
            Channel::SelfVelocityFlow,
        ] {
            assert!(!ch.is_spatial(), "{:?} should be scalar", ch);
        }
        for ch in [
            Channel::OpponentPosition,
            Channel::OpponentApproach,
            Channel::OpponentHook,
            Channel::OtherPlayers,
            Channel::Walls,
            Channel::FreezeDeathTiles,
            Channel::NoHookTiles,
        ] {
            assert!(ch.is_spatial(), "{:?} should be spatial", ch);
        }
    }

    // --- Ring angle geometry --------------------------------------------------------------------

    #[test]
    fn ring_angle_matches_hand_worked_directions() {
        // Right (same height): 0. Up (screen -y): pi/2. Left: pi. Down: -pi/2.
        assert!(ring_angle_from_screen_delta(1.0, 0.0).abs() < 1e-6);
        assert!((ring_angle_from_screen_delta(0.0, -1.0) - PI / 2.0).abs() < 1e-6);
        assert!((ring_angle_from_screen_delta(-1.0, 0.0).abs() - PI).abs() < 1e-6);
        assert!((ring_angle_from_screen_delta(0.0, 1.0) + PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn mirroring_azimuth_sign_gives_pi_minus_theta() {
        // Review round 1, F9: checked at the real default `elevation_gain` (not just `1.0`) --
        // the mirror argument doesn't depend on the gain's value (see `ring_angle_from_rf`'s doc
        // comment), but the test should exercise the gain actually used in practice.
        let gain = RayGridConfig::default().elevation_gain;
        for (az, el) in [(-30.0f32, 10.0), (-70.0, -20.0), (-5.0, 0.0)] {
            let theta_l = ring_angle_from_rf(az, el, gain);
            let theta_r = ring_angle_from_rf(-az, el, gain);
            let expected = angular_diff(PI, theta_l); // pi - theta_l, wrapped
            let actual = angular_diff(theta_r, 0.0); // just theta_r, wrapped into (-pi,pi]
            assert!(
                (angular_diff(actual, expected)).abs() < 1e-5,
                "theta_l={theta_l} theta_r={theta_r} expected(pi-theta_l)={expected}"
            );
        }
    }

    /// Review round 1, F3 (major, confirmed): the looming feature was effectively always zero
    /// because `looming_k` was tuned assuming `v` in px/s, while `CharacterObservation::vel` is
    /// actually px/tick (F4). A realistic approach — 3 tiles away, closing head-on at `10` px/tick
    /// (`tuning.ground_control_speed()`'s own default) — must give a *meaningfully* nonzero
    /// feature, not `~1e-4`.
    #[test]
    fn looming_feature_is_meaningfully_nonzero_for_a_realistic_approach() {
        let cfg = RayGridConfig::default();
        let me = CharacterObservation {
            vel: ddai_physics::vmath::Vec2::new(0.0, 0.0),
            ..CharacterObservation::at_rest(0)
        };
        let mut opp = CharacterObservation::at_rest(1);
        let d = 96.0f32; // 3 tiles
        opp.vel = ddai_physics::vmath::Vec2::new(-10.0, 0.0); // closing head-on at 10 px/tick
        let feature = compute_looming_feature(&me, &opp, d, 0.0, d, cfg.looming_k);
        assert!(
            feature >= 0.3,
            "expected a meaningfully nonzero looming feature, got {feature}"
        );

        // A receding opponent must still saturate to (near) zero.
        let mut receding = opp;
        receding.vel = ddai_physics::vmath::Vec2::new(10.0, 0.0);
        let feature_receding = compute_looming_feature(&me, &receding, d, 0.0, d, cfg.looming_k);
        assert!(
            feature_receding < 1e-6,
            "a receding opponent must not loom: {feature_receding}"
        );
    }

    /// Review round 1, F1 (blocker, confirmed): elevation was inverted (dorsal cells were
    /// responding to things *below*, not above). A neuron whose RF sits straight up (elevation
    /// +90, azimuth 0) must line up with an opponent placed straight above the tee; one at
    /// elevation -90 must line up with an opponent straight below. Checked via each neuron's own
    /// `ring_angle_from_rf` against the same ring angle an actual "opponent above"/"opponent
    /// below" bearing produces (`ring_angle_from_screen_delta`), not just a sign flip in
    /// isolation — this is the same kind of real, end-to-end check the reviewer's own repro used.
    #[test]
    fn opponent_above_drives_dorsal_cells_more_than_ventral_ones() {
        // Azimuth is exactly 0 here, so `elevation_gain` cannot change the result (`atan2(y, 0)`
        // is `+-pi/2` regardless of `y`'s magnitude) -- `1.0` is fine for this specific check.
        let theta_dorsal = ring_angle_from_rf(0.0, 90.0, 1.0);
        let theta_ventral = ring_angle_from_rf(0.0, -90.0, 1.0);
        let theta_up = ring_angle_from_screen_delta(0.0, -1.0); // opponent straight above self
        let theta_down = ring_angle_from_screen_delta(0.0, 1.0); // opponent straight below self

        assert!(
            angular_diff(theta_dorsal, theta_up).abs() < 1e-5,
            "a dorsal (elevation +90) receptive field must line up with 'opponent above': theta_dorsal={theta_dorsal} theta_up={theta_up}"
        );
        assert!(
            angular_diff(theta_ventral, theta_down).abs() < 1e-5,
            "a ventral (elevation -90) receptive field must line up with 'opponent below': theta_ventral={theta_ventral} theta_down={theta_down}"
        );

        // End-to-end: an opponent placed above the tee must inject more current into a dorsal
        // neuron's spatial channel than into a ventral one at the same azimuth (not just the
        // angle in isolation -- the actual Gaussian-weighted current the encoder computes).
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![tile(TILE_AIR); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(300.0, 200.0); // straight above (smaller screen y)
        let obs = ddai_brain::Observation {
            map: Arc::new(map),
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        feat.compute(&obs, &cfg);

        let ray_weight_at = |theta: f32| -> f32 {
            let sigma_rad = cfg.rf_sigma_deg.to_radians();
            let d = feat.num_directions();
            (0..d)
                .map(|k| angular_gaussian(theta, ray_angle(k, d), sigma_rad))
                .zip(feat.spatial(Channel::OpponentPosition).chunks(feat.num_bins()))
                .map(|(w, bins)| w * bins.iter().sum::<f32>())
                .sum()
        };
        let current_dorsal = ray_weight_at(theta_dorsal);
        let current_ventral = ray_weight_at(theta_ventral);
        assert!(
            current_dorsal > current_ventral * 4.0,
            "opponent above should drive the dorsal RF far more than the ventral one: dorsal={current_dorsal} ventral={current_ventral}"
        );
    }

    fn tile(index: u8) -> Tile {
        Tile {
            index,
            ..Default::default()
        }
    }

    fn map_with_wall_to_the_right() -> ddai_physics::map::MapData {
        // 10x3 tiles; self spawns at tile (1,1) (center row); a solid wall sits at (5,1), i.e.
        // 4 tiles directly to the right.
        let w = 10usize;
        let h = 3usize;
        let mut game = vec![tile(TILE_AIR); w * h];
        game[w + 5] = tile(TILE_SOLID); // row 1, column 5
        ddai_physics::map::MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn ray_cast_detects_a_wall_at_the_right_distance_and_direction() {
        let map = map_with_wall_to_the_right();
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        // Self at the center of tile (1,1): pixel (48, 48).
        feat.cast_ray(0, (48.0, 48.0), &map, &cfg); // direction 0 = ring angle 0 = +x = "right"

        let grid = feat.spatial(Channel::Walls);
        let peak_bin = grid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Wall is at tile x=5, self at tile x=1 -> 4 tiles away; max_range=20 tiles, 4 bins ->
        // normalized distance 4/20=0.2 falls in bin 0 ([0, 0.25)).
        assert_eq!(peak_bin, 0, "grid={grid:?}");
        assert!(grid[peak_bin] > 0.5, "peak weight should be close to 1.0: {grid:?}");
    }

    /// A brute-force reference for the exact ray/wall crossing distance (tiles), marching in
    /// fine (`step`-tile) pixel increments and classifying each sampled point directly — a much
    /// simpler, much slower, independent implementation to check the real DDA in `cast_ray`
    /// against (review round 1, F7's own methodology: a fine-step reference, not another
    /// DDA-shaped implementation that could share the same bug).
    fn brute_force_wall_distance_tiles(
        map: &ddai_physics::map::MapData,
        origin: (f32, f32),
        theta: f32,
        max_range_tiles: f32,
    ) -> Option<f32> {
        let (ddx, ddy) = (theta.cos(), -theta.sin());
        let step = 0.02f32; // tiles
        let mut t = 0.0f32;
        while t <= max_range_tiles {
            let px = origin.0 + ddx * t * 32.0;
            let py = origin.1 + ddy * t * 32.0;
            if tile_class(map, px, py).is_solid() {
                return Some(t);
            }
            t += step;
        }
        None
    }

    fn checkerboard_map(width: u32, height: u32, period: i32) -> ddai_physics::map::MapData {
        let mut game = vec![tile(TILE_AIR); (width * height) as usize];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if (x + y) % period == 0 && (x * 7 + y * 13) % 5 == 0 {
                    game[(y * width as i32 + x) as usize] = tile(TILE_SOLID);
                }
            }
        }
        ddai_physics::map::MapData {
            width,
            height,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    /// Review round 1, F7 (major, CONFIRMED): the reviewer's own brute-force check against a real
    /// map found 0.19% of rays seeing through a wall and 0.23% overshooting by more than a tile
    /// under the old fixed-32px-step march. Reproduced here with an irregular synthetic
    /// checkerboard-ish map (deterministic, not hand-picked to favor either implementation) and
    /// every one of `cast_ray`'s 48 default directions from several origins: every hit distance
    /// must agree with the brute-force reference to within one fine step, and — the actual
    /// failure mode the reviewer found — `cast_ray` must never report "no wall" when the
    /// brute-force reference found one closer than `max_range_tiles` (seeing through a wall),
    /// nor overshoot past a wall the reference found within one tile.
    #[test]
    fn cast_ray_matches_a_brute_force_reference_on_an_irregular_map() {
        let map = checkerboard_map(40, 40, 3);
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        let mut checked_at_least_one_wall = false;
        for &origin in &[(150.0, 150.0), (300.0, 620.0), (500.0, 900.0), (80.0, 1100.0)] {
            for d in 0..cfg.num_directions {
                let theta = ray_angle(d, cfg.num_directions);
                let reference = brute_force_wall_distance_tiles(&map, origin, theta, cfg.max_range_tiles);
                feat.cast_ray(d, origin, &map, &cfg);
                let grid = feat.spatial(Channel::Walls);
                let peak = grid[d * cfg.num_distance_bins..(d + 1) * cfg.num_distance_bins]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);
                let dda_saw_a_wall = peak > 0.05;

                match reference {
                    Some(ref_t) => {
                        checked_at_least_one_wall = true;
                        assert!(
                            dda_saw_a_wall,
                            "DDA saw through a wall the brute-force reference found at {ref_t} tiles (origin={origin:?}, d={d})"
                        );
                    }
                    None => {
                        // The reference found no wall within range; DDA must not report a wall
                        // more than one tile *inside* range either (an overshoot in the other
                        // direction -- reporting a hit that isn't really there -- would also be
                        // wrong, though the reviewer's repro was specifically the opposite case).
                    }
                }
            }
        }
        assert!(
            checked_at_least_one_wall,
            "test map/origins must actually exercise some wall hits"
        );
    }

    /// The exact failure class the reviewer's repro found: a specific origin/direction pair
    /// where a naive fixed-step march's sample points straddle a wall without ever landing
    /// inside it, while the true DDA traversal (which visits *every* crossed tile, not just
    /// sampled points) correctly detects it. Constructed directly rather than relying on the
    /// checkerboard test to happen to hit this case: a single wall tile one map-diagonal-step
    /// away from the origin, at an angle where a 32px-spaced sample march can straddle it.
    #[test]
    fn cast_ray_does_not_see_through_a_thin_diagonal_wall() {
        let mut game = vec![tile(TILE_AIR); 400];
        // A single solid tile at (7, 3) -- not on any axis-aligned line from the origin tile
        // (2, 2), so the ray from the origin towards its center crosses several tile boundaries
        // at fractional (non-32px-multiple) distances.
        game[3 * 20 + 7] = tile(TILE_SOLID);
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let origin = (2.0 * 32.0 + 16.0, 2.0 * 32.0 + 16.0);
        let target = (7.0 * 32.0 + 16.0, 3.0 * 32.0 + 16.0);
        let theta = ring_angle_from_screen_delta(target.0 - origin.0, target.1 - origin.1);
        // The nearest of the 48 default ray directions to this exact bearing.
        let cfg = RayGridConfig::default();
        let d = (0..cfg.num_directions)
            .min_by(|&a, &b| {
                angular_diff(ray_angle(a, cfg.num_directions), theta)
                    .abs()
                    .partial_cmp(&angular_diff(ray_angle(b, cfg.num_directions), theta).abs())
                    .unwrap()
            })
            .unwrap();

        let mut feat = RayGridFeatures::new(&cfg);
        feat.cast_ray(d, origin, &map, &cfg);
        let grid = feat.spatial(Channel::Walls);
        let peak = grid[d * cfg.num_distance_bins..(d + 1) * cfg.num_distance_bins]
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        assert!(
            peak > 0.05,
            "DDA must detect the wall along the nearest ray direction: grid slice={grid:?}"
        );
    }

    #[test]
    fn ray_cast_does_not_see_a_wall_behind_it() {
        let map = map_with_wall_to_the_right();
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        let left_dir = cfg.num_directions / 2; // ring angle pi = "left" = away from the wall
        feat.cast_ray(left_dir, (48.0, 48.0), &map, &cfg);
        assert!(
            feat.spatial(Channel::Walls).iter().all(|&x| x < 1e-6),
            "no wall should be detected looking away from it"
        );
    }

    #[test]
    fn hazard_tile_does_not_occlude_the_ray_but_a_wall_does() {
        let w = 10usize;
        let h = 3usize;
        let mut game = vec![tile(TILE_AIR); w * h];
        game[w + 3] = tile(TILE_FREEZE); // row 1, column 3 -- hazard closer
        game[w + 6] = tile(TILE_SOLID); // row 1, column 6 -- wall further, still visible past the hazard
        let map = ddai_physics::map::MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        feat.cast_ray(0, (48.0, 48.0), &map, &cfg);
        assert!(feat.spatial(Channel::FreezeDeathTiles).iter().any(|&x| x > 0.1));
        assert!(feat.spatial(Channel::Walls).iter().any(|&x| x > 0.1));
    }

    // --- Point-object placement -----------------------------------------------------------------

    #[test]
    fn opponent_directly_to_the_right_peaks_at_ray_zero() {
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![tile(TILE_AIR); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(400.0, 300.0); // directly right, 100px away
        let obs = ddai_brain::Observation {
            map: Arc::new(map),
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        feat.compute(&obs, &cfg);
        let grid = feat.spatial(Channel::OpponentPosition);
        let (peak_idx, &peak_val) = grid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let peak_dir = peak_idx / cfg.num_distance_bins;
        assert_eq!(peak_dir, 0, "should peak at ray direction 0 (bearing 0): grid={grid:?}");
        assert!(peak_val > 0.5);
    }

    /// Review round 1, F13 (CONFIRMED): with two `others`, `compute` must drive
    /// `opponent_position` from whichever `Observation::target_id` actually selects, not
    /// unconditionally from `others[0]` -- checked by putting the *first*-listed character
    /// directly above (bearing pi/2) and the *second*-listed, `target_id`-selected one directly
    /// to the right (bearing 0): the peak must land at bearing 0 (the selected target), and the
    /// un-selected character above must show up in `other_players` instead.
    #[test]
    fn compute_drives_opponent_position_from_target_id_not_others_first() {
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![tile(TILE_AIR); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let cfg = RayGridConfig::default();
        let mut feat = RayGridFeatures::new(&cfg);
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let mut listed_first = CharacterObservation::at_rest(1);
        listed_first.pos = ddai_physics::vmath::Vec2::new(300.0, 200.0); // directly above
        let mut selected_target = CharacterObservation::at_rest(2);
        selected_target.pos = ddai_physics::vmath::Vec2::new(400.0, 300.0); // directly right
        let obs = ddai_brain::Observation {
            map: Arc::new(map),
            tick: 0,
            self_state: me,
            others: vec![listed_first, selected_target],
            target_id: Some(2),
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        feat.compute(&obs, &cfg);

        let opp_grid = feat.spatial(Channel::OpponentPosition);
        let (peak_idx, &peak_val) = opp_grid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let peak_dir = peak_idx / cfg.num_distance_bins;
        assert_eq!(
            peak_dir, 0,
            "opponent_position must peak at bearing 0 (the target_id-selected character), not \
             pi/2 (others[0]): grid={opp_grid:?}"
        );
        assert!(peak_val > 0.5);

        // The un-selected, first-listed character (directly above) must appear in
        // `other_players` instead, not be silently dropped.
        let others_grid = feat.spatial(Channel::OtherPlayers);
        assert!(
            others_grid.iter().any(|&v| v > 0.5),
            "the un-selected character must still show up in other_players: grid={others_grid:?}"
        );
    }

    // --- EncoderModel validation -----------------------------------------------------------------

    fn vpn_an_flyg() -> ddai_flyg::Flyg {
        let types = [
            FxType {
                name: "VPN_T",
                sign: Sign::Excitatory,
            },
            FxType {
                name: "AN_T",
                sign: Sign::Excitatory,
            },
        ];
        let neurons = [
            FxNeuron {
                type_index: 0,
                role: NeuronRole::InputVisual,
                side: Side::L,
                full_connectome_in: 10,
                rf: (-45.0, 0.0),
            },
            FxNeuron {
                type_index: 0,
                role: NeuronRole::InputVisual,
                side: Side::R,
                full_connectome_in: 10,
                rf: (45.0, 0.0),
            },
            FxNeuron {
                type_index: 1,
                role: NeuronRole::InputAscending,
                side: Side::M,
                full_connectome_in: 10,
                rf: (0.0, 0.0),
            },
        ];
        build_brain_flyg(
            &types,
            &neurons,
            &[],
            &[FxInputChannel {
                type_name: "VPN_T",
                channels: vec![CH_OPPONENT_POSITION],
            }],
            &[],
        )
    }

    fn model_from(flyg: ddai_flyg::Flyg) -> FlyModel {
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, 1);
        FlyModel::new(flyg, config, params).unwrap()
    }

    fn default_proprio() -> ProprioceptionConfig {
        ProprioceptionConfig {
            grounded: vec!["AN_T".to_string()],
            ..ProprioceptionConfig::default()
        }
    }

    #[test]
    fn encoder_model_builds_for_a_valid_graph_and_config() {
        let model = model_from(vpn_an_flyg());
        let enc = EncoderModel::new(&model, RayGridConfig::default(), &default_proprio()).unwrap();
        assert_eq!(enc.num_inputs(), 3);
        assert_eq!(enc.num_params(), 2); // (VPN_T, opponent_position), (AN_T, grounded)
    }

    #[test]
    fn unknown_an_type_name_in_proprioception_config_is_rejected() {
        let model = model_from(vpn_an_flyg());
        let bad = ProprioceptionConfig {
            grounded: vec!["NOT_A_REAL_TYPE".to_string()],
            ..ProprioceptionConfig::default()
        };
        assert!(matches!(
            EncoderModel::new(&model, RayGridConfig::default(), &bad),
            Err(EncoderError::UnknownAnType(_))
        ));
    }

    #[test]
    fn assigning_a_visual_type_name_as_an_an_type_is_rejected() {
        let model = model_from(vpn_an_flyg());
        let bad = ProprioceptionConfig {
            grounded: vec!["VPN_T".to_string()], // exists, but is InputVisual, not InputAscending
            ..ProprioceptionConfig::default()
        };
        assert!(matches!(
            EncoderModel::new(&model, RayGridConfig::default(), &bad),
            Err(EncoderError::UnknownAnType(_))
        ));
    }

    #[test]
    fn invalid_ray_grid_config_is_rejected_before_touching_the_graph() {
        let model = model_from(vpn_an_flyg());
        let bad_cfg = RayGridConfig {
            num_directions: 3,
            ..RayGridConfig::default()
        };
        assert!(matches!(
            EncoderModel::new(&model, bad_cfg, &default_proprio()),
            Err(EncoderError::InvalidConfig(_))
        ));
    }

    #[test]
    fn visual_channels_with_no_neurons_reports_the_unused_ones() {
        let model = model_from(vpn_an_flyg());
        let enc = EncoderModel::new(&model, RayGridConfig::default(), &default_proprio()).unwrap();
        let unused = enc.visual_channels_with_no_neurons();
        assert!(unused.contains(&CH_WALLS));
        assert!(!unused.contains(&CH_OPPONENT_POSITION));
        assert_eq!(unused.len(), ALL_VISUAL_CHANNELS.len() - 1);
    }

    // --- Forward / backward consistency ----------------------------------------------------------

    #[test]
    fn forward_matches_a_hand_computed_current_for_a_scalar_channel() {
        let model = model_from(vpn_an_flyg());
        let enc = EncoderModel::new(&model, RayGridConfig::default(), &default_proprio()).unwrap();
        let mut params = EncoderParams::init_default(enc.num_params());
        // Find the (AN_T, grounded) param id.
        let pid = enc
            .assignments()
            .iter()
            .find(|a| a.type_name == "AN_T" && a.channel == "grounded")
            .unwrap()
            .param_id as usize;
        params.g[pid] = 3.0;
        params.c[pid] = 0.5;

        let features = RayGridFeatures::new(enc.ray_grid_config());
        let an_values = ProprioceptionValues {
            grounded: 1.0,
            airborne: 0.0,
            own_hook: 0.0,
            jumps_left: 0.0,
            freeze_timer: 0.0,
            speed: 0.0,
        };
        let mut out = vec![0.0; enc.num_inputs()];
        enc.forward(&features, &an_values, &params, &mut out);

        // Input order is model.input_neuron_indices(): VPN_L(0), VPN_R(1), AN(2). AN's only term
        // is (grounded, pid): I = g*1.0 + c = 3.0 + 0.5 = 3.5. The two VPN neurons have g=1,c=0
        // (untouched) and zero opponent feature (no opponent in `features`), so I=0 for them.
        assert!((out[2] - 3.5).abs() < 1e-6, "out={out:?}");
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1]).abs() < 1e-6);
    }

    /// Per-distance-bin gains (task 8.2): with unit gains the current equals the 7.3 encoder's,
    /// with a near-bin-only gain a near object drives a neuron and a far one does not (distance is
    /// no longer summed away), and `backward_with_params` matches finite differences of `forward`
    /// for `g`, `c` and the bin gains.
    #[test]
    fn distance_bin_gains_restore_distance_and_have_correct_gradients() {
        let model = model_from(vpn_an_flyg());
        let cfg = RayGridConfig {
            learn_distance_gains: true,
            ..RayGridConfig::default()
        };
        let enc = EncoderModel::new(&model, cfg, &default_proprio()).unwrap();
        let nb = cfg.num_distance_bins;
        let plain = enc.init_params();
        assert_eq!(plain.bin_gain.len(), enc.num_params() * nb);
        assert!(plain.bin_gain.iter().all(|&g| g == 1.0));

        let map = Arc::new(ddai_physics::map::MapData {
            width: 40,
            height: 40,
            game: vec![tile(TILE_AIR); 1600],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        });
        let obs_at = |dx: f32| {
            let mut me = CharacterObservation::at_rest(0);
            me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
            let mut opp = CharacterObservation::at_rest(1);
            opp.pos = ddai_physics::vmath::Vec2::new(300.0 + dx, 300.0);
            ddai_brain::Observation {
                map: map.clone(),
                tick: 0,
                self_state: me,
                others: vec![opp],
                target_id: None,
                tuning: ddai_physics::tuning::TuningParams::default(),
            }
        };
        let an = ProprioceptionValues {
            grounded: 0.0,
            airborne: 0.0,
            own_hook: 0.0,
            jumps_left: 0.0,
            freeze_timer: 0.0,
            speed: 0.0,
        };
        let currents = |p: &EncoderParams, dx: f32| {
            let mut f = RayGridFeatures::new(enc.ray_grid_config());
            f.compute(&obs_at(dx), enc.ray_grid_config());
            let mut out = vec![0.0; enc.num_inputs()];
            enc.forward(&f, &an, p, &mut out);
            out
        };
        // Unit gains equal the encoder without gains.
        let legacy = EncoderParams::init_default(enc.num_params());
        for dx in [60.0f32, 400.0] {
            for (a, b) in currents(&plain, dx).iter().zip(&currents(&legacy, dx)) {
                assert!((a - b).abs() < 1e-5);
            }
        }
        // Only the nearest bin has gain: a near opponent drives the VPN neurons, a far one hardly.
        let mut near_only = plain.clone();
        for p in 0..enc.num_params() {
            for b in 0..nb {
                near_only.bin_gain[p * nb + b] = f32::from(b == 0);
            }
        }
        let near: f32 = currents(&near_only, 60.0).iter().map(|x| x.abs()).sum();
        let far: f32 = currents(&near_only, 500.0).iter().map(|x| x.abs()).sum();
        assert!(near > 5.0 * far.max(1e-6), "near {near} vs far {far}");
        // With uniform gains the two distances are indistinguishable in magnitude (the 7.3 flaw).
        let near_u: f32 = currents(&plain, 60.0).iter().map(|x| x.abs()).sum();
        let far_u: f32 = currents(&plain, 500.0).iter().map(|x| x.abs()).sum();
        assert!(
            (near_u - far_u).abs() < 0.35 * near_u.max(far_u),
            "uniform: near {near_u} far {far_u}"
        );

        // Gradients against finite differences of the forward pass.
        let mut params = plain.clone();
        for (i, g) in params.g.iter_mut().enumerate() {
            *g = 0.8 + 0.2 * i as f32;
        }
        for (i, g) in params.bin_gain.iter_mut().enumerate() {
            *g = 0.5 + 0.1 * (i % 7) as f32;
        }
        let mut f = RayGridFeatures::new(enc.ray_grid_config());
        f.compute(&obs_at(90.0), enc.ray_grid_config());
        let grad_input: Vec<f32> = (0..enc.num_inputs()).map(|k| 1.0 + 0.5 * k as f32).collect();
        let mut analytic = enc.zero_grads();
        enc.backward_with_params(&f, &an, &params, &grad_input, &mut analytic);
        let loss = |p: &EncoderParams| -> f64 {
            let mut out = vec![0.0f32; enc.num_inputs()];
            enc.forward(&f, &an, p, &mut out);
            out.iter()
                .zip(&grad_input)
                .map(|(&i, &g)| f64::from(i) * f64::from(g))
                .sum()
        };
        let h = 1e-3f32;
        let mut checked = 0;
        for i in 0..params.bin_gain.len() {
            let (mut a, mut b) = (params.clone(), params.clone());
            a.bin_gain[i] += h;
            b.bin_gain[i] -= h;
            let fd = (loss(&a) - loss(&b)) / (2.0 * f64::from(h));
            if fd.abs() > 0.05 {
                checked += 1;
                assert!(
                    (f64::from(analytic.bin_gain[i]) - fd).abs() / fd.abs() < 1e-3,
                    "bin_gain[{i}]: {} vs {fd}",
                    analytic.bin_gain[i]
                );
            }
        }
        assert!(checked > 0, "the check must exercise at least one non-zero gradient");
        for pid in 0..enc.num_params() {
            let (mut a, mut b) = (params.clone(), params.clone());
            a.g[pid] += h;
            b.g[pid] -= h;
            let fd = (loss(&a) - loss(&b)) / (2.0 * f64::from(h));
            assert!(
                (f64::from(analytic.g[pid]) - fd).abs() <= 1e-3 * fd.abs().max(1e-3),
                "g[{pid}]"
            );
        }
    }

    /// Encoder-only finite-difference gradient check (task spec's "tiny graphs, f64 reference"):
    /// with `grad_input_currents` fixed (as if handed down from `backward::backward`), `backward`
    /// must give exactly `dL/dg`/`dL/dc` for `L = sum_k grad_input_currents[k] * I_k(g, c)` — a
    /// pure, self-contained check of this module's own math, decoupled from the rest of the fly
    /// model (the full encoder+fly+decoder chain is gradient-checked separately, in
    /// `tests/gradcheck_brain.rs`).
    #[test]
    fn backward_matches_finite_differences_of_forward() {
        let model = model_from(vpn_an_flyg());
        let enc = EncoderModel::new(&model, RayGridConfig::default(), &default_proprio()).unwrap();
        let mut params = EncoderParams::init_default(enc.num_params());
        for (i, g) in params.g.iter_mut().enumerate() {
            *g = 0.7 + 0.3 * i as f32;
        }
        for (i, c) in params.c.iter_mut().enumerate() {
            *c = -0.2 + 0.1 * i as f32;
        }

        let mut features = RayGridFeatures::new(enc.ray_grid_config());
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![tile(TILE_AIR); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(350.0, 280.0);
        let obs = ddai_brain::Observation {
            map: Arc::new(map),
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        features.compute(&obs, enc.ray_grid_config());
        let an_values = ProprioceptionValues {
            grounded: 0.6,
            airborne: 0.4,
            own_hook: 0.0,
            jumps_left: 0.5,
            freeze_timer: 0.0,
            speed: 0.1,
        };

        let grad_input: Vec<f32> = (0..enc.num_inputs()).map(|k| 1.0 + 0.5 * k as f32).collect();
        let mut analytic = EncoderGradients::zeros(enc.num_params());
        enc.backward(&features, &an_values, &grad_input, &mut analytic);

        let loss = |p: &EncoderParams| -> f64 {
            let mut out = vec![0.0f32; enc.num_inputs()];
            enc.forward(&features, &an_values, p, &mut out);
            out.iter()
                .zip(&grad_input)
                .map(|(&i, &g)| f64::from(i) * f64::from(g))
                .sum()
        };

        let h = 1e-3f32;
        for pid in 0..enc.num_params() {
            let mut p_plus = params.clone();
            p_plus.g[pid] += h;
            let mut p_minus = params.clone();
            p_minus.g[pid] -= h;
            let fd_g = (loss(&p_plus) - loss(&p_minus)) / (2.0 * f64::from(h));
            let rel = (f64::from(analytic.g[pid]) - fd_g).abs() / fd_g.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "grad_g[{pid}]: analytic={} fd={fd_g} rel={rel}",
                analytic.g[pid]
            );

            let mut p_plus = params.clone();
            p_plus.c[pid] += h;
            let mut p_minus = params.clone();
            p_minus.c[pid] -= h;
            let fd_c = (loss(&p_plus) - loss(&p_minus)) / (2.0 * f64::from(h));
            let rel = (f64::from(analytic.c[pid]) - fd_c).abs() / fd_c.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "grad_c[{pid}]: analytic={} fd={fd_c} rel={rel}",
                analytic.c[pid]
            );
        }
    }

    /// A deliberately zeroed gradient must NOT pass the same check (review lesson from task 7.2:
    /// gradient checks must be able to fail).
    #[test]
    fn a_zeroed_out_gradient_is_caught_by_the_same_check() {
        let model = model_from(vpn_an_flyg());
        let enc = EncoderModel::new(&model, RayGridConfig::default(), &default_proprio()).unwrap();
        let params = EncoderParams::init_default(enc.num_params());

        // A real opponent in view (not an all-zero `features`, which would make the VPN
        // opponent_position param's gradient genuinely zero regardless of sabotage — every
        // param actually needs a nonzero effect for this check to be meaningful).
        let mut features = RayGridFeatures::new(enc.ray_grid_config());
        let map = ddai_physics::map::MapData {
            width: 20,
            height: 20,
            game: vec![tile(TILE_AIR); 400],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let mut me = CharacterObservation::at_rest(0);
        me.pos = ddai_physics::vmath::Vec2::new(300.0, 300.0);
        let mut opp = CharacterObservation::at_rest(1);
        opp.pos = ddai_physics::vmath::Vec2::new(400.0, 300.0);
        let obs = ddai_brain::Observation {
            map: Arc::new(map),
            tick: 0,
            self_state: me,
            others: vec![opp],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        features.compute(&obs, enc.ray_grid_config());
        let an_values = ProprioceptionValues {
            grounded: 0.8,
            airborne: 0.2,
            own_hook: 0.0,
            jumps_left: 0.0,
            freeze_timer: 0.0,
            speed: 0.0,
        };
        let grad_input = vec![1.0f32; enc.num_inputs()];
        let mut analytic = EncoderGradients::zeros(enc.num_params());
        enc.backward(&features, &an_values, &grad_input, &mut analytic);

        let loss = |p: &EncoderParams| -> f64 {
            let mut out = vec![0.0f32; enc.num_inputs()];
            enc.forward(&features, &an_values, p, &mut out);
            out.iter()
                .zip(&grad_input)
                .map(|(&i, &g)| f64::from(i) * f64::from(g))
                .sum()
        };

        // Sabotage the largest-|g| entry (review lesson from task 7.2: check the parameter most
        // likely to actually matter, not just index 0).
        let biggest = (0..enc.num_params())
            .max_by(|&a, &b| analytic.g[a].abs().partial_cmp(&analytic.g[b].abs()).unwrap())
            .unwrap();
        analytic.g[biggest] = 0.0;

        let h = 1e-3f32;
        let mut p_plus = params.clone();
        p_plus.g[biggest] += h;
        let mut p_minus = params.clone();
        p_minus.g[biggest] -= h;
        let fd_g = (loss(&p_plus) - loss(&p_minus)) / (2.0 * f64::from(h));
        assert!(
            fd_g.abs() > 1e-6,
            "the finite-difference itself must be nonzero for this to be a meaningful check"
        );
        assert_ne!(
            analytic.g[biggest] as f64, fd_g,
            "sabotaged gradient must disagree with the finite difference"
        );
    }

    // --- Review round 1, F9: weak vision directly above/below ------------------------------------

    /// Diagnostic, not a pass/fail check (`#[ignore]`d, needs real connectome data): dumps the
    /// real S graph's receptive-field elevation distribution and the resulting ring-angle
    /// distribution before/after [`RayGridConfig::elevation_gain`]'s fix (review round 1, F9) —
    /// the numbers this field's chosen default (see its own doc comment) were picked from. Run
    /// with `cargo test -p ddai-fly --lib encoder::tests::dump_real_s_elevation_and_ring_angle_distribution -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_real_s_elevation_and_ring_angle_distribution() {
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
        let flyg = ddai_flyg::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));

        let mut els: Vec<f32> = Vec::new();
        for n in &flyg.neurons {
            if let Some(rf) = &n.rf {
                els.push(rf.elevation_deg.abs());
            }
        }
        els.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = els.len();
        let pct = |v: &[f32], p: f32| v[(((v.len() - 1) as f32) * p) as usize];
        eprintln!("n_rf = {n}");
        for p in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0] {
            eprintln!("|elevation_deg| p{:.0} = {:.2}", p * 100.0, pct(&els, p));
        }

        // Fraction of ring angles landing within +-60deg of the horizontal axis (theta near 0 or
        // pi), at gain 1.0 (no fix) vs a candidate gain, for every VPN receptive field.
        let frac_near_horizontal = |gain: f32| -> f32 {
            let mut near = 0usize;
            let mut total = 0usize;
            for n in &flyg.neurons {
                if let Some(rf) = &n.rf {
                    let theta = ring_angle_from_rf(rf.azimuth_deg, rf.elevation_deg, gain);
                    let dist_from_horizontal = theta.sin().abs().asin(); // in [0, pi/2], 0 = exactly horizontal
                    if dist_from_horizontal.to_degrees() < 60.0 {
                        near += 1;
                    }
                    total += 1;
                }
            }
            near as f32 / total.max(1) as f32
        };
        let median_dist_from_horizontal_deg = |gain: f32| -> f32 {
            let mut dists: Vec<f32> = flyg
                .neurons
                .iter()
                .filter_map(|n| n.rf.as_ref())
                .map(|rf| {
                    let theta = ring_angle_from_rf(rf.azimuth_deg, rf.elevation_deg, gain);
                    theta.sin().abs().asin().to_degrees()
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dists[dists.len() / 2]
        };
        for gain in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] {
            eprintln!(
                "gain={gain}: fraction within +-60deg of horizontal = {:.3}, median distance from horizontal = {:.1}deg",
                frac_near_horizontal(gain),
                median_dist_from_horizontal_deg(gain)
            );
        }
    }
}
