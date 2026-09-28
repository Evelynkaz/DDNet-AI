//! Task 7.3, acceptance criterion 6 / review round 1's F14 (CONFIRMED, minor): an independent
//! **dual-number** (forward-mode automatic differentiation) reference for the decoder's and the
//! world-model head's own loss functions — an *exact* derivative (no finite-difference truncation
//! error at all, only `f64` rounding), computed by an entirely separate implementation of the
//! same math from the one under test (`crate::decoder::decoder_loss_and_grad`/`crate::world_model
//! ::world_model_loss_and_grad`), not the finite-difference comparisons the rest of this crate
//! already uses everywhere else. Own implementation (not a copy of anything) — the `Dual` type
//! and every loss function below are written fresh for this file.
//!
//! ## Scope: why *these* two groups, not all four
//! Acceptance criterion 6's four trainable groups are the connectome's `a`/`b`/`theta`, the
//! encoder's `g`/`c`, the decoder's heads, and the world model's heads. This file covers the
//! **decoder** and **world model** — the two groups whose own loss functions are genuinely
//! nonlinear (softmax, sigmoid, `atan2`/`cos`/`sin` for the von Mises aim loss) in a way a
//! finite-difference check's `h`-dependent truncation error could in principle mask a subtly
//! wrong analytic derivative. The other two already have an equivalent or stronger independent
//! check:
//! - **Connectome `a`/`b`/`theta`**: `tests/backward_real_graph.rs`'s own `dual_number_exact_
//!   gradient_matches_analytic_on_the_real_s_graph` (task 7.2, pre-existing, unchanged by this
//!   task) already is a dual-number check, through the actual nonlinear recurrent dynamics
//!   (exponential Euler, `tanh`/`relu` activation) this file's decoder/world-model checks
//!   deliberately do *not* need to re-derive (see below).
//! - **Encoder `g`/`c`**: `EncoderModel::forward`'s own formula (FLY.md §5) is *exactly linear* in
//!   `g`/`c` for a fixed observation (`I_i = Σ feature * g + c`, no nonlinearity at all) — a
//!   central finite difference of a linear function has **zero** truncation error for *any* `h`
//!   (the Taylor remainder term that `h` controls is identically zero when the second derivative
//!   is zero), so the existing finite-difference checks (`encoder::tests::backward_matches_
//!   finite_differences_of_forward`, `tests/gradcheck_brain.rs`'s real-S `encoder.g` spot check)
//!   are already exact up to the same `f64`/`f32` rounding a dual-number check would also be
//!   limited by, not merely "close enough" — reimplementing the encoder's per-ray Gaussian
//!   weighting a third time here would add code without adding rigor.
//!
//! Because decoder/world-model params don't feed into the connectome's own recurrence at all
//! (only *out* of it, reading a fixed `dn_rates`/`r_full` snapshot), this file only needs to
//! reimplement each head's own loss function in dual arithmetic — not the whole encode ->
//! recurrence -> decode pipeline — while still exercising the exact same nonlinear math
//! (softmax cross-entropy, binary cross-entropy, the von Mises aim loss's population vector +
//! `atan2`) the analytic implementation under test computes.

use ddai_fly::brain_fixtures::{FxNeuron, FxOutputGroup, FxType, build_brain_flyg};
use ddai_fly::config::FlyConfig;
use ddai_fly::decoder::{DecoderConfig, DecoderModel, DecoderTargets, DnCalibration, decoder_loss_and_grad};
use ddai_fly::model::FlyModel;
use ddai_fly::params::FlyParams;
use ddai_fly::world_model::{
    HorizonTargets, NUM_BINARY_TARGETS, NUM_REGRESSION_TARGETS, WorldModelConfig, WorldModelHead,
    world_model_loss_and_grad,
};
use ddai_flyg::{NeuronRole, Side, Sign};

// --- A minimal forward-mode dual number (own implementation, see the module doc comment) --------

#[derive(Debug, Clone, Copy)]
struct Dual {
    val: f64,
    deriv: f64,
}

impl Dual {
    fn constant(val: f64) -> Self {
        Dual { val, deriv: 0.0 }
    }
    fn variable(val: f64) -> Self {
        Dual { val, deriv: 1.0 }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val + rhs.val,
            deriv: self.deriv + rhs.deriv,
        }
    }
}
impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val - rhs.val,
            deriv: self.deriv - rhs.deriv,
        }
    }
}
impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val * rhs.val,
            deriv: self.deriv * rhs.val + self.val * rhs.deriv,
        }
    }
}
impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val / rhs.val,
            deriv: (self.deriv * rhs.val - self.val * rhs.deriv) / (rhs.val * rhs.val),
        }
    }
}
impl std::ops::Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            val: -self.val,
            deriv: -self.deriv,
        }
    }
}
impl std::iter::Sum for Dual {
    fn sum<I: Iterator<Item = Dual>>(iter: I) -> Dual {
        iter.fold(Dual::constant(0.0), |a, b| a + b)
    }
}

fn dexp(d: Dual) -> Dual {
    let e = d.val.exp();
    Dual {
        val: e,
        deriv: d.deriv * e,
    }
}
fn dln(d: Dual) -> Dual {
    Dual {
        val: d.val.ln(),
        deriv: d.deriv / d.val,
    }
}
fn dsin(d: Dual) -> Dual {
    Dual {
        val: d.val.sin(),
        deriv: d.deriv * d.val.cos(),
    }
}
fn dcos(d: Dual) -> Dual {
    Dual {
        val: d.val.cos(),
        deriv: -d.deriv * d.val.sin(),
    }
}
fn datan2(y: Dual, x: Dual) -> Dual {
    let denom = x.val * x.val + y.val * y.val;
    Dual {
        val: y.val.atan2(x.val),
        deriv: (x.val * y.deriv - y.val * x.deriv) / denom,
    }
}
fn dsigmoid(d: Dual) -> Dual {
    let s = 1.0 / (1.0 + (-d.val).exp());
    Dual {
        val: s,
        deriv: d.deriv * s * (1.0 - s),
    }
}
fn dscale(d: Dual, k: f64) -> Dual {
    Dual {
        val: d.val * k,
        deriv: d.deriv * k,
    }
}

/// `softmax` over exactly 3 dual logits (no max-subtraction stability trick -- unneeded at this
/// test's modest magnitudes, and it would itself introduce a `max`-kink into the dual arithmetic
/// for no benefit here).
fn dsoftmax3(logits: [Dual; 3]) -> [Dual; 3] {
    let exps = [dexp(logits[0]), dexp(logits[1]), dexp(logits[2])];
    let sum = exps[0] + exps[1] + exps[2];
    [exps[0] / sum, exps[1] / sum, exps[2] / sum]
}

// --- The tiny fixture (own construction, shared shape with `decoder::tests::tiny_decoder_flyg`
// but built independently here since integration test binaries can't import another test file's
// private items) ----------------------------------------------------------------------------

/// Dense output slot layout (Output-role neurons, ascending dense index -> ascending slot):
/// `0,1` = `DN_LR` `L,R` (tied `direction_left`/`direction_right`); `2,3` = `DN_STOP` `L,R`
/// (pooled `direction_stop`); `4,5` = `DN_JUMP`; `6,7` = `DN_HOOK`; `8,9` = `DN_FIRE`; `10,11` =
/// `DN_AIM_A` `L,R` (tied aim pair); `12` = `DN_AIM_UNPAIRED` (side `M`, untied -- label **П**).
/// Plus 3 `Hidden`-role neurons (dense `0..3`, for the world-model subset) before all of the above.
fn tiny_flyg() -> ddai_flyg::Flyg {
    let type_names = [
        "HID",
        "DN_LR",
        "DN_STOP",
        "DN_JUMP",
        "DN_HOOK",
        "DN_FIRE",
        "DN_AIM_A",
        "DN_AIM_UNPAIRED",
    ];
    let types: Vec<FxType> = type_names
        .iter()
        .map(|&name| FxType {
            name,
            sign: Sign::Excitatory,
        })
        .collect();
    let mut neurons = Vec::new();
    for _ in 0..3 {
        neurons.push(FxNeuron {
            type_index: 0,
            role: NeuronRole::Hidden,
            side: Side::M,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    for ti in [1u32, 2, 3, 4, 5, 6] {
        neurons.push(FxNeuron {
            type_index: ti,
            role: NeuronRole::Output,
            side: Side::L,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
        neurons.push(FxNeuron {
            type_index: ti,
            role: NeuronRole::Output,
            side: Side::R,
            full_connectome_in: 1,
            rf: (0.0, 0.0),
        });
    }
    neurons.push(FxNeuron {
        type_index: 7,
        role: NeuronRole::Output,
        side: Side::M,
        full_connectome_in: 1,
        rf: (0.0, 0.0),
    });

    let output_groups = vec![
        FxOutputGroup {
            action: "direction_left",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::L),
        },
        FxOutputGroup {
            action: "direction_right",
            member_type_names: vec!["DN_LR"],
            side_filter: Some(Side::R),
        },
        FxOutputGroup {
            action: "direction_stop",
            member_type_names: vec!["DN_STOP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "jump",
            member_type_names: vec!["DN_JUMP"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "hook",
            member_type_names: vec!["DN_HOOK"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "fire",
            member_type_names: vec!["DN_FIRE"],
            side_filter: None,
        },
        FxOutputGroup {
            action: "aim",
            member_type_names: vec!["DN_AIM_A", "DN_AIM_UNPAIRED"],
            side_filter: None,
        },
    ];
    build_brain_flyg(&types, &neurons, &[], &[], &output_groups)
}

fn model_from(flyg: ddai_flyg::Flyg) -> FlyModel {
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    FlyModel::new(flyg, config, params).unwrap()
}

/// The 13 `z`-scores (== `dn_rates` directly here: `mu=0`, `sigma=1`, well inside the clip bound,
/// so the frozen calibration is a no-op identity and doesn't need its own dual-number treatment --
/// only the decoder's *own* parameters are ever `Dual::variable` in this file) at the slot layout
/// `tiny_flyg`'s doc comment describes.
fn z_values() -> [f64; 13] {
    std::array::from_fn(|i| 0.3 + 0.11 * i as f64)
}

// The *dual* side is exact to `f64` rounding, but `decoder_loss_and_grad`/`world_model_loss_and_
// grad` (the analytic side) compute entirely in `f32` (~1.19e-7 relative precision) -- so `1e-5`
// here isn't a concession to approximate differentiation (unlike every finite-difference
// tolerance elsewhere in this crate, which has to absorb real truncation error too), it is purely
// the `f32` analytic value's own rounding floor, comfortably below what would indicate an
// actually-wrong derivative (a wrong analytic formula shows up as `rel` of order `0.1`-`1`, not
// `1e-7`).
const RTOL: f64 = 1e-5;
const ATOL: f64 = 1e-6;

fn assert_close(label: &str, analytic: f32, dual_deriv: f64) {
    let a = f64::from(analytic);
    let rel = (a - dual_deriv).abs() / dual_deriv.abs().max(1e-9);
    assert!(
        rel < RTOL || (a - dual_deriv).abs() < ATOL,
        "{label}: analytic={a} dual={dual_deriv} rel={rel}"
    );
}

/// The decoder's total loss (matching `crate::decoder::decoder_loss_and_grad`'s exact
/// composition: cross-entropy + BCE + BCE + BCE + von Mises NLL, summed, L1 excluded -- see that
/// function's own doc comment for why L1 is deliberately kept separate), computed in dual
/// arithmetic. `params`: `[direction_lr_w, direction_lr_b, direction_stop_w, direction_stop_b,
/// jump_w, jump_b, hook_w, hook_b, fire_w, fire_b, aim_pair_theta, aim_unpaired_theta]` (12 duals,
/// matching this fixture's one-type-per-pooled-head shape).
#[allow(clippy::too_many_arguments)]
fn dual_decoder_loss(
    z: &[f64; 13],
    direction_lr_w: Dual,
    direction_lr_b: Dual,
    direction_stop_w: Dual,
    direction_stop_b: Dual,
    jump_w: Dual,
    jump_b: Dual,
    hook_w: Dual,
    hook_b: Dual,
    fire_w: Dual,
    fire_b: Dual,
    aim_pair_theta: Dual,
    aim_unpaired_theta: Dual,
    target: &DecoderTargets,
) -> Dual {
    let zc = |i: usize| Dual::constant(z[i]);
    let mean2 = |a: usize, b: usize| Dual::constant((z[a] + z[b]) / 2.0);

    let mut loss = Dual::constant(0.0);

    if let Some(t) = target.direction {
        let left_logit = direction_lr_b + direction_lr_w * zc(0);
        let right_logit = direction_lr_b + direction_lr_w * zc(1);
        let stop_logit = direction_stop_b + direction_stop_w * mean2(2, 3);
        let probs = dsoftmax3([left_logit, stop_logit, right_logit]);
        loss = loss - dln(probs[t as usize]);
    }
    if let Some(y) = target.jump {
        let logit = jump_b + jump_w * mean2(4, 5);
        let p = dsigmoid(logit);
        let yf = Dual::constant(f64::from(y));
        loss = loss - (yf * dln(p) + (Dual::constant(1.0) - yf) * dln(Dual::constant(1.0) - p));
    }
    if let Some(y) = target.hook {
        let logit = hook_b + hook_w * mean2(6, 7);
        let p = dsigmoid(logit);
        let yf = Dual::constant(f64::from(y));
        loss = loss - (yf * dln(p) + (Dual::constant(1.0) - yf) * dln(Dual::constant(1.0) - p));
    }
    if let Some(y) = target.fire {
        let logit = fire_b + fire_w * mean2(8, 9);
        let p = dsigmoid(logit);
        let yf = Dual::constant(f64::from(y));
        loss = loss - (yf * dln(p) + (Dual::constant(1.0) - yf) * dln(Dual::constant(1.0) - p));
    }
    if let Some(target_angle) = target.aim {
        // Population vector (matches `crate::decoder::population_vector`'s exact math): a tied
        // pair's asymmetry (`zl-zr`) loads onto `C` (`cos theta`), its sum (`zl+zr`) onto `S`
        // (`sin theta`); an unpaired member contributes its own rate directly to both.
        let (zl, zr) = (zc(10), zc(11));
        let c = (zl - zr) * dcos(aim_pair_theta) + zc(12) * dcos(aim_unpaired_theta);
        let s = (zl + zr) * dsin(aim_pair_theta) + zc(12) * dsin(aim_unpaired_theta);
        let mu = datan2(s, c);
        // Von Mises NLL, kappa=4.0 fixed (matches `DecoderConfig::aim_kappa`'s default;
        // `log(2*pi*I0(kappa))` dropped -- constant w.r.t. every trainable quantity here, same as
        // the analytic implementation under test).
        loss = loss - dscale(dcos(Dual::constant(f64::from(target_angle)) - mu), 4.0);
    }
    loss
}

#[test]
fn dual_number_gradients_match_analytic_for_every_decoder_head_on_a_tiny_graph() {
    let flyg = tiny_flyg();
    let model = model_from(flyg);
    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();

    // Sanity: this fixture actually resolves to the shape `tiny_flyg`'s doc comment claims -- via
    // `DecoderParams`' own (public) field lengths, since `DecoderModel`'s internal tying
    // structure itself is private and this is a separate integration-test binary, not
    // `decoder::tests`' own descendant module.
    let default_params = decoder.init_default_params();
    assert_eq!(default_params.aim_pair_theta.len(), 1, "1 tied aim pair (DN_AIM_A)");
    assert_eq!(
        default_params.aim_unpaired_theta.len(),
        1,
        "1 unpaired aim member (DN_AIM_UNPAIRED)"
    );
    assert_eq!(default_params.direction_lr_w.len(), 1, "1 tied direction type (DN_LR)");
    assert_eq!(default_params.direction_stop_w.len(), 1, "1 pooled stop type (DN_STOP)");

    let calib = DnCalibration {
        mu: vec![0.0; 13],
        sigma: vec![1.0; 13],
    };
    let z = z_values();
    let dn_rates: Vec<f32> = z.iter().map(|&v| v as f32).collect();

    let mut params = decoder.init_default_params();
    params.direction_lr_w = vec![0.7];
    params.direction_lr_b = -0.2;
    params.direction_stop_w = vec![-0.4];
    params.direction_stop_b = 0.1;
    params.jump_w = vec![0.5];
    params.jump_b = 0.05;
    params.hook_w = vec![0.3];
    params.hook_b = -0.1;
    params.fire_w = vec![-0.6];
    params.fire_b = 0.2;
    params.aim_pair_theta = vec![0.4];
    params.aim_unpaired_theta = vec![1.1];

    let targets = DecoderTargets {
        direction: Some(2),
        jump: Some(true),
        hook: Some(false),
        fire: Some(true),
        aim: Some(1.0),
    };

    let (_loss, analytic, _grad_dn) = decoder_loss_and_grad(&decoder, &dn_rates, &calib, &params, &targets);

    // Flat `f64` view of the 12 scalar parameters, in `dual_decoder_loss`'s own argument order.
    let values: [f64; 12] = [
        f64::from(params.direction_lr_w[0]),
        f64::from(params.direction_lr_b),
        f64::from(params.direction_stop_w[0]),
        f64::from(params.direction_stop_b),
        f64::from(params.jump_w[0]),
        f64::from(params.jump_b),
        f64::from(params.hook_w[0]),
        f64::from(params.hook_b),
        f64::from(params.fire_w[0]),
        f64::from(params.fire_b),
        f64::from(params.aim_pair_theta[0]),
        f64::from(params.aim_unpaired_theta[0]),
    ];
    let analytic_flat: [f32; 12] = [
        analytic.direction_lr_w[0],
        analytic.direction_lr_b,
        analytic.direction_stop_w[0],
        analytic.direction_stop_b,
        analytic.jump_w[0],
        analytic.jump_b,
        analytic.hook_w[0],
        analytic.hook_b,
        analytic.fire_w[0],
        analytic.fire_b,
        analytic.aim_pair_theta[0],
        analytic.aim_unpaired_theta[0],
    ];
    let labels = [
        "direction_lr_w",
        "direction_lr_b",
        "direction_stop_w",
        "direction_stop_b",
        "jump_w",
        "jump_b",
        "hook_w",
        "hook_b",
        "fire_w",
        "fire_b",
        "aim_pair_theta",
        "aim_unpaired_theta",
    ];

    // `vary`: which of the 12 slots is `Dual::variable`; every other slot is `Dual::constant`.
    let eval = |vary: usize| -> Dual {
        let d = |i: usize| -> Dual {
            if i == vary {
                Dual::variable(values[i])
            } else {
                Dual::constant(values[i])
            }
        };
        dual_decoder_loss(
            &z,
            d(0),
            d(1),
            d(2),
            d(3),
            d(4),
            d(5),
            d(6),
            d(7),
            d(8),
            d(9),
            d(10),
            d(11),
            &targets,
        )
    };

    for i in 0..12 {
        assert_close(labels[i], analytic_flat[i], eval(i).deriv);
    }
}

/// Review lesson from task 7.2 (applied throughout this crate): a gradient check must be able to
/// fail. Sabotaging one analytic decoder gradient must break this dual-number comparison too, not
/// just the finite-difference ones elsewhere.
#[test]
fn a_sabotaged_decoder_gradient_is_caught_by_the_dual_number_check() {
    let flyg = tiny_flyg();
    let model = model_from(flyg);
    let decoder = DecoderModel::new(&model, DecoderConfig::default()).unwrap();
    let calib = DnCalibration {
        mu: vec![0.0; 13],
        sigma: vec![1.0; 13],
    };
    let z = z_values();
    let dn_rates: Vec<f32> = z.iter().map(|&v| v as f32).collect();
    let mut params = decoder.init_default_params();
    params.jump_w = vec![0.5];
    params.jump_b = 0.05;
    let targets = DecoderTargets {
        jump: Some(true),
        ..Default::default()
    };
    let (_loss, mut analytic, _grad_dn) = decoder_loss_and_grad(&decoder, &dn_rates, &calib, &params, &targets);
    analytic.jump_w[0] = 0.0; // sabotage

    let dual_deriv = dual_decoder_loss(
        &z,
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::variable(f64::from(params.jump_w[0])),
        Dual::constant(f64::from(params.jump_b)),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        &targets,
    )
    .deriv;
    assert!(
        dual_deriv.abs() > 1e-3,
        "the dual derivative itself must be meaningfully nonzero, got {dual_deriv}"
    );
    assert_ne!(f64::from(analytic.jump_w[0]), dual_deriv);
}

// --- World model (MSE regression + BCE binary, horizon 0) ----------------------------------------

/// Matches `crate::world_model::world_model_loss_and_grad`'s exact composition for one horizon:
/// `Σ_o (pred_reg[o] - target[o])^2` + `Σ_o BCE(sigmoid(logit_bin[o]), target[o])`, both linear
/// readouts from the (fixed, constant) subset rates `x`.
fn dual_world_model_loss(
    x: &[f64; 3],
    w_reg: &[Dual; NUM_REGRESSION_TARGETS * 3],
    b_reg: &[Dual; NUM_REGRESSION_TARGETS],
    w_bin: &[Dual; NUM_BINARY_TARGETS * 3],
    b_bin: &[Dual; NUM_BINARY_TARGETS],
    target_reg: &[f64; NUM_REGRESSION_TARGETS],
    target_bin: &[bool; NUM_BINARY_TARGETS],
) -> Dual {
    let xc: [Dual; 3] = std::array::from_fn(|j| Dual::constant(x[j]));
    let mut loss = Dual::constant(0.0);
    for o in 0..NUM_REGRESSION_TARGETS {
        let pred = b_reg[o] + (0..3).map(|j| w_reg[o * 3 + j] * xc[j]).sum::<Dual>();
        let diff = pred - Dual::constant(target_reg[o]);
        loss = loss + diff * diff;
    }
    for o in 0..NUM_BINARY_TARGETS {
        let logit = b_bin[o] + (0..3).map(|j| w_bin[o * 3 + j] * xc[j]).sum::<Dual>();
        let p = dsigmoid(logit);
        let y = Dual::constant(f64::from(target_bin[o]));
        loss = loss - (y * dln(p) + (Dual::constant(1.0) - y) * dln(Dual::constant(1.0) - p));
    }
    loss
}

#[test]
fn dual_number_gradients_match_analytic_for_the_world_model_head_on_a_tiny_graph() {
    let flyg = tiny_flyg();
    let model = model_from(flyg);
    let world_model = WorldModelHead::new(&model, &WorldModelConfig::default()).unwrap();
    assert_eq!(world_model.num_subset(), 3, "this fixture's 3 Hidden-role neurons");

    let mut params = world_model.init_default_params();
    for (i, w) in params.horizons[0].w_reg.iter_mut().enumerate() {
        *w = 0.1 * (i as f32 - 4.0);
    }
    for (o, b) in params.horizons[0].b_reg.iter_mut().enumerate() {
        *b = 0.05 * (o as f32 - 4.0);
    }
    for (i, w) in params.horizons[0].w_bin.iter_mut().enumerate() {
        *w = 0.08 * (i as f32 - 4.0);
    }
    params.horizons[0].b_bin = [0.1, -0.1, 0.0];

    let x = [0.3f64, -0.2, 0.5];
    let mut r_full = vec![0.0f32; model.num_neurons()];
    for (i, &xi) in x.iter().enumerate() {
        r_full[i] = xi as f32;
    }
    let target_reg: [f32; NUM_REGRESSION_TARGETS] = [0.1, -0.2, 0.3, -0.1, 0.05, -0.05, 0.2, -0.3];
    let target_bin = [true, false, true];
    let targets: [HorizonTargets; 3] = std::array::from_fn(|k| {
        if k == 0 {
            HorizonTargets {
                regression: Some(target_reg),
                binary: Some(target_bin),
            }
        } else {
            HorizonTargets::default()
        }
    });

    let (_loss, analytic, _grad_r) =
        world_model_loss_and_grad(&world_model, &r_full, model.num_neurons(), &params, &targets);

    let target_reg64: [f64; NUM_REGRESSION_TARGETS] = std::array::from_fn(|o| f64::from(target_reg[o]));
    let w_reg64: Vec<f64> = params.horizons[0].w_reg.iter().map(|&w| f64::from(w)).collect();
    let b_reg64: Vec<f64> = params.horizons[0].b_reg.iter().map(|&b| f64::from(b)).collect();
    let w_bin64: Vec<f64> = params.horizons[0].w_bin.iter().map(|&w| f64::from(w)).collect();
    let b_bin64: Vec<f64> = params.horizons[0].b_bin.iter().map(|&b| f64::from(b)).collect();

    // `slot`: a flat index into the concatenation `[w_reg | b_reg | w_bin | b_bin]`; `eval` marks
    // exactly that one entry `Dual::variable`, everything else `Dual::constant`.
    let n_w_reg = w_reg64.len();
    let n_b_reg = b_reg64.len();
    let n_w_bin = w_bin64.len();
    let n_b_bin = b_bin64.len();
    let eval = |slot: usize| -> Dual {
        let mk = |flat_offset: usize, values: &[f64], take: usize| -> Vec<Dual> {
            (0..take)
                .map(|i| {
                    if flat_offset + i == slot {
                        Dual::variable(values[i])
                    } else {
                        Dual::constant(values[i])
                    }
                })
                .collect()
        };
        let w_reg_d = mk(0, &w_reg64, n_w_reg);
        let b_reg_d = mk(n_w_reg, &b_reg64, n_b_reg);
        let w_bin_d = mk(n_w_reg + n_b_reg, &w_bin64, n_w_bin);
        let b_bin_d = mk(n_w_reg + n_b_reg + n_w_bin, &b_bin64, n_b_bin);
        dual_world_model_loss(
            &x,
            w_reg_d.as_slice().try_into().unwrap(),
            b_reg_d.as_slice().try_into().unwrap(),
            w_bin_d.as_slice().try_into().unwrap(),
            b_bin_d.as_slice().try_into().unwrap(),
            &target_reg64,
            &target_bin,
        )
    };

    for i in 0..n_w_reg {
        assert_close(&format!("w_reg[{i}]"), analytic.horizons[0].w_reg[i], eval(i).deriv);
    }
    for i in 0..n_b_reg {
        assert_close(
            &format!("b_reg[{i}]"),
            analytic.horizons[0].b_reg[i],
            eval(n_w_reg + i).deriv,
        );
    }
    for i in 0..n_w_bin {
        assert_close(
            &format!("w_bin[{i}]"),
            analytic.horizons[0].w_bin[i],
            eval(n_w_reg + n_b_reg + i).deriv,
        );
    }
    for i in 0..n_b_bin {
        assert_close(
            &format!("b_bin[{i}]"),
            analytic.horizons[0].b_bin[i],
            eval(n_w_reg + n_b_reg + n_w_bin + i).deriv,
        );
    }
}
