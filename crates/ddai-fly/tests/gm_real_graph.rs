//! The `Gm` neuron model on the real S and M graphs (task 8.8). Skipped (with a note) when the compiled
//! graphs are not on this machine.
//!
//! - `gm_s_graph_shape_and_activity`: the descriptor keys, the afferent / DN types and the activity of a fresh
//!   network under input (not saturated, not dead), the numbers behind `docs/research/fly-flygm.md` section 3.
//! - `gm_cost_per_decision` (`--ignored`): wall time of one decision of the network alone on S and M,
//!   `cargo test -p ddai-fly --release --test gm_real_graph -- --ignored --nocapture gm_cost`.

use std::path::PathBuf;
use std::time::Instant;

use ddai_fly::config::FlyConfig;
use ddai_fly::gm::{GmConfig, GmDescriptors, GmUpdate};
use ddai_fly::model::FlyModel;
use ddai_fly::params::FlyParams;
use ddai_fly::rng::SplitMix64;
use ddai_fly::state::FlyState;

fn load(name: &str) -> Option<ddai_flyg::Flyg> {
    let path = PathBuf::from(std::env::var("HOME").ok()?).join(format!("aiddnet/data/connectome/compiled/{name}"));
    if !path.exists() {
        eprintln!("skipped: {} is not on this machine", path.display());
        return None;
    }
    Some(ddai_flyg::load(&path).expect("load graph"))
}

fn build(flyg: ddai_flyg::Flyg, gm: GmConfig) -> FlyModel {
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 1);
    FlyModel::new(flyg, config, params)
        .unwrap()
        .with_gm(gm, None, 1)
        .unwrap()
}

/// Inputs shaped like the encoder's: mostly zero with a few active channels of order one.
fn inputs(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            if rng.next_f32_unit() < 0.15 {
                2.0 * rng.next_f32_unit()
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn gm_s_graph_shape_and_activity() {
    let Some(flyg) = load("fly-S-v1.flyg") else { return };
    let n = flyg.neurons.len();
    let with_group = flyg.neurons.iter().filter(|x| x.group_id.is_some()).count();
    eprintln!(
        "S: {n} neurons, {} edges, {with_group} with a group id",
        flyg.edges.num_edges()
    );
    for descriptors in [GmDescriptors::PerType, GmDescriptors::PerNeuronTied] {
        let model = build(
            flyg.clone(),
            GmConfig {
                descriptors,
                ..GmConfig::default()
            },
        );
        let gm = model.gm().unwrap();
        eprintln!(
            "  {descriptors:?}: shape {:?}, {} parameters, {} MACs/step",
            gm.shape(),
            gm.shape().total(),
            gm.macs_per_step()
        );
    }
    // Activity of a fresh network: rms of the state per role and the fraction near the bounds.
    for update in [GmUpdate::Plain, GmUpdate::Gated] {
        let model = build(
            flyg.clone(),
            GmConfig {
                update,
                ..GmConfig::default()
            },
        );
        let mut state = FlyState::new(&model);
        let mut rng = SplitMix64::new(3);
        let mut last = Vec::new();
        for _ in 0..40 {
            let e = inputs(&mut rng, model.num_inputs());
            last = state.step_decision(&model, &e).dn_rates.to_vec();
        }
        let h = state.v();
        let d = 8;
        let rms = |sel: &dyn Fn(usize) -> bool| {
            let mut s = 0.0f64;
            let mut c = 0usize;
            for v in (0..n).filter(|&v| sel(v)) {
                for x in &h[v * d..(v + 1) * d] {
                    s += f64::from(*x) * f64::from(*x);
                    c += 1;
                }
            }
            (s / c.max(1) as f64).sqrt()
        };
        let neurons = &flyg.neurons;
        let role = |r: ddai_flyg::NeuronRole| move |v: usize| neurons[v].role == r;
        let sat = h.iter().filter(|x| x.abs() > 0.95).count() as f64 / h.len() as f64;
        let dead = h.iter().filter(|x| x.abs() < 0.01).count() as f64 / h.len() as f64;
        eprintln!(
            "  {update:?}: rms(H) all {:.3}, input-visual {:.3}, hidden {:.3}, output {:.3}; |H|>0.95: {:.3}, |H|<0.01: {:.3}; dn rate range [{:.3}, {:.3}]",
            rms(&|_| true),
            rms(&role(ddai_flyg::NeuronRole::InputVisual)),
            rms(&role(ddai_flyg::NeuronRole::Hidden)),
            rms(&role(ddai_flyg::NeuronRole::Output)),
            sat,
            dead,
            last.iter().copied().fold(f32::INFINITY, f32::min),
            last.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );
        assert!(
            h.iter().all(|x| x.is_finite() && x.abs() <= 1.0 + 1e-5),
            "the state is bounded by the squash"
        );
        assert!(
            sat < 0.5 && dead < 0.9,
            "a fresh network must be neither saturated nor dead"
        );
    }
}

#[test]
#[ignore = "timing; run with --release --ignored --nocapture"]
fn gm_cost_per_decision() {
    for (name, file) in [("S", "fly-S-v1.flyg"), ("M", "fly-M-v1.flyg")] {
        let Some(flyg) = load(file) else { continue };
        // The rate model for reference.
        {
            let config = FlyConfig::default();
            let params = FlyParams::init_default(&flyg, &config, 1);
            let model = FlyModel::new(flyg.clone(), config, params).unwrap();
            let t = time_decisions(&model);
            eprintln!("{name} rate (4 substeps): p50 {:.0} us, p99 {:.0} us", t.0, t.1);
        }
        for (d, hidden, steps, update) in [
            (8, 8, 1, GmUpdate::Plain),
            (8, 16, 1, GmUpdate::Plain),
            (8, 16, 2, GmUpdate::Plain),
            (8, 16, 2, GmUpdate::Gated),
            (8, 16, 4, GmUpdate::Plain),
            (8, 16, 4, GmUpdate::Gated),
            (16, 32, 2, GmUpdate::Plain),
        ] {
            let cfg = GmConfig {
                d,
                hidden,
                steps,
                update,
                ..GmConfig::default()
            };
            let model = build(flyg.clone(), cfg);
            let macs = model.gm().unwrap().macs_per_step() * steps as usize;
            let t = time_decisions(&model);
            eprintln!(
                "{name} {}: p50 {:.0} us, p99 {:.0} us ({:.2} M MAC/decision, {:.2} GMAC/s at p50)",
                cfg.label(),
                t.0,
                t.1,
                macs as f64 / 1e6,
                macs as f64 / (t.0 * 1e-6) / 1e9
            );
        }
    }
}

fn time_decisions(model: &FlyModel) -> (f64, f64) {
    let mut state = FlyState::new(model);
    let mut rng = SplitMix64::new(9);
    let es: Vec<Vec<f32>> = (0..64).map(|_| inputs(&mut rng, model.num_inputs())).collect();
    for e in es.iter().cycle().take(64) {
        std::hint::black_box(state.step_decision(model, e));
    }
    let mut us = Vec::new();
    for e in es.iter().cycle().take(600) {
        let t0 = Instant::now();
        std::hint::black_box(state.step_decision(model, e));
        us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    us.sort_by(f64::total_cmp);
    (us[us.len() / 2], us[us.len() * 99 / 100])
}
