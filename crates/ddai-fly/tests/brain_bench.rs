//! Acceptance criterion 7: "decision latency on S (encode + 4 substeps + decode): report p50/
//! p99; target p99 <= 2ms on one core." `#[ignore]`d: needs the real `fly-S-v1.flyg` and
//! `configs/fly/S-brain.toml`.

use std::sync::Arc;
use std::time::Instant;

use ddai_brain::{Brain, CharacterObservation, Observation};
use ddai_fly::brain::{ActionSelection, FlyBrain, FlyBrainConfig};
use ddai_fly::decoder::{DecoderModel, DnCalibration};
use ddai_fly::encoder::EncoderModel;
use ddai_fly::model::FlyModel;
use ddai_fly::params::FlyParams;

fn home() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").expect("HOME must be set"))
}

fn tiny_open_map() -> ddai_physics::map::MapData {
    ddai_physics::map::MapData {
        width: 60,
        height: 60,
        game: vec![Default::default(); 3600],
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

fn sample_observation(rng: &mut ddai_fly::rng::SplitMix64, map: &Arc<ddai_physics::map::MapData>) -> Observation {
    let mut me = CharacterObservation::at_rest(0);
    me.pos = ddai_physics::vmath::Vec2::new(600.0 + rng.next_f32_unit() * 200.0, 600.0 + rng.next_f32_unit() * 200.0);
    me.vel = ddai_physics::vmath::Vec2::new(rng.next_f32_unit() * 200.0 - 100.0, rng.next_f32_unit() * 100.0 - 50.0);
    let mut opp = CharacterObservation::at_rest(1);
    opp.pos = ddai_physics::vmath::Vec2::new(600.0 + rng.next_f32_unit() * 400.0, 600.0 + rng.next_f32_unit() * 400.0);
    Observation {
        map: Arc::clone(map),
        tick: 0,
        self_state: me,
        others: vec![opp],
        target_id: None,
        tuning: ddai_physics::tuning::TuningParams::default(),
    }
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    let idx = ((p / 100.0) * (sorted_ms.len() - 1) as f64).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

#[test]
#[ignore]
fn decide_latency_on_the_real_s_graph() {
    let flyg_path = home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    let flyg = ddai_flyg::load(&flyg_path).unwrap_or_else(|e| panic!("failed to load {}: {e}", flyg_path.display()));
    let brain_cfg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    let brain_cfg =
        ddai_fly::brain_config::load_brain_config(&brain_cfg_path).expect("configs/fly/S-brain.toml should parse");

    let config = ddai_fly::FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 5);
    let model = FlyModel::new(flyg, config, params).expect("build FlyModel");
    let encoder = EncoderModel::new(&model, brain_cfg.ray_grid, &brain_cfg.proprioception).expect("build EncoderModel");
    let encoder_params = ddai_fly::encoder::EncoderParams::init_default(encoder.num_params());
    let decoder = DecoderModel::new(&model, brain_cfg.decoder).expect("build DecoderModel");
    let decoder_params = decoder.init_default_params();
    let calib = DnCalibration {
        mu: vec![0.0; model.num_outputs()],
        sigma: vec![1.0; model.num_outputs()],
    };

    let mut brain = FlyBrain::new(
        model,
        encoder,
        encoder_params,
        decoder,
        decoder_params,
        calib,
        FlyBrainConfig {
            action_selection: ActionSelection::Argmax,
            seed: 5,
        },
    );
    brain.reset(&ddai_brain::ResetContext {
        map: Arc::new(tiny_open_map()),
        self_id: 0,
        seed: 5,
    });

    let map = Arc::new(tiny_open_map());
    let mut rng = ddai_fly::rng::SplitMix64::new(9);
    let observations: Vec<Observation> = (0..1000).map(|_| sample_observation(&mut rng, &map)).collect();

    // Warm-up calls (allowed to be slow: lazy init, cache warming), not measured.
    for obs in observations.iter().take(10) {
        let _ = brain.decide(obs);
    }

    let mut latencies_ms = Vec::with_capacity(observations.len());
    for obs in &observations {
        let start = Instant::now();
        let action = brain.decide(obs);
        let elapsed = start.elapsed();
        std::hint::black_box(action);
        latencies_ms.push(elapsed.as_secs_f64() * 1000.0);
    }
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = percentile(&latencies_ms, 50.0);
    let p99 = percentile(&latencies_ms, 99.0);
    let max = *latencies_ms.last().unwrap();
    let over_2ms = latencies_ms.iter().filter(|&&x| x > 2.0).count();

    eprintln!(
        "FlyBrain::decide latency on S ({} decisions): median={median:.3}ms p99={p99:.3}ms max={max:.3}ms over_2ms={over_2ms}/{}",
        observations.len(),
        observations.len()
    );
}
