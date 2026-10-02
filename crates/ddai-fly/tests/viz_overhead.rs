//! Task 7.4, acceptance criterion 1: "zero allocations and <= 1% decision-time overhead when enabled with no subscriber. Measure
//! it." `#[ignore]`d (a timing run, needs the real `fly-S-v1.flyg` and `configs/fly/S-brain.toml`); run it with
//!
//! ```text
//! cargo test --release -p ddai-fly --test viz_overhead -- --ignored --nocapture
//! ```
//!
//! The zero-allocation half is `tests/no_alloc_brain.rs` (an exact count, not a timing). This measures wall time per
//! `FlyBrain::decide` in four modes on four identical flies fed the same observations: every decision is timed once on each,
//! one right after the other in a rotating order, so whatever the shared machine is doing at that moment hits all four alike
//! (block-wise runs on a loaded machine drift by more than the effect). The modes:
//!
//! * `off` — `decide` alone;
//! * `off (again)` — the same, a second time: the spread between the two is the noise floor of the measurement;
//! * `enabled, nobody watching` — the loop the bot's driver runs with the stream compiled in: `decide`, then
//!   `if watched { viz_frame }` with `watched == false` (what `Bridge::fly_wanted` returns with no subscriber);
//! * `watched` — `decide` plus a frame pulled after every decision (every = 1, the worst case; the bot's default builds one
//!   per two decisions).

use std::sync::Arc;
use std::time::{Duration, Instant};

use ddai_brain::{Brain, CharacterObservation, Observation};
use ddai_fly::brain::{ActionSelection, FlyBrain, FlyBrainConfig};
use ddai_fly::decoder::{DecoderModel, DnCalibration};
use ddai_fly::encoder::EncoderModel;
use ddai_fly::model::FlyModel;
use ddai_fly::params::FlyParams;

fn open_map() -> ddai_physics::map::MapData {
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

fn observations(n: usize) -> Vec<Observation> {
    let map = Arc::new(open_map());
    let mut rng = ddai_fly::rng::SplitMix64::new(9);
    (0..n)
        .map(|_| {
            let mut me = CharacterObservation::at_rest(0);
            me.pos = ddai_physics::vmath::Vec2::new(
                600.0 + rng.next_f32_unit() * 200.0,
                600.0 + rng.next_f32_unit() * 200.0,
            );
            me.vel =
                ddai_physics::vmath::Vec2::new(rng.next_f32_unit() * 200.0 - 100.0, rng.next_f32_unit() * 100.0 - 50.0);
            let mut opp = CharacterObservation::at_rest(1);
            opp.pos = ddai_physics::vmath::Vec2::new(
                600.0 + rng.next_f32_unit() * 400.0,
                600.0 + rng.next_f32_unit() * 400.0,
            );
            Observation {
                map: Arc::clone(&map),
                tick: 0,
                self_state: me,
                others: vec![opp],
                target_id: None,
                tuning: ddai_physics::tuning::TuningParams::default(),
            }
        })
        .collect()
}

fn real_brain() -> FlyBrain {
    let home = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"));
    let flyg = ddai_flyg::load(&home.join("aiddnet/data/connectome/compiled/fly-S-v1.flyg")).expect("the S graph");
    let cfg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    let cfg = ddai_fly::brain_config::load_brain_config(&cfg_path).expect("S-brain.toml");
    let config = ddai_fly::FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 5);
    let model = FlyModel::new(flyg, config, params).unwrap();
    let encoder = EncoderModel::new(&model, cfg.ray_grid, &cfg.proprioception).unwrap();
    let encoder_params = ddai_fly::encoder::EncoderParams::init_default(encoder.num_params());
    let decoder = DecoderModel::new(&model, cfg.decoder).unwrap();
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
        map: Arc::new(open_map()),
        self_id: 0,
        seed: 5,
    });
    brain.set_viz_every(1);
    brain
}

/// One decision of `brain` in `mode`, timed (0 and 1: `decide` alone; 2: the bot driver's pattern with nobody watching;
/// 3: a frame pulled after every decision).
fn one(brain: &mut FlyBrain, o: &Observation, mode: usize, watched: bool) -> Duration {
    let t = Instant::now();
    std::hint::black_box(brain.decide(o));
    match mode {
        2 => {
            // The bot driver's pattern, run for what it costs when nobody watches.
            if std::hint::black_box(watched) {
                std::hint::black_box(brain.viz_frame(0));
            }
        }
        3 => {
            std::hint::black_box(brain.viz_frame(0));
        }
        _ => {}
    }
    t.elapsed()
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[test]
#[ignore = "timing run on the real S graph; see the module docs"]
fn viz_overhead_on_the_real_s_graph() {
    // Four identical flies fed the same observations. Every decision is timed once on each of them, one right after the
    // other and in a rotating order, so whatever the shared machine is doing at that moment hits all four alike.
    let mut brains: Vec<FlyBrain> = (0..4).map(|_| real_brain()).collect();
    let obs = observations(400);
    for o in &obs {
        for b in &mut brains {
            let _ = b.decide(o); // warm up
        }
    }
    let names = [
        "off",
        "off (again)",
        "enabled, nobody watching",
        "watched (every decision)",
    ];
    let mut all: Vec<Vec<Duration>> = vec![Vec::new(); 4];
    for round in 0..30 {
        for (i, o) in obs.iter().enumerate() {
            for k in 0..4 {
                let mode = (round + i + k) % 4;
                all[mode].push(one(&mut brains[mode], o, mode, false));
            }
        }
    }
    let n = all[0].len();
    let base = median(&mut all[0].clone()).as_secs_f64() * 1e6;
    println!("FlyBrain::decide on S, {n} decisions per mode, interleaved ({base:.0} us median in `off`):");
    for k in 0..4 {
        let med = median(&mut all[k].clone()).as_secs_f64() * 1e6;
        let mean = all[k].iter().map(|d| d.as_secs_f64()).sum::<f64>() / n as f64 * 1e6;
        let mean0 = all[0].iter().map(|d| d.as_secs_f64()).sum::<f64>() / n as f64 * 1e6;
        // Paired: the same decision (same index) on each fly; the median of the per-decision ratios.
        let mut ratios: Vec<f64> = (0..n)
            .map(|j| all[k][j].as_secs_f64() / all[0][j].as_secs_f64())
            .collect();
        ratios.sort_by(f64::total_cmp);
        println!(
            "  {:<28} median {med:7.1} us ({:+.2}%)  mean {mean:7.1} us ({:+.2}%)  median paired ratio {:+.2}%",
            names[k],
            (med / base - 1.0) * 100.0,
            (mean / mean0 - 1.0) * 100.0,
            (ratios[n / 2] - 1.0) * 100.0,
        );
    }
}
