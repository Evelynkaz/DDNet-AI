//! Task 3.15 (E-028): what one prediction call costs (wall time, release build): the whole `WindowModel::predict` (frame features, rays, input
//! assembly, the network, decoding) and the network forward alone, for the production size or the sizes given.
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example opp_bench -- [h1 h2] [--model file]
//! ```

use std::sync::Arc;
use std::time::Instant;

use ddai_oppnet::feature::{INPUT_DIM, OUT_DIM};
use ddai_oppnet::net::{Mlp, Scratch};
use ddai_oppnet::{OppBundle, OppPredictor};
use ddai_physics::core::PlayerInput as Wire;
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};
use ddai_planner::hybrid::window::{WindowCtx, WindowModel};
use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::vmath::Vec2;

fn hall() -> Arc<MapData> {
    let (w, h) = (40usize, 16usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            let solid = y >= 10 || x == 0 || x == w - 1 || y == 0;
            let freeze = y == 1 && (10..=20).contains(&x);
            game[y * w + x] = Tile {
                index: if freeze {
                    TILE_FREEZE
                } else if solid {
                    TILE_SOLID
                } else {
                    0
                },
                ..Tile::default()
            };
        }
    }
    Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    })
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_file = args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1));
    let nums: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let (h1, h2) = (
        nums.first().copied().unwrap_or(160),
        nums.get(1).copied().unwrap_or(128),
    );
    let bundle = match model_file {
        Some(f) => OppBundle::load(std::path::Path::new(f)).expect("model"),
        None => OppBundle::new(Mlp::new(INPUT_DIM, h1, h2, OUT_DIM, 1), 1, 0, 0.0, "bench".into()),
    };
    let net = bundle.net.clone();
    println!(
        "network {} -> {} -> {} -> {}: {} parameters",
        net.n_in,
        net.h1,
        net.h2,
        net.n_out,
        net.params.len()
    );
    let mut p = OppPredictor::new(bundle, "bench").expect("predictor");
    let mut pw = PhysicsWorld::new(hall(), 1);
    for i in 0..2 {
        pw.add_tee(
            i,
            Vec2 {
                x: (17.5 + 4.0 * f64::from(i)) * 32.0,
                y: 9.5 * 32.0,
            },
        );
    }
    for _ in 0..4 {
        pw.step();
    }
    let w = pw.inner();
    let inflight = vec![Wire::default(); 3];
    let mut out = vec![None; 3];
    for _ in 0..2000 {
        p.predict(
            &WindowCtx {
                world: w,
                self_id: 0,
                victim_id: 1,
                in_flight: &inflight,
            },
            &mut out,
        );
    }
    let mut us = Vec::new();
    for _ in 0..5000 {
        let t = Instant::now();
        p.predict(
            &WindowCtx {
                world: w,
                self_id: 0,
                victim_id: 1,
                in_flight: &inflight,
            },
            &mut out,
        );
        us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    println!(
        "predict (whole call): min {:.1} us, median {:.1} us, p99 {:.1} us",
        pct(&mut us.clone(), 0.0),
        pct(&mut us.clone(), 0.5),
        pct(&mut us, 0.99)
    );
    let x = [0.3f32; INPUT_DIM];
    let mut s = Scratch::new(&net);
    let mut fw = Vec::new();
    for i in 0..7000 {
        let t = Instant::now();
        net.forward(&x, &mut s);
        if i >= 2000 {
            fw.push(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    println!(
        "forward, dense input: min {:.1} us, median {:.1} us",
        pct(&mut fw.clone(), 0.0),
        pct(&mut fw, 0.5)
    );
}
