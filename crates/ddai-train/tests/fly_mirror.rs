//! The fly gets mirror augmentation like the controls (review F4 of E-005): its brain is only
//! approximately mirror-symmetric on the real S graph (`ddai-fly/tests/brain_mirror.rs` compares
//! per-type means to a tolerance), so a mirrored window is new data for it, and "trained
//! identically" (D-014) requires that the controls are not the only ones that see twice the data.
//! Skipped (with a note) when the compiled S graph is not on this machine.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::CharacterObservation;
use ddai_dataset::types::ActionRec;
use ddai_fly::bc::{HeadMask, LossConfig};
use ddai_fly::rng::SplitMix64;
use ddai_physics::map::{MapData, TILE_SOLID, Tile};
use ddai_train::learner::{FlyLearner, FlyTrainConfig, Learner};
use ddai_train::seq::{Corpus, MapEntry, Seq, SeqStep, Source};
use ddai_train::types::char_rec;

fn graph() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").ok()?).join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    p.exists().then_some(p)
}

fn corpus() -> Corpus {
    let (w, h) = (30usize, 20usize);
    let mut game = vec![Tile::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            if y >= 14 || x == 0 || x == w - 1 || y == 0 {
                game[y * w + x] = Tile {
                    index: TILE_SOLID,
                    ..Tile::default()
                };
            }
        }
    }
    let map = MapEntry::new(Arc::new(MapData {
        width: w as u32,
        height: h as u32,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }));
    let mut rng = SplitMix64::new(3);
    let steps = (0..24)
        .map(|t| {
            let x = 200.0 + rng.next_f32_unit() * 300.0;
            let dx = 60.0 + rng.next_f32_unit() * 200.0;
            let mut me = CharacterObservation::at_rest(0);
            me.pos = ddai_physics::vmath::Vec2::new(x, 400.0);
            me.grounded = true;
            let mut opp = CharacterObservation::at_rest(1);
            opp.pos = ddai_physics::vmath::Vec2::new(x + dx, 400.0);
            SeqStep {
                tick: 2 * t,
                me: char_rec(&me),
                others: vec![char_rec(&opp)],
                target: 1,
                label: ActionRec {
                    direction: 1,
                    jump: false,
                    hook: false,
                    fire: false,
                    aim: [100, 0],
                },
                soft: None,
                weight: 1.0,
                mask: HeadMask::ALL,
                latch: false,
            }
        })
        .collect();
    Corpus::new(vec![Seq {
        map,
        steps,
        source: Source::Human { demo: 0 },
    }])
}

#[test]
fn the_fly_learner_mirrors_windows_and_trains_on_them() {
    let Some(flyg) = graph() else {
        eprintln!("note: fly-S-v1.flyg not found, skipping");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let learner = FlyLearner::init(
        &flyg,
        &root.join("configs/fly/S-brain.toml"),
        1,
        FlyTrainConfig::default(),
        &[],
    )
    .unwrap();
    assert!(learner.mirror_augment(), "the fly is mirrored like the controls");

    // A mirrored window goes through the fly's own forward/backward: finite loss, finite gradient,
    // and the mirrored target of a rightward step is leftward.
    let c = corpus();
    let mut rng = SplitMix64::new(11);
    let w = c.sample_window(&mut rng, 12, 2, true);
    assert!(w.mirrored);
    let scored = w.targets.iter().find(|t| t.weight > 0.0).unwrap();
    assert_eq!(scored.dir, 0, "right (2) becomes left (0) in the mirror");
    let mut grad = vec![0.0f32; learner.num_params()];
    let mut ws = learner.new_workspace(12);
    let stats = learner.window_grad(&w, &LossConfig::default(), &mut ws, &mut grad);
    assert!(stats.loss.total.is_finite() && stats.weight_sum > 0.0);
    assert!(grad.iter().all(|g| g.is_finite()));
    assert!(grad.iter().any(|g| *g != 0.0));
}
