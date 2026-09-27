//! Acceptance criterion 5b: the real S graph learns a synthetic supervised task — "left vs right
//! visual sector active" -> "direction_left vs direction_right DN group high" (FLY.md §6's own
//! example), loss decreasing substantially, held-out patterns generalising, training stable (no
//! NaN, activity not dead). Numbers/curves written as CSV to `~/aiddnet/data/runs/7.2-demo/` (task
//! instructions: reports live under `~/aiddnet/data`, never in the repo).
//!
//! `#[ignore]`d: needs the real compiled `.flyg` (see `tests/stability.rs`'s doc comment for the
//! project's standard convention). Run with:
//! `cargo test -p ddai-fly --release --test train_demo_real_graph -- --ignored --nocapture`

use std::io::Write as _;
use std::path::PathBuf;

use ddai_fly::demo::{DirectionDemoConfig, direction_inputs_outputs_from_flyg, evaluate_direction, run_direction_demo};
use ddai_fly::optim::{AdamConfig, GuardedAdamConfig};
use ddai_fly::{BackwardIndex, FlyConfig, FlyModel, FlyParams, FlyState};

fn compiled_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to find ~/aiddnet/data/connectome/compiled");
    PathBuf::from(home).join("aiddnet/data/connectome/compiled")
}

fn runs_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to find ~/aiddnet/data/runs");
    PathBuf::from(home).join("aiddnet/data/runs/7.2-demo")
}

/// Builds the demo config from the real S graph's own data — see `ddai_fly::demo::
/// direction_inputs_outputs_from_flyg`'s doc comment (shared with the `ddnet-ai fly train-demo`
/// CLI, so the two can't drift apart on what "left"/"right" mean).
fn build_config(model: &FlyModel, seed: u64) -> DirectionDemoConfig {
    let (left_inputs, right_inputs, left_outputs, right_outputs) =
        direction_inputs_outputs_from_flyg(model, "direction_left", "direction_right");

    DirectionDemoConfig {
        left_inputs,
        right_inputs,
        left_outputs,
        right_outputs,
        target_high: 6.0,
        target_low: 1.0,
        activation_prob: 0.5, // a random subset of the active side's neurons each pattern -> many distinct instances
        batch_size: 24,
        t_decisions: 6,
        readout_decisions: 2,
        steps: 300,
        grad_clip_norm: 1.0,
        adam: GuardedAdamConfig {
            adam: AdamConfig {
                lr_a: 2e-2,
                lr_b: 2e-2,
                lr_theta: 2e-2,
                ..AdamConfig::default()
            },
            ..GuardedAdamConfig::default()
        },
        seed,
    }
}

fn write_csv(path: &std::path::Path, metrics: &[ddai_fly::demo::StepMetric]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "step,loss,grad_norm,applied,lr_scale,max_abs_v")?;
    for m in metrics {
        writeln!(
            f,
            "{},{},{},{},{},{}",
            m.step, m.loss, m.grad_norm, m.applied, m.lr_scale, m.max_abs_v
        )?;
    }
    Ok(())
}

#[test]
#[ignore]
fn real_s_graph_learns_direction_task_and_generalises() {
    let path = compiled_dir().join("fly-S-v1.flyg");
    let flyg = ddai_flyg::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));

    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 42);
    let mut model = FlyModel::new(flyg, config, params).expect("build model from real S graph");
    let index = BackwardIndex::build(&model);

    let mut warm = FlyState::new(&model);
    let warm_report = warm.warm_up(&model);
    assert!(warm_report.converged, "warm-up should converge on the real S graph");
    let v_init = warm.v().to_vec();

    let demo = build_config(&model, 20260927);
    eprintln!(
        "real S direction demo: {} left inputs, {} right inputs, {} left_outputs, {} right_outputs, \
         batch={}, T={}, readout={}, steps={}",
        demo.left_inputs.len(),
        demo.right_inputs.len(),
        demo.left_outputs.len(),
        demo.right_outputs.len(),
        demo.batch_size,
        demo.t_decisions,
        demo.readout_decisions,
        demo.steps,
    );

    // Held-out generalisation check *before* training too, on the exact same pattern set `seed`
    // will draw *after* training (so the before/after comparison is apples-to-apples) — a
    // `seed` distinct from `demo.seed` (training's own), so these exact random masks are never
    // trained on either side of the comparison.
    let held_out_seed = 424_242;
    let held_out_patterns = 60;
    let eval_before = evaluate_direction(&model, &demo, &v_init, held_out_seed, held_out_patterns);

    let metrics = run_direction_demo(&mut model, &index, &demo, &v_init);

    assert!(
        metrics.iter().all(|m| m.applied),
        "no step should have hit the NaN guard on the real S graph"
    );
    let max_abs_v_ever = metrics.iter().map(|m| m.max_abs_v).fold(0.0f32, f32::max);
    assert!(
        max_abs_v_ever.is_finite(),
        "activity exploded to a non-finite value during training"
    );
    // r_max=10 by default; a healthy network's V rarely needs to go far beyond a handful of
    // r_max's worth to saturate f(V) — this is a loose "still alive, not diverging" bound, not a
    // tight one (see the crate README's stability tables for the real graphs' typical |V| range).
    assert!(
        max_abs_v_ever < 200.0,
        "|V| grew implausibly large during training: {max_abs_v_ever}"
    );

    let initial_loss = metrics[0].loss;
    let final_loss: f32 = metrics[metrics.len() - 10..].iter().map(|m| m.loss).sum::<f32>() / 10.0;
    eprintln!("real S direction demo: initial loss={initial_loss:.4}, final loss (last 10 avg)={final_loss:.4}");
    assert!(
        final_loss < initial_loss * 0.5,
        "loss should decrease substantially: initial={initial_loss}, final={final_loss}"
    );

    let eval_after = evaluate_direction(&model, &demo, &v_init, held_out_seed, held_out_patterns);
    eprintln!(
        "real S direction demo: held-out accuracy before={:.3} after={:.3}; mean margin before={:.4} after={:.4}",
        eval_before.accuracy, eval_after.accuracy, eval_before.mean_margin, eval_after.mean_margin
    );
    // `accuracy` alone can already sit at 1.0 before training (the connectome's own wiring is
    // somewhat lateralised by anatomy — see `evaluate_direction`'s doc comment), so the sharper,
    // non-ceilinged signal that training actually helped is the *margin* growing, not accuracy
    // (checked too, but with `>=` rather than `>` for exactly that reason).
    assert!(
        eval_after.mean_margin > eval_before.mean_margin,
        "training should increase the held-out margin: before={} after={}",
        eval_before.mean_margin,
        eval_after.mean_margin
    );
    assert!(
        eval_after.accuracy >= eval_before.accuracy,
        "training should not make held-out accuracy worse: before={} after={}",
        eval_before.accuracy,
        eval_after.accuracy
    );
    assert!(
        eval_after.accuracy >= 0.8,
        "held-out accuracy should be well above chance (0.5) after training: {}",
        eval_after.accuracy
    );

    // Not-dead check (same spirit as `tests/stability.rs`): some neurons should actually be firing
    // after training, not just the constrained output groups.
    let mut probe = FlyState::new(&model);
    probe.set_v(&model, &v_init);
    let out = probe.step_decision(&model, &vec![0.5f32; model.num_inputs()]);
    let fraction_active = out.per_type_mean_rate.iter().filter(|&&r| r > 0.0).count() as f32 / model.num_types() as f32;
    eprintln!("real S direction demo: fraction of types with r>0 after training = {fraction_active:.3}");
    assert!(
        fraction_active > 0.5,
        "network should not have collapsed to near-total silence: {fraction_active}"
    );

    let dir = runs_dir();
    std::fs::create_dir_all(&dir).expect("create ~/aiddnet/data/runs/7.2-demo/");
    let csv_path = dir.join("s-graph-direction-demo.csv");
    write_csv(&csv_path, &metrics).expect("write demo CSV");
    eprintln!(
        "real S direction demo: wrote loss/grad_norm curve to {}",
        csv_path.display()
    );
}
