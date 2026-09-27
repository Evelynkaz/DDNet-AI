//! Criterion benchmarks for `step_decision` and `set_params` on the real S/M `.flyg` graphs
//! (acceptance criterion 4). Loads `~/aiddnet/data/connectome/compiled/fly-{S,M}-v1.flyg`; skips
//! (with a message, not a failure) whatever isn't present, since that data lives outside the repo
//! and isn't guaranteed to exist in every environment this crate is built in — see the crate
//! README for how to fetch/build it (`ddai-connectome build-subgraph`).
//!
//! Pins this thread to one core (best-effort, `core_affinity` — a dev-dependency only, see
//! `Cargo.toml`) before benchmarking, per acceptance criterion 4's methodology
//! (`docs/research/rust-stack.md` §4).

use std::hint::black_box;
use std::path::{Path, PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};
use ddai_fly::{FlyConfig, FlyModel, FlyParams, FlyState};

fn pin_this_thread_to_one_core() {
    if let Some(mut ids) = core_affinity::get_core_ids()
        && let Some(id) = ids.pop()
    {
        // Best-effort: a `false` return (unsupported platform/sandbox) just means this run's
        // numbers may carry a bit more host-scheduler noise, not a benchmark failure.
        let _ = core_affinity::set_for_current(id);
    }
}

fn bench_one_graph(c: &mut Criterion, name: &str, path: &Path) {
    if !path.exists() {
        eprintln!(
            "decision bench: skipping {name} ({} not found in this environment)",
            path.display()
        );
        return;
    }
    let flyg = ddai_flyg::load(path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 42);
    let model = FlyModel::new(flyg, config, params.clone()).expect("build FlyModel from a real .flyg");
    let mut state = FlyState::new(&model);
    let warm_up_report = state.warm_up(&model);
    let inputs = vec![0.3f32; model.num_inputs()];

    eprintln!(
        "decision bench: {name}: {} neurons, {} edges, memory {:.1} MiB, warm-up: converged={} in {} decisions ({:.0}ms, final max|dV|={:.4})",
        model.num_neurons(),
        model.flyg().edges.num_edges(),
        model.memory_footprint_bytes() as f64 / (1024.0 * 1024.0),
        warm_up_report.converged,
        warm_up_report.decisions_run,
        warm_up_report.elapsed_ms,
        warm_up_report.final_max_delta_v,
    );

    c.bench_function(&format!("step_decision_{name}"), |b| {
        b.iter(|| {
            let out = state.step_decision(&model, &inputs);
            black_box(out.dn_rates[0]);
        });
    });

    let mut settable = model.clone();
    c.bench_function(&format!("set_params_{name}"), |b| {
        b.iter(|| {
            settable
                .set_params(params.clone())
                .expect("set_params on its own graph must succeed");
        });
    });
}

fn benches(c: &mut Criterion) {
    pin_this_thread_to_one_core();
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let base: PathBuf = Path::new(&home).join("aiddnet/data/connectome/compiled");
    bench_one_graph(c, "S", &base.join("fly-S-v1.flyg"));
    bench_one_graph(c, "M", &base.join("fly-M-v1.flyg"));
}

criterion_group!(fly_benches, benches);
criterion_main!(fly_benches);
