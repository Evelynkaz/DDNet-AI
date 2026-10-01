//! Task 7.2b throughput harness (`#[ignore]`d: needs the compiled real graphs). Measures
//! sample-steps/s of the per-sequence backend (`train_step`) and the batched one
//! (`train_step_batched`) on the real S/M graphs. One "sample-step" is one decision of one
//! sequence (4 substeps), forward + backward -- the unit `train_throughput.rs` (task 7.2) used.
//!
//! ```text
//! FLY_BENCH_GRAPHS=S,M FLY_BENCH_THREADS=1,8 FLY_BENCH_BATCHES=32,64,128 FLY_BENCH_BACKENDS=batched \
//!   cargo test -p ddai-fly --release --test batched_throughput -- --ignored --nocapture
//! ```
//! Also `FLY_BENCH_T` (decisions per sequence, default 32) and `FLY_BENCH_REPEATS` (default 3).
//! The load average is printed right before every timed run, and peak RSS is the high-water mark
//! since the previous measurement (reset through `/proc/self/clear_refs`).

use std::path::PathBuf;
use std::time::Instant;

use ddai_fly::batched::{BatchedEngine, BatchedForwardOptions, BatchedSeqGrad, BatchedSeqInput};
use ddai_fly::rng::SplitMix64;
use ddai_fly::{BackwardIndex, FlyConfig, FlyModel, FlyParams, FlyState, Sequence, train_step};

fn compiled_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME")).join("aiddnet/data/connectome/compiled")
}

fn load_avg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| "n/a".into())
}

/// User + system CPU seconds of this process (`/proc/self/stat`, 100 Hz ticks): on a shared
/// machine the wall clock of a run includes time waiting for a CPU, this does not.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // The command name (field 2) may contain spaces; everything after the last ')' is plain.
    let rest = stat.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    // After ')': state is f[0] (field 3), utime is field 14 -> f[11], stime field 15 -> f[12].
    let ticks = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn peak_rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|r| r.split_whitespace().next()?.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn env_list<T: std::str::FromStr>(name: &str, default: &str) -> Vec<T> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

fn build_sequences(model: &FlyModel, rng: &mut SplitMix64, batch: usize, t: usize) -> Vec<Sequence> {
    let (k_in, k_out, n) = (model.num_inputs(), model.num_outputs(), model.num_neurons());
    (0..batch)
        .map(|_| Sequence {
            v_init: (0..n).map(|_| rng.next_f32_unit() * 0.4 - 0.2).collect(),
            inputs: (0..t)
                .map(|_| (0..k_in).map(|_| rng.next_f32_unit() * 0.6 - 0.3).collect())
                .collect(),
            grad_dn: (0..t)
                .map(|_| (0..k_out).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
                .collect(),
            extra_taps: vec![],
        })
        .collect()
}

#[test]
#[ignore]
fn throughput_real_graphs() {
    let graphs: Vec<String> = env_list("FLY_BENCH_GRAPHS", "S,M");
    let threads: Vec<usize> = env_list("FLY_BENCH_THREADS", "1,8");
    let batches: Vec<usize> = env_list("FLY_BENCH_BATCHES", "32,64,128");
    let backends: Vec<String> = env_list("FLY_BENCH_BACKENDS", "batched");
    let t: usize = env_list("FLY_BENCH_T", "32")[0];
    let repeats: usize = env_list("FLY_BENCH_REPEATS", "3")[0];
    for g in &graphs {
        let path = compiled_dir().join(format!("fly-{g}-v1.flyg"));
        if !path.exists() {
            eprintln!("skipping {g}: {} not found", path.display());
            continue;
        }
        let flyg = ddai_flyg::load(&path).unwrap();
        {
            let config = FlyConfig::default();
            let params = FlyParams::init_default(&flyg, &config, 42);
            let model = FlyModel::new(flyg, config, params).unwrap();
            let index = BackwardIndex::build(&model);
            let mut warm = FlyState::new(&model);
            let _ = warm.warm_up(&model);
            let s = model.config().substeps_per_decision as usize;
            eprintln!(
                "\n=== {g}: {} neurons, {} edges, T={t}, S={s} ===",
                model.num_neurons(),
                model.flyg().edges.num_edges()
            );
            let mut engine = BatchedEngine::new(&model);
            eprintln!(
                "plan: {} chunks, {:.1} MiB",
                engine.plan().num_chunks(),
                engine.plan().memory_bytes() as f64 / 1048576.0
            );
            for &b in &batches {
                let mut rng = SplitMix64::new(20261001 + b as u64);
                let seqs = build_sequences(&model, &mut rng, b, t);
                for &nt in &threads {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
                    for backend in &backends {
                        // warm-up (first touch, caches) -- also sizes the engine's buffers
                        let run_once = |engine: &mut BatchedEngine, timing: &mut (f64, f64)| match backend.as_str() {
                            "batched" => {
                                let ins: Vec<BatchedSeqInput<'_>> = seqs
                                    .iter()
                                    .map(|q| BatchedSeqInput {
                                        v_init: &q.v_init,
                                        inputs: &q.inputs,
                                    })
                                    .collect();
                                let gr: Vec<BatchedSeqGrad<'_>> = seqs
                                    .iter()
                                    .map(|q| BatchedSeqGrad {
                                        grad_dn: &q.grad_dn,
                                        extra_taps: &q.extra_taps,
                                    })
                                    .collect();
                                let t0 = Instant::now();
                                engine.forward(&model, &ins, &BatchedForwardOptions::default()).unwrap();
                                let t1 = Instant::now();
                                let _ = engine.backward(&model, &gr, false);
                                let t2 = Instant::now();
                                timing.0 += (t1 - t0).as_secs_f64();
                                timing.1 += (t2 - t1).as_secs_f64();
                            }
                            "perseq" => {
                                let t0 = Instant::now();
                                let _ = train_step(&model, &index, &seqs, None).unwrap();
                                timing.0 += t0.elapsed().as_secs_f64();
                            }
                            other => panic!("unknown backend {other}"),
                        };
                        pool.install(|| {
                            let mut warm_timing = (0.0, 0.0);
                            run_once(&mut engine, &mut warm_timing);
                        });
                        reset_peak_rss();
                        let load = load_avg();
                        let mut timing = (0.0, 0.0);
                        let t0 = Instant::now();
                        let cpu0 = cpu_seconds();
                        let mut best_step = f64::MAX;
                        pool.install(|| {
                            for _ in 0..repeats {
                                let before = timing.0 + timing.1;
                                run_once(&mut engine, &mut timing);
                                best_step = best_step.min(timing.0 + timing.1 - before);
                            }
                        });
                        let el = t0.elapsed().as_secs_f64();
                        let cpu = cpu_seconds() - cpu0;
                        let samples = (b * t * repeats) as f64;
                        eprintln!(
                            "{g} {backend:<8} B={b:<4} threads={nt:<2} load={load:<17} step={:>8.1} ms (fwd {:>7.1} bwd {:>7.1}; best {:>7.1}; cpu {:>8.1}) \
                         sample-steps/s={:>8.0} (best {:>8.0}) substep-samples/s={:>8.0} peakRSS={:>7.1} MiB",
                            el / repeats as f64 * 1e3,
                            timing.0 / repeats as f64 * 1e3,
                            timing.1 / repeats as f64 * 1e3,
                            best_step * 1e3,
                            cpu / repeats as f64 * 1e3,
                            samples / el,
                            (b * t) as f64 / best_step,
                            samples * s as f64 / el,
                            peak_rss_mib(),
                        );
                    }
                }
            }
        }
    }
}
