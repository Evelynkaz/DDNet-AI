//! Task 7.3, acceptance criterion 9's fair evaluation protocol (review round 1, F2; review round
//! 2, F20, CONFIRMED): trains on one real block map, evaluates across seeds — not a pass/fail
//! assertion (there is no task-given bar for "the fly must beat the MLP"; the task asked for
//! honest numbers, reported in the build report, not a green check hiding a real gap).
//! `#[ignore]`d: needs `~/aiddnet/data/connectome/compiled/fly-S-v1.flyg`,
//! `configs/fly/S-brain.toml`, the six `~/aiddnet/data/maps/copy-love-box/*.map` files, and
//! `~/aiddnet/data/maps/cache/BlmapChill_*.map` (none of which are in the repo — real
//! connectome/map data, per `CLAUDE.md`).
//!
//! ## Review round 2, F20 (CONFIRMED): the six "Copy Love Box" files are not independent maps
//! An earlier revision of this file called the other five Copy Love Box variants "held-out maps
//! it never trained on" without checking what they actually are: they are **revisions of the same
//! base map** (85.7-88.6% identical game tiles against the training file; two of the six are
//! byte-for-byte identical in tile layout) — evaluating on them measures generalization across
//! minor map edits, not across genuinely different geometry. This file now:
//! - prints each Copy Love Box file's identical-game-tile fraction against the training file, so
//!   the similarity is visible, not asserted away;
//! - labels every Copy Love Box evaluation "same base map (revision)", not "held out";
//! - adds `BlmapChill` (a different block map entirely) as the one genuinely unseen map available
//!   locally (`ChillBlock5`/`Blockdale`, which the review suggested too, are not present under
//!   `~/aiddnet/data/maps` on this host — noted, not silently substituted);
//! - reports metrics under **both** [`ddai_fly::demo_brain::sample_scenario`]'s natural,
//!   unstratified distribution and [`ddai_fly::demo_brain::sample_scenario_stratified`]'s `jump`-
//!   stratified one, side by side, for every map — stratifying `jump` (review round 1, F2) shifts
//!   `hook`'s own prevalence too (they are not drawn independently), which inflated `hook`'s
//!   apparent performance in a fly-vs-MLP comparison that only ever looked at the stratified
//!   numbers.
//!
//! Run with `cargo test -p ddai-fly --test brain_demo_generalization -- --ignored --nocapture`.

use std::path::PathBuf;
use std::sync::Arc;

use ddai_brain::Observation;
use ddai_fly::decoder::DecoderModel;
use ddai_fly::demo_brain::{
    BrainDemoConfig, HeadMetrics, TeacherLabels, evaluate_fly, evaluate_mlp, run_brain_demo, sample_scenario,
    sample_scenario_stratified, scripted_teacher,
};
use ddai_fly::encoder::EncoderModel;
use ddai_fly::rng::SplitMix64;
use ddai_fly::{BackwardIndex, FlyConfig, FlyModel, FlyParams, FlyState};

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME must be set"))
}

fn load_real_s() -> ddai_flyg::Flyg {
    let path = home().join("aiddnet/data/connectome/compiled/fly-S-v1.flyg");
    ddai_flyg::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
}

fn load_s_brain_config() -> ddai_fly::brain_config::BrainConfig {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/fly/S-brain.toml");
    ddai_fly::brain_config::load_brain_config(&path).expect("configs/fly/S-brain.toml should parse")
}

fn load_map(path: &std::path::Path) -> Arc<ddai_physics::map::MapData> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let loaded = ddai_map::load_map(&bytes).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
    Arc::new(loaded.data)
}

/// One evaluation map: a name, the loaded data, and whether it is a genuinely different map or a
/// revision of the training file (review round 2, F20 — see the module doc comment).
struct EvalMap {
    name: String,
    map: Arc<ddai_physics::map::MapData>,
    kind: MapKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MapKind {
    /// The exact file trained on this run.
    TrainingMap,
    /// A different file, but the same underlying base map (Copy Love Box's own revisions).
    SameBaseMapRevision,
    /// A genuinely different map (`BlmapChill`).
    GenuinelyDifferentMap,
}

/// The six real "Copy Love Box" revisions (sorted, so "train on the first" is deterministic
/// across runs) plus `BlmapChill` if present locally (review round 2, F20).
fn load_eval_maps() -> Vec<EvalMap> {
    let dir = home().join("aiddnet/data/maps/copy-love-box");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "map"))
        .collect();
    paths.sort();
    assert!(
        paths.len() >= 6,
        "expected at least 6 Copy Love Box map files in {}, found {}",
        dir.display(),
        paths.len()
    );

    let mut maps: Vec<EvalMap> = paths
        .into_iter()
        .enumerate()
        .map(|(i, p)| EvalMap {
            name: p.file_stem().unwrap().to_string_lossy().into_owned(),
            map: load_map(&p),
            kind: if i == 0 {
                MapKind::TrainingMap
            } else {
                MapKind::SameBaseMapRevision
            },
        })
        .collect();

    let cache_dir = home().join("aiddnet/data/maps/cache");
    let blmap_chill = std::fs::read_dir(&cache_dir).ok().and_then(|entries| {
        entries.filter_map(|e| e.ok().map(|e| e.path())).find(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("BlmapChill_"))
        })
    });
    match blmap_chill {
        Some(p) => maps.push(EvalMap {
            name: "BlmapChill".to_string(),
            map: load_map(&p),
            kind: MapKind::GenuinelyDifferentMap,
        }),
        None => eprintln!(
            "warning: BlmapChill not found under {} -- evaluating without a genuinely different \
             held-out map (review round 2, F20 asked for one); ChillBlock5/Blockdale, suggested as \
             alternatives, are also not present locally",
            cache_dir.display()
        ),
    }
    maps
}

/// Fraction of tiles whose `game` layer index matches `train`'s, at the same `(x, y)` — `NaN` if
/// the dimensions differ outright. Printed once per run (review round 2, F20) so the Copy Love Box
/// files' similarity to the training map is visible data, not an assumed "these are independent".
fn identical_game_tile_fraction(m: &ddai_physics::map::MapData, train: &ddai_physics::map::MapData) -> f64 {
    if m.width != train.width || m.height != train.height {
        return f64::NAN;
    }
    let matching = m
        .game
        .iter()
        .zip(&train.game)
        .filter(|(a, b)| a.index == b.index)
        .count();
    matching as f64 / train.game.len() as f64
}

fn print_head_metrics(label: &str, m: &HeadMetrics) {
    println!(
        "    {label}: direction acc={:.3} (majority={:.3}, bal={:.3}) | jump prev={:.3} acc={:.3} (majority={:.3}, bal={:.3}, auroc={:.3}) | hook prev={:.3} acc={:.3} (majority={:.3}, bal={:.3}, auroc={:.3})",
        m.direction.accuracy,
        m.direction.majority_baseline_accuracy,
        m.direction.balanced_accuracy,
        m.jump.prevalence,
        m.jump.accuracy,
        m.jump.majority_baseline_accuracy,
        m.jump.balanced_accuracy,
        m.jump.auroc,
        m.hook.prevalence,
        m.hook.accuracy,
        m.hook.majority_baseline_accuracy,
        m.hook.balanced_accuracy,
        m.hook.auroc,
    );
}

/// Draws `n` samples from `map` under either distribution (review round 2, F20: both are reported
/// side by side for every evaluation map, not just the stratified one).
fn draw_samples(
    map: &Arc<ddai_physics::map::MapData>,
    cfg: &BrainDemoConfig,
    seed: u64,
    n: usize,
    stratified: bool,
) -> (Vec<Observation>, Vec<TeacherLabels>) {
    let mut rng = SplitMix64::new(seed.wrapping_add(0x5EED));
    let mut obs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let (o, l) = if stratified {
            sample_scenario_stratified(Arc::clone(map), cfg, &mut rng)
        } else {
            let o = sample_scenario(Arc::clone(map), cfg, &mut rng);
            let l = scripted_teacher(&o, cfg);
            (o, l)
        };
        obs.push(o);
        labels.push(l);
    }
    (obs, labels)
}

#[test]
#[ignore]
fn fly_vs_mlp_across_seeds_and_maps_stratified_and_natural() {
    let maps = load_eval_maps();
    let brain_cfg = load_s_brain_config();

    const SEEDS: [u64; 3] = [1, 2, 3];

    // Printed once (map similarity doesn't depend on the seed): review round 2, F20's own
    // transparency ask.
    {
        let train = &maps[0].map;
        println!("=== map similarity vs. the training file ({}) ===", maps[0].name);
        for m in &maps {
            let frac = identical_game_tile_fraction(&m.map, train);
            println!(
                "  {:<40} {}x{} identical-game-tiles-vs-train={frac:.4} [{:?}]",
                m.name, m.map.width, m.map.height, m.kind
            );
        }
    }

    for &seed in &SEEDS {
        println!("=== seed {seed} ===");
        let flyg = load_real_s();
        let config = FlyConfig::default();
        let params = FlyParams::init_default(&flyg, &config, seed);
        let mut model = FlyModel::new(flyg, config, params).expect("build FlyModel");
        let index = BackwardIndex::build(&model);

        let encoder =
            EncoderModel::new(&model, brain_cfg.ray_grid, &brain_cfg.proprioception).expect("build EncoderModel");
        let mut encoder_params = ddai_fly::encoder::EncoderParams::init_default(encoder.num_params());

        let decoder = DecoderModel::new(&model, brain_cfg.decoder.clone()).expect("build DecoderModel");
        let mut decoder_params = decoder.init_default_params();

        let calib =
            ddai_fly::calibrate_from_rest(&model, seed, decoder.config().min_sigma).expect("calibration must fit");

        let demo_cfg = BrainDemoConfig {
            seed,
            ..BrainDemoConfig::default()
        };

        let train_map = Arc::clone(&maps[0].map);
        let report = run_brain_demo(
            &mut model,
            &index,
            &encoder,
            &mut encoder_params,
            &decoder,
            &mut decoder_params,
            &calib,
            Arc::clone(&train_map),
            &demo_cfg,
        );

        println!(
            "trained on {} ({} steps, fly_params={}, mlp_params={}, guarded_steps_skipped={})",
            maps[0].name,
            report.metrics.len(),
            report.fly_num_params,
            report.mlp_num_params,
            report.guarded_steps_skipped
        );

        // Warm state to decode from, matching `evaluate_fly`'s own convention.
        let mut warm = FlyState::new(&model);
        let _ = warm.warm_up(&model);
        let v_init = warm.v().to_vec();

        const N: usize = 400;
        for m in &maps {
            let label = match m.kind {
                MapKind::TrainingMap => "TRAINING MAP",
                MapKind::SameBaseMapRevision => "same base map (revision, NOT an independent held-out map)",
                MapKind::GenuinelyDifferentMap => "GENUINELY DIFFERENT MAP (never trained on)",
            };
            println!(" -- {} [{label}], n={N}:", m.name);
            for &stratified in &[true, false] {
                let dist_label = if stratified { "stratified" } else { "natural" };
                let (obs, labels) = draw_samples(&m.map, &demo_cfg, seed, N, stratified);
                let fly_metrics = evaluate_fly(
                    &model,
                    &encoder,
                    &encoder_params,
                    &decoder,
                    &decoder_params,
                    &calib,
                    &v_init,
                    demo_cfg.t_decisions,
                    &obs,
                    &labels,
                );
                let mlp_metrics = evaluate_mlp(&report.mlp, encoder.ray_grid_config(), &obs, &labels);
                println!("  [{dist_label}]");
                print_head_metrics("fly", &fly_metrics);
                print_head_metrics("mlp", &mlp_metrics);
            }
        }
    }
}
