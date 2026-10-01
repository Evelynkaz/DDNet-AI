//! The shipped experiment and collection configs parse, and mean what E-005 says they mean.

use std::path::PathBuf;

use ddai_train::experiment::CollectConfig;
use ddai_train::runner::ExperimentConfig;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/train")
}

fn experiment(name: &str) -> ExperimentConfig {
    let text = std::fs::read_to_string(dir().join(format!("{name}.toml"))).unwrap();
    toml::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn every_e005_config_parses_and_the_models_share_data_and_schedule() {
    let names = [
        "e005-fly",
        "e005-fly-noclb",
        "e005-mlp-s",
        "e005-mlp-w",
        "e005-gru-s",
        "e005-gru-w",
        "e005-mlp-w-noclb",
    ];
    let cfgs: Vec<ExperimentConfig> = names.iter().map(|n| experiment(n)).collect();
    let fly = &cfgs[0];
    for (n, c) in names.iter().zip(&cfgs) {
        assert_eq!(c.name, *n);
        // "Trained identically": the same data, schedule, batch shape and DAgger plan.
        assert_eq!(
            (c.bc_steps, c.dagger.steps_per_round),
            (fly.bc_steps, fly.dagger.steps_per_round)
        );
        assert_eq!(c.teacher_base, fly.teacher_base);
        assert_eq!(c.train, fly.train, "{n}");
        assert_eq!(c.dagger.betas, fly.dagger.betas);
        assert_eq!(c.dagger.jobs, fly.dagger.jobs);
        assert!(c.dagger.betas.len() >= 3, "at least three DAgger rounds");
        assert_eq!(c.human.as_ref().unwrap().dataset_dirs.len(), 2, "both human datasets");
        // ChillBlock5 is the clean holdout: never trained on, in any variant.
        assert!(
            c.human
                .as_ref()
                .unwrap()
                .exclude_map_names
                .iter()
                .any(|m| m == "ChillBlock5")
        );
    }
    // The clean variants differ in exactly one thing: no Copy-Love-Box-family human play.
    for (with, without) in [(0, 1), (3, 6)] {
        assert!(cfgs[with].human.as_ref().unwrap().exclude_map_substrings.is_empty());
        assert_eq!(
            cfgs[without].human.as_ref().unwrap().exclude_map_substrings,
            vec!["Copy".to_string()]
        );
    }
}

#[test]
fn the_round0_collection_config_parses_and_labels_the_holdout_arenas_too() {
    let text = std::fs::read_to_string(dir().join("round0-v1.toml")).unwrap();
    let c: CollectConfig = toml::from_str(&text).unwrap();
    assert_eq!(c.actor, "teacher");
    let arenas: std::collections::BTreeSet<&str> = c.jobs.iter().map(|j| j.arena.as_str()).collect();
    for a in ["clb-left", "pit", "platform", "clb-right", "chillblock5-ruler"] {
        assert!(arenas.contains(a), "{a}");
    }
    let seeds: std::collections::BTreeSet<u64> = c.jobs.iter().map(|j| j.base_seed).collect();
    assert_eq!(seeds.len(), c.jobs.len(), "every job has its own seed range");
    assert!(c.jobs.iter().any(|j| j.opponents.len() == 3), "1v3 data");
}
