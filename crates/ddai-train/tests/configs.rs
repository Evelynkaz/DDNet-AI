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

/// E-008: the generated configs parse; arms of one model differ only in what they are meant to differ in; the
/// clean holdout stays excluded everywhere but in the one ablation that says so; nothing asks for more than 3
/// threads; no DAgger round plays without the teacher (beta > 0 in every round).
#[test]
fn every_e008_config_parses_and_the_arms_of_a_model_differ_only_in_their_arm() {
    let mut names: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("e008-") && n.ends_with(".toml"))
        .map(|n| n.trim_end_matches(".toml").to_string())
        .collect();
    names.sort();
    assert!(names.len() >= 12, "{names:?}");
    let cfgs: Vec<(String, ExperimentConfig)> = names.iter().map(|n| (n.clone(), experiment(n))).collect();
    for (n, c) in &cfgs {
        assert_eq!(&c.name, n);
        assert!(c.train.threads <= 3, "{n}");
        assert!(c.dagger.betas.iter().all(|b| *b > 0.0), "{n}: beta > 0 in every round");
        assert!(c.dagger.betas.len() >= 3, "{n}");
        assert!(
            c.run_dir.contains("E-008") && c.teacher_dagger.contains(n.as_str()),
            "{n}"
        );
        let excludes_cb5 = c
            .human
            .as_ref()
            .unwrap()
            .exclude_map_names
            .iter()
            .any(|m| m == "ChillBlock5");
        assert_eq!(
            excludes_cb5,
            !n.contains("cb5"),
            "{n}: only a cb5 ablation may train on ChillBlock5 human data"
        );
        // Checkpoints are selected on training arenas only.
        for a in &c.dagger.eval_arenas {
            assert!(
                a != "chillblock5-ruler" && a != "clb-right",
                "{n}: holdout arena {a} in the in-run evaluation"
            );
        }
    }
    // Phase 1: the arms of one model share data, schedule and seeds' meaning; only the own-hook arm differs.
    let p1: Vec<&(String, ExperimentConfig)> = cfgs.iter().filter(|(n, _)| n.starts_with("e008-p1-")).collect();
    for model in ["mlpw", "fly"] {
        let of: Vec<_> = p1.iter().filter(|(n, _)| n.contains(&format!("-{model}-"))).collect();
        assert!(of.len() >= 6, "{model}: at least 3 arms x 2 seeds");
        let base = &of[0].1;
        for (n, c) in &of {
            assert_eq!(c.teacher_base, base.teacher_base, "{n}");
            assert_eq!(c.human, base.human, "{n}");
            assert_eq!(c.dagger.betas, base.dagger.betas, "{n}");
            assert_eq!(c.dagger.jobs, base.dagger.jobs, "{n}");
            assert_eq!(
                (c.bc_steps, c.dagger.steps_per_round),
                (base.bc_steps, base.dagger.steps_per_round),
                "{n}"
            );
            let (mut a, mut b) = (c.train.clone(), base.train.clone());
            (a.own_hook, b.own_hook, a.seed, b.seed) = (Default::default(), Default::default(), 0, 0);
            assert_eq!(a, b, "{n}: only own_hook and the seed may differ");
        }
        let arms: std::collections::BTreeSet<String> = of
            .iter()
            .map(|(_, c)| format!("{:?}/{}", c.train.own_hook.mode, c.train.own_hook.switch_weight))
            .collect();
        assert!(arms.len() >= 3, "{model}: {arms:?}");
        for arm in ["base", "drop", "maskhook"] {
            let seeds: std::collections::BTreeSet<u64> = of
                .iter()
                .filter(|(n, _)| n.contains(&format!("-{arm}-s")))
                .map(|(_, c)| c.train.seed)
                .collect();
            assert!(seeds.len() >= 2, "{model} {arm}: at least two seeds: {seeds:?}");
        }
    }
}

#[test]
fn the_round1_collection_config_adds_1vn_and_solved_technique_scenarios_on_training_arenas_only() {
    let text = std::fs::read_to_string(dir().join("round1-v2.toml")).unwrap();
    let c: CollectConfig = toml::from_str(&text).unwrap();
    assert_eq!(c.actor, "teacher");
    assert!(c.scenarios_dir.is_some());
    let holdout = ["clb-right", "chillblock5-ruler"];
    assert!(
        c.jobs.iter().all(|j| !holdout.contains(&j.arena.as_str())),
        "no holdout arena is collected"
    );
    let scen: std::collections::BTreeSet<&str> = c.jobs.iter().filter_map(|j| j.scenario.as_deref()).collect();
    for t in ["T1", "T3", "T10", "T14"] {
        assert!(scen.contains(t), "{t}");
    }
    assert!(
        c.jobs.iter().filter(|j| j.scenario.is_some()).all(|j| j.only_success),
        "technique demonstrations are solved trials"
    );
    assert!(
        c.jobs.iter().any(|j| j.opponents.len() == 2) && c.jobs.iter().any(|j| j.opponents.len() == 3),
        "1v2 and 1v3"
    );
    let seeds: std::collections::BTreeSet<u64> = c.jobs.iter().map(|j| j.base_seed).collect();
    assert_eq!(seeds.len(), c.jobs.len(), "every job has its own seed range");
}

#[test]
fn the_batched_m_smoke_config_parses_and_the_e005_configs_keep_the_per_sequence_backend() {
    use ddai_fly::TrainBackend;
    let smoke = experiment("smoke-fly-m-batched");
    assert_eq!(smoke.fly.backend, TrainBackend::Batched);
    assert!(smoke.flyg.ends_with("fly-M-v1.flyg") && smoke.brain_config.ends_with("M-brain.toml"));
    assert_eq!(smoke.fly.batched_memory_cap_mb, 3072);
    for name in ["e005-fly", "e005-fly-noclb", "e005-fly-s2"] {
        assert_eq!(experiment(name).fly.backend, TrainBackend::PerSequence, "{name}");
    }
}
