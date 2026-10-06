//! Task 8.5a on a tiny fly and the synthetic `pit` arena (no local data needed): the post-freeze bank replays its source games tick for
//! tick, and the ES is deterministic (same parameters on any thread count), resumable (a run stopped and continued ends where a
//! continuous one does) and refuses to resume under a changed configuration.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_brain::IdleBrain;
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::game::play_game;
use ddai_env::sim::PlayerSetup;
use ddai_fly::bc::HookView;
use ddai_fly::bundle::read_zstd_postcard;
use ddai_train::bank::{Bank, BankBuildSpec, ReplayThenBrain, build_bank, play_from_start};
use ddai_train::es::{EsConfig, EsState, run_es};
use ddai_train::experiment::{Env, load_env};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Fixture {
    _dir: tempfile::TempDir,
    dir: PathBuf,
    env: Env,
    bundle: PathBuf,
    flyg: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let (bundle, flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle(&dir, HookView::Shared);
    let env = load_env(
        &root().join("configs/arenas"),
        Path::new("/nonexistent"),
        Some(flyg.clone()),
    )
    .unwrap();
    Fixture {
        _dir: tmp,
        dir,
        env,
        bundle,
        flyg,
    }
}

fn small_bank(f: &Fixture) -> Bank {
    let spec = BankBuildSpec {
        arenas: vec!["pit".into()],
        blockers: vec![("scripted".into(), 120)],
        base_seed: 5_000,
        window_ticks: 250,
        threads: 2,
    };
    build_bank(&f.env, &spec, &mut |_| {}).unwrap()
}

#[test]
fn a_bank_start_replays_its_source_game_tick_for_tick() {
    let f = fixture();
    let bank = small_bank(&f);
    assert!(
        bank.starts.len() >= 8,
        "only {} credited freezes in 120 games",
        bank.starts.len()
    );
    let arena = &f.env.arenas["pit"];
    let rules = Rules {
        after_ticks: 0,
        ..bank.rules.clone()
    };
    let scripted = || f.env.models.factory()(&PlayerSpec::simple("scripted")).unwrap();
    for s in bank.starts.iter().take(12) {
        // The source game: scripted against scripted.
        let setups = || {
            vec![
                PlayerSetup {
                    brain: scripted(),
                    lag: 0,
                    label: "a".into(),
                },
                PlayerSetup {
                    brain: scripted(),
                    lag: 0,
                    label: "b".into(),
                },
            ]
        };
        let src = play_game(arena, &rules, s.seed, s.layout(), setups()).unwrap();
        assert_eq!(src.end_tick, s.end_tick);
        // The replay: an idle brain whose seat is played from the log until the (never reached) handover tick.
        let replay = ReplayThenBrain::new(Box::new(IdleBrain), Arc::new(s.actions.clone()), i32::MAX, 0);
        let mut players = setups();
        players[0].brain = Box::new(replay);
        let rep = play_game(arena, &rules, s.seed, s.layout(), players).unwrap();
        assert_eq!(rep.end_tick, src.end_tick, "seed {}", s.seed);
        assert_eq!(rep.result, src.result);
        assert_eq!(rep.players[0].hash, src.players[0].hash, "the blocker's decisions");
        assert_eq!(rep.players[1].hash, src.players[1].hash, "the victim's decisions");
    }
}

#[test]
fn every_start_replays_through_the_helper_and_idle_is_tagged() {
    let f = fixture();
    let bank = small_bank(&f);
    let arena = &f.env.arenas["pit"];
    let mut idle_holds = 0;
    let (mut idle_out, mut idle_out_victim_stays) = (0usize, 0usize);
    for s in &bank.starts {
        let idle = f.env.models.factory()(&PlayerSpec::simple("idle")).unwrap();
        let scripted = f.env.models.factory()(&PlayerSpec::simple("scripted")).unwrap();
        let o = play_from_start(arena, &bank.rules, s, idle, scripted, 250, 0).unwrap();
        assert_eq!(o.held_block, s.idle_held, "the tag is what an idle blocker does");
        // The split tags (review of 8.5a, F1): the victim's escape is tracked even after the idle blocker went out.
        assert_eq!(s.victim_escapes_under_idle, Some(o.escape_tick.is_some()));
        assert_eq!(s.idle_blocker_out, Some(o.focal_out_in_window));
        if o.held_block {
            assert!(!o.escape_tick.is_some() && !o.focal_out_in_window);
        }
        if o.focal_out_in_window {
            idle_out += 1;
            idle_out_victim_stays += usize::from(o.escape_tick.is_none());
        }
        assert!(o.credited && o.end_tick == s.end_tick);
        // The shaping of a recorded episode telescopes to the difference of its end potentials (G0 of the research plan).
        assert!((o.shaping - o.shaping_closed_form()).abs() < 1e-4, "{o:?}");
        assert_eq!(o.window_played, 250);
        idle_holds += usize::from(s.idle_held);
    }
    assert!(
        idle_holds < bank.starts.len(),
        "some start must be escapable for the bank to be useful"
    );
    // The old "escapable" (`!idle_held`) mixes two things: starts where the victim escapes under an idle blocker (V) and starts where
    // only the idle blocker falls (B); the split tags tell them apart, and the victim is tracked after the blocker went out.
    let not_held = bank.starts.iter().filter(|s| !s.idle_held).count();
    let victim_escapes = bank
        .starts
        .iter()
        .filter(|s| s.victim_escapes_under_idle == Some(true))
        .count();
    assert!(victim_escapes <= not_held, "V is a part of `!idle_held`");
    assert!(
        idle_out >= idle_out_victim_stays,
        "{idle_out} idle blockers out, {idle_out_victim_stays} with the victim staying"
    );
    // The bank survives a save and a load.
    let p = f.dir.join("bank.bin");
    bank.save(&p).unwrap();
    assert_eq!(Bank::load(&p).unwrap(), bank);
}

fn config(f: &Fixture, run: &str, generations: u64) -> EsConfig {
    let text = format!(
        r#"
name = "tiny"
flyg = "{flyg}"
init_bundle = "{bundle}"
arenas_dir = "{arenas}"
map_dir = "/nonexistent"
run_dir = "{run}"
bank = "{bank}"
seed = 7
pairs = 3
generations = {generations}
post_episodes = 3
normal_games = 2
train_arenas = ["pit"]
threads = 1
[space]
sigma_a = 0.02
[eval]
every = 0
starts = 3
games = 2
holdout_arenas = []
"#,
        flyg = f.flyg.display(),
        bundle = f.bundle.display(),
        arenas = root().join("configs/arenas").display(),
        run = f.dir.join(run).display(),
        bank = f.dir.join("bank.bin").display(),
    );
    EsConfig::parse(&text).unwrap()
}

fn state_of(run: &Path) -> EsState {
    read_zstd_postcard(&run.join("state.bin")).unwrap()
}

#[test]
fn the_es_is_deterministic_on_any_thread_count_and_resumable() {
    let f = fixture();
    small_bank(&f).save(&f.dir.join("bank.bin")).unwrap();
    // The same two generations twice, on 1 and on 3 threads.
    let a = config(&f, "run-a", 2);
    let mut b = config(&f, "run-b", 2);
    b.threads = 3;
    run_es(&a, false, &mut |_| {}).unwrap();
    run_es(&b, false, &mut |_| {}).unwrap();
    let (sa, sb) = (state_of(&f.dir.join("run-a")), state_of(&f.dir.join("run-b")));
    assert_eq!(sa.generation, 2);
    assert_eq!(
        sa, sb,
        "the parameters, Adam's moments and the bookkeeping are bit-identical"
    );
    // The parameters moved (the run did something) and the files the web tab reads are there.
    let start = ddai_fly::bundle::load_bundle(&f.bundle).unwrap();
    assert_ne!(ddai_train::es::space::flatten(&start), sa.theta);
    for file in [
        "config.toml",
        "metrics.jsonl",
        "status.json",
        "state.bin",
        "checkpoints/final.bundle",
        "checkpoints/last.bundle",
        "checkpoints/selected.bundle",
    ] {
        assert!(f.dir.join("run-a").join(file).exists(), "{file}");
    }
    let metrics = std::fs::read_to_string(f.dir.join("run-a/metrics.jsonl")).unwrap();
    let kinds: Vec<String> = metrics
        .lines()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["kind"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    for k in ["train", "es", "es_eval", "arena", "selection"] {
        assert!(kinds.iter().any(|x| x == k), "no {k} line in {kinds:?}");
    }
    // A run stopped after one generation and continued (to two) ends exactly where the continuous one did.
    let part = config(&f, "run-c", 1);
    run_es(&part, false, &mut |_| {}).unwrap();
    assert_eq!(state_of(&f.dir.join("run-c")).generation, 1);
    let full = config(&f, "run-c", 2);
    run_es(&full, false, &mut |_| {}).unwrap();
    let sc = state_of(&f.dir.join("run-c"));
    assert_eq!((sc.generation, &sc.theta, &sc.adam), (2, &sa.theta, &sa.adam));
    // Changing something that matters is refused on resume, and accepted with the flag.
    let mut changed = config(&f, "run-c", 3);
    changed.seed = 8;
    let err = run_es(&changed, false, &mut |_| {}).unwrap_err();
    assert!(err.contains("refusing to resume"), "{err}");
    run_es(&changed, true, &mut |_| {}).unwrap();
    assert!(f.dir.join("run-c/config-before-1.toml").exists());
}

#[test]
fn a_holdout_arena_is_refused_for_training() {
    let f = fixture();
    small_bank(&f).save(&f.dir.join("bank.bin")).unwrap();
    let mut c = config(&f, "run-h", 1);
    c.train_arenas = vec!["clb-right".into()];
    let e = run_es(&c, false, &mut |_| {}).unwrap_err();
    assert!(e.contains("clb-right"), "{e}");
}

#[test]
fn post_freeze_starts_are_labelled_into_a_teacher_dataset_the_bc_trainer_reads() {
    use ddai_train::bank_collect::collect_starts;
    use ddai_train::collect::Mixing;
    use ddai_train::store::TeacherStore;
    let f = fixture();
    let bank = small_bank(&f);
    let starts: Vec<_> = bank.starts.iter().take(3).collect();
    let mut store = TeacherStore::create(&f.dir.join("pf"), "post-freeze", "test").unwrap();
    // The teacher plays after the freeze.
    let s = collect_starts(
        &f.env,
        &mut store,
        &starts,
        &bank.rules,
        "teacher",
        Mixing::default(),
        250,
        50,
        1,
        2,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(s.games, 3);
    // Every episode holds the burn-in (25 decisions) and the window (125) at least, i.e. labels after the first freeze.
    assert!(s.steps >= 3 * 140, "{} decisions", s.steps);
    store.verify().unwrap();
    // A student (here the tiny fly) plays, the teacher labels what it visits.
    let actor = format!("fly:{}", f.bundle.display());
    let s2 = collect_starts(
        &f.env,
        &mut store,
        &starts,
        &bank.rules,
        &actor,
        Mixing::default(),
        250,
        50,
        2,
        2,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(s2.games, 3);
    assert_eq!(store.manifest.total_episodes(), 6);
    let reopened = TeacherStore::open(&f.dir.join("pf")).unwrap();
    reopened.verify().unwrap();
    // The ticks of the steps run from the burn-in to the end of the window.
    let chunk = reopened.read_chunk(0).unwrap();
    let ep = &chunk.episodes[0];
    let first = ep.steps.first().unwrap().tick;
    assert!(
        first >= (starts[0].end_tick - 50).max(0) && first <= (starts[0].end_tick - 49).max(1),
        "{first}"
    );
    assert!(ep.steps.last().unwrap().tick >= starts[0].end_tick + 240);
}

#[test]
fn a_resumed_run_refuses_a_replaced_start_bundle_and_keeps_its_best_score_with_the_evaluation() {
    let f = fixture();
    small_bank(&f).save(&f.dir.join("bank.bin")).unwrap();
    let mut c = config(&f, "run-p", 1);
    c.eval.every = 1;
    let run = f.dir.join("run-p");
    let es_scores = |run: &Path| -> Vec<(u64, f64)> {
        std::fs::read_to_string(run.join("metrics.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(|v| v["kind"] == "es_eval")
            .filter_map(|v| Some((v["generation"].as_u64()?, v["score"].as_f64()?)))
            .collect()
    };
    // A kill right after the first evaluation (the log line of an evaluation comes last in it): the process dies before the first update.
    // With `state.bin` written only after the update (the order before the review of 8.5a, F7), no state existed at this point, and a
    // resume skipped the evaluation (its metrics line was there) with a best score of -inf, so a later, worse evaluation could overwrite
    // `selected.bundle`.
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run_es(&c, false, &mut |l| {
            assert!(!l.contains("eval generation 0"), "killed after the first evaluation");
        });
    }));
    assert!(killed.is_err(), "the simulated kill did not happen");
    assert!(
        run.join("state.bin").exists(),
        "state.bin must exist right after an evaluation"
    );
    let st = state_of(&run);
    let scores = es_scores(&run);
    assert_eq!(scores.len(), 1, "{scores:?}");
    assert_eq!(st.generation, 0, "no update happened yet");
    assert!(
        (st.best_score - scores[0].1).abs() < 1e-9,
        "{} vs {:?}",
        st.best_score,
        scores
    );
    assert!(run.join("checkpoints/selected.bundle").exists());
    // The resume goes on from there: it does not evaluate generation 0 again and finishes the run.
    run_es(&c, false, &mut |_| {}).unwrap();
    let scores = es_scores(&run);
    assert_eq!(scores.iter().filter(|(g, _)| *g == 0).count(), 1, "{scores:?}");
    assert!(run.join("init.sha256").exists());
    let st = state_of(&run);
    assert!((st.best_score - scores.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max)).abs() < 1e-9);
    // Replace the start bundle by another one: the resume is refused.
    let mut b = ddai_fly::bundle::load_bundle(&f.bundle).unwrap();
    b.decoder_params.jump_b += 0.5;
    ddai_fly::bundle::save_bundle(&f.bundle, &b).unwrap();
    let mut longer = c.clone();
    longer.generations = 2;
    let e = run_es(&longer, false, &mut |_| {}).unwrap_err();
    assert!(e.contains("start bundle"), "{e}");
}
