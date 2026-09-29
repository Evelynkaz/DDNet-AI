//! The T1-T18 technique scenarios (task 8.1, criterion 4b): the catalogue is complete, every
//! scenario is solvable (its reference solution passes on the exact start), a brain that does
//! nothing fails the ones where action is required, and scoring is reproducible.

use std::path::{Path, PathBuf};

use ddai_brain::IdleBrain;
use ddai_env::config::builtin_brain;
use ddai_env::scenario::*;

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

fn map_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join("aiddnet/data/maps"))
        .unwrap_or_default()
}

fn defs() -> Vec<ScenarioDef> {
    ScenarioDef::load_dir(&repo("configs/scenarios")).unwrap()
}

/// Scenarios on real maps need the local map file; on a machine without it they are skipped.
fn world_or_skip(def: &ScenarioDef) -> Option<ddai_env::arena::BuiltWorld> {
    match load_world(def, &map_dir()) {
        Ok(w) => Some(w),
        Err(e) => {
            eprintln!("skipping {}: {e}", def.id);
            None
        }
    }
}

#[test]
fn the_whole_t1_to_t18_catalogue_is_present() {
    let ids: Vec<String> = defs().into_iter().map(|d| d.id).collect();
    let mut want: Vec<String> = (1..=18)
        .flat_map(|n| {
            if n == 15 {
                vec!["T15a".to_string(), "T15b".to_string()]
            } else {
                vec![format!("T{n}")]
            }
        })
        .collect();
    want.sort();
    let mut got = ids.clone();
    got.sort();
    assert_eq!(got, want, "the catalogue rows T1..T18 (T15 in two spawn orders)");
    let mut dedup = ids.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), ids.len(), "scenario ids are unique");
}

#[test]
fn every_scenario_has_a_reference_solution_that_passes_the_exact_start() {
    let mut checked = 0;
    for def in defs() {
        assert!(!def.reference.is_empty(), "{} has no reference solution", def.id);
        let Some(world) = world_or_skip(&def) else { continue };
        let out = run_trial(&def, &world, reference_brain(&def), 0, 1, 0, false).unwrap();
        assert!(
            out.success,
            "{}: the reference solution does not solve its own scenario",
            def.id
        );
        checked += 1;
    }
    assert!(
        checked >= 17,
        "at most the two real-map scenarios may be skipped, checked {checked}"
    );
}

/// The control: a brain that does nothing must not solve a scenario that asks for action, and
/// must solve the discipline scenarios flagged `idle_should_pass`.
#[test]
fn doing_nothing_fails_the_scenarios_that_need_action() {
    for def in defs() {
        let Some(world) = world_or_skip(&def) else { continue };
        let score = score_brain(&def, &world, "idle", 0, 1, None, &|| Ok(Box::new(IdleBrain))).unwrap();
        if def.idle_should_pass {
            assert!(
                score.rate >= 0.9,
                "{}: idle should pass a discipline scenario, got {score:?}",
                def.id
            );
        } else {
            assert!(
                score.rate <= 0.1,
                "{}: a scenario an idle brain solves does not test anything: {score:?}",
                def.id
            );
        }
    }
}

/// Jitter turns one setup into many trials but the reference solution should still cope with most
/// of them for the simple open-loop ones -- and different trials really do start differently.
#[test]
fn trials_start_from_different_jittered_positions() {
    let def = defs().into_iter().find(|d| d.id == "T1").unwrap();
    let world = load_world(&def, &map_dir()).unwrap();
    let start = |k: u32| {
        run_trial(&def, &world, Box::new(IdleBrain), 0, 1, k, true)
            .unwrap()
            .trace
            .snaps[0][0]
            .pos
    };
    assert_ne!(start(0), start(1));
    assert_eq!(start(3), start(3), "a trial is deterministic in (seed, trial)");
    let exact = run_trial(&def, &world, Box::new(IdleBrain), 0, 1, 5, false)
        .unwrap()
        .trace
        .snaps[0][0]
        .pos;
    assert_eq!(
        exact,
        [20.5 * 32.0, 9.5 * 32.0],
        "jitter = false is the nominal start (before settling)"
    );
}

#[test]
fn scores_are_reproducible_and_wilson_bounded() {
    let def = defs().into_iter().find(|d| d.id == "T13").unwrap();
    let world = load_world(&def, &map_dir()).unwrap();
    let make = || builtin_brain(&ddai_env::config::PlayerSpec::simple("scripted"));
    let a = score_brain(&def, &world, "scripted", 0, 7, Some(12), &make).unwrap();
    let b = score_brain(&def, &world, "scripted", 0, 7, Some(12), &make).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.trials, 12);
    assert!(a.lo <= a.rate && a.rate <= a.hi);
}

/// The hook duel exists in both spawn orders because the order decides which hook is strong: the
/// solution of one order does not solve the other.
#[test]
fn the_hook_duel_depends_on_the_spawn_order() {
    let all = defs();
    let a = all.iter().find(|d| d.id == "T15a").unwrap();
    let b = all.iter().find(|d| d.id == "T15b").unwrap();
    let world = load_world(a, &map_dir()).unwrap();
    assert!(a.spawn_reverse != b.spawn_reverse);
    // T15a's solution (hook and walk away) drags the subject into the pit as well when it is
    // spawned last.
    let wrong = run_trial(b, &world, reference_brain(a), 0, 1, 0, false).unwrap();
    assert!(!wrong.success, "the same inputs must not win in both orders");
}

/// Predicates on hand-built traces.
#[test]
fn predicates_evaluate_on_traces() {
    let snap = |out: bool, onset: bool, credit: Option<usize>| TeeSnap {
        out,
        onset,
        credit,
        pos: [64.0, 64.0],
        hook_state: -1,
        hooked_player: -1,
        grounded: false,
    };
    // tee 1 goes out at tick 2 credited to 0 and stays out; tee 0 never.
    let trace = Trace {
        snaps: vec![
            vec![snap(false, false, None), snap(false, false, None)],
            vec![snap(false, false, None), snap(false, false, None)],
            vec![snap(false, false, None), snap(true, true, Some(0))],
            vec![snap(false, false, None), snap(true, false, None)],
        ],
    };
    let parse = |t: &str| {
        let def = ScenarioDef::parse(&format!(
            "id='X'\nname='x'\nhorizon=3\n[map]\nkind='synthetic'\nwidth=4\nheight=4\n[[tee]]\npos=[1.0,1.0]\n[[tee]]\npos=[2.0,1.0]\n[success]\n{t}\n"
        ))
        .unwrap();
        def.success
    };
    let holds = |t: &str| eval(&parse(t), &trace, 3);
    assert!(holds("kind='out'\ntee=1\nwithin=2\ncredited_to=0"));
    assert!(!holds("kind='out'\ntee=1\nwithin=1"));
    assert!(!holds("kind='out'\ntee=1\ncredited_to=1"));
    assert!(holds("kind='not_out'\ntee=0"));
    assert!(!holds("kind='not_out'\ntee=1"));
    assert!(holds("kind='not_out'\ntee=1\nuntil=1"));
    assert!(holds("kind='stays_out'\ntee=1\nfrom=2\nto=3"));
    assert!(!holds("kind='stays_out'\ntee=1\nfrom=1\nto=3"));
    assert!(holds("kind='free'\ntee=1\ntick=1"));
    assert!(!holds("kind='free'\ntee=1\ntick=3"));
    assert!(holds("kind='onsets_at_most'\ntee=1\nn=1"));
    assert!(!holds("kind='onsets_at_most'\ntee=1\nn=0"));
    assert!(holds("kind='in_box'\ntee=0\ntick=2\nx0=1.5\ny0=1.5\nx1=2.5\ny1=2.5"));
    assert!(!holds("kind='in_box'\ntee=0\ntick=2\nx0=3.0\ny0=1.5\nx1=4.0\ny1=2.5"));
    assert!(holds("kind='any'\nof=[{kind='not_out',tee=1},{kind='not_out',tee=0}]"));
    assert!(!holds("kind='all'\nof=[{kind='not_out',tee=1},{kind='not_out',tee=0}]"));
    assert!(ScenarioDef::parse("id='X'\nname='x'\nhorizon=3\n[map]\nkind='synthetic'\nwidth=4\nheight=4\n[[tee]]\npos=[1.0,1.0]\n[[tee]]\npos=[2.0,1.0]\n[success]\nkind='not_out'\ntee=5\n").is_err());
}
