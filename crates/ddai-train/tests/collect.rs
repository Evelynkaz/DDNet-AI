//! Teacher labelling in the arena: determinism, thread independence, and that a game the teacher
//! plays itself is exactly the game the plain planner brain plays (same decisions, tick by tick).

use std::path::PathBuf;

use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::brains::RecordingBrain;
use ddai_env::config::{PlayerSpec, Rules, builtin_brain};
use ddai_env::game::{Layout, play_game};
use ddai_env::sim::PlayerSetup;
use ddai_train::collect::{CollectJob, Mixing, collect, collect_game_held};
use ddai_train::types::{Episode, action_of, step_flags};

fn pit() -> Arena {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/arenas");
    let defs = load_arena_defs(&dir).expect("arena defs");
    Arena::build(&defs["pit"], std::path::Path::new("/nonexistent")).expect("pit is synthetic")
}

fn job(actor: Option<PlayerSpec>, mixing: Mixing, games: u32) -> CollectJob {
    CollectJob {
        arena: "pit".into(),
        games,
        opponents: vec![PlayerSpec::simple("scripted")],
        actor,
        mixing,
        base_seed: 11,
    }
}

/// Short games: the planner is slow in a test build, and a few hundred ticks show everything.
fn rules() -> Rules {
    Rules {
        max_ticks: 240,
        after_ticks: 30,
        ..Rules::default()
    }
}

fn run(job: &CollectJob, threads: usize) -> Vec<Episode> {
    collect(&pit(), 0, &rules(), job, &builtin_brain, threads).expect("collect")
}

#[test]
fn a_game_the_teacher_plays_is_the_planner_brains_game() {
    let j = job(None, Mixing::default(), 3);
    let eps = run(&j, 2);
    let rules = rules();
    let arena = pit();
    for (g, ep) in eps.iter().enumerate() {
        // The plain planner brain plays the same game (same seed, same layout), recorded.
        let (rec, log) = RecordingBrain::new(builtin_brain(&PlayerSpec::simple("planner")).unwrap());
        let players = vec![
            PlayerSetup {
                brain: Box::new(rec),
                lag: 0,
                label: "planner".into(),
            },
            PlayerSetup {
                brain: builtin_brain(&PlayerSpec::simple("scripted")).unwrap(),
                lag: 0,
                label: "scripted".into(),
            },
        ];
        let g32 = g as u32;
        let report = play_game(
            &arena,
            &rules,
            j.base_seed + u64::from(g32),
            Layout {
                swap: g32 % 2 == 1,
                reverse_order: (g32 / 2) % 2 == 1,
            },
            players,
        )
        .unwrap();
        let plain = log.lock().unwrap().clone();
        assert_eq!(ep.steps.len(), plain.len(), "game {g}: decisions");
        assert_eq!(ep.end_tick, report.end_tick, "game {g}");
        for (i, (s, (tick, a))) in ep.steps.iter().zip(&plain).enumerate() {
            // The record has no weapon-switch field (the planner always asks for the hammer).
            let mut mine = action_of(&s.played);
            mine.wanted_weapon = a.wanted_weapon;
            assert_eq!((s.tick, mine), (*tick, *a), "game {g} decision {i}");
            assert_eq!(s.label, s.played);
            assert!(s.teacher_acted() && !s.noise());
            assert!(s.soft.is_some(), "a fixed-iteration decision has a soft target");
        }
    }
}

#[test]
fn labels_are_deterministic_and_independent_of_the_thread_count() {
    let j = job(
        Some(PlayerSpec::simple("scripted")),
        Mixing {
            beta: 0.3,
            noise_prob: 0.1,
            noise_len: (2, 4),
        },
        3,
    );
    let (a, b, c) = (run(&j, 1), run(&j, 1), run(&j, 3));
    assert_eq!(a, b, "same seed, same labels");
    assert_eq!(a, c, "1 and 3 threads give the same episodes");
    let bytes = |e: &Vec<Episode>| postcard::to_allocvec(e).unwrap();
    assert_eq!(bytes(&a), bytes(&c));
}

#[test]
fn a_student_plays_and_the_teacher_still_labels_every_state() {
    // The scripted bot is the "student"; with beta = 0 it always plays, so played != label
    // somewhere, and every step still carries the teacher's label and soft target.
    let eps = run(&job(Some(PlayerSpec::simple("scripted")), Mixing::default(), 3), 2);
    let steps: Vec<_> = eps.iter().flat_map(|e| &e.steps).collect();
    assert!(steps.len() > 30);
    assert!(steps.iter().all(|s| !s.teacher_acted() && s.soft.is_some()));
    assert!(
        steps.iter().any(|s| s.label != s.played),
        "a student differs from the teacher"
    );
}

#[test]
fn beta_one_makes_the_teacher_play_and_noise_marks_its_steps() {
    let mixed = run(
        &job(
            Some(PlayerSpec::simple("scripted")),
            Mixing {
                beta: 1.0,
                ..Mixing::default()
            },
            4,
        ),
        2,
    );
    assert!(
        mixed
            .iter()
            .flat_map(|e| &e.steps)
            .all(|s| s.label == s.played && s.teacher_acted())
    );

    let noisy = run(
        &job(
            None,
            Mixing {
                beta: 0.0,
                noise_prob: 0.5,
                noise_len: (2, 3),
            },
            4,
        ),
        2,
    );
    let steps: Vec<_> = noisy.iter().flat_map(|e| &e.steps).collect();
    let n_noise = steps.iter().filter(|s| s.noise()).count();
    assert!(
        n_noise > 0 && n_noise < steps.len(),
        "some but not all steps are noise ({n_noise}/{})",
        steps.len()
    );
    for s in steps.iter().filter(|s| s.noise()) {
        assert_eq!(s.flags & step_flags::TEACHER_ACTED, 0);
    }
    // Noise never changes what the teacher says about the state: label steps are still searched.
    assert!(steps.iter().all(|s| s.searched()));
}

/// Technique scenarios as teacher data: the labelled subject plays a scenario trial (synthetic map, no map files
/// needed), the episode carries the scenario's tick horizon and a win when the success predicate held, and the
/// trials come back identical at any thread count.
#[test]
fn scenario_trials_are_labelled_deterministically() {
    use ddai_env::scenario::{ScenarioDef, load_world};
    use ddai_train::collect::collect_scenario;
    use ddai_train::types::Outcome;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/scenarios");
    let defs = ScenarioDef::load_dir(&dir).expect("scenarios");
    let def = defs.iter().find(|d| d.id == "T12").expect("T12 (freeze jump) exists");
    let world = load_world(def, std::path::Path::new("/nonexistent")).expect("synthetic map");
    let job = CollectJob {
        arena: "scn:T12".into(),
        games: 3,
        opponents: Vec::new(),
        actor: None,
        mixing: Mixing::default(),
        base_seed: 5,
    };
    let one = collect_scenario(def, &world, 0, &job, &builtin_brain, 1).expect("collect");
    let three = collect_scenario(def, &world, 0, &job, &builtin_brain, 3).expect("collect");
    assert_eq!(one, three, "trials do not depend on the thread count");
    assert_eq!(one.len(), 3);
    for (k, ep) in one.iter().enumerate() {
        assert_eq!(ep.seed, 5 + k as u64);
        assert_eq!(ep.end_tick, def.horizon);
        assert_eq!(usize::from(ep.players), def.tee.len());
        assert!(!ep.steps.is_empty());
        assert!(
            ep.steps.iter().all(|s| s.teacher_acted() || s.noise()),
            "the teacher plays itself"
        );
    }
    // T12 is a planner scenario (100/100): the teacher's trials are wins.
    assert!(
        one.iter().all(|e| e.outcome == Outcome::Win),
        "{:?}",
        one.iter().map(|e| e.outcome).collect::<Vec<_>>()
    );
    // Jitter differs between trials, so the visited states differ.
    assert_ne!(one[0].steps[0].me.pos, one[1].steps[0].me.pos);
}

/// Task 3.10, the API the fly's training builds on: an episode past the first freeze. With `Rules::held_block_window` the game is played on
/// for 250 ticks after the deciding freeze, the episode holds those decisions too, and the `HeldOutcome` carries the reward facts.
#[test]
fn an_episode_can_play_on_past_the_first_freeze_and_report_whether_the_block_held() {
    let arena = pit();
    let j = job(None, Mixing::default(), 1);
    let short = Rules {
        max_ticks: 240,
        after_ticks: 0,
        ..Rules::default()
    };
    let long = short.clone().held_block_window();
    assert_eq!(long.after_ticks, ddai_env::config::HELD_BLOCK_TICKS);
    let (mut decided, mut longer) = (0, 0);
    for g in 0..6 {
        let (e0, h0) = collect_game_held(&arena, 0, &short, &j, g, &builtin_brain).unwrap();
        let (e1, h1) = collect_game_held(&arena, 0, &long, &j, g, &builtin_brain).unwrap();
        // The game is the same up to the deciding tick; the long episode only adds decisions.
        assert_eq!((e0.outcome, e0.end_tick), (e1.outcome, e1.end_tick), "game {g}");
        assert_eq!(h1.window_ticks, 250);
        assert!(e1.steps.len() >= e0.steps.len());
        if h1.victim_out_ticks > 0 || h1.held_block {
            decided += 1;
        }
        if e1.steps.len() > e0.steps.len() {
            longer += 1;
            assert!(
                e1.steps.last().unwrap().tick >= e1.end_tick,
                "game {g}: steps after the deciding tick"
            );
        }
        // The facts agree with themselves: held means out on every tick of the window.
        assert_eq!(
            h1.held_block,
            h1.victim_out_ticks == 250 && h1.escape_tick.is_none() && h1.victim_out_ticks > 0,
            "game {g}: {h1:?}"
        );
        assert!((-1.0..=1.0).contains(&h1.held_return()));
        assert_eq!(h1.held_return() == 1.0, h1.strict_held_win(), "game {g}");
        assert_eq!(h0.result, h1.result);
    }
    assert!(
        decided > 0 && longer > 0,
        "the pit decides games: {decided} decided, {longer} played on"
    );
}
