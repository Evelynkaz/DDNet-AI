//! Task 3.7b: the loss diagnosis (`ddai_planner::diag`) must not change the game it watches. A hybrid on the work clock
//! plays the same games alone and inside a `DiagBrain` (a shadowing fixed planner, the pool recorded and the planner's
//! plan scored by the hybrid's own evaluator after every decision, the truth rollouts run at the end): the decision
//! hashes of every game are identical.

use std::path::{Path, PathBuf};

use ddai_env::arena::{Arena, load_arena_defs};
use ddai_env::config::{PlayerSpec, Rules, builtin_brain, hybrid_config};
use ddai_env::game::play_game;
use ddai_env::run::layout_of;
use ddai_env::sim::PlayerSetup;
use ddai_planner::diag::{DiagBrain, DiagOptions};
use ddai_planner::hybrid::{HybridBrain, NoProposer};

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

fn arena(name: &str) -> Arena {
    let defs = load_arena_defs(&repo("configs/arenas")).unwrap();
    Arena::build(&defs[name], Path::new("/nonexistent")).unwrap()
}

fn hybrid_spec() -> PlayerSpec {
    let mut s = PlayerSpec::simple("hybrid");
    s.mode = Some("deadline".into());
    s.clock = Some("work".into());
    s.budget_ms = Some(4.0);
    s
}

fn rules() -> Rules {
    Rules {
        max_ticks: 300,
        after_ticks: 20,
        ..Rules::default()
    }
}

fn opponent() -> PlayerSetup {
    let brain = builtin_brain(&PlayerSpec::simple("planner")).unwrap();
    PlayerSetup {
        brain,
        lag: 0,
        label: "planner".into(),
    }
}

fn plain(arena: &Arena, g: u32) -> (String, String) {
    let brain = builtin_brain(&hybrid_spec()).unwrap();
    let players = vec![
        PlayerSetup {
            brain,
            lag: 0,
            label: "hybrid".into(),
        },
        opponent(),
    ];
    let rep = play_game(arena, &rules(), 1 + u64::from(g), layout_of(arena, g), players).unwrap();
    (rep.players[0].hash.clone(), rep.players[1].hash.clone())
}

fn watched(arena: &Arena, g: u32, opts: DiagOptions) -> ((String, String), usize, usize) {
    let (mut cfg, clock) = hybrid_config(&hybrid_spec()).unwrap();
    cfg.debug_pool = true;
    let hybrid = HybridBrain::new(cfg, clock, Box::new(NoProposer)).unwrap();
    let (brain, inner) = DiagBrain::new(hybrid, opts);
    let players = vec![
        PlayerSetup {
            brain: Box::new(brain),
            lag: 0,
            label: "diag".into(),
        },
        opponent(),
    ];
    let rep = play_game(arena, &rules(), 1 + u64::from(g), layout_of(arena, g), players).unwrap();
    inner.borrow_mut().finalize();
    let inner = inner.borrow();
    let scored = inner.records.iter().filter(|r| r.teacher_score.is_some()).count();
    let truth = inner.records.iter().filter(|r| r.truth_hybrid.is_some()).count();
    (
        (rep.players[0].hash.clone(), rep.players[1].hash.clone()),
        scored,
        truth,
    )
}

#[test]
fn watching_does_not_change_the_game() {
    // A decision keeps world-sized values on the stack: the 2 MiB test-thread default overflows (3.7b review F1).
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(watching_body)
        .unwrap()
        .join()
        .unwrap();
}

fn watching_body() {
    let arena = arena("platform");
    for g in 0..3 {
        let base = plain(&arena, g);
        let opts = DiagOptions {
            window: Some((0, 10_000)),
            mirror: true,
        };
        let (diag, scored, truth) = watched(&arena, g, opts);
        assert_eq!(base, diag, "game {g}: the diagnosing brain played differently");
        assert!(scored > 0, "game {g}: the planner's plan was never scored");
        assert!(truth > 0, "game {g}: no truth rollouts ran");
    }
}
