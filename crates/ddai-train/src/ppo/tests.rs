//! Task 8.5b on a tiny fly and the synthetic `pit` arena (no local data needed): the actor in arg-max mode plays exactly like the bundle, the
//! stored recurrent state reproduces the actor's logits in the batched windows, the policy backward is the gradient of the logits (also for
//! the opponent-state channels), the rewards decompose into the 8.5a return, and a run is deterministic on any thread count, resumable and
//! refuses a changed configuration.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ddai_brain::Brain;
use ddai_env::config::{PlayerSpec, Rules};
use ddai_env::game::{Layout, play_game};
use ddai_env::sim::PlayerSetup;
use ddai_fly::batched::{BatchedEngine, BatchedPlan};
use ddai_fly::bc::{HeadLogits, HookView};
use ddai_fly::brain::FlyBrainConfig;
use ddai_fly::brain_policy_batched::{PolicyWindow, policy_backward, policy_forward};
use ddai_fly::bundle::{
    FlyBrainTemplate, FlyBundle, load_bundle, read_zstd_postcard, save_bundle, upgrade_with_opponent_state,
};
use ddai_fly::rng::SplitMix64;

use super::actor::{ActMode, Decision, PpoActor, WindowGrid};
use super::config::PpoConfig;
use super::learner::{PpoState, plan_windows};
use super::reward::{EpisodeKind, PpoReward, shaping_terms};
use super::rollout::{EpisodeSpec, RolloutCtx, play_episode_ppo};
use super::{rollout, run_ppo};
use crate::bank::{Bank, BankBuildSpec, build_bank};
use crate::experiment::{Env, load_env};
use crate::learner::{FlyLearner, FlyTrainConfig, Learner};

const SECTION: &str = r#"
[opponent_state]
frozen = ["VPN_OPP", "VPN_WALL"]
freeze_left = ["VPN_OPP"]
velocity_x = ["VPN_OPP"]
velocity_y = ["VPN_OPP"]
hook_state = ["AN_GROUND", "VPN_WALL"]
"#;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Fx {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    env: Env,
    bundle: PathBuf,
    flyg: PathBuf,
    bank: Bank,
}

fn fixture(view: HookView) -> Fx {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let (plain, flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle(&dir, view);
    // The tiny fly upgraded with the opponent-state channels (zero weights), as the real one is.
    let b = load_bundle(&plain).unwrap();
    let up = upgrade_with_opponent_state(&b, ddai_flyg::load(&flyg).unwrap(), SECTION).unwrap();
    let bundle = dir.join("up.bundle");
    save_bundle(&bundle, &up).unwrap();
    let env = load_env(
        &root().join("configs/arenas"),
        Path::new("/nonexistent"),
        Some(flyg.clone()),
    )
    .unwrap();
    let spec = BankBuildSpec {
        arenas: vec!["pit".into()],
        blockers: vec![("scripted".into(), 120)],
        base_seed: 5_000,
        window_ticks: 250,
        threads: 2,
    };
    let bank = build_bank(&env, &spec, &mut |_| {}).unwrap();
    Fx {
        _tmp: tmp,
        dir,
        env,
        bundle,
        flyg,
        bank,
    }
}

fn scripted(f: &Fx) -> Box<dyn Brain> {
    f.env.models.factory()(&PlayerSpec::simple("scripted")).unwrap()
}

#[test]
fn the_actor_in_argmax_mode_plays_exactly_like_the_bundle() {
    for view in [HookView::Shared, HookView::MaskedForHookHead] {
        let f = fixture(view);
        let t = FlyBrainTemplate::load(&f.bundle, Some(&f.flyg)).unwrap();
        let arena = &f.env.arenas["pit"];
        let rules = Rules {
            max_ticks: 600,
            after_ticks: 0,
            ..Rules::default()
        };
        let grid = WindowGrid {
            chunk: 32,
            burn_in: 8,
            decide_every: rules.decide_every,
        };
        for (seed, swap) in [(11u64, false), (12, true), (13, false), (14, true)] {
            let layout = Layout {
                swap,
                reverse_order: false,
            };
            let setups = |focal: Box<dyn Brain>| {
                vec![
                    PlayerSetup {
                        brain: focal,
                        lag: 0,
                        label: "f".into(),
                    },
                    PlayerSetup {
                        brain: scripted(&f),
                        lag: 0,
                        label: "o".into(),
                    },
                ]
            };
            let a = play_game(
                arena,
                &rules,
                seed,
                layout,
                setups(t.instantiate_played(FlyBrainConfig::default())),
            )
            .unwrap();
            let sink = Arc::new(Mutex::new(Vec::new()));
            let actor = PpoActor::new(
                &t,
                ActMode::Argmax,
                8.0,
                ddai_fly::policy::Temperatures::uniform(0.3),
                0,
                grid,
                sink.clone(),
            );
            let b = play_game(arena, &rules, seed, layout, setups(Box::new(actor))).unwrap();
            assert_eq!(
                a.players[0].hash, b.players[0].hash,
                "{view:?} seed {seed}: the decisions"
            );
            assert_eq!((a.result, a.end_tick), (b.result, b.end_tick));
            assert!(sink.lock().unwrap().len() > 20);
        }
    }
}

/// The longest episode of the tiny fly (the tiny fly freezes itself early in many starts) among the first bank starts.
fn an_episode(f: &Fx, template: &FlyBrainTemplate, mode: ActMode) -> (super::rollout::Episode, crate::bank::BankStart) {
    let rules = f.bank.rules.clone();
    let fields = f
        .env
        .arenas
        .iter()
        .map(|(n, a)| (n.clone(), super::critic::MapFields::new(&a.map)))
        .collect();
    let reward = PpoReward::default();
    let ctx = RolloutCtx {
        arenas: &f.env.arenas,
        fields: &fields,
        template,
        rules: &rules,
        window: 250,
        burn_in_ticks: 50,
        grid: WindowGrid {
            chunk: 32,
            burn_in: 8,
            decide_every: rules.decide_every,
        },
        aim_kappa: 8.0,
        temps: ddai_fly::policy::Temperatures::uniform(0.3),
        gamma: 0.995,
        reward: &reward,
        mode,
    };
    let sc = || f.env.models.factory()(&PlayerSpec::simple("scripted"));
    f.bank
        .starts
        .iter()
        .take(12)
        .map(|s| {
            (
                play_episode_ppo(&ctx, &EpisodeSpec::Post(s), &sc, None).unwrap(),
                s.clone(),
            )
        })
        .max_by_key(|(e, _)| e.acted_decisions())
        .unwrap()
}

#[test]
fn stored_states_reproduce_the_actors_logits_in_the_batched_windows() {
    for view in [HookView::Shared, HookView::MaskedForHookHead] {
        let f = fixture(view);
        let t = FlyBrainTemplate::load(&f.bundle, Some(&f.flyg)).unwrap();
        let (ep, _) = an_episode(&f, &t, ActMode::Sample);
        assert!(ep.acted_decisions() > 40, "{}", ep.acted_decisions());
        // The actor's logits, decision by decision, re-run from the first recorded decision with fresh brains.
        let mut full = t.instantiate(FlyBrainConfig::default());
        let mut masked = (view == HookView::MaskedForHookHead).then(|| t.instantiate(FlyBrainConfig::default()));
        let ctx0 = ddai_brain::ResetContext {
            map: ep.decisions[0].obs.map.clone(),
            self_id: 0,
            seed: 1,
        };
        full.reset(&ctx0);
        if let Some(m) = &mut masked {
            m.reset(&ctx0);
        }
        let seq: Vec<HeadLogits> = ep
            .decisions
            .iter()
            .map(|d| {
                let lf = full.forward_logits(&d.obs);
                match &mut masked {
                    Some(m) => {
                        ddai_fly::bc::combine_hook_view(&lf, &m.forward_logits(&ddai_fly::bc::mask_own_hook(&d.obs)))
                    }
                    None => lf,
                }
            })
            .collect();
        // The batched windows from the stored states.
        let learner =
            FlyLearner::from_bundle(load_bundle(&f.bundle).unwrap(), &f.flyg, FlyTrainConfig::default()).unwrap();
        let plans = plan_windows(std::slice::from_ref(&ep), 32, 8, 2);
        assert!(plans.len() >= 3, "{} windows", plans.len());
        let windows: Vec<PolicyWindow> = plans
            .iter()
            .map(|p| {
                let snap = ep.decisions[p.b]
                    .snap
                    .as_ref()
                    .expect("a stored state at the start of a window");
                PolicyWindow {
                    v_init: snap.0.clone(),
                    v_init_masked: snap.1.clone(),
                    observations: ep.decisions[p.b..p.i1].iter().map(|d| d.obs.clone()).collect(),
                }
            })
            .collect();
        let mut engine = BatchedEngine::with_plan(BatchedPlan::new(learner.model()));
        let fwd = policy_forward(
            learner.policy_net(),
            &mut engine,
            &windows,
            view == HookView::MaskedForHookHead,
            None,
        )
        .unwrap();
        let mut worst = 0.0f32;
        for (w, p) in plans.iter().enumerate() {
            for (i, b) in seq.iter().enumerate().take(p.i1).skip(p.i0) {
                let a = &fwd.logits[w][i - p.b];
                for (x, y) in [
                    (a.dir[0], b.dir[0]),
                    (a.dir[1], b.dir[1]),
                    (a.dir[2], b.dir[2]),
                    (a.jump, b.jump),
                    (a.hook, b.hook),
                    (a.fire, b.fire),
                    (a.aim_c, b.aim_c),
                    (a.aim_s, b.aim_s),
                ] {
                    worst = worst.max((x - y).abs() / (1.0 + y.abs()));
                }
            }
        }
        assert!(
            worst < 2e-3,
            "{view:?}: the batched window differs from the actor by {worst}"
        );
        // Every scored decision is in exactly one window, the burn-in comes before it.
        let acted = ep.acted_decisions();
        assert_eq!(plans.iter().map(|p| p.i1 - p.i0).sum::<usize>(), acted);
        assert!(plans.iter().all(|p| p.b <= p.i0));
    }
}

#[test]
fn the_policy_backward_is_the_gradient_of_the_logits_including_the_opponent_channels() {
    for view in [HookView::Shared, HookView::MaskedForHookHead] {
        let f = fixture(view);
        let t = FlyBrainTemplate::load(&f.bundle, Some(&f.flyg)).unwrap();
        let (ep, _) = an_episode(&f, &t, ActMode::Sample);
        let plans = plan_windows(std::slice::from_ref(&ep), 32, 8, 2);
        let plans = &plans[..3.min(plans.len())];
        let windows: Vec<PolicyWindow> = plans
            .iter()
            .map(|p| {
                let snap = ep.decisions[p.b].snap.as_ref().unwrap();
                PolicyWindow {
                    v_init: snap.0.clone(),
                    v_init_masked: snap.1.clone(),
                    observations: ep.decisions[p.b..p.i1].iter().map(|d| d.obs.clone()).collect(),
                }
            })
            .collect();
        let two_view = view == HookView::MaskedForHookHead;
        // Make the new channels non-zero so that their gradients are not the trivial ones, and give the decoder some signal.
        let mut learner =
            FlyLearner::from_bundle(load_bundle(&f.bundle).unwrap(), &f.flyg, FlyTrainConfig::default()).unwrap();
        let mut theta = learner.params();
        let mut rng = SplitMix64::new(3);
        struct L {
            a: usize,
            b: usize,
            theta: usize,
            g: usize,
            c: usize,
            decoder_start: usize,
        }
        let layout = {
            let l = learner.layout();
            L {
                a: l.a,
                b: l.b,
                theta: l.theta,
                g: l.g,
                c: l.c,
                decoder_start: l.decoder_start(),
            }
        };
        let g0 = layout.a + layout.b + layout.theta;
        for i in 0..layout.g {
            if theta[g0 + i] == 0.0 {
                theta[g0 + i] = 0.3 * rng.next_gaussian();
            }
        }
        let c0 = g0 + layout.g;
        for i in 0..layout.c {
            if theta[c0 + i] == 0.0 {
                theta[c0 + i] = 0.1 * rng.next_gaussian();
            }
        }
        learner.set_params(&theta).unwrap();
        // Fixed random weights on every logit of every scored decision: L = sum c * logit.
        let coef = |w: usize, t: usize, h: usize| -> f32 {
            let mut r = SplitMix64::new(((w * 1000 + t) * 10 + h) as u64 + 77);
            r.next_gaussian()
        };
        let weights = |w: usize, t: usize| HeadLogits {
            dir: [coef(w, t, 0), coef(w, t, 1), coef(w, t, 2)],
            jump: coef(w, t, 3),
            hook: coef(w, t, 4),
            fire: coef(w, t, 5),
            aim_c: coef(w, t, 6),
            aim_s: coef(w, t, 7),
        };
        let loss_of = |learner: &FlyLearner| -> f64 {
            let mut engine = BatchedEngine::with_plan(BatchedPlan::new(learner.model()));
            let fwd = policy_forward(learner.policy_net(), &mut engine, &windows, two_view, None).unwrap();
            let mut l = 0.0f64;
            for (w, p) in plans.iter().enumerate() {
                for i in p.i0..p.i1 {
                    let (x, c) = (&fwd.logits[w][i - p.b], weights(w, i - p.b));
                    l += f64::from(
                        x.dir[0] * c.dir[0]
                            + x.dir[1] * c.dir[1]
                            + x.dir[2] * c.dir[2]
                            + x.jump * c.jump
                            + x.hook * c.hook
                            + x.fire * c.fire
                            + x.aim_c * c.aim_c
                            + x.aim_s * c.aim_s,
                    );
                }
            }
            l
        };
        // Analytic gradient.
        let mut engine = BatchedEngine::with_plan(BatchedPlan::new(learner.model()));
        let fwd = policy_forward(learner.policy_net(), &mut engine, &windows, two_view, None).unwrap();
        let d: Vec<Vec<HeadLogits>> = plans
            .iter()
            .enumerate()
            .map(|(w, p)| {
                (0..p.i1 - p.b)
                    .map(|t| {
                        if t + p.b >= p.i0 {
                            weights(w, t)
                        } else {
                            HeadLogits::default()
                        }
                    })
                    .collect()
            })
            .collect();
        let bwd = policy_backward(learner.policy_net(), &mut engine, &fwd, &d);
        let mut grad = vec![0.0f32; theta.len()];
        let no_dec = learner.policy_net().decoder.zeros_gradients();
        let zero_enc = learner.encoder().zero_grads();
        learner.add_parts_grads(&bwd.fly, &zero_enc, &no_dec, &mut grad);
        let no_fly = ddai_fly::optim::ParamGradients::zeros_like(learner.policy_net().model.params());
        for (e, dd) in bwd.encoder.iter().zip(&bwd.decoder) {
            learner.add_parts_grads(&no_fly, e, dd, &mut grad);
        }
        // Finite differences on the parameters with the largest gradients of every group, and on the new channels.
        let groups: [(&str, usize, usize); 6] = [
            ("a", 0, layout.a),
            ("b", layout.a, layout.b),
            ("theta", layout.a + layout.b, layout.theta),
            ("enc_g", g0, layout.g),
            ("enc_c", c0, layout.c),
            ("dec", layout.decoder_start, theta.len() - layout.decoder_start),
        ];
        let mut checked = 0;
        for (name, start, len) in groups {
            let mut idx: Vec<usize> = (start..start + len).collect();
            idx.sort_by(|&x, &y| grad[y].abs().partial_cmp(&grad[x].abs()).unwrap());
            for &i in idx.iter().take(4) {
                if grad[i].abs() < 1e-3 {
                    continue;
                }
                let eps = 2e-3f32 * (1.0 + theta[i].abs());
                let (mut up, mut dn) = (theta.clone(), theta.clone());
                up[i] += eps;
                dn[i] -= eps;
                let (mut lu, mut ld) = (learner_with(&f, &up), learner_with(&f, &dn));
                let _ = (&mut lu, &mut ld);
                let numeric = ((loss_of(&lu) - loss_of(&ld)) / (2.0 * f64::from(eps))) as f32;
                assert!(
                    (numeric - grad[i]).abs() < 0.04 * (1.0 + numeric.abs().max(grad[i].abs())),
                    "{view:?} {name}[{i}]: analytic {} vs numeric {numeric}",
                    grad[i]
                );
                checked += 1;
            }
        }
        assert!(checked >= 8, "{checked} parameters checked");
        // The opponent-state channels got a gradient (the reason this task trains them at all).
        let new_ids: Vec<usize> = learner
            .encoder()
            .assignments()
            .iter()
            .filter(|a| {
                ddai_fly::encoder::OPPONENT_CHANNELS
                    .iter()
                    .any(|c| c.name() == a.channel)
            })
            .map(|a| a.param_id as usize)
            .collect();
        assert_eq!(new_ids.len(), 7);
        let norm: f32 = new_ids
            .iter()
            .map(|&i| grad[g0 + i].powi(2) + grad[c0 + i].powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(norm > 1e-3, "{view:?}: the new channels' gradient is {norm}");
    }
}

fn learner_with(f: &Fx, theta: &[f32]) -> FlyLearner {
    let mut l = FlyLearner::from_bundle(load_bundle(&f.bundle).unwrap(), &f.flyg, FlyTrainConfig::default()).unwrap();
    l.set_params(theta).unwrap();
    l
}

#[test]
fn the_rewards_of_an_episode_are_the_8_5a_return_plus_a_telescoping_shaping() {
    let f = fixture(HookView::Shared);
    let t = FlyBrainTemplate::load(&f.bundle, Some(&f.flyg)).unwrap();
    let fields: std::collections::BTreeMap<_, _> = f
        .env
        .arenas
        .iter()
        .map(|(n, a)| (n.clone(), super::critic::MapFields::new(&a.map)))
        .collect();
    let rules = f.bank.rules.clone();
    let reward = PpoReward::default();
    let plain = PpoReward {
        shaping: 0.0,
        free_victim: 0.0,
        ..PpoReward::default()
    };
    let eight_five_a = crate::heldblock::RewardConfig::default();
    let sc = || f.env.models.factory()(&PlayerSpec::simple("scripted"));
    let mut seen_held = false;
    let mut n_checked = 0;
    for (k, s) in f.bank.starts.iter().take(10).enumerate() {
        for (cfg, shaped) in [(&reward, true), (&plain, false)] {
            let ctx = RolloutCtx {
                arenas: &f.env.arenas,
                fields: &fields,
                template: &t,
                rules: &rules,
                window: 250,
                burn_in_ticks: 50,
                grid: WindowGrid {
                    chunk: 32,
                    burn_in: 8,
                    decide_every: 2,
                },
                aim_kappa: 8.0,
                temps: ddai_fly::policy::Temperatures::uniform(0.3),
                gamma: 0.995,
                reward: cfg,
                mode: ActMode::Sample,
            };
            let ep = play_episode_ppo(&ctx, &EpisodeSpec::Post(s), &sc, None).unwrap();
            assert_eq!(ep.kind, EpisodeKind::Post);
            assert_eq!(ep.rewards.len(), ep.decisions.len());
            // Burn-in decisions carry nothing.
            assert!(ep.rewards[..ep.first_acted()].iter().all(|&r| r == 0.0));
            let total_terminal = f64::from(ep.parts.terminal);
            let o = &ep.outcome;
            let want = eight_five_a.post_freeze(&crate::heldblock::EpisodeOutcome {
                shaping: 0.0,
                ..o.clone()
            });
            assert!(
                (total_terminal - f64::from(want)).abs() < 1e-6,
                "start {k}: {total_terminal} vs {want}"
            );
            seen_held |= o.held_block;
            if shaped {
                // The potential-based part: its discounted sum is exactly -scale * Phi(first acted decision).
                let first = ep.first_acted();
                let phi: Vec<f32> = ep.decisions[first..]
                    .iter()
                    .map(|d| super::reward::potential(&d.obs, &fields[&ep.arena], cfg.own_hazard))
                    .collect();
                let f_terms = shaping_terms(&phi, 0.995);
                let disc: f64 = f_terms
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| 0.995f64.powi(i as i32) * f64::from(x))
                    .sum();
                assert!((disc + f64::from(phi[0])).abs() < 1e-4, "{disc} vs {}", -phi[0]);
                let in_reward: f64 = ep.rewards[first..]
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| 0.995f64.powi(i as i32) * f64::from(r))
                    .sum();
                let expected = f64::from(reward.shaping) * disc
                    + f64::from(ep.parts.terminal) * 0.995f64.powi((ep.decisions.len() - 1 - first) as i32)
                    + f64::from(ep.parts.free_victim) * 0.0;
                // (the free-victim penalty, if it fired, sits at its own decision: allow for it)
                if !ep.parts.freed_victim {
                    assert!(
                        (in_reward - expected).abs() < 2e-3,
                        "start {k}: {in_reward} vs {expected}"
                    );
                }
            } else {
                assert!((f64::from(ep.total_reward()) - total_terminal).abs() < 1e-5);
            }
            n_checked += 1;
        }
    }
    assert_eq!(n_checked, 20);
    let _ = seen_held;
}

fn plan_of(c: &PpoConfig, iteration: u64, classes: &[Vec<usize>; 3], snapshots: usize) -> Vec<super::Plan> {
    let none: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    super::iteration_plan(
        c,
        iteration,
        classes,
        &none,
        &super::curriculum::CurriculumState::new(&c.curriculum),
        snapshots,
    )
}

fn config_text(f: &Fx, run: &str, iterations: u64, bc_dir: Option<&Path>) -> String {
    format!(
        r#"
name = "tiny"
flyg = "{flyg}"
init_bundle = "{bundle}"
arenas_dir = "{arenas}"
map_dir = "/nonexistent"
run_dir = "{run}"
bank = "{bank}"
seed = 7
iterations = {iterations}
train_arenas = ["pit"]
threads = 1
snapshot_every = 1
[rollout]
post_episodes = 4
game_episodes = 2
opponents = [["scripted", 1.0], ["past", 1.0]]
[ppo]
epochs = 2
minibatch_windows = 4
critic_warmup_iters = 1
critic_hidden = 16
bc_coef = {bc}
bc_windows = 2
[bc]
teacher_dirs = [{bcdir}]
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
        bc = if bc_dir.is_some() { "0.5" } else { "0.0" },
        bcdir = bc_dir.map_or(String::new(), |d| format!("\"{}\"", d.display())),
    )
}

fn state_of(run: &Path) -> PpoState {
    read_zstd_postcard(&run.join("state.bin")).unwrap()
}

#[test]
fn a_ppo_run_is_deterministic_on_any_thread_count_resumable_and_refuses_a_changed_config() {
    let f = fixture(HookView::MaskedForHookHead);
    f.bank.save(&f.dir.join("bank.bin")).unwrap();
    // A teacher dataset of planner labels on the post-freeze starts for the BC term.
    {
        use crate::bank_collect::collect_starts;
        use crate::collect::Mixing;
        use crate::store::TeacherStore;
        let starts: Vec<_> = f.bank.starts.iter().take(3).collect();
        let mut store = TeacherStore::create(&f.dir.join("pf"), "post-freeze", "test").unwrap();
        collect_starts(
            &f.env,
            &mut store,
            &starts,
            &f.bank.rules,
            "teacher",
            Mixing::default(),
            250,
            50,
            1,
            2,
            &mut |_| {},
        )
        .unwrap();
    }
    let bc = f.dir.join("pf");
    let text = |run: &str, it: u64| config_text(&f, run, it, Some(&bc));
    let a = PpoConfig::parse(&text("run-a", 2)).unwrap();
    let mut b = PpoConfig::parse(&text("run-b", 2)).unwrap();
    b.threads = 3;
    run_ppo(&a, false, &mut |_| {}).unwrap();
    run_ppo(&b, false, &mut |_| {}).unwrap();
    let (sa, sb) = (state_of(&f.dir.join("run-a")), state_of(&f.dir.join("run-b")));
    assert_eq!(sa.iteration, 2);
    assert_eq!(
        sa.theta, sb.theta,
        "the parameters are bit-identical on 1 and on 3 threads"
    );
    assert_eq!(sa.critic, sb.critic);
    assert_eq!(sa.adam.m, sb.adam.m);
    assert_eq!((sa.beta_kl, sa.snapshots.len()), (sb.beta_kl, sb.snapshots.len()));
    // The policy moved (iteration 1 trains it; iteration 0 only the critic), and so did the new channels' weights.
    let start = load_bundle(&f.bundle).unwrap();
    assert_ne!(crate::es::space::flatten(&start), sa.theta);
    let l = FlyLearner::from_bundle(start.clone(), &f.flyg, FlyTrainConfig::default()).unwrap();
    let g0 = l.layout().a + l.layout().b + l.layout().theta;
    let moved = l
        .encoder()
        .assignments()
        .iter()
        .filter(|a| {
            ddai_fly::encoder::OPPONENT_CHANNELS
                .iter()
                .any(|c| c.name() == a.channel)
        })
        .any(|a| sa.theta[g0 + a.param_id as usize] != 0.0);
    assert!(moved, "no gradient reached the opponent-state channels");
    for file in [
        "config.toml",
        "metrics.jsonl",
        "status.json",
        "state.bin",
        "checkpoints/final.bundle",
        "checkpoints/last.bundle",
        "checkpoints/selected.bundle",
        "snapshots/it-00001.bundle",
    ] {
        assert!(f.dir.join("run-a").join(file).exists(), "{file}");
    }
    let metrics = std::fs::read_to_string(f.dir.join("run-a/metrics.jsonl")).unwrap();
    let rows: Vec<serde_json::Value> = metrics.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    for k in ["train", "ppo", "ppo_eval", "arena", "selection"] {
        assert!(rows.iter().any(|r| r["kind"] == k), "no {k} line");
    }
    let ppo: Vec<&serde_json::Value> = rows.iter().filter(|r| r["kind"] == "ppo").collect();
    assert_eq!(ppo.len(), 2);
    assert_eq!(
        ppo[0]["policy_updated"], false,
        "the critic warm-up iteration does not move the policy"
    );
    assert_eq!(ppo[1]["policy_updated"], true);
    assert!(ppo[1]["bc_loss"].as_f64().unwrap() > 0.0, "the BC term ran");
    assert!(ppo[1]["new_g_norm"].as_f64().unwrap() + ppo[1]["new_c_norm"].as_f64().unwrap() > 0.0);
    // A run stopped after one iteration and continued ends where the continuous one does.
    run_ppo(&PpoConfig::parse(&text("run-c", 1)).unwrap(), false, &mut |_| {}).unwrap();
    assert_eq!(state_of(&f.dir.join("run-c")).iteration, 1);
    run_ppo(&PpoConfig::parse(&text("run-c", 2)).unwrap(), false, &mut |_| {}).unwrap();
    let sc = state_of(&f.dir.join("run-c"));
    assert_eq!((sc.iteration, &sc.theta, &sc.critic), (2, &sa.theta, &sa.critic));
    // Changing something that matters is refused on resume, and accepted with the flag.
    let mut changed = PpoConfig::parse(&text("run-c", 3)).unwrap();
    changed.ppo.clip = 0.1;
    let e = run_ppo(&changed, false, &mut |_| {}).unwrap_err();
    assert!(e.contains("refusing to resume"), "{e}");
    run_ppo(&changed, true, &mut |_| {}).unwrap();
    assert!(f.dir.join("run-c/config-before-1.toml").exists());
}

#[test]
fn a_holdout_arena_is_refused_for_training() {
    let f = fixture(HookView::Shared);
    f.bank.save(&f.dir.join("bank.bin")).unwrap();
    let mut c = PpoConfig::parse(&config_text(&f, "run-h", 1, None)).unwrap();
    c.train_arenas = vec!["clb-right".into()];
    let e = run_ppo(&c, false, &mut |_| {}).unwrap_err();
    assert!(e.contains("clb-right"), "{e}");
}

#[test]
fn the_iteration_plan_is_a_function_of_the_seed_and_the_iteration() {
    let f = fixture(HookView::Shared);
    let c = PpoConfig::parse(&config_text(&f, "run-p", 1, None)).unwrap();
    let classes: [Vec<usize>; 3] = [(0..20).collect(), (20..35).collect(), (35..50).collect()];
    let p1 = plan_of(&c, 5, &classes, 2);
    assert_eq!(p1, plan_of(&c, 5, &classes, 2));
    assert_ne!(p1, plan_of(&c, 6, &classes, 2));
    assert_eq!(p1.len(), c.rollout.post_episodes + c.rollout.game_episodes);
    // Only class V when it alone has weight, and a `past` opponent only picks a snapshot that exists.
    let mut c2 = c.clone();
    c2.rollout.start_mix = [1.0, 0.0, 0.0];
    c2.rollout.post_episodes = 40;
    for p in plan_of(&c2, 1, &classes, 0) {
        if let super::Pick::Post(i) = p.pick {
            assert!(i < 20);
        }
        assert!(p.past.is_none());
    }
    // The mix is followed on average, and an empty class is never drawn.
    c2.rollout.start_mix = [1.0, 1.0, 2.0];
    c2.rollout.post_episodes = 400;
    let mut counts = [0usize; 3];
    for p in plan_of(&c2, 3, &classes, 0) {
        if let super::Pick::Post(i) = p.pick {
            counts[if i < 20 {
                0
            } else if i < 35 {
                1
            } else {
                2
            }] += 1;
        }
    }
    assert!(
        counts[2] > counts[0] + 60 && counts[0].abs_diff(counts[1]) < 70,
        "{counts:?}"
    );
    let no_h: [Vec<usize>; 3] = [(0..20).collect(), (20..35).collect(), Vec::new()];
    for p in plan_of(&c2, 3, &no_h, 0) {
        if let super::Pick::Post(i) = p.pick {
            assert!(i < 35);
        }
    }
    let _ = rollout;
    let _: Option<&Decision> = None;
    let _: Option<FlyBundle> = None;
}

#[test]
fn the_curriculum_plan_mixes_levels_and_a_curriculum_episode_replays_the_demonstration() {
    use super::curriculum::{CurriculumState, build_demos, resumed_log};
    let f = fixture(HookView::Shared);
    f.bank.save(&f.dir.join("bank.bin")).unwrap();
    let mut c = PpoConfig::parse(&config_text(&f, "run-cur", 1, None)).unwrap();
    c.curriculum.enabled = true;
    c.curriculum.demos = f.dir.join("demos.bin").display().to_string();
    c.curriculum.step = 50;
    c.rollout.post_episodes = 20;
    // The plan: half of the post-freeze episodes are curriculum ones, at the current level or an easier one.
    let classes: [Vec<usize>; 3] = [(0..20).collect(), (20..35).collect(), (35..50).collect()];
    let demos: [Vec<usize>; 3] = [(0..10).collect(), (20..25).collect(), Vec::new()];
    let mut st = CurriculumState::new(&c.curriculum);
    st.offsets = [100, 150, 100];
    let p = super::iteration_plan(&c, 4, &classes, &demos, &st, 0);
    let cur: Vec<(usize, i32)> = p
        .iter()
        .filter_map(|p| match p.pick {
            super::Pick::Demo(i, o) => Some((i, o)),
            _ => None,
        })
        .collect();
    assert_eq!(cur.len(), 10);
    // Every class has its own ladder: V from 100 (its first level is 104, step 50: nothing easier), B from 150 up to its first level 236.
    assert!(
        cur.iter()
            .all(|&(i, o)| (i < 10 && o == 100) || ((20..25).contains(&i) && [150, 200].contains(&o))),
        "{cur:?}"
    );
    assert!(cur.iter().any(|&(i, o)| i < 10 && o == 100) && cur.iter().any(|&(i, o)| i >= 20 && o == 150));
    assert_eq!(p.len(), 20 + c.rollout.game_episodes);
    assert_eq!(p, super::iteration_plan(&c, 4, &classes, &demos, &st, 0));
    // Demonstrations of the tiny bank: the planner plays, and a replayed hand-over reproduces the planner's own game when the fly is the
    // planner again (the open-loop log equals what it did).
    let starts: Vec<&crate::bank::BankStart> = f.bank.starts.iter().take(6).collect();
    let pool = super::make_pool(2).unwrap();
    let set = build_demos(&f.env, &f.bank, &starts, &pool, &mut |_| {}).unwrap();
    assert_eq!(set.demos.len(), 6);
    let mut checked = 0;
    for s in &starts {
        let Some(d) = set.held_demo(s) else { continue };
        let first = d.actions.first().unwrap();
        assert!(
            (s.end_tick..s.end_tick + 2).contains(&first.tick),
            "the demonstration starts at the freeze"
        );
        let (log, handover) = resumed_log(s, d, 100);
        assert_eq!(handover, s.end_tick + 100);
        // The log replayed with an idle brain from the handover: up to there it is the planner's game, after it nobody acts.
        let o = super::curriculum::play_resumed(
            &f.env,
            &f.bank.rules,
            s,
            d,
            100,
            Box::new(ddai_brain::IdleBrain),
            250,
            50,
        )
        .unwrap();
        assert!(
            o.credited && o.end_tick == s.end_tick,
            "the replay reproduces the freeze"
        );
        assert!(log.iter().all(|a| a.tick < handover));
        checked += 1;
    }
    let _ = checked;
}

#[test]
fn a_run_with_the_curriculum_and_dagger_is_deterministic_resumable_and_moves_the_level() {
    let f = fixture(HookView::MaskedForHookHead);
    f.bank.save(&f.dir.join("bank.bin")).unwrap();
    let text = |run: &str, it: u64| {
        let mut t = config_text(&f, run, it, None);
        t = t.replace("bc_coef = 0.0", "bc_coef = 0.5");
        t.push_str(&format!(
            "[curriculum]\nenabled = true\ndemos = \"{}\"\nstart_offsets = [200, 200, 200]\nstep = 100\nthreshold = 0.0\nmin_episodes = 2\nshare = 0.5\nmix = [1.0, 1.0, 1.0]\n[dagger]\nevery = 1\nstarts = 3\nmix = [1.0, 1.0, 1.0]\n",
            f.dir.join("demos.bin").display()
        ));
        t
    };
    let a = PpoConfig::parse(&text("run-a", 3)).unwrap();
    let mut b = PpoConfig::parse(&text("run-b", 3)).unwrap();
    b.threads = 3;
    run_ppo(&a, false, &mut |_| {}).unwrap();
    run_ppo(&b, false, &mut |_| {}).unwrap();
    let (sa, sb) = (state_of(&f.dir.join("run-a")), state_of(&f.dir.join("run-b")));
    assert_eq!(
        sa.theta, sb.theta,
        "bit-identical on 1 and 3 threads, curriculum and dagger included"
    );
    assert_eq!(sa.curriculum, sb.curriculum);
    assert_eq!(sa.dagger_ids, sb.dagger_ids);
    assert!(!sa.dagger_ids.is_empty(), "the DAgger rounds added chunks");
    // A threshold of 0 moves a level at every opportunity (of the classes that have demonstrations).
    assert!(!sa.curriculum.moves.is_empty(), "{:?}", sa.curriculum);
    let metrics = std::fs::read_to_string(f.dir.join("run-a/metrics.jsonl")).unwrap();
    let rows: Vec<serde_json::Value> = metrics.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let ppo: Vec<&serde_json::Value> = rows.iter().filter(|r| r["kind"] == "ppo").collect();
    assert!(
        ppo.iter().any(|r| r["curr_n"].as_u64().unwrap() > 0),
        "curriculum episodes were played"
    );
    // Resume: one iteration, then two more, ends where the continuous run does (the DAgger chunks are loaded back exactly).
    run_ppo(&PpoConfig::parse(&text("run-c", 1)).unwrap(), false, &mut |_| {}).unwrap();
    run_ppo(&PpoConfig::parse(&text("run-c", 3)).unwrap(), false, &mut |_| {}).unwrap();
    let sc = state_of(&f.dir.join("run-c"));
    assert_eq!(
        (&sc.theta, &sc.curriculum, &sc.dagger_ids),
        (&sa.theta, &sa.curriculum, &sa.dagger_ids)
    );
}

#[test]
fn a_resumed_episode_warms_the_fly_up_before_the_handover() {
    // Review 8.5b F1: the probe handed over a cold brain. With a burn-in of B ticks the brain decides on every decision of the last B ticks before
    // the handover (its action is discarded, the logged one is played), so its recurrent state is warm; with 0 it is first asked at the handover.
    use super::curriculum::{build_demos, play_resumed, resumed_log};
    let f = fixture(HookView::Shared);
    let starts: Vec<&crate::bank::BankStart> = f.bank.starts.iter().take(6).collect();
    let pool = super::make_pool(2).unwrap();
    let set = build_demos(&f.env, &f.bank, &starts, &pool, &mut |_| {}).unwrap();
    let (s, d) = starts
        .iter()
        .find_map(|s| set.held_demo(s).map(|d| (*s, d)))
        .expect("a held demonstration");
    let asked = |burn: i32| -> Vec<i32> {
        let (rec, log) = ddai_env::brains::RecordingBrain::new(Box::new(ddai_brain::IdleBrain));
        play_resumed(&f.env, &f.bank.rules, s, d, 100, Box::new(rec), 250, burn).unwrap();
        let ticks: Vec<i32> = log.lock().unwrap().iter().map(|(t, _)| *t).collect();
        ticks
    };
    let handover = resumed_log(s, d, 100).1;
    let warm = asked(50);
    let cold = asked(0);
    assert_eq!(
        warm.iter().filter(|&&t| t < handover).count(),
        25,
        "25 decisions of warm-up"
    );
    assert!(warm.iter().filter(|&&t| t < handover).all(|&t| t >= handover - 50));
    assert_eq!(
        cold.iter().filter(|&&t| t < handover).count(),
        0,
        "a cold brain is first asked at the handover"
    );
    assert!(cold.iter().all(|&t| t >= handover));
}
