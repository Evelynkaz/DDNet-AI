//! Brains that live in files (task 8.2): the fly and the MLP/GRU controls, loaded from their
//! checkpoints, next to the built-in ones of [`crate::config::builtin_brain`].
//!
//! A batch plays thousands of games and each game needs its own brain, so a checkpoint is loaded
//! and validated **once** ([`ModelBrains`] caches the loaded template per file) and every game
//! gets a fresh brain from it ([`FlyBrainTemplate::instantiate`], [`ControlTemplate::instantiate`]).
//!
//! Spec syntax: a player is `{ brain = "fly", model = "<checkpoint>" }` in a run config, or, on a
//! command line, `fly:<checkpoint>` / `mlp:<checkpoint>` / `gru:<checkpoint>` ([`player_from_arg`]).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ddai_brain::Brain;
use ddai_controls::ControlTemplate;
use ddai_fly::bc::HookView;
use ddai_fly::brain::{ActionSelection, FlyBrainConfig};
use ddai_fly::bundle::FlyBrainTemplate;

use crate::EnvError;
use crate::config::{HybridSpec, PlayerSpec, builtin_brain, hybrid_config};

pub use ddai_fly::two_view::TwoViewBrain;

enum Loaded {
    Fly(Arc<FlyBrainTemplate>),
    Control(ControlTemplate),
}

/// Loads model brains once per file and hands out a fresh brain per call.
pub struct ModelBrains {
    /// Overrides the `.flyg` path stored in a fly checkpoint (the hash still has to match).
    flyg: Option<PathBuf>,
    cache: Mutex<HashMap<(String, String), Arc<Loaded>>>,
}

/// The model kinds a spec can name.
const KINDS: [&str; 3] = ["fly", "mlp", "gru"];

/// Splits `kind:path`; `None` when `brain` is a plain name.
fn split_kind(brain: &str) -> Option<(&str, &str)> {
    let (kind, path) = brain.split_once(':')?;
    KINDS.contains(&kind).then_some((kind, path))
}

/// A player spec from a command-line brain argument: `idle`, `scripted`, `planner` or
/// `fly:<checkpoint>` / `mlp:<checkpoint>` / `gru:<checkpoint>` (a leading `~/` is expanded); a model
/// argument may end in `#sampled` to draw every head from its probabilities instead of taking the
/// most likely value.
pub fn player_from_arg(arg: &str) -> PlayerSpec {
    // `hybrid:mlp:<bundle>` / `hybrid:gru:<bundle>`: the hybrid with a trained control network as its proposer.
    for kind in ["mlp", "gru"] {
        if let Some(path) = arg.strip_prefix(&format!("hybrid:{kind}:")) {
            let mut spec = PlayerSpec::simple("hybrid");
            spec.hybrid = Some(HybridSpec {
                proposer: Some(kind.to_string()),
                control_model: Some(expand_home(path).to_string_lossy().into_owned()),
                ..HybridSpec::default()
            });
            return spec;
        }
    }
    // `hybrid:fly:<bundle>`: the hybrid brain with a trained fly as its proposer.
    if let Some(path) = arg.strip_prefix("hybrid:fly:") {
        let mut spec = PlayerSpec::simple("hybrid");
        spec.hybrid = Some(HybridSpec {
            proposer: Some("fly".to_string()),
            fly_model: Some(expand_home(path).to_string_lossy().into_owned()),
            ..HybridSpec::default()
        });
        return spec;
    }
    match split_kind(arg) {
        Some((kind, rest)) => {
            let (path, select) = match rest.rsplit_once('#') {
                Some((p, sel)) => (p, Some(sel.to_string())),
                None => (rest, None),
            };
            let mut spec = PlayerSpec::simple(kind);
            spec.model = Some(expand_home(path).to_string_lossy().into_owned());
            spec.select = select;
            spec
        }
        None => PlayerSpec::simple(arg),
    }
}

/// A short table label for a `--brain` argument: the run directory of a checkpoint under
/// `<run>/checkpoints/` or `<run>/rounds/` (`fly:.../e005-fly/checkpoints/final.bundle` becomes
/// `e005-fly`, plus the `#sampled` marker when present), else the file stem or the argument itself.
pub fn label_of_arg(arg: &str) -> String {
    for kind in ["fly", "mlp", "gru"] {
        if let Some(path) = arg.strip_prefix(&format!("hybrid:{kind}:")) {
            return format!("hybrid+{}", label_of_arg(&format!("{kind}:{path}")));
        }
    }
    let Some((_, rest)) = split_kind(arg) else {
        return arg.to_string();
    };
    let (path, select) = match rest.rsplit_once('#') {
        Some((p, s)) => (p, Some(s)),
        None => (rest, None),
    };
    let p = std::path::Path::new(path);
    let stem = p
        .file_stem()
        .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    let in_run_dir = p
        .parent()
        .and_then(|d| d.file_name())
        .is_some_and(|d| d == "checkpoints" || d == "rounds");
    let base = if in_run_dir {
        let run = p.parent().and_then(|d| d.parent()).and_then(|d| d.file_name());
        let run = run.map_or_else(|| stem.clone(), |r| r.to_string_lossy().into_owned());
        if p.parent().and_then(|d| d.file_name()).is_some_and(|d| d == "rounds") {
            format!("{run}@{stem}")
        } else {
            run
        }
    } else {
        stem
    };
    match select {
        Some(s) => format!("{base}#{s}"),
        None => base,
    }
}

fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(h)) => PathBuf::from(h).join(rest),
        _ => PathBuf::from(p),
    }
}

impl ModelBrains {
    pub fn new(flyg_override: Option<PathBuf>) -> Self {
        ModelBrains {
            flyg: flyg_override,
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn load(&self, kind: &str, path: &str) -> Result<Arc<Loaded>, EnvError> {
        let key = (kind.to_string(), path.to_string());
        if let Some(l) = self
            .cache
            .lock()
            .map_err(|_| EnvError::new("model cache poisoned"))?
            .get(&key)
        {
            return Ok(l.clone());
        }
        let file = expand_home(path);
        let loaded = match kind {
            "fly" => {
                let t = FlyBrainTemplate::load(&file, self.flyg.as_deref())
                    .map_err(|e| EnvError::new(format!("fly model {}: {e}", file.display())))?;
                Loaded::Fly(Arc::new(t))
            }
            _ => Loaded::Control(
                ControlTemplate::load(&file)
                    .map_err(|e| EnvError::new(format!("{kind} model {}: {e}", file.display())))?,
            ),
        };
        let loaded = Arc::new(loaded);
        self.cache
            .lock()
            .map_err(|_| EnvError::new("model cache poisoned"))?
            .insert(key, loaded.clone());
        Ok(loaded)
    }

    /// The `hybrid` brain whose proposer is the fly: a trained bundle (`hybrid.fly_model`), or the
    /// untrained fly of the `.flyg` named by `model` (task 3.5: plumbing and cost only).
    fn hybrid_with_fly(&self, spec: &PlayerSpec) -> Result<ddai_planner::hybrid::HybridBrain, EnvError> {
        let h = spec.hybrid.as_ref().expect("checked by the caller");
        let (cfg, clock) = hybrid_config(spec)?;
        let seed = 1;
        let proposer = if let Some(bundle) = &h.fly_model {
            let Loaded::Fly(t) = &*self.load("fly", bundle)? else {
                return Err(EnvError::new("hybrid fly_model must be a fly bundle"));
            };
            // A model trained with the hook head masked proposes through its second view (`from_template`).
            ddai_fly::proposer::FlyProposer::from_template(t, FlyBrainConfig::default(), seed)
        } else {
            let flyg = spec.model.as_deref().ok_or_else(|| {
                EnvError::new("hybrid with proposer = \"fly\" needs model = <.flyg path> or fly_model = <bundle>")
            })?;
            let flyg = expand_home(flyg);
            let cfg_path = h.fly_config.as_deref().map_or_else(
                || {
                    let m = flyg.file_name().is_some_and(|n| n.to_string_lossy().contains("-M-"));
                    PathBuf::from(if m {
                        "configs/fly/M-brain.toml"
                    } else {
                        "configs/fly/S-brain.toml"
                    })
                },
                expand_home,
            );
            let brain = ddai_fly::proposer::untrained_fly_brain(&flyg, &cfg_path, seed).map_err(EnvError::new)?;
            ddai_fly::proposer::FlyProposer::new(brain, seed)
        };
        let proposer = Box::new(proposer);
        let hybrid = ddai_planner::hybrid::HybridBrain::new(cfg, clock, proposer).map_err(EnvError::new)?;
        Ok(hybrid)
    }

    /// The `hybrid` brain whose proposer is a trained MLP or GRU (`hybrid.control_model`).
    fn hybrid_with_control(&self, spec: &PlayerSpec) -> Result<Box<dyn Brain>, EnvError> {
        let h = spec.hybrid.as_ref().expect("checked by the caller");
        let kind = h.proposer_name();
        let bundle = h.control_model.as_deref().ok_or_else(|| {
            EnvError::new(format!(
                "hybrid with proposer = {kind:?} needs control_model = <bundle>"
            ))
        })?;
        let Loaded::Control(t) = &*self.load(kind, bundle)? else {
            return Err(EnvError::new("hybrid control_model must be a control bundle"));
        };
        let (cfg, clock) = hybrid_config(spec)?;
        let seed = 1;
        let mut proposer = ddai_controls::proposer::ControlProposer::new(t.instantiate(), seed);
        if t.hook_view() == HookView::MaskedForHookHead {
            proposer = proposer.with_hook_brain(t.instantiate());
        }
        let proposer = Box::new(proposer);
        let hybrid = ddai_planner::hybrid::HybridBrain::new(cfg, clock, proposer).map_err(EnvError::new)?;
        Ok(Box::new(hybrid))
    }

    /// A brain for `spec`: a model brain when it names one, `hybrid` with the fly as proposer, else
    /// [`builtin_brain`].
    pub fn make(&self, spec: &PlayerSpec) -> Result<Box<dyn Brain>, EnvError> {
        if spec.brain == "hybrid" && spec.hybrid.as_ref().is_some_and(|h| h.proposer_name() == "fly") {
            return self.hybrid_with_fly(spec).map(|h| Box::new(h) as Box<dyn Brain>);
        }
        if spec.brain == "hybrid"
            && spec
                .hybrid
                .as_ref()
                .is_some_and(|h| matches!(h.proposer_name(), "mlp" | "gru"))
        {
            return self.hybrid_with_control(spec);
        }
        let (kind, path) = match split_kind(&spec.brain) {
            Some((k, p)) => (k, Some(p.to_string())),
            None if KINDS.contains(&spec.brain.as_str()) => (spec.brain.as_str(), spec.model.clone()),
            None => return builtin_brain(spec),
        };
        let path = path.or_else(|| spec.model.clone()).ok_or_else(|| {
            EnvError::new(format!(
                "brain {kind:?} needs a model path (`model = \"...\"` or `{kind}:<path>`)"
            ))
        })?;
        let selection = match spec.select.as_deref().unwrap_or("argmax") {
            "argmax" => ActionSelection::Argmax,
            "sampled" => ActionSelection::Sampled,
            other => {
                return Err(EnvError::new(format!(
                    "select must be argmax or sampled, got {other:?}"
                )));
            }
        };
        Ok(match &*self.load(kind, &path)? {
            Loaded::Fly(t) => t.instantiate_played(FlyBrainConfig {
                action_selection: selection,
                ..FlyBrainConfig::default()
            }),
            Loaded::Control(t) => {
                if t.hook_view() == HookView::MaskedForHookHead {
                    Box::new(TwoViewBrain::new(
                        Box::new(t.instantiate_with(selection)),
                        Box::new(t.instantiate_with(selection)),
                    ))
                } else {
                    Box::new(t.instantiate_with(selection))
                }
            }
        })
    }

    /// The closure form the arena's batch functions take (`BrainFactory` is a `'static` trait
    /// object, so the closure owns a handle to the loader).
    pub fn factory(self: &Arc<Self>) -> impl Fn(&PlayerSpec) -> Result<Box<dyn Brain>, EnvError> + Sync + 'static {
        let me = Arc::clone(self);
        move |spec| me.make(spec)
    }

    /// Path of the model a spec names, if it names one (for logs).
    pub fn model_path(spec: &PlayerSpec) -> Option<PathBuf> {
        match split_kind(&spec.brain) {
            Some((_, p)) => Some(expand_home(p)),
            None => KINDS
                .contains(&spec.brain.as_str())
                .then(|| spec.model.as_deref().map(expand_home))
                .flatten(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_arguments_become_player_specs() {
        let s = player_from_arg("fly:/tmp/x/ckpt.bin");
        assert_eq!((s.brain.as_str(), s.model.as_deref()), ("fly", Some("/tmp/x/ckpt.bin")));
        let s = player_from_arg("gru:/a/b");
        assert_eq!((s.brain.as_str(), s.model.as_deref()), ("gru", Some("/a/b")));
        assert_eq!(s.select, None);
        let s = player_from_arg("mlp:/a/b.bundle#sampled");
        assert_eq!(
            (s.model.as_deref(), s.select.as_deref()),
            (Some("/a/b.bundle"), Some("sampled"))
        );
        let s = player_from_arg("planner");
        assert_eq!((s.brain.as_str(), s.model.as_deref()), ("planner", None));
        // An unknown prefix is left alone (the built-in factory will reject the name).
        assert_eq!(player_from_arg("banana:/x").brain, "banana:/x");
        let s = player_from_arg("hybrid:fly:/a/runs/e005-fly/checkpoints/final.bundle");
        let h = s.hybrid.as_ref().unwrap();
        assert_eq!(s.brain, "hybrid");
        assert_eq!(
            (h.proposer_name(), h.fly_model.as_deref()),
            ("fly", Some("/a/runs/e005-fly/checkpoints/final.bundle"))
        );
        assert_eq!(
            label_of_arg("hybrid:fly:/a/runs/e005-fly/checkpoints/final.bundle"),
            "hybrid+e005-fly"
        );
    }

    #[test]
    fn labels_name_the_run_not_the_checkpoint_file() {
        assert_eq!(
            label_of_arg("fly:/x/runs/e005-fly/checkpoints/final.bundle"),
            "e005-fly"
        );
        assert_eq!(
            label_of_arg("fly:/x/runs/e005-fly/rounds/round-2.bundle#sampled"),
            "e005-fly@round-2#sampled"
        );
        assert_eq!(label_of_arg("mlp:/tmp/a.bundle"), "a");
        assert_eq!(label_of_arg("planner"), "planner");
    }

    #[test]
    fn builtin_names_pass_through_and_model_kinds_need_a_path() {
        let m = ModelBrains::new(None);
        assert_eq!(m.make(&PlayerSpec::simple("scripted")).unwrap().name(), "scripted");
        assert!(m.make(&PlayerSpec::simple("nope")).is_err());
        let e = m.make(&PlayerSpec::simple("fly")).err().unwrap();
        assert!(e.to_string().contains("needs a model path"), "{e}");
        let e = m.make(&player_from_arg("mlp:/definitely/not/here.bin")).err().unwrap();
        assert!(e.to_string().contains("mlp model"), "{e}");
    }

    /// A random MLP whose hook logit depends strongly on the own hook input (feature index `d - 4`): one
    /// weight of the hook output row is large, so a model that sees the own hook state follows it.
    fn own_hook_follower(hook_view: HookView) -> (tempfile::TempDir, String) {
        use ddai_controls::bundle::{ControlBundle, save_control_bundle};
        use ddai_controls::features::input_dim;
        use ddai_controls::mlp::Mlp;
        use ddai_controls::net::SeqNet;
        use ddai_fly::bc::HeadThresholds;
        use ddai_fly::bundle::BundleMeta;
        use ddai_fly::encoder::RayGridConfig;
        let cfg = RayGridConfig::default();
        let d = input_dim(&cfg);
        let mut net = Mlp::new(d, 4, 3);
        // Hidden unit 0 reads the own-hook feature only; the hook head (output row 4 of 8) reads hidden unit 0.
        let h = 4usize;
        let p = net.params_mut();
        p.fill(0.0);
        p[d - 4] = 6.0; // W1[0][own_hook]
        let head = h * d + h; // start of the head weights
        p[head + 4 * h] = 6.0; // hook head, hidden unit 0
        p[head + 8 * h + 4] = -3.0; // hook head bias: off unless the own hook is out
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.bundle");
        let b = ControlBundle::from_net(&net, cfg, BundleMeta::default(), HeadThresholds::default(), hook_view);
        save_control_bundle(&path, &b).unwrap();
        let s = path.to_string_lossy().into_owned();
        (dir, s)
    }

    fn obs_with_own_hook(state: i32) -> ddai_brain::Observation {
        let map = std::sync::Arc::new(ddai_physics::map::MapData {
            width: 4,
            height: 4,
            game: vec![Default::default(); 16],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        });
        let mut me = ddai_brain::CharacterObservation::at_rest(0);
        me.hook_state = state;
        ddai_brain::Observation {
            map,
            tick: 0,
            self_state: me,
            others: vec![ddai_brain::CharacterObservation::at_rest(1)],
            target_id: Some(1),
            tuning: ddai_physics::tuning::TuningParams::default(),
        }
    }

    #[test]
    fn a_model_trained_with_the_hook_head_masked_is_played_in_two_views() {
        let reset = |b: &mut Box<dyn Brain>, obs: &ddai_brain::Observation| {
            b.reset(&ddai_brain::ResetContext {
                map: obs.map.clone(),
                self_id: 0,
                seed: 1,
            });
        };
        let (_k1, shared) = own_hook_follower(HookView::Shared);
        let (_k2, masked) = own_hook_follower(HookView::MaskedForHookHead);
        let m = ModelBrains::new(None);
        let (idle, grabbed) = (
            obs_with_own_hook(ddai_brain::HOOK_IDLE),
            obs_with_own_hook(ddai_brain::HOOK_GRABBED),
        );

        // A shared-view model follows its own hook state (the copycat shortcut, built in on purpose).
        let mut b = m.make(&player_from_arg(&format!("mlp:{shared}"))).unwrap();
        reset(&mut b, &idle);
        let (a_idle, a_grabbed) = (b.decide(&idle), b.decide(&grabbed));
        assert!(
            !a_idle.hook && a_grabbed.hook,
            "the shared model copies its own hook state"
        );

        // The masked one cannot: its hook head sees the observation without the own hook state.
        let mut b = m.make(&player_from_arg(&format!("mlp:{masked}"))).unwrap();
        assert!(b.name().ends_with("+hookview"), "{}", b.name());
        reset(&mut b, &idle);
        let (a_idle, a_grabbed) = (b.decide(&idle), b.decide(&grabbed));
        assert_eq!(
            a_idle.hook, a_grabbed.hook,
            "the hook decision does not depend on the own hook state"
        );
        // A fly bundle path that is not a fly bundle is an error naming the problem.
        let e = m.make(&player_from_arg(&format!("hybrid:fly:{masked}"))).err().unwrap();
        assert!(e.to_string().contains("fly model"), "{e}");
    }

    #[test]
    fn a_trained_control_can_be_the_hybrids_proposer() {
        let (_k, shared) = own_hook_follower(HookView::Shared);
        let spec = player_from_arg(&format!("hybrid:mlp:{shared}"));
        let h = spec.hybrid.as_ref().unwrap();
        assert_eq!((spec.brain.as_str(), h.proposer_name()), ("hybrid", "mlp"));
        assert_eq!(h.control_model.as_deref(), Some(shared.as_str()));
        assert_eq!(label_of_arg(&format!("hybrid:gru:{shared}")), "hybrid+m");
        let m = ModelBrains::new(None);
        let b = m.make(&spec).unwrap();
        assert!(b.name().starts_with("hybrid"), "{}", b.name());
        // A missing bundle is an error that names the problem, not a panic.
        let e = m
            .make(&player_from_arg("hybrid:gru:/definitely/not/here.bundle"))
            .err()
            .unwrap();
        assert!(e.to_string().contains("gru model"), "{e}");
        // A model trained with the hook head masked proposes through two views.
        let (_k2, masked) = own_hook_follower(HookView::MaskedForHookHead);
        let b = m.make(&player_from_arg(&format!("hybrid:mlp:{masked}"))).unwrap();
        assert!(b.name().starts_with("hybrid"), "{}", b.name());
    }

    /// Review F8 (round 3): the arena's own wiring plays a masked FLY bundle in two views, standalone and as the hybrid's
    /// proposer, and the work clock charges each network run (twice the price of a one-view proposer).
    #[test]
    fn a_masked_fly_bundle_is_played_in_two_views_by_the_arena_standalone_and_as_a_proposer() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = |view: HookView, substeps: u32| {
            let sub = dir.path().join(format!("{view:?}-{substeps}"));
            std::fs::create_dir_all(&sub).unwrap();
            let (b, _flyg) = ddai_fly::brain_fixtures::write_tiny_fly_bundle_with(&sub, view, substeps);
            b.to_string_lossy().into_owned()
        };
        let m = ModelBrains::new(None);
        let name = |view: HookView| {
            m.make(&player_from_arg(&format!("fly:{}", bundle(view, 4))))
                .unwrap()
                .name()
                .to_string()
        };
        assert!(name(HookView::MaskedForHookHead).ends_with("+hookview"));
        assert!(!name(HookView::Shared).ends_with("+hookview"));

        // Many substeps, so that the proposal price (nnz x substeps x rate) does not round to 0 on the tiny graph.
        let units = |view: HookView| {
            let spec = player_from_arg(&format!("hybrid:fly:{}", bundle(view, 4000)));
            m.hybrid_with_fly(&spec).unwrap().proposer_work_units()
        };
        let one_view = units(HookView::Shared);
        assert!(one_view > 100, "{one_view}");
        assert_eq!(units(HookView::MaskedForHookHead), 2 * one_view);
    }
}
