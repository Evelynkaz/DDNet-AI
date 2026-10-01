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
use ddai_fly::brain::{ActionSelection, FlyBrainConfig};
use ddai_fly::bundle::FlyBrainTemplate;

use crate::EnvError;
use crate::config::{HybridSpec, PlayerSpec, builtin_brain, hybrid_config};

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
    if let Some(path) = arg.strip_prefix("hybrid:fly:") {
        return format!("hybrid+{}", label_of_arg(&format!("fly:{path}")));
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
    fn hybrid_with_fly(&self, spec: &PlayerSpec) -> Result<Box<dyn Brain>, EnvError> {
        let h = spec.hybrid.as_ref().expect("checked by the caller");
        let (cfg, clock) = hybrid_config(spec)?;
        let seed = 1;
        let brain = if let Some(bundle) = &h.fly_model {
            let Loaded::Fly(t) = &*self.load("fly", bundle)? else {
                return Err(EnvError::new("hybrid fly_model must be a fly bundle"));
            };
            t.instantiate(FlyBrainConfig::default())
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
            ddai_fly::proposer::untrained_fly_brain(&flyg, &cfg_path, seed).map_err(EnvError::new)?
        };
        let proposer = Box::new(ddai_fly::proposer::FlyProposer::new(brain, seed));
        let hybrid = ddai_planner::hybrid::HybridBrain::new(cfg, clock, proposer).map_err(EnvError::new)?;
        Ok(Box::new(hybrid))
    }

    /// A brain for `spec`: a model brain when it names one, `hybrid` with the fly as proposer, else
    /// [`builtin_brain`].
    pub fn make(&self, spec: &PlayerSpec) -> Result<Box<dyn Brain>, EnvError> {
        if spec.brain == "hybrid" && spec.hybrid.as_ref().is_some_and(|h| h.proposer_name() == "fly") {
            return self.hybrid_with_fly(spec);
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
            Loaded::Fly(t) => Box::new(t.instantiate(FlyBrainConfig {
                action_selection: selection,
                ..FlyBrainConfig::default()
            })),
            Loaded::Control(t) => Box::new(t.instantiate_with(selection)),
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
}
