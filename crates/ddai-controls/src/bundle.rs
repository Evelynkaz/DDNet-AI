//! The control checkpoint: one file holding a network's flat parameters plus what is needed to
//! rebuild it (kind, sizes, the ray-grid configuration its features were built with). Same
//! on-disk discipline as the fly's bundle (`ddai_fly::bundle`: hash + zstd + atomic rename).

use std::path::Path;

use ddai_fly::bc::HeadThresholds;
use ddai_fly::bundle::{BundleError, BundleMeta, decode_payload, peek_version, read_zstd_bytes, write_zstd_postcard};
use ddai_fly::encoder::RayGridConfig;
use serde::{Deserialize, Serialize};

use crate::gru::Gru;
use crate::mlp::Mlp;
use crate::net::{NetKind, SeqNet};

/// Version 2 added [`ControlBundle::thresholds`]; version 1 files still load (default thresholds).
pub const CONTROL_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlBundle {
    pub format_version: u32,
    pub kind: NetKind,
    pub input_dim: usize,
    pub hidden: usize,
    pub ray_grid: RayGridConfig,
    pub params: Vec<f32>,
    pub meta: BundleMeta,
    /// Decision thresholds of the jump/hook/fire heads (format v2).
    pub thresholds: HeadThresholds,
}

/// The format-version-1 layout (no thresholds), kept to read old files.
#[derive(Deserialize)]
struct ControlBundleV1 {
    #[allow(dead_code)] // decoded to keep the layout; the version was peeked already
    format_version: u32,
    kind: NetKind,
    input_dim: usize,
    hidden: usize,
    ray_grid: RayGridConfig,
    params: Vec<f32>,
    meta: BundleMeta,
}

impl ControlBundle {
    pub fn from_net(net: &dyn SeqNet, ray_grid: RayGridConfig, meta: BundleMeta, thresholds: HeadThresholds) -> Self {
        ControlBundle {
            format_version: CONTROL_FORMAT_VERSION,
            kind: net.kind(),
            input_dim: net.input_dim(),
            hidden: net.hidden(),
            ray_grid,
            params: net.params().to_vec(),
            meta,
            thresholds,
        }
    }

    /// Rebuilds the network; checks the parameter count and that every value is finite.
    pub fn build(&self) -> Result<Box<dyn SeqNet>, BundleError> {
        if self.params.iter().any(|x| !x.is_finite()) {
            return Err(BundleError(
                "control bundle contains a non-finite parameter".to_string(),
            ));
        }
        let bad = || {
            BundleError(format!(
                "{}: parameter count {} does not match the sizes",
                self.kind.name(),
                self.params.len()
            ))
        };
        Ok(match self.kind {
            NetKind::Mlp => {
                Box::new(Mlp::from_params(self.input_dim, self.hidden, self.params.clone()).ok_or_else(bad)?)
            }
            NetKind::Gru => {
                Box::new(Gru::from_params(self.input_dim, self.hidden, self.params.clone()).ok_or_else(bad)?)
            }
        })
    }
}

pub fn save_control_bundle(path: &Path, bundle: &ControlBundle) -> Result<(), BundleError> {
    write_zstd_postcard(path, bundle, 3)
}

pub fn load_control_bundle(path: &Path) -> Result<ControlBundle, BundleError> {
    let bytes = read_zstd_bytes(path)?;
    match peek_version(path, &bytes)? {
        CONTROL_FORMAT_VERSION => {
            let b: ControlBundle = decode_payload(path, &bytes, "a v2 bundle")?;
            b.thresholds
                .validate()
                .map_err(|e| BundleError(format!("{}: {e}", path.display())))?;
            Ok(b)
        }
        1 => {
            let b: ControlBundleV1 = decode_payload(path, &bytes, "a v1 bundle")?;
            Ok(ControlBundle {
                format_version: CONTROL_FORMAT_VERSION,
                kind: b.kind,
                input_dim: b.input_dim,
                hidden: b.hidden,
                ray_grid: b.ray_grid,
                params: b.params,
                meta: b.meta,
                thresholds: HeadThresholds::default(),
            })
        }
        v => Err(BundleError(format!(
            "{}: control bundle format version {v} is not supported (expected 1 or {CONTROL_FORMAT_VERSION})",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::input_dim;

    #[test]
    fn bundles_round_trip_for_both_kinds_and_reject_bad_sizes() {
        let cfg = RayGridConfig::default();
        let dir = tempfile::tempdir().unwrap();
        for net in [
            Box::new(Mlp::new(input_dim(&cfg), 3, 1)) as Box<dyn SeqNet>,
            Box::new(Gru::new(input_dim(&cfg), 2, 1)),
        ] {
            let path = dir.path().join(format!("{}.bin", net.kind().name()));
            let b = ControlBundle::from_net(net.as_ref(), cfg, BundleMeta::default(), HeadThresholds::default());
            save_control_bundle(&path, &b).unwrap();
            let back = load_control_bundle(&path).unwrap();
            assert_eq!(back, b);
            let rebuilt = back.build().unwrap();
            assert_eq!(rebuilt.params(), net.params());
            assert_eq!(rebuilt.kind(), net.kind());
        }
        let mut b = ControlBundle::from_net(
            &Mlp::new(20, 3, 1),
            cfg,
            BundleMeta::default(),
            HeadThresholds::default(),
        );
        b.params.pop();
        assert!(b.build().is_err());
        let mut b = ControlBundle::from_net(
            &Mlp::new(20, 3, 1),
            cfg,
            BundleMeta::default(),
            HeadThresholds::default(),
        );
        b.params[0] = f32::NAN;
        assert!(b.build().is_err());
    }

    #[test]
    fn thresholds_round_trip_and_version_one_files_load_with_the_default() {
        let cfg = RayGridConfig::default();
        let dir = tempfile::tempdir().unwrap();
        let net = Mlp::new(input_dim(&cfg), 3, 1);
        let th = HeadThresholds {
            jump: 0.7,
            hook: 0.35,
            fire: 0.9,
        };
        let path = dir.path().join("v2.bin");
        save_control_bundle(&path, &ControlBundle::from_net(&net, cfg, BundleMeta::default(), th)).unwrap();
        assert_eq!(load_control_bundle(&path).unwrap().thresholds, th);

        // A file written by the version-1 layout (what every E-005 checkpoint is).
        #[derive(Serialize)]
        struct V1 {
            format_version: u32,
            kind: NetKind,
            input_dim: usize,
            hidden: usize,
            ray_grid: RayGridConfig,
            params: Vec<f32>,
            meta: BundleMeta,
        }
        let v1 = V1 {
            format_version: 1,
            kind: net.kind(),
            input_dim: net.input_dim(),
            hidden: net.hidden(),
            ray_grid: cfg,
            params: net.params().to_vec(),
            meta: BundleMeta::default(),
        };
        let old = dir.path().join("v1.bin");
        write_zstd_postcard(&old, &v1, 3).unwrap();
        let b = load_control_bundle(&old).unwrap();
        assert_eq!(b.thresholds, HeadThresholds::default());
        assert_eq!(b.build().unwrap().params(), net.params());

        // Unknown versions and invalid thresholds are refused.
        let mut bad = ControlBundle::from_net(&net, cfg, BundleMeta::default(), th);
        bad.format_version = 9;
        let p9 = dir.path().join("v9.bin");
        write_zstd_postcard(&p9, &bad, 3).unwrap();
        assert!(load_control_bundle(&p9).is_err());
        let mut bad = ControlBundle::from_net(&net, cfg, BundleMeta::default(), th);
        bad.thresholds.hook = 1.5;
        let pt = dir.path().join("bad-th.bin");
        write_zstd_postcard(&pt, &bad, 3).unwrap();
        assert!(load_control_bundle(&pt).is_err());
    }
}
