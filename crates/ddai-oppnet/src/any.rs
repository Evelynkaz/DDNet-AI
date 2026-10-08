//! One loader for the model files of both generations: [`AnyPredictor`] is the v1 predictor (`m1`, task 3.15) or the v2 one (task 3.21), told apart by the
//! file's version header, behind the same `WindowModel` interface. The live module and the arena use it, so a model file of either kind runs where the other did.

use std::path::Path;

use ddai_planner::hybrid::window::{PredictedInput, WindowCtx, WindowModel};
use serde::Deserialize;

use crate::blob::read_blob;
use crate::predictor::OppPredictor;
use crate::v2::feature::KnownTick;
use crate::v2::predictor::Predictor;

/// The first two fields of both bundle structs (the file starts with them; the rest is ignored here).
#[derive(Deserialize)]
struct Head {
    format_version: u32,
    feature_version: u32,
}

pub enum AnyPredictor {
    V1(OppPredictor),
    V2(Predictor),
}

impl AnyPredictor {
    /// Loads a model file of either generation; a file of neither (or a corrupt one) is an error.
    pub fn load(path: &Path) -> Result<AnyPredictor, String> {
        let head: Head = read_blob(path)?;
        match (head.format_version, head.feature_version) {
            (1, 1) => OppPredictor::load(path).map(AnyPredictor::V1),
            (2, 2) => Predictor::load(path).map(AnyPredictor::V2),
            (f, v) => Err(format!(
                "{}: opponent model of format {f}, feature layout {v}: this build reads 1/1 and 2/2",
                path.display()
            )),
        }
    }

    /// 1 or 2.
    pub fn generation(&self) -> u32 {
        match self {
            AnyPredictor::V1(_) => 1,
            AnyPredictor::V2(_) => 2,
        }
    }

    /// The longest window the model is asked about (its in-flight slots and its one-hot length).
    pub fn max_window(&self) -> usize {
        match self {
            AnyPredictor::V1(_) => crate::feature::IF_SLOTS,
            AnyPredictor::V2(_) => crate::v2::feature::MAX_LAG,
        }
    }

    /// The opponent's real inputs for the first ticks of the next window (pre-inputs). Only v2 reads them; v1 ignores them.
    pub fn set_known(&mut self, known: &[Option<KnownTick>]) {
        if let AnyPredictor::V2(p) = self {
            p.set_known(known);
        }
    }
}

impl From<OppPredictor> for AnyPredictor {
    fn from(p: OppPredictor) -> AnyPredictor {
        AnyPredictor::V1(p)
    }
}

impl From<Predictor> for AnyPredictor {
    fn from(p: Predictor) -> AnyPredictor {
        AnyPredictor::V2(p)
    }
}

impl WindowModel for AnyPredictor {
    fn name(&self) -> &str {
        match self {
            AnyPredictor::V1(p) => p.name(),
            AnyPredictor::V2(p) => p.name(),
        }
    }

    fn reset(&mut self) {
        match self {
            AnyPredictor::V1(p) => p.reset(),
            AnyPredictor::V2(p) => p.reset(),
        }
    }

    fn predict(&mut self, ctx: &WindowCtx<'_>, out: &mut [Option<PredictedInput>]) {
        match self {
            AnyPredictor::V1(p) => p.predict(ctx, out),
            AnyPredictor::V2(p) => p.predict(ctx, out),
        }
    }

    fn work_units(&self) -> u64 {
        match self {
            AnyPredictor::V1(p) => p.work_units(),
            AnyPredictor::V2(p) => p.work_units(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::OppBundle;
    use crate::net::Mlp;
    use crate::v2::predictor::{Bundle, Decode};

    #[test]
    fn the_loader_tells_the_generations_apart_and_refuses_junk() {
        let dir = tempfile::tempdir().unwrap();
        let (p1, p2, junk) = (
            dir.path().join("a.oppnet"),
            dir.path().join("b.oppnet"),
            dir.path().join("c.oppnet"),
        );
        OppBundle::new(
            Mlp::new(crate::feature::INPUT_DIM, 8, 4, crate::feature::OUT_DIM, 1),
            1,
            1,
            0.0,
            "m1".into(),
        )
        .save(&p1)
        .unwrap();
        Bundle::new(
            Mlp::new(crate::v2::feature::INPUT_DIM, 8, 4, crate::v2::feature::OUT_DIM, 1),
            Decode::default(),
            1,
            1,
            0.0,
            "m2".into(),
        )
        .save(&p2)
        .unwrap();
        let (a, b) = (AnyPredictor::load(&p1).unwrap(), AnyPredictor::load(&p2).unwrap());
        assert_eq!((a.generation(), a.max_window()), (1, 8));
        assert_eq!((b.generation(), b.max_window()), (2, 4));
        std::fs::write(&junk, b"nothing").unwrap();
        assert!(AnyPredictor::load(&junk).is_err());
        assert!(AnyPredictor::load(&dir.path().join("none")).is_err());
    }
}
