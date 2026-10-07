//! Task 3.15 (E-028): trains the opponent-input predictor on recorded games and reports its offline accuracy against "hold".
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example opp_train -- --train <dir>... --val <dir>... [--test <dir>] --out model.oppnet \
//!     [--epochs 12] [--h1 160] [--h2 128] [--batch 256] [--lr 0.0015] [--threads 3] [--seed 1] [--lag <n>]
//! cargo run --release -p ddai-oppnet --example opp_train -- --model model.oppnet --test <dir> [--lag <n>]     # evaluation only
//! ```
//! `--lag n` keeps only the games whose own lag is `n` (for the metrics tables); `--gates 0.5,1,2` adds compact tables of the model with its heads gated by a
//! logit margin (a head less sure than that answers "hold").

use std::path::{Path, PathBuf};

use ddai_oppnet::blob::read_blob;
use ddai_oppnet::bundle::OppBundle;
use ddai_oppnet::data::{Chunk, GameRec};
use ddai_oppnet::train::{Corpus, TrainCfg, evaluate, evaluate_gated, train};

fn load_dir(dir: &Path, lag: Option<u8>) -> Result<Vec<GameRec>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "opp"))
        .collect();
    files.sort();
    let mut games = Vec::new();
    for f in files {
        let chunk: Chunk = read_blob(&f)?;
        games.extend(chunk.into_iter().filter(|g| lag.is_none_or(|l| g.lag[0] == l)));
    }
    Ok(games)
}

fn main() -> Result<(), String> {
    let mut train_dirs: Vec<PathBuf> = Vec::new();
    let mut val_dirs: Vec<PathBuf> = Vec::new();
    let mut test_dir = None;
    let mut feat_dir = None;
    let mut out = None;
    let mut model = None;
    let mut lag = None;
    let mut gates: Vec<f32> = Vec::new();
    let mut cfg = TrainCfg::default();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--train" => train_dirs.push(PathBuf::from(v()?)),
            "--val" => val_dirs.push(PathBuf::from(v()?)),
            "--feat-means" => feat_dir = Some(PathBuf::from(v()?)),
            "--test" => test_dir = Some(PathBuf::from(v()?)),
            "--out" => out = Some(PathBuf::from(v()?)),
            "--model" => model = Some(PathBuf::from(v()?)),
            "--gates" => {
                gates = v()?
                    .split(',')
                    .map(|g| g.parse::<f32>().map_err(|e| format!("--gates: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--lag" => lag = Some(v()?.parse::<u8>().map_err(|e| e.to_string())?),
            "--epochs" => cfg.epochs = v()?.parse().map_err(|e| format!("{e}"))?,
            "--h1" => cfg.h1 = v()?.parse().map_err(|e| format!("{e}"))?,
            "--h2" => cfg.h2 = v()?.parse().map_err(|e| format!("{e}"))?,
            "--batch" => cfg.batch = v()?.parse().map_err(|e| format!("{e}"))?,
            "--lr" => cfg.lr = v()?.parse().map_err(|e| format!("{e}"))?,
            "--threads" => cfg.threads = v()?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => cfg.seed = v()?.parse().map_err(|e| format!("{e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let bundle = if let Some(m) = &model {
        OppBundle::load(m)?
    } else {
        let Some(out) = &out else {
            return Err(
                "usage: opp_train --train <dir>... --val <dir>... --out <file> | --model <file> --test <dir>".into(),
            );
        };
        if train_dirs.is_empty() || val_dirs.is_empty() {
            return Err(
                "usage: opp_train --train <dir>... --val <dir>... --out <file> | --model <file> --test <dir>".into(),
            );
        }
        let t0 = std::time::Instant::now();
        let mut train_games = Vec::new();
        for d in &train_dirs {
            train_games.extend(load_dir(d, None)?);
        }
        let mut val_games = Vec::new();
        for d in &val_dirs {
            val_games.extend(load_dir(d, None)?);
        }
        let c_train = Corpus::new(train_games);
        let c_val = Corpus::new(val_games);
        println!(
            "train: {} games, {} samples; val: {} games, {} samples ({:.0}s to load)",
            c_train.games.len(),
            c_train.samples().len(),
            c_val.games.len(),
            c_val.samples().len(),
            t0.elapsed().as_secs_f64()
        );
        let t1 = std::time::Instant::now();
        let (net, val_loss) = train(&c_train, &c_val, &cfg, &mut |l| println!("{l}"))?;
        println!(
            "trained in {:.0}s: best val loss {val_loss:.4}, {} parameters",
            t1.elapsed().as_secs_f64(),
            net.params.len()
        );
        let b = OppBundle::new(
            net,
            cfg.seed,
            cfg.epochs as u32,
            val_loss,
            format!(
                "h1 {} h2 {} batch {} lr {} train games {}",
                cfg.h1,
                cfg.h2,
                cfg.batch,
                cfg.lr,
                c_train.games.len()
            ),
        );
        b.save(out)?;
        println!("saved {}", out.display());
        let val_s = c_val.samples();
        println!(
            "\n## validation (seeds of the val split)\n\n{}",
            evaluate(&b.net, &c_val, &val_s).table()
        );
        for g in &gates {
            println!(
                "\n### validation with the confidence gate at logit margin {g}\n\n{}",
                evaluate_gated(&b.net, &c_val, &val_s, *g).compact()
            );
        }
        b
    };
    if let Some(td) = &feat_dir {
        let c = Corpus::new(load_dir(td, lag)?);
        println!("mean frame features of {}: {:?}", td.display(), c.mean_features());
    }
    if let Some(td) = &test_dir {
        let c = Corpus::new(load_dir(td, lag)?);
        let s = c.samples();
        println!(
            "\n## test ({} games{})\n\n{}",
            c.games.len(),
            lag.map_or(String::new(), |l| format!(", our lag {l}")),
            evaluate(&bundle.net, &c, &s).table()
        );
        for g in &gates {
            println!(
                "\n### test with the confidence gate at logit margin {g}\n\n{}",
                evaluate_gated(&bundle.net, &c, &s, *g).compact()
            );
        }
    }
    Ok(())
}
