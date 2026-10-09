//! Task 3.21 (E-036): offline evaluation of a v2 model per arena file (one file = one lag cell) and on clips, optionally with `n` known ticks (pre-inputs).
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example opp_eval2 -- --model m2.oppnet --arena DIR [--clips FILE...] [--human FILE...] [--known N] [--decode "press=-1"]
//! ```

use std::path::PathBuf;

use ddai_oppnet::blob::read_blob;
use ddai_oppnet::clipdata::ClipGame;
use ddai_oppnet::humandata::HumanGame;
use ddai_oppnet::v2::corpus::{Corpus, CorpusCfg};
use ddai_oppnet::v2::data::GameRec;
use ddai_oppnet::v2::predictor::{Bundle, apply_decode_overrides};
use ddai_oppnet::v2::train::evaluate;

fn main() -> Result<(), String> {
    let (mut model, mut arena, mut clips, mut humans) = (PathBuf::new(), Vec::new(), Vec::new(), Vec::<PathBuf>::new());
    let (mut known, mut decode, mut swing) = (0usize, String::new(), false);
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" => model = PathBuf::from(v()?),
            "--arena" => arena.push(PathBuf::from(v()?)),
            "--clips" => clips.push(PathBuf::from(v()?)),
            "--human" => humans.push(PathBuf::from(v()?)),
            "--known" => known = v()?.parse().map_err(|e| format!("--known: {e}"))?,
            "--decode" => decode = v()?,
            "--swing-label" => swing = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let b = Bundle::load(&model)?;
    let mut dec = b.decode;
    apply_decode_overrides(&mut dec, &decode)?;
    let mut known_p = [0.0f32; 5];
    known_p[known.min(4)] = 1.0;
    let cfg = CorpusCfg {
        known_p,
        swing_label: swing,
        ..CorpusCfg::default()
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for p in &arena {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .map_err(|e| e.to_string())?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "opp2"))
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p.clone());
        }
    }
    for f in files {
        let games: Vec<GameRec> = read_blob(&f)?;
        let c = Corpus::new(games, vec![], cfg.clone());
        let s = c.samples();
        println!(
            "\n## {} (known ticks {known}), decode {dec:?}\n\n{}",
            f.display(),
            evaluate(&b.net, &c, &s, &dec).table()
        );
    }
    for f in clips {
        let g: Vec<ClipGame> = read_blob(&f)?;
        let c = Corpus::new(vec![], g, cfg.clone());
        let s = c.samples();
        println!(
            "\n## clips {}\n\n{}",
            f.display(),
            evaluate(&b.net, &c, &s, &dec).table()
        );
    }
    for f in humans {
        let g: Vec<HumanGame> = read_blob(&f)?;
        let c = Corpus::new(vec![], vec![], cfg.clone()).with_humans(g);
        let s = c.samples();
        println!(
            "\n## humans {} (known ticks {known})\n\n{}",
            f.display(),
            evaluate(&b.net, &c, &s, &dec).table()
        );
    }
    Ok(())
}
