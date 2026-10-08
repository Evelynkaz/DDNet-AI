//! Task 3.21 (E-036): trains and evaluates a v2 opponent-input model (`ddai_oppnet::v2`) on arena games and live clips.
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example opp_train2 -- --train-arena DIR --train-clips s0.clipgames \
//!     --val-arena DIR --val-clips s0.clipgames --out m2.oppnet [--init m.oppnet] [--epochs 14] [--threads 3]
//! ```
//!
//! Model selection and threshold tuning use the validation sets only; the tables printed at the end are on them, split by source (arena / clips).

use std::path::PathBuf;

use ddai_oppnet::blob::read_blob;
use ddai_oppnet::clipdata::ClipGame;
use ddai_oppnet::net::Mlp;
use ddai_oppnet::v2::corpus::{Corpus, CorpusCfg, SampleRef};
use ddai_oppnet::v2::data::GameRec;
use ddai_oppnet::v2::feature::HORIZON;
use ddai_oppnet::v2::predictor::{Bundle, Decode};
use ddai_oppnet::v2::train::{TrainCfg, auc, best_f1, evaluate, train};

struct Args {
    train_arena: Vec<PathBuf>,
    train_clips: Vec<PathBuf>,
    val_arena: Vec<PathBuf>,
    val_clips: Vec<PathBuf>,
    /// Clips held out of everything: evaluated at the end only (a final test).
    test_clips: Vec<PathBuf>,
    out: PathBuf,
    init: Option<PathBuf>,
    cfg: TrainCfg,
    known_p: [f32; 5],
    clip_lag: usize,
    hist_keep: usize,
    notes: String,
    /// The fire threshold set by hand (otherwise tuned on the validation set).
    press_thr: Option<f32>,
    /// Decoding overrides applied after tuning (`press=100,jump=100`).
    decode: String,
    /// Hold out whole clip files of the training clips: every `fold_mod`-th game with index % fold_mod == fold goes to the validation set instead.
    fold: Option<(usize, usize)>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        train_arena: vec![],
        train_clips: vec![],
        val_arena: vec![],
        val_clips: vec![],
        test_clips: vec![],
        out: PathBuf::new(),
        init: None,
        cfg: TrainCfg::default(),
        known_p: [1.0, 0.0, 0.0, 0.0, 0.0],
        clip_lag: 2,
        hist_keep: ddai_oppnet::v2::feature::K_HIST,
        notes: String::new(),
        press_thr: None,
        decode: String::new(),
        fold: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        let f = |s: String| s.parse::<f32>().map_err(|e| format!("{k}: {e}"));
        let u = |s: String| s.parse::<usize>().map_err(|e| format!("{k}: {e}"));
        match k.as_str() {
            "--train-arena" => a.train_arena.push(PathBuf::from(v()?)),
            "--train-clips" => a.train_clips.push(PathBuf::from(v()?)),
            "--val-arena" => a.val_arena.push(PathBuf::from(v()?)),
            "--val-clips" => a.val_clips.push(PathBuf::from(v()?)),
            "--test-clips" => a.test_clips.push(PathBuf::from(v()?)),
            "--out" => a.out = PathBuf::from(v()?),
            "--init" => a.init = Some(PathBuf::from(v()?)),
            "--epochs" => a.cfg.epochs = u(v()?)?,
            "--h1" => a.cfg.h1 = u(v()?)?,
            "--h2" => a.cfg.h2 = u(v()?)?,
            "--batch" => a.cfg.batch = u(v()?)?,
            "--seed" => a.cfg.seed = u(v()?)? as u64,
            "--threads" => a.cfg.threads = u(v()?)?,
            "--lr" => a.cfg.lr = f(v()?)?,
            "--lr-min" => a.cfg.lr_min = f(v()?)?,
            "--wd" => a.cfg.weight_decay = f(v()?)?,
            "--clip-weight" => a.cfg.loss.clip_weight = f(v()?)?,
            "--press-pos" => a.cfg.loss.press_pos = f(v()?)?,
            "--w-press" => a.cfg.loss.w_press = f(v()?)?,
            "--w-aim" => a.cfg.loss.w_aim = f(v()?)?,
            "--k-weight" => {
                let w: Vec<f32> = v()?
                    .split(',')
                    .map(|x| x.parse::<f32>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?;
                if w.len() != HORIZON {
                    return Err(format!("--k-weight needs {HORIZON} numbers"));
                }
                a.cfg.loss.k_weight.copy_from_slice(&w);
            }
            "--known-p" => {
                let p: Vec<f32> = v()?
                    .split(',')
                    .map(|x| x.parse::<f32>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?;
                if p.len() != 5 || (p.iter().sum::<f32>() - 1.0).abs() > 1e-4 {
                    return Err("--known-p needs 5 probabilities summing to 1".into());
                }
                a.known_p.copy_from_slice(&p);
            }
            "--clip-lag" => a.clip_lag = u(v()?)?,
            "--hist-keep" => a.hist_keep = u(v()?)?,
            "--notes" => a.notes = v()?,
            "--press-thr" => a.press_thr = Some(f(v()?)?),
            "--decode" => a.decode = v()?,
            "--fold" => {
                let s = v()?;
                let (i, n) = s.split_once('/').ok_or("--fold i/n")?;
                a.fold = Some((u(i.into())?, u(n.into())?));
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.out.as_os_str().is_empty() || (a.train_arena.is_empty() && a.train_clips.is_empty()) {
        return Err("usage: opp_train2 --train-arena DIR|FILE... --train-clips FILE... --val-arena ... --val-clips ... --out FILE [options]".into());
    }
    Ok(a)
}

fn load_arena(paths: &[PathBuf]) -> Result<Vec<GameRec>, String> {
    let mut out = Vec::new();
    for p in paths {
        let files: Vec<PathBuf> = if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .map_err(|e| format!("{}: {e}", p.display()))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "opp"))
                .collect();
            v.sort();
            v
        } else {
            vec![p.clone()]
        };
        for f in files {
            out.extend(read_blob::<Vec<GameRec>>(&f)?);
        }
    }
    Ok(out)
}

fn load_clips(paths: &[PathBuf]) -> Result<Vec<ClipGame>, String> {
    let mut out = Vec::new();
    for p in paths {
        out.extend(read_blob::<Vec<ClipGame>>(p)?);
    }
    Ok(out)
}

/// The decoding thresholds that do best on the validation samples at the ticks the live window uses (`k = 0, 1`).
fn tune(m: &Mlp, c: &Corpus, samples: &[SampleRef], press_thr: Option<f32>) -> Decode {
    let base = Decode::default();
    let m0 = evaluate(m, c, samples, &base);
    let mut press: Vec<(f32, bool)> = Vec::new();
    for k in 0..2 {
        press.extend_from_slice(&m0.press.by_k[k]);
    }
    let (f1, thr) = best_f1(&press);
    println!(
        "tuning: fire over k = 0, 1: {} samples, {} swings, AUC {:.3}, best F1 {:.3} at logit {:.2}",
        press.len(),
        press.iter().filter(|x| x.1).count(),
        auc(&press),
        f1,
        thr
    );
    println!("fire precision / recall over k = 0, 1 by logit threshold:");
    for t in [-2.0f32, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0] {
        let pos = press.iter().filter(|x| x.0 > t).count();
        let tp = press.iter().filter(|x| x.0 > t && x.1).count();
        let ev = press.iter().filter(|x| x.1).count();
        println!(
            "  logit > {t:>4}: predicted {pos:>6}, precision {:.3}, recall {:.3}",
            tp as f64 / pos.max(1) as f64,
            tp as f64 / ev.max(1) as f64
        );
    }
    let mut d = Decode {
        press: press_thr.unwrap_or(thr),
        ..base
    };
    // Direction margin, hook and jump thresholds: accuracy at k = 0, 1 pooled.
    let acc = |dec: &Decode, head: u8| -> f64 {
        let m = evaluate(m, c, samples, dec);
        let cells = match head {
            0 => &m.dir,
            1 => &m.hook,
            _ => &m.jump,
        };
        let (n, ok) = cells[..2].iter().fold((0u64, 0u64), |a, c| (a.0 + c.n, a.1 + c.model));
        ok as f64 / n.max(1) as f64
    };
    let mut best = (acc(&d, 0), d.dir_margin);
    for margin in [0.5f32, 1.0, 1.5, 2.0, 3.0] {
        let a = acc(
            &Decode {
                dir_margin: margin,
                ..d
            },
            0,
        );
        if a > best.0 + 1e-9 {
            best = (a, margin);
        }
    }
    d.dir_margin = best.1;
    let mut best = (acc(&d, 1), d.hook_margin);
    for t in [0.5f32, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let a = acc(&Decode { hook_margin: t, ..d }, 1);
        if a > best.0 + 1e-9 {
            best = (a, t);
        }
    }
    d.hook_margin = best.1;
    let mut best = (acc(&d, 2), d.jump);
    for t in [-1.0f32, -0.5, 0.5, 1.0, 1.5, 2.0] {
        let a = acc(&Decode { jump: t, ..d }, 2);
        if a > best.0 + 1e-9 {
            best = (a, t);
        }
    }
    d.jump = best.1;
    println!("tuning: decode {d:?}");
    d
}

fn main() -> Result<(), String> {
    let a = parse()?;
    let t0 = std::time::Instant::now();
    let tr_arena = load_arena(&a.train_arena)?;
    let mut tr_clips = load_clips(&a.train_clips)?;
    let va_arena = load_arena(&a.val_arena)?;
    let mut va_clips = load_clips(&a.val_clips)?;
    let te_clips = load_clips(&a.test_clips)?;
    if let Some((fold, n)) = a.fold {
        // Whole runs of a clip: the games of one source never split between the sets.
        let mut sources: Vec<String> = tr_clips
            .iter()
            .map(|g| g.source.split('p').next().unwrap_or("").to_string())
            .collect();
        sources.sort();
        sources.dedup();
        let held: Vec<&String> = sources
            .iter()
            .enumerate()
            .filter(|(i, _)| i % n == fold)
            .map(|(_, s)| s)
            .collect();
        let (mut keep, mut out) = (Vec::new(), Vec::new());
        for g in tr_clips.drain(..) {
            let src = g.source.split('p').next().unwrap_or("").to_string();
            if held.contains(&&src) {
                out.push(g);
            } else {
                keep.push(g);
            }
        }
        println!("fold {fold}/{n}: held out clips {held:?}");
        tr_clips = keep;
        va_clips.extend(out);
    }
    println!(
        "train: {} arena games, {} clip runs; val: {} arena games, {} clip runs; test: {} clip runs",
        tr_arena.len(),
        tr_clips.len(),
        va_arena.len(),
        va_clips.len(),
        te_clips.len()
    );
    let cfg_train = CorpusCfg {
        clip_lag: a.clip_lag,
        known_p: a.known_p,
        hist_keep: a.hist_keep,
    };
    let cfg_val = CorpusCfg {
        clip_lag: a.clip_lag,
        known_p: [1.0, 0.0, 0.0, 0.0, 0.0],
        hist_keep: a.hist_keep,
    };
    let c_train = Corpus::new(tr_arena, tr_clips, cfg_train);
    let c_val = Corpus::new(va_arena, va_clips, cfg_val.clone());
    let init = match &a.init {
        Some(p) => Some(Bundle::load(p)?.net),
        None => None,
    };
    if init.is_some() {
        println!(
            "fine-tuning from {}",
            a.init.as_ref().map_or_else(String::new, |p| p.display().to_string())
        );
    }
    let n_tr = c_train.samples().len();
    let n_va = c_val.samples().len();
    println!(
        "samples: train {n_tr}, val {n_va}; loaded in {:.0}s",
        t0.elapsed().as_secs_f64()
    );
    let (net, val) = train(&c_train, &c_val, &a.cfg, init.as_ref(), &mut |l| println!("{l}"))?;
    let samples = c_val.samples();
    // The thresholds are tuned for the real opponent when there are clips to tune on, else on the arena.
    let clip_samples: Vec<SampleRef> = samples.iter().copied().filter(|r| r.src == 1).collect();
    let tune_on = if clip_samples.is_empty() { &samples } else { &clip_samples };
    println!("tuning the decoding on {} {} samples", tune_on.len(), if clip_samples.is_empty() { "arena" } else { "clip" });
    let mut decode = tune(&net, &c_val, tune_on, a.press_thr);
    ddai_oppnet::v2::predictor::apply_decode_overrides(&mut decode, &a.decode)?;
    println!("decoding used: {decode:?}");
    let bundle = Bundle::new(
        net.clone(),
        decode,
        a.cfg.seed,
        a.cfg.epochs as u32,
        val,
        a.notes.clone(),
    );
    bundle.save(&a.out)?;
    println!(
        "saved {} ({} parameters), val loss {val:.4}",
        a.out.display(),
        net.params.len()
    );
    for (name, src) in [("arena", 0u8), ("clips", 1u8)] {
        let s: Vec<SampleRef> = samples.iter().copied().filter(|r| r.src == src).collect();
        if s.is_empty() {
            continue;
        }
        println!(
            "\n## validation, {name}\n\n{}",
            evaluate(&net, &c_val, &s, &decode).table()
        );
    }
    if !te_clips.is_empty() {
        let c_test = Corpus::new(vec![], te_clips, cfg_val);
        let s = c_test.samples();
        println!(
            "\n## test clips ({} samples)\n\n{}",
            s.len(),
            evaluate(&net, &c_test, &s, &decode).table()
        );
    }
    Ok(())
}
