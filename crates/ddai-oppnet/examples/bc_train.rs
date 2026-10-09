//! Task 3.24 (E-039), the imitation pilot, step 2: trains the behaviour-cloning head of `ddai_oppnet::bc` on `b<N>.bcsamples` files (from `human_bc`) and prints
//! how it does on the validation demos against the always-the-commonest-class baseline, per head.
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example bc_train -- --train b0.bcsamples [--train ...] --val b9.bcsamples [--val ...] --out prior.bc [--epochs 20] [--seed 1]
//! ```

use std::path::PathBuf;

use ddai_oppnet::bc::{BC_IN, BC_OUT, BcModel, BcSample, BcTrainCfg, distribution, train_bc};
use ddai_oppnet::blob::read_blob;
use ddai_oppnet::net::Scratch;

fn load(paths: &[PathBuf]) -> Result<Vec<BcSample>, String> {
    let mut out = Vec::new();
    for p in paths {
        out.extend(read_blob::<Vec<BcSample>>(p)?);
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    let (mut train, mut val, mut out) = (Vec::<PathBuf>::new(), Vec::<PathBuf>::new(), PathBuf::new());
    let mut cfg = BcTrainCfg::default();
    let mut notes = String::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--train" => train.push(PathBuf::from(v()?)),
            "--val" => val.push(PathBuf::from(v()?)),
            "--out" => out = PathBuf::from(v()?),
            "--epochs" => cfg.epochs = v()?.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--seed" => cfg.seed = v()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--h1" => cfg.h1 = v()?.parse().map_err(|e| format!("--h1: {e}"))?,
            "--h2" => cfg.h2 = v()?.parse().map_err(|e| format!("--h2: {e}"))?,
            "--lr" => cfg.lr = v()?.parse().map_err(|e| format!("--lr: {e}"))?,
            "--notes" => notes = v()?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if train.is_empty() || val.is_empty() || out.as_os_str().is_empty() {
        return Err("usage: bc_train --train FILE... --val FILE... --out FILE [--epochs N] [--seed N]".into());
    }
    let (tr, va) = (load(&train)?, load(&val)?);
    println!(
        "train {} samples, val {} samples (inputs {BC_IN}, outputs {BC_OUT})",
        tr.len(),
        va.len()
    );
    let (net, val_loss) = train_bc(&tr, &va, &cfg, &mut |l| println!("{l}"))?;
    BcModel::new(net.clone(), val_loss, notes).save(&out)?;
    println!(
        "saved {} ({} parameters), val loss {val_loss:.4}",
        out.display(),
        net.params.len()
    );
    // per-head accuracy on the validation demos against the commonest-class baseline
    let mut s = Scratch::new(&net);
    let n = va.len() as f64;
    let (mut dir_ok, mut jump_ok, mut hook_ok, mut fire_ok) = (0u64, 0u64, 0u64, 0u64);
    let mut dir_n = [0u64; 3];
    let (mut jump_n, mut hook_n, mut fire_n) = (0u64, 0u64, 0u64);
    let mut aim_err = (0.0f64, 0.0f64);
    for smp in &va {
        net.forward(&smp.x, &mut s);
        let d = distribution(&s.out, 0.0);
        let arg = (0..3)
            .max_by(|&a, &b| d.direction[a].total_cmp(&d.direction[b]))
            .unwrap_or(1);
        dir_ok += u64::from(arg == usize::from(smp.y.dir));
        dir_n[usize::from(smp.y.dir)] += 1;
        jump_ok += u64::from((d.jump >= 0.5) == smp.y.jump);
        hook_ok += u64::from((d.hook >= 0.5) == smp.y.hook);
        fire_ok += u64::from((d.fire >= 0.5) == smp.y.fire);
        jump_n += u64::from(smp.y.jump);
        hook_n += u64::from(smp.y.hook);
        fire_n += u64::from(smp.y.fire);
        aim_err.0 += (d.aim_angle - f64::from(smp.y.aim_rel)).abs().min(std::f64::consts::PI);
        aim_err.1 += f64::from(smp.y.aim_rel).abs();
    }
    let maj = |pos: u64| (pos.max(va.len() as u64 - pos)) as f64 / n * 100.0;
    println!("\n| head | model accuracy % | commonest class % | positives % |\n|---|---:|---:|---:|");
    println!(
        "| direction | {:.1} | {:.1} | - |",
        100.0 * dir_ok as f64 / n,
        100.0 * *dir_n.iter().max().unwrap_or(&0) as f64 / n
    );
    for (name, ok, pos) in [
        ("jump", jump_ok, jump_n),
        ("hook", hook_ok, hook_n),
        ("fire press", fire_ok, fire_n),
    ] {
        println!(
            "| {name} | {:.1} | {:.1} | {:.1} |",
            100.0 * ok as f64 / n,
            maj(pos),
            100.0 * pos as f64 / n
        );
    }
    println!(
        "\naim relative to the line to the victim: mean abs error {:.3} rad, 'aim straight at him' baseline {:.3} rad",
        aim_err.0 / n,
        aim_err.1 / n
    );
    Ok(())
}
