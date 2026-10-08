//! Task 3.21 (E-036): how the real opponent behaves in the clips -- the numbers `docs/research/opponent-predictor-v2.md` §2 compares with the arena's.
//!
//! For every `*.clipgames` file (see `live_data`): over the pairs of consecutive duel frames (2 ticks apart, nobody frozen) the share of unchanged directions, the chance
//! that an idle hook starts and that a grabbing hook is let go within 2 ticks, for all frames and for the last 75 / 150 / 300 frames of each clip; then the parity of the
//! tick of the opponent's weapon use (the snapshots are all on even ticks, so an even use is window tick `k = 0`, an odd one `k = 1`).
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example clip_stats -- ~/aiddnet/data/runs/E-036/live/s0.clipgames ~/aiddnet/data/runs/E-036/live/s1.clipgames
//! ```

use ddai_oppnet::blob::read_blob;
use ddai_oppnet::clipdata::ClipGame;

fn main() {
    for p in std::env::args().skip(1) {
        let games: Vec<ClipGame> = read_blob(std::path::Path::new(&p)).expect("a .clipgames file");
        // The last run of each source is the end of the clip (the round's last frames).
        let mut by_src: std::collections::BTreeMap<String, Vec<&ClipGame>> = Default::default();
        for g in &games {
            by_src
                .entry(g.source.split('p').next().unwrap_or("").to_string())
                .or_default()
                .push(g);
        }
        println!("{p}");
        for last_n in [75usize, 150, 300, usize::MAX / 2] {
            let mut tr = [[0u64; 3]; 3];
            let (mut same, mut n) = (0u64, 0u64);
            for runs in by_src.values() {
                // The last run of each clip for the end-of-round slices, every run for the whole.
                let used: &[&ClipGame] = if last_n > 100_000 {
                    runs
                } else {
                    &runs[runs.len().saturating_sub(1)..]
                };
                for g in used {
                    let start = g.ticks.len().saturating_sub(last_n);
                    for w in g.ticks[start..].windows(2) {
                        if !(w[0].duel && w[1].duel) || w[1].tick != w[0].tick + 2 {
                            continue;
                        }
                        let (a, b) = (&w[0].frames[1], &w[1].frames[1]);
                        if a.freeze_left > 0 || b.freeze_left > 0 {
                            continue;
                        }
                        // Hook class: 0 idle, 2 grabbed, 1 anything else.
                        let cls = |s: i8| {
                            if s == 0 {
                                0
                            } else if s == 5 {
                                2
                            } else {
                                1
                            }
                        };
                        tr[cls(a.hook_state)][cls(b.hook_state)] += 1;
                        n += 1;
                        same += u64::from(a.direction == b.direction);
                    }
                }
            }
            let (idle, grab): (u64, u64) = (tr[0].iter().sum(), tr[2].iter().sum());
            let label = if last_n > 100_000 {
                "every run of every clip".to_string()
            } else {
                format!("last {last_n} frames")
            };
            println!(
                "  {label}: pairs {n}, direction unchanged {:.3}, P(hook starts | idle) {:.3} (n {idle}), P(hook let go | grabbed) {:.3} (n {grab})",
                same as f64 / n.max(1) as f64,
                1.0 - tr[0][0] as f64 / idle.max(1) as f64,
                1.0 - tr[2][2] as f64 / grab.max(1) as f64
            );
        }
        let mut sw = [0u64; 2];
        for g in &games {
            for w in g.ticks.windows(2) {
                if w[0].duel && w[1].duel && w[1].tick == w[0].tick + 2 && w[1].opp_attack_tick != w[0].opp_attack_tick
                {
                    sw[(w[1].opp_attack_tick & 1) as usize] += 1;
                }
            }
        }
        println!("  weapon use on even / odd ticks (window tick k = 0 / 1): {sw:?}");
    }
}
