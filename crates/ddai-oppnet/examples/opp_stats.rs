//! Task 3.21 (E-036): how the arena's opponent behaves in v2 arena files (`opp_record --v2`), the numbers `docs/research/opponent-predictor-v2.md` §2 compares with the
//! clips' (see `clip_stats`): direction unchanged after 2 ticks, the chance that an idle hook starts / a grabbing hook is let go, swings per tick (attack_age == 1).
//!
//! ```text
//! cargo run --release -p ddai-oppnet --example opp_stats -- ~/aiddnet/data/runs/E-036/data/train/*.opp2
//! ```

use ddai_oppnet::blob::read_blob;
use ddai_oppnet::v2::data::GameRec;

fn main() {
    for p in std::env::args().skip(1) {
        let games: Vec<GameRec> = read_blob(std::path::Path::new(&p)).expect("a v2 .opp file");
        let mut tr = [[0u64; 3]; 3];
        let (mut same, mut n, mut swings, mut ticks) = (0u64, 0u64, 0u64, 0u64);
        for g in &games {
            for i in (0..g.ticks.len().saturating_sub(2)).step_by(2) {
                let (a, b) = (&g.ticks[i].frames[1], &g.ticks[i + 2].frames[1]);
                if !a.alive || !b.alive || a.freeze_left > 0 || b.freeze_left > 0 {
                    continue;
                }
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
            // Weapon uses (swings): the frame a step led to shows the weapon used one tick ago (`attack_age == 1`); fire-counter presses during the reload are not swings.
            // The first 30 ticks are skipped: a tee that never swung has `attack_tick` 0, so the first tick of every game reads as a use (3.17 note).
            for (j, t) in g.ticks.iter().enumerate() {
                if g.tick0 + j as i32 >= 30 && t.frames[1].alive && t.frames[1].freeze_left == 0 {
                    ticks += 1;
                    swings += u64::from(t.frames[1].attack_age == 1);
                }
            }
        }
        let (idle, grab): (u64, u64) = (tr[0].iter().sum(), tr[2].iter().sum());
        println!(
            "{}: direction unchanged {:.3}, P(hook starts | idle) {:.3} (n {idle}), P(hook let go | grabbed) {:.3} (n {grab}), swings per tick (attack_age == 1) {:.4}",
            p.rsplit('/').next().unwrap_or(&p),
            same as f64 / n.max(1) as f64,
            1.0 - tr[0][0] as f64 / idle.max(1) as f64,
            1.0 - tr[2][2] as f64 / grab.max(1) as f64,
            swings as f64 / ticks.max(1) as f64
        );
    }
}
