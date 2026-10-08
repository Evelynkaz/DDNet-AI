use ddai_oppnet::blob::read_blob;
use ddai_oppnet::v2::data::GameRec;
fn main() {
    for p in std::env::args().skip(1) {
        let games: Vec<GameRec> = read_blob(std::path::Path::new(&p)).unwrap();
        let mut tr = [[0u64; 3]; 3];
        let (mut dsame, mut dn) = (0u64, 0u64);
        let (mut swings, mut ticks) = (0u64, 0u64);
        for g in &games {
            for i in (0..g.ticks.len().saturating_sub(2)).step_by(2) {
                let (a, b) = (&g.ticks[i].frames[1], &g.ticks[i + 2].frames[1]);
                if !a.alive || !b.alive || a.freeze_left > 0 || b.freeze_left > 0 { continue; }
                let cls = |s: i8| if s == 0 { 0 } else if s == 5 { 2 } else { 1 };
                tr[cls(a.hook_state)][cls(b.hook_state)] += 1;
                dn += 1;
                dsame += u64::from(a.direction == b.direction);
                ticks += 2;
                swings += u64::from(g.ticks[i + 1].applied[1].fire != g.ticks[i].applied[1].fire && g.ticks[i + 1].applied[1].fire & 1 == 1) + u64::from(g.ticks[i + 2].applied[1].fire != g.ticks[i + 1].applied[1].fire && g.ticks[i + 2].applied[1].fire & 1 == 1);
            }
        }
        let idle: u64 = tr[0].iter().sum(); let grab: u64 = tr[2].iter().sum();
        println!("{}: dir same {:.3}, onset {:.3} (n {idle}), release {:.3} (n {grab}), swings/tick {:.4}", p.rsplit('/').next().unwrap(), dsame as f64 / dn as f64, 1.0 - tr[0][0] as f64 / idle as f64, 1.0 - tr[2][2] as f64 / grab as f64, swings as f64 / ticks as f64);
    }
}
