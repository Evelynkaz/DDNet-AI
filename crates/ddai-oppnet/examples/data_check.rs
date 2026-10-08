use ddai_oppnet::blob::read_blob;
use ddai_oppnet::clipdata::ClipGame;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let games: Vec<ClipGame> = read_blob(std::path::Path::new(&p)).unwrap();
    // runs per source; the last run of each source is the end of the clip (the round's last frames)
    let mut by_src: std::collections::BTreeMap<String, Vec<&ClipGame>> = Default::default();
    for g in &games { by_src.entry(g.source.split('p').next().unwrap().to_string()).or_default().push(g); }
    for last_n in [75usize, 150, 300, 100000] {
        let mut tr = [[0u64; 3]; 3];
        let (mut dsame, mut dn) = (0u64, 0u64);
        for (_, runs) in &by_src {
            let g = runs.last().unwrap();
            let start = g.ticks.len().saturating_sub(last_n);
            for w in g.ticks[start..].windows(2) {
                if !(w[0].duel && w[1].duel) || w[1].tick != w[0].tick + 2 { continue; }
                let (a, b) = (&w[0].frames[1], &w[1].frames[1]);
                if a.freeze_left > 0 || b.freeze_left > 0 { continue; }
                let cls = |s: i8| if s == 0 { 0 } else if s == 5 { 2 } else { 1 };
                tr[cls(a.hook_state)][cls(b.hook_state)] += 1;
                dn += 1;
                dsame += u64::from(a.direction == b.direction);
            }
        }
        let idle: u64 = tr[0].iter().sum(); let grab: u64 = tr[2].iter().sum();
        println!("last {last_n} frames of each source: pairs {dn}, dir same {:.3}, P(onset|idle) {:.3} (n {idle}), P(release|grabbed) {:.3} (n {grab})", dsame as f64 / dn as f64, 1.0 - tr[0][0] as f64 / idle as f64, 1.0 - tr[2][2] as f64 / grab as f64);
    }
}
