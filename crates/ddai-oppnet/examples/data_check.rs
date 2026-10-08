use ddai_oppnet::blob::read_blob;
use ddai_oppnet::clipdata::ClipGame;
fn main() {
    for p in std::env::args().skip(1) {
        let games: Vec<ClipGame> = read_blob(std::path::Path::new(&p)).unwrap();
        let (mut fr, mut sw) = ([0u64; 2], [0u64; 2]);
        let mut per_src: std::collections::BTreeMap<String, ([u64; 2], [u64; 2])> = Default::default();
        for g in &games {
            let e = per_src.entry(g.source.split('p').next().unwrap().to_string()).or_default();
            for w in g.ticks.windows(2) {
                if !(w[0].duel && w[1].duel) || w[1].tick != w[0].tick + 2 { continue; }
                fr[(w[0].tick & 1) as usize] += 1;
                e.0[(w[0].tick & 1) as usize] += 1;
                if w[1].opp_attack_tick != w[0].opp_attack_tick {
                    sw[(w[1].opp_attack_tick & 1) as usize] += 1;
                    e.1[(w[1].opp_attack_tick & 1) as usize] += 1;
                }
            }
        }
        println!("{p}: frame tick parity [even, odd] {fr:?}, swing attack_tick parity {sw:?}");
        for (s, (f, w)) in per_src { if w[0] + w[1] > 3 { println!("   {s}: frames {f:?} swings {w:?}"); } }
    }
}
