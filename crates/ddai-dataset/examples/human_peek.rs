//! Task 3.24 debugging aid: prints the frames of one demo in a tick window next to the players' real
//! inputs (anonymous labels only). `human_peek <demo> <from_tick> <n_frames> [skip_frames]`.
use ddai_dataset::config::Config;
use ddai_dataset::humaninput::true_table;
use ddai_dataset::ingest::{FrameSource, LabelTracker};
use ddai_recorder::format::Frame;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&a[0]).unwrap();
    let from: i32 = a[1].parse().unwrap();
    let n: usize = a[2].parse().unwrap();
    let skip: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let demo = ddai_demo::Demo::parse(&bytes).unwrap();
    let table = true_table(&demo);
    let _ = Config::default();
    let mut tracker = LabelTracker::new();
    let mut shown = 0;
    let mut seen = 0;
    for (frame, _) in FrameSource::new(&demo) {
        let Frame::Snapshot { tick, characters, .. } = &frame else {
            continue;
        };
        let row = tracker.push(&frame);
        if *tick < from {
            continue;
        }
        seen += 1;
        if seen <= skip {
            continue;
        }
        print!("t={tick}");
        for (c, (_, label)) in characters.iter().zip(&row) {
            let ev = table.track(*label).and_then(|t| t.at(*tick));
            let ch = &c.character;
            print!(
                " | L{label} id{} ({},{}) v({},{}) hk{}/{} jm{} w{} fz{} in[{}]",
                c.id,
                ch.x / 32,
                ch.y / 32,
                ch.vel_x / 256,
                ch.vel_y / 256,
                ch.hook_state,
                ch.hooked_player,
                ch.jumped,
                ch.weapon,
                c.ddnet.map_or(-1, |d| d.freeze_end - tick),
                ev.map_or("?".to_string(), |e| format!(
                    "d{} j{} h{} f{}",
                    e.direction,
                    u8::from(e.jump),
                    u8::from(e.hook),
                    e.fire
                )),
            );
        }
        println!();
        shown += 1;
        if shown >= n {
            break;
        }
    }
}
