//! Task 3.24 debugging aid: how many player-frames of a demo have a real input in force, with and without the snapshot-trust pass, and why not.
//! `human_cover <demo>...` (prints ids by sha prefix only).
use ddai_dataset::humaninput::{true_table, true_table_raw};
use ddai_dataset::ingest::{FrameSource, LabelTracker};
use ddai_recorder::format::Frame;
use sha2::{Digest, Sha256};

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let id: String = Sha256::digest(&bytes)
            .iter()
            .take(6)
            .map(|b| format!("{b:02x}"))
            .collect();
        let demo = ddai_demo::Demo::parse(&bytes).unwrap();
        let raw = true_table_raw(&demo);
        let trusted = true_table(&demo);
        let (mut total, mut k_raw, mut k_tr, mut no_track, mut before_first) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut tracker = LabelTracker::new();
        for (frame, _) in FrameSource::new(&demo).min_spacing(2) {
            let Frame::Snapshot { tick, characters, .. } = &frame else {
                continue;
            };
            let row = tracker.push(&frame);
            for (_, &(_, label)) in characters.iter().zip(&row) {
                total += 1;
                match raw.track(label) {
                    None => no_track += 1,
                    Some(t) => {
                        before_first += u64::from(t.events.first().is_some_and(|e| e.intended > *tick));
                        k_raw += u64::from(t.at(*tick).is_some());
                        k_tr += u64::from(trusted.track(label).is_some_and(|t| t.at(*tick).is_some()));
                    }
                }
            }
        }
        let desyncs: usize = trusted.tracks.values().map(|t| t.desyncs.len()).sum();
        let restarts: usize = trusted.tracks.values().map(|t| t.restarts.len()).sum();
        println!(
            "{id}: player-frames {total}, no track {no_track}, before first event {before_first}, known raw {k_raw} ({:.0}%), known after trust {k_tr} ({:.0}%); desyncs {desyncs}, restarts {restarts}",
            100.0 * k_raw as f64 / total.max(1) as f64,
            100.0 * k_tr as f64 / total.max(1) as f64
        );
    }
}
