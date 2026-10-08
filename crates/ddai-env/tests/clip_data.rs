//! Task 3.21 (E-036): the live-clip data set (`live_data`'s conversion through `LiveWorld`) against what the clip itself recorded.
//!
//! On a real duel clip of the lead's diagnosis folder (read-only; the test is skipped when it is not on this machine): the runs are consecutive 2-tick frames, every
//! frame is a duel frame, the weapon-use ticks the labels are built from (`attack_tick`) agree one for one with the weapon-use events the bot's own clip recorder saw
//! (`ClipEvent::HammerFire` of the opponent), and a labelled sample puts a swing in the window tick the tick of its `attack_tick` says.

#[path = "../examples/live_data/convert.rs"]
mod convert;

use ddai_clip::format::{Clip, ClipEvent};
use ddai_oppnet::clipdata::labels_at;
use ddai_oppnet::live::RegimeGate;

fn clip() -> Option<(Clip, std::sync::Arc<ddai_physics::map::MapData>)> {
    let home = std::path::PathBuf::from(std::env::var_os("HOME")?);
    let clip = Clip::read(&home.join("aiddnet/data/scratch/duel-diag/self-freeze-11626416-s186.clip")).ok()?;
    let map = convert::load_map(&home.join("aiddnet/data/maps/cache"), &clip).ok()?;
    Some((clip, map))
}

#[test]
fn a_real_duel_clip_becomes_consecutive_labelled_runs_that_agree_with_its_recorded_events() {
    let Some((clip, map)) = clip() else {
        eprintln!("skipped: the duel clip is not on this machine");
        return;
    };
    let own = clip.header.own_id;
    let games = convert::convert(&clip, map, &RegimeGate::default(), "t", 0);
    assert!(!games.is_empty());
    let frames: usize = games.iter().map(|g| g.ticks.len()).sum();
    let duel: usize = games.iter().flat_map(|g| g.ticks.iter()).filter(|t| t.duel).count();
    assert!(
        frames + 2 >= clip.frames.len(),
        "a duel clip keeps (nearly) all its frames: {frames} of {}",
        clip.frames.len()
    );
    assert!(duel > frames * 9 / 10, "{duel} duel frames of {frames}");
    for g in &games {
        assert!(
            g.ticks.windows(2).all(|w| w[1].tick == w[0].tick + 2),
            "frames are 2 ticks apart"
        );
        assert!(
            g.ticks.iter().all(|t| t.tick % 2 == 0),
            "the snapshots of this clip are on even ticks"
        );
        assert!(
            g.ticks.iter().all(|t| t.sent[1].is_some()),
            "our own input at the frame's tick is always known"
        );
        assert!(!g.source.contains(' '), "no names: {}", g.source);
    }
    // The opponent's weapon uses: the clip recorder's events against the attack-tick changes the labels use.
    let opp = clip
        .frames
        .iter()
        .flat_map(|f| f.tees.iter())
        .map(|t| t.id)
        .find(|&id| id != own)
        .expect("an opponent");
    let events: usize = clip
        .frames
        .iter()
        .flat_map(|f| f.events.iter())
        .filter(|e| matches!(e, ClipEvent::HammerFire { from, .. } if *from == opp))
        .count();
    let mut changes = 0usize;
    let mut labelled = 0usize;
    for g in &games {
        for (i, w) in g.ticks.windows(2).enumerate() {
            if w[1].opp_attack_tick != w[0].opp_attack_tick {
                changes += 1;
                // The sample at frame `i` labels the use at its window tick (`attack_tick - T`), when the later frame is free to show it.
                let l = labels_at(g, i, 4);
                if l.v_press & 0b11 == 0b11 {
                    let k = w[1].opp_attack_tick - w[0].tick;
                    assert!(
                        (0..2).contains(&k),
                        "a use shown by the frame at T + 2 happened at T or T + 1, not at {k}"
                    );
                    assert_eq!(l.press & 0b11, 1 << k, "the swing is labelled at window tick {k}");
                    labelled += 1;
                }
            }
        }
    }
    eprintln!("opponent weapon uses: {events} recorded events, {changes} attack-tick changes, {labelled} labelled");
    assert!(events > 3 && changes > 3, "the clip has swings: {events} / {changes}");
    assert!(
        changes.abs_diff(events) <= 2,
        "the labels' swings ({changes}) are the swings the recorder saw ({events})"
    );
    assert!(
        labelled * 10 >= changes * 8,
        "most swings fall in labelled samples: {labelled} of {changes}"
    );
}

/// 3.21 review F5: two ring clips of one life that overlap hold the same snapshots twice; the overlap is kept once. The same tick numbers with other content (a
/// restarted server) are not an overlap.
#[test]
fn overlapping_clips_keep_each_frame_once() {
    let Some((clip, _)) = clip() else {
        eprintln!("skipped: the duel clip is not on this machine");
        return;
    };
    let n = clip.frames.len();
    let mut early = clip.clone();
    early.frames.truncate(n * 2 / 3);
    let mut late = clip.clone();
    late.frames.drain(..n / 3);
    // Handed over in the wrong order on purpose: the earlier clip is the one that keeps its frames.
    let mut v = vec![
        (std::path::PathBuf::from("late"), late, 0u8),
        (std::path::PathBuf::from("early"), early, 0u8),
    ];
    convert::drop_overlap(&mut v);
    let mut ticks: Vec<i32> = v.iter().flat_map(|(_, c, _)| c.frames.iter().map(|f| f.tick)).collect();
    let total = ticks.len();
    ticks.sort_unstable();
    ticks.dedup();
    assert_eq!(
        (total, ticks.len()),
        (n, n),
        "every frame of the whole clip exactly once"
    );
    // Same ticks, another life: nothing is cut.
    let mut moved = clip.clone();
    for f in &mut moved.frames {
        for t in &mut f.tees {
            t.ch.x += 64;
        }
    }
    let mut v = vec![
        (std::path::PathBuf::from("a"), clip.clone(), 0u8),
        (std::path::PathBuf::from("b"), moved, 0u8),
    ];
    convert::drop_overlap(&mut v);
    // Only frames with no tee at all look the same after the shift.
    let empty = clip.frames.iter().filter(|f| f.tees.is_empty()).count();
    assert_eq!(
        v.iter().map(|(_, c, _)| c.frames.len()).collect::<Vec<_>>(),
        vec![n, n - empty]
    );
}
