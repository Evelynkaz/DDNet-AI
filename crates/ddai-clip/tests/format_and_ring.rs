//! The clip file format (magic, version, damage) and the ring recorder's no-allocation steady state.

use ddai_clip::format::*;
use ddai_clip::record::{ClipMeta, Recorder};

fn meta() -> ClipMeta {
    ClipMeta {
        map_name: "room".into(),
        map_sha256: [7; 32],
        own_id: 0,
        brain: "hybrid".into(),
        reason: ClipReason {
            kind: "manual".into(),
            severity: 0,
            tick: 0,
            note: "x".into(),
        },
        players: vec![PlayerTag {
            id: 1,
            tag: "p-1a2b".into(),
        }],
    }
}

fn tee(id: i32, x: i32) -> TeeRec {
    TeeRec {
        id,
        ch: CharRec {
            x,
            y: 500,
            hooked_player: -1,
            ..CharRec::default()
        },
        ..TeeRec::default()
    }
}

/// Fills one frame the way the bot does: 8 tees, projectiles, sent inputs, events, the bot record.
fn fill(r: &mut Recorder, tick: i32) {
    let mut b = r.begin(tick, true);
    for id in 0..8 {
        assert!(b.add_tee(tee(id, tick + id)));
    }
    for k in 0..30 {
        b.add_projectile(ProjRec {
            id: k,
            ..ProjRec::default()
        }); // more than fit: the overflow is counted, not stored
    }
    for k in 0..3 {
        b.add_sent(SentRec {
            tick: tick - k,
            input: InputRec::default(),
            timing_known: true,
        });
    }
    b.add_event(ClipEvent::HammerFire { from: 0, hits: 1 });
    b.set_bot(BotRec {
        target: 3,
        ..BotRec::default()
    });
    b.finish();
}

#[test]
fn recording_a_frame_allocates_nothing_in_the_steady_state() {
    let mut r = Recorder::new(RING_FRAMES);
    r.intern("walk to the exit"); // strings are interned up front
    for t in 0..(RING_FRAMES as i32 + 10) {
        fill(&mut r, t); // warm: the ring has wrapped
    }
    let info = allocation_counter::measure(|| {
        for t in 0..500 {
            fill(&mut r, 10_000 + t);
        }
    });
    assert_eq!(info.count_total, 0, "recording allocated: {info:?}");
    assert_eq!(r.len(), RING_FRAMES);
}

#[test]
fn the_ring_keeps_the_newest_frames_oldest_first() {
    let mut r = Recorder::new(10);
    for t in 0..25 {
        fill(&mut r, t);
    }
    let clip = r.to_clip(meta(), None).unwrap();
    let ticks: Vec<i32> = clip.frames.iter().map(|f| f.tick).collect();
    assert_eq!(ticks, (15..25).collect::<Vec<_>>());
    let last3 = r.to_clip(meta(), Some(3)).unwrap();
    assert_eq!(
        last3.frames.iter().map(|f| f.tick).collect::<Vec<_>>(),
        vec![22, 23, 24]
    );
    assert_eq!(clip.frames[0].tees.len(), 8);
    assert_eq!(clip.frames[0].projectiles.len(), MAX_PROJECTILES);
    assert_eq!(clip.frames[0].projectiles_dropped as usize, 30 - MAX_PROJECTILES);
    assert!(
        Recorder::new(4).to_clip(meta(), None).is_none(),
        "an empty ring has no clip"
    );
}

#[test]
fn a_full_ring_round_trips_and_stays_small() {
    let mut r = Recorder::new(RING_FRAMES);
    for t in 0..RING_FRAMES as i32 {
        fill(&mut r, 1000 + 2 * t);
    }
    let clip = r.to_clip(meta(), None).unwrap();
    assert_eq!(clip.frames.len(), RING_FRAMES);
    let bytes = clip.encode().unwrap();
    assert_eq!(&bytes[..7], b"DDCLIP\0");
    assert_eq!(bytes[7], FORMAT_VERSION);
    assert!(bytes.len() < 400_000, "a 30 s clip is {} bytes", bytes.len());
    assert_eq!(Clip::decode(&bytes).unwrap(), clip);
}

#[test]
fn damaged_and_foreign_files_are_errors_never_panics() {
    let mut r = Recorder::new(8);
    for t in 0..8 {
        fill(&mut r, t);
    }
    let bytes = r.to_clip(meta(), None).unwrap().encode().unwrap();
    assert!(matches!(Clip::decode(b""), Err(ClipError::BadMagic)));
    assert!(matches!(Clip::decode(b"not a clip at all"), Err(ClipError::BadMagic)));
    let mut newer = bytes.clone();
    newer[7] = FORMAT_VERSION + 1;
    assert!(matches!(Clip::decode(&newer), Err(ClipError::Version(v)) if v == FORMAT_VERSION + 1));
    // Every truncation and a flipped byte in the body: an error or (a lucky flip) a different clip, no panic.
    for cut in 8..bytes.len() {
        let _ = Clip::decode(&bytes[..cut]);
    }
    assert!(Clip::decode(&bytes[..bytes.len() - 5]).is_err());
    for at in (8..bytes.len()).step_by(7) {
        let mut b = bytes.clone();
        b[at] ^= 0xFF;
        let _ = Clip::decode(&b);
    }
    // Valid zstd of garbage postcard.
    let mut garbage = MAGIC.to_vec();
    garbage.extend(zstd_of(&[0xFF; 64]));
    assert!(matches!(Clip::decode(&garbage), Err(ClipError::Decode(_))));
}

fn zstd_of(raw: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut out = Vec::new();
    let mut z = zstd::Encoder::new(&mut out, 3).unwrap();
    z.write_all(raw).unwrap();
    z.finish().unwrap();
    out
}

#[test]
fn a_clip_carries_no_nicknames_only_tags() {
    let mut r = Recorder::new(4);
    fill(&mut r, 1);
    let mut m = meta();
    m.players = vec![PlayerTag {
        id: 1,
        tag: "p-1a2b".into(),
    }];
    let clip = r.to_clip(m, None).unwrap();
    let text = format!("{clip:?}");
    assert!(text.contains("p-1a2b"));
    // The only strings of the format are: map name, brain, reason, labels, tags. No name field exists.
    assert!(!format!("{:?}", clip.frames[0]).contains("name"));
}
