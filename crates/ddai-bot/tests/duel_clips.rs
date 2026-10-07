//! Task 4.12 (D-108): the automatic duel detection on the clips of the real F-DDrace `/1vs1` test duel (joniTee, 2026-10-06), and on a
//! synthetic clip that needs no data.
//!
//! The real clips are data and are never committed: this test reads them (read only) from `DDAI_DUEL_CLIPS` (default
//! `~/aiddnet/data/scratch/duel-diag`) and `DDAI_BOT_CLIPS` (default `~/aiddnet/data/bot/clips`), looks each file up by name and skips the
//! names that are not there. What they show: in the duel our DDRace team is 1 or 2 with exactly one other player in it; outside it we
//! are in team 0 with a dozen others.

use std::path::{Path, PathBuf};

use ddai_bot::duel::DuelWhy;
use ddai_bot::kill_replay::scan_duel;
use ddai_clip::format::{
    BotRec, CharRec, Clip, ClipEvent, ClipHeader, ClipReason, Frame, KillWhy, PlayerTag, TeamsChange, TeeRec,
};

/// Clips of the test duel: the detector says duel (the bot was in a two-player team the whole time or from a tick on).
const DUEL: &[&str] = &[
    "self-freeze-11626416-s186.clip",
    "self-freeze-11630670-s252.clip",
    "self-freeze-11642608-s248.clip",
    "self-freeze-11651834-s194.clip",
    "self-freeze-11702118-s276.clip",
    "self-freeze-11704686-s238.clip",
    "self-freeze-11717198-s198.clip",
    "self-freeze-11718944-s408.clip",
    "cross-fail-11719900-s0.clip",
    "goto-into-freeze-11639386-s208.clip",
];

/// Clips on the same joniTee map from before or between the duels (team 0, a dozen others): not a duel.
const NOT_DUEL: &[&str] = &[
    "goto-into-freeze-11620270-s182.clip",
    "self-freeze-11623550-s202.clip",
    "goto-into-freeze-11633676-s188.clip",
    "cross-fail-11637422-s0.clip",
    "self-freeze-11635726-s264.clip",
    "self-freeze-11694638-s252.clip",
];

fn dirs() -> Vec<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    [
        ("DDAI_DUEL_CLIPS", "aiddnet/data/scratch/duel-diag"),
        ("DDAI_BOT_CLIPS", "aiddnet/data/bot/clips"),
    ]
    .iter()
    .map(|(var, default)| std::env::var_os(var).map_or_else(|| home.join(default), PathBuf::from))
    .collect()
}

fn find(name: &str) -> Option<PathBuf> {
    dirs().into_iter().map(|d| d.join(name)).find(|p| p.is_file())
}

fn read(p: &Path) -> Clip {
    Clip::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn the_real_duel_clips_are_detected_by_the_team_and_the_rest_are_not() {
    let (mut seen_duel, mut seen_not) = (0, 0);
    for name in DUEL {
        let Some(p) = find(name) else { continue };
        let clip = read(&p);
        // Without F-DDrace evidence (a clip holds no chat) a two-player team is no duel; with it, it is.
        assert_eq!(
            scan_duel(&clip, false).first,
            None,
            "{name}: a team alone is not a duel"
        );
        let scan = scan_duel(&clip, true);
        let first = scan
            .first
            .unwrap_or_else(|| panic!("{name}: no duel detected; team: {:?}", scan.own_team));
        assert_eq!(first.1, DuelWhy::Team, "{name}: the structural signal, not the chat");
        assert!(
            scan.own_team
                .iter()
                .any(|&(_, team, others)| (1..64).contains(&team) && others == 1),
            "{name}: {:?}",
            scan.own_team
        );
        assert!(
            scan.frames_in_duel * 10 >= scan.frames * 4,
            "{name}: {} of {}",
            scan.frames_in_duel,
            scan.frames
        );
        seen_duel += 1;
    }
    for name in NOT_DUEL {
        let Some(p) = find(name) else { continue };
        let scan = scan_duel(&read(&p), true);
        assert_eq!(scan.first, None, "{name}: {:?}", scan.own_team);
        assert!(
            scan.own_team.iter().all(|&(_, team, others)| team == 0 && others > 2),
            "{name}: {:?}",
            scan.own_team
        );
        seen_not += 1;
    }
    eprintln!("duel clips read: {seen_duel} duel, {seen_not} not duel");
}

/// The `Cl_Kill`s the legacy bot sent inside the test duel are the ones the detector now stops.
#[test]
fn the_self_kills_sent_inside_the_real_duel_are_the_detectors() {
    let mut inside = 0;
    for (name, tick) in [
        ("self-freeze-11642608-s248.clip", 11642842),
        ("self-freeze-11702118-s276.clip", 11702370),
        ("cross-fail-11719900-s0.clip", 11719162),
        ("goto-into-freeze-11639386-s208.clip", 11639254),
    ] {
        let Some(p) = find(name) else { continue };
        let scan = scan_duel(&read(&p), true);
        assert!(
            scan.kills_in_duel.contains(&tick),
            "{name}: {tick} not in {:?}",
            scan.kills_in_duel
        );
        inside += 1;
    }
    // The clip of the transition holds one kill before the duel (a wayblock one in team 0): not stopped.
    if let Some(p) = find("goto-into-freeze-11639386-s208.clip") {
        let scan = scan_duel(&read(&p), true);
        assert!(scan.kills_outside.contains(&11638362), "{:?}", scan.kills_outside);
    }
    eprintln!("kills inside a duel checked on {inside} clips");
}

// ---- a clip made here ----------------------------------------------------------------------------------------------------------

fn tee_rec(id: i32) -> TeeRec {
    TeeRec {
        id,
        ch: CharRec {
            x: 1000 + id,
            y: 1000,
            ..CharRec::default()
        },
        ..TeeRec::default()
    }
}

fn frame(tick: i32, kill: bool) -> Frame {
    Frame {
        tick,
        own_alive: true,
        tees: vec![tee_rec(0), tee_rec(1)],
        events: if kill {
            vec![ClipEvent::KillSent {
                why: KillWhy::Unstick as u8,
            }]
        } else {
            Vec::new()
        },
        bot: BotRec::default(),
        ..Frame::default()
    }
}

/// Team 0 with the other player, then both in team 5 from tick 120, then back in team 0 from tick 300 (the duel is over).
#[test]
fn a_synthetic_clip_with_a_two_player_team_is_a_duel_from_the_tick_the_team_appears_to_the_release() {
    let teams_of = |from_tick: i32, own_and_other: i32| {
        let mut teams = vec![0; 2];
        teams[0] = own_and_other;
        teams[1] = own_and_other;
        TeamsChange {
            from_tick,
            received: 2,
            teams,
        }
    };
    let clip = Clip {
        header: ClipHeader {
            map_name: "synthetic".into(),
            map_sha256: [0; 32],
            own_id: 0,
            brain: "test".into(),
            reason: ClipReason {
                kind: "manual".into(),
                severity: 0,
                tick: 0,
                note: String::new(),
            },
            labels: vec![String::new()],
            players: vec![
                PlayerTag {
                    id: 0,
                    tag: "c0-aaaaaaaa".into(),
                },
                PlayerTag {
                    id: 1,
                    tag: "c1-bbbbbbbb".into(),
                },
            ],
            tuning: Vec::new(),
            teams: vec![teams_of(0, 0), teams_of(120, 5), teams_of(300, 0)],
            world_seed: 1,
        },
        // Frames every 2 ticks from 100 to 500; a kill sent at 200 (inside the duel) and one at 450 (after it).
        frames: (50..250).map(|i| frame(i * 2, i * 2 == 200 || i * 2 == 450)).collect(),
    };
    assert_eq!(
        scan_duel(&clip, false).first,
        None,
        "no F-DDrace evidence: a two-player team is not a duel"
    );
    let scan = scan_duel(&clip, true);
    assert_eq!(scan.first, Some((120, DuelWhy::Team)));
    assert_eq!(scan.kills_in_duel, vec![200]);
    assert_eq!(scan.kills_outside, vec![450]);
    assert_eq!(scan.own_team, vec![(100, 0, 1), (120, 5, 1), (300, 0, 1)]);
    // On from 120; the team was last seen at 298, and the detector holds 100 ticks more: frames 120..=396 (every 2 ticks).
    assert_eq!(scan.frames_in_duel, (398 - 120) / 2);
}
