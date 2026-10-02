//! `ddnet-ai fly watch` (task 7.4): the offline source of the web's «Муха» tab, end to end through the real binary and a real
//! Unix socket. A client plays the web unit's part (HELLO, a subscription, then `FLYMETA` and `FLY` messages); the frames must
//! decode with the layout the stream announced, arrive decimated and in order, and the process must finish its game and exit
//! cleanly. Needs a trained bundle (E-005's `final.bundle` or `DDAI_FLY_BUNDLE`), the S graph and the CLB map on disk;
//! without them the test says so and passes (the data never lives in git).

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn data(sub: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join("aiddnet/data")
        .join(sub)
}

fn read_message(s: &mut UnixStream) -> (u8, Vec<u8>) {
    let mut len = [0u8; 4];
    s.read_exact(&mut len).expect("a message");
    let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
    s.read_exact(&mut body).expect("its body");
    (body[0], body[1..].to_vec())
}

#[test]
fn watch_serves_the_trained_flys_frames_to_a_subscriber_and_finishes_its_game() {
    watch_and_read(false);
}

#[test]
fn watching_the_fly_as_the_hybrids_proposer_adds_how_often_its_proposal_is_played() {
    watch_and_read(true);
}

fn watch_and_read(hybrid: bool) {
    let bundle = std::env::var_os("DDAI_FLY_BUNDLE")
        .map_or_else(|| data("runs/E-005/e005-fly/checkpoints/final.bundle"), PathBuf::from);
    let flyg = data("connectome/compiled/fly-S-v1.flyg");
    let map_dir = data("maps");
    if !bundle.is_file() || !flyg.is_file() || !map_dir.join("copy-love-box").is_dir() {
        eprintln!(
            "skipping: needs {} , {} and the CLB map (data is never in git)",
            bundle.display(),
            flyg.display()
        );
        return;
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("watch.sock");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ddnet-ai"));
    cmd.current_dir(&repo_root).args([
        "fly",
        "watch",
        "--arena",
        "clb-left",
        "--games",
        "1",
        "--speed",
        "6",
        "--pause-ms",
        "0",
        "--seed",
        "1",
    ]);
    if hybrid {
        cmd.arg("--hybrid");
    }
    let child = cmd
        .arg("--bundle")
        .arg(&bundle)
        .arg("--bridge")
        .arg(&sock)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn ddnet-ai");

    let started = Instant::now();
    let mut stream = loop {
        match UnixStream::connect(&sock) {
            Ok(s) => break s,
            Err(_) if started.elapsed() < Duration::from_secs(30) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => panic!("the watch socket never appeared: {e}"),
        }
    };
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let (kind, hello) = read_message(&mut stream);
    assert_eq!((kind, hello.as_slice()), (1, &b"DDBL\x01"[..]));
    // Nothing but HELLO until we subscribe: the stream is off with nobody watching.
    stream.write_all(&[2, 0, 0, 0, 1, 1]).unwrap();
    let (kind, meta) = read_message(&mut stream);
    assert_eq!(kind, 6, "the layout comes first");
    let meta: serde_json::Value = serde_json::from_slice(&meta).expect("JSON layout");
    assert_eq!(meta["v"], 1);
    assert_eq!(meta["role"], if hybrid { "proposer" } else { "fly" });
    let bundle_name = meta["bundle"]["name"].as_str().unwrap();
    assert!(!bundle_name.starts_with('/'), "a name, never a path: {bundle_name}");
    assert_eq!(meta["bundle"]["sha256"].as_str().unwrap().len(), 64);
    let rate_max = meta["rate_max"].as_f64().unwrap() as f32;
    let z_clip = meta["z_clip"].as_f64().unwrap() as f32;
    let groups = meta["groups"].as_array().unwrap().len();
    assert!(groups >= 20, "{groups} groups");
    assert_eq!(meta["dn"].as_array().unwrap().len(), 100);

    let mut seqs = Vec::new();
    let mut ticks = Vec::new();
    let mut decisions = Vec::new();
    while seqs.len() < 30 {
        let (kind, payload) = read_message(&mut stream);
        if kind != 7 {
            continue;
        }
        assert_eq!(payload.len() as u64, meta["frame_bytes"].as_u64().unwrap());
        let f = ddai_fly::viz::decode_frame(&payload, rate_max, z_clip).expect("a frame of the announced layout");
        assert_eq!(f.groups.len(), groups);
        let valid = f.flags & ddai_fly::viz::flag::CHOSEN_VALID != 0;
        assert_eq!(valid, hybrid, "only a proposer carries the chosen flag");
        if hybrid {
            assert!(
                f.chosen_total <= f.decisions_total,
                "{} of {}",
                f.chosen_total,
                f.decisions_total
            );
            decisions.push(f.decisions_total);
        }
        assert!(f.groups.iter().all(|g| g.is_finite() && *g >= 0.0));
        seqs.push(f.seq);
        ticks.push(f.tick);
    }
    // Decimated by two decisions, in order, one decision every second tick.
    assert!(seqs.windows(2).all(|w| w[1] - w[0] == 2), "{seqs:?}");
    assert!(ticks.windows(2).all(|w| w[1] > w[0]), "{ticks:?}");

    if hybrid {
        assert!(
            decisions.windows(2).all(|w| w[1] >= w[0]) && *decisions.last().unwrap() > 0,
            "{decisions:?}"
        );
    }
    drop(stream);
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{:?}", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("game 0:"), "{text}");
}

/// Task 7.4, acceptance criterion 4: a game's decisions are bit-identical whether or not a viewer pulls the fly's frames. The
/// same seed is played twice, once plain (`play_game`) and once with an observer that asks slot 0 for a frame after every tick
/// (a viewer attached, nothing dropped); the fly's decision hash (SHA-1 of every wire input it sent), the result and the
/// end tick must be equal. Once for the fly alone and once for the hybrid with the fly as proposer in its deterministic
/// (fixed-work) mode, where the viewer also gets the search's verdict on each proposal.
#[test]
fn a_viewer_pulling_frames_changes_no_decision_of_a_whole_game() {
    use ddai_env::arena::{Arena, load_arena_defs};
    use ddai_env::config::{PlayerSpec, Rules};
    use ddai_env::game::{Layout, play_game, play_game_observed};
    use ddai_env::models::{ModelBrains, player_from_arg};
    use ddai_env::sim::PlayerSetup;

    let bundle = std::env::var_os("DDAI_FLY_BUNDLE")
        .map_or_else(|| data("runs/E-005/e005-fly/checkpoints/final.bundle"), PathBuf::from);
    if !bundle.is_file() || !data("connectome/compiled/fly-S-v1.flyg").is_file() || !data("maps/copy-love-box").is_dir()
    {
        eprintln!("skipping: needs the trained bundle, the S graph and the CLB map");
        return;
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let defs = load_arena_defs(&repo_root.join("configs/arenas")).unwrap();
    let arena = Arena::build(&defs["clb-left"], &data("maps")).unwrap();
    let models = ModelBrains::new(None);
    let rules = Rules::default();

    let mut hybrid = player_from_arg(&format!("hybrid:fly:{}", bundle.display()));
    hybrid.mode = Some("fixed".to_string());
    let specs: [(&str, PlayerSpec); 2] = [
        ("fly", player_from_arg(&format!("fly:{}", bundle.display()))),
        ("hybrid", hybrid),
    ];
    for (name, focal) in specs {
        let players = |models: &ModelBrains| -> Vec<PlayerSetup> {
            [focal.clone(), PlayerSpec::simple("scripted")]
                .iter()
                .map(|s| PlayerSetup {
                    brain: models.make(s).unwrap(),
                    lag: 0,
                    label: name.to_string(),
                })
                .collect()
        };
        let layout = Layout::default();
        let plain = play_game(&arena, &rules, 7, layout, players(&models)).unwrap();
        let mut pulled = 0usize;
        let watched = play_game_observed(&arena, &rules, 7, layout, players(&models), &mut |p, tick| {
            if p[0].brain.viz_frame(tick as u32).is_some() {
                pulled += 1;
            }
            true
        })
        .unwrap();
        assert!(pulled > 20, "{name}: the viewer got {pulled} frames");
        assert_eq!(
            plain.players[0].hash, watched.players[0].hash,
            "{name}: the fly's decision stream"
        );
        assert_eq!(plain.players[1].hash, watched.players[1].hash, "{name}: the opponent's");
        assert_eq!(
            (plain.result, plain.end_tick),
            (watched.result, watched.end_tick),
            "{name}"
        );
    }
}
