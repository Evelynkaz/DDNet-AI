//! Task 3.21 (E-036): builds the real-opponent data set from live clips (read-only) -- `ClipGame`s per session, through `LiveWorld` as the live bot does.
//!
//! Sessions (the unit of the train / held-out split; clips are never mixed between them):
//! * `0` -- the 06.10 duels on the JoniTee map (the competitor's bot);
//! * `1` -- the 07.10 test duel (the competitor's bot; ticks below 200 000, not a `manual` rehearsal clip);
//! * `2` -- everything else (public-server clips with humans; only the frames the regime gate admits are duel frames).
//!
//! Identical clips (same map, own id, first and last tick) found in several directories are taken once. No names: a game carries `s<session>c<n>p<part>`.
//!
//! ```text
//! cargo run --release -p ddai-env --example live_data -- build --out ~/aiddnet/data/runs/E-036/live DIR...
//! ```

#[path = "convert.rs"]
mod convert;

use std::path::PathBuf;

use ddai_oppnet::clipdata::ClipGame;
use ddai_oppnet::live::RegimeGate;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let cmd = args
        .next()
        .ok_or("usage: live_data build --out <dir> [--maps <dir>] <clip dir>...")?;
    if cmd != "build" {
        return Err(format!("unknown command {cmd}"));
    }
    let mut out = PathBuf::new();
    let mut maps = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("aiddnet/data/maps/cache");
    let mut dirs = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out needs a value")?),
            "--maps" => maps = PathBuf::from(args.next().ok_or("--maps needs a value")?),
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    if out.as_os_str().is_empty() || dirs.is_empty() {
        return Err("usage: live_data build --out <dir> [--maps <dir>] <clip dir>...".into());
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let gate = RegimeGate::default();
    let mut per_session: [Vec<ClipGame>; 3] = Default::default();
    let mut counter = [0usize; 3];
    for (_, clip, session) in convert::collect_clips(&dirs)? {
        let map = convert::load_map(&maps, &clip)?;
        counter[usize::from(session)] += 1;
        let source = format!("s{session}c{}", counter[usize::from(session)]);
        let games = convert::convert(&clip, map, &gate, &source, session);
        per_session[usize::from(session)].extend(games);
    }
    for (s, games) in per_session.iter().enumerate() {
        let frames: usize = games.iter().map(|g| g.ticks.len()).sum();
        let duel: usize = games.iter().flat_map(|g| g.ticks.iter()).filter(|t| t.duel).count();
        println!(
            "session {s}: {} clips, {} runs, {frames} frames, {duel} duel frames",
            counter[s],
            games.len()
        );
        ddai_oppnet::blob::write_blob(&out.join(format!("s{s}.clipgames")), games, 3)?;
    }
    Ok(())
}
