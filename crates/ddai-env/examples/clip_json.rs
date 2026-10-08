//! Post-mortem helper (duel 2026-10-07): a clip as JSON lines, for offline analysis in a script.
//!
//! Line 1 is the header (map, own id, reason, walk labels, players as tags, tuning, teams); every further line is one frame
//! (`ddai_clip::format::Frame`, serde field names). Read-only: the clip file is only read.
//!
//! ```text
//! cargo run --release -p ddai-env --example clip_json -- <file.clip> > out.jsonl
//! ```

use std::io::Write;

use ddai_clip::format::Clip;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: clip_json <file.clip>")?;
    let clip = Clip::read(std::path::Path::new(&path)).map_err(|e| format!("{path}: {e}"))?;
    let out = std::io::stdout();
    let mut out = out.lock();
    let header = serde_json::to_string(&clip.header).map_err(|e| e.to_string())?;
    writeln!(out, "{header}").map_err(|e| e.to_string())?;
    for f in &clip.frames {
        let line = serde_json::to_string(f).map_err(|e| e.to_string())?;
        writeln!(out, "{line}").map_err(|e| e.to_string())?;
    }
    Ok(())
}
