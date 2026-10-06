//! Task 3.11: an opt-in per-input trace for timing experiments (`BotConfig::input_trace`, or `DDAI_INPUT_TRACE=<file>` for a process with one bot), the bot's half of the "input-to-apply
//! tick error" measurement. One JSON object per line:
//!
//! - `{"k":"s","tick":T,"in":[direction,target_x,target_y,jump,fire,hook,flags,weapon,next,prev]}` — a `NETMSG_INPUT` for
//!   `IntendedTick = T` left the socket;
//! - `{"k":"t","tick":T,"left":MS}` — the server's `NETMSG_INPUTTIMING` for it (`left < 0`: it arrived late and the server moved it);
//! - `{"k":"d","tick":T,"first":F,"exp":E,"brain":0|1,"wire_us":..,"handed_us":..,"pickup_us":..}` — the decision whose first input
//!   was `T`: the first slot after its snapshot (`F`), the tick it was aimed at (`E`), and the times of the stages;
//!
//! `tools/e2e/live_timing_analyze.py` joins these with the server's teehistorian by tick (the line order is the order of the events).
//! Nothing here runs unless the variable is set; it never influences a decision.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use ddai_net::generated::objects::PlayerInput;

/// The trace file.
pub struct InputTrace {
    out: BufWriter<File>,
}

impl InputTrace {
    /// Creates (truncates) the file at `path`.
    pub fn create(path: &Path) -> std::io::Result<InputTrace> {
        Ok(InputTrace {
            out: BufWriter::with_capacity(1 << 16, File::create(path)?),
        })
    }

    /// The trace named by `DDAI_INPUT_TRACE`, if set and creatable.
    pub fn from_env() -> Option<InputTrace> {
        let path = std::env::var_os("DDAI_INPUT_TRACE")?;
        match InputTrace::create(Path::new(&path)) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!(error = %e, "DDAI_INPUT_TRACE: the file could not be created; no trace");
                None
            }
        }
    }

    pub fn sent(&mut self, tick: i32, i: &PlayerInput) {
        let _ = writeln!(
            self.out,
            r#"{{"k":"s","tick":{tick},"in":[{},{},{},{},{},{},{},{},{},{}]}}"#,
            i.direction,
            i.target_x,
            i.target_y,
            i.jump,
            i.fire,
            i.hook,
            i.player_flags,
            i.wanted_weapon,
            i.next_weapon,
            i.prev_weapon
        );
    }

    pub fn timing(&mut self, tick: i32, time_left: i32) {
        let _ = writeln!(self.out, r#"{{"k":"t","tick":{tick},"left":{time_left}}}"#);
    }

    pub fn decision(
        &mut self,
        tick: i32,
        tag: Option<ddai_client::InputTag>,
        wire: Duration,
        handed_after: Duration,
        pickup: Duration,
    ) {
        let (first, exp) = tag.map_or((0, 0), |t| (t.first_slot, t.expected_tick));
        let _ = writeln!(
            self.out,
            r#"{{"k":"d","tick":{tick},"first":{first},"exp":{exp},"brain":{},"wire_us":{},"handed_us":{},"pickup_us":{}}}"#,
            u8::from(tag.is_some_and(|t| t.brain)),
            wire.as_micros(),
            handed_after.as_micros(),
            pickup.as_micros()
        );
    }

    pub fn flush(&mut self) {
        let _ = self.out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three kinds of line are valid JSON with the documented fields (the analyser reads exactly these).
    #[test]
    fn the_trace_lines_are_json_with_the_documented_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        let mut t = InputTrace::create(&path).unwrap();
        let input = PlayerInput {
            direction: -1,
            target_x: 10,
            target_y: -20,
            jump: 1,
            fire: 3,
            hook: 1,
            player_flags: 0,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        };
        t.sent(1001, &input);
        t.timing(1001, -3);
        t.decision(
            1001,
            Some(ddai_client::InputTag {
                first_slot: 1000,
                expected_tick: 1001,
                brain: true,
            }),
            Duration::from_micros(12_345),
            Duration::from_micros(4_500),
            Duration::from_micros(80),
        );
        t.flush();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["k"], "s");
        assert_eq!(lines[0]["tick"], 1001);
        assert_eq!(lines[0]["in"], serde_json::json!([-1, 10, -20, 1, 3, 1, 0, 0, 0, 0]));
        assert_eq!(lines[1], serde_json::json!({"k": "t", "tick": 1001, "left": -3}));
        assert_eq!(
            lines[2],
            serde_json::json!({"k": "d", "tick": 1001, "first": 1000, "exp": 1001, "brain": 1,
                               "wire_us": 12345, "handed_us": 4500, "pickup_us": 80})
        );
    }
}
