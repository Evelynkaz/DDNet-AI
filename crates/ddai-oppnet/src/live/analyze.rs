//! The analysis of the live log (`oppnet-live.jsonl`): accuracy of the model against hold, by tick of the window (`k`) and by our lag (`w`).
//!
//! The metric is `opp_clips`' (task 3.15): the share of snapshots where the predicted **direction** equals the one the snapshot shows
//! (hold: the direction the window's first snapshot showed), plus the hook state (out or not) and the aim error in radians, for the model and for
//! hold. `k` is the tick of the window (the input applied in the step from `T + k`; the snapshot at `T + k + 1` shows it). The log has
//! no raw opponent inputs (nobody can see them); "actual" is what the snapshots show.
//!
//! `ddnet-ai oppnet-live report <file>...` prints it; the library part is here so the numbers are tested.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::BufRead;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Line {
    ev: Option<String>,
    w: Option<u8>,
    u: Option<u8>,
    s: Option<Vec<Vec<i64>>>,
    t: Option<i64>,
    to: Option<String>,
    start: Option<u64>,
    server: Option<String>,
    /// A string in `open` lines (the model's sha256), a number in `guard` lines.
    model: Option<serde_json::Value>,
}

/// Sums over the samples of one cell (a `(w, k)` pair, or a total).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cell {
    pub n: u64,
    pub dir_hold: u64,
    pub dir_model: u64,
    pub hook_hold: u64,
    pub hook_model: u64,
    /// Direction and hook both right.
    pub both_hold: u64,
    pub both_model: u64,
    /// Sum of aim errors, radians.
    pub aim_hold: f64,
    pub aim_model: f64,
    /// Jump events seen / flagged by the model when seen / flagged by the model with no event.
    pub jump_events: u64,
    pub jump_hit: u64,
    pub jump_false: u64,
}

impl Cell {
    fn add(&mut self, o: &Cell) {
        self.n += o.n;
        self.dir_hold += o.dir_hold;
        self.dir_model += o.dir_model;
        self.hook_hold += o.hook_hold;
        self.hook_model += o.hook_model;
        self.both_hold += o.both_hold;
        self.both_model += o.both_model;
        self.aim_hold += o.aim_hold;
        self.aim_model += o.aim_model;
        self.jump_events += o.jump_events;
        self.jump_hit += o.jump_hit;
        self.jump_false += o.jump_false;
    }

    fn pct(a: u64, n: u64) -> f64 {
        100.0 * a as f64 / n.max(1) as f64
    }
}

/// Absolute difference of two angles given in milli-radians, in radians.
fn gap_mrad(a: i64, b: i64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut d = ((a - b) as f64 / 1000.0).rem_euclid(two_pi);
    if d > std::f64::consts::PI {
        d = two_pi - d;
    }
    d
}

/// One run of the bot in the log (one `ev: open` line and the lines after it).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Run {
    pub model: String,
    pub start: u64,
    pub server: String,
    pub windows: u64,
    pub cell: Cell,
}

#[derive(Debug, Default)]
pub struct Report {
    /// The runs in file order; lines before any `open` line belong to an unnamed first run.
    pub runs: Vec<Run>,
    cells: BTreeMap<(u8, u8), Cell>,
    /// Windows by `used` (0 shadow, 1 the model drove).
    windows: [u64; 2],
    /// Guard transitions: `(tick, to)`.
    pub transitions: Vec<(i64, String)>,
    /// Files whose header said which model produced them.
    pub models: Vec<String>,
    pub bad_lines: u64,
}

impl Report {
    pub fn new() -> Report {
        Report::default()
    }

    /// Adds one line of a log file.
    pub fn add_line(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let Ok(line) = serde_json::from_str::<Line>(text) else {
            self.bad_lines += 1;
            return;
        };
        match line.ev.as_deref() {
            Some("open") => {
                self.runs.push(Run {
                    model: line
                        .model
                        .as_ref()
                        .and_then(|m| m.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    start: line.start.unwrap_or(0),
                    server: line.server.clone().unwrap_or_default(),
                    ..Run::default()
                });
                if let Some(m) = serde_json::from_str::<serde_json::Value>(text)
                    .ok()
                    .and_then(|v| v.get("model").and_then(|m| m.as_str().map(str::to_string)))
                    && !self.models.contains(&m)
                {
                    self.models.push(m);
                }
            }
            Some("guard") => {
                self.transitions
                    .push((line.t.unwrap_or(0), line.to.clone().unwrap_or_default()));
            }
            Some(_) => {}
            None => {
                let (Some(w), Some(s)) = (line.w, line.s) else {
                    self.bad_lines += 1;
                    return;
                };
                self.windows[usize::from(line.u.unwrap_or(0).min(1))] += 1;
                if self.runs.is_empty() {
                    self.runs.push(Run::default());
                }
                let run = self.runs.last_mut().expect("a run exists");
                run.windows += 1;
                for f in s {
                    if f.len() != 12 {
                        self.bad_lines += 1;
                        continue;
                    }
                    let k = u8::try_from(f[0]).unwrap_or(0);
                    let (a_dir, m_dir, h_dir) = (f[1], f[2], f[3]);
                    let (a_hook, m_hook, h_hook) = (f[4], f[5], f[6]);
                    let (a_aim, m_aim, h_aim) = (f[7], f[8], f[9]);
                    let (a_jump, m_jump) = (f[10] != 0, f[11] != 0);
                    let mut one = Cell {
                        n: 1,
                        dir_hold: u64::from(a_dir == h_dir),
                        dir_model: u64::from(a_dir == m_dir),
                        hook_hold: u64::from(a_hook == h_hook),
                        hook_model: u64::from(a_hook == m_hook),
                        aim_hold: gap_mrad(a_aim, h_aim),
                        aim_model: gap_mrad(a_aim, m_aim),
                        ..Cell::default()
                    };
                    one.both_hold = u64::from(a_dir == h_dir && a_hook == h_hook);
                    one.both_model = u64::from(a_dir == m_dir && a_hook == m_hook);
                    self.runs.last_mut().expect("a run exists").cell.add(&one);
                    let c = self.cells.entry((w, k)).or_default();
                    c.n += 1;
                    c.dir_hold += u64::from(a_dir == h_dir);
                    c.dir_model += u64::from(a_dir == m_dir);
                    c.hook_hold += u64::from(a_hook == h_hook);
                    c.hook_model += u64::from(a_hook == m_hook);
                    c.both_hold += u64::from(a_dir == h_dir && a_hook == h_hook);
                    c.both_model += u64::from(a_dir == m_dir && a_hook == m_hook);
                    c.aim_hold += gap_mrad(a_aim, h_aim);
                    c.aim_model += gap_mrad(a_aim, m_aim);
                    c.jump_events += u64::from(a_jump);
                    c.jump_hit += u64::from(a_jump && m_jump);
                    c.jump_false += u64::from(!a_jump && m_jump);
                }
            }
        }
    }

    /// Adds a whole file.
    pub fn add_reader(&mut self, r: impl BufRead) -> std::io::Result<()> {
        for line in r.lines() {
            self.add_line(&line?);
        }
        Ok(())
    }

    /// Samples in total.
    pub fn samples(&self) -> u64 {
        self.cells.values().map(|c| c.n).sum()
    }

    /// The pooled cell of `k` over all lags (`None`: any k) or of lag `w` over all k (`None`: any lag).
    pub fn pooled(&self, w: Option<u8>, k: Option<u8>) -> Cell {
        let mut t = Cell::default();
        for (&(cw, ck), c) in &self.cells {
            if w.is_none_or(|w| w == cw) && k.is_none_or(|k| k == ck) {
                t.add(c);
            }
        }
        t
    }

    fn row(label: &str, c: &Cell) -> String {
        let n = c.n.max(1) as f64;
        format!(
            "| {label} | {} | {:.1} -> {:.1} | {:.1} -> {:.1} | {:.1} -> {:.1} | {:.3} -> {:.3} | {} / {} / {} |\n",
            c.n,
            Cell::pct(c.dir_hold, c.n),
            Cell::pct(c.dir_model, c.n),
            Cell::pct(c.hook_hold, c.n),
            Cell::pct(c.hook_model, c.n),
            Cell::pct(c.both_hold, c.n),
            Cell::pct(c.both_model, c.n),
            c.aim_hold / n,
            c.aim_model / n,
            c.jump_events,
            c.jump_hit,
            c.jump_false,
        )
    }

    /// The report as Markdown tables: by `k`, by our lag `w`, and (with `detail`) by both.
    pub fn render(&self, detail: bool) -> String {
        const HEAD: &str = "| n | direction % (hold -> model) | hook % | direction+hook % | aim error rad | jump events / hit / false flags |\n|---|---|---|---|---|---|---|\n";
        let mut s = String::new();
        let _ = writeln!(
            s,
            "## live window-model log: {} samples in {} windows ({} model-driven, {} shadow){}\n",
            self.samples(),
            self.windows[0] + self.windows[1],
            self.windows[1],
            self.windows[0],
            if self.models.is_empty() {
                String::new()
            } else {
                format!(", model sha256 {}", self.models.join(", "))
            }
        );
        let _ = write!(s, "### by tick of the window (k)\n\n| k {HEAD}");
        let ks: Vec<u8> = {
            let mut v: Vec<u8> = self.cells.keys().map(|&(_, k)| k).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let ws: Vec<u8> = {
            let mut v: Vec<u8> = self.cells.keys().map(|&(w, _)| w).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        for &k in &ks {
            s.push_str(&Self::row(&k.to_string(), &self.pooled(None, Some(k))));
        }
        s.push_str(&Self::row("all", &self.pooled(None, None)));
        let _ = write!(s, "\n### by our lag (w, ticks of the window)\n\n| w {HEAD}");
        for &w in &ws {
            s.push_str(&Self::row(&w.to_string(), &self.pooled(Some(w), None)));
        }
        if detail {
            let _ = write!(s, "\n### by lag and tick\n\n| w/k {HEAD}");
            for (&(w, k), c) in &self.cells {
                s.push_str(&Self::row(&format!("{w}/{k}"), c));
            }
        }
        if self.runs.len() > 1 || self.runs.iter().any(|r| r.start != 0) {
            let _ = write!(
                s,
                "\n### runs (one per `ev: open` line)\n\n| # | start (unix) | server | model | windows | n | direction % (hold -> model) | hook % | direction+hook % | aim error rad | jump events / hit / false flags |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
            );
            for (i, r) in self.runs.iter().enumerate() {
                let model: String = r.model.chars().take(12).collect();
                let head = format!("{} | {} | {} | {} | {}", i + 1, r.start, r.server, model, r.windows);
                s.push_str(&Self::row(&head, &r.cell));
            }
        }
        if self.transitions.is_empty() {
            let _ = writeln!(s, "\nguard: no change of state in the files");
        } else {
            let _ = writeln!(
                s,
                "\nguard: {} changes of state (tick -> driver):",
                self.transitions.len()
            );
            for (t, to) in &self.transitions {
                let _ = writeln!(s, "  - {t} -> {to}");
            }
        }
        if self.bad_lines > 0 {
            let _ = writeln!(s, "\n{} lines could not be read", self.bad_lines);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One window, lag 3: k = 1 (everything right for the model, hold wrong on the direction), k = 3 (hold right, model wrong on the hook).
    const W1: &str = r#"{"v":1,"t":100,"w":3,"u":1,"o":"c2-aabbccdd","s":[[1,1,1,0,1,1,1,1000,1000,2000,0,0],[3,-1,0,-1,0,1,0,3000,3100,3000,1,1]]}"#;

    #[test]
    fn a_window_line_counts_by_k_and_lag() {
        let mut r = Report::new();
        r.add_line(r#"{"ev":"open","v":1,"model":"f244c47d"}"#);
        r.add_line(W1);
        r.add_line(W1);
        r.add_line("");
        assert_eq!(r.samples(), 4);
        assert_eq!(r.models, ["f244c47d"]);
        let k1 = r.pooled(None, Some(1));
        assert_eq!((k1.n, k1.dir_hold, k1.dir_model), (2, 0, 2));
        assert_eq!((k1.hook_hold, k1.hook_model), (2, 2));
        assert!((k1.aim_hold / 2.0 - 1.0).abs() < 1e-9, "hold aim 1.0 rad off");
        assert!(k1.aim_model.abs() < 1e-9);
        let k3 = r.pooled(Some(3), Some(3));
        assert_eq!((k3.dir_hold, k3.dir_model), (2, 0));
        assert_eq!((k3.hook_hold, k3.hook_model), (2, 0));
        assert_eq!((k3.both_hold, k3.both_model), (2, 0));
        assert_eq!((k3.jump_events, k3.jump_hit, k3.jump_false), (2, 2, 0));
        assert_eq!(r.pooled(Some(4), None).n, 0, "no window of lag 4");
        assert_eq!(r.pooled(None, None).n, 4);
    }

    #[test]
    fn every_open_line_starts_a_run_and_the_report_labels_them() {
        let mut r = Report::new();
        r.add_line(r#"{"ev":"open","v":1,"start":100,"server":"127.0.0.1:8303","model":"aaaaaaaaaaaaaaaa"}"#);
        r.add_line(W1);
        r.add_line(r#"{"ev":"open","v":1,"start":200,"server":"10.0.0.1:8303","model":"bbbbbbbbbbbbbbbb"}"#);
        r.add_line(W1);
        r.add_line(W1);
        assert_eq!(r.runs.len(), 2);
        assert_eq!((r.runs[0].windows, r.runs[1].windows), (1, 2));
        assert_eq!((r.runs[0].cell.n, r.runs[1].cell.n), (2, 4));
        assert_eq!(r.runs[1].server, "10.0.0.1:8303");
        let text = r.render(false);
        assert!(
            text.contains("### runs") && text.contains("10.0.0.1:8303") && text.contains("bbbbbbbbbbbb"),
            "{text}"
        );
    }

    #[test]
    fn angle_gaps_wrap_around() {
        assert!((gap_mrad(100, 6283 - 100) - 0.2).abs() < 0.01, "across zero");
        assert!((gap_mrad(0, 3141) - std::f64::consts::PI).abs() < 0.01);
        assert!(gap_mrad(500, 500).abs() < 1e-12);
    }

    #[test]
    fn guard_events_and_garbage_are_kept_apart_and_the_report_renders() {
        let mut r = Report::new();
        r.add_line(r#"{"ev":"guard","t":4242,"to":"hold","model":1.2,"hold":1.0,"samples":400,"windows":100}"#);
        r.add_line("not json");
        r.add_line(r#"{"v":1,"t":1,"w":2,"u":0,"o":"x","s":[[1,0,0]]}"#);
        r.add_line(W1);
        assert_eq!(r.transitions, [(4242, "hold".to_string())]);
        assert_eq!(r.bad_lines, 2, "garbage and a sample with the wrong width");
        let text = r.render(true);
        for want in [
            "by tick of the window",
            "by our lag",
            "by lag and tick",
            "3/3",
            "4242 -> hold",
            "2 lines could not",
        ] {
            assert!(text.contains(want), "{want}: {text}");
        }
        assert!(text.contains("1 model-driven, 1 shadow"), "{text}");
    }
}
