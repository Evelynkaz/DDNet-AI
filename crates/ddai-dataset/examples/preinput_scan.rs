//! Task 3.24, step 1: does a demo carry `Sv_PreInput` messages, and what do they look like?
//! One line per demo: sha256 prefix (never the file name), size, ticks, message counts and the lead
//! `intended_tick - demo tick`. Usage: `preinput_scan <demo>...`.
use std::collections::BTreeMap;

use ddai_net::generated::messages::ExGameMsg;
use ddai_net::message::Msg;
use sha2::{Digest, Sha256};

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("read error: {e}");
                continue;
            }
        };
        let sha = Sha256::digest(&bytes);
        let id: String = sha.iter().take(6).map(|b| format!("{b:02x}")).collect();
        let demo = match ddai_demo::Demo::parse(&bytes) {
            Ok(d) => d,
            Err(e) => {
                println!("{id} parse error {e}");
                continue;
            }
        };
        let mut ticks = 0u64;
        let (mut first, mut last) = (i32::MAX, i32::MIN);
        let mut n = 0u64;
        let mut owners: BTreeMap<i32, u64> = BTreeMap::new();
        let mut lead: BTreeMap<i32, u64> = BTreeMap::new();
        let (mut fire, mut hook, mut jump, mut dir) = (0u64, 0u64, 0u64, 0u64);
        let mut invalid = 0u64;
        let mut spacing = [0u64; 6];
        let mut prev_snap: Option<std::sync::Arc<ddai_net::snapshot::Snapshot>> = None;
        let mut prev_fresh_tick = i32::MIN;
        let mut err = None;
        for t in demo.ticks() {
            let t = match t {
                Ok(t) => t,
                Err(e) => {
                    err = Some(e.to_string());
                    break;
                }
            };
            ticks += 1;
            if let Some(sn) = &t.snapshot
                && prev_snap.as_ref().is_none_or(|p| !std::sync::Arc::ptr_eq(p, sn))
            {
                if prev_fresh_tick != i32::MIN {
                    spacing[((t.tick - prev_fresh_tick).clamp(1, 5)) as usize] += 1;
                }
                prev_fresh_tick = t.tick;
                prev_snap = Some(std::sync::Arc::clone(sn));
            }
            first = first.min(t.tick);
            last = last.max(t.tick);
            for m in &t.messages {
                match m {
                    Msg::ExGame(ExGameMsg::SvPreInput(p)) => {
                        n += 1;
                        *owners.entry(p.owner).or_default() += 1;
                        *lead.entry((p.intended_tick - t.tick).clamp(-20, 20)).or_default() += 1;
                        fire += u64::from(p.fire != 0);
                        hook += u64::from(p.hook != 0);
                        jump += u64::from(p.jump != 0);
                        dir += u64::from(p.direction != 0);
                    }
                    Msg::Invalid => invalid += 1,
                    _ => {}
                }
            }
        }
        let minutes = f64::from(last.saturating_sub(first).max(0)) / 50.0 / 60.0;
        println!(
            "{id} v{} {:.1}MB ticks={ticks} span={minutes:.1}min preinputs={n} owners={} invalid_msgs={invalid} err={err:?}",
            demo.header.version,
            bytes.len() as f64 / 1e6,
            owners.len()
        );
        println!("   fresh-snapshot spacing 1/2/3/4/5+ ticks: {:?}", &spacing[1..]);
        if n > 0 {
            println!(
                "   hook!=0 {:.1}% jump!=0 {:.1}% dir!=0 {:.1}% fire!=0 {:.1}%  lead(intended-demo tick) {:?}",
                100.0 * hook as f64 / n as f64,
                100.0 * jump as f64 / n as f64,
                100.0 * dir as f64 / n as f64,
                100.0 * fire as f64 / n as f64,
                lead
            );
        }
    }
}
