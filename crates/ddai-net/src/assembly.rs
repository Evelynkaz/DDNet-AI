// Ported from DDNet `src/engine/client/client.cpp` (`CClient::ProcessServerPacket`'s
// `NETMSG_SNAP`/`NETMSG_SNAPSINGLE`/`NETMSG_SNAPEMPTY` branch, `client.cpp:2110-2312`, pinned rev
// c9d208138f85755521f16a0096b6fe036c5c8698, "20.1"), which carries the original Teeworlds
// zlib-style notice:
//
//   (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information.
//   If you are missing that file, acquire a complete release at teeworlds.com.
//
// This is an *altered* source version: rewritten in safe Rust, same part/size limits and
// tick/ack bookkeeping, so it stays compatible with the assembly a real DDNet 0.6+DDNet server
// expects a client to perform. See docs/formats.md for the byte layout.
//
//! Multi-part `NETMSG_SNAP` assembly: turns a stream of decoded [`crate::sysmsg::SysMsg::Snap`] /
//! `SnapSingle` / `SnapEmpty` messages into complete [`crate::snapshot::Snapshot`]s — task 2.2b
//! acceptance criterion 3.
//!
//! [`SysMsg::SnapSmall`] is deliberately **not** fed into assembly at all (review round 1,
//! finding F7): the real DDNet 20.1 client's own dispatch (`client.cpp:2110`) is
//! `Msg == NETMSG_SNAP || Msg == NETMSG_SNAPSINGLE || Msg == NETMSG_SNAPEMPTY` — `SNAPSMALL` is
//! simply not in that condition, so a real client ignores it outright regardless of what a
//! server puts in it (`protocol.h:47`: "not used" — no real 20.1 server sends it either).
//! [`SnapAssembler::feed`] mirrors that by returning `None` unconditionally for it, same as any
//! other message this module does not recognise; [`crate::sysmsg`] still decodes its byte layout
//! (identical to `SnapSingle`'s), since the task's message list names it explicitly.
//!
//! Deliberately **not** ported (out of this task's scope, task 2.3's client session instead):
//! `AckGameTick`/`SendInput` — a real client resends its ack every tick via `NETMSG_INPUT`; this
//! module only *tracks* the ack value ([`SnapAssembler::ack_game_tick`]) and lets the caller decide
//! when/whether to act on a resync request ([`Event::Resync`]). Similarly, "which snapshot to
//! purge up to" here is simply this delta's own base tick (`snapshot.cpp:2247-2251`'s `PurgeTick`
//! also considers the client's *displayed* prev/current snapshot ticks, which this crate has no
//! concept of — a caller that needs those kept longer should hold its own reference, e.g. by
//! reading [`SnapAssembler::storage`] before advancing).
//!
//! **The caller must call [`SnapAssembler::reset`] every time it is about to send
//! `NETMSG_ENTERGAME`** (review round 1, finding F1) — see that method's docs for why: without
//! it, every snapshot after a map change is silently dropped forever.

use crate::delta::{self, StaticSizes};
use crate::snapshot::{Snapshot, SnapshotStorage};
use crate::sysmsg::SysMsg;

/// `MAX_SNAPSHOT_PACKSIZE` (`protocol.h:107`): max bytes of snapshot data per `NETMSG_SNAP` part.
pub const MAX_SNAPSHOT_PACKSIZE: usize = 900;

/// What [`SnapAssembler::feed`] observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// A complete snapshot was assembled, delta-unpacked, and (for a non-empty delta) CRC
    /// verified — now stored (`SnapAssembler::storage`) and returned.
    Snapshot { tick: i32, snap: Snapshot },
    /// The assembled delta's CRC did not match what the server claimed (`crc_errors` is the
    /// running count of *consecutive* such mismatches so far, including this one) — the snapshot
    /// is **not** stored or returned as [`Event::Snapshot`] (matches the C++ reference: a CRC
    /// mismatch discards the snapshot outright, matching `client.cpp:2221-2231`'s bare `return;`).
    CrcMismatch {
        tick: i32,
        wanted: u32,
        got: u32,
        crc_errors: u32,
    },
    /// The delta's base tick is no longer in [`SnapAssembler::storage`] (purged, or never
    /// received) — the server needs to resync (send a full snapshot); a real client would set
    /// `AckGameTick = -1` and send that in its next `NETMSG_INPUT`, which this module already did
    /// internally (see [`SnapAssembler::ack_game_tick`]) — the caller only needs to act on this to
    /// know a resync is *why* the ack changed, e.g. for logging/counting.
    Resync { tick: i32 },
    /// The delta bytes themselves were malformed (see [`crate::delta::DeltaError`]) — dropped,
    /// like DDNet's own `dbg_msg("client", "delta unpack failed...")` path.
    DeltaError { tick: i32, error: delta::DeltaError },
    /// `tick` is at or before the already-acked tick, or older than the tick currently being
    /// assembled — dropped without further processing, matching `client.cpp:2144-2145`'s guard
    /// (review round 1, finding F1: an earlier version returned a bare `None` here, identical to
    /// "still waiting on more parts of an in-progress tick", so nothing distinguished a normal,
    /// benign duplicate from *every* snapshot after a map change silently vanishing because the
    /// caller forgot to call [`SnapAssembler::reset`]). A large run of these right after a
    /// `NETMSG_ENTERGAME` (first entry or a later map change) almost always means exactly that —
    /// see [`SnapAssembler::reset`].
    Stale { tick: i32 },
}

/// Assembles multi-part `NETMSG_SNAP` messages into complete snapshots and keeps recent ones in a
/// [`SnapshotStorage`] so later deltas can find their base — the decode-side half of task 2.2b's
/// "complete snapshot handling".
#[derive(Debug, Clone)]
pub struct SnapAssembler {
    static_sizes: StaticSizes,
    storage: SnapshotStorage,
    ack_game_tick: i32,
    current_recv_tick: i32,
    parts_received: u64,
    incoming: Vec<u8>,
    incoming_size: usize,
    crc_errors: u32,
}

impl SnapAssembler {
    pub fn new(static_sizes: StaticSizes) -> Self {
        SnapAssembler {
            static_sizes,
            storage: SnapshotStorage::new(),
            ack_game_tick: -1,
            current_recv_tick: 0,
            parts_received: 0,
            incoming: vec![0u8; crate::snapshot::MAX_PARTS * MAX_SNAPSHOT_PACKSIZE],
            incoming_size: 0,
            crc_errors: 0,
        }
    }

    /// Resets all session-scoped state — mirrors DDNet's `CClient::OnEnterGame`
    /// (`client.cpp:472-508`) for the pieces this crate owns (input/prediction/display timing are
    /// task 2.3's): purges [`SnapAssembler::storage`], clears any in-progress multi-part
    /// reassembly, and resets the tick/ack/CRC-error bookkeeping to the same values a freshly
    /// [`SnapAssembler::new`]'d instance would have.
    ///
    /// **The caller must call this every time it is about to send `NETMSG_ENTERGAME`** — which
    /// happens both on the very first map entry and after *every later* `NETMSG_MAP_CHANGE`
    /// (`client.cpp:511-524`'s `EnterGame` calls `OnEnterGame` unconditionally, every time). The
    /// server also resets its own tick counter on a map change (`server.cpp:3505`), so every
    /// snapshot tick after a map change starts again from a small number — at or below whatever
    /// this assembler last saw on the *previous* map. Without this reset, every one of those
    /// ticks looks [`Event::Stale`] forever (confirmed against a real capture of a live map
    /// change, review round 1 finding F1: a single un-reset assembler recovered only 384 of 1,626
    /// real snapshots across two map changes; resetting on every `ENTERGAME` recovered all
    /// 1,626 — see `tests/real_traffic.rs`).
    pub fn reset(&mut self) {
        self.storage = SnapshotStorage::new();
        self.ack_game_tick = -1;
        self.current_recv_tick = 0;
        self.parts_received = 0;
        self.incoming_size = 0;
        self.crc_errors = 0;
    }

    /// The ack value a real client would report in its next `NETMSG_INPUT` — starts at `-1`
    /// ("no snapshot acked yet") and is reset to `-1` whenever a resync is needed
    /// ([`Event::Resync`]) or 10 consecutive CRC errors accumulate (`client.cpp:2226-2231`).
    pub fn ack_game_tick(&self) -> i32 {
        self.ack_game_tick
    }

    /// Snapshots successfully assembled so far, most recent last.
    pub fn storage(&self) -> &SnapshotStorage {
        &self.storage
    }

    /// Feeds one decoded system message. Returns `None` for anything other than a
    /// `Snap`/`SnapSingle`/`SnapEmpty` variant (including `SnapSmall`, see the module docs), for a
    /// part that is structurally invalid, or while still waiting on more parts of an in-progress
    /// tick — every one of those is silently dropped, matching the C++ reference's bare `return;`
    /// in every such case (never panics regardless of how malformed/hostile the fields are). A
    /// stale/already-acked tick is reported as `Some(Event::Stale)` instead — see that variant's
    /// docs.
    pub fn feed(&mut self, msg: &SysMsg) -> Option<Event> {
        let (tick, delta_tick, num_parts, part, crc, data) = match msg {
            SysMsg::Snap {
                tick,
                delta_tick,
                num_parts,
                part,
                crc,
                data,
            } => (*tick, *delta_tick, *num_parts, *part, Some(*crc), data.as_slice()),
            SysMsg::SnapSingle {
                tick,
                delta_tick,
                crc,
                data,
            } => (*tick, *delta_tick, 1, 0, Some(*crc), data.as_slice()),
            SysMsg::SnapEmpty { tick, delta_tick } => (*tick, *delta_tick, 1, 0, None, [].as_slice()),
            // `SysMsg::SnapSmall` and everything else: not fed into assembly at all — see the
            // module docs (F7).
            _ => return None,
        };

        if !(1..=crate::snapshot::MAX_PARTS as i32).contains(&num_parts)
            || !(0..num_parts).contains(&part)
            || data.len() > MAX_SNAPSHOT_PACKSIZE
        {
            return None;
        }
        if !(tick >= self.current_recv_tick && tick > self.ack_game_tick) {
            // Stale or already-acked: matches `client.cpp:2144-2145`'s guard — see
            // `Event::Stale`'s docs for why this is reported, not a bare drop.
            return Some(Event::Stale { tick });
        }

        if tick != self.current_recv_tick {
            self.parts_received = 0;
            self.current_recv_tick = tick;
            self.incoming_size = 0;
        }

        let offset = part as usize * MAX_SNAPSHOT_PACKSIZE;
        let copy_len = data.len().min(self.incoming.len().saturating_sub(offset));
        self.incoming[offset..offset + copy_len].copy_from_slice(&data[..copy_len]);
        self.parts_received |= 1u64 << part;

        if part == num_parts - 1 {
            self.incoming_size = (num_parts - 1) as usize * MAX_SNAPSHOT_PACKSIZE + data.len();
        }

        let all_parts_mask = if num_parts as usize == crate::snapshot::MAX_PARTS {
            u64::MAX
        } else {
            (1u64 << num_parts) - 1
        };
        if self.parts_received & all_parts_mask != all_parts_mask {
            return None; // still waiting on more parts
        }
        self.parts_received = 0;

        let base: Snapshot = if delta_tick < 0 {
            Snapshot::empty()
        } else {
            match self.storage.get(delta_tick) {
                Some(s) => s.clone(),
                None => {
                    self.ack_game_tick = -1;
                    return Some(Event::Resync { tick });
                }
            }
        };

        let mut ints = Vec::new();
        if self.incoming_size == 0 {
            ints.extend_from_slice(&[0, 0, 0]); // `CSnapshotDelta::EmptyDelta`
        } else if crate::packer::unpack_ints(&self.incoming[..self.incoming_size], &mut ints).is_none() {
            return None; // decompression failed: `client.cpp:2199-2202`'s bare `return;`
        }

        let snap = match delta::unpack_delta(&base, &ints, &self.static_sizes) {
            Ok(s) => s,
            Err(error) => return Some(Event::DeltaError { tick, error }),
        };

        if let Some(wanted) = crc {
            let got = snap.crc();
            if wanted as u32 != got {
                self.crc_errors += 1;
                if self.crc_errors > 10 {
                    self.ack_game_tick = -1;
                    self.crc_errors = 0;
                }
                return Some(Event::CrcMismatch {
                    tick,
                    wanted: wanted as u32,
                    got,
                    crc_errors: self.crc_errors,
                });
            } else if self.crc_errors > 0 {
                self.crc_errors -= 1;
            }
        }

        self.storage.purge_until(delta_tick);
        self.storage.add(tick, snap.clone());
        self.ack_game_tick = tick;
        Some(Event::Snapshot { tick, snap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated::objects;
    use crate::snapshot::SnapshotItem;

    fn flag_snap(x: i32) -> Snapshot {
        Snapshot {
            items: vec![SnapshotItem {
                key: objects::Flag::ID << 16,
                data: vec![x, 0, 0],
            }],
        }
    }

    fn compress(ints: &[i32]) -> Vec<u8> {
        let mut buf = vec![0u8; ints.len() * crate::packer::MAX_BYTES_PACKED];
        let n = crate::packer::pack_ints(&mut buf, ints).unwrap();
        buf.truncate(n);
        buf
    }

    #[test]
    fn snapempty_against_empty_base_produces_empty_snapshot() {
        let mut asm = SnapAssembler::new(StaticSizes::ddnet_06());
        let msg = SysMsg::SnapEmpty {
            tick: 10,
            delta_tick: -1,
        };
        let event = asm.feed(&msg).unwrap();
        assert_eq!(
            event,
            Event::Snapshot {
                tick: 10,
                snap: Snapshot::empty()
            }
        );
        assert_eq!(asm.ack_game_tick(), 10);
    }

    #[test]
    fn snapsingle_full_flow_and_delta_against_prior_snapshot() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);

        // First: an empty base at tick 10.
        asm.feed(&SysMsg::SnapEmpty {
            tick: 10,
            delta_tick: -1,
        });

        // Then: a SnapSingle at tick 12, delta against tick 10, adding one Flag item.
        let target = flag_snap(42);
        let delta_ints = delta::create_delta(&Snapshot::empty(), &target, &StaticSizes::ddnet_06()).unwrap();
        let data = compress(&delta_ints);
        let msg = SysMsg::SnapSingle {
            tick: 12,
            delta_tick: 10,
            crc: target.crc() as i32,
            data,
        };
        let event = asm.feed(&msg).unwrap();
        assert_eq!(
            event,
            Event::Snapshot {
                tick: 12,
                snap: target.clone()
            }
        );
        assert_eq!(asm.storage().get(12), Some(&target));
        assert_eq!(asm.ack_game_tick(), 12);
    }

    #[test]
    fn multi_part_snap_assembles_only_once_all_parts_arrive() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        // A big-enough snapshot that its compressed delta needs 2 parts (every non-last part is
        // exactly MAX_SNAPSHOT_PACKSIZE bytes on the real wire, `client.cpp:2154`'s
        // `Part * MAX_SNAPSHOT_PACKSIZE` addressing — a real server never sends a short
        // intermediate part).
        let target = Snapshot {
            items: (0..150)
                .map(|id| SnapshotItem {
                    key: (objects::Flag::ID << 16) | id,
                    data: vec![100_000 + id, 200_000 + id, 0],
                })
                .collect(),
        };
        let delta_ints = delta::create_delta(&Snapshot::empty(), &target, &StaticSizes::ddnet_06()).unwrap();
        let data = compress(&delta_ints);
        let num_parts = data.len().div_ceil(MAX_SNAPSHOT_PACKSIZE);
        assert!(
            (2..=crate::snapshot::MAX_PARTS).contains(&num_parts),
            "test setup: need a delta spanning 2+ parts, got {} bytes ({num_parts} parts)",
            data.len()
        );

        let mut last_event = None;
        for (part, chunk) in data.chunks(MAX_SNAPSHOT_PACKSIZE).enumerate() {
            let msg = SysMsg::Snap {
                tick: 5,
                delta_tick: -1,
                num_parts: num_parts as i32,
                part: part as i32,
                crc: target.crc() as i32,
                data: chunk.to_vec(),
            };
            let event = asm.feed(&msg);
            if part + 1 < num_parts {
                assert_eq!(event, None, "must not assemble before every part has arrived");
            } else {
                last_event = event;
            }
        }
        assert_eq!(last_event, Some(Event::Snapshot { tick: 5, snap: target }));
    }

    #[test]
    fn crc_mismatch_is_reported_and_snapshot_is_not_stored() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        let target = flag_snap(1);
        let delta_ints = delta::create_delta(&Snapshot::empty(), &target, &StaticSizes::ddnet_06()).unwrap();
        let data = compress(&delta_ints);
        let msg = SysMsg::SnapSingle {
            tick: 1,
            delta_tick: -1,
            crc: 0xdead, // wrong on purpose
            data,
        };
        let event = asm.feed(&msg).unwrap();
        assert!(matches!(event, Event::CrcMismatch { tick: 1, .. }));
        assert_eq!(asm.storage().get(1), None);
    }

    #[test]
    fn missing_delta_base_triggers_resync() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        let msg = SysMsg::SnapEmpty {
            tick: 100,
            delta_tick: 50, // never received / already purged
        };
        let event = asm.feed(&msg).unwrap();
        assert_eq!(event, Event::Resync { tick: 100 });
        assert_eq!(asm.ack_game_tick(), -1);
    }

    #[test]
    fn stale_tick_is_reported_as_event_not_reprocessed_and_not_panicked() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        asm.feed(&SysMsg::SnapEmpty {
            tick: 10,
            delta_tick: -1,
        });
        // A tick we've already acked: must be reported as `Event::Stale`, not reprocessed as a
        // fresh `Event::Snapshot` (F1, review round 1 — this used to be a silent `None`,
        // indistinguishable from "still assembling a multi-part tick").
        let event = asm.feed(&SysMsg::SnapEmpty {
            tick: 10,
            delta_tick: -1,
        });
        assert_eq!(event, Some(Event::Stale { tick: 10 }));
        let event2 = asm.feed(&SysMsg::SnapEmpty {
            tick: 5,
            delta_tick: -1,
        });
        assert_eq!(event2, Some(Event::Stale { tick: 5 }));
    }

    #[test]
    fn snapsmall_is_never_assembled() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        let event = asm.feed(&SysMsg::SnapSmall {
            tick: 10,
            delta_tick: -1,
            crc: 0,
            data: vec![],
        });
        assert_eq!(event, None, "F7: a real DDNet client ignores NETMSG_SNAPSMALL outright");
        assert_eq!(asm.storage().len(), 0);
    }

    #[test]
    fn reset_recovers_snapshots_after_a_simulated_map_change() {
        // Directly exercises F1's repro shape: a map change resets the server's tick counter, so
        // without `reset()` every post-change tick looks stale forever.
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);

        asm.feed(&SysMsg::SnapEmpty {
            tick: 5000,
            delta_tick: -1,
        });
        assert_eq!(asm.ack_game_tick(), 5000);

        // Without a reset, a post-map-change tick smaller than 5000 is stale forever.
        let stale = asm.feed(&SysMsg::SnapEmpty {
            tick: 10,
            delta_tick: -1,
        });
        assert_eq!(stale, Some(Event::Stale { tick: 10 }));

        // The session driver calls `reset()` right when it sends `NETMSG_ENTERGAME` again...
        asm.reset();
        assert_eq!(asm.ack_game_tick(), -1);
        assert_eq!(asm.storage().len(), 0);

        // ... and the very same tick that was "stale" a moment ago now assembles cleanly.
        let event = asm.feed(&SysMsg::SnapEmpty {
            tick: 10,
            delta_tick: -1,
        });
        assert_eq!(
            event,
            Some(Event::Snapshot {
                tick: 10,
                snap: Snapshot::empty()
            })
        );
    }

    #[test]
    fn structurally_invalid_part_fields_never_panic() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        for (num_parts, part) in [(0, 0), (-1, 0), (65, 0), (2, 2), (2, -1)] {
            let msg = SysMsg::Snap {
                tick: 1,
                delta_tick: -1,
                num_parts,
                part,
                crc: 0,
                data: vec![],
            };
            assert_eq!(asm.feed(&msg), None);
        }
    }

    #[test]
    fn malformed_delta_bytes_never_panic() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        let msg = SysMsg::SnapSingle {
            tick: 1,
            delta_tick: -1,
            crc: 0,
            data: vec![0xff, 0xff, 0xff, 0xff, 0xff], // garbage, no valid varint termination
        };
        let _ = asm.feed(&msg); // must not panic, whatever it returns
    }

    #[test]
    fn non_snap_message_returns_none() {
        let sizes = StaticSizes::ddnet_06();
        let mut asm = SnapAssembler::new(sizes);
        assert_eq!(asm.feed(&SysMsg::Ready), None);
    }
}
