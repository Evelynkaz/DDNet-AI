//! Offline per-player trajectory/input reconstruction (task 8.4a acceptance criterion 3) from a
//! sequence of already-decoded [`crate::format::Frame`]s (typically everything
//! [`crate::reader::RecordingReader`] yields for one recording, collected into memory — this is
//! explicitly an *offline* tool, not a streaming one; a recording is this project's own compact
//! format, not raw demo footage, so holding one in memory is the same order of cost as holding
//! the physics trace formats `ddai-trace` already loads whole).
//!
//! Two things come out of this per player:
//! - [`PlayerReconstruction::trajectory`]: position/velocity/hook/weapon/direction/aim per tick,
//!   taken from `character.tick` (`ddai_net`'s "dead-reckoning tick", task 2.2b's F4 finding,
//!   `docs/formats.md` §13.8) — not the snapshot's own tick, exactly like that section's own
//!   trajectory extraction. These fields are transmitted as-is by the server; nothing here
//!   *estimates* them.
//! - [`PlayerReconstruction::inputs`]/[`PlayerReconstruction::fire_ticks`]: the actual estimation
//!   this task asks for — see [`FieldConfidence`] and the module-level docs on each field's own
//!   heuristic, all derived from reading DDNet 20.1's own server source
//!   (`game/server/entities/character.cpp`/`game/gamecore.cpp` via this project's own
//!   `ddai-physics` port, which shares identical field semantics) and validated empirically
//!   against known scripted inputs (`ddnet-ai play --brain random-scripted --input-log`,
//!   `ddnet-ai rec reconstruct --validate`) — see the crate's BUILD REPORT for the measured
//!   accuracy per field, which is the number that actually matters, not the theory below.

use crate::format::{CharacterRecord, Frame};
use ddai_net::generated::enums::characterflagflag;
use std::collections::{BTreeMap, HashMap};

/// `HOOK_IDLE` (`game/gamecore.h:115`) — the only hook state this crate treats as "button not
/// held"; every other value (`HOOK_RETRACTED = -1` included: a hook can retract on its own, e.g.
/// hitting nothing at max range, while the player is still holding the button) counts as "held".
const HOOK_IDLE: i32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Transmitted by the server as-is; this crate does not infer it.
    Exact,
    /// Inferred from a physics side-effect (a state-bit transition, a velocity change, a
    /// timestamp field) because the raw button state is never transmitted to a spectator at all.
    Estimated,
}

/// Per-field confidence for one player's reconstructed inputs — task acceptance criterion 3's
/// "mark each input field as exact or estimated". `aim` depends on whether the DDNet character
/// extension (`character@netobj.ddnet.tw`, carrying `target_x`/`target_y`) was present on *every*
/// observed sample for this player — true for any real DDNet 20.x server (`ddai_net::view`'s own
/// docs: "it always does for a real DDNet 20.x server"), so `Estimated` here in practice only
/// means a non-DDNet 0.6 peer was seen, or the sample set was empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldConfidence {
    pub direction: Confidence,
    pub aim: Confidence,
    pub jump: Confidence,
    pub hook: Confidence,
    pub fire: Confidence,
}

impl FieldConfidence {
    fn with_aim(aim: Confidence) -> Self {
        FieldConfidence {
            direction: Confidence::Exact,
            aim,
            jump: Confidence::Estimated,
            hook: Confidence::Estimated,
            fire: Confidence::Estimated,
        }
    }
}

/// One tick of a player's physics state — task acceptance criterion 3's trajectory: "position/
/// velocity per tick from the reckoning core (`character.tick`) ... hook state, freeze state,
/// weapon, direction, aim angle as sent in the snapshot". `tick` is `character.tick` (see the
/// module docs), not the snapshot's own tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
    pub tick: i32,
    pub x: i32,
    pub y: i32,
    pub vel_x: i32,
    pub vel_y: i32,
    pub hook_state: i32,
    pub hooked_player: i32,
    pub weapon: i32,
    pub direction: i32,
    /// Review round 1, finding F3: whether `Direction`/`Jump`/(`Hook`, while additionally
    /// `freeze_end != 0`) were server-forced to zero *this tick* — `character.cpp:2235-2253`
    /// (`CCharacter::DDRaceTick`): `freeze_end != 0` (a real, active timed freeze — `-1` while
    /// deep-frozen, or `Server()->Tick() + m_FreezeTime` otherwise, `character.cpp:1343`, so this
    /// is never `0` while `m_FreezeTime > 0`) **or** `flags & CHARACTERFLAG_MOVEMENTS_DISABLED`
    /// (`m_Core.m_LiveFrozen`, `character.cpp:1340-1341`). `Some` only when the DDNet extension
    /// was present — exact when present, never guessed when absent. This is **not** the same
    /// thing as [`Self::in_freeze_tile`] — round 1 of this crate conflated the two (using the tile
    /// flag here instead), which is exactly finding F3: `CHARACTERFLAG_IN_FREEZE` only means "on a
    /// freeze/death/switch tile right now", a map-position fact that does not by itself force any
    /// input to zero (a character can be on such a tile for one tick before its freeze timer
    /// actually starts, or keep residual `IN_FREEZE` after `freeze_end`/`MOVEMENTS_DISABLED`
    /// already cleared).
    pub freeze: Option<bool>,
    /// Review round 1, finding F3: "standing on a freeze/death/switch tile right now"
    /// (`CHARACTERFLAG_IN_FREEZE`, `m_Core.m_IsInFreeze`, computed by the tile-check loop at
    /// `character.cpp:2258-2275`) — a tile/position property, kept separate from [`Self::freeze`]
    /// (whether inputs were actually forced to zero) precisely because the two are not
    /// interchangeable; see `freeze`'s own doc comment for the distinction and why conflating them
    /// was itself a bug.
    pub in_freeze_tile: Option<bool>,
    pub aim_x: i32,
    pub aim_y: i32,
    /// Review round 2, finding F18: the raw `Character::jumped` bitfield as sent by the server —
    /// bit 0 (`& 1`) is "jump button currently held" (already used for [`EstimatedInput::jump`]'s
    /// own rising-edge detection), bit 1 (`& 2`) is "the second/air jump has already been used this
    /// flight" (`gamecore.cpp:227-256`: once set, no further jump executes until the tee touches
    /// ground again, *regardless* of a still-held or freshly-pressed button) — a validator judging
    /// whether a ground-truth jump *press* could possibly have executed needs this bit, which
    /// [`EstimatedInput::jump`] alone cannot answer (that field only reports whether a press was
    /// *detected*, not why one might legitimately fail to register). Always present on the base
    /// `Character` object (unlike `freeze`/`in_freeze_tile`, which need the DDNet extension) — no
    /// `Option` wrapper, no confidence caveat.
    pub jumped: i32,
}

/// One tick's estimated direction/aim/jump/hook — task acceptance criterion 3's input estimation,
/// minus `fire` (see [`PlayerReconstruction::fire_ticks`] for why that one is reported
/// separately, on its own tick grid, rather than folded in here).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimatedInput {
    pub tick: i32,
    /// Exact (`Character::direction`, echoed back by the server's physics unmodified).
    pub direction: i32,
    pub aim_x: i32,
    pub aim_y: i32,
    /// Estimated: a rising edge of `Character::jumped`'s bit 0 *or* bit 1 (`gamecore.cpp`/
    /// `ddai-physics`'s `core.rs`: bit 0 is set the tick a jump actually executes, held at 1 while
    /// the button stays down, cleared the instant the button is released; bit 1 is "the second/air
    /// jump has already been used this flight", set in the *same* tick an air jump executes
    /// (`gamecore.cpp:249-253`) and cleared only on landing — see the module docs) between this
    /// sample and the previous one for the same player. Either bit's own rising edge, on its own,
    /// counts as one jump event (never double-counted when both happen to rise on the very same
    /// sample, e.g. a grounded jump with the double-jump tuning disabled sets both at once).
    ///
    /// Review round 3, finding F20 step (4): bit 1's rising edge is the one addition over round
    /// 1-2's bit-0-only detection — it exists because bit 1, once set by an air jump, *persists*
    /// until landing (unlike bit 0, which a brief press/release can clear again within a single
    /// dead-reckoning sampling gap), so a sample taken even one tick *after* a missed air-jump
    /// press can still catch it via bit 1's own rising edge. Live testing (a per-event
    /// classification across 8 recordings, see the crate's BUILD REPORT) found this exact
    /// pattern: a miss at the sample immediately after the press, with `jumped == 2` (bit 0
    /// already released, bit 1 freshly set) where the previous sample had bit 1 clear — bit 1's
    /// rising edge reports that as a jump event even though bit 0's own rising edge never had a
    /// sample land on it at all.
    ///
    /// **Known, undetectable residual gap, documented rather than silently claimed fixed**: a
    /// single ground jump that lasts only one input tick, with the double-jump tuning *enabled*
    /// (the DDNet default, `m_Jumps == 2`), sets *only* bit 0 (`gamecore.cpp:241-247`) — no bit 1
    /// transition happens at all for it. If dead-reckoning sampling misses that one tick entirely
    /// (the same structural cause `random_scripted_input`'s own `PULSE_TICKS` fix, finding F20
    /// step (1), addresses on the *validation* side by never sending a pulse that brief), this
    /// specific case has no bit left to reveal it after the fact — a real, physical limit of what
    /// is observable from a spectator's own dead-reckoning stream at DDNet's ~25Hz snapshot rate,
    /// not a gap this crate's own estimation logic can close.
    ///
    /// The very first observed sample for a player is judged in isolation for *each* bit (no
    /// earlier sample to compare against): a bit already set there is still reported as a jump
    /// event, since a continuously-held jump from before recording started is indistinguishable
    /// from one that started exactly on this tick, and the task's "known scripted input"
    /// validation never holds jump across the very first observed sample anyway
    /// (`random_scripted_input`'s pulses are brief, `ddnet-ai`'s crate).
    pub jump: bool,
    /// Estimated: `hook_state != HOOK_IDLE` — see [`HOOK_IDLE`]'s doc comment for why every other
    /// state value (including the involuntary `HOOK_RETRACTED`) counts as "held".
    pub hook: bool,
}

/// One player's full reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerReconstruction {
    pub client_id: i32,
    /// Review round 1, finding F5: a client id *slot* can be occupied by a different real person
    /// than it was earlier in the same recording (the earlier occupant disconnected — the server
    /// stops sending `PlayerInfo`/`ClientInfo` for that id at all until someone new connects into
    /// it — and a later connection was assigned the same, now-free, id). `stint` counts these
    /// occupancies for one `client_id`, starting at `0`: a new stint begins whenever this id was
    /// completely absent from a `Snapshot` frame's `players` list for at least one whole frame
    /// (see [`reconstruct`]'s own doc comment for exactly how that is detected), or its
    /// `ClientInfo::name` changed while continuously present (a real nick change, `Cl_ChangeInfo`
    /// — distinct from a slot handoff, but just as much "not the same track" for trajectory/input
    /// purposes). Each `(client_id, stint)` pair gets its own, independent [`Self::trajectory`]/
    /// [`Self::inputs`]/[`Self::fire_ticks`] — one is never allowed to bleed into another.
    pub stint: u32,
    pub trajectory: Vec<TrajectorySample>,
    pub inputs: Vec<EstimatedInput>,
    /// Estimated: every *newly seen* `Character::attack_tick` value, in the order first
    /// observed — `character.cpp:649`'s `m_AttackTick = Server()->Tick()`, set to the exact
    /// server tick `FireWeapon()` ran on the tick it happens, not merely "some tick around when
    /// we noticed it changed" — so this list is tick numbers straight from that field, not from
    /// whichever of our own samples happened to observe the change first. Reported separately
    /// from [`Self::inputs`] (rather than a per-sample `fire: bool` on the trajectory's own tick
    /// grid) because its natural granularity is "one tick per fire event", which will not
    /// generally land exactly on an observed trajectory sample.
    pub fire_ticks: Vec<i32>,
    pub confidence: FieldConfidence,
}

fn aim_from_ddnet_or_angle(character: &CharacterRecord) -> ((i32, i32), Confidence) {
    if let Some(ddnet) = &character.ddnet {
        ((ddnet.target_x, ddnet.target_y), Confidence::Exact)
    } else {
        (aim_from_angle(character.character.angle), Confidence::Estimated)
    }
}

/// Inverts `ddai_physics::core::angle_from_target`/`gamecore.cpp`'s `atan2(y, x) * 256`
/// wire-angle encoding — the *direction* recovers exactly (mod the ±1/256 turn quantization
/// already baked into the wire format itself), but the original aim vector's *magnitude* is lost
/// (the wire angle carries no magnitude at all), so this picks an arbitrary nominal magnitude
/// (1000, a typical real aim distance) purely so the returned vector points the right way — never
/// treated as [`Confidence::Exact`] by any caller.
fn aim_from_angle(angle_raw: i32) -> (i32, i32) {
    const NOMINAL_MAGNITUDE: f64 = 1000.0;
    let angle_rad = f64::from(angle_raw) / 256.0;
    (
        (angle_rad.cos() * NOMINAL_MAGNITUDE).round() as i32,
        (angle_rad.sin() * NOMINAL_MAGNITUDE).round() as i32,
    )
}

/// Review round 1, finding F3: see [`TrajectorySample::freeze`]'s doc comment for the exact
/// citations — this is "were inputs forced to zero", not "on a freeze tile" (that one is
/// [`is_in_freeze_tile`]).
fn is_movement_frozen(character: &CharacterRecord) -> Option<bool> {
    character
        .ddnet
        .as_ref()
        .map(|d| d.freeze_end != 0 || d.flags & characterflagflag::MOVEMENTS_DISABLED != 0)
}

/// Review round 1, finding F3: see [`TrajectorySample::in_freeze_tile`]'s doc comment.
fn is_in_freeze_tile(character: &CharacterRecord) -> Option<bool> {
    character
        .ddnet
        .as_ref()
        .map(|d| d.flags & characterflagflag::IN_FREEZE != 0)
}

/// Reconstructs every player seen across `frames` — task acceptance criterion 3's whole surface,
/// as a single entry point `ddnet-ai rec reconstruct` calls. Frames are consumed in the order
/// given (the same order [`crate::reader::RecordingReader::next_frame`] yields them — a
/// recording's own arrival order); only [`Frame::Snapshot`] frames contribute (game messages do
/// not carry per-player physics state).
/// Review round 1, finding F5: per-`client_id` identity tracking across `Snapshot` frames, kept
/// separate from [`Building`] (which accumulates one *stint*'s actual trajectory/input data) —
/// this only decides *which* stint a given frame's data belongs to. `pub(crate)` (round 2, finding
/// F19): [`crate::anonymize::Anonymizer`] needs the exact same stint boundaries this module uses,
/// so it can key its own per-player anonymized ids by `(client_id, stint)` too, not just
/// `client_id` — reusing this type/[`stint_for`] directly (rather than a second, independent
/// implementation) is the only way the two can never silently drift out of agreement.
#[derive(Debug)]
pub(crate) struct Identity {
    /// The stint index currently in effect for this id.
    current_stint: u32,
    /// This id's `ClientInfo::name` as of the last frame it was seen in `players` — `None` if it
    /// has never had one (no `ClientInfo` at all, an unusual but tolerated case: never itself
    /// treated as "the name changed").
    name: Option<String>,
    /// The ordinal index (into the sequence of `Snapshot` frames only, 0-based) of the last frame
    /// this id was present in `players` — used to detect a gap (id vanished from `players` for at
    /// least one whole frame, then came back) as a slot handoff.
    last_seen_frame: usize,
}

/// Review round 1, finding F5's core mechanism: given `frame_index` (this `Snapshot` frame's
/// 0-based ordinal among `Snapshot` frames only) and this frame's `players` list, updates
/// `identities` and returns the current stint for `client_id` — creating a fresh stint (`0` if
/// never seen before, otherwise the previous stint's index `+ 1`) exactly when:
/// - this id was absent from `players` for at least one whole frame since it was last seen there
///   (`frame_index` jumped by more than 1 past `last_seen_frame`) — the server stops sending
///   `PlayerInfo`/`ClientInfo` for a disconnected client entirely, so a gap here means the
///   underlying connection actually dropped and something new took the slot; or
/// - its `ClientInfo::name` changed since it was last seen (a real nick change, `Cl_ChangeInfo`)
///   even with no such gap.
///
/// Called once per id actually present in `players` this frame; a `client_id` seen in
/// `characters` but *not* in this frame's own `players` (should not happen on a well-formed
/// server, but this crate never assumes it can't) simply keeps whatever stint it was last
/// assigned — a single frame's absence from `characters` (dead reckoning, no `Character` update
/// this tick, or genuinely no character right now) is normal and must not, by itself, split
/// anything; only `players` absence is the identity signal.
pub(crate) fn stint_for(
    identities: &mut HashMap<i32, Identity>,
    client_id: i32,
    frame_index: usize,
    name: Option<&str>,
) -> u32 {
    match identities.get_mut(&client_id) {
        None => {
            identities.insert(
                client_id,
                Identity {
                    current_stint: 0,
                    name: name.map(str::to_string),
                    last_seen_frame: frame_index,
                },
            );
            0
        }
        Some(id) => {
            let gap = frame_index.saturating_sub(id.last_seen_frame) > 1;
            let name_changed = match (&id.name, name) {
                (Some(old), Some(new)) => old != new,
                _ => false,
            };
            if gap || name_changed {
                id.current_stint += 1;
            }
            id.name = name.map(str::to_string).or_else(|| id.name.take());
            id.last_seen_frame = frame_index;
            id.current_stint
        }
    }
}

/// See the module docs for the overall shape; review round 1, finding F5 added the stint
/// mechanism ([`stint_for`]) this function now runs on every `Snapshot` frame *before* looking at
/// that frame's `characters`, so every character sample is attributed to the right
/// `(client_id, stint)` track from the start — a two-pass "attribute first, then split
/// afterwards" design would have to re-derive exactly the same per-tick boundary anyway, with no
/// benefit.
pub fn reconstruct(frames: &[Frame]) -> Vec<PlayerReconstruction> {
    // Per (client id, stint): the trajectory built so far, the last `character.tick` pushed (to
    // dedup repeated identical dead-reckoning ticks across consecutive snapshots — §13.8's own
    // rationale), the last `jumped` bit-0 value seen (for rising-edge jump detection), the last
    // `attack_tick` value seen (for fire-event detection — review round 1, finding F11's baseline/
    // monotonic-only tracking), and whether every sample so far had the DDNet extension.
    struct Building {
        trajectory: Vec<TrajectorySample>,
        inputs: Vec<EstimatedInput>,
        fire_ticks: Vec<i32>,
        last_tick: Option<i32>,
        last_jump_bit0: Option<bool>,
        last_jump_bit1: Option<bool>,
        last_attack_tick: Option<i32>,
        all_had_ddnet_ext: bool,
        saw_any_ddnet_ext: bool,
    }

    let mut identities: HashMap<i32, Identity> = HashMap::new();
    let mut by_player: BTreeMap<(i32, u32), Building> = BTreeMap::new();
    let mut frame_index: usize = 0;

    for frame in frames {
        let Frame::Snapshot {
            characters, players, ..
        } = frame
        else {
            continue;
        };

        // Review round 1, finding F5: update identity/stint tracking from *every* id present in
        // `players` this frame, before looking at `characters` at all — a player with no live
        // `Character` (dead, or between rounds) still has a `PlayerInfo`/`ClientInfo` entry and
        // must still be tracked, so a later stint boundary is computed against the right
        // `last_seen_frame`.
        for p in players {
            let name = p.client_info.as_ref().map(|ci| ci.name.as_str());
            stint_for(&mut identities, p.id, frame_index, name);
        }

        for c in characters {
            // The stint for `c.id` as of *this* frame — `stint_for` was already called above for
            // every id in `players` this frame (including `c.id`, on any well-formed server); if
            // `c.id` was somehow missing from `players` this frame, this reuses whatever stint it
            // was last assigned (or starts stint 0 if this id has genuinely never been seen in
            // `players` at all) rather than inventing a spurious new one.
            let stint = stint_for(&mut identities, c.id, frame_index, None);
            let key = (c.id, stint);

            let entry = by_player.entry(key).or_insert_with(|| Building {
                trajectory: Vec::new(),
                inputs: Vec::new(),
                fire_ticks: Vec::new(),
                last_tick: None,
                last_jump_bit0: None,
                last_jump_bit1: None,
                last_attack_tick: None,
                all_had_ddnet_ext: true,
                saw_any_ddnet_ext: false,
            });

            if entry.last_tick == Some(c.character.tick) {
                continue; // same dead-reckoning tick already recorded — a repeat, not new data.
            }
            entry.last_tick = Some(c.character.tick);

            if c.ddnet.is_some() {
                entry.saw_any_ddnet_ext = true;
            } else {
                entry.all_had_ddnet_ext = false;
            }

            let ((aim_x, aim_y), _aim_confidence_this_sample) = aim_from_ddnet_or_angle(c);

            entry.trajectory.push(TrajectorySample {
                tick: c.character.tick,
                x: c.character.x,
                y: c.character.y,
                vel_x: c.character.vel_x,
                vel_y: c.character.vel_y,
                hook_state: c.character.hook_state,
                hooked_player: c.character.hooked_player,
                weapon: c.character.weapon,
                direction: c.character.direction,
                freeze: is_movement_frozen(c),
                in_freeze_tile: is_in_freeze_tile(c),
                aim_x,
                aim_y,
                jumped: c.character.jumped,
            });

            // Review round 3, finding F20 step (4): a rising edge of bit 0 ("jump button held")
            // OR of bit 1 ("second/air jump already used", `gamecore.cpp:249-253` — set the same
            // tick an air jump executes, exactly like bit 0 is for a ground jump) each count as a
            // jump event on their own, independently — see `EstimatedInput::jump`'s own doc
            // comment for why bit 1 catches genuine air jumps that bit 0's own rising edge can
            // miss (a 1-tick-long ground jump with the double-jump tuning enabled is the one case
            // neither bit can see — documented there too, not silently claimed fixed).
            let jump_bit0_now = c.character.jumped & 1 != 0;
            let jump_bit1_now = c.character.jumped & 2 != 0;
            let bit0_rising = match entry.last_jump_bit0 {
                Some(prev) => jump_bit0_now && !prev,
                // First observed sample for this player: no prior context — see
                // `EstimatedInput::jump`'s doc comment for why this still counts as an event.
                None => jump_bit0_now,
            };
            let bit1_rising = match entry.last_jump_bit1 {
                Some(prev) => jump_bit1_now && !prev,
                None => jump_bit1_now,
            };
            let jump_event = bit0_rising || bit1_rising;
            entry.last_jump_bit0 = Some(jump_bit0_now);
            entry.last_jump_bit1 = Some(jump_bit1_now);

            entry.inputs.push(EstimatedInput {
                tick: c.character.tick,
                direction: c.character.direction,
                aim_x,
                aim_y,
                jump: jump_event,
                hook: c.character.hook_state != HOOK_IDLE,
            });

            // Review round 1, finding F11: two fixes over round 1's "any change is a fire event".
            // (1) The very *first* observed sample only establishes the baseline — the pre-
            // existing value it reveals was not necessarily a fire that happened while this crate
            // was watching (it could be the value the character already had). (2) Only a *strict
            // increase* counts as a real fire: `attack_tick` is stamped from the server's own
            // always-increasing tick counter (`character.cpp:649`, `m_AttackTick =
            // Server()->Tick()`), so a genuine fire can never produce a value smaller than the
            // last one seen — a decrease only happens on a reset (a respawn's fresh `CCharacter`,
            // or simply the start of a new stint/segment), which round 1 miscounted as a fire.
            // Residual, undetectable-from-sparse-sampling edge case, documented rather than
            // silently claimed fixed: `CCharacter::TickPaused` (`character.cpp:975`) also
            // increments `m_AttackTick` by exactly 1 for every tick the *whole game* is paused (an
            // admin/vote pause, not a normal part of block play) to keep tick-relative timers in
            // sync — indistinguishable, from sparse dead-reckoning samples alone, from a real
            // fire's tick jump. This project never sends a pause request itself (D-007: no chat,
            // no vote); a human player or admin pausing while being recorded remains a known,
            // accepted gap in `fire_ticks`' precision, not a false "fixed".
            match entry.last_attack_tick {
                None => entry.last_attack_tick = Some(c.character.attack_tick),
                Some(prev) if c.character.attack_tick > prev => {
                    entry.last_attack_tick = Some(c.character.attack_tick);
                    entry.fire_ticks.push(c.character.attack_tick);
                }
                Some(prev) if c.character.attack_tick < prev => {
                    entry.last_attack_tick = Some(c.character.attack_tick);
                }
                Some(_) => {}
            }
        }

        frame_index += 1;
    }

    by_player
        .into_iter()
        .map(|((client_id, stint), b)| PlayerReconstruction {
            client_id,
            stint,
            trajectory: b.trajectory,
            inputs: b.inputs,
            fire_ticks: b.fire_ticks,
            confidence: FieldConfidence::with_aim(if b.saw_any_ddnet_ext && b.all_had_ddnet_ext {
                Confidence::Exact
            } else {
                Confidence::Estimated
            }),
        })
        .collect()
}

/// One `(client_id, stint)` pair's last-known name and the (0-based, `Snapshot`-frame-only) index
/// of the last frame it was confirmed under that name — [`stint_names`]'s per-entry payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StintName {
    pub name: String,
    pub last_seen_frame: usize,
}

/// Review round 1, finding F5: a single pass over `frames` producing every `(client_id, stint)`
/// pair actually seen (in the exact same key space [`reconstruct`] itself uses, via the same
/// [`stint_for`] mechanism) together with the name last confirmed for it and when — the one
/// canonical source both `ddnet-ai rec inspect`'s player listing and [`resolve_name_to_stints`]
/// build on, so neither has to re-derive stint boundaries its own way (which could silently drift
/// out of sync with [`reconstruct`]'s own).
pub fn stint_names(frames: &[Frame]) -> BTreeMap<(i32, u32), StintName> {
    let mut identities: HashMap<i32, Identity> = HashMap::new();
    let mut names: BTreeMap<(i32, u32), StintName> = BTreeMap::new();
    let mut frame_index: usize = 0;
    for frame in frames {
        let Frame::Snapshot { players, .. } = frame else {
            continue;
        };
        for p in players {
            let this_name = p.client_info.as_ref().map(|ci| ci.name.as_str());
            let stint = stint_for(&mut identities, p.id, frame_index, this_name);
            if let Some(name) = this_name {
                names.insert(
                    (p.id, stint),
                    StintName {
                        name: name.to_string(),
                        last_seen_frame: frame_index,
                    },
                );
            }
        }
        frame_index += 1;
    }
    names
}

/// Review round 1, finding F5: resolves `name` to every `(client_id, stint)` pair it was ever seen
/// under, oldest first — built from [`stint_names`], so it can never disagree with how
/// [`reconstruct`]'s own output is actually keyed. A name that was reused across a presence gap
/// (the same real person reconnecting under the same nick, or an unrelated later occupant
/// coincidentally choosing the same nick) yields more than one pair — callers that want
/// "whichever stint most recently ran under this name" (e.g. `ddnet-ai rec reconstruct --validate
/// --player-name`, matching `load_input_log`'s own last-epoch-only semantics on the ground-truth
/// side) should take the last element.
pub fn resolve_name_to_stints(frames: &[Frame], name: &str) -> Vec<(i32, u32)> {
    let mut matches: Vec<(usize, (i32, u32))> = stint_names(frames)
        .into_iter()
        .filter(|(_, info)| info.name == name)
        .map(|(key, info)| (info.last_seen_frame, key))
        .collect();
    matches.sort_by_key(|(last_seen_frame, _)| *last_seen_frame);
    matches.into_iter().map(|(_, key)| key).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_net::generated::objects;

    fn character(id: i32, tick: i32, jumped: i32, attack_tick: i32, hook_state: i32) -> CharacterRecord {
        CharacterRecord {
            id,
            character: objects::Character {
                tick,
                x: tick * 10,
                y: 0,
                vel_x: 0,
                vel_y: 0,
                angle: 0,
                direction: 1,
                jumped,
                hooked_player: -1,
                hook_state,
                hook_tick: 0,
                hook_x: 0,
                hook_y: 0,
                hook_dx: 0,
                hook_dy: 0,
                player_flags: 1,
                health: 10,
                armor: 0,
                ammo_count: -1,
                weapon: 1,
                emote: 0,
                attack_tick,
            },
            ddnet: Some(objects::DDNetCharacter {
                flags: 0,
                freeze_end: 0,
                jumps: 2,
                tele_checkpoint: -1,
                strong_weak_id: 0,
                jumped_total: -1,
                ninja_activation_tick: -1,
                freeze_start: -1,
                target_x: 100,
                target_y: 0,
                tune_zone_override: -1,
            }),
        }
    }

    fn snapshot(tick: i32, characters: Vec<CharacterRecord>) -> Frame {
        Frame::Snapshot {
            tick,
            characters,
            players: vec![],
        }
    }

    #[test]
    fn trajectory_dedups_repeated_character_tick_across_snapshots() {
        // The same dead-reckoning tick (5) sent in two consecutive snapshots (server only resends
        // m_SendCore periodically, task 2.2b §13.8) must produce exactly one trajectory sample.
        let c = character(0, 5, 0, 0, 0);
        let frames = vec![snapshot(100, vec![c]), snapshot(101, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon.len(), 1);
        assert_eq!(recon[0].trajectory.len(), 1);
        assert_eq!(recon[0].trajectory[0].tick, 5);
    }

    #[test]
    fn direction_and_exact_aim_pass_through_unchanged() {
        let c = character(0, 5, 0, 0, 0);
        let frames = vec![snapshot(100, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].confidence.aim, Confidence::Exact);
        assert_eq!(recon[0].confidence.direction, Confidence::Exact);
        assert_eq!(recon[0].inputs[0].direction, 1);
        assert_eq!((recon[0].inputs[0].aim_x, recon[0].inputs[0].aim_y), (100, 0));
    }

    #[test]
    fn jump_rising_edge_is_detected_once_per_press_not_once_per_tick_held() {
        // jumped&1 goes 0 -> 1 -> 1 -> 0 (a 2-tick hold, then release) across 4 distinct ticks.
        let frames = vec![
            snapshot(0, vec![character(0, 0, 0b00, 0, 0)]),
            snapshot(1, vec![character(0, 1, 0b01, 0, 0)]), // rising edge here
            snapshot(2, vec![character(0, 2, 0b01, 0, 0)]), // still held, no new event
            snapshot(3, vec![character(0, 3, 0b00, 0, 0)]), // released
        ];
        let recon = reconstruct(&frames);
        let jumps: Vec<bool> = recon[0].inputs.iter().map(|i| i.jump).collect();
        assert_eq!(jumps, vec![false, true, false, false]);
    }

    /// Review round 3, finding F20 step (4): a rising edge of bit 1 alone (bit 0 already released
    /// again by the time this sample was taken — the exact miss pattern live testing found for a
    /// 1-tick air jump) must still be reported as a jump event.
    #[test]
    fn bit1_rising_edge_alone_is_detected_as_a_jump_when_bit0_was_already_missed() {
        let frames = vec![
            snapshot(0, vec![character(0, 0, 0b00, 0, 0)]), // neither bit set
            // The press+release of bit 0 happened entirely between these two samples (never
            // observed) — but bit 1 (second jump used) persists until landing, so its own rising
            // edge (0 -> 2) is still visible here.
            snapshot(1, vec![character(0, 1, 0b10, 0, 0)]),
            snapshot(2, vec![character(0, 2, 0b10, 0, 0)]), // still airborne, no new event
            snapshot(3, vec![character(0, 3, 0b00, 0, 0)]), // landed, bit 1 clears — no event either
        ];
        let recon = reconstruct(&frames);
        let jumps: Vec<bool> = recon[0].inputs.iter().map(|i| i.jump).collect();
        assert_eq!(jumps, vec![false, true, false, false]);
    }

    /// A grounded jump with the double-jump tuning disabled sets both bits in the very same
    /// sample (`0b00 -> 0b11`) — this must be reported as exactly one jump event, not two.
    #[test]
    fn both_bits_rising_on_the_same_sample_is_one_jump_event_not_two() {
        let frames = vec![
            snapshot(0, vec![character(0, 0, 0b00, 0, 0)]),
            snapshot(1, vec![character(0, 1, 0b11, 0, 0)]), // both bits rise together
        ];
        let recon = reconstruct(&frames);
        let jumps: Vec<bool> = recon[0].inputs.iter().map(|i| i.jump).collect();
        assert_eq!(jumps, vec![false, true]);
    }

    /// Review round 2, finding F18: `TrajectorySample::jumped` must carry the raw
    /// `Character::jumped` bitfield through verbatim (not just the derived rising-edge bit that
    /// `EstimatedInput::jump` reports) — a validator needs bit 1 (`& 2`, "second jump already
    /// used") too, which no other field exposes.
    #[test]
    fn trajectory_sample_carries_the_raw_jumped_bitfield_through_verbatim() {
        let frames = vec![snapshot(0, vec![character(0, 0, 0b11, 0, 0)])]; // held + second-jump-used
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].trajectory[0].jumped, 0b11);
    }

    #[test]
    fn hook_is_active_whenever_state_is_not_idle() {
        let frames = vec![
            snapshot(0, vec![character(0, 0, 0, 0, 0)]),  // HOOK_IDLE
            snapshot(1, vec![character(0, 1, 0, 0, 4)]),  // HOOK_FLYING
            snapshot(2, vec![character(0, 2, 0, 0, -1)]), // HOOK_RETRACTED — still "held"
            snapshot(3, vec![character(0, 3, 0, 0, 0)]),  // released
        ];
        let recon = reconstruct(&frames);
        let hooks: Vec<bool> = recon[0].inputs.iter().map(|i| i.hook).collect();
        assert_eq!(hooks, vec![false, true, true, false]);
    }

    #[test]
    fn fire_ticks_come_from_attack_tick_changes_on_its_own_grid() {
        let frames = vec![
            snapshot(0, vec![character(0, 0, 0, 0, 0)]),
            snapshot(1, vec![character(0, 1, 0, 0, 0)]),  // no change yet
            snapshot(2, vec![character(0, 2, 0, 42, 0)]), // fire happened at tick 42
            snapshot(3, vec![character(0, 3, 0, 42, 0)]), // same fire, no new event
            snapshot(4, vec![character(0, 4, 0, 50, 0)]), // second fire at tick 50
        ];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].fire_ticks, vec![42, 50]);
    }

    /// Review round 1, finding F11: the very first observed sample only establishes the
    /// `attack_tick` baseline — it must never itself be reported as a fire event, even when that
    /// first-seen value is non-zero (a player who was already alive with a nonzero attack_tick
    /// before this crate started observing).
    #[test]
    fn the_first_observed_attack_tick_sample_is_a_baseline_not_a_fire_event() {
        let frames = vec![snapshot(0, vec![character(0, 0, 0, 77, 0)])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].fire_ticks, Vec::<i32>::new());
    }

    /// Review round 1, finding F11: a *decrease* in `attack_tick` (a respawn resetting the
    /// character, or the start of a fresh segment) must rebase silently, not be misread as a fire.
    #[test]
    fn a_decrease_in_attack_tick_rebases_silently_instead_of_firing() {
        let frames = vec![
            snapshot(0, vec![character(0, 0, 0, 100, 0)]), // baseline
            snapshot(1, vec![character(0, 1, 0, 5, 0)]),   // respawn: attack_tick resets low
            snapshot(2, vec![character(0, 2, 0, 20, 0)]),  // a real fire after the respawn
        ];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].fire_ticks, vec![20]);
    }

    #[test]
    fn missing_ddnet_extension_falls_back_to_angle_and_is_marked_estimated() {
        let mut c = character(0, 0, 0, 0, 0);
        c.ddnet = None;
        c.character.angle = 0; // pointing right
        let frames = vec![snapshot(0, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].confidence.aim, Confidence::Estimated);
        // angle 0 -> pointing along +x.
        assert!(recon[0].inputs[0].aim_x > 0);
        assert_eq!(recon[0].inputs[0].aim_y, 0);
    }

    /// Review round 1, finding F3: `freeze` (inputs forced to zero) must come from
    /// `freeze_end`/`MOVEMENTS_DISABLED`, *not* `CHARACTERFLAG_IN_FREEZE` — merely being on a
    /// freeze tile (`IN_FREEZE`) with neither of those set must leave `freeze` at `Some(false)`.
    #[test]
    fn being_on_a_freeze_tile_alone_does_not_count_as_freeze() {
        let mut c = character(0, 0, 0, 0, 0);
        c.ddnet.as_mut().unwrap().flags = characterflagflag::IN_FREEZE; // tile only, no timer/live-frozen
        c.ddnet.as_mut().unwrap().freeze_end = 0;
        let frames = vec![snapshot(0, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].trajectory[0].freeze, Some(false));
        assert_eq!(recon[0].trajectory[0].in_freeze_tile, Some(true));
    }

    /// A real, active timed freeze (`freeze_end != 0`) is `freeze = true`, independent of whether
    /// `IN_FREEZE` (the tile flag) happens to be set too.
    #[test]
    fn an_active_timed_freeze_is_freeze_true_via_freeze_end() {
        let mut c = character(0, 0, 0, 0, 0);
        c.ddnet.as_mut().unwrap().freeze_end = 500; // Server()->Tick() + m_FreezeTime, some future tick
        let frames = vec![snapshot(0, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].trajectory[0].freeze, Some(true));
    }

    /// Deep-frozen encodes `freeze_end = -1` (`character.cpp:1343`), which must still count.
    #[test]
    fn deep_frozen_freeze_end_negative_one_is_freeze_true() {
        let mut c = character(0, 0, 0, 0, 0);
        c.ddnet.as_mut().unwrap().freeze_end = -1;
        let frames = vec![snapshot(0, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].trajectory[0].freeze, Some(true));
    }

    /// `CHARACTERFLAG_MOVEMENTS_DISABLED` (`m_LiveFrozen`) alone, with no active timed freeze,
    /// must also count as `freeze = true` (`character.cpp:1340-1341`/`2235-2240`).
    #[test]
    fn movements_disabled_alone_is_freeze_true() {
        let mut c = character(0, 0, 0, 0, 0);
        c.ddnet.as_mut().unwrap().flags = characterflagflag::MOVEMENTS_DISABLED;
        c.ddnet.as_mut().unwrap().freeze_end = 0;
        let frames = vec![snapshot(0, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].trajectory[0].freeze, Some(true));
    }

    #[test]
    fn freeze_is_none_without_the_ddnet_extension() {
        let mut c = character(0, 0, 0, 0, 0);
        c.ddnet = None;
        let frames = vec![snapshot(0, vec![c])];
        let recon = reconstruct(&frames);
        assert_eq!(recon[0].trajectory[0].freeze, None);
        assert_eq!(recon[0].trajectory[0].in_freeze_tile, None);
    }

    #[test]
    fn multiple_players_are_kept_independent() {
        let frames = vec![snapshot(
            0,
            vec![character(0, 0, 0b01, 0, 0), character(1, 0, 0b00, 0, 4)],
        )];
        let recon = reconstruct(&frames);
        assert_eq!(recon.len(), 2);
        assert_eq!(recon[0].client_id, 0);
        assert!(recon[0].inputs[0].jump);
        assert_eq!(recon[1].client_id, 1);
        assert!(recon[1].inputs[0].hook);
    }

    #[test]
    fn empty_frames_yields_no_players() {
        assert!(reconstruct(&[]).is_empty());
    }

    fn player(id: i32, name: &str) -> crate::format::PlayerRecord {
        crate::format::PlayerRecord {
            id,
            info: objects::PlayerInfo {
                local: 0,
                client_id: id,
                team: 0,
                score: 0,
                latency: 0,
            },
            client_info: Some(objects::ClientInfo {
                name: name.to_string(),
                clan: String::new(),
                country: -1,
                skin: String::new(),
                use_custom_color: 0,
                color_body: 0,
                color_feet: 0,
            }),
            ddnet: None,
        }
    }

    fn snapshot_with_players(
        tick: i32,
        characters: Vec<CharacterRecord>,
        players: Vec<crate::format::PlayerRecord>,
    ) -> Frame {
        Frame::Snapshot {
            tick,
            characters,
            players,
        }
    }

    /// Review round 1, finding F5: the same `client_id` slot, occupied by two different named
    /// people back-to-back with no presence gap in `players` at all (edge case: the server could
    /// in principle reuse a slot the very next snapshot), must still split into two stints because
    /// the name changed.
    #[test]
    fn a_name_change_on_the_same_slot_starts_a_new_stint() {
        let frames = vec![
            snapshot_with_players(0, vec![character(0, 0, 0, 10, 0)], vec![player(0, "Alice")]),
            snapshot_with_players(1, vec![character(0, 1, 0, 10, 0)], vec![player(0, "Alice")]),
            snapshot_with_players(2, vec![character(0, 2, 0, 10, 0)], vec![player(0, "Bob")]),
            snapshot_with_players(3, vec![character(0, 3, 0, 10, 0)], vec![player(0, "Bob")]),
        ];
        let recon = reconstruct(&frames);
        assert_eq!(
            recon.len(),
            2,
            "must produce two independent tracks, not one merged one"
        );
        assert_eq!(recon[0].client_id, 0);
        assert_eq!(recon[0].stint, 0);
        assert_eq!(recon[0].trajectory.len(), 2);
        assert_eq!(recon[1].client_id, 0);
        assert_eq!(recon[1].stint, 1);
        assert_eq!(recon[1].trajectory.len(), 2);
    }

    /// Review round 1, finding F5: a slot that disappears from `players` for at least one whole
    /// frame (the earlier occupant disconnected) and then reappears — even under the *same*
    /// name (e.g. a coincidence, or the same person reconnecting) — must still split into a new
    /// stint, since presence itself, not just the name, is the identity signal (a real
    /// reconnect resets in-game state such as spawn position that must not be glued onto the
    /// previous stint's trajectory).
    #[test]
    fn a_presence_gap_on_the_same_slot_starts_a_new_stint_even_with_the_same_name() {
        let frames = vec![
            snapshot_with_players(0, vec![character(0, 0, 0, 10, 0)], vec![player(0, "Alice")]),
            snapshot_with_players(1, vec![], vec![]), // slot vanishes entirely for a whole frame
            snapshot_with_players(2, vec![character(0, 2, 0, 10, 0)], vec![player(0, "Alice")]),
        ];
        let recon = reconstruct(&frames);
        assert_eq!(recon.len(), 2);
        assert_eq!(recon[0].stint, 0);
        assert_eq!(recon[0].trajectory.len(), 1);
        assert_eq!(recon[1].stint, 1);
        assert_eq!(recon[1].trajectory.len(), 1);
    }

    /// A single frame with no `Character` update for a slot that otherwise stays in `players`
    /// continuously (dead reckoning skipped a tick, or the player is temporarily dead with no
    /// live character) must NOT split anything — only absence from `players` is the identity
    /// signal, per `stint_for`'s own doc comment.
    #[test]
    fn a_gap_in_characters_alone_with_continuous_players_presence_does_not_split() {
        let frames = vec![
            snapshot_with_players(0, vec![character(0, 0, 0, 10, 0)], vec![player(0, "Alice")]),
            snapshot_with_players(1, vec![], vec![player(0, "Alice")]), // no character this tick, still "present"
            snapshot_with_players(2, vec![character(0, 2, 0, 10, 0)], vec![player(0, "Alice")]),
        ];
        let recon = reconstruct(&frames);
        assert_eq!(recon.len(), 1);
        assert_eq!(recon[0].stint, 0);
        assert_eq!(recon[0].trajectory.len(), 2);
    }

    /// Review round 1, finding F5: `resolve_name_to_stints` must agree with `reconstruct`'s own
    /// stint boundaries — the presence-gap case from
    /// `a_presence_gap_on_the_same_slot_starts_a_new_stint_even_with_the_same_name` must resolve
    /// "Alice" to both `(0, 0)` and `(0, 1)`, in that order.
    #[test]
    fn resolve_name_to_stints_finds_every_stint_that_name_was_seen_under() {
        let frames = vec![
            snapshot_with_players(0, vec![character(0, 0, 0, 10, 0)], vec![player(0, "Alice")]),
            snapshot_with_players(1, vec![], vec![]), // presence gap
            snapshot_with_players(2, vec![character(0, 2, 0, 10, 0)], vec![player(0, "Alice")]),
        ];
        assert_eq!(resolve_name_to_stints(&frames, "Alice"), vec![(0, 0), (0, 1)]);
        assert_eq!(resolve_name_to_stints(&frames, "Nobody"), Vec::<(i32, u32)>::new());
    }

    #[test]
    fn game_event_frames_are_ignored_not_erroring() {
        let frames = vec![Frame::GameEvent {
            tick_hint: 0,
            message: crate::format::RecordedGameMessage::Broadcast {
                message: "hi".to_string(),
            },
        }];
        assert!(reconstruct(&frames).is_empty());
    }
}
