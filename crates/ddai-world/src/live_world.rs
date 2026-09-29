//! [`LiveWorld`]: the client-side counterpart of `ddai-physics::World` — turns the network
//! snapshot stream (task 2.2b/2.3) into a `World<f32>` that tracks the *server's* state (task
//! spec: "builds or updates a `World<f32>` from the snapshot") and can then be stepped forward a
//! few extra ticks ([`LiveWorld::predict`]) to cover the round-trip latency between "the tick this
//! snapshot describes" and "the tick our next input will actually land on" (PredTick).

use std::sync::Arc;

use ddai_brain::{CharacterObservation, Observation};
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::{TeamsState, TuneParams};
use ddai_net::view::CharacterView;
use ddai_physics::core::{self, MAX_CLIENTS, NUM_WEAPONS, PlayerInput, WEAPON_HAMMER, WEAPON_NINJA};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use ddai_physics::world::{self, Player, TickInput, World};

use crate::reckoning::{MAX_EVOLVE_AGE_TICKS, evolve_character_core, to_net_ddnet_character};

/// `TILE_SWITCH{TIMEDOPEN,TIMEDCLOSE,OPEN,CLOSE}` (`ddai_physics::map`) re-exported here only so
/// [`apply_switch_state`] doesn't need a second `use` for constants nothing else in this module
/// touches.
use ddai_physics::map::{TILE_SWITCHCLOSE, TILE_SWITCHOPEN, TILE_SWITCHTIMEDCLOSE, TILE_SWITCHTIMEDOPEN};

/// Converts the wire `PlayerInput` (`ddai_net::generated::objects::PlayerInput`, what
/// `ClientEvent::InputSent`/`ddai_net::view::View` hand a caller) into the physics one
/// (`ddai_physics::core::PlayerInput`, what [`LiveWorld::predict`]'s `own_inputs_in_flight`
/// wants) — review round 1, finding N2. The two are structurally identical (same fields, same
/// order, same types — both are ports of the same `CNetObj_PlayerInput`), so this is a plain
/// field-for-field copy; it exists as a named, tested conversion instead of repeating that copy
/// at every call site (this crate's own `tests/e2e_local_server.rs` is exactly such a call site).
///
/// Not a `std::convert::From` impl: neither type is defined in this crate, and Rust's orphan
/// rule forbids `impl From<ddai_net::...::PlayerInput> for ddai_physics::core::PlayerInput` from
/// anywhere outside `ddai-net` or `ddai-physics` — putting it in either would give that crate an
/// unwanted new dependency on the other (`ddai-physics` is deliberately zero-dependency; see its
/// own `lib.rs`). A free function here, where both types are already in scope for `LiveWorld`'s
/// own sake, is the conversion's only legal, non-disruptive home.
pub fn player_input_from_net(net: objects::PlayerInput) -> PlayerInput {
    PlayerInput {
        direction: net.direction,
        target_x: net.target_x,
        target_y: net.target_y,
        jump: net.jump,
        fire: net.fire,
        hook: net.hook,
        player_flags: net.player_flags,
        wanted_weapon: net.wanted_weapon,
        next_weapon: net.next_weapon,
        prev_weapon: net.prev_weapon,
    }
}

/// The reverse of [`player_input_from_net`] — provided for symmetry (e.g. a caller that builds a
/// physics-side `PlayerInput` for its own decision and needs it in wire shape to actually send).
pub fn player_input_to_net(input: PlayerInput) -> objects::PlayerInput {
    objects::PlayerInput {
        direction: input.direction,
        target_x: input.target_x,
        target_y: input.target_y,
        jump: input.jump,
        fire: input.fire,
        hook: input.hook,
        player_flags: input.player_flags,
        wanted_weapon: input.wanted_weapon,
        next_weapon: input.next_weapon,
        prev_weapon: input.prev_weapon,
    }
}

/// Corrects a raw "what I queued for each tick" log into "what the server actually used for each
/// tick" (review round 1, finding F5 — confirmed live: without this, own-tee prediction dropped
/// to ~0.998 exact instead of the task spec's `>= 0.999` bar): a tick `NETMSG_INPUTTIMING`
/// reported as late (or one nothing was ever sent for at all, e.g. a gap) still had the server
/// keep using whichever input was already in effect, not the new one queued for exactly that
/// tick — `PredTick` is only ever a *request*, and the real client's own prediction has this
/// same correctness gap for exactly the same reason (it has no better information either).
///
/// `sent`: ascending-or-not `(tick, input)` pairs — e.g. built by a caller from
/// `ClientEvent::InputSent { tick, input }` (`ddai_client::ClientConfig::emit_input_sent`),
/// converted via [`player_input_from_net`]. `late_ticks`: the set of `tick`s
/// `ClientEvent::InputTiming { tick, time_left }` reported `time_left < 0` for. Returns ascending
/// `(tick, input)` pairs suitable for [`LiveWorld::predict`]'s `own_inputs_in_flight`, covering
/// every tick from `sent`'s lowest to its highest key (`None` for `sent` empty).
pub fn correct_late_inputs(
    sent: &[(i32, PlayerInput)],
    late_ticks: &std::collections::BTreeSet<i32>,
) -> Vec<(i32, PlayerInput)> {
    if sent.is_empty() {
        return Vec::new();
    }
    let sent_map: std::collections::BTreeMap<i32, PlayerInput> = sent.iter().copied().collect();
    let lo = *sent_map.keys().next().expect("just checked non-empty");
    let hi = *sent_map.keys().next_back().expect("just checked non-empty");

    let mut corrected = Vec::with_capacity((hi - lo).max(0) as usize + 1);
    let mut last: Option<PlayerInput> = None;
    for t in lo..=hi {
        let v = if late_ticks.contains(&t) || !sent_map.contains_key(&t) {
            last
        } else {
            sent_map.get(&t).copied()
        };
        if let Some(v) = v {
            corrected.push((t, v));
            last = Some(v);
        }
    }
    corrected
}

/// Review round 3, finding F11: [`correct_late_inputs`] models a late/missing tick as "the server
/// kept using whichever input was already in effect" — i.e. the *tick itself* just never got a
/// new input. That is not what DDNet 20.1's server actually does: a `NETMSG_INPUT` that arrives
/// too late for the tick it was meant for is not discarded, it is *re-targeted forward* —
/// `IntendedTick = std::max(IntendedTick, Server()->Tick() + 1)` (`server.cpp:1921`) — so it still
/// takes effect, just on a *later* tick than the one it was queued for, and the first input to
/// claim a given tick this way wins it (a second, still-later input re-targeted to the same tick
/// no longer has anywhere to go but even later still — this fn's `.or_insert` below never
/// overwrites an already-claimed effective tick, matching that "first wins" behavior).
///
/// Reviewer-provided model (round 3 report, cross-validated against a live capture — the
/// reviewer's own measured table put this at `1.0000` exact on a `BlockField` margin-5 capture,
/// where the previous [`correct_late_inputs`]-only model scored measurably lower): for a tick `T`
/// `time_left_ms` reports as late (`< 0`), its effective tick is `T + ceil(-time_left_ms / 20)`
/// (20ms being one server tick at the standard 50 Tps) — e.g. 25ms late re-targets to `T + 2`
/// (`ceil(25/20) = 2`), matching the server's own `+1`-per-tick-of-lateness re-target exactly for
/// the common "one tick late" case and extrapolating the same rule for a larger margin. Every
/// tick `time_left_ms` doesn't even mention is assumed on-time (`eff = T`), same as
/// [`correct_late_inputs`]'s own convention.
///
/// **Known limitation** (task requirement — document as an approximation): the server's own
/// re-target is driven by its *own* tick clock at the moment the packet actually arrives, not by
/// a clean multiple of 20ms of claimed lateness — for a strongly negative margin (a badly
/// mistimed/misconfigured client, or a connection so jittery `time_left_ms` swings wildly), the
/// ceiling-division estimate above can land one tick off from what the real server chose,
/// especially once several inputs' re-targeted ticks start colliding with each other. This crate's
/// own measured accuracy bar (`>= 0.999` exact) is against a real, healthy connection at the
/// default `prediction_margin_ms`, where lateness this large essentially never happens; nothing
/// here specifically detects or reports "estimate is likely off" for the strongly-negative case.
///
/// `sent`: `(tick, input)` pairs in any order (sorted internally) — see [`correct_late_inputs`]'s
/// own doc comment for where a caller gets these. `time_left_ms`: every tick a
/// `SessionEvent::InputTiming { tick, time_left }` was ever seen for (not just the late ones —
/// unlike `late_ticks` on [`correct_late_inputs`], this needs the *value*, not just lateness).
/// Returns ascending `(tick, input)` pairs, gaps already filled by holding the previous input
/// (delegates to [`correct_late_inputs`] for exactly that, with an empty late-set: every gap in
/// the *retargeted* map is an honest "nothing landed on this tick", not a second layer of
/// lateness).
pub fn retarget_late_inputs(
    sent: &[(i32, PlayerInput)],
    time_left_ms: &std::collections::BTreeMap<i32, i32>,
) -> Vec<(i32, PlayerInput)> {
    let mut sorted: Vec<(i32, PlayerInput)> = sent.to_vec();
    sorted.sort_by_key(|&(t, _)| t);

    let mut retargeted: std::collections::BTreeMap<i32, PlayerInput> = std::collections::BTreeMap::new();
    for (t, input) in sorted {
        let time_left = time_left_ms.get(&t).copied().unwrap_or(0);
        let eff = if time_left < 0 {
            // ceil(-time_left / 20), 20ms being one server tick at the standard 50 Tps.
            t + (-time_left + 19) / 20
        } else {
            t
        };
        retargeted.entry(eff).or_insert(input); // First (earliest original tick) wins `eff`.
    }
    let flattened: Vec<(i32, PlayerInput)> = retargeted.into_iter().collect();
    correct_late_inputs(&flattened, &std::collections::BTreeSet::new())
}

/// The client-side reconstruction of the server's `World<f32>`, kept up to date from the
/// snapshot stream ([`LiveWorld::on_snapshot`]) and steppable a few ticks into the future
/// ([`LiveWorld::predict`]) to cover network + processing latency.
pub struct LiveWorld {
    /// Our own client id — [`Observation::self_state`] is always this id; every other present
    /// character becomes an [`Observation::others`] entry (task spec's `target_id` selection
    /// hooks are left to the caller/brain, via [`LiveWorld::build_observation`]'s parameter).
    own_id: i32,
    map: Arc<MapData>,
    /// The reconstructed state at the *last* snapshot's own game tick — never touched by
    /// [`LiveWorld::predict`] (which works on [`LiveWorld::scratch`] instead), so it is always
    /// ready to be the base of the *next* `predict()` call, however many extra ticks the
    /// previous one stepped through.
    world: World<f32>,
    /// Reused across [`LiveWorld::predict`] calls: every call re-syncs it to [`Self::world`] via
    /// `World::restore_from` (task 1.10), which reuses this struct's own already-grown buffers
    /// instead of allocating fresh ones — see [`LiveWorld::predict`]'s own doc comment for the
    /// measured cost.
    scratch: World<f32>,
    /// Reused `Vec<TickInput>` buffer for [`LiveWorld::predict`]'s per-tick `World::step` calls.
    tick_inputs_scratch: Vec<TickInput>,
    /// Every other character's "held" input (task spec: "holding the other players' last known
    /// input, as the client does") — see [`derive_held_input`] for exactly what this
    /// approximates and why. Indexed by client id; `None` for an id with no character right now.
    held_input: [Option<PlayerInput>; MAX_CLIENTS],
    seed: u64,
}

impl LiveWorld {
    /// Builds a fresh `LiveWorld` for `map` (the currently loaded map — task spec: characters/
    /// tune zones/switches are all relative to this map's own layout). `seed`: forwarded to
    /// `World::from_map` — this crate never actually *uses* the world's PRNG (Stage A's PRNG-
    /// consuming logic, e.g. crazy-shotgun/tele-hook randomness, is a minor, already-bounded
    /// source of prediction error the accuracy measurement mode reports on, not something this
    /// constructor can eliminate: the real server's own PRNG state is never observable over the
    /// network at all), so any fixed value is as good as any other; callers that already know the
    /// server's own scenario seed may pass it for a marginally closer match on maps whose physics
    /// actually consult the PRNG (tele-hook, crazy shotguns).
    pub fn new(map: Arc<MapData>, own_id: i32, seed: u64) -> Self {
        let mut world = World::from_map(&map, seed);
        // Applies the map's own baked-in "Settings" strings (`sv_hit`, `tune`/`tune_zone`, ...),
        // exactly like `CGameContext::OnInit()` — see `World::init`'s doc comment. No pre-init
        // cfg pass: this crate never has a scenario/`--cfg` file, only the live map and network
        // messages (`Sv_TuneParams`) to go on.
        let _ = world.init(std::iter::empty::<&str>());
        let scratch = world.clone();
        LiveWorld {
            own_id,
            map,
            world,
            scratch,
            tick_inputs_scratch: Vec::with_capacity(8),
            held_input: [None; MAX_CLIENTS],
            seed,
        }
    }

    /// Our own client id, as given to [`LiveWorld::new`].
    pub fn own_id(&self) -> i32 {
        self.own_id
    }

    /// The tick [`LiveWorld::on_snapshot`] last reconstructed exact state for — `predict()`'s
    /// implicit starting point.
    pub fn base_tick(&self) -> i32 {
        self.world.tick
    }

    /// A read-only look at the reconstructed base state (the last snapshot's own tick, no
    /// prediction) — mainly for tests/diagnostics; live callers normally want
    /// [`LiveWorld::predict`]'s result instead.
    pub fn base_world(&self) -> &World<f32> {
        &self.world
    }

    /// Builds/updates the reconstructed world from one snapshot (task spec, acceptance criterion
    /// 1): reckoning-core extrapolation per character (see [`crate::reckoning`]), `DDNetCharacter`
    /// merge (flags/freeze/jumps/telegun/weapons), `StrongWeakId` tick order (D-022), switch
    /// states, and tuning (map zones, already loaded at [`LiveWorld::new`] time, plus the live
    /// `Sv_TuneParams` message layered on top of zone 0 — see [`apply_tune_params`]'s doc comment
    /// for that layering's one known limitation).
    ///
    /// `tick`: the snapshot's own game tick (`SessionEvent::Snapshot { tick }` /
    /// `ddai_net::view::View::game_info()`'s tick — whichever the caller already has to hand;
    /// this fn never reads it from `characters` itself, since a snapshot with *no* characters at
    /// all, e.g. before anyone has spawned, must still advance the base tick).
    /// `characters`: `ddai_net::view::View::characters()`'s result (or an equivalent slice built
    /// for a test — see `crate::reckoning`/this crate's tests for how little that takes).
    /// `switch_states`: `ddai_net::view::View::switch_states()`'s result.
    /// `teams`: `ddai_client::session::Session::teams_state()`'s result, if the caller has kept
    /// one — DDRace team assignment has "carry forward the last known value" semantics (see
    /// [`TeamsState`]'s own doc comment), so a caller that never received one at all may simply
    /// pass `None` every time (every character then stays on the default team, `0`).
    /// `own_input_at_tick`: review round 3, finding F10 — the input the server actually applied to
    /// *our own* tee at this exact snapshot's tick (i.e. this connection's own logged
    /// `SessionEvent::InputSent`/`InputTiming` history for `tick`, corrected for late delivery —
    /// see [`retarget_late_inputs`]/[`correct_late_inputs`] — and converted via
    /// [`player_input_from_net`]), if known. `None` when it isn't (a dropped `InputSent` — F12's
    /// own droppability — or the caller's own bookkeeping simply has a gap for this exact tick)
    /// reuses *our own last known seed* instead of a fresh guess (review round 3, finding F15 —
    /// falling all the way back to [`derive_held_input`]'s neutral `fire: 0` guess here would
    /// reintroduce F10's own phantom-fire bug for every dropped/missing tick); only this
    /// character's very first-ever sighting (no previous seed to reuse either) falls all the way
    /// back to that guess — see this method's own doc comment on [`upsert_character`]'s seeding
    /// for why any of this matters only for our own tee, never for anyone else's.
    pub fn on_snapshot(
        &mut self,
        tick: i32,
        characters: &[CharacterView],
        tuning: TuneParams,
        switch_states: &[(i32, objects::SwitchState)],
        teams: Option<&TeamsState>,
        own_input_at_tick: Option<PlayerInput>,
    ) {
        self.world.tick = tick;

        if let Some(teams) = teams {
            for (id, &team) in teams.teams[..teams.received].iter().enumerate() {
                self.world.teams_core.set_team(id as i32, team);
            }
        }

        // Review round 1, finding F3: apply *before* the characters pass, so the tune-zone lookup
        // below (`apply_tune_params`) reads each present switch's freshly-cleared/updated state,
        // not last snapshot's — switches themselves have no tune-zone dependency either way.
        let touched_a_switch = !switch_states.is_empty();
        for &(team, sw) in switch_states {
            apply_switch_state(&mut self.world, team, &sw);
        }
        if touched_a_switch {
            rebuild_active_timed_switchers(&mut self.world);
        }

        let collision = Arc::clone(&self.world.collision);
        let mut present = [false; MAX_CLIENTS];
        let mut own_tune_zone_override = -1;
        for cv in characters {
            let id = cv.id;
            if !(0..MAX_CLIENTS as i32).contains(&id) {
                continue; // Hostile/malformed id — never trust the network past a bounds check.
            }
            present[id as usize] = true;
            let core = evolve_character_core(&cv.character, tick, &collision);
            // Review round 3, finding F10: the *own* tee is seeded from the real applied input
            // when the caller can supply one, never from `derive_held_input`'s always-`fire: 0`
            // guess (see `upsert_character`'s own doc comment on why that guess, while a fine
            // approximation for every other character, is actively wrong for our own tee).
            //
            // Review round 3, finding F15: `own_input_at_tick` being `None` (a dropped
            // `SessionEvent::InputSent` — F12's own droppability — or simply a snapshot that
            // arrived before this connection had sent anything at all yet) must **not** fall
            // straight through to `derive_held_input`'s fresh guess (`fire: 0`) the way it used
            // to — that reintroduces F10's exact phantom-fire bug (a real, already-settled
            // nonzero fire counter momentarily replaced by `0` again) for every tick a caller
            // simply couldn't supply a value for, which is precisely the case F12 says must be
            // tolerated. The correct fallback is *our own last known seed*
            // (`self.held_input[self.own_id]`, itself either a previous `own_input_at_tick` or,
            // transitively, this same fallback) — only when there has never been *any* known
            // value at all (the character's very first sighting) does this fall through to
            // `derive_held_input`'s guess, same as it always has for that one genuinely
            // information-free case.
            let seed_input = if id == self.own_id {
                own_input_at_tick
                    .or(self.held_input[id as usize])
                    .unwrap_or_else(|| derive_held_input(&core, self.held_input[id as usize]))
            } else {
                derive_held_input(&core, self.held_input[id as usize])
            };
            self.upsert_character(id, core, &cv.character, cv.ddnet.as_ref(), tick, seed_input);
            self.held_input[id as usize] = Some(seed_input);
            if id == self.own_id {
                own_tune_zone_override = cv.ddnet.map(|d| d.tune_zone_override).unwrap_or(-1);
            }
        }
        for (id, &is_present) in present.iter().enumerate() {
            if !is_present && self.world.characters[id].is_some() {
                self.remove_character(id as i32);
            }
        }

        self.rebuild_entity_order();

        // Review round 1, finding F4: apply the live `Sv_TuneParams` message to *our own tee's
        // current* tune zone (`tune_zone_override` if the server set one, else the map-position-
        // derived zone `handle_tune_layer` — called for every present character above, including
        // our own — already recomputed), not unconditionally to zone 0
        // (`gameclient.cpp:3440-3459`'s own `TuneZone` computation). This can only be done *after*
        // the characters pass above, so our own zone reflects *this* snapshot's position, not the
        // previous one's.
        let own_zone = if own_tune_zone_override >= 0 {
            own_tune_zone_override
        } else {
            self.own_character().map(|c| c.tune_zone).unwrap_or(0)
        };
        apply_tune_params(self.world.tuning.zone_mut(own_zone), &tuning);
    }

    /// [`Self::own_id`]'s current [`world::Character`], bounds-checked — review round 1, finding
    /// N3: `own_id` is caller-supplied at [`LiveWorld::new`] and never itself validated (an
    /// out-of-`0..MAX_CLIENTS` value, e.g. `-1`, is a legitimate "no local player yet" sentinel a
    /// caller might reasonably pass before it has one), so nothing in this crate may index
    /// `world.characters`/`world.cores`/`self.held_input` with it directly without going through
    /// this (or an equivalent bounds check) first.
    fn own_character(&self) -> Option<&world::Character<f32>> {
        if !(0..MAX_CLIENTS as i32).contains(&self.own_id) {
            return None;
        }
        self.world.characters[self.own_id as usize].as_ref()
    }

    fn upsert_character(
        &mut self,
        id: i32,
        mut core: core::CharacterCore<f32>,
        net: &objects::Character,
        ddnet: Option<&objects::DDNetCharacter>,
        tick: i32,
        seed_input: PlayerInput,
    ) {
        // `evolve_character_core` leaves `core.id == -1` (matching the real client's `Evolve`
        // exactly — see that function's doc comment); the *world*, unlike that isolated one-slot
        // scratch space, does need the real id (every sibling-aware lookup — hook targets, the
        // hammer/collision loops — reads it), matching `WorldCore::insert`'s own invariant.
        core.id = id;
        if let Some(d) = ddnet {
            core.read_ddnet(&to_net_ddnet_character(d));
            // Review round 1, finding F6: `CharacterCore::read_ddnet` already sets `core.solo`
            // from the wire flag, but the real `SetSolo` (`character.cpp:183-188`) *also* pokes
            // `TeamsCore()->SetSolo` — a second, independent piece of state `character_can_collide`/
            // hook-target selection reads (`core.rs`'s own `TeamsCore::get_solo`), which
            // `read_ddnet` alone (a `CCharacterCore`-only method) has no way to touch.
            self.world.teams_core.set_solo(id, core.solo);
        }
        // Review round 3, finding F1 (reopened): must run *after* `read_ddnet` above, which is
        // where `core.weapons[*].got` actually comes from — see `reconstruct_weapon_ammo`'s own
        // doc comment for why this can't live in `evolve_character_core` (called before `got` is
        // known at all) the way round 1's fix originally tried to.
        reconstruct_weapon_ammo(&mut core, net);

        let world = &mut self.world;
        let mut character = world.characters[id as usize].unwrap_or_default();
        // A character with no `CNetObj_Character` in the snapshot isn't reconstructed at all
        // (handled by `on_snapshot`'s `remove_character` pass instead) — so being here at all
        // means the server considers `id` alive right now. Without this, `character.alive` would
        // stay at `Character::default()`'s `false` while `Player::has_character` (set below) is
        // `true` — an inconsistency `world::player_tick` itself resolves *for* us every predicted
        // tick (`has_character && !alive` clears `has_character`), which then makes the *next*
        // predicted tick's `try_respawn` call `spawn_character` on an id [`World::cores`] never
        // actually lost (found live, against the local server: a real character older than
        // `sv_respawn`-equivalent's ~150-tick delay panicked `WorldCore::insert` with "duplicate
        // client id" the moment `LiveWorld::predict` stepped that far — see this crate's `BUILD
        // REPORT`).
        character.alive = true;
        // `HandleTiles`' anti-skip loop (`ddrace_post_core_tick`, only reachable once `alive` is
        // `true` — see the comment above) walks every tile between `prev_pos` and the current
        // position looking for freeze/speedup/tele/kill tiles it must not let a fast-moving tee
        // skip past. `prev_pos` is meant to be "one server tick ago"; a one-shot network
        // reconstruction has no way to know that exactly, but leaving it at whatever stale value
        // (or `Character::default()`'s `(0, 0)`, the map's own top-left corner) it last held is
        // far worse: found live, against the local server, once `alive` started being set
        // correctly above — a giant bogus "skip" line from `(0, 0)` all the way to the tee's real
        // position, crossing unrelated tiles across the whole map and corrupting the very next
        // predicted tick (see this crate's `BUILD REPORT`). Snapping it to the just-reconstructed
        // current position instead is the conservative choice: "assume no tile was skipped since
        // last tick" is the same assumption `evolve_character_core`'s own idealized
        // (no-interaction) continuation already rests its correctness on.
        character.prev_pos = core.pos;
        if let Some(d) = ddnet {
            character.strong_weak_id = d.strong_weak_id;
            character.tele_checkpoint = d.tele_checkpoint;
            // `m_FreezeEnd` is an absolute tick (`-1` = deep freeze, modeled on `core.deep_frozen`
            // instead — see `CharacterCore::read_ddnet`); `Character::freeze_time` is DDRace's own
            // *remaining-ticks* countdown (`ddrace_tick` decrements it every tick and zeroes the
            // input while it's positive — see `world::ddrace_tick`), so it has to be derived here,
            // not copied.
            character.freeze_time = if d.freeze_end <= 0 {
                0
            } else {
                (d.freeze_end - tick).max(0)
            };
        } else {
            character.freeze_time = 0;
        }

        // Review round 1, finding F2 (confirmed live): a fresh reconstruction never gets to
        // "already >1 inputs old" the way a real, continuously-ticking character always is, so
        // `world::on_direct_input`'s own `num_inputs > 1` gate (`world.rs:3393`) would otherwise
        // suppress `handle_weapon_switch`/`fire_weapon` for the *entire first predicted tick*
        // every single time — never `> 1` on the very first `on_direct_input` call from `1`.
        // Seeding it to `2` (matching "this character has clearly existed for a while") makes the
        // gate a no-op immediately, exactly like a real, already-running character's own counter
        // (which only ever increases, so once above `1` it never revisits the gate at all).
        character.num_inputs = character.num_inputs.max(2);
        // `latest_input`/`latest_prev_input` back `handle_weapon_switch`/`fire_weapon`'s press-edge
        // detection (`count_input_presses` on their `next_weapon`/`prev_weapon`/`fire` counters) on
        // this character's very first predicted tick, before `on_direct_input` has run even once
        // to refresh them from a real supplied input.
        //
        // Both fields are seeded to the *same* value (`seed_input`, this call's caller-supplied
        // parameter) deliberately: `count_input_presses(x, x)` is always `0`, so whatever
        // `seed_input` actually is, seeding both identically guarantees no phantom press/switch is
        // ever manufactured purely from the seed itself — only a genuinely *different* value
        // supplied on the first predicted tick can register as a press, exactly like a real,
        // already-running character's own counters (which only ever change when a real new input
        // actually arrives).
        //
        // Review round 3, finding F10 (blocker, confirmed live: `fire_first_tick`/`PHANTOM=1` —
        // holding fire constant at a nonzero counter value for hundreds of ticks, no new press at
        // all — still predicted a phantom weapon fire on the very first predicted tick): for every
        // *other* character, `seed_input` is [`derive_held_input`]'s neutral guess, whose `fire: 0`
        // is a fine approximation (this crate never predicts anyone else pulling a fresh trigger
        // anyway — see that function's own doc comment). For *our own* tee, `fire` is not a
        // level, it is a press *counter* that only ever increases (`INPUT_STATE_MASK`-wrapped) and
        // never resets to `0` just because the button is still being held — so seeding it to `0`
        // while its true steady-state value is anything else (e.g. `2`, held) makes
        // `count_input_presses(0, 2)` see a spurious press the instant the first real (still-held,
        // no new press) input tick is applied. `on_snapshot`'s caller now threads in the actual
        // applied input for our own tee specifically (`own_input_at_tick`) so `seed_input` here is
        // the *real* value whenever it's known, not a synthesized one — see this method's own doc
        // comment.
        character.latest_input = seed_input;
        character.latest_prev_input = seed_input;
        character.input = seed_input;
        character.saved_input = seed_input;

        // Review round 1, finding F2: `reload_timer` reconstruction — own character only,
        // mirroring the real client's own `IsLocal`-gated derivation exactly
        // (`prediction/entities/character.cpp:1572-1579`): skipped for `WEAPON_HAMMER` (whose
        // real reload can be either of two different delays depending on whether the last swing
        // *hit* someone, `world::fire_hammer`'s own special case — not reconstructible from the
        // network alone) and while ninja is held (a temporary weapon overlay). Without this, our
        // own reconstructed `reload_timer` starts at `0` ("ready to fire *right now*") regardless
        // of how recently the real character actually fired, letting a predicted rapid-fire
        // weapon re-fire far faster than the real server would ever allow.
        character.attack_tick = net.attack_tick;
        if id == self.own_id
            && core.active_weapon != -1
            && core.active_weapon != WEAPON_HAMMER
            && !core.weapons[WEAPON_NINJA as usize].got
        {
            let fire_delay_ticks =
                (core.tuning.get_weapon_fire_delay(core.active_weapon) * core::SERVER_TICK_SPEED as f32) as i32;
            character.reload_timer = (net.attack_tick + fire_delay_ticks - tick).max(0);
        } else if id == self.own_id {
            character.reload_timer = 0;
        }

        // Recomputed eagerly (not just left for the next `predict()` tick) so a caller that
        // builds an `Observation` right after `on_snapshot`, before ever calling `predict`, still
        // sees the right zone/tuning — `handle_tune_layer` is otherwise called every tick anyway
        // (`world::ddrace_tick`), so this duplicates no real work, it just also runs once now.
        world::handle_tune_layer(&mut character, &mut core, world.collision.as_ref(), &world.tuning);

        if world.cores.get(id as u8).is_some() {
            *world.cores.get_mut(id as u8).expect("just checked Some") = core;
        } else {
            world.cores.insert(id as u8, core);
        }
        world.characters[id as usize] = Some(character);
        if world.players[id as usize].is_none() {
            let mut player = Player::new(tick);
            player.has_character = true;
            player.team = world::TEAM_GAME;
            world.players[id as usize] = Some(player);
        }
    }

    fn remove_character(&mut self, id: i32) {
        self.world.cores.remove(id as u8);
        self.world.characters[id as usize] = None;
        self.world.players[id as usize] = None;
        self.held_input[id as usize] = None;
        // Review round 1, finding F6: mirrors `upsert_character`'s own `teams_core.set_solo` —
        // a departed character must not leave a stale `solo` entry another (later, id-reused)
        // character could inherit.
        self.world.teams_core.set_solo(id, false);
    }

    /// D-022: orders [`World::entity_order`] by ascending `StrongWeakId`, matching the server's
    /// own "new characters tick first" order (`World::entity_order`'s doc comment: index `0` =
    /// processed first) — this is what decides who wins a strong/weak hook contest during
    /// [`LiveWorld::predict`]'s stepping, so getting the order wrong silently flips every such
    /// contest's outcome.
    fn rebuild_entity_order(&mut self) {
        let mut ids: Vec<(i32, u8)> = self
            .world
            .characters
            .iter()
            .enumerate()
            .filter_map(|(id, c)| c.as_ref().map(|c| (c.strong_weak_id, id as u8)))
            .collect();
        ids.sort_by_key(|&(strong_weak_id, _)| strong_weak_id);
        self.world.entity_order.clear();
        self.world.entity_order.extend(ids.into_iter().map(|(_, id)| id));
    }

    /// Steps a scratch copy of the reconstructed base world ([`LiveWorld::base_tick`]) forward to
    /// `to_tick` (task spec: "PredTick"), applying our own unacknowledged inputs by tick
    /// (`own_inputs_in_flight`, ascending `(tick, input)` pairs — a tick with no entry repeats the
    /// most recent earlier one, falling back to [`derive_held_input`]'s own-character guess if
    /// `own_inputs_in_flight` is empty or starts after `base_tick() + 1`) and holding every other
    /// present character's last known input (see [`derive_held_input`]). Returns the predicted
    /// world — the caller's `Observation` should be built from *this*, not [`LiveWorld::base_world`].
    ///
    /// Cost (task spec, D-041; measured in `tests/predict_cost.rs`): task 1.10 (merged after this
    /// crate's first round) added `World::restore_from` — a save/restore that reuses every
    /// buffer instead of `Clone`'s fresh allocation-per-`Vec`-field (task 1.10's own `BUILD
    /// REPORT`: ~0.45-4 µs vs 17-24 µs for `.clone()`, 0 allocations once every buffer has grown
    /// to fit) — `predict()` uses that instead of `clone_from` now (review round 1's API note),
    /// so the whole call is zero-allocation in steady state, not just its step loop.
    ///
    /// Review round 1, finding F7: `to_tick` is capped at `base_tick() +
    /// crate::reckoning::MAX_EVOLVE_AGE_TICKS` (3 seconds) — the same defensive cap
    /// `evolve_character_core` applies to reckoning age, for the same reason: a caller (or a bug
    /// upstream of one) asking to predict arbitrarily far ahead must not turn this into an
    /// effectively-unbounded `World::step` loop. D-041's own live decision cadence (a few ticks
    /// ahead at most) never comes close to this cap.
    ///
    /// `to_tick < base_tick()` is a no-op (returns the base world unchanged, `to_tick > base_tick`
    /// is the only case this function ever actually steps).
    pub fn predict(&mut self, to_tick: i32, own_inputs_in_flight: &[(i32, PlayerInput)]) -> &World<f32> {
        self.scratch.restore_from(&self.world);
        let to_tick = to_tick.min(self.scratch.tick.saturating_add(MAX_EVOLVE_AGE_TICKS));
        if to_tick <= self.scratch.tick {
            return &self.scratch;
        }

        let own_id = self.own_id;
        let mut latest_own_input = self
            .held_input
            .get(own_id as usize)
            .copied()
            .flatten()
            .unwrap_or_default();

        while self.scratch.tick < to_tick {
            let next_tick = self.scratch.tick + 1;
            if let Some(&(_, input)) = own_inputs_in_flight.iter().find(|&&(t, _)| t == next_tick) {
                latest_own_input = input;
            }

            self.tick_inputs_scratch.clear();
            for id in 0..MAX_CLIENTS {
                if self.scratch.characters[id].is_none() {
                    continue;
                }
                let input = if id as i32 == own_id {
                    latest_own_input
                } else {
                    self.held_input[id].unwrap_or_default()
                };
                self.tick_inputs_scratch.push(TickInput {
                    id: id as u8,
                    input,
                    kill: false,
                });
            }
            self.scratch.step(&self.tick_inputs_scratch);
        }
        &self.scratch
    }

    /// Builds a [`ddai_brain::Observation`] from `world` (normally [`LiveWorld::predict`]'s
    /// result — see the task spec's "the result is the observation for the brain"). `target_id`
    /// is passed straight through to [`Observation::target_id`] (task spec: "`target_id` selection
    /// hooks" — target selection itself is outside this crate's scope, per `ddai-brain`'s own
    /// doc comment on that field).
    pub fn build_observation(&self, world: &World<f32>, target_id: Option<i32>) -> Observation {
        let self_state =
            character_observation(world, self.own_id).unwrap_or_else(|| CharacterObservation::at_rest(self.own_id));
        let mut others = Vec::new();
        for id in 0..MAX_CLIENTS as i32 {
            if id == self.own_id {
                continue;
            }
            if let Some(obs) = character_observation(world, id) {
                others.push(obs);
            }
        }
        // Review round 1, finding N3: bounds-checked, not a direct `world.characters[self.own_id
        // as usize]` index — `self.own_id` is caller-supplied and never itself validated (see
        // `Self::own_character`'s doc comment), so an out-of-range value (e.g. `-1`, "no local
        // player yet") must fall back to zone `0`, not panic.
        let tune_zone = if (0..MAX_CLIENTS as i32).contains(&self.own_id) {
            world.characters[self.own_id as usize].map(|c| c.tune_zone).unwrap_or(0)
        } else {
            0
        };
        Observation {
            map: Arc::clone(&self.map),
            tick: world.tick,
            self_state,
            others,
            target_id,
            tuning: *world.tuning.zone(tune_zone),
        }
    }

    /// The map this `LiveWorld` was built from ([`ddai_brain::Observation::map`] shares this same
    /// allocation — see [`LiveWorld::build_observation`]).
    pub fn map(&self) -> &Arc<MapData> {
        &self.map
    }

    /// The seed passed to [`LiveWorld::new`] (diagnostics only).
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Review round 3, finding F1 (reopened MAJOR): reconstructs every weapon slot's `ammo` for one
/// character, called from [`LiveWorld::upsert_character`] *after* `core.read_ddnet` has already
/// set `core.weapons[*].got` from the wire flags (this function reads that, so it cannot run any
/// earlier — see [`crate::reckoning::evolve_character_core`]'s own doc comment for why the old
/// round-1 fix living there was wrong).
///
/// The server's `AmmoCount` (`net.ammo_count`) is *not* a plain ammo reading in general
/// (`character.cpp:1091,1140-1148`): it defaults to `0` and is only ever replaced with the tee's
/// real current-weapon ammo for the *snapping client's own* character, and only while that
/// character isn't frozen (`m_FreezeTime == 0`) — for every other player, and for our own tee
/// while frozen, the wire value is unconditionally `0`, indistinguishable on its own from a
/// genuine "no ammo left". The client's own reconstruction never trusts it blindly either
/// (`prediction/entities/character.cpp:1538-1548`): DDRace has no ammo-limited weapons at all, so
/// `m_WorldConfig.m_InfiniteAmmo` is `true` on every server this bot targets (a DDRace/block
/// server never gives a client real, positive `AmmoCount` for a non-ninjajetpack weapon — the one
/// signal that would ever clear it, `gameclient.cpp:3538-3539` — since nothing on such a server
/// ever finitely limits ammo) — reflected here directly, per review round 3: every weapon this tee
/// has ever picked up (`got`) reads back as infinite (`-1`) *unless* this exact wire value proves
/// otherwise for the one currently-*active* weapon (`net.ammo_count > 0`, a real positive reading
/// — the one case this crate's DDRace-only scope can still honor exactly). Hammer is always `-1`
/// regardless of `got`/active (never reload-gated by ammo — `pChar->m_Weapon == WEAPON_HAMMER` in
/// the same client line); ninja is left untouched (Stage B — its wire "ammo" is actually a
/// duration counter, not modeled by this crate at all, see `reckoning`'s own doc comment).
fn reconstruct_weapon_ammo(core: &mut core::CharacterCore<f32>, net: &objects::Character) {
    for i in 0..NUM_WEAPONS as i32 {
        if i == WEAPON_NINJA {
            continue;
        }
        let has_it = core.weapons[i as usize].got || i == net.weapon;
        if !has_it {
            continue; // Never had it, and it isn't even the (possibly not-yet-`got`) active one.
        }
        core.weapons[i as usize].ammo = if i == WEAPON_HAMMER {
            -1
        } else if i == net.weapon && net.ammo_count > 0 {
            net.ammo_count
        } else {
            -1
        };
    }
}

/// Derives the [`PlayerInput`] LiveWorld holds constant for a non-local character across
/// [`LiveWorld::predict`]'s forward ticks (task spec: "holding the other players' last known
/// input, as the client does" — see `gameclient.cpp:2612-2726`'s `OnPredict`, which simply never
/// calls `OnDirectInput`/`OnPredictedInput` for anyone but the local/dummy character, leaving
/// their existing `m_Input` untouched for the whole predicted window).
///
/// DDNet's network protocol never sends another player's raw input booleans at all (only
/// `direction`, part of the core) — so this is necessarily an *approximation* of "their last
/// known input", built from what the reconstructed core already tells us:
/// - `direction`: exact (`CharacterCore::direction`, straight off the wire, untouched by
///   `evolve_character_core`'s `use_input=false` extrapolation).
/// - `hook`: held **pressed** iff `hook_state != HOOK_IDLE` (continues an in-progress hook,
///   never launches a fresh one — seeing `hook_state == HOOK_FLYING/GRABBED/...` already proves
///   they were holding hook a moment ago) and **released** otherwise, i.e. never manufactures a
///   brand-new hook launch for someone we last saw not hooking.
/// - `target_x`/`target_y`: only read by `tick()` when hook input transitions `0 -> 1` while
///   `hook_state == HOOK_IDLE` (a fresh launch) — which the rule above deliberately never
///   triggers for an *already*-idle-hook character, so this only matters while continuing an
///   existing hook, where it's set from that hook's own already-reconstructed direction
///   (`hook_dir`) so a continued hook never appears to bend.
/// - `jump`: always **not pressed**. Holding `jump = 1` across many ticks would cause a fresh
///   jump on every single ground contact `tick()` sees (jump is edge-triggered, not "as long as
///   held" — see `core::tick`'s `me.jumped & 1 == 0` guard), which is almost certainly a worse
///   guess than assuming no new jump input arrives during the (short, a handful of ticks)
///   prediction window; already-airborne jump state is unaffected either way (the reconstructed
///   core's own `jumped`/`vel` already capture it).
/// - `fire`/weapon-switch fields: always neutral (no new weapon fire/switch modeled for others).
///
/// `previous`: the same character's held input from the *last* `on_snapshot` call (`None` for a
/// character just seen for the first time), reused only for `target_x`/`target_y` when not
/// currently hooking (an arbitrary-but-stable aim guess is preferable to snapping to `(0, -1)`
/// every snapshot).
fn derive_held_input(core: &core::CharacterCore<f32>, previous: Option<PlayerInput>) -> PlayerInput {
    let hooking = core.hook_state != core::HOOK_IDLE;
    let (target_x, target_y) = if hooking {
        (
            (core.hook_dir.x * 1000.0).round() as i32,
            (core.hook_dir.y * 1000.0).round() as i32,
        )
    } else if let Some(p) = previous {
        (p.target_x, p.target_y)
    } else {
        (0, -1)
    };
    PlayerInput {
        direction: core.direction,
        target_x,
        target_y,
        jump: 0,
        fire: 0,
        hook: i32::from(hooking),
        player_flags: playerflagflag::PLAYING,
        wanted_weapon: 0,
        next_weapon: 0,
        prev_weapon: 0,
    }
}

/// `Sv_TuneParams`'s wire order (`ddai_net::tuning::TuneParams`) is `tuning.h`'s declaration
/// order — the exact same order `ddai_physics::tuning::TuningParams` stores its 47 `TuneParam`s
/// at (both are direct ports of the same header; cross-checked field-by-field against
/// `ddai_physics::tuning`'s own `NAMES`/`idx` table) — so this is a plain positional copy via
/// [`TuningParams::set_raw`] (already-quantized `×100` ints both sides, no float round-trip).
///
/// **Known limitation:** `Sv_TuneParams` carries only the *receiving client's own current* zone's
/// tuning (`CGameContext::SendTuningParams(ClientId, Zone)`, sent whenever that client's own
/// `m_TuneZone` changes — `character.cpp:2126-2139`), not which zone number it is. This applies it
/// to zone `0` unconditionally; while our own tee is standing in a *non-zero* tune zone, an
/// incoming live `Sv_TuneParams` would misapply to zone 0 instead of that zone (the map's own
/// baked-in zone table, loaded once at [`LiveWorld::new`] via `World::init`'s "Settings" pass,
/// already covers the static, no-live-rcon-change case correctly either way — see this crate's
/// `BUILD REPORT`).
fn apply_tune_params(zone: &mut TuningParams, net: &TuneParams) {
    let raw: [i32; 47] = [
        net.ground_control_speed,
        net.ground_control_accel,
        net.ground_friction,
        net.ground_jump_impulse,
        net.air_jump_impulse,
        net.air_control_speed,
        net.air_control_accel,
        net.air_friction,
        net.hook_length,
        net.hook_fire_speed,
        net.hook_drag_accel,
        net.hook_drag_speed,
        net.gravity,
        net.velramp_start,
        net.velramp_range,
        net.velramp_curvature,
        net.gun_curvature,
        net.gun_speed,
        net.gun_lifetime,
        net.shotgun_curvature,
        net.shotgun_speed,
        net.shotgun_speeddiff,
        net.shotgun_lifetime,
        net.grenade_curvature,
        net.grenade_speed,
        net.grenade_lifetime,
        net.laser_reach,
        net.laser_bounce_delay,
        net.laser_bounce_num,
        net.laser_bounce_cost,
        net.laser_damage,
        net.player_collision,
        net.player_hooking,
        net.jetpack_strength,
        net.shotgun_strength,
        net.explosion_strength,
        net.hammer_strength,
        net.hook_duration,
        net.hammer_fire_delay,
        net.gun_fire_delay,
        net.shotgun_fire_delay,
        net.grenade_fire_delay,
        net.laser_fire_delay,
        net.ninja_fire_delay,
        net.hammer_hit_fire_delay,
        net.ground_elasticity_x,
        net.ground_elasticity_y,
    ];
    for (i, v) in raw.into_iter().enumerate() {
        zone.set_raw(i, v);
    }
}

/// Merges one `CNetObj_SwitchState` (task spec: "switch states from `SwitchState`") into
/// `world.cores.switchers`. `team`: `View::switch_states()`'s item id, i.e. `SentTeam`
/// (`gamecontext.cpp:504-557`) — the DDRace team this particular state describes (see that
/// method's own doc comment: normally just our own team).
///
/// The steady on/off status (`sw.status`, a 256-bit map over every switch number) is exact — the
/// server packs every switch's live status into it every time, no approximation.
///
/// Review round 1, finding F3 (confirmed live: a switch closed for over 3 seconds still showed a
/// long-expired `end_tick`/`TILE_SWITCHTIMEDOPEN`, flipping it on the very first predicted tick):
/// the server only ever reports up to 4 switches whose real end tick is *currently* less than 3
/// seconds away (`gamecontext.cpp:531-545`'s own `EndTick < Server()->Tick() + 3 * TickSpeed`
/// filter) — every switch this message does *not* list that way genuinely has no known-imminent
/// flip right now, whether because it never had one, it already happened, or it's still more than
/// 3 seconds off. The real client recomputes `kind` from *current* `status` and whatever
/// `end_tick` it holds on *every* message (`gameclient.cpp:2100-2107`) but — unlike this crate —
/// never predicts a switch flipping at all, so a stale nonzero `end_tick` lingering forever
/// between updates is harmless for it and would not be for `LiveWorld::predict`'s own
/// `tick_switch_expiry` call (`ddai_physics::World::step`'s normal per-tick pipeline). So: for
/// every switch number this team's status covers, `end_tick`/`kind` are set from *this message
/// alone* — cleared to the plain (non-timed) `OPEN`/`CLOSE` kind unless this exact message
/// reports a live (`> tick`) end tick for it.
fn apply_switch_state(world: &mut World<f32>, team: i32, sw: &objects::SwitchState) {
    if !(0..core::NUM_DDRACE_TEAMS).contains(&team) {
        return;
    }
    let team = team as usize;
    let tick = world.tick;
    let highest = (sw.highest_switch_number.max(0) as usize).min(world.cores.switchers.len().saturating_sub(1));

    // Up to 4 (number, end_tick) pairs this exact message reports as "live" (a real, current,
    // not-yet-passed end tick) — built once, not re-scanned per switch number below.
    let mut live_end_tick = [None::<i32>; 4];
    for (i, slot) in live_end_tick.iter_mut().enumerate() {
        let end_tick = sw.end_ticks[i];
        let number = sw.switch_numbers[i];
        if end_tick > tick && (0..=highest as i32).contains(&number) {
            *slot = Some(end_tick);
        }
    }

    for number in 0..=highest {
        if number >= world.cores.switchers.len() {
            break;
        }
        let word = sw.status[number / 32];
        let status = (word >> (number % 32)) & 1 != 0;
        world.cores.switchers[number].status[team] = status;

        let reported_live = (0..sw.switch_numbers.len())
            .find(|&i| live_end_tick[i].is_some() && sw.switch_numbers[i] as usize == number)
            .and_then(|i| live_end_tick[i]);
        match reported_live {
            Some(end_tick) => {
                world.cores.switchers[number].end_tick[team] = end_tick;
                world.cores.switchers[number].kind[team] = if status {
                    TILE_SWITCHTIMEDOPEN as i32
                } else {
                    TILE_SWITCHTIMEDCLOSE as i32
                };
            }
            None => {
                world.cores.switchers[number].end_tick[team] = 0;
                world.cores.switchers[number].kind[team] = if status {
                    TILE_SWITCHOPEN as i32
                } else {
                    TILE_SWITCHCLOSE as i32
                };
            }
        }
    }
}

/// Review round 1, finding F3's follow-up: [`World::active_timed_switchers`] is the perf side
/// list `switch::tick_switch_expiry` alone consults — [`apply_switch_state`] can both *add* a
/// newly-timed switcher and *clear* one back to steady state, so it must be rebuilt (not just
/// appended to) whenever any switch state was just applied. `O(switchers × NUM_DDRACE_TEAMS)`,
/// run once per `on_snapshot` call that actually touched a switch (not per tick) — see that
/// call site.
fn rebuild_active_timed_switchers(world: &mut World<f32>) {
    world.active_timed_switchers.clear();
    for (number, switcher) in world.cores.switchers.iter().enumerate() {
        let timed = switcher
            .kind
            .iter()
            .any(|&k| k == TILE_SWITCHTIMEDOPEN as i32 || k == TILE_SWITCHTIMEDCLOSE as i32);
        if timed {
            world.active_timed_switchers.push(number as u8);
        }
    }
}

/// The [`CharacterObservation`] of client `id` in `world`, `None` when there is no such character.
/// Public so offline consumers (the human-play dataset pipeline, task 8.4c) can read many
/// characters out of one reconstructed world without building a `LiveWorld` per viewpoint.
pub fn character_observation(world: &World<f32>, id: i32) -> Option<CharacterObservation> {
    if !(0..MAX_CLIENTS as i32).contains(&id) {
        return None;
    }
    let core = world.cores.get(id as u8)?;
    let character = world.characters[id as usize].as_ref()?;
    let grounded = world.collision.is_on_ground(core.pos, core::physical_size::<f32>());
    Some(CharacterObservation {
        id,
        team: world.teams_core.team(id),
        pos: core.pos,
        vel: core.vel,
        hook_state: core.hook_state,
        hook_pos: core.hook_pos,
        hooked_player: core.hooked_player(),
        is_frozen: character.freeze_time > 0,
        is_deep_frozen: core.deep_frozen,
        is_live_frozen: core.live_frozen,
        freeze_ticks_remaining: character.freeze_time,
        jumps_left: core.jumps,
        jumps_used: core.jumped_total,
        grounded,
        weapon: core.active_weapon,
        direction: core.direction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::map::{SwitchTile, TILE_SOLID, Tile};

    /// A flat 4x2 room with switch numbers 0-5 laid out on the game layer's row 0 (`number = x`)
    /// — enough for [`apply_switch_state`]/[`rebuild_active_timed_switchers`] to have real
    /// `cores.switchers` slots to touch (`Collision::new` sizes them to the highest `number` any
    /// switch-layer tile declares, regardless of `kind`).
    fn map_with_switches() -> MapData {
        let (w, h) = (6i32, 2i32);
        let mut game = vec![Tile::default(); (w * h) as usize];
        for x in 0..w {
            game[(w + x) as usize] = Tile {
                index: TILE_SOLID,
                ..Default::default()
            };
        }
        let mut switch = vec![SwitchTile::default(); (w * h) as usize];
        for x in 0..w {
            switch[x as usize] = SwitchTile {
                number: x as u8,
                kind: 0,
                flags: 0,
                delay: 0,
            };
        }
        MapData {
            width: w as u32,
            height: h as u32,
            game,
            front: None,
            tele: None,
            speedup: None,
            switch: Some(switch),
            tune: None,
            settings: Vec::new(),
        }
    }

    fn switch_state(highest: i32, status_word: i32, timed: &[(i32, i32)]) -> objects::SwitchState {
        let mut switch_numbers = [0i32; 4];
        let mut end_ticks = [0i32; 4];
        for (i, &(number, end_tick)) in timed.iter().enumerate().take(4) {
            switch_numbers[i] = number;
            end_ticks[i] = end_tick;
        }
        objects::SwitchState {
            highest_switch_number: highest,
            status: [status_word, 0, 0, 0, 0, 0, 0, 0],
            switch_numbers,
            end_ticks,
        }
    }

    #[test]
    fn player_input_conversion_round_trips_every_field() {
        let net = objects::PlayerInput {
            direction: -1,
            target_x: 123,
            target_y: -45,
            jump: 1,
            fire: 7,
            hook: 1,
            player_flags: playerflagflag::PLAYING,
            wanted_weapon: 3,
            next_weapon: 2,
            prev_weapon: 1,
        };
        let physics = player_input_from_net(net);
        assert_eq!(physics.direction, net.direction);
        assert_eq!(physics.target_x, net.target_x);
        assert_eq!(physics.target_y, net.target_y);
        assert_eq!(physics.jump, net.jump);
        assert_eq!(physics.fire, net.fire);
        assert_eq!(physics.hook, net.hook);
        assert_eq!(physics.player_flags, net.player_flags);
        assert_eq!(physics.wanted_weapon, net.wanted_weapon);
        assert_eq!(physics.next_weapon, net.next_weapon);
        assert_eq!(physics.prev_weapon, net.prev_weapon);
        let back = player_input_to_net(physics);
        assert_eq!(back, net);
    }

    #[test]
    fn apply_switch_state_sets_steady_status_bits_exactly() {
        let map = map_with_switches();
        let mut world: World<f32> = World::from_map(&map, 1);
        // Switches 0,2,4 open (bits 0,2,4 set), 1,3,5 closed.
        let sw = switch_state(5, 0b010101, &[]);
        apply_switch_state(&mut world, 0, &sw);
        for n in 0..=5 {
            assert_eq!(world.cores.switchers[n].status[0], n % 2 == 0, "switch {n}");
            // No live end tick reported: must be the plain (non-timed) steady kind.
            assert_eq!(world.cores.switchers[n].end_tick[0], 0, "switch {n} end_tick");
            let expected_kind = if n % 2 == 0 {
                TILE_SWITCHOPEN as i32
            } else {
                TILE_SWITCHCLOSE as i32
            };
            assert_eq!(world.cores.switchers[n].kind[0], expected_kind, "switch {n} kind");
        }
    }

    /// Review round 1, finding F3's exact regression: a switch previously reported as timed
    /// (`end_tick` in the future) must have that cleared back to a plain steady kind once a
    /// *later* message no longer reports it as live — whether because it already flipped, or
    /// because the real remaining time is now past the server's own "< 3s" reporting window —
    /// never left dangling with a stale, already-passed `end_tick` a later `tick_switch_expiry`
    /// call would wrongly act on.
    #[test]
    fn stale_end_tick_is_cleared_once_no_longer_reported_live() {
        let map = map_with_switches();
        let mut world: World<f32> = World::from_map(&map, 1);
        world.tick = 1000;

        // First message: switch 2 open, ending (closing) at tick 1010 — still live (> 1000).
        let sw1 = switch_state(5, 0b0100, &[(2, 1010)]);
        apply_switch_state(&mut world, 0, &sw1);
        assert_eq!(world.cores.switchers[2].end_tick[0], 1010);
        assert_eq!(world.cores.switchers[2].kind[0], TILE_SWITCHTIMEDOPEN as i32);

        // A later message no longer lists switch 2 among the (up to 4) live end ticks at all
        // (either it already flipped — status now closed — or it's simply outside the 3s
        // window) — the stale end_tick must not survive.
        world.tick = 1100;
        let sw2 = switch_state(5, 0b0000, &[]);
        apply_switch_state(&mut world, 0, &sw2);
        assert!(!world.cores.switchers[2].status[0]);
        assert_eq!(
            world.cores.switchers[2].end_tick[0], 0,
            "stale end_tick must be cleared"
        );
        assert_eq!(world.cores.switchers[2].kind[0], TILE_SWITCHCLOSE as i32);
    }

    /// An `end_tick` that has already passed *by the time this exact message arrived* (a
    /// same-message edge case, e.g. a slow-processed snapshot) must be treated the same as "not
    /// reported live" — never applied as if it were still in the future.
    #[test]
    fn end_tick_already_passed_in_the_same_message_is_not_applied() {
        let map = map_with_switches();
        let mut world: World<f32> = World::from_map(&map, 1);
        world.tick = 2000;
        let sw = switch_state(5, 0b0100, &[(2, 2000)]); // end_tick == tick, not > tick.
        apply_switch_state(&mut world, 0, &sw);
        assert_eq!(world.cores.switchers[2].end_tick[0], 0);
        assert_eq!(world.cores.switchers[2].kind[0], TILE_SWITCHOPEN as i32);
    }

    #[test]
    fn rebuild_active_timed_switchers_reflects_only_currently_timed_switches() {
        let map = map_with_switches();
        let mut world: World<f32> = World::from_map(&map, 1);
        world.tick = 100;
        let sw = switch_state(5, 0b0100, &[(2, 200)]);
        apply_switch_state(&mut world, 0, &sw);
        rebuild_active_timed_switchers(&mut world);
        assert_eq!(world.active_timed_switchers, vec![2u8]);

        // Once no longer reported live, it must drop back out of the active list too.
        let sw2 = switch_state(5, 0b0000, &[]);
        apply_switch_state(&mut world, 0, &sw2);
        rebuild_active_timed_switchers(&mut world);
        assert!(world.active_timed_switchers.is_empty());
    }

    #[test]
    fn apply_switch_state_ignores_out_of_range_team() {
        let map = map_with_switches();
        let mut world: World<f32> = World::from_map(&map, 1);
        // `Switcher` (`ddai_physics::core`) has no `PartialEq`, so snapshot the one field this
        // test cares about (team 0's status, everywhere `false` for a freshly-built world)
        // instead of comparing the whole struct.
        let before: Vec<bool> = world.cores.switchers.iter().map(|s| s.status[0]).collect();
        let sw = switch_state(5, 0b111111, &[]);
        apply_switch_state(&mut world, -1, &sw);
        apply_switch_state(&mut world, core::NUM_DDRACE_TEAMS, &sw);
        let after: Vec<bool> = world.cores.switchers.iter().map(|s| s.status[0]).collect();
        assert_eq!(after, before, "out-of-range team must be a no-op");
    }

    #[test]
    fn apply_tune_params_is_a_plain_positional_copy() {
        let mut zone = TuningParams::default();
        let mut net = ddai_net::tuning::DEFAULT_TUNE_PARAMS;
        net.gravity = 12345;
        net.ground_friction = 6789;
        apply_tune_params(&mut zone, &net);
        assert_eq!(zone.get_by_name("gravity"), Some(123.45));
        assert_eq!(zone.get_by_name("ground_friction"), Some(67.89));
    }

    fn input_with_direction(direction: i32) -> PlayerInput {
        PlayerInput {
            direction,
            target_x: 0,
            target_y: -1,
            jump: 0,
            fire: 0,
            hook: 0,
            player_flags: playerflagflag::PLAYING,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }

    #[test]
    fn correct_late_inputs_is_identity_when_nothing_is_late_or_missing() {
        let sent: Vec<(i32, PlayerInput)> = (1..=5).map(|t| (t, input_with_direction(t % 3 - 1))).collect();
        let corrected = correct_late_inputs(&sent, &Default::default());
        assert_eq!(corrected, sent);
    }

    /// Review round 1, finding F5's exact regression: a tick reported late must keep the
    /// *previous* (already-corrected) input, not the new one queued for it.
    #[test]
    fn correct_late_inputs_holds_the_previous_input_for_a_late_tick() {
        let sent = vec![
            (1, input_with_direction(1)),
            (2, input_with_direction(-1)), // reported late below.
            (3, input_with_direction(-1)),
        ];
        let mut late = std::collections::BTreeSet::new();
        late.insert(2);
        let corrected = correct_late_inputs(&sent, &late);
        assert_eq!(
            corrected,
            vec![
                (1, input_with_direction(1)),
                (2, input_with_direction(1)), // held over from tick 1, not tick 2's own value.
                (3, input_with_direction(-1)),
            ]
        );
    }

    /// A tick with no `InputSent` entry at all (a gap — e.g. a dropped event) is treated exactly
    /// like a late one: hold the previous tick's (already-corrected) value.
    #[test]
    fn correct_late_inputs_fills_gaps_by_holding_the_previous_input() {
        let sent = vec![(1, input_with_direction(1)), (3, input_with_direction(-1))]; // tick 2 missing.
        let corrected = correct_late_inputs(&sent, &Default::default());
        assert_eq!(
            corrected,
            vec![
                (1, input_with_direction(1)),
                (2, input_with_direction(1)),
                (3, input_with_direction(-1)),
            ]
        );
    }

    #[test]
    fn correct_late_inputs_of_empty_log_is_empty() {
        assert!(correct_late_inputs(&[], &Default::default()).is_empty());
    }

    /// A late-reported *first* tick has no earlier corrected value to hold over — correctly
    /// produces nothing for it (there is no honest answer), not a manufactured default.
    #[test]
    fn correct_late_inputs_drops_a_late_first_tick_with_nothing_to_hold_over() {
        let sent = vec![(1, input_with_direction(1)), (2, input_with_direction(-1))];
        let mut late = std::collections::BTreeSet::new();
        late.insert(1);
        let corrected = correct_late_inputs(&sent, &late);
        assert_eq!(corrected, vec![(2, input_with_direction(-1))]);
    }

    // --- Review round 3, finding F11: `retarget_late_inputs` ------------------------------------

    /// The task's own required regression: a late-reported input is retargeted *forward* by the
    /// ceiling-division rule, not held in place — a tick reported 25ms late (`ceil(25/20) = 2`)
    /// lands on `T + 2`, and every tick strictly between the previous real input and that retarget
    /// destination holds the *previous* (pre-retarget) input, exactly like a plain gap.
    #[test]
    fn retarget_late_inputs_shifts_a_late_input_forward_by_the_ceiling_of_its_lateness() {
        let sent = vec![(1, input_with_direction(1)), (2, input_with_direction(-1))];
        let mut timing = std::collections::BTreeMap::new();
        timing.insert(2, -25); // 25ms late.
        let retargeted = retarget_late_inputs(&sent, &timing);
        assert_eq!(
            retargeted,
            vec![
                (1, input_with_direction(1)),
                (2, input_with_direction(1)), // gap: holds tick 1's input, not tick 2's own.
                (3, input_with_direction(1)), // gap: still holding.
                (4, input_with_direction(-1)), // 2's input finally lands here (2 + ceil(25/20)).
            ]
        );
    }

    /// A tick `time_left_ms` never mentions at all is assumed on-time — identical output to
    /// [`correct_late_inputs`] with an empty late-set when nothing is ever reported late.
    #[test]
    fn retarget_late_inputs_is_identity_when_nothing_is_reported_late() {
        let sent: Vec<(i32, PlayerInput)> = (1..=5).map(|t| (t, input_with_direction(t % 3 - 1))).collect();
        let retargeted = retarget_late_inputs(&sent, &std::collections::BTreeMap::new());
        assert_eq!(retargeted, sent);
    }

    /// Two different original ticks whose retarget destination collides on the same effective
    /// tick: the *earlier* original tick wins it (`server.cpp:1921`'s own "first arrival" — the
    /// later one has nowhere earlier left to go and simply never lands at all, same as a
    /// late-first-tick in [`correct_late_inputs`]).
    #[test]
    fn retarget_late_inputs_first_original_tick_wins_a_collision() {
        let sent = vec![
            (1, input_with_direction(1)),
            (2, input_with_direction(-1)), // 1ms late -> retargets to tick 3 (ceil(1/20)=1).
            (3, input_with_direction(0)),  // on time, but tick 2 already claims tick 3 first.
        ];
        let mut timing = std::collections::BTreeMap::new();
        timing.insert(2, -1);
        let retargeted = retarget_late_inputs(&sent, &timing);
        // Tick 2's input (direction -1) wins effective tick 3 (processed first, ascending order);
        // tick 3's own input (direction 0) has no earlier slot left and is dropped entirely.
        // Tick 2 itself is a gap now (its own input moved on to tick 3), so it holds tick 1's.
        assert_eq!(
            retargeted,
            vec![
                (1, input_with_direction(1)),
                (2, input_with_direction(1)),
                (3, input_with_direction(-1)),
            ]
        );
    }

    #[test]
    fn retarget_late_inputs_of_empty_log_is_empty() {
        assert!(retarget_late_inputs(&[], &std::collections::BTreeMap::new()).is_empty());
    }

    // --- Review round 3, finding F1 (reopened): `reconstruct_weapon_ammo` -----------------------

    fn net_char(weapon: i32, ammo_count: i32) -> objects::Character {
        objects::Character {
            tick: 0,
            x: 0,
            y: 0,
            vel_x: 0,
            vel_y: 0,
            angle: 0,
            direction: 0,
            jumped: 0,
            hooked_player: -1,
            hook_state: -1,
            hook_tick: 0,
            hook_x: 0,
            hook_y: 0,
            hook_dx: 0,
            hook_dy: 0,
            player_flags: 0,
            health: 10,
            armor: 0,
            ammo_count,
            weapon,
            emote: 0,
            attack_tick: 0,
        }
    }

    /// Review round 3, finding F1's exact regression: `AmmoCount == 0` is the server's "no
    /// information" sentinel (sent for every other player, and for our own tee while frozen), not
    /// a real "zero ammo" reading — every weapon this tee has ever picked up must read back as
    /// infinite (`-1`), never `0`, when that's all the wire gives us.
    #[test]
    fn ammo_sentinel_zero_never_produces_a_real_zero_for_any_got_weapon() {
        let mut core = core::CharacterCore::<f32>::default();
        core.weapons[WEAPON_HAMMER as usize].got = true;
        core.weapons[core::WEAPON_GUN as usize].got = true;
        core.weapons[core::WEAPON_SHOTGUN as usize].got = true;
        let net = net_char(core::WEAPON_GUN, 0);
        reconstruct_weapon_ammo(&mut core, &net);
        assert_eq!(core.weapons[WEAPON_HAMMER as usize].ammo, -1);
        assert_eq!(
            core.weapons[core::WEAPON_GUN as usize].ammo,
            -1,
            "AmmoCount==0 is the sentinel, not a real zero, even for the active weapon"
        );
        assert_eq!(core.weapons[core::WEAPON_SHOTGUN as usize].ammo, -1);
    }

    /// A genuine positive `AmmoCount` reading (only ever sent for the snapping client's own,
    /// unfrozen, active weapon) is honored exactly — but only for that one weapon; every other
    /// `got` weapon still reads back as infinite.
    #[test]
    fn a_real_positive_ammo_count_is_honored_for_the_active_weapon_only() {
        let mut core = core::CharacterCore::<f32>::default();
        core.weapons[WEAPON_HAMMER as usize].got = true;
        core.weapons[core::WEAPON_GUN as usize].got = true;
        let net = net_char(core::WEAPON_GUN, 7);
        reconstruct_weapon_ammo(&mut core, &net);
        assert_eq!(core.weapons[core::WEAPON_GUN as usize].ammo, 7);
        assert_eq!(
            core.weapons[WEAPON_HAMMER as usize].ammo, -1,
            "hammer is always infinite regardless of any AmmoCount reading"
        );
    }

    /// Hammer is unconditionally `-1` even while it happens to be the *active* weapon and the
    /// wire carries a stale positive `ammo_count` left over from a previous weapon (the server
    /// only ever means that field for the currently active weapon; hammer never reads it).
    #[test]
    fn hammer_is_always_minus_one_even_as_the_active_weapon() {
        let mut core = core::CharacterCore::<f32>::default();
        core.weapons[WEAPON_HAMMER as usize].got = true;
        let net = net_char(WEAPON_HAMMER, 9);
        reconstruct_weapon_ammo(&mut core, &net);
        assert_eq!(core.weapons[WEAPON_HAMMER as usize].ammo, -1);
    }

    /// A weapon never picked up (`got == false`) and not the active one is left completely
    /// untouched (still `CharacterCore::default()`'s `0`) — this function must never manufacture
    /// inventory data for a weapon nothing ever proved this tee actually has.
    #[test]
    fn never_gotten_non_active_weapon_is_left_untouched() {
        let mut core = core::CharacterCore::<f32>::default();
        core.weapons[WEAPON_HAMMER as usize].got = true;
        let net = net_char(WEAPON_HAMMER, -1);
        reconstruct_weapon_ammo(&mut core, &net);
        assert_eq!(core.weapons[core::WEAPON_GRENADE as usize].ammo, 0);
        assert!(!core.weapons[core::WEAPON_GRENADE as usize].got);
    }

    /// Ninja is Stage B (not modeled) — must never be touched, matching
    /// `crate::reckoning::evolve_character_core`'s own exclusion.
    #[test]
    fn ninja_is_never_touched() {
        let mut core = core::CharacterCore::<f32>::default();
        core.weapons[WEAPON_NINJA as usize].got = true;
        core.weapons[WEAPON_NINJA as usize].ammo = 12345;
        let net = net_char(WEAPON_NINJA, 999);
        reconstruct_weapon_ammo(&mut core, &net);
        assert_eq!(core.weapons[WEAPON_NINJA as usize].ammo, 12345);
    }

    // --- Review round 3, finding F10: own-tee seeding, end to end -------------------------------

    /// Builds a minimal [`CharacterView`] straight from a physics core (no network round trip),
    /// for tests that drive [`LiveWorld::on_snapshot`] directly from a `ddai_physics::world::World`
    /// under this crate's own control — mirrors the reviewer's `fire_first_tick` repro's own
    /// `to_character_view` helper, trimmed to what these tests actually need (freeze/telegun/solo
    /// etc. are irrelevant here; only weapon "got" flags and the core fields `evolve_character_core`
    /// reads matter).
    fn character_view_from_core(id: i32, core: &core::CharacterCore<f32>) -> CharacterView {
        let net_core = core.write();
        let character = objects::Character {
            tick: 0, // "already current" (`evolve_character_core`'s own `net.tick == 0` fast path).
            x: net_core.x,
            y: net_core.y,
            vel_x: net_core.vel_x,
            vel_y: net_core.vel_y,
            angle: net_core.angle,
            direction: net_core.direction,
            jumped: net_core.jumped,
            hooked_player: net_core.hooked_player,
            hook_state: net_core.hook_state,
            hook_tick: net_core.hook_tick,
            hook_x: net_core.hook_x,
            hook_y: net_core.hook_y,
            hook_dx: net_core.hook_dx,
            hook_dy: net_core.hook_dy,
            player_flags: playerflagflag::PLAYING,
            health: 10,
            armor: 0,
            ammo_count: -1,
            weapon: core.active_weapon,
            emote: 0,
            attack_tick: 0,
        };
        let mut flags = 0;
        if core.weapons[WEAPON_HAMMER as usize].got {
            flags |= core::CHARACTERFLAG_WEAPON_HAMMER;
        }
        if core.weapons[core::WEAPON_GUN as usize].got {
            flags |= core::CHARACTERFLAG_WEAPON_GUN;
        }
        let ddnet = objects::DDNetCharacter {
            flags,
            freeze_end: 0,
            jumps: core.jumps,
            tele_checkpoint: 0,
            strong_weak_id: 0,
            jumped_total: core.jumped_total,
            ninja_activation_tick: -1,
            freeze_start: -1,
            target_x: 0,
            target_y: 0,
            tune_zone_override: -1,
        };
        CharacterView {
            id,
            character,
            ddnet: Some(ddnet),
        }
    }

    fn held_hammer_input(fire: i32) -> PlayerInput {
        PlayerInput {
            direction: 0,
            target_x: 100,
            target_y: 0,
            jump: 0,
            fire,
            hook: 0,
            player_flags: playerflagflag::PLAYING,
            wanted_weapon: 0,
            next_weapon: 0,
            prev_weapon: 0,
        }
    }

    /// Review round 3, finding F10 (blocker) — end-to-end reproduction through the real
    /// `on_snapshot`/`predict` pipeline (mirrors the reviewer's own `fire_first_tick`/`PHANTOM=1`
    /// repro): a hammer held at a steady, nonzero fire-press-counter value long enough to be
    /// fully settled, reconstructed from a snapshot taken at that steady state, must never
    /// manufacture a phantom fire on the first predicted tick purely because the seed used to
    /// reset `fire` to `0` — while a *genuine* new press within the same predicted window (the
    /// counter actually changing) must still be detected exactly like the real server would
    /// (proving the fix does not just blanket-suppress firing).
    #[test]
    fn own_tee_reconstruction_has_no_phantom_fire_but_still_detects_a_real_press() {
        let map = Arc::new(ddai_trace::synthetic::build("arena").expect("recipe must exist"));
        let mut truth: World<f32> = World::from_map(&map, 5);
        truth.init(std::iter::empty::<&str>()).expect("map init");
        truth.players[0] = Some(Player::new(0));
        world::spawn_character(&mut truth, 0, ddai_physics::vmath::Vec2::new(200.0, 620.0));
        truth.cores.get_mut(0).unwrap().active_weapon = WEAPON_HAMMER;

        // Settle into a steady state holding `fire == 2` (an even, "released" counter value —
        // `count_input_presses`'s own docs) for long enough that the warm-up's own single real
        // press (the very first `0 -> 2` transition from a fresh spawn) has long since resolved.
        for _ in 0..300 {
            truth.step(&[TickInput {
                id: 0,
                input: held_hammer_input(2),
                kill: false,
            }]);
        }
        let base = truth.tick;

        let view = character_view_from_core(0, truth.cores.get(0).unwrap());
        let mut lw = LiveWorld::new(Arc::clone(&map), 0, 5);
        lw.on_snapshot(
            base,
            &[view],
            ddai_net::tuning::DEFAULT_TUNE_PARAMS,
            &[],
            None,
            Some(held_hammer_input(2)),
        );

        // k=1,2: still holding `fire == 2` — no new press, must not phantom-fire.
        // k=3..5: a genuine new press (`fire`: 2 -> 3 at k=3), held afterward — must still fire.
        let own_in: Vec<(i32, PlayerInput)> = (1..=5)
            .map(|k| (base + k, held_hammer_input(if k < 3 { 2 } else { 3 })))
            .collect();

        for k in 1..=5 {
            truth.step(&[TickInput {
                id: 0,
                input: own_in[(k - 1) as usize].1,
                kill: false,
            }]);
            let predicted = lw.predict(base + k, &own_in);
            let expected = truth.characters[0].unwrap().reload_timer;
            let got = predicted.characters[0].unwrap().reload_timer;
            assert_eq!(got, expected, "reload_timer mismatch at k={k} (phantom or missed fire)");
        }
    }

    /// The same scenario with `own_input_at_tick: None`, but with no *previous* seed either (this
    /// character's very first-ever sighting — [`LiveWorld::held_input`] is `None` for it) is a
    /// *known, documented* degradation, not this test's concern: it deliberately is **not**
    /// asserted to match `truth` here (it would, in fact, still phantom-fire at k=1 — falling all
    /// the way back to [`derive_held_input`]'s neutral guess is the only honest answer when there
    /// is truly no information at all, not even a previous seed to reuse — see review round 3,
    /// finding F15's own doc comment on [`LiveWorld::on_snapshot`] for the case that *does* have a
    /// previous seed to reuse, [`no_seed_after_a_fire_press_gives_no_phantom_hit`]). This test only
    /// pins down that `None` doesn't panic and produces *some* result, so a future refactor can't
    /// silently change `on_snapshot`'s signature in a way that breaks the fallback path outright.
    #[test]
    fn own_tee_reconstruction_fallback_without_a_known_input_does_not_panic() {
        let map = Arc::new(ddai_trace::synthetic::build("arena").expect("recipe must exist"));
        let mut truth: World<f32> = World::from_map(&map, 5);
        truth.init(std::iter::empty::<&str>()).expect("map init");
        truth.players[0] = Some(Player::new(0));
        world::spawn_character(&mut truth, 0, ddai_physics::vmath::Vec2::new(200.0, 620.0));
        truth.cores.get_mut(0).unwrap().active_weapon = WEAPON_HAMMER;
        for _ in 0..300 {
            truth.step(&[TickInput {
                id: 0,
                input: held_hammer_input(2),
                kill: false,
            }]);
        }
        let base = truth.tick;
        let view = character_view_from_core(0, truth.cores.get(0).unwrap());
        let mut lw = LiveWorld::new(Arc::clone(&map), 0, 5);
        lw.on_snapshot(base, &[view], ddai_net::tuning::DEFAULT_TUNE_PARAMS, &[], None, None);
        let own_in: Vec<(i32, PlayerInput)> = (1..=5).map(|k| (base + k, held_hammer_input(2))).collect();
        let _ = lw.predict(base + 5, &own_in);
    }

    /// Review round 3, finding F15: once a real seed *has* been established (a successfully
    /// delivered `SessionEvent::InputSent`, matching [`own_tee_reconstruction_has_no_phantom_fire_
    /// but_still_detects_a_real_press`]'s first `on_snapshot` call), a *later* snapshot whose own
    /// `InputSent` was dropped (F12's own droppability) must reuse that previous seed — not fall
    /// all the way back to [`derive_held_input`]'s fresh `fire: 0` guess, which would reintroduce
    /// F10's exact phantom-fire bug for every dropped/missing tick. Repro: seed once with a real,
    /// settled `fire == 2` press; a later `on_snapshot` call passes `None` (simulating the drop)
    /// while the button is still just being *held* (no new press) — the reconstructed seed must
    /// still read `fire == 2`, and predicting forward must not phantom-fire.
    #[test]
    fn no_seed_after_a_fire_press_gives_no_phantom_hit() {
        let map = Arc::new(ddai_trace::synthetic::build("arena").expect("recipe must exist"));
        let mut truth: World<f32> = World::from_map(&map, 5);
        truth.init(std::iter::empty::<&str>()).expect("map init");
        truth.players[0] = Some(Player::new(0));
        world::spawn_character(&mut truth, 0, ddai_physics::vmath::Vec2::new(200.0, 620.0));
        truth.cores.get_mut(0).unwrap().active_weapon = WEAPON_HAMMER;

        // Settle into a steady state holding `fire == 2` — the warm-up's own single real press
        // (`0 -> 2` from a fresh spawn) has long since resolved by the time either snapshot below
        // is taken.
        for _ in 0..300 {
            truth.step(&[TickInput {
                id: 0,
                input: held_hammer_input(2),
                kill: false,
            }]);
        }

        let mut lw = LiveWorld::new(Arc::clone(&map), 0, 5);
        // First snapshot: a real seed *is* known (a successfully delivered `InputSent`).
        let tick_a = truth.tick;
        let view_a = character_view_from_core(0, truth.cores.get(0).unwrap());
        lw.on_snapshot(
            tick_a,
            &[view_a],
            ddai_net::tuning::DEFAULT_TUNE_PARAMS,
            &[],
            None,
            Some(held_hammer_input(2)),
        );

        // Advance a few more ticks, still just holding `fire == 2` (no new press) — matches a
        // real connection's next snapshot.
        for _ in 0..4 {
            truth.step(&[TickInput {
                id: 0,
                input: held_hammer_input(2),
                kill: false,
            }]);
        }
        let tick_b = truth.tick;
        let view_b = character_view_from_core(0, truth.cores.get(0).unwrap());
        // Second snapshot: `own_input_at_tick` is `None` — simulates a dropped `InputSent`.
        lw.on_snapshot(
            tick_b,
            &[view_b],
            ddai_net::tuning::DEFAULT_TUNE_PARAMS,
            &[],
            None,
            None,
        );

        // The reused seed must be the previous known value (`fire == 2`), not a fresh `fire: 0`
        // guess — pin the seeding itself directly, not just its downstream effect.
        let seeded = lw.base_world().characters[0].unwrap();
        assert_eq!(
            seeded.latest_input.fire, 2,
            "must reuse the previous own seed, not reset to fire: 0"
        );
        assert_eq!(seeded.latest_prev_input.fire, 2);

        // And predicting forward (still just holding `fire == 2`, no new press) must not
        // phantom-fire.
        truth.step(&[TickInput {
            id: 0,
            input: held_hammer_input(2),
            kill: false,
        }]);
        let own_in = vec![(tick_b + 1, held_hammer_input(2))];
        let predicted = lw.predict(tick_b + 1, &own_in);
        assert_eq!(
            predicted.characters[0].unwrap().reload_timer,
            truth.characters[0].unwrap().reload_timer,
            "reload_timer mismatch — phantom fire from a dropped InputSent"
        );
    }
}
