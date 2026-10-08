//! Task 3.19 (D-116): the arena's **live view** -- what the live bot's brain sees, built from what a server tells a client.
//!
//! The arena normally hands a brain the true `World<f32>` of the tick (`WorldView { world, lag_ticks, in_flight }`): every character's real
//! reload timer, input and unquantised state, no gaps. The live bot sees none of that. A DDNet server sends a snapshot every second tick;
//! each character in it carries the *dead-reckoned send core* of the server (an old state plus the tick it was taken at, re-synced
//! only when the client's own idealised continuation would differ), the quantised core fields and the DDNet extension, but not the other
//! tees' inputs, not their reload timers; `LiveWorld` rebuilds a world from that and rolls it forward to the tick our input will reach the
//! server (`bot.rs`, step 9), holding every other tee's last visible input.
//!
//! This module puts that path into the arena:
//!
//! * [`ServerView`] is the server's side: the dead-reckoning memory of every tee (`CCharacter::TickDeferred`: an idealised dummy core that
//!   is compared with the true core every tick; a mismatch, or three seconds of age, re-syncs the send core) and the snapshot objects
//!   (`CNetObj_Character` + `CNetObj_DDNetCharacter`) of the world after a tick, quantised as on the wire. The same conversion
//!   `tests/live_features.rs` (task 3.17) proved bit-exact against `LiveWorld`, now with the per-tick re-sync the server does.
//! * [`LiveSeat`] is the client's side for one player: a `LiveWorld`, fed the snapshot at each decision tick (every `decide_every`
//!   ticks, the snapshot cadence), our own applied input as `own_input_at_tick`, and our own unacknowledged inputs as `in_flight`; the
//!   brain then decides on `predict_local_observation(to_tick = tick + lag, ...)` with `WorldView { lag_ticks: 0, in_flight: [] }`, exactly
//!   as `Bot::decide` does.
//!
//! What is *not* modelled: network jitter (every snapshot arrives on time, our lag is the fixed `PlayerSetup::lag`), snapshot loss and
//! the bot's post-filters (hook/hammer vetoes act on spared tees, none in a duel). The arena's other player keeps the true view.

use ddai_brain::{Action, Brain, LiveContext, Observation, WorldView};
use ddai_net::generated::enums::playerflagflag;
use ddai_net::generated::objects;
use ddai_net::tuning::DEFAULT_TUNE_PARAMS;
use ddai_net::view::CharacterView;
use ddai_physics::core::{self, MAX_CLIENTS, PlayerInput};
use ddai_physics::map::MapData;
use ddai_physics::world::World;
use ddai_world::{LiveWorld, SnapshotInput};
use std::collections::VecDeque;
use std::sync::Arc;

/// `m_ReckoningTick + TickSpeed * 3 < Tick`: the server never lets a send core grow older than three seconds.
const RECKONING_MAX_AGE: i32 = 3 * core::SERVER_TICK_SPEED;

/// The server's memory of one tee: the send core (`m_SendCore`, with `m_ReckoningTick` in `tick`) and the idealised dummy that continues
/// it (`m_ReckoningCore`, as the snapshot object it quantises to).
#[derive(Clone, Copy)]
struct Reck {
    sent: objects::Character,
    dummy: objects::Character,
}

/// The server side of the live view: the dead-reckoning state of every tee. One per game, advanced once per world tick.
pub struct ServerView {
    recks: Vec<Option<Reck>>,
}

impl Default for ServerView {
    fn default() -> Self {
        Self::new()
    }
}

/// The wire object of a character's current (true, quantised) state; `tick` 0 as the server writes a fresh core.
fn char_object(world: &World<f32>, id: u8) -> Option<objects::Character> {
    let core = world.cores.get(id)?;
    let ch = world.characters[id as usize].as_ref().filter(|c| c.alive)?;
    let n = core.write();
    Some(objects::Character {
        tick: 0,
        x: n.x,
        y: n.y,
        vel_x: n.vel_x,
        vel_y: n.vel_y,
        angle: n.angle,
        direction: n.direction,
        jumped: n.jumped,
        hooked_player: n.hooked_player,
        hook_state: n.hook_state,
        hook_tick: n.hook_tick,
        hook_x: n.hook_x,
        hook_y: n.hook_y,
        hook_dx: n.hook_dx,
        hook_dy: n.hook_dy,
        player_flags: playerflagflag::PLAYING,
        health: 10,
        armor: ch.armor,
        ammo_count: -1,
        weapon: core.active_weapon,
        emote: 0,
        attack_tick: ch.attack_tick,
    })
}

/// The core fields of two character objects are equal (`mem_comp` of `CNetObj_Character` after `Write(core)`; the non-core fields are the same
/// by construction).
fn same_core(a: &objects::Character, b: &objects::Character) -> bool {
    (
        a.x,
        a.y,
        a.vel_x,
        a.vel_y,
        a.angle,
        a.direction,
        a.jumped,
        a.hooked_player,
    ) == (
        b.x,
        b.y,
        b.vel_x,
        b.vel_y,
        b.angle,
        b.direction,
        b.jumped,
        b.hooked_player,
    ) && (a.hook_state, a.hook_tick, a.hook_x, a.hook_y, a.hook_dx, a.hook_dy)
        == (b.hook_state, b.hook_tick, b.hook_x, b.hook_y, b.hook_dx, b.hook_dy)
}

/// `CCharacter::Snap`'s `CNetObj_DDNetCharacter` for `id`.
fn ddnet_object(world: &World<f32>, id: u8) -> Option<objects::DDNetCharacter> {
    let c = world.cores.get(id)?;
    let ch = world.characters[id as usize].as_ref().filter(|c| c.alive)?;
    use ddai_net::generated::enums::characterflagflag as f;
    let mut flags = 0;
    let mut set = |on: bool, bit: i32| {
        if on {
            flags |= bit;
        }
    };
    set(c.solo, f::SOLO);
    set(c.jetpack, f::JETPACK);
    set(c.collision_disabled, f::COLLISION_DISABLED);
    set(c.endless_hook, f::ENDLESS_HOOK);
    set(c.endless_jump, f::ENDLESS_JUMP);
    set(c.is_super, f::SUPER);
    set(c.hammer_hit_disabled, f::HAMMER_HIT_DISABLED);
    set(c.shotgun_hit_disabled, f::SHOTGUN_HIT_DISABLED);
    set(c.grenade_hit_disabled, f::GRENADE_HIT_DISABLED);
    set(c.laser_hit_disabled, f::LASER_HIT_DISABLED);
    set(c.hook_hit_disabled, f::HOOK_HIT_DISABLED);
    set(c.has_telegun_gun, f::TELEGUN_GUN);
    set(c.has_telegun_grenade, f::TELEGUN_GRENADE);
    set(c.has_telegun_laser, f::TELEGUN_LASER);
    set(c.weapons[core::WEAPON_HAMMER as usize].got, f::WEAPON_HAMMER);
    set(c.weapons[core::WEAPON_GUN as usize].got, f::WEAPON_GUN);
    set(c.weapons[core::WEAPON_SHOTGUN as usize].got, f::WEAPON_SHOTGUN);
    set(c.weapons[core::WEAPON_GRENADE as usize].got, f::WEAPON_GRENADE);
    set(c.weapons[core::WEAPON_LASER as usize].got, f::WEAPON_LASER);
    set(c.weapons[core::WEAPON_NINJA as usize].got, f::WEAPON_NINJA);
    set(c.live_frozen, f::MOVEMENTS_DISABLED);
    set(c.is_in_freeze, f::IN_FREEZE);
    set(c.invincible, f::INVINCIBLE);
    let freeze_end = if c.deep_frozen {
        -1
    } else if ch.freeze_time > 0 {
        world.tick + ch.freeze_time
    } else {
        0
    };
    Some(objects::DDNetCharacter {
        flags,
        freeze_end,
        jumps: c.jumps,
        tele_checkpoint: 0,
        strong_weak_id: ch.strong_weak_id,
        jumped_total: c.jumped_total,
        ninja_activation_tick: c.ninja.activation_tick,
        freeze_start: c.freeze_start,
        target_x: ch.input.target_x,
        target_y: ch.input.target_y,
        tune_zone_override: -1,
    })
}

impl ServerView {
    pub fn new() -> Self {
        ServerView {
            recks: vec![None; MAX_CLIENTS],
        }
    }

    /// `CCharacter::TickDeferred`'s last block, once per world tick, for every tee: advance the idealised dummy by one tick, compare it with the
    /// true core, and re-sync the send core on a mismatch or when it is three seconds old. Call it after each `World::step` (and once for the
    /// initial world).
    pub fn after_tick(&mut self, world: &World<f32>) {
        let now = world.tick;
        for id in 0..MAX_CLIENTS {
            let cur = char_object(world, id as u8);
            let Some(cur) = cur else {
                self.recks[id] = None;
                continue;
            };
            let resync = match self.recks[id] {
                None => true,
                Some(r) => {
                    // The dummy of the previous tick, one idealised tick on (no input, an empty world, default tuning).
                    let ideal = ddai_world::reckoning::evolve_character_core(
                        &objects::Character {
                            tick: now - 1,
                            ..r.dummy
                        },
                        now,
                        &world.collision,
                    )
                    .write();
                    let ideal_obj = objects::Character {
                        x: ideal.x,
                        y: ideal.y,
                        vel_x: ideal.vel_x,
                        vel_y: ideal.vel_y,
                        angle: ideal.angle,
                        direction: ideal.direction,
                        jumped: ideal.jumped,
                        hooked_player: ideal.hooked_player,
                        hook_state: ideal.hook_state,
                        hook_tick: ideal.hook_tick,
                        hook_x: ideal.hook_x,
                        hook_y: ideal.hook_y,
                        hook_dx: ideal.hook_dx,
                        hook_dy: ideal.hook_dy,
                        ..r.dummy
                    };
                    // A mismatch (the true core left the idealised path) or a send core three seconds old re-syncs it.
                    if !same_core(&ideal_obj, &cur) || r.sent.tick + RECKONING_MAX_AGE < now {
                        true
                    } else {
                        self.recks[id] = Some(Reck {
                            sent: r.sent,
                            dummy: ideal_obj,
                        });
                        false
                    }
                }
            };
            if resync {
                let fresh = objects::Character { tick: now, ..cur };
                self.recks[id] = Some(Reck {
                    sent: fresh,
                    dummy: fresh,
                });
            }
        }
    }

    /// The snapshot characters the server would send for the world as it is after the last [`ServerView::after_tick`], in client id order.
    pub fn characters(&self, world: &World<f32>) -> Vec<CharacterView> {
        let mut out = Vec::with_capacity(2);
        for id in 0..MAX_CLIENTS {
            let (Some(rk), Some(cur), Some(dd)) = (
                self.recks[id].as_ref(),
                char_object(world, id as u8),
                ddnet_object(world, id as u8),
            ) else {
                continue;
            };
            // The core fields and the tick are the send core's; the direction, the weapon and the attack tick are the live ones.
            let character = objects::Character {
                direction: cur.direction,
                weapon: cur.weapon,
                attack_tick: cur.attack_tick,
                armor: cur.armor,
                ..rk.sent
            };
            out.push(CharacterView {
                id: id as i32,
                character,
                ddnet: Some(dd),
            });
        }
        out
    }
}

/// The wire input (applied by the world) in the shape `LiveWorld` takes: the same type, the physics' own `PlayerInput`.
pub type Wire = PlayerInput;

/// The client side of the live view for one player.
pub struct LiveSeat {
    live: LiveWorld,
    own_id: i32,
    obs: Observation,
    keep: [bool; MAX_CLIENTS],
    in_flight: Vec<(i32, Wire)>,
}

impl LiveSeat {
    pub fn new(map: Arc<MapData>, own_id: i32, ids: &[i32], seed: u64) -> LiveSeat {
        let live = LiveWorld::new(Arc::clone(&map), own_id, seed);
        let obs = live.build_observation(live.base_world(), None);
        let mut keep = [false; MAX_CLIENTS];
        for &id in ids {
            if let Some(k) = usize::try_from(id).ok().and_then(|i| keep.get_mut(i)) {
                *k = true;
            }
        }
        LiveSeat {
            live,
            own_id,
            obs,
            keep,
            in_flight: Vec::with_capacity(8),
        }
    }

    /// The decision at snapshot tick `tick` (the world after `tick` steps): feed the snapshot, roll to `tick + lag` with our unacknowledged
    /// inputs, and let the brain decide on that world as the live bot does.
    ///
    /// `applied` is the input the server applied to our tee in the step that made this tick (`own_input_at_tick`); `pending` holds our sent but
    /// not yet applied inputs as `(tick they act in the step that ends at, input)`, oldest first (the sim's queue).
    #[allow(clippy::too_many_arguments)]
    pub fn decide(
        &mut self,
        brain: &mut dyn Brain,
        server: &ServerView,
        world: &World<f32>,
        applied: Wire,
        pending: &VecDeque<(i32, Wire)>,
        lag: u32,
        target_id: Option<i32>,
    ) -> Action {
        let tick = world.tick;
        let characters = server.characters(world);
        let mut input = SnapshotInput::new(tick, &characters, DEFAULT_TUNE_PARAMS);
        input.own_input_at_tick = Some(applied);
        self.live.on_snapshot(input);
        let to_tick = tick + i32::try_from(lag).unwrap_or(0);
        // The claims the live bot's `SentLog::in_flight(tick, to_tick)` would give: our sent inputs for the ticks in `(tick, to_tick]`.
        self.in_flight.clear();
        self.in_flight
            .extend(pending.iter().filter(|(t, _)| *t > tick && *t <= to_tick).copied());
        let predicted =
            self.live
                .predict_local_observation(to_tick, &self.in_flight, &self.keep, target_id, &mut self.obs);
        let view = WorldView {
            world: predicted,
            self_id: self.own_id,
            lag_ticks: 0,
            in_flight: &[],
        };
        // The bot tells its brain what it knows besides the world; here that is the duel.
        brain.set_live_context(&LiveContext {
            duel: true,
            ..LiveContext::default()
        });
        brain.decide_in(&self.obs, Some(&view))
    }

    pub fn live(&self) -> &LiveWorld {
        &self.live
    }

    /// The observation of the last decision (the predicted world at `tick + lag`): what the brain was asked about.
    pub fn observation(&self) -> &Observation {
        &self.obs
    }
}
