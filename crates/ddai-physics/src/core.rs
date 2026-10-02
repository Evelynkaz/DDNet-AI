// Ported from DDNet 20.1 `src/game/gamecore.{h,cpp}` (`CCharacterCore`, `CWorldCore`,
// `CTeamsCore` — the last one is a thin re-export of `crate::teams::TeamsCore`, ported from
// `src/game/teamscore.{h,cpp}`). DDNet's zlib-style license notice for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//
// Altered for DDNet-AI: rewritten in Rust, generic over `R: Real`, with **no `unsafe`** and
// therefore no raw `m_pWorld`/`m_pCollision`/`m_pTeams` pointers inside [`CharacterCore`] — C++
// keeps a character "aware of its world" via those pointers; here, `Tick`/`TickDeferred`/`Move`
// become free functions that take `&mut WorldCore<R, CAP>` (the array of sibling characters,
// keyed by client id — see that type's doc comment) plus `&Collision<R>`/`&TeamsCore` explicitly.
// Every such function follows the same pattern to satisfy the borrow checker without pointers or
// interior mutability: copy `self`'s [`CharacterCore`] out of the array by value (it's `Copy`),
// read/write *other* slots of the array freely (no aliasing, since `self`'s copy no longer
// borrows from it), then write the (possibly mutated) copy back into `self`'s slot before
// returning. This exactly reproduces the C++ observable behavior — including order-dependent
// effects like "does sibling X's `TickDeferred` see sibling Y's *this-tick* or *last-tick*
// `m_MoveRestrictions`" — because every field access still happens against the same shared
// array, at the same points in the same per-character call sequence; only *how* Rust expresses
// "read/write a sibling" differs from C++'s pointer dereference.
//
// `m_AntiPingInterfereCallback` (`SetAntiPingInterfereCallback`) is explicitly commented
// "clientside only" in `gamecore.h` and defaults to a no-op lambda that Oracle A/`core_world`
// never replaces — calling it has zero effect on any field this crate's state schema records, so
// it is not ported at all (not even as a no-op call) to keep the hot path lean.
//
// `m_aWeapons`/`m_Ninja`/`m_HasTelegunGun/Grenade/Laser`/`m_FreezeStart`/`m_FreezeEnd`/
// `m_IsInFreeze`/`m_DeepFrozen`/`m_LiveFrozen` ARE ported as [`CharacterCore`] fields (review
// round 2, finding F2: an earlier revision of this crate omitted them, on the mistaken claim
// that they're "only ever written by `ReadDDNet`" — true for the telegun/weapon/ninja fields,
// but wrong for `m_LiveFrozen`/`m_DeepFrozen`, which `character.cpp`'s `Freeze`/`UnFreeze` write
// directly). None of `Tick`/`TickDeferred`/`Move`/`Reset` themselves read or write any of these
// (only [`CharacterCore::read_ddnet`], `character.cpp`'s DDRace/weapon logic — out of scope per
// the task spec — and, for `m_LiveFrozen`/`m_DeepFrozen`, `character.cpp`'s freeze handling do),
// so every one of them stays at its `Reset()`/zero-init default for the whole of any scenario
// Oracle A/`core_world` build in this task's scope — structurally present (satisfying
// acceptance criterion 1.e and giving task 1.6+ real fields/`read_ddnet` to build on), but inert
// here. `m_Solo`/`m_Jetpack`/`m_CollisionDisabled`/`m_EndlessHook`/`m_EndlessJump`/
// `m_HammerHitDisabled`/`m_GrenadeHitDisabled`/`m_LaserHitDisabled`/`m_ShotgunHitDisabled`/
// `m_HookHitDisabled`/`m_Super`/`m_Invincible` are ALSO only ever written by `ReadDDNet` in this
// crate's scope, but unlike the fields above, `Tick`/`TickDeferred`/`Move` themselves *read*
// every one of them — the actual reason they were already present before this finding.

use crate::collision::{self, Collision};
use crate::map;
use crate::prng::Prng;
use crate::real::Real;
use crate::tuning::TuningParams;
use crate::vmath::{self, Vec2};

/// DDNet's `MAX_CLIENTS` (`engine/shared/protocol.h`): the valid range for a client id (`0` to
/// this, exclusive) and the size of `CWorldCore::m_apCharacters` in the original C++. See
/// [`WorldCore`]'s doc comment for why this crate's `WorldCore` doesn't literally allocate an
/// array of this size.
pub const MAX_CLIENTS: usize = 128;

/// `SERVER_TICK_SPEED` (`engine/shared/protocol.h`).
pub const SERVER_TICK_SPEED: i32 = 50;

/// `CCharacterCore::PhysicalSize()`: `28.0f`.
pub fn physical_size<R: Real>() -> R {
    R::from_i32(28)
}

/// `CCharacterCore::PhysicalSizeVec2()`: `(28.0f, 28.0f)`.
pub fn physical_size_vec2<R: Real>() -> Vec2<R> {
    Vec2::new(physical_size(), physical_size())
}

// --- Hook state machine values (`gamecore.h` anonymous enum) -----------------------------------

/// Hook has been released and is snapping back with no further effect.
pub const HOOK_RETRACTED: i32 = -1;
/// No hook in flight.
pub const HOOK_IDLE: i32 = 0;
/// First tick of the (3-tick) retract animation.
pub const HOOK_RETRACT_START: i32 = 1;
/// Last tick of the retract animation, transitions to [`HOOK_RETRACTED`] next tick.
pub const HOOK_RETRACT_END: i32 = 3;
/// Hook is flying outward.
pub const HOOK_FLYING: i32 = 4;
/// Hook is attached (to the ground or another player).
pub const HOOK_GRABBED: i32 = 5;

// --- `m_TriggeredEvents` bits (`gamecore.h` anonymous enum) -------------------------------------

/// A ground jump happened this tick.
pub const COREEVENT_GROUND_JUMP: i32 = 0x01;
/// An air jump happened this tick.
pub const COREEVENT_AIR_JUMP: i32 = 0x02;
/// The hook was fired this tick.
pub const COREEVENT_HOOK_LAUNCH: i32 = 0x04;
/// The hook attached to another player this tick.
pub const COREEVENT_HOOK_ATTACH_PLAYER: i32 = 0x08;
/// The hook attached to the ground this tick.
pub const COREEVENT_HOOK_ATTACH_GROUND: i32 = 0x10;
/// The hook hit a `TILE_NOHOOK` tile this tick.
pub const COREEVENT_HOOK_HIT_NOHOOK: i32 = 0x20;
/// The hook started retracting this tick.
pub const COREEVENT_HOOK_RETRACT: i32 = 0x40;

/// `SaturatedAdd<T>(T Min, T Max, T Current, T Modifier)` (`gamecore.h` template).
pub fn saturated_add<R: Real>(min: R, max: R, current: R, modifier: R) -> R {
    if modifier < R::ZERO {
        if current < min {
            return current;
        }
        let mut current = current + modifier;
        if current < min {
            current = min;
        }
        current
    } else {
        if current > max {
            return current;
        }
        let mut current = current + modifier;
        if current > max {
            current = max;
        }
        current
    }
}

/// `VelocityRamp(float Value, float Start, float Range, float Curvature)` (`gamecore.cpp`).
pub fn velocity_ramp<R: Real>(value: R, start: R, range: R, curvature: R) -> R {
    if value < start {
        return R::ONE;
    }
    R::ONE / curvature.powf((value - start) / range)
}

/// `m_Angle`'s computation (`gamecore.cpp` `Tick`): `std::atan2(m_Input.m_TargetY,
/// m_Input.m_TargetX)` — both `int` arguments promote to `double` in C++ (`std::atan2` has no
/// `(int, int)` overload), so this is computed in `f64` regardless of which `Real` the rest of
/// the core is instantiated over, exactly like the task spec calls out as a required exception
/// to the "generic over `R`" rule; the result narrows to `f32` (`float TmpAngle = ...;`) before
/// the rest of the expression, which is genuinely `float` in the C++ source.
pub fn angle_from_target(target_x: i32, target_y: i32) -> i32 {
    let tmp_angle_f64 = (target_y as f64).atan2(target_x as f64);
    let tmp_angle = tmp_angle_f64 as f32;
    let pi = std::f32::consts::PI;
    // `atan2` of two finite `f64`s is always finite and in `[-pi, pi]`, so this narrowing
    // `(int)` cast can never actually hit the NaN/out-of-range case review round 1's finding F1
    // is about — routed through `to_i32_trunc` anyway (rather than a bare `as i32`) so nothing in
    // this crate's float→int narrowing bypasses that shared, C++-`cvttss2si`-matching rule.
    if tmp_angle < -(pi / 2.0) {
        <f32 as Real>::to_i32_trunc((tmp_angle + 2.0 * pi) * 256.0)
    } else {
        <f32 as Real>::to_i32_trunc(tmp_angle * 256.0)
    }
}

/// `CNetObj_PlayerInput` (`generated/protocol.h`): the 10-field applied input, in wire order.
/// Mirrors `ddai_trace::scenario::PlayerInput` field-for-field, but is this crate's own type —
/// `ddai-physics` cannot depend on `ddai-trace` (the dependency runs the other way).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlayerInput {
    /// `-1`/`0`/`1`: desired horizontal movement.
    pub direction: i32,
    /// Aim vector X (pixels, relative to the character).
    pub target_x: i32,
    /// Aim vector Y (pixels, relative to the character).
    pub target_y: i32,
    /// Jump key held.
    pub jump: i32,
    /// Fire key press-counter (odd = currently held) — not read by core-level physics.
    pub fire: i32,
    /// Hook key held.
    pub hook: i32,
    /// Client display flags — not read by core-level physics.
    pub player_flags: i32,
    /// Requested weapon slot — not read by core-level physics.
    pub wanted_weapon: i32,
    /// Next-weapon key press-counter — not read by core-level physics.
    pub next_weapon: i32,
    /// Previous-weapon key press-counter — not read by core-level physics.
    pub prev_weapon: i32,
}

/// `CNetObj_CharacterCore` (`generated/protocol.h`), the subset `Write`/`Read` touch (`m_Tick` is
/// never written by `CCharacterCore::Write`, so it's omitted here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetCharacterCore {
    /// Quantized `m_Pos.x` (`round_to_int`).
    pub x: i32,
    /// Quantized `m_Pos.y`.
    pub y: i32,
    /// Quantized `m_Vel.x * 256`.
    pub vel_x: i32,
    /// Quantized `m_Vel.y * 256`.
    pub vel_y: i32,
    /// `m_Angle`, unchanged by the round trip.
    pub angle: i32,
    /// `m_Direction`, unchanged by the round trip.
    pub direction: i32,
    /// `m_Jumped`, unchanged by the round trip.
    pub jumped: i32,
    /// `m_HookedPlayer` (`-1` = none), unchanged by the round trip.
    pub hooked_player: i32,
    /// `m_HookState`, unchanged by the round trip.
    pub hook_state: i32,
    /// `m_HookTick`, unchanged by the round trip.
    pub hook_tick: i32,
    /// Quantized `m_HookPos.x`.
    pub hook_x: i32,
    /// Quantized `m_HookPos.y`.
    pub hook_y: i32,
    /// Quantized `m_HookDir.x * 256`.
    pub hook_dx: i32,
    /// Quantized `m_HookDir.y * 256`.
    pub hook_dy: i32,
}

/// `CCharacterCore::m_AttachedPlayers` (`std::set<int>`): which client ids currently have this
/// character hooked. A 128-bit set (one bit per possible client id, [`MAX_CLIENTS`]) instead of a
/// tree/hash set — no heap allocation, `Copy`, and cheap to include in the hot
/// [`CharacterCore`]/[`WorldCore`] structs the performance criterion cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttachedPlayers([u64; 2]);

impl AttachedPlayers {
    fn word_bit(id: i32) -> (usize, u32) {
        debug_assert!((0..MAX_CLIENTS as i32).contains(&id), "invalid client id {id}");
        ((id as usize) / 64, (id as u32) % 64)
    }

    /// Adds `id` to the set.
    pub fn insert(&mut self, id: i32) {
        let (w, b) = Self::word_bit(id);
        self.0[w] |= 1u64 << b;
    }

    /// Removes `id` from the set, if present.
    pub fn remove(&mut self, id: i32) {
        let (w, b) = Self::word_bit(id);
        self.0[w] &= !(1u64 << b);
    }

    /// Whether `id` is currently in the set.
    pub fn contains(&self, id: i32) -> bool {
        let (w, b) = Self::word_bit(id);
        (self.0[w] >> b) & 1 != 0
    }

    /// Whether the set has no members.
    pub fn is_empty(&self) -> bool {
        self.0 == [0, 0]
    }

    /// Ascending client id order (matches `std::set<int>`'s iteration order).
    pub fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        (0..MAX_CLIENTS as i32).filter(move |&id| self.contains(id))
    }
}

/// Number of weapon slots (`gamecore.h`'s `m_aWeapons[NUM_WEAPONS]`; `generated/protocol.h`'s
/// `WEAPON_*`/`NUM_WEAPONS` enum, from `datasrc/network.py`'s `weapons.id` list).
pub const NUM_WEAPONS: usize = 6;
/// `WEAPON_HAMMER`.
pub const WEAPON_HAMMER: i32 = 0;
/// `WEAPON_GUN`.
pub const WEAPON_GUN: i32 = 1;
/// `WEAPON_SHOTGUN`.
pub const WEAPON_SHOTGUN: i32 = 2;
/// `WEAPON_GRENADE`.
pub const WEAPON_GRENADE: i32 = 3;
/// `WEAPON_LASER`.
pub const WEAPON_LASER: i32 = 4;
/// `WEAPON_NINJA`.
pub const WEAPON_NINJA: i32 = 5;

/// `CCharacterCore::CWeaponStat` (`gamecore.h`). Never read/written by anything this crate ports
/// (`Tick`/`TickDeferred`/`Move`/`Reset`) — only by `character.cpp` (weapons, out of scope) and
/// [`CharacterCore::read_ddnet`] — kept for structural completeness (acceptance criterion 1.e).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeaponStat {
    /// `m_AmmoRegenStart`.
    pub ammo_regen_start: i32,
    /// `m_Ammo`.
    pub ammo: i32,
    /// `m_Ammocost`.
    pub ammo_cost: i32,
    /// `m_Got`.
    pub got: bool,
}

/// `CCharacterCore::m_Ninja`'s anonymous struct type (`gamecore.h`). Same scope note as
/// [`WeaponStat`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NinjaState<R: Real> {
    /// `m_ActivationDir`.
    pub activation_dir: Vec2<R>,
    /// `m_ActivationTick`.
    pub activation_tick: i32,
    /// `m_CurrentMoveTime`.
    pub current_move_time: i32,
    /// `m_OldVelAmount`.
    pub old_vel_amount: i32,
}

impl<R: Real> Default for NinjaState<R> {
    fn default() -> Self {
        NinjaState {
            activation_dir: Vec2::zero(),
            activation_tick: 0,
            current_move_time: 0,
            old_vel_amount: 0,
        }
    }
}

/// Port of 20.1 `CCharacterCore` (`gamecore.h`/`gamecore.cpp`). See the module doc comment for
/// exactly which C++ members are/aren't ported and why. `Tick`/`TickDeferred`/`Move` are free
/// functions ([`tick`]/[`tick_deferred`]/[`move_character`]) rather than methods — see the module
/// doc comment for why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterCore<R: Real> {
    /// `m_Pos`.
    pub pos: Vec2<R>,
    /// `m_Vel`.
    pub vel: Vec2<R>,

    /// `m_HookPos`.
    pub hook_pos: Vec2<R>,
    /// `m_HookDir`.
    pub hook_dir: Vec2<R>,
    /// `m_HookTeleBase`.
    pub hook_tele_base: Vec2<R>,
    /// `m_HookTick`.
    pub hook_tick: i32,
    /// `m_HookState` (see [`HOOK_IDLE`] and friends).
    pub hook_state: i32,
    /// `m_AttachedPlayers`.
    pub attached_players: AttachedPlayers,
    hooked_player: i32,

    /// `m_ActiveWeapon` — always `0` in this crate (never written by core-level `Tick`; only
    /// `character.cpp`'s `SetWeapon`, out of scope, ever changes it).
    pub active_weapon: i32,
    /// `m_NewHook`.
    pub new_hook: bool,

    /// `m_Jumped`.
    pub jumped: i32,
    /// `m_JumpedTotal`.
    pub jumped_total: i32,
    /// `m_Jumps`.
    pub jumps: i32,

    /// `m_Direction`.
    pub direction: i32,
    /// `m_Angle`.
    pub angle: i32,
    /// `m_Input`.
    pub input: PlayerInput,

    /// `m_TriggeredEvents` (see the `COREEVENT_*` constants).
    pub triggered_events: i32,

    // DDRace
    /// `m_Id`: this character's own client id.
    pub id: i32,
    /// `m_Reset`.
    pub reset: bool,
    /// `m_Colliding`.
    pub colliding: i32,
    /// `m_LeftWall`.
    pub left_wall: bool,

    // DDNet Character flags — read (not just written) by core-level Tick/TickDeferred/Move; see
    // the module doc comment for why these are kept but weapons/freeze/telegun aren't.
    /// `m_Solo`.
    pub solo: bool,
    /// `m_Jetpack` — stored for structural completeness; not read by any ported function.
    pub jetpack: bool,
    /// `m_CollisionDisabled`.
    pub collision_disabled: bool,
    /// `m_EndlessHook` — stored for structural completeness; endless-hook selection is task 1.7.
    pub endless_hook: bool,
    /// `m_EndlessJump` — stored for structural completeness; not read by any ported function.
    pub endless_jump: bool,
    /// `m_HammerHitDisabled` — stored for structural completeness (weapons are out of scope).
    pub hammer_hit_disabled: bool,
    /// `m_GrenadeHitDisabled` — stored for structural completeness (weapons are out of scope).
    pub grenade_hit_disabled: bool,
    /// `m_LaserHitDisabled` — stored for structural completeness (weapons are out of scope).
    pub laser_hit_disabled: bool,
    /// `m_ShotgunHitDisabled` — stored for structural completeness (weapons are out of scope).
    pub shotgun_hit_disabled: bool,
    /// `m_HookHitDisabled`.
    pub hook_hit_disabled: bool,
    /// `m_Super`. Named `is_super` since `super` is a Rust keyword.
    pub is_super: bool,
    /// `m_Invincible` — stored for structural completeness; not read by any ported function.
    pub invincible: bool,

    // Weapons/ninja/telegun/freeze: written by `ReadDDNet` *and* — for `m_LiveFrozen`/
    // `m_DeepFrozen` specifically — directly by `character.cpp` (`Freeze`/`UnFreeze`, review
    // round 2 finding F2: an earlier revision of this module claimed these were "only ever
    // written by ReadDDNet", which is true for the telegun/weapon/ninja fields but wrong for
    // `m_LiveFrozen`/`m_DeepFrozen`). None of `character.cpp`'s writers run in this crate's scope
    // (Oracle A/`core_world` are core-only — see the task spec), so every one of these still
    // stays at its `Reset()`/zero-init default for the whole of any scenario this crate builds;
    // kept for structural completeness (acceptance criterion 1.e) and so [`CharacterCore::read_ddnet`]
    // has real fields to write into for task 1.6+.
    /// `m_aWeapons` (`CWeaponStat[NUM_WEAPONS]`).
    pub weapons: [WeaponStat; NUM_WEAPONS],
    /// `m_Ninja`.
    pub ninja: NinjaState<R>,
    /// `m_HasTelegunGun`.
    pub has_telegun_gun: bool,
    /// `m_HasTelegunGrenade`.
    pub has_telegun_grenade: bool,
    /// `m_HasTelegunLaser`.
    pub has_telegun_laser: bool,
    /// `m_FreezeStart`.
    pub freeze_start: i32,
    /// `m_FreezeEnd`.
    pub freeze_end: i32,
    /// `m_IsInFreeze`.
    pub is_in_freeze: bool,
    /// `m_DeepFrozen`.
    pub deep_frozen: bool,
    /// `m_LiveFrozen`.
    pub live_frozen: bool,

    /// Copied from the world's tuning at setup (`CCharacterCore::m_Tuning`), matching C++: each
    /// character has its own copy so a future per-character/tune-zone override (task 1.7) can
    /// change it without affecting siblings.
    pub tuning: TuningParams,

    /// `CCharacterCore::m_MoveRestrictions` (private in C++, `gamecore.h:282`) — computed once
    /// per tick, at the start of `Tick()`, from `CCollision::GetMoveRestrictions(callback, this,
    /// m_Pos)` (no `MapIndex` override), and read by `TickDeferred`'s hook-drag `ClampVel` calls
    /// only (both by `self` and, for the hooked character's own `ClampVel`, by *another*
    /// `CharacterCore`'s `TickDeferred`, matching C++'s same-class private-field access from a
    /// different instance — `gamecore.cpp:517,521`). See [`CharacterCore::move_restrictions`]'s
    /// doc comment for why this is a genuinely *different* field from `CCharacter`'s own
    /// `m_MoveRestrictions` (task 1.6's [`crate::world::Character::move_restrictions`]) despite
    /// the identical name — conflating the two was a real, found-empirically bug this crate had.
    move_restrictions: i32,
}

impl<R: Real> Default for CharacterCore<R> {
    /// Value-initialized (every field zero/false/`Vec2::zero()`), matching Oracle A's
    /// `std::vector<CCharacterCore>(N)` — see `docs/formats.md` §5.6 for why this is the
    /// documented starting value for fields `Reset()` itself doesn't touch (`active_weapon`,
    /// `colliding`, `left_wall`, `move_restrictions`, `id`, `reset`).
    fn default() -> Self {
        CharacterCore {
            pos: Vec2::zero(),
            vel: Vec2::zero(),
            hook_pos: Vec2::zero(),
            hook_dir: Vec2::zero(),
            hook_tele_base: Vec2::zero(),
            hook_tick: 0,
            hook_state: 0,
            attached_players: AttachedPlayers::default(),
            hooked_player: 0,
            active_weapon: 0,
            new_hook: false,
            jumped: 0,
            jumped_total: 0,
            jumps: 0,
            direction: 0,
            angle: 0,
            input: PlayerInput::default(),
            triggered_events: 0,
            id: 0,
            reset: false,
            colliding: 0,
            left_wall: false,
            solo: false,
            jetpack: false,
            collision_disabled: false,
            endless_hook: false,
            endless_jump: false,
            hammer_hit_disabled: false,
            grenade_hit_disabled: false,
            laser_hit_disabled: false,
            shotgun_hit_disabled: false,
            hook_hit_disabled: false,
            is_super: false,
            invincible: false,
            weapons: [WeaponStat::default(); NUM_WEAPONS],
            ninja: NinjaState::default(),
            has_telegun_gun: false,
            has_telegun_grenade: false,
            has_telegun_laser: false,
            freeze_start: 0,
            freeze_end: 0,
            is_in_freeze: false,
            deep_frozen: false,
            live_frozen: false,
            tuning: TuningParams::default(),
            move_restrictions: 0,
        }
    }
}

impl<R: Real> CharacterCore<R> {
    /// `CCharacterCore::HookedPlayer()`.
    pub fn hooked_player(&self) -> i32 {
        self.hooked_player
    }

    /// `CCharacterCore::m_MoveRestrictions`'s current value — computed once per tick by [`tick`]
    /// alone (`gamecore.cpp:197`, `m_Pos` alone, no `MapIndex` override). **Not** the same field
    /// as `CCharacter::m_MoveRestrictions` (`character.h:136`,
    /// [`crate::world::Character::move_restrictions`]), which `HandleTiles` recomputes with a
    /// `MapIndex` override every tile its anti-skip loop visits — see that field's doc comment
    /// (task 1.6, root-caused against an instrumented Oracle B binary: this crate's
    /// `handle_tiles` used to overwrite *this* field, conflating the two). This core-level field
    /// is read only by [`tick_deferred`]'s hook-drag `ClampVel` calls (`gamecore.cpp:517,521`)
    /// and by the parity trace comparison (which dumps exactly this field,
    /// `oracle_server.cpp:2375`, `pChar->m_Core.m_MoveRestrictions`) — every *other* DDRace-level
    /// consumer (`TakeDamage`, `ApplyMoveRestrictions`, speedup tiles, the stopper jump-reset,
    /// the hammer-hit target clamp) must use the character-level field instead.
    pub fn move_restrictions(&self) -> i32 {
        self.move_restrictions
    }

    /// `CCharacterCore::Init(CWorldCore*, CCollision*, CTeamsCore*)`: in the C++ source this
    /// stores the three pointers (unnecessary here — see the module doc comment — `tick`/
    /// `tick_deferred`/`move_character`/[`quantize`] take the equivalent state as explicit
    /// parameters instead) and sets `m_Id = -1`; only that last, durable side effect applies.
    /// Callers set the real id afterward (matching every real call site, which does the same).
    pub fn init(&mut self) {
        self.id = -1;
    }

    /// `CCharacterCore::Reset()`. Does *not* set `active_weapon`/`colliding`/`left_wall`/
    /// `move_restrictions`/`id` — matches the C++ source exactly (see `docs/formats.md` §5.6);
    /// keeps `tuning`/`input` untouched too (`Reset()` doesn't touch either in the C++ source).
    pub fn reset(&mut self) {
        self.pos = Vec2::zero();
        self.vel = Vec2::zero();
        self.new_hook = false;
        self.hook_pos = Vec2::zero();
        self.hook_dir = Vec2::zero();
        self.hook_tele_base = Vec2::zero();
        self.hook_tick = 0;
        self.hook_state = HOOK_IDLE;
        self.set_hooked_player_self_only(-1);
        self.attached_players = AttachedPlayers::default();
        self.jumped = 0;
        self.jumped_total = 0;
        self.jumps = 2;
        self.triggered_events = 0;

        self.solo = false;
        self.jetpack = false;
        self.collision_disabled = false;
        self.endless_hook = false;
        self.endless_jump = false;
        self.hammer_hit_disabled = false;
        self.grenade_hit_disabled = false;
        self.laser_hit_disabled = false;
        self.shotgun_hit_disabled = false;
        self.hook_hit_disabled = false;
        self.is_super = false;
        self.invincible = false;
        self.has_telegun_gun = false;
        self.has_telegun_grenade = false;
        self.has_telegun_laser = false;
        self.freeze_start = 0;
        self.freeze_end = 0;
        self.is_in_freeze = false;
        self.deep_frozen = false;
        self.live_frozen = false;

        // "never initialize both to 0"
        self.input.target_x = 0;
        self.input.target_y = -1;
    }

    /// Sets `hooked_player` without touching any sibling's `attached_players` — used only by
    /// [`CharacterCore::reset`] (a character being reset has no siblings that could have it
    /// hooked yet in any scenario this crate builds) and by [`quantize`] (which reads back the
    /// exact value it just wrote, so the "did it change" guard `SetHookedPlayer` normally applies
    /// is trivially false — see [`quantize`]'s doc comment). Any other caller wanting the full
    /// `SetHookedPlayer` behavior (updating a sibling's `attached_players`) must use
    /// [`set_hooked_player`] instead, which has `WorldCore` access.
    fn set_hooked_player_self_only(&mut self, hooked_player: i32) {
        self.hooked_player = hooked_player;
    }

    /// `CCharacterCore::Write(CNetObj_CharacterCore*)`.
    pub fn write(&self) -> NetCharacterCore {
        NetCharacterCore {
            x: vmath::round_to_int(self.pos.x),
            y: vmath::round_to_int(self.pos.y),
            vel_x: vmath::round_to_int(self.vel.x * R::from_i32(256)),
            vel_y: vmath::round_to_int(self.vel.y * R::from_i32(256)),
            angle: self.angle,
            direction: self.direction,
            jumped: self.jumped,
            hooked_player: self.hooked_player,
            hook_state: self.hook_state,
            hook_tick: self.hook_tick,
            hook_x: vmath::round_to_int(self.hook_pos.x),
            hook_y: vmath::round_to_int(self.hook_pos.y),
            hook_dx: vmath::round_to_int(self.hook_dir.x * R::from_i32(256)),
            hook_dy: vmath::round_to_int(self.hook_dir.y * R::from_i32(256)),
        }
    }

    /// `CCharacterCore::Read(const CNetObj_CharacterCore*)`. Sets `hooked_player` directly,
    /// bypassing `SetHookedPlayer`'s sibling-`attached_players` bookkeeping — see
    /// [`quantize`]'s doc comment for why that's exactly right for this crate's only caller.
    pub fn read(&mut self, net: &NetCharacterCore) {
        self.pos = Vec2::new(R::from_i32(net.x), R::from_i32(net.y));
        self.vel = Vec2::new(R::from_i32(net.vel_x), R::from_i32(net.vel_y)) / R::from_i32(256);
        self.hook_state = net.hook_state;
        self.hook_tick = net.hook_tick;
        self.hook_pos = Vec2::new(R::from_i32(net.hook_x), R::from_i32(net.hook_y));
        self.hook_dir = Vec2::new(R::from_i32(net.hook_dx), R::from_i32(net.hook_dy)) / R::from_i32(256);
        self.set_hooked_player_self_only(net.hooked_player);
        self.jumped = net.jumped;
        self.direction = net.direction;
        self.angle = net.angle;
    }

    /// `CCharacterCore::ReadDDNet(const CNetObj_DDNetCharacter*)`. Not called by Oracle A/
    /// `core_world` (there is no network snapshot in a core-only harness — see the task spec),
    /// but ported for structural completeness (review round 2, finding F2) and for task 1.6+,
    /// which will call it from a real snapshot round trip.
    pub fn read_ddnet(&mut self, net: &NetDDNetCharacter) {
        let flag = |bit: i32| (net.flags & bit) != 0;

        // Collision
        self.solo = flag(CHARACTERFLAG_SOLO);
        self.jetpack = flag(CHARACTERFLAG_JETPACK);
        self.collision_disabled = flag(CHARACTERFLAG_COLLISION_DISABLED);
        self.hammer_hit_disabled = flag(CHARACTERFLAG_HAMMER_HIT_DISABLED);
        self.shotgun_hit_disabled = flag(CHARACTERFLAG_SHOTGUN_HIT_DISABLED);
        self.grenade_hit_disabled = flag(CHARACTERFLAG_GRENADE_HIT_DISABLED);
        self.laser_hit_disabled = flag(CHARACTERFLAG_LASER_HIT_DISABLED);
        self.hook_hit_disabled = flag(CHARACTERFLAG_HOOK_HIT_DISABLED);
        self.is_super = flag(CHARACTERFLAG_SUPER);
        self.invincible = flag(CHARACTERFLAG_INVINCIBLE);

        // Endless
        self.endless_hook = flag(CHARACTERFLAG_ENDLESS_HOOK);
        self.endless_jump = flag(CHARACTERFLAG_ENDLESS_JUMP);

        // Freeze
        self.freeze_end = net.freeze_end;
        self.deep_frozen = net.freeze_end == -1;
        self.live_frozen = flag(CHARACTERFLAG_MOVEMENTS_DISABLED);

        // Telegun
        self.has_telegun_grenade = flag(CHARACTERFLAG_TELEGUN_GRENADE);
        self.has_telegun_gun = flag(CHARACTERFLAG_TELEGUN_GUN);
        self.has_telegun_laser = flag(CHARACTERFLAG_TELEGUN_LASER);

        // Weapons
        self.weapons[WEAPON_HAMMER as usize].got = flag(CHARACTERFLAG_WEAPON_HAMMER);
        self.weapons[WEAPON_GUN as usize].got = flag(CHARACTERFLAG_WEAPON_GUN);
        self.weapons[WEAPON_SHOTGUN as usize].got = flag(CHARACTERFLAG_WEAPON_SHOTGUN);
        self.weapons[WEAPON_GRENADE as usize].got = flag(CHARACTERFLAG_WEAPON_GRENADE);
        self.weapons[WEAPON_LASER as usize].got = flag(CHARACTERFLAG_WEAPON_LASER);
        self.weapons[WEAPON_NINJA as usize].got = flag(CHARACTERFLAG_WEAPON_NINJA);

        // Available jumps
        self.jumps = net.jumps;

        // Display information: only accepted when actually received (not `-1`).
        if net.jumped_total != -1 {
            self.jumped_total = net.jumped_total;
        }
        if net.ninja_activation_tick != -1 {
            self.ninja.activation_tick = net.ninja_activation_tick;
        }
        if net.freeze_start != -1 {
            self.freeze_start = net.freeze_start;
            self.is_in_freeze = flag(CHARACTERFLAG_IN_FREEZE);
        }
    }
}

// --- `CHARACTERFLAG_*` (`generated/protocol.h`, from `datasrc/network.py`'s `CharacterFlags`) —
// only the bits `read_ddnet` above actually reads. ------------------------------------------

/// `CHARACTERFLAG_SOLO`.
pub const CHARACTERFLAG_SOLO: i32 = 1 << 0;
/// `CHARACTERFLAG_JETPACK`.
pub const CHARACTERFLAG_JETPACK: i32 = 1 << 1;
/// `CHARACTERFLAG_COLLISION_DISABLED`.
pub const CHARACTERFLAG_COLLISION_DISABLED: i32 = 1 << 2;
/// `CHARACTERFLAG_ENDLESS_HOOK`.
pub const CHARACTERFLAG_ENDLESS_HOOK: i32 = 1 << 3;
/// `CHARACTERFLAG_ENDLESS_JUMP`.
pub const CHARACTERFLAG_ENDLESS_JUMP: i32 = 1 << 4;
/// `CHARACTERFLAG_SUPER`.
pub const CHARACTERFLAG_SUPER: i32 = 1 << 5;
/// `CHARACTERFLAG_HAMMER_HIT_DISABLED`.
pub const CHARACTERFLAG_HAMMER_HIT_DISABLED: i32 = 1 << 6;
/// `CHARACTERFLAG_SHOTGUN_HIT_DISABLED`.
pub const CHARACTERFLAG_SHOTGUN_HIT_DISABLED: i32 = 1 << 7;
/// `CHARACTERFLAG_GRENADE_HIT_DISABLED`.
pub const CHARACTERFLAG_GRENADE_HIT_DISABLED: i32 = 1 << 8;
/// `CHARACTERFLAG_LASER_HIT_DISABLED`.
pub const CHARACTERFLAG_LASER_HIT_DISABLED: i32 = 1 << 9;
/// `CHARACTERFLAG_HOOK_HIT_DISABLED`.
pub const CHARACTERFLAG_HOOK_HIT_DISABLED: i32 = 1 << 10;
/// `CHARACTERFLAG_TELEGUN_GUN`.
pub const CHARACTERFLAG_TELEGUN_GUN: i32 = 1 << 11;
/// `CHARACTERFLAG_TELEGUN_GRENADE`.
pub const CHARACTERFLAG_TELEGUN_GRENADE: i32 = 1 << 12;
/// `CHARACTERFLAG_TELEGUN_LASER`.
pub const CHARACTERFLAG_TELEGUN_LASER: i32 = 1 << 13;
/// `CHARACTERFLAG_WEAPON_HAMMER`.
pub const CHARACTERFLAG_WEAPON_HAMMER: i32 = 1 << 14;
/// `CHARACTERFLAG_WEAPON_GUN`.
pub const CHARACTERFLAG_WEAPON_GUN: i32 = 1 << 15;
/// `CHARACTERFLAG_WEAPON_SHOTGUN`.
pub const CHARACTERFLAG_WEAPON_SHOTGUN: i32 = 1 << 16;
/// `CHARACTERFLAG_WEAPON_GRENADE`.
pub const CHARACTERFLAG_WEAPON_GRENADE: i32 = 1 << 17;
/// `CHARACTERFLAG_WEAPON_LASER`.
pub const CHARACTERFLAG_WEAPON_LASER: i32 = 1 << 18;
/// `CHARACTERFLAG_WEAPON_NINJA`.
pub const CHARACTERFLAG_WEAPON_NINJA: i32 = 1 << 19;
/// `CHARACTERFLAG_MOVEMENTS_DISABLED`.
pub const CHARACTERFLAG_MOVEMENTS_DISABLED: i32 = 1 << 20;
/// `CHARACTERFLAG_IN_FREEZE`.
pub const CHARACTERFLAG_IN_FREEZE: i32 = 1 << 21;
/// `CHARACTERFLAG_INVINCIBLE`.
pub const CHARACTERFLAG_INVINCIBLE: i32 = 1 << 25;

/// `CNetObj_DDNetCharacter` (`generated/protocol.h`), the subset [`CharacterCore::read_ddnet`]
/// reads (`m_TeleCheckpoint`/`m_StrongWeakId`/`m_TargetX`/`m_TargetY`/`m_TuneZoneOverride` exist
/// on the real network object but are never read by `ReadDDNet`, so they're omitted here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetDDNetCharacter {
    /// `m_Flags` (`CHARACTERFLAG_*` bits).
    pub flags: i32,
    /// `m_FreezeEnd`.
    pub freeze_end: i32,
    /// `m_Jumps`.
    pub jumps: i32,
    /// `m_JumpedTotal` (`-1` = not sent).
    pub jumped_total: i32,
    /// `m_NinjaActivationTick` (`-1` = not sent).
    pub ninja_activation_tick: i32,
    /// `m_FreezeStart` (`-1` = not sent).
    pub freeze_start: i32,
}

/// [`crate::vmath::Vec2`]'s `/` isn't implemented (see that module) — `Read`/`quantize` need
/// vector-over-scalar division for velocity/hook-direction unquantization, spelled out via `Div`
/// here rather than adding a general `Vec2 / R` operator only one call site uses.
impl<R: Real> std::ops::Div<R> for Vec2<R> {
    type Output = Vec2<R>;
    fn div(self, s: R) -> Vec2<R> {
        Vec2::new(self.x / s, self.y / s)
    }
}

/// `CCharacterCore::Quantize()` = `Write()` then `Read()`. Bypasses `SetHookedPlayer`'s
/// sibling-bookkeeping (see [`CharacterCore::read`]): `Quantize` always reads back the exact
/// `hooked_player` it just wrote, so `SetHookedPlayer`'s "did it change" guard is always false
/// here — calling the full sibling-aware version would be correct too, just pure overhead every
/// single tick for an update that can never do anything.
pub fn quantize<R: Real>(core: &mut CharacterCore<R>) {
    let net = core.write();
    core.read(&net);
}

/// `CTeamsCore` (`teamscore.{h,cpp}`), unchanged in spirit: fixed [`MAX_CLIENTS`]-sized arrays,
/// no heap. `g_Config.m_SvTeam == SV_TEAM_FORCED_SOLO` is never modeled (Oracle A's team default
/// is `TEAM_FLOCK` for everyone, see `docs/formats.md` §5.4) — [`TeamsCore::default`] matches
/// that (never the forced-solo per-client-id assignment `CTeamsCore::Reset` does when
/// `sv_team == SV_TEAM_FORCED_SOLO`).
#[derive(Debug, Clone, Copy)]
pub struct TeamsCore {
    team: [i32; MAX_CLIENTS],
    solo: [bool; MAX_CLIENTS],
}

/// `TEAM_FLOCK` (`teamscore.h`).
pub const TEAM_FLOCK: i32 = 0;
/// `TEAM_SUPER` (`teamscore.h`): `MAX_CLIENTS`.
pub const TEAM_SUPER: i32 = MAX_CLIENTS as i32;
/// `NUM_DDRACE_TEAMS` (`teamscore.h`): `TEAM_SUPER + 1`.
pub const NUM_DDRACE_TEAMS: i32 = TEAM_SUPER + 1;

impl Default for TeamsCore {
    fn default() -> Self {
        TeamsCore {
            team: [TEAM_FLOCK; MAX_CLIENTS],
            solo: [false; MAX_CLIENTS],
        }
    }
}

impl TeamsCore {
    /// `CTeamsCore()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `CTeamsCore::TeamSuper()`: `m_NumDDRaceTeams - 1`, always [`TEAM_SUPER`] for a freshly
    /// reset instance (this crate never changes `m_NumDDRaceTeams`, matching every real DDNet
    /// server too — it's set once in `Reset()` and never mutated elsewhere in `teamscore.cpp`).
    pub fn team_super(&self) -> i32 {
        TEAM_SUPER
    }

    /// `CTeamsCore::Team(int ClientId)`.
    ///
    /// # Panics
    ///
    /// If `client_id` isn't `0..MAX_CLIENTS` (C++'s `m_aTeam[ClientId]` has no such check — an
    /// out-of-range id there is undefined behavior; review round 2, finding F3 specifically
    /// flagged a client id of `-1` reaching this method via a stale/mismatched
    /// [`CharacterCore::id`] and panicking on the raw array index with no clear message — this
    /// gives that same "never happens for a well-formed caller" precondition a clear panic
    /// instead of a confusing one, while [`WorldCore::from_characters`]/[`WorldCore::insert`]
    /// now make the specific F3 scenario impossible by construction).
    pub fn team(&self, client_id: i32) -> i32 {
        assert!(
            (0..MAX_CLIENTS as i32).contains(&client_id),
            "invalid client id {client_id}"
        );
        self.team[client_id as usize]
    }

    /// `CTeamsCore::Team(int ClientId, int Team)`.
    pub fn set_team(&mut self, client_id: i32, team: i32) {
        assert!((TEAM_FLOCK..NUM_DDRACE_TEAMS).contains(&team), "invalid team {team}");
        self.team[client_id as usize] = team;
    }

    /// `CTeamsCore::SetSolo(int ClientId, bool Value)`.
    pub fn set_solo(&mut self, client_id: i32, value: bool) {
        self.solo[client_id as usize] = value;
    }

    /// `CTeamsCore::GetSolo(int ClientId)`.
    pub fn get_solo(&self, client_id: i32) -> bool {
        if !(0..MAX_CLIENTS as i32).contains(&client_id) {
            return false;
        }
        self.solo[client_id as usize]
    }

    /// `CTeamsCore::SameTeam`.
    pub fn same_team(&self, id1: i32, id2: i32) -> bool {
        self.team(id1) == self.team_super() || self.team(id2) == self.team_super() || self.team(id1) == self.team(id2)
    }

    /// `CTeamsCore::CanKeepHook`.
    pub fn can_keep_hook(&self, id1: i32, id2: i32) -> bool {
        if self.team(id1) == self.team_super() || self.team(id2) == self.team_super() || id1 == id2 {
            return true;
        }
        self.team(id1) == self.team(id2)
    }

    /// `CTeamsCore::CanCollide`.
    pub fn can_collide(&self, id1: i32, id2: i32) -> bool {
        if self.team(id1) == self.team_super() || self.team(id2) == self.team_super() || id1 == id2 {
            return true;
        }
        if self.get_solo(id1) || self.get_solo(id2) {
            return false;
        }
        self.team(id1) == self.team(id2)
    }
}

/// `SSwitchers` (`gamecore.h`): per-switch-number door/timer state. A placeholder for task 1.7
/// (switch-state selection, out of scope here) — [`WorldCore::switchers`] stays empty for every
/// scenario this crate's `core_world` builds (Oracle A never calls `InitSwitchers` either; see
/// `docs/formats.md` §5.4), so nothing in `tick`/`tick_deferred`/`move_character` reads it yet.
#[derive(Debug, Clone)]
pub struct Switcher {
    /// `m_aStatus`: open/closed, per team.
    pub status: [bool; NUM_DDRACE_TEAMS as usize],
    /// `m_Initial`.
    pub initial: bool,
    /// `m_aEndTick`: tick a timed open/close reverts, per team.
    pub end_tick: [i32; NUM_DDRACE_TEAMS as usize],
    /// `m_aType`: the switch control type currently in effect, per team.
    pub kind: [i32; NUM_DDRACE_TEAMS as usize],
    /// `m_aLastUpdateTick`, per team.
    pub last_update_tick: [i32; NUM_DDRACE_TEAMS as usize],
}

impl Default for Switcher {
    fn default() -> Self {
        Switcher {
            status: [false; NUM_DDRACE_TEAMS as usize],
            initial: false,
            end_tick: [0; NUM_DDRACE_TEAMS as usize],
            kind: [0; NUM_DDRACE_TEAMS as usize],
            last_update_tick: [0; NUM_DDRACE_TEAMS as usize],
        }
    }
}

/// Port of 20.1 `CWorldCore` (`gamecore.h`): [`Prng`] (`m_pPrng`, `None` unless a caller seeds
/// one — Oracle A always runs with `None`, see `docs/formats.md` §5.4) and [`Switcher`]s
/// (`m_vSwitchers`), plus the character roster `m_apCharacters` addresses.
///
/// **`m_apCharacters` is not a literal `[Option<CharacterCore<R>>; MAX_CLIENTS]`.** In C++ it is
/// an array of *pointers* (`class CCharacterCore *m_apCharacters[MAX_CLIENTS]`, 128 × 8 bytes on
/// a 64-bit build) into `CCharacterCore` objects that live elsewhere (owned by each `CCharacter`
/// entity) — `CWorldCore` itself never owns 128 copies of the actual (~300+-byte) character
/// state. This crate has no raw pointers to borrow that way, so `WorldCore` instead *owns* up to
/// `CAP` characters directly, compactly (slots `0..len`, no gaps), kept **sorted by ascending
/// client id** at all times — which is exactly the order `CWorldCore::m_apCharacters[0..
/// MAX_CLIENTS]` is iterated in (ascending array index = ascending client id, skipping absent
/// slots), so every order-dependent effect (which sibling's `TickDeferred` sees which other
/// sibling's velocity first, tie-breaks in the closest-hook-target search, ...) matches C++
/// bit-for-bit despite the different underlying representation. `CAP` is chosen by the caller
/// (small — the performance criterion's benchmark uses `CAP = 2`, `core_world`'s parity tests use
/// something comfortably above the generator's 1..=8 characters) rather than fixed at
/// [`MAX_CLIENTS`], so cloning/iterating a small world stays cheap regardless of how large a
/// valid client id (`0..MAX_CLIENTS`) happens to be — the id *range* `CAP` must support is
/// unrelated to how many characters are actually present (see `docs/formats.md`'s note on
/// scenario ids not needing to be small or contiguous).
///
/// `Clone` is implemented by hand (not derived) purely to override `clone_from` (task 1.10,
/// acceptance criterion 2's "save/restore without cloning Vecs"): `ids`/`cores`/`len` are plain
/// `Copy` data (no heap involved either way), but `switchers` is a `Vec` — `clone_from`'s default
/// trait-level implementation (`*self = source.clone()`) would still drop this `WorldCore`'s own
/// `switchers` allocation and replace it with a freshly allocated one every call; delegating to
/// `Vec::clone_from` instead reuses the destination's existing allocation whenever its capacity
/// already fits (the common case: a search loop's `World::restore_from` calling this repeatedly
/// against the same, already-grown `World`). `clone()` itself is unchanged from what `#[derive]`
/// would generate.
#[derive(Debug)]
pub struct WorldCore<R: Real, const CAP: usize> {
    ids: [u8; CAP],
    cores: [CharacterCore<R>; CAP],
    len: usize,
    /// `m_pPrng`: `None` (matching a null pointer) unless a caller seeds one. Always `None` for
    /// every scenario Oracle A/`core_world` builds (see [`WorldCore::random_or_0`]).
    pub prng: Option<Prng>,
    /// `m_vSwitchers`: empty unless a caller calls [`WorldCore::init_switchers`] (task 1.7).
    pub switchers: Vec<Switcher>,
}

impl<R: Real, const CAP: usize> Default for WorldCore<R, CAP> {
    fn default() -> Self {
        WorldCore {
            ids: [0; CAP],
            cores: std::array::from_fn(|_| CharacterCore::default()),
            len: 0,
            prng: None,
            switchers: Vec::new(),
        }
    }
}

/// See the struct doc comment: hand-written only to override `clone_from`.
impl<R: Real, const CAP: usize> Clone for WorldCore<R, CAP> {
    fn clone(&self) -> Self {
        WorldCore {
            ids: self.ids,
            cores: self.cores,
            len: self.len,
            prng: self.prng.clone(),
            switchers: self.switchers.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        // Only `ids[..source.len]`/`cores[..source.len]` are ever read by anything (the struct
        // doc comment: "compactly (slots 0..len, no gaps)" — every accessor above bounds its own
        // reads by `self.len`), so copying only that prefix (typically 2-8 characters, not
        // `CAP` — up to `MAX_CLIENTS` = 128) still reproduces every *observable* bit of `source`
        // once `self.len` is set to match; the stale tail past `source.len` is exactly as
        // unreachable in `self` afterward as it already was in `source`. Task 1.10, acceptance
        // criterion 2 ("save/restore without cloning Vecs" — the same idea applied to a fixed-size
        // array instead of a `Vec`): measured effect on a `World::restore_from` this feeds into,
        // this task's `BUILD REPORT`.
        let len = source.len;
        self.ids[..len].copy_from_slice(&source.ids[..len]);
        self.cores[..len].copy_from_slice(&source.cores[..len]);
        self.len = len;
        self.prng.clone_from(&source.prng);
        self.switchers.clone_from(&source.switchers);
    }
}

impl<R: Real, const CAP: usize> WorldCore<R, CAP> {
    /// An empty world (no characters, no PRNG, no switchers).
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a world from `entries` (client id, initial core), stored sorted by ascending
    /// client id (see the struct doc comment).
    ///
    /// # Panics
    ///
    /// If `entries.len() > CAP`, any id is `>= MAX_CLIENTS`, or two entries share an id.
    pub fn from_characters(entries: &[(u8, CharacterCore<R>)]) -> Self {
        assert!(
            entries.len() <= CAP,
            "too many characters ({}) for this WorldCore's capacity ({CAP})",
            entries.len()
        );
        let mut sorted: Vec<(u8, CharacterCore<R>)> = entries.to_vec();
        sorted.sort_by_key(|(id, _)| *id);
        for w in sorted.windows(2) {
            assert_ne!(w[0].0, w[1].0, "duplicate client id {}", w[0].0);
        }
        for &(id, _) in &sorted {
            assert!((id as usize) < MAX_CLIENTS, "client id {id} must be < MAX_CLIENTS");
        }
        let mut world = Self::new();
        world.len = sorted.len();
        for (slot, (id, mut core)) in sorted.into_iter().enumerate() {
            // Review round 2, finding F3: `id` (this array's own key/index) and `core.id`
            // (`m_Id`, read independently by e.g. `tick`'s `HOOK_GRABBED` handling, matching
            // `pCharCore->m_Id` in `gamecore.cpp`) must never be allowed to disagree — a caller
            // passing a `core` whose `id` field doesn't match its key would otherwise silently
            // create exactly that inconsistency (and, downstream, an out-of-bounds `TeamsCore`
            // index if the mismatched value were `-1` or `>= MAX_CLIENTS`). Setting it here,
            // unconditionally, mirrors every real call site in `gamecore.cpp`/`oracle_core.cpp`
            // (`Core.m_Id = Id;`, set once, right after `Reset()`/`Init()`, from the exact same
            // id the character is being registered under) and makes the two impossible to
            // desync at construction time.
            core.id = id as i32;
            world.ids[slot] = id;
            world.cores[slot] = core;
        }
        world
    }

    /// Inserts a new character at client id `id`, keeping slots sorted by ascending id (see the
    /// struct doc comment) — not called by any parity test in this task (Oracle A never spawns
    /// characters mid-scenario), provided for task 1.6's full `World` (spawn/respawn). Forces
    /// `core.id = id` for the same reason [`WorldCore::from_characters`] does.
    ///
    /// # Panics
    ///
    /// If `id >= MAX_CLIENTS`, `id` is already present, or this world is already at its `CAP`
    /// capacity.
    pub fn insert(&mut self, id: u8, mut core: CharacterCore<R>) {
        assert!((id as usize) < MAX_CLIENTS, "client id {id} must be < MAX_CLIENTS");
        assert!(self.len < CAP, "WorldCore is at capacity ({CAP})");
        let pos = self.ids[..self.len].partition_point(|&x| x < id);
        assert!(pos == self.len || self.ids[pos] != id, "duplicate client id {id}");
        core.id = id as i32;
        for i in (pos..self.len).rev() {
            self.ids[i + 1] = self.ids[i];
            self.cores[i + 1] = self.cores[i];
        }
        self.ids[pos] = id;
        self.cores[pos] = core;
        self.len += 1;
    }

    /// Removes the character with client id `id`, if present, keeping the remaining slots
    /// sorted by ascending id. Returns the removed character.
    pub fn remove(&mut self, id: u8) -> Option<CharacterCore<R>> {
        let pos = self.slot_of(id)?;
        let removed = self.cores[pos];
        for i in pos..self.len - 1 {
            self.ids[i] = self.ids[i + 1];
            self.cores[i] = self.cores[i + 1];
        }
        self.len -= 1;
        Some(removed)
    }

    /// Number of characters currently in this world.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether this world has no characters.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The client id stored at compact slot `slot` (`0..self.len()`).
    pub fn id_at(&self, slot: usize) -> u8 {
        self.ids[slot]
    }

    /// Finds which compact slot (if any) holds client id `id` — `O(log len)` (binary search,
    /// since slots are kept sorted by id).
    pub fn slot_of(&self, id: u8) -> Option<usize> {
        self.ids[..self.len].binary_search(&id).ok()
    }

    /// The character with client id `id`, if present.
    pub fn get(&self, id: u8) -> Option<&CharacterCore<R>> {
        self.slot_of(id).map(|s| &self.cores[s])
    }

    /// Mutable version of [`WorldCore::get`].
    pub fn get_mut(&mut self, id: u8) -> Option<&mut CharacterCore<R>> {
        let slot = self.slot_of(id)?;
        Some(&mut self.cores[slot])
    }

    /// The character at compact slot `slot` (`0..self.len()`; see [`WorldCore::slot_of`]).
    pub fn core_at(&self, slot: usize) -> &CharacterCore<R> {
        &self.cores[slot]
    }

    /// Mutable version of [`WorldCore::core_at`].
    pub fn core_at_mut(&mut self, slot: usize) -> &mut CharacterCore<R> {
        &mut self.cores[slot]
    }

    /// `(client_id, core)` pairs, in ascending client-id order.
    pub fn iter(&self) -> impl Iterator<Item = (u8, &CharacterCore<R>)> {
        self.ids[..self.len].iter().copied().zip(self.cores[..self.len].iter())
    }

    /// `CWorldCore::RandomOr0(int BelowThis)`.
    pub fn random_or_0(&mut self, below_this: i32) -> i32 {
        if below_this <= 1 {
            return 0;
        }
        match &mut self.prng {
            None => 0,
            Some(p) => (p.random_bits() % (below_this as u32)) as i32,
        }
    }

    /// `CWorldCore::InitSwitchers(int HighestSwitchNumber)`.
    pub fn init_switchers(&mut self, highest_switch_number: i32) {
        if highest_switch_number > 0 {
            self.switchers = (0..=highest_switch_number).map(|_| Switcher::default()).collect();
        } else {
            self.switchers.clear();
        }
        for switcher in &mut self.switchers {
            switcher.initial = true;
            for j in 0..NUM_DDRACE_TEAMS as usize {
                switcher.status[j] = true;
                switcher.end_tick[j] = 0;
                switcher.kind[j] = 0;
                switcher.last_update_tick[j] = 0;
            }
        }
    }
}

/// `CCharacterCore::SetHookedPlayer(int HookedPlayer)`, generalized to operate on `me` (a local
/// copy extracted from `world`, per the module doc comment) instead of `this` — updates the
/// *previous* and *new* hooked player's `attached_players` in `world` (only for ids that are
/// actually present — matches the C++ null-pointer checks), then `me.hooked_player`. `pub`
/// (task 1.6): `CCharacter::ReleaseHook`/`ResetHook` (`character.cpp:766-777`) call this directly
/// on a character's own core from outside this module, exactly like every call site here does.
pub fn set_hooked_player<R: Real, const CAP: usize>(
    world: &mut WorldCore<R, CAP>,
    me: &mut CharacterCore<R>,
    self_id: u8,
    hooked_player: i32,
) {
    let old = me.hooked_player;
    if hooked_player == old {
        return;
    }
    if old != -1
        && let Some(slot) = world.slot_of(old as u8)
    {
        world.cores[slot].attached_players.remove(i32::from(self_id));
    }
    if hooked_player != -1
        && let Some(slot) = world.slot_of(hooked_player as u8)
    {
        world.cores[slot].attached_players.insert(i32::from(self_id));
    }
    me.hooked_player = hooked_player;
}

/// `CCharacterCore::Tick(bool UseInput, bool DoDeferredTick)`. `self_slot` identifies which
/// character (by its compact slot in `world`) is ticking; see the module doc comment for the
/// extract/mutate-others/write-back pattern this and its sibling functions use in place of a
/// `this->m_pWorld` pointer.
///
/// The `pfnSwitchActive` callback `GetMoveRestrictions` takes mirrors `CCharacterCore::
/// IsSwitchActiveCb` (`gamecore.cpp:740-747`) exactly: `false` whenever `world.switchers` is
/// empty, `id == -1`, or `teams.team(id) == teams.team_super()`, else `world.switchers[Number]
/// .status[team]`. Oracle A/task 1.3's parity tests always have empty `world.switchers` (switch
/// tiles are out of Oracle A's core-only scope — `docs/formats.md` §5.5), so for every one of
/// them this is provably identical to always passing `None` (the callback can never even be
/// invoked — `get_move_restrictions` only calls it when a door tile's number is `<=
/// m_HighestSwitchNumber`, and an empty-switchers collision never reports a nonzero highest
/// switch number since no switch layer was ever loaded); task 1.6's `World` populates real
/// switchers, where this now matters (a character standing in a closed door's `TILE_STOPA`
/// cell must have `CANTMOVE_*` bits set exactly when the real server's callback would report the
/// door active for that character's team).
pub fn tick<R: Real, const CAP: usize>(
    world: &mut WorldCore<R, CAP>,
    self_slot: usize,
    collision: &Collision<R>,
    teams: &TeamsCore,
    use_input: bool,
    do_deferred_tick: bool,
) {
    let self_id = world.ids[self_slot];
    let mut me = world.cores[self_slot];

    me.move_restrictions = if world.switchers.is_empty() {
        collision.get_move_restrictions_simple(me.pos, R::from_i32(18))
    } else {
        let self_team = if me.id != -1 { teams.team(me.id) } else { -1 };
        let team_super = teams.team_super();
        let switchers = &world.switchers;
        collision.get_move_restrictions(
            Some(|number: u8| {
                self_team != -1
                    && self_team != team_super
                    && (number as usize) < switchers.len()
                    && switchers[number as usize].status[self_team as usize]
            }),
            me.pos,
            R::from_i32(18),
            None,
        )
    };
    me.triggered_events = 0;

    let grounded = collision.is_on_ground(me.pos, physical_size());
    let target_direction = vmath::normalize(Vec2::new(
        R::from_i32(me.input.target_x),
        R::from_i32(me.input.target_y),
    ));

    me.vel.y += me.tuning.gravity::<R>();

    let max_speed = if grounded {
        me.tuning.ground_control_speed::<R>()
    } else {
        me.tuning.air_control_speed::<R>()
    };
    let accel = if grounded {
        me.tuning.ground_control_accel::<R>()
    } else {
        me.tuning.air_control_accel::<R>()
    };
    let friction = if grounded {
        me.tuning.ground_friction::<R>()
    } else {
        me.tuning.air_friction::<R>()
    };

    if use_input {
        me.direction = me.input.direction;
        me.angle = angle_from_target(me.input.target_x, me.input.target_y);

        if me.input.jump != 0 {
            if me.jumped & 1 == 0 {
                if grounded && (me.jumped & 2 == 0 || me.jumps != 0) {
                    me.triggered_events |= COREEVENT_GROUND_JUMP;
                    me.vel.y = -me.tuning.ground_jump_impulse::<R>();
                    if me.jumps > 1 {
                        me.jumped |= 1;
                    } else {
                        me.jumped |= 3;
                    }
                    me.jumped_total = 0;
                } else if me.jumped & 2 == 0 {
                    me.triggered_events |= COREEVENT_AIR_JUMP;
                    me.vel.y = -me.tuning.air_jump_impulse::<R>();
                    me.jumped |= 3;
                    me.jumped_total += 1;
                }
            }
        } else {
            me.jumped &= !1;
        }

        if me.input.hook != 0 {
            if me.hook_state == HOOK_IDLE {
                me.hook_state = HOOK_FLYING;
                me.hook_pos = me.pos + target_direction * physical_size::<R>() * R::from_f64(1.5);
                me.hook_dir = target_direction;
                set_hooked_player(world, &mut me, self_id, -1);
                me.hook_tick = (R::from_i32(SERVER_TICK_SPEED) * (R::from_f64(1.25) - me.tuning.hook_duration::<R>()))
                    .to_i32_trunc();
                me.triggered_events |= COREEVENT_HOOK_LAUNCH;
            }
        } else {
            set_hooked_player(world, &mut me, self_id, -1);
            me.hook_state = HOOK_IDLE;
            me.hook_pos = me.pos;
        }
    }

    if grounded {
        me.jumped &= !2;
        me.jumped_total = 0;
    }

    if me.direction < 0 {
        me.vel.x = saturated_add(-max_speed, max_speed, me.vel.x, -accel);
    }
    if me.direction > 0 {
        me.vel.x = saturated_add(-max_speed, max_speed, me.vel.x, accel);
    }
    if me.direction == 0 {
        me.vel.x *= friction;
    }

    if me.hook_state == HOOK_IDLE {
        set_hooked_player(world, &mut me, self_id, -1);
        me.hook_pos = me.pos;
    } else if me.hook_state >= HOOK_RETRACT_START && me.hook_state < HOOK_RETRACT_END {
        me.hook_state += 1;
    } else if me.hook_state == HOOK_RETRACT_END {
        me.triggered_events |= COREEVENT_HOOK_RETRACT;
        me.hook_state = HOOK_RETRACTED;
    } else if me.hook_state == HOOK_FLYING {
        let hook_base = if me.new_hook { me.hook_tele_base } else { me.pos };
        let mut new_pos = me.hook_pos + me.hook_dir * me.tuning.hook_fire_speed::<R>();
        if vmath::distance(hook_base, new_pos) > me.tuning.hook_length::<R>() {
            me.hook_state = HOOK_RETRACT_START;
            new_pos = hook_base + vmath::normalize(new_pos - hook_base) * me.tuning.hook_length::<R>();
            me.reset = true;
        }

        // `IntersectLineTeleHook`'s `sv_old_teleport_hook`: `false`, matching Oracle A's
        // zero-initialized `g_Config.m_SvOldTeleportHook` (see `collision`'s module doc comment).
        let hit_result = collision.intersect_line_tele_hook(me.hook_pos, new_pos, false);
        let mut going_to_hit_ground = false;
        let mut going_to_retract = false;
        let mut going_through_tele = false;
        let mut tele_nr = 0;
        if hit_result.hit != 0 {
            new_pos = hit_result.collision;
            if hit_result.hit == map::TILE_NOHOOK as i32 {
                going_to_retract = true;
            } else if hit_result.hit == map::TILE_TELEINHOOK as i32 {
                going_through_tele = true;
                tele_nr = hit_result.tele_nr;
            } else {
                going_to_hit_ground = true;
            }
            me.reset = true;
        }

        // Check against other players first (ascending client id, matching `for(int i = 0; i <
        // MAX_CLIENTS; i++)` — `world`'s slots are kept sorted by id, see `WorldCore`'s doc
        // comment).
        if !me.hook_hit_disabled
            && me.tuning.player_hooking::<R>() != R::ZERO
            && (me.hook_state == HOOK_FLYING || !me.new_hook)
        {
            let mut best_distance = R::ZERO;
            for slot in 0..world.len {
                if slot == self_slot {
                    continue;
                }
                let other_id = world.ids[slot];
                let other_pos = world.cores[slot].pos;
                let other_solo = world.cores[slot].solo;
                let other_is_super = world.cores[slot].is_super;
                if !(me.is_super || other_is_super)
                    && ((me.id != -1 && !teams.can_collide(other_id as i32, me.id)) || other_solo || me.solo)
                {
                    continue;
                }
                if let Some(closest) = vmath::closest_point_on_line(me.hook_pos, new_pos, other_pos)
                    && vmath::distance(other_pos, closest) < physical_size::<R>() + R::from_i32(2)
                    && (me.hooked_player() == -1 || vmath::distance(me.hook_pos, other_pos) < best_distance)
                {
                    me.triggered_events |= COREEVENT_HOOK_ATTACH_PLAYER;
                    me.hook_state = HOOK_GRABBED;
                    set_hooked_player(world, &mut me, self_id, other_id as i32);
                    best_distance = vmath::distance(me.hook_pos, other_pos);
                }
            }
        }

        if me.hook_state == HOOK_FLYING {
            if going_to_hit_ground {
                me.triggered_events |= COREEVENT_HOOK_ATTACH_GROUND;
                me.hook_state = HOOK_GRABBED;
            } else if going_to_retract {
                me.triggered_events |= COREEVENT_HOOK_HIT_NOHOOK;
                me.hook_state = HOOK_RETRACT_START;
            }

            let tele_outs = if going_through_tele {
                collision.tele_outs((tele_nr - 1) as u8)
            } else {
                &[]
            };
            if going_through_tele && !tele_outs.is_empty() {
                me.triggered_events = 0;
                set_hooked_player(world, &mut me, self_id, -1);
                me.new_hook = true;
                let random_out = world.random_or_0(tele_outs.len() as i32) as usize;
                me.hook_pos = tele_outs[random_out] + target_direction * physical_size::<R>() * R::from_f64(1.5);
                me.hook_dir = target_direction;
                me.hook_tele_base = me.hook_pos;
            } else {
                me.hook_pos = new_pos;
            }
        }
    }

    if me.hook_state == HOOK_GRABBED {
        if me.hooked_player() != -1 {
            match world.slot_of(me.hooked_player() as u8) {
                Some(slot) => {
                    let other_pos = world.cores[slot].pos;
                    let other_id = world.cores[slot].id;
                    if me.id != -1 && teams.can_keep_hook(me.id, other_id) {
                        me.hook_pos = other_pos;
                    } else {
                        set_hooked_player(world, &mut me, self_id, -1);
                        me.hook_state = HOOK_RETRACTED;
                        me.hook_pos = me.pos;
                    }
                }
                None => {
                    set_hooked_player(world, &mut me, self_id, -1);
                    me.hook_state = HOOK_RETRACTED;
                    me.hook_pos = me.pos;
                }
            }
        }

        if me.hooked_player() == -1 && vmath::distance(me.hook_pos, me.pos) > R::from_i32(46) {
            let mut hook_vel = vmath::normalize(me.hook_pos - me.pos) * me.tuning.hook_drag_accel::<R>();
            if hook_vel.y > R::ZERO {
                hook_vel.y *= R::from_f64(0.3);
            }
            if (hook_vel.x < R::ZERO && me.direction < 0) || (hook_vel.x > R::ZERO && me.direction > 0) {
                hook_vel.x *= R::from_f64(0.95);
            } else {
                hook_vel.x *= R::from_f64(0.75);
            }
            let new_vel = me.vel + hook_vel;
            let new_vel_length = vmath::length(new_vel);
            if new_vel_length < me.tuning.hook_drag_speed::<R>() || new_vel_length < vmath::length(me.vel) {
                me.vel = new_vel;
            }
        }

        me.hook_tick += 1;
        if me.hooked_player() != -1 {
            let hooked_exists = world.slot_of(me.hooked_player() as u8).is_some();
            if me.hook_tick > SERVER_TICK_SPEED + SERVER_TICK_SPEED / 5 || !hooked_exists {
                set_hooked_player(world, &mut me, self_id, -1);
                me.hook_state = HOOK_RETRACTED;
                me.hook_pos = me.pos;
            }
        }
    }

    if do_deferred_tick {
        tick_deferred_body(world, self_slot, teams, &mut me);
    }
    world.cores[self_slot] = me;
}

/// `CCharacterCore::TickDeferred()`.
pub fn tick_deferred<R: Real, const CAP: usize>(world: &mut WorldCore<R, CAP>, self_slot: usize, teams: &TeamsCore) {
    let mut me = world.cores[self_slot];
    tick_deferred_body(world, self_slot, teams, &mut me);
    world.cores[self_slot] = me;
}

/// `TickDeferred()`'s body, factored out so [`tick`] (when `do_deferred_tick` is `true`) can run
/// it against the *same* extracted `me` its own logic already has, instead of writing `me` back
/// and having [`tick_deferred`] extract a second copy — review round 2, finding F4 ("avoid
/// copying the whole ~460-byte `CharacterCore` in/out 3x per character-tick"): this removes one
/// of those three extract/write-back round trips for the common (non-`no_weak_hook`) path.
/// [`tick_deferred`] itself (used for `no_weak_hook`'s separate deferred pass, where nothing else
/// touches `self`'s slot between `Tick(true, false)` and `TickDeferred()`, so a fresh extract is
/// unavoidable there) still calls this the same way, just with its own freshly extracted `me`.
fn tick_deferred_body<R: Real, const CAP: usize>(
    world: &mut WorldCore<R, CAP>,
    self_slot: usize,
    teams: &TeamsCore,
    me: &mut CharacterCore<R>,
) {
    for slot in 0..world.len {
        if slot == self_slot {
            continue;
        }
        let other_id = world.ids[slot];
        let other_pos = world.cores[slot].pos;
        let other_is_super = world.cores[slot].is_super;
        let other_solo = world.cores[slot].solo;
        let other_collision_disabled = world.cores[slot].collision_disabled;

        if me.id != -1 && !teams.can_collide(me.id, other_id as i32) {
            continue;
        }
        if !(me.is_super || other_is_super) && (me.solo || other_solo) {
            continue;
        }

        let distance = vmath::distance(me.pos, other_pos);
        if distance > R::ZERO {
            let dir = vmath::normalize(me.pos - other_pos);
            let can_collide = (me.is_super || other_is_super)
                || (!me.collision_disabled
                    && !other_collision_disabled
                    && me.tuning.player_collision::<R>() != R::ZERO);

            if can_collide && distance < physical_size::<R>() * R::from_f64(1.25) {
                let a = physical_size::<R>() * R::from_f64(1.45) - distance;
                let mut velocity = R::from_f64(0.5);
                if vmath::length(me.vel) > R::from_f64(0.0001) {
                    velocity = R::ONE - (vmath::dot(vmath::normalize(me.vel), dir) + R::ONE) / R::from_i32(2);
                }
                me.vel += dir * a * (velocity * R::from_f64(0.75));
                me.vel *= R::from_f64(0.85);
            }

            if !me.hook_hit_disabled
                && me.hooked_player() == other_id as i32
                && me.tuning.player_hooking::<R>() != R::ZERO
                && distance > physical_size::<R>() * R::from_f64(1.50)
            {
                let hook_accel = me.tuning.hook_drag_accel::<R>() * (distance / me.tuning.hook_length::<R>());
                let drag_speed = me.tuning.hook_drag_speed::<R>();

                let other_vel = world.cores[slot].vel;
                let other_move_restrictions = world.cores[slot].move_restrictions;
                let temp_other = Vec2::new(
                    saturated_add(
                        -drag_speed,
                        drag_speed,
                        other_vel.x,
                        hook_accel * dir.x * R::from_f64(1.5),
                    ),
                    saturated_add(
                        -drag_speed,
                        drag_speed,
                        other_vel.y,
                        hook_accel * dir.y * R::from_f64(1.5),
                    ),
                );
                world.cores[slot].vel = collision::clamp_vel(other_move_restrictions, temp_other);

                let temp_self = Vec2::new(
                    saturated_add(
                        -drag_speed,
                        drag_speed,
                        me.vel.x,
                        -hook_accel * dir.x * R::from_f64(0.25),
                    ),
                    saturated_add(
                        -drag_speed,
                        drag_speed,
                        me.vel.y,
                        -hook_accel * dir.y * R::from_f64(0.25),
                    ),
                );
                me.vel = collision::clamp_vel(me.move_restrictions, temp_self);
            }
        }
    }

    if me.hook_state != HOOK_FLYING {
        me.new_hook = false;
    }

    if vmath::length(me.vel) > R::from_i32(6000) {
        me.vel = vmath::normalize(me.vel) * R::from_i32(6000);
    }
}

/// `CCharacterCore::Move()`.
pub fn move_character<R: Real, const CAP: usize>(
    world: &mut WorldCore<R, CAP>,
    self_slot: usize,
    collision: &Collision<R>,
    teams: &TeamsCore,
) {
    let mut me = world.cores[self_slot];

    let ramp_value = velocity_ramp(
        vmath::length(me.vel) * R::from_i32(50),
        me.tuning.velramp_start::<R>(),
        me.tuning.velramp_range::<R>(),
        me.tuning.velramp_curvature::<R>(),
    );
    me.vel.x *= ramp_value;

    let old_vel = me.vel;
    let (new_pos, new_vel, grounded) = collision.move_box(
        me.pos,
        me.vel,
        physical_size_vec2(),
        Vec2::new(
            me.tuning.ground_elasticity_x::<R>(),
            me.tuning.ground_elasticity_y::<R>(),
        ),
    );
    me.vel = new_vel;

    if grounded {
        me.jumped &= !2;
        me.jumped_total = 0;
    }

    me.colliding = 0;
    if me.vel.x < R::from_f64(0.001) && me.vel.x > R::from_f64(-0.001) {
        if old_vel.x > R::ZERO {
            me.colliding = 1;
        } else if old_vel.x < R::ZERO {
            me.colliding = 2;
        }
    } else {
        me.left_wall = true;
    }
    me.vel.x *= R::ONE / ramp_value;

    if me.is_super || (me.tuning.player_collision::<R>() != R::ZERO && !me.collision_disabled && !me.solo) {
        let distance = vmath::distance(me.pos, new_pos);
        if distance > R::ZERO {
            let end = (distance + R::ONE).to_i32_trunc();
            let mut last_pos = me.pos;
            for i in 0..end {
                let a = R::from_i32(i) / distance;
                let pos = vmath::mix(me.pos, new_pos, a);
                for slot in 0..world.len {
                    if slot == self_slot {
                        continue;
                    }
                    let other_id = world.ids[slot];
                    // Copy only the handful of `Copy` scalars this loop actually reads — NOT
                    // the whole (488-byte as of the F2 fields, `TuningParams`-carrying)
                    // `CharacterCore` — this loop runs `end` (up to `~(int)Distance`) times per
                    // `Move()` call, so a full-struct copy here was the single largest cost in
                    // this function (verified: removing it dropped `move_character` from ~300ns
                    // to ~17ns per call in a walking-speed micro-benchmark, back when the struct
                    // was 352 bytes, before review round 2's finding F2 added more fields).
                    let other_pos = world.cores[slot].pos;
                    let other_is_super = world.cores[slot].is_super;
                    let other_solo = world.cores[slot].solo;
                    let other_collision_disabled = world.cores[slot].collision_disabled;
                    if !(other_is_super || me.is_super)
                        && (me.solo
                            || other_solo
                            || other_collision_disabled
                            || (me.id != -1 && !teams.can_collide(me.id, other_id as i32)))
                    {
                        continue;
                    }
                    let d = vmath::distance(pos, other_pos);
                    if d < physical_size::<R>() {
                        if a > R::ZERO {
                            me.pos = last_pos;
                        } else if vmath::distance(new_pos, other_pos) > d {
                            me.pos = new_pos;
                        }
                        world.cores[self_slot] = me;
                        return;
                    }
                }
                last_pos = pos;
            }
        }
    }

    me.pos = new_pos;
    world.cores[self_slot] = me;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_matches_cpp_defaults_and_leaves_undocumented_fields_alone() {
        // Poison fields `Reset()` does NOT touch, to prove `reset()` really leaves them be.
        let mut c = CharacterCore::<f32> {
            active_weapon: 7,
            colliding: 2,
            left_wall: true,
            id: 42,
            ..Default::default()
        };
        c.tuning.set_by_name("gravity", 1.0);
        c.reset();
        assert_eq!(c.pos, Vec2::zero());
        assert_eq!(c.vel, Vec2::zero());
        assert!(!c.new_hook);
        assert_eq!(c.hook_state, HOOK_IDLE);
        assert_eq!(c.hooked_player(), -1);
        assert_eq!(c.jumped, 0);
        assert_eq!(c.jumped_total, 0);
        assert_eq!(c.jumps, 2);
        assert_eq!(c.triggered_events, 0);
        assert!(!c.solo && !c.is_super && !c.invincible);
        assert_eq!(c.input.target_x, 0);
        assert_eq!(c.input.target_y, -1);
        // Untouched by Reset():
        assert_eq!(c.active_weapon, 7);
        assert_eq!(c.colliding, 2);
        assert!(c.left_wall);
        assert_eq!(c.id, 42);
        assert_eq!(c.tuning.get_by_name("gravity"), Some(1.0));
    }

    #[test]
    fn default_matches_value_initialization() {
        let c = CharacterCore::<f32>::default();
        assert_eq!(c.active_weapon, 0);
        assert_eq!(c.colliding, 0);
        assert!(!c.left_wall);
        assert_eq!(c.id, 0);
        assert_eq!(c.hooked_player(), 0); // zero-init, NOT -1 (Reset() hasn't run yet)
        assert_eq!(c.move_restrictions(), 0);
    }

    #[test]
    fn init_sets_id_to_minus_one() {
        let mut c = CharacterCore::<f32> {
            id: 5,
            ..Default::default()
        };
        c.init();
        assert_eq!(c.id, -1);
    }

    #[test]
    fn write_read_round_trip_is_lossless_for_already_quantized_values() {
        let mut c = CharacterCore::<f32>::default();
        c.reset();
        c.pos = Vec2::new(100.0, -50.0);
        c.vel = Vec2::new(2.5, -1.0); // 2.5*256, -1.0*256 are exact integers
        c.hook_pos = Vec2::new(10.0, 20.0);
        c.hook_dir = Vec2::new(1.0, 0.0);
        c.hook_state = HOOK_FLYING;
        c.hook_tick = 12;
        c.jumped = 3;
        c.direction = -1;
        c.angle = 123;
        let before = c;
        quantize(&mut c);
        assert_eq!(c.pos, before.pos);
        assert_eq!(c.vel, before.vel);
        assert_eq!(c.hook_pos, before.hook_pos);
        assert_eq!(c.hook_dir, before.hook_dir);
        assert_eq!(c.hook_state, before.hook_state);
        assert_eq!(c.hook_tick, before.hook_tick);
        assert_eq!(c.jumped, before.jumped);
        assert_eq!(c.direction, before.direction);
        assert_eq!(c.angle, before.angle);
        assert_eq!(c.hooked_player(), before.hooked_player());
    }

    #[test]
    fn quantize_rounds_velocity_to_the_nearest_256th() {
        let mut c = CharacterCore::<f32>::default();
        c.reset();
        // 1/512 * 256 = 0.5 exactly (both exact powers of 2) -> round_to_int(0.5) = 1 (`f>0`
        // branch: `(0.5+0.5) as i32 = 1`), so this must land exactly on 1/256, not truncate to 0.
        c.vel = Vec2::new(1.0 / 512.0, 0.0);
        quantize(&mut c);
        assert_eq!(c.vel.x, 1.0 / 256.0);
    }

    #[test]
    fn attached_players_tracks_inserts_and_removes() {
        let mut set = AttachedPlayers::default();
        assert!(set.is_empty());
        set.insert(3);
        set.insert(0);
        set.insert(127);
        assert!(set.contains(3) && set.contains(0) && set.contains(127));
        assert!(!set.contains(1));
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 3, 127]); // ascending, like std::set
        set.remove(3);
        assert!(!set.contains(3));
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 127]);
    }

    #[test]
    fn teams_core_default_is_all_flock_no_solo() {
        let teams = TeamsCore::new();
        assert_eq!(teams.team(0), TEAM_FLOCK);
        assert_eq!(teams.team(5), TEAM_FLOCK);
        assert!(!teams.get_solo(0));
        assert!(teams.can_collide(0, 1));
        assert!(teams.same_team(0, 1));
        assert!(teams.can_keep_hook(0, 1));
    }

    #[test]
    fn teams_core_solo_prevents_collide_but_not_self() {
        let mut teams = TeamsCore::new();
        teams.set_solo(2, true);
        assert!(!teams.can_collide(2, 3));
        assert!(!teams.can_collide(3, 2));
        assert!(teams.can_collide(2, 2)); // id1 == id2 always collides (same tee)
    }

    #[test]
    fn teams_core_super_overrides_solo_and_team_mismatch() {
        let mut teams = TeamsCore::new();
        let super_id = teams.team_super();
        teams.set_team(4, super_id);
        teams.set_solo(4, true);
        assert!(teams.can_collide(4, 5));
        assert!(teams.same_team(4, 5));
    }

    #[test]
    fn teams_core_different_teams_cannot_collide_or_keep_hook() {
        let mut teams = TeamsCore::new();
        teams.set_team(0, 1);
        teams.set_team(1, 2);
        assert!(!teams.can_collide(0, 1));
        assert!(!teams.can_keep_hook(0, 1));
        assert!(!teams.same_team(0, 1));
    }

    /// `CCharacterCore::Tick` (`gamecore.cpp:416`) keeps a player hook only `if(pCharCore && m_Id != -1
    /// && m_pTeams->CanKeepHook(m_Id, pCharCore->m_Id))`: `m_Id` is the core's own field. A core
    /// registered through `WorldCore` never has `m_Id == -1`, so only a caller poking `id` through
    /// `get_mut` can reach the release branch; this pins that the check reads the field.
    #[test]
    fn grabbed_player_hook_is_released_only_when_own_id_is_minus_one() {
        let air = map::Tile {
            index: 0,
            flags: 0,
            skip: 0,
            reserved: 0,
        };
        let collision = Collision::<f32>::new(&map::MapData {
            width: 8,
            height: 8,
            game: vec![air; 64],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        });
        let teams = TeamsCore::new();
        let build = |own_id: i32| {
            let mut a = CharacterCore::<f32>::default();
            a.reset();
            let mut b = CharacterCore::<f32>::default();
            b.reset();
            b.pos = Vec2::new(64.0, 0.0);
            let mut world: WorldCore<f32, 2> = WorldCore::from_characters(&[(0, a), (1, b)]);
            let me = world.get_mut(0).unwrap();
            me.set_hooked_player_self_only(1);
            me.hook_state = HOOK_GRABBED;
            me.hook_pos = Vec2::new(64.0, 0.0);
            me.input.hook = 1;
            me.id = own_id;
            world
        };
        let mut kept = build(0);
        tick(&mut kept, 0, &collision, &teams, true, true);
        assert_eq!(kept.core_at(0).hook_state, HOOK_GRABBED);
        assert_eq!(kept.core_at(0).hooked_player(), 1);

        let mut released = build(-1);
        tick(&mut released, 0, &collision, &teams, true, true);
        assert_eq!(released.core_at(0).hook_state, HOOK_RETRACTED);
        assert_eq!(released.core_at(0).hooked_player(), -1);
    }

    #[test]
    #[should_panic(expected = "duplicate client id")]
    fn world_core_from_characters_rejects_duplicate_ids() {
        let mut a = CharacterCore::<f32>::default();
        a.reset();
        let mut b = CharacterCore::<f32>::default();
        b.reset();
        let _world: WorldCore<f32, 4> = WorldCore::from_characters(&[(0, a), (0, b)]);
    }

    #[test]
    #[should_panic(expected = "must be < MAX_CLIENTS")]
    fn world_core_from_characters_rejects_id_at_or_above_max_clients() {
        let mut a = CharacterCore::<f32>::default();
        a.reset();
        let _world: WorldCore<f32, 4> = WorldCore::from_characters(&[(MAX_CLIENTS as u8, a)]);
    }

    #[test]
    #[should_panic(expected = "too many characters")]
    fn world_core_from_characters_rejects_more_entries_than_capacity() {
        let mut a = CharacterCore::<f32>::default();
        a.reset();
        let mut b = CharacterCore::<f32>::default();
        b.reset();
        let mut c = CharacterCore::<f32>::default();
        c.reset();
        let _world: WorldCore<f32, 2> = WorldCore::from_characters(&[(0, a), (1, b), (2, c)]);
    }

    #[test]
    fn world_core_slots_are_sorted_by_ascending_client_id_regardless_of_input_order() {
        let mut a = CharacterCore::<f32>::default();
        a.reset();
        let mut b = CharacterCore::<f32>::default();
        b.reset();
        let world: WorldCore<f32, 4> = WorldCore::from_characters(&[(9, a), (2, b)]);
        assert_eq!(world.id_at(0), 2);
        assert_eq!(world.id_at(1), 9);
        assert_eq!(world.slot_of(2), Some(0));
        assert_eq!(world.slot_of(9), Some(1));
        assert_eq!(world.slot_of(5), None);
    }

    #[test]
    fn random_or_0_is_always_zero_without_a_seeded_prng() {
        let mut world: WorldCore<f32, 2> = WorldCore::from_characters(&[]);
        assert_eq!(world.random_or_0(100), 0);
        assert_eq!(world.random_or_0(0), 0);
        assert_eq!(world.random_or_0(1), 0);
    }

    #[test]
    fn random_or_0_with_a_seeded_prng_is_bounded_and_not_always_zero() {
        let mut world: WorldCore<f32, 2> = WorldCore::from_characters(&[]);
        let mut prng = Prng::new();
        prng.seed([42, 7]);
        world.prng = Some(prng);
        let mut saw_nonzero = false;
        for _ in 0..100 {
            let v = world.random_or_0(10);
            assert!((0..10).contains(&v));
            if v != 0 {
                saw_nonzero = true;
            }
        }
        assert!(
            saw_nonzero,
            "expected at least one nonzero draw out of 100 with bound 10"
        );
    }

    #[test]
    fn saturated_add_clamps_at_bounds() {
        assert_eq!(saturated_add(-10.0f32, 10.0, 9.0, 5.0), 10.0); // clamps at max
        assert_eq!(saturated_add(-10.0f32, 10.0, -9.0, -5.0), -10.0); // clamps at min
        assert_eq!(saturated_add(-10.0f32, 10.0, 11.0, 1.0), 11.0); // already over max, modifier>0 -> no-op
        assert_eq!(saturated_add(-10.0f32, 10.0, -11.0, -1.0), -11.0); // already under min, modifier<0 -> no-op
        assert_eq!(saturated_add(-10.0f32, 10.0, 0.0, 3.0), 3.0); // normal add, within bounds
    }

    #[test]
    fn velocity_ramp_is_one_below_start() {
        assert_eq!(velocity_ramp(100.0f32, 550.0, 2000.0, 1.4), 1.0);
        assert_eq!(velocity_ramp(550.0f32, 550.0, 2000.0, 1.4), 1.0); // not strictly less -> still 1.0
    }

    #[test]
    fn velocity_ramp_decreases_above_start() {
        let r = velocity_ramp(1550.0f32, 550.0, 2000.0, 1.4);
        assert!(r < 1.0 && r > 0.0, "got {r}");
    }

    #[test]
    fn angle_from_target_matches_known_directions() {
        assert_eq!(angle_from_target(1000, 0), 0); // pointing right -> angle 0
        // Pointing straight down: atan2(1,0) = pi/2 -> (pi/2)*256 ≈ 402.
        assert_eq!(angle_from_target(0, 1000), 402);
        // Pointing left: atan2(0,-1) = pi -> wraps via the `< -pi/2` branch? atan2(0,-1)=pi, not
        // negative, so takes the `else` branch: (pi)*256 ≈ 804.
        assert_eq!(angle_from_target(-1000, 0), 804);
    }

    #[test]
    fn clamp_vel_matches_collision_module() {
        // Sanity: `core` re-exports/uses `collision::clamp_vel` consistently.
        let v = Vec2::new(5.0f32, -5.0f32);
        assert_eq!(collision::clamp_vel(collision::CANTMOVE_RIGHT, v), Vec2::new(0.0, -5.0));
    }

    #[test]
    fn physical_size_matches_ddnet_constant() {
        assert_eq!(physical_size::<f32>(), 28.0);
        assert_eq!(physical_size_vec2::<f32>(), Vec2::new(28.0, 28.0));
    }
}
