// Ported from DDNet 20.1 `src/game/tuning.h` and the `CTuneParam`/`CTuningParams` classes in
// `src/game/gamecore.h`/`gamecore.cpp`. DDNet's zlib-style license notice for the ported logic:
//
//   /* (c) Magnus Auvinen. See licence.txt in the root of the distribution for more information. */
//   /* If you are missing that file, acquire a complete release at teeworlds.com.                */
//
// Altered for DDNet-AI: rewritten in Rust; `CTuningParams` is stored as a plain array (not named
// struct fields reinterpreted through a pointer cast, which `NetworkArray()`'s
// `(int*)this` relies on in C++ and which Rust has no safe equivalent of) indexed by
// [`TuneIndex`] constants, with named accessor methods for the ergonomics the C++ named fields
// give for free; `GetWeaponFireDelay` is omitted (it belongs to weapon handling, out of scope —
// see the task spec's constraints).

/// One tuning parameter's storage: `CTuneParam` (`gamecore.h`). Stored as a fixed-point integer
/// (`(int)(v * 100.0f)`), read back as `value / 100.0f` — this representation is the same
/// regardless of which `Real` the physics core is instantiated over (DDNet's C++ tuning is
/// always `float`; see `crate::real::Real::PI`'s doc comment for the analogous point about
/// `base/math.h`'s `pi`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TuneParam(i32);

impl TuneParam {
    /// `CTuneParam::operator=(float v)` (`gamecore.h`): `Fixed = v * 100.0f`, stored as `(int)
    /// Fixed` unless out of `int` range or `Fixed` is NaN, in which case `i32::MIN` (matching
    /// DDNet 20.1/#12664's overflow/NaN handling — see `docs/research/ddnet-physics.md` §1.1).
    pub fn from_f32(v: f32) -> Self {
        let fixed = v * 100.0f32;
        let value = if fixed >= i32::MIN as f32 && fixed < i32::MAX as f32 {
            fixed as i32
        } else {
            i32::MIN
        };
        TuneParam(value)
    }

    /// `CTuneParam::operator float() const` (`gamecore.h`): `m_Value / 100.0f`.
    pub fn as_f32(self) -> f32 {
        self.0 as f32 / 100.0f32
    }

    /// The stored fixed-point integer, as `CTuneParam::Get()`/`NetworkArray()` expose it.
    pub fn raw(self) -> i32 {
        self.0
    }

    /// Wraps an already-computed fixed-point integer (e.g. a scenario's `value_x100` override).
    pub fn from_raw(raw: i32) -> Self {
        TuneParam(raw)
    }

    /// `as_f32`, widened/narrowed to whichever `Real` the calling physics core is instantiated
    /// over — this is what `CCharacterCore<R>::tick` actually reads (mirroring the implicit
    /// `CTuneParam -> float` conversion DDNet's C++ performs wherever a tuning field like
    /// `m_Tuning.m_Gravity` is used in an expression).
    pub fn get<R: crate::real::Real>(self) -> R {
        // Equivalent to `R::from_f64(self.as_f32() as f64)` (for `R = f32` that round trip is an
        // exact no-op: widening an `f32` to `f64` and narrowing back reproduces the identical
        // bit pattern) but without the redundant `f64` detour — this is on the hot path (`Tick`
        // reads a dozen-plus tuning fields every call).
        R::from_i32(self.0) / R::from_i32(100)
    }
}

/// One entry per `MACRO_TUNING_PARAM` in `src/game/tuning.h`, in file order — the indices
/// [`TuningParams`] stores its 47 [`TuneParam`]s at, and the order [`TuningParams::NAMES`]
/// mirrors `CTuningParams::ms_apNames`/`Name(int)`.
// Every index is kept (matching `tuning.h`'s full 47-parameter list) even though only the
// subset `core`'s ported `Tick`/`TickDeferred`/`Move` actually read gets a named accessor below
// (the rest — weapon/jetpack/hammer tuning — belongs to `character.cpp`, out of scope per the
// task spec; they're still reachable via `get`/`get_by_name`/`network_array` for tooling and for
// tasks 1.6-1.8 to add accessors for as needed).
#[allow(dead_code)]
#[rustfmt::skip]
mod idx {
    pub const GROUND_CONTROL_SPEED: usize = 0;
    pub const GROUND_CONTROL_ACCEL: usize = 1;
    pub const GROUND_FRICTION: usize = 2;
    pub const GROUND_JUMP_IMPULSE: usize = 3;
    pub const AIR_JUMP_IMPULSE: usize = 4;
    pub const AIR_CONTROL_SPEED: usize = 5;
    pub const AIR_CONTROL_ACCEL: usize = 6;
    pub const AIR_FRICTION: usize = 7;
    pub const HOOK_LENGTH: usize = 8;
    pub const HOOK_FIRE_SPEED: usize = 9;
    pub const HOOK_DRAG_ACCEL: usize = 10;
    pub const HOOK_DRAG_SPEED: usize = 11;
    pub const GRAVITY: usize = 12;
    pub const VELRAMP_START: usize = 13;
    pub const VELRAMP_RANGE: usize = 14;
    pub const VELRAMP_CURVATURE: usize = 15;
    pub const GUN_CURVATURE: usize = 16;
    pub const GUN_SPEED: usize = 17;
    pub const GUN_LIFETIME: usize = 18;
    pub const SHOTGUN_CURVATURE: usize = 19;
    pub const SHOTGUN_SPEED: usize = 20;
    pub const SHOTGUN_SPEEDDIFF: usize = 21;
    pub const SHOTGUN_LIFETIME: usize = 22;
    pub const GRENADE_CURVATURE: usize = 23;
    pub const GRENADE_SPEED: usize = 24;
    pub const GRENADE_LIFETIME: usize = 25;
    pub const LASER_REACH: usize = 26;
    pub const LASER_BOUNCE_DELAY: usize = 27;
    pub const LASER_BOUNCE_NUM: usize = 28;
    pub const LASER_BOUNCE_COST: usize = 29;
    pub const LASER_DAMAGE: usize = 30;
    pub const PLAYER_COLLISION: usize = 31;
    pub const PLAYER_HOOKING: usize = 32;
    pub const JETPACK_STRENGTH: usize = 33;
    pub const SHOTGUN_STRENGTH: usize = 34;
    pub const EXPLOSION_STRENGTH: usize = 35;
    pub const HAMMER_STRENGTH: usize = 36;
    pub const HOOK_DURATION: usize = 37;
    pub const HAMMER_FIRE_DELAY: usize = 38;
    pub const GUN_FIRE_DELAY: usize = 39;
    pub const SHOTGUN_FIRE_DELAY: usize = 40;
    pub const GRENADE_FIRE_DELAY: usize = 41;
    pub const LASER_FIRE_DELAY: usize = 42;
    pub const NINJA_FIRE_DELAY: usize = 43;
    pub const HAMMER_HIT_FIRE_DELAY: usize = 44;
    pub const GROUND_ELASTICITY_X: usize = 45;
    pub const GROUND_ELASTICITY_Y: usize = 46;
}

/// Number of tuning parameters `src/game/tuning.h` declares (`CTuningParams::Num()`).
pub const NUM: usize = 47;

/// `CTuningParams` (`gamecore.h`): all 47 tuning parameters, in `tuning.h`'s declaration order,
/// with `tuning.h`'s exact default values (computed through [`TuneParam::from_f32`] just like
/// `CTuningParams`'s constructor does via `MACRO_TUNING_PARAM(Name, ScriptName, Value, ...)`
/// expanding to `m_##Name = (Value);`, so a default like `ground_jump_impulse`'s `13.2f` ends up
/// stored/read back as the same `13.1999998...f` DDNet's C++ actually uses, not the "clean"
/// `13.2f` — see `docs/research/ddnet-physics.md` §3.A2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuningParams([TuneParam; NUM]);

/// `ms_apNames`/`Name(int)` (`gamecore.cpp`): each parameter's `ScriptName`, in the same order.
pub const NAMES: [&str; NUM] = [
    "ground_control_speed",
    "ground_control_accel",
    "ground_friction",
    "ground_jump_impulse",
    "air_jump_impulse",
    "air_control_speed",
    "air_control_accel",
    "air_friction",
    "hook_length",
    "hook_fire_speed",
    "hook_drag_accel",
    "hook_drag_speed",
    "gravity",
    "velramp_start",
    "velramp_range",
    "velramp_curvature",
    "gun_curvature",
    "gun_speed",
    "gun_lifetime",
    "shotgun_curvature",
    "shotgun_speed",
    "shotgun_speeddiff",
    "shotgun_lifetime",
    "grenade_curvature",
    "grenade_speed",
    "grenade_lifetime",
    "laser_reach",
    "laser_bounce_delay",
    "laser_bounce_num",
    "laser_bounce_cost",
    "laser_damage",
    "player_collision",
    "player_hooking",
    "jetpack_strength",
    "shotgun_strength",
    "explosion_strength",
    "hammer_strength",
    "hook_duration",
    "hammer_fire_delay",
    "gun_fire_delay",
    "shotgun_fire_delay",
    "grenade_fire_delay",
    "laser_fire_delay",
    "ninja_fire_delay",
    "hammer_hit_fire_delay",
    "ground_elasticity_x",
    "ground_elasticity_y",
];

/// `tuning.h`'s literal default values, `f32`, in declaration order — kept separate from
/// [`TuningParams::default`] only so the long list reads like `tuning.h` itself.
const DEFAULTS: [f32; NUM] = [
    10.0,
    100.0 / 50.0,
    0.5,
    13.2,
    12.0,
    250.0 / 50.0,
    1.5,
    0.95,
    380.0,
    80.0,
    3.0,
    15.0,
    0.5,
    550.0,
    2000.0,
    1.4,
    1.25,
    2200.0,
    2.0,
    1.25,
    2750.0,
    0.8,
    0.20,
    7.0,
    1000.0,
    2.0,
    800.0,
    150.0,
    1000.0,
    0.0,
    5.0,
    1.0,
    1.0,
    400.0,
    10.0,
    6.0,
    1.0,
    1.25,
    125.0,
    125.0,
    500.0,
    500.0,
    800.0,
    800.0,
    320.0,
    0.0,
    0.0,
];

impl Default for TuningParams {
    fn default() -> Self {
        let mut params = [TuneParam::default(); NUM];
        for (slot, &default) in params.iter_mut().zip(DEFAULTS.iter()) {
            *slot = TuneParam::from_f32(default);
        }
        TuningParams(params)
    }
}

impl TuningParams {
    /// `CTuningParams::Num()`.
    pub const fn num() -> usize {
        NUM
    }

    /// `CTuningParams::Name(int Index)`.
    pub fn name(index: usize) -> &'static str {
        NAMES[index]
    }

    /// `CTuningParams::NetworkArray()` (`const` overload): the 47 raw fixed-point integers, in
    /// order — e.g. for a scenario file's tuning override to write into directly, bypassing the
    /// `Set`/`from_f32` float round trip (see `docs/formats.md` §2, "Tuning-переопределения").
    pub fn network_array(&self) -> [i32; NUM] {
        self.0.map(TuneParam::raw)
    }

    /// `CTuningParams::Set(int Index, float Value)`.
    pub fn set(&mut self, index: usize, value: f32) -> bool {
        if index >= NUM {
            return false;
        }
        self.0[index] = TuneParam::from_f32(value);
        true
    }

    /// Sets a raw fixed-point value directly (`NetworkArray()[index] = value_x100`), without
    /// going through the `f32` round trip `set`/`from_f32` would — see `docs/formats.md` §2 for
    /// why a scenario's tuning override must be applied this way.
    pub fn set_raw(&mut self, index: usize, value_x100: i32) -> bool {
        if index >= NUM {
            return false;
        }
        self.0[index] = TuneParam::from_raw(value_x100);
        true
    }

    /// `CTuningParams::Get(int Index, float *pValue)`.
    pub fn get(&self, index: usize) -> Option<f32> {
        self.0.get(index).map(|p| p.as_f32())
    }

    /// `CTuningParams::Set(const char *pName, float Value)`: case-insensitive name lookup
    /// (`str_comp_nocase`).
    pub fn set_by_name(&mut self, name: &str, value: f32) -> bool {
        match Self::index_of(name) {
            Some(i) => self.set(i, value),
            None => false,
        }
    }

    /// `CTuningParams::Get(const char *pName, float *pValue)`.
    pub fn get_by_name(&self, name: &str) -> Option<f32> {
        Self::index_of(name).and_then(|i| self.get(i))
    }

    fn index_of(name: &str) -> Option<usize> {
        NAMES.iter().position(|n| n.eq_ignore_ascii_case(name))
    }

    fn param<R: crate::real::Real>(&self, index: usize) -> R {
        self.0[index].get()
    }

    // Named accessors mirroring `m_<Name>` field access in `gamecore.cpp` — see `idx` above for
    // the index each corresponds to. Only the ones `core`/`collision` actually read are named;
    // the rest are reachable via `get`/`get_by_name`/`network_array` for completeness (tune
    // zones, later tasks, tooling).
    /// `m_GroundControlSpeed` (`ground_control_speed`).
    pub fn ground_control_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::GROUND_CONTROL_SPEED)
    }
    /// `m_GroundControlAccel` (`ground_control_accel`).
    pub fn ground_control_accel<R: crate::real::Real>(&self) -> R {
        self.param(idx::GROUND_CONTROL_ACCEL)
    }
    /// `m_GroundFriction` (`ground_friction`).
    pub fn ground_friction<R: crate::real::Real>(&self) -> R {
        self.param(idx::GROUND_FRICTION)
    }
    /// `m_GroundJumpImpulse` (`ground_jump_impulse`).
    pub fn ground_jump_impulse<R: crate::real::Real>(&self) -> R {
        self.param(idx::GROUND_JUMP_IMPULSE)
    }
    /// `m_AirJumpImpulse` (`air_jump_impulse`).
    pub fn air_jump_impulse<R: crate::real::Real>(&self) -> R {
        self.param(idx::AIR_JUMP_IMPULSE)
    }
    /// `m_AirControlSpeed` (`air_control_speed`).
    pub fn air_control_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::AIR_CONTROL_SPEED)
    }
    /// `m_AirControlAccel` (`air_control_accel`).
    pub fn air_control_accel<R: crate::real::Real>(&self) -> R {
        self.param(idx::AIR_CONTROL_ACCEL)
    }
    /// `m_AirFriction` (`air_friction`).
    pub fn air_friction<R: crate::real::Real>(&self) -> R {
        self.param(idx::AIR_FRICTION)
    }
    /// `m_HookLength` (`hook_length`).
    pub fn hook_length<R: crate::real::Real>(&self) -> R {
        self.param(idx::HOOK_LENGTH)
    }
    /// `m_HookFireSpeed` (`hook_fire_speed`).
    pub fn hook_fire_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::HOOK_FIRE_SPEED)
    }
    /// `m_HookDragAccel` (`hook_drag_accel`).
    pub fn hook_drag_accel<R: crate::real::Real>(&self) -> R {
        self.param(idx::HOOK_DRAG_ACCEL)
    }
    /// `m_HookDragSpeed` (`hook_drag_speed`).
    pub fn hook_drag_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::HOOK_DRAG_SPEED)
    }
    /// `m_Gravity` (`gravity`).
    pub fn gravity<R: crate::real::Real>(&self) -> R {
        self.param(idx::GRAVITY)
    }
    /// `m_VelrampStart` (`velramp_start`).
    pub fn velramp_start<R: crate::real::Real>(&self) -> R {
        self.param(idx::VELRAMP_START)
    }
    /// `m_VelrampRange` (`velramp_range`).
    pub fn velramp_range<R: crate::real::Real>(&self) -> R {
        self.param(idx::VELRAMP_RANGE)
    }
    /// `m_VelrampCurvature` (`velramp_curvature`).
    pub fn velramp_curvature<R: crate::real::Real>(&self) -> R {
        self.param(idx::VELRAMP_CURVATURE)
    }
    /// `m_PlayerCollision` (`player_collision`).
    pub fn player_collision<R: crate::real::Real>(&self) -> R {
        self.param(idx::PLAYER_COLLISION)
    }
    /// `m_PlayerHooking` (`player_hooking`).
    pub fn player_hooking<R: crate::real::Real>(&self) -> R {
        self.param(idx::PLAYER_HOOKING)
    }
    /// `m_HookDuration` (`hook_duration`).
    pub fn hook_duration<R: crate::real::Real>(&self) -> R {
        self.param(idx::HOOK_DURATION)
    }
    /// `m_GroundElasticityX` (`ground_elasticity_x`).
    pub fn ground_elasticity_x<R: crate::real::Real>(&self) -> R {
        self.param(idx::GROUND_ELASTICITY_X)
    }
    /// `m_GroundElasticityY` (`ground_elasticity_y`).
    pub fn ground_elasticity_y<R: crate::real::Real>(&self) -> R {
        self.param(idx::GROUND_ELASTICITY_Y)
    }

    // Task 1.6: weapon/projectile tuning accessors (`character.cpp`/`projectile.cpp`, out of
    // task 1.3's core-only scope, in scope here). Same pattern as the accessors above.
    /// `m_GunCurvature` (`gun_curvature`).
    pub fn gun_curvature<R: crate::real::Real>(&self) -> R {
        self.param(idx::GUN_CURVATURE)
    }
    /// `m_GunSpeed` (`gun_speed`).
    pub fn gun_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::GUN_SPEED)
    }
    /// `m_GunLifetime` (`gun_lifetime`), in seconds — `character.cpp:574`:
    /// `(int)(Server()->TickSpeed() * GetTuning(m_TuneZone)->m_GunLifetime)`.
    pub fn gun_lifetime(&self) -> f32 {
        self.param::<f32>(idx::GUN_LIFETIME)
    }
    /// `m_GrenadeCurvature` (`grenade_curvature`).
    pub fn grenade_curvature<R: crate::real::Real>(&self) -> R {
        self.param(idx::GRENADE_CURVATURE)
    }
    /// `m_GrenadeSpeed` (`grenade_speed`).
    pub fn grenade_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::GRENADE_SPEED)
    }
    /// `m_GrenadeLifetime` (`grenade_lifetime`), in seconds — see [`Self::gun_lifetime`]'s doc
    /// comment for the same pattern (`character.cpp:605`).
    pub fn grenade_lifetime(&self) -> f32 {
        self.param::<f32>(idx::GRENADE_LIFETIME)
    }
    /// `m_ShotgunCurvature` (`shotgun_curvature`) — used by the `ENTITY_CRAZY_SHOTGUN[_EX]` map
    /// fixture's `CProjectile` (a real `WEAPON_SHOTGUN`-typed projectile; Stage A scope), not by
    /// the player's own shotgun fire (a `CLaser`, Stage B).
    pub fn shotgun_curvature<R: crate::real::Real>(&self) -> R {
        self.param(idx::SHOTGUN_CURVATURE)
    }
    /// `m_ShotgunSpeed` (`shotgun_speed`).
    pub fn shotgun_speed<R: crate::real::Real>(&self) -> R {
        self.param(idx::SHOTGUN_SPEED)
    }
    /// `m_JetpackStrength` (`jetpack_strength`) — read by `HandleJetpack` (`character.cpp:289`);
    /// never actually applied in this corpus (`jetpack_ticks == 0` — no scenario/map ever sets
    /// `m_Core.m_Jetpack`), kept for structural completeness.
    pub fn jetpack_strength<R: crate::real::Real>(&self) -> R {
        self.param(idx::JETPACK_STRENGTH)
    }
    /// `m_ExplosionStrength` (`explosion_strength`) — `CGameContext::CreateExplosion`
    /// (`gamecontext.cpp:372`).
    pub fn explosion_strength<R: crate::real::Real>(&self) -> R {
        self.param(idx::EXPLOSION_STRENGTH)
    }
    /// `m_HammerStrength` (`hammer_strength`) — `FireWeapon`'s `WEAPON_HAMMER` case
    /// (`character.cpp:547`).
    pub fn hammer_strength<R: crate::real::Real>(&self) -> R {
        self.param(idx::HAMMER_STRENGTH)
    }

    /// `CTuningParams::GetWeaponFireDelay(int Weapon)` (`gamecore.cpp:56-68`): always plain
    /// `float` arithmetic in the C++ source (the return type is `float`, unconditionally, never
    /// widened to whatever `R` the calling `CharacterCore<R>` uses) — so this returns `f32`
    /// rather than being generic over [`crate::real::Real`], matching that exactly.
    ///
    /// # Panics
    ///
    /// If `weapon` isn't one of `WEAPON_HAMMER..=WEAPON_NINJA` (`0..=5`) — matches the C++
    /// `dbg_assert_failed` on its `default` switch case (a caller passing e.g. `-1`, "no
    /// weapon", is a logic error the same way it would be in the original).
    pub fn get_weapon_fire_delay(&self, weapon: i32) -> f32 {
        let idx = match weapon {
            0 => idx::HAMMER_FIRE_DELAY,
            1 => idx::GUN_FIRE_DELAY,
            2 => idx::SHOTGUN_FIRE_DELAY,
            3 => idx::GRENADE_FIRE_DELAY,
            4 => idx::LASER_FIRE_DELAY,
            5 => idx::NINJA_FIRE_DELAY,
            other => panic!("GetWeaponFireDelay: invalid weapon {other}"),
        };
        self.param::<f32>(idx) / 1000.0f32
    }

    /// `m_HammerFireDelay`/`m_HammerHitFireDelay` — the "miss"/"hit" reload-timer formulas
    /// `FireWeapon`'s `WEAPON_HAMMER` case picks between (`character.cpp:561-566`), each already
    /// divided into [`Self::get_weapon_fire_delay`]'s `WEAPON_HAMMER` case for the "miss" one;
    /// this is the "hit" one specifically, in **milliseconds** (not yet divided by 1000, and not
    /// yet multiplied by `TickSpeed` — `character.cpp:564-565` does
    /// `FireDelay * Server()->TickSpeed() / 1000` as a single `float` expression, so callers
    /// reproduce that exact expression shape themselves rather than composing two already-scaled
    /// helpers).
    pub fn hammer_hit_fire_delay_ms(&self) -> f32 {
        self.param::<f32>(idx::HAMMER_HIT_FIRE_DELAY)
    }
    /// `m_HammerFireDelay`, in **milliseconds** — see [`Self::hammer_hit_fire_delay_ms`]'s doc
    /// comment for why this is the raw millisecond value, not a ticks conversion.
    pub fn hammer_fire_delay_ms(&self) -> f32 {
        self.param::<f32>(idx::HAMMER_FIRE_DELAY)
    }
}

/// `CWorldCore`/`CCharacter::m_Core.m_Tuning` per-zone overrides (`TILE_TUNE`, `CTuneTile`):
/// a placeholder container for task 1.7 (tune-zone selection, `HandleTuneLayer` in
/// `character.cpp`, out of scope here). Not read by `core`/`core_world` yet.
#[derive(Debug, Clone, Default)]
pub struct TuneZones {
    zones: std::collections::HashMap<u8, TuningParams>,
}

impl TuneZones {
    /// No zone overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// The zone-specific tuning for zone `number`, or `None` if zone `number` has no override
    /// (a tee outside any tune zone, or in zone `0`, uses the world's default tuning instead).
    pub fn get(&self, number: u8) -> Option<&TuningParams> {
        self.zones.get(&number)
    }

    /// Sets (or replaces) zone `number`'s tuning override.
    pub fn set(&mut self, number: u8, params: TuningParams) {
        self.zones.insert(number, params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_and_names_agree() {
        assert_eq!(NAMES.len(), NUM);
        assert_eq!(TuningParams::num(), NUM);
    }

    #[test]
    fn default_gravity_is_zero_point_five() {
        let t = TuningParams::default();
        assert_eq!(t.get_by_name("gravity"), Some(0.5));
        assert_eq!(t.gravity::<f32>(), 0.5f32);
    }

    #[test]
    fn default_ground_jump_impulse_round_trips_through_fixed_point_like_cpp() {
        // docs/research/ddnet-physics.md §3.A2: `13.2f32` is not exactly representable in
        // binary32 (it's really ~13.199999809265137); the *stored* representation is the
        // fixed-point integer `(int)(13.2f * 100.0f)`, not a float, and it must be exactly 1320
        // (computed via `TuneParam::from_f32`, the same expression `CTuneParam::operator=` uses)
        // — not 1319, which a naive "13.2*100 truncates because floats round down" assumption
        // would predict.
        let t = TuningParams::default();
        let idx = NAMES.iter().position(|&n| n == "ground_jump_impulse").unwrap();
        assert_eq!(TuneParam::from_f32(13.2).raw(), 1320);
        assert_eq!(t.network_array()[idx], 1320);
        let v = t.get_by_name("ground_jump_impulse").unwrap();
        assert_eq!(v, 1320.0f32 / 100.0f32);
    }

    #[test]
    fn default_velramp_start_and_range_are_integers_stored_as_float() {
        let t = TuningParams::default();
        assert_eq!(t.get_by_name("velramp_start"), Some(550.0));
        assert_eq!(t.get_by_name("velramp_range"), Some(2000.0));
    }

    #[test]
    fn get_and_set_by_name_are_case_insensitive() {
        let mut t = TuningParams::default();
        assert!(t.set_by_name("GRAVITY", 1.0));
        assert_eq!(t.get_by_name("gravity"), Some(1.0));
        assert_eq!(t.get_by_name("GrAvItY"), Some(1.0));
    }

    #[test]
    fn set_by_name_rejects_unknown_name() {
        let mut t = TuningParams::default();
        assert!(!t.set_by_name("not_a_real_param", 1.0));
    }

    #[test]
    fn get_rejects_out_of_range_index() {
        let t = TuningParams::default();
        assert_eq!(t.get(NUM), None);
        assert_eq!(t.get(NUM + 100), None);
    }

    #[test]
    fn set_rejects_out_of_range_index() {
        let mut t = TuningParams::default();
        assert!(!t.set(NUM, 1.0));
    }

    #[test]
    fn set_raw_bypasses_the_float_round_trip() {
        // docs/formats.md §2: a scenario's tuning override stores the exact fixed-point integer
        // (`value_x100`), applied directly via `set_raw` (mirrors the oracle's
        // `NetworkArray()[index] = ValueX100`), never through `set`'s `f32` conversion — applying
        // e.g. value_x100=25 through `set(idx, 0.25)` could round differently.
        let mut t = TuningParams::default();
        let idx = NAMES.iter().position(|&n| n == "gravity").unwrap();
        assert!(t.set_raw(idx, 25));
        assert_eq!(t.network_array()[idx], 25);
        assert_eq!(t.get(idx), Some(0.25));
    }

    #[test]
    fn network_array_reflects_set_values() {
        let mut t = TuningParams::default();
        let idx = NAMES.iter().position(|&n| n == "hook_length").unwrap();
        t.set(idx, 100.0);
        assert_eq!(t.network_array()[idx], 10000);
    }

    #[test]
    fn tune_param_clamps_overflowing_fixed_point_to_i32_min() {
        // #12664 (docs/research/ddnet-physics.md §1.1): `v*100` overflowing `int` range stores
        // `i32::MIN`, not a wrapped/UB value.
        let huge = TuneParam::from_f32(f32::MAX);
        assert_eq!(huge.raw(), i32::MIN);
    }

    #[test]
    fn tune_param_clamps_nan_to_i32_min() {
        let nan = TuneParam::from_f32(f32::NAN);
        assert_eq!(nan.raw(), i32::MIN);
    }

    #[test]
    fn tune_zones_starts_empty() {
        let zones = TuneZones::new();
        assert_eq!(zones.get(1), None);
    }

    #[test]
    fn tune_zones_get_and_set() {
        let mut zones = TuneZones::new();
        let mut params = TuningParams::default();
        params.set_by_name("gravity", 0.0);
        zones.set(1, params);
        assert_eq!(zones.get(1).unwrap().get_by_name("gravity"), Some(0.0));
        assert_eq!(zones.get(2), None);
    }

    #[test]
    fn param_generic_over_f64_matches_f32_value() {
        let t = TuningParams::default();
        let g32: f32 = t.gravity();
        let g64: f64 = t.gravity();
        assert_eq!(g64 as f32, g32);
    }
}
