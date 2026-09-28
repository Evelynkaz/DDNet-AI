//! Minimal readers for scenario v3 and trace-b v2 (`docs/formats.md` §11/§12.6) — just enough to
//! drive `tests/parity_oracle_b.rs` against the real Oracle B corpus. Not part of `ddai-trace`'s
//! public API (task 1.6 is a `ddai-physics` task; these two formats are Oracle-B/task-1.5
//! specific test-input formats, read here directly rather than growing `ddai-trace`'s surface for
//! a single consumer). Reuses `ddai_trace::io::Reader` (task 1.2) for the primitive decodes.
//!
//! Lives under `tests/common/` (like `tests/common/mod.rs`) so Cargo doesn't treat it as its own
//! test binary; `tests/parity_oracle_b.rs` pulls it in via `#[path = ...] mod`.
//!
//! Several fields below (`SwitchEntry`'s, most of `EntityRecord`'s, `DDRaceState`'s
//! `died_this_tick`/`respawned_this_tick`, `TraceBTick::switches`, `TraceBReader::
//! switch_team_ids`) are read off the stream — required to parse the fixed-layout binary
//! correctly and stay in sync with the next field — but not currently consumed by
//! `parity_oracle_b.rs`'s comparisons; kept as documentation of the exact wire format (and for
//! whichever future comparison needs them) rather than deleted.
#![allow(dead_code)]

use ddai_trace::io::Reader;

fn read_u64(r: &mut Reader, context: &'static str) -> u64 {
    let b = r.bytes(8, context).unwrap();
    u64::from_le_bytes(b.try_into().unwrap())
}

/// One scenario-v3 character entry (`docs/formats.md` §12.6).
#[derive(Debug, Clone, Copy)]
pub struct ScenarioV3Character {
    pub id: u32,
    pub spawn_x: i32,
    pub spawn_y: i32,
    pub team: i32,
}

/// One resolved scenario-v3 input row (`docs/formats.md` §11/§12.6): the same 11-field layout as
/// scenario v2's `ScenarioInput`, plus `kill` — since scenario v3 stores *already-resolved*
/// inputs (`aim_slot` always `-1`), the first 10 fields are directly a
/// [`ddai_physics::core::PlayerInput`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioV3Input {
    pub direction: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub jump: i32,
    pub fire: i32,
    pub hook: i32,
    pub player_flags: i32,
    pub wanted_weapon: i32,
    pub next_weapon: i32,
    pub prev_weapon: i32,
    pub kill: i32,
}

/// A parsed scenario v3 file (`docs/formats.md` §12.6). `inputs[tick][character_slot]`.
pub struct ScenarioV3 {
    pub rawmap_path: String,
    pub map_sha256: [u8; 32],
    pub characters: Vec<ScenarioV3Character>,
    pub inputs: Vec<Vec<ScenarioV3Input>>,
    pub cfg_lines: Vec<String>,
    pub generator_id: String,
    pub embedded_seed: u64,
}

impl ScenarioV3 {
    pub fn read_bytes(bytes: &[u8]) -> Self {
        let mut r = Reader::new(bytes);
        r.expect_magic(b"SCN1").expect("bad scenario magic");
        let version = r.u32("version").unwrap();
        assert_eq!(version, 3, "expected scenario v3");
        let map_ref_tag = r.u8("map ref tag").unwrap();
        assert_eq!(
            map_ref_tag, 1,
            "oracle-b-emitted scenario v3 always has tag 1 (RawmapFile)"
        );
        let rawmap_path = r.string16("rawmap path").unwrap();
        let map_sha256 = r.array32::<32>("map sha256").unwrap();
        let _no_weak_hook = r.u8("no_weak_hook").unwrap();
        let override_count = r.u32("tuning override count").unwrap();
        assert_eq!(override_count, 0, "v3 files use cfg_lines, not tuning_overrides");
        let char_count = r.u32("character count").unwrap();
        let mut characters = Vec::with_capacity(char_count as usize);
        for _ in 0..char_count {
            let id = r.u32("character id").unwrap();
            let spawn_x = r.i32("spawn x").unwrap();
            let spawn_y = r.i32("spawn y").unwrap();
            let team = r.i32("team").unwrap();
            characters.push(ScenarioV3Character {
                id,
                spawn_x,
                spawn_y,
                team,
            });
        }
        let tick_count = r.u32("tick count").unwrap();
        let mut inputs = Vec::with_capacity(tick_count as usize);
        for _ in 0..tick_count {
            let mut tick_inputs = Vec::with_capacity(characters.len());
            for _ in 0..characters.len() {
                let direction = r.i32("direction").unwrap();
                let target_x = r.i32("target_x").unwrap();
                let target_y = r.i32("target_y").unwrap();
                let _aim_slot = r.i32("aim_slot").unwrap();
                let jump = r.i32("jump").unwrap();
                let fire = r.i32("fire").unwrap();
                let hook = r.i32("hook").unwrap();
                let player_flags = r.i32("player_flags").unwrap();
                let wanted_weapon = r.i32("wanted_weapon").unwrap();
                let next_weapon = r.i32("next_weapon").unwrap();
                let prev_weapon = r.i32("prev_weapon").unwrap();
                let kill = r.i32("kill").unwrap();
                tick_inputs.push(ScenarioV3Input {
                    direction,
                    target_x,
                    target_y,
                    jump,
                    fire,
                    hook,
                    player_flags,
                    wanted_weapon,
                    next_weapon,
                    prev_weapon,
                    kill,
                });
            }
            inputs.push(tick_inputs);
        }
        let cfg_line_count = r.u32("cfg line count").unwrap();
        let mut cfg_lines = Vec::with_capacity(cfg_line_count as usize);
        for _ in 0..cfg_line_count {
            cfg_lines.push(r.string16("cfg line").unwrap());
        }
        let generator_id = r.string16("generator id").unwrap();
        let embedded_seed = read_u64(&mut r, "embedded seed");
        r.expect_eof().unwrap();
        ScenarioV3 {
            rawmap_path,
            map_sha256,
            characters,
            inputs,
            cfg_lines,
            generator_id,
            embedded_seed,
        }
    }
}

/// One switch's dumped state for one (team, tick) — `docs/formats.md` §11.2.
#[derive(Debug, Clone, Copy, Default)]
pub struct SwitchEntry {
    pub status: i32,
    pub end_tick: i32,
    pub kind: i32,
    pub last_update_tick: i32,
}

/// One `EntityRecord` (`docs/formats.md` §11.2) — 36 bytes.
#[derive(Debug, Clone, Copy)]
pub struct EntityRecord {
    pub kind: i32,
    pub owner_client_id: i32,
    pub weapon_type: i32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub dir_x: f32,
    pub dir_y: f32,
    pub start_tick: i32,
    pub extra: i32,
}

/// One character's `CoreStateFields` (28 fields, `docs/formats.md` §6.2) — bit-exact layout with
/// task 1.3's `CharacterCoreState`. Read as raw values here (not `ddai_trace::trace
/// ::CharacterCoreState`, to avoid depending on `ddai-trace`'s exact field order beyond what this
/// file itself pins down independently — see this module's doc comment on scope).
#[derive(Debug, Clone, Copy, Default)]
pub struct CoreState {
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub hook_pos_x: f32,
    pub hook_pos_y: f32,
    pub hook_dir_x: f32,
    pub hook_dir_y: f32,
    pub hook_tele_base_x: f32,
    pub hook_tele_base_y: f32,
    pub hook_tick: i32,
    pub hook_state: i32,
    pub hooked_player: i32,
    pub active_weapon: i32,
    pub new_hook: i32,
    pub jumped: i32,
    pub jumped_total: i32,
    pub jumps: i32,
    pub direction: i32,
    pub angle: i32,
    pub triggered_events: i32,
    pub colliding: i32,
    pub left_wall: i32,
    pub move_restrictions: i32,
    pub solo: i32,
    pub collision_disabled: i32,
    pub endless_hook: i32,
    pub hook_hit_disabled: i32,
}

/// One character's `DDRaceStateFields` (54 fields, `docs/formats.md` §11.3), in table order.
#[derive(Debug, Clone, Copy, Default)]
pub struct DDRaceState {
    pub alive: i32,
    pub died_this_tick: i32,
    pub respawned_this_tick: i32,
    pub freeze_time: i32,
    pub is_in_freeze: i32,
    pub deep_frozen: i32,
    pub live_frozen: i32,
    pub frozen_last_tick: i32,
    pub reload_timer: i32,
    pub attack_tick: i32,
    pub queued_weapon: i32,
    pub last_weapon: i32,
    pub weapon_got_mask: i32,
    pub weapon_ammo: [i32; 6],
    pub weapon_ammo_regen_start: [i32; 6],
    pub ninja_activation_tick: i32,
    pub ninja_current_move_time: i32,
    pub ninja_old_vel_amount: i32,
    pub ninja_activation_dir_x: f32,
    pub ninja_activation_dir_y: f32,
    pub tele_checkpoint: i32,
    pub endless_jump: i32,
    pub jetpack: i32,
    pub is_super: i32,
    pub invincible: i32,
    pub hammer_hit_disabled: i32,
    pub grenade_hit_disabled: i32,
    pub laser_hit_disabled: i32,
    pub shotgun_hit_disabled: i32,
    pub has_telegun_gun: i32,
    pub has_telegun_grenade: i32,
    pub has_telegun_laser: i32,
    pub team: i32,
    pub strong_weak_id: i32,
    pub freeze_start: i32,
    pub freeze_end: i32,
    pub tune_zone: i32,
    pub num_inputs: i32,
    pub last_refill_jumps: i32,
    pub ddrace_state: i32,
    pub start_time: i32,
    pub die_tick: i32,
    pub spawning: i32,
    pub previous_die_tick: i32,
}

fn read_ddrace_state(r: &mut Reader) -> DDRaceState {
    let mut s = DDRaceState {
        alive: r.i32("alive").unwrap(),
        died_this_tick: r.i32("died_this_tick").unwrap(),
        respawned_this_tick: r.i32("respawned_this_tick").unwrap(),
        freeze_time: r.i32("freeze_time").unwrap(),
        is_in_freeze: r.i32("is_in_freeze").unwrap(),
        deep_frozen: r.i32("deep_frozen").unwrap(),
        live_frozen: r.i32("live_frozen").unwrap(),
        frozen_last_tick: r.i32("frozen_last_tick").unwrap(),
        reload_timer: r.i32("reload_timer").unwrap(),
        attack_tick: r.i32("attack_tick").unwrap(),
        queued_weapon: r.i32("queued_weapon").unwrap(),
        last_weapon: r.i32("last_weapon").unwrap(),
        weapon_got_mask: r.i32("weapon_got_mask").unwrap(),
        ..Default::default()
    };
    for slot in s.weapon_ammo.iter_mut() {
        *slot = r.i32("weapon_ammo").unwrap();
    }
    for slot in s.weapon_ammo_regen_start.iter_mut() {
        *slot = r.i32("weapon_ammo_regen_start").unwrap();
    }
    s.ninja_activation_tick = r.i32("ninja_activation_tick").unwrap();
    s.ninja_current_move_time = r.i32("ninja_current_move_time").unwrap();
    s.ninja_old_vel_amount = r.i32("ninja_old_vel_amount").unwrap();
    s.ninja_activation_dir_x = r.f32("ninja_activation_dir_x").unwrap();
    s.ninja_activation_dir_y = r.f32("ninja_activation_dir_y").unwrap();
    s.tele_checkpoint = r.i32("tele_checkpoint").unwrap();
    s.endless_jump = r.i32("endless_jump").unwrap();
    s.jetpack = r.i32("jetpack").unwrap();
    s.is_super = r.i32("super").unwrap();
    s.invincible = r.i32("invincible").unwrap();
    s.hammer_hit_disabled = r.i32("hammer_hit_disabled").unwrap();
    s.grenade_hit_disabled = r.i32("grenade_hit_disabled").unwrap();
    s.laser_hit_disabled = r.i32("laser_hit_disabled").unwrap();
    s.shotgun_hit_disabled = r.i32("shotgun_hit_disabled").unwrap();
    s.has_telegun_gun = r.i32("has_telegun_gun").unwrap();
    s.has_telegun_grenade = r.i32("has_telegun_grenade").unwrap();
    s.has_telegun_laser = r.i32("has_telegun_laser").unwrap();
    s.team = r.i32("team").unwrap();
    s.strong_weak_id = r.i32("strong_weak_id").unwrap();
    s.freeze_start = r.i32("freeze_start").unwrap();
    s.freeze_end = r.i32("freeze_end").unwrap();
    s.tune_zone = r.i32("tune_zone").unwrap();
    s.num_inputs = r.i32("num_inputs").unwrap();
    s.last_refill_jumps = r.i32("last_refill_jumps").unwrap();
    s.ddrace_state = r.i32("ddrace_state").unwrap();
    s.start_time = r.i32("start_time").unwrap();
    s.die_tick = r.i32("die_tick").unwrap();
    s.spawning = r.i32("spawning").unwrap();
    s.previous_die_tick = r.i32("previous_die_tick").unwrap();
    s
}

fn read_core_state(r: &mut Reader) -> CoreState {
    CoreState {
        pos_x: r.f32("pos_x").unwrap(),
        pos_y: r.f32("pos_y").unwrap(),
        vel_x: r.f32("vel_x").unwrap(),
        vel_y: r.f32("vel_y").unwrap(),
        hook_pos_x: r.f32("hook_pos_x").unwrap(),
        hook_pos_y: r.f32("hook_pos_y").unwrap(),
        hook_dir_x: r.f32("hook_dir_x").unwrap(),
        hook_dir_y: r.f32("hook_dir_y").unwrap(),
        hook_tele_base_x: r.f32("hook_tele_base_x").unwrap(),
        hook_tele_base_y: r.f32("hook_tele_base_y").unwrap(),
        hook_tick: r.i32("hook_tick").unwrap(),
        hook_state: r.i32("hook_state").unwrap(),
        hooked_player: r.i32("hooked_player").unwrap(),
        active_weapon: r.i32("active_weapon").unwrap(),
        new_hook: r.i32("new_hook").unwrap(),
        jumped: r.i32("jumped").unwrap(),
        jumped_total: r.i32("jumped_total").unwrap(),
        jumps: r.i32("jumps").unwrap(),
        direction: r.i32("direction").unwrap(),
        angle: r.i32("angle").unwrap(),
        triggered_events: r.i32("triggered_events").unwrap(),
        colliding: r.i32("colliding").unwrap(),
        left_wall: r.i32("left_wall").unwrap(),
        move_restrictions: r.i32("move_restrictions").unwrap(),
        solo: r.i32("solo").unwrap(),
        collision_disabled: r.i32("collision_disabled").unwrap(),
        endless_hook: r.i32("endless_hook").unwrap(),
        hook_hit_disabled: r.i32("hook_hit_disabled").unwrap(),
    }
}

/// One character's full row for one tick: applied input (11 fields, incl. `kill`) + `CoreState`
/// (28) + `DDRaceState` (54).
#[derive(Debug, Clone, Copy)]
pub struct CharacterRow {
    pub input: ScenarioV3Input,
    pub core: CoreState,
    pub ddrace: DDRaceState,
}

/// One tick's `GlobalTickFields` + every character's row (`docs/formats.md` §11/§11.2).
pub struct TraceBTick {
    pub game_tick: i32,
    /// `switches[team_idx * switch_highest_number + switch_idx]`.
    pub switches: Vec<SwitchEntry>,
    pub entities: Vec<EntityRecord>,
    pub characters: Vec<CharacterRow>,
}

/// A streaming trace-b v2 reader: parses the header up front, then yields one tick at a time via
/// [`TraceBReader::next_tick`] — the whole file's bytes are held in memory (`Vec<u8>`, read once
/// by the caller), but only one [`TraceBTick`] exists at a time, keeping this test's own working
/// set small regardless of a trace's tick count (see this module's doc comment on the
/// "load one file's bytes, not the whole corpus" memory contract this crate's `BUILD REPORT`
/// documents).
pub struct TraceBReader<'a> {
    r: Reader<'a>,
    pub metadata_json: String,
    pub character_ids: Vec<u32>,
    pub switch_highest_number: u32,
    pub switch_team_count: u32,
    pub switch_team_ids: Vec<i32>,
    pub tick_count: u32,
    ticks_read: u32,
}

impl<'a> TraceBReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let mut r = Reader::new(bytes);
        r.expect_magic(b"TRB1").expect("bad trace-b magic");
        let version = r.u32("version").unwrap();
        assert_eq!(version, 2, "expected trace-b v2");
        let metadata_len = r.u32("metadata len").unwrap();
        let metadata_bytes = r.bytes(metadata_len as usize, "metadata json").unwrap();
        let metadata_json = String::from_utf8(metadata_bytes.to_vec()).unwrap();
        let character_count = r.u32("character count").unwrap();
        let mut character_ids = Vec::with_capacity(character_count as usize);
        for _ in 0..character_count {
            character_ids.push(r.u32("character id").unwrap());
        }
        let switch_highest_number = r.u32("switch highest number").unwrap();
        let switch_team_count = r.u32("switch team count").unwrap();
        let mut switch_team_ids = Vec::with_capacity(switch_team_count as usize);
        for _ in 0..switch_team_count {
            switch_team_ids.push(r.i32("switch team id").unwrap());
        }
        let tick_count = r.u32("tick count").unwrap();
        TraceBReader {
            r,
            metadata_json,
            character_ids,
            switch_highest_number,
            switch_team_count,
            switch_team_ids,
            tick_count,
            ticks_read: 0,
        }
    }

    pub fn next_tick(&mut self) -> Option<TraceBTick> {
        if self.ticks_read >= self.tick_count {
            return None;
        }
        self.ticks_read += 1;
        let r = &mut self.r;
        let game_tick = r.i32("game_tick").unwrap();
        let switch_slots = (self.switch_team_count * self.switch_highest_number) as usize;
        let mut switches = Vec::with_capacity(switch_slots);
        for _ in 0..switch_slots {
            switches.push(SwitchEntry {
                status: r.i32("switch status").unwrap(),
                end_tick: r.i32("switch end_tick").unwrap(),
                kind: r.i32("switch kind").unwrap(),
                last_update_tick: r.i32("switch last_update_tick").unwrap(),
            });
        }
        let entity_count = r.u32("entity count").unwrap();
        let mut entities = Vec::with_capacity(entity_count as usize);
        for _ in 0..entity_count {
            entities.push(EntityRecord {
                kind: r.i32("kind").unwrap(),
                owner_client_id: r.i32("owner_client_id").unwrap(),
                weapon_type: r.i32("weapon_type").unwrap(),
                pos_x: r.f32("pos_x").unwrap(),
                pos_y: r.f32("pos_y").unwrap(),
                dir_x: r.f32("dir_x").unwrap(),
                dir_y: r.f32("dir_y").unwrap(),
                start_tick: r.i32("start_tick").unwrap(),
                extra: r.i32("extra").unwrap(),
            });
        }
        let mut characters = Vec::with_capacity(self.character_ids.len());
        for _ in 0..self.character_ids.len() {
            let direction = r.i32("direction").unwrap();
            let target_x = r.i32("target_x").unwrap();
            let target_y = r.i32("target_y").unwrap();
            let jump = r.i32("jump").unwrap();
            let fire = r.i32("fire").unwrap();
            let hook = r.i32("hook").unwrap();
            let player_flags = r.i32("player_flags").unwrap();
            let wanted_weapon = r.i32("wanted_weapon").unwrap();
            let next_weapon = r.i32("next_weapon").unwrap();
            let prev_weapon = r.i32("prev_weapon").unwrap();
            let kill = r.i32("kill").unwrap();
            let input = ScenarioV3Input {
                direction,
                target_x,
                target_y,
                jump,
                fire,
                hook,
                player_flags,
                wanted_weapon,
                next_weapon,
                prev_weapon,
                kill,
            };
            let core = read_core_state(r);
            let ddrace = read_ddrace_state(r);
            characters.push(CharacterRow { input, core, ddrace });
        }
        Some(TraceBTick {
            game_tick,
            switches,
            entities,
            characters,
        })
    }
}

/// Pulls `"seed": <N>` out of trace-b's metadata JSON (`docs/formats.md` §11.1) without a JSON
/// dependency — the field is always a bare integer at the top level in every producer this file
/// reads (`ddnet-oracle-b`).
pub fn metadata_seed(metadata_json: &str) -> u64 {
    let key = "\"seed\":";
    let start = metadata_json.find(key).expect("metadata_json has no \"seed\" field") + key.len();
    let rest = metadata_json[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().expect("seed is not a valid integer")
}
