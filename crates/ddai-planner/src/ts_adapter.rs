//! `PlanWorld`/`PlanCollision` for `ddai_tsworld::SimWorld` (feature `ts-parity`) — the backend
//! that proves the port is decision-for-decision identical to the real TS planner (acceptance
//! criterion 3). Every field of this crate's own [`crate::types::PlayerInput`]/
//! [`crate::types::TeeState`] matches `ddai-tsworld`'s equivalent 1:1 (both are themselves literal
//! ports of the same `src/core/types.ts`), so this adapter is a plain field-for-field conversion,
//! never a re-derivation of any value.

use crate::plan_world::{CFLAG_DEATH, CFLAG_NOHOOK, CFLAG_SOLID, LineHit, PlanCollision, PlanWorld};
use crate::types::{PlayerInput, TeeState, WorldEvent};
use crate::vmath::Vec2;

fn to_ts_vec2(v: Vec2) -> ddai_tsworld::Vec2 {
    ddai_tsworld::Vec2 { x: v.x, y: v.y }
}

fn from_ts_vec2(v: ddai_tsworld::Vec2) -> Vec2 {
    Vec2 { x: v.x, y: v.y }
}

fn to_ts_input(i: &PlayerInput) -> ddai_tsworld::PlayerInput {
    ddai_tsworld::PlayerInput {
        direction: i.direction,
        target_x: i.target_x,
        target_y: i.target_y,
        jump: i.jump,
        fire: i.fire,
        hook: i.hook,
        player_flags: i.player_flags,
        wanted_weapon: i.wanted_weapon,
        next_weapon: i.next_weapon,
        prev_weapon: i.prev_weapon,
    }
}

fn from_ts_tee(t: ddai_tsworld::TeeState) -> TeeState {
    TeeState {
        id: t.id,
        alive: t.alive,
        pos: from_ts_vec2(t.pos),
        vel: from_ts_vec2(t.vel),
        hook_state: t.hook_state,
        hook_pos: from_ts_vec2(t.hook_pos),
        hook_dir: from_ts_vec2(t.hook_dir),
        hooked_player: t.hooked_player,
        jumped: t.jumped,
        jumps_left: t.jumps_left,
        direction: t.direction,
        angle: t.angle,
        active_weapon: t.active_weapon,
        frozen: t.frozen,
        freeze_ticks_left: t.freeze_ticks_left,
        attack_tick: t.attack_tick,
        hook_tick: t.hook_tick,
        jumped_total: t.jumped_total,
        reload_ticks: t.reload_ticks,
        frozen_for: t.frozen_for,
        deep_frozen: t.deep_frozen,
        jumps: t.jumps,
        ddnet_flags: t.ddnet_flags,
        since_attack: t.since_attack,
    }
}

fn to_ts_tee(t: &TeeState) -> ddai_tsworld::TeeState {
    ddai_tsworld::TeeState {
        id: t.id,
        alive: t.alive,
        pos: to_ts_vec2(t.pos),
        vel: to_ts_vec2(t.vel),
        hook_state: t.hook_state,
        hook_pos: to_ts_vec2(t.hook_pos),
        hook_dir: to_ts_vec2(t.hook_dir),
        hooked_player: t.hooked_player,
        jumped: t.jumped,
        jumps_left: t.jumps_left,
        direction: t.direction,
        angle: t.angle,
        active_weapon: t.active_weapon,
        frozen: t.frozen,
        freeze_ticks_left: t.freeze_ticks_left,
        attack_tick: t.attack_tick,
        hook_tick: t.hook_tick,
        jumped_total: t.jumped_total,
        reload_ticks: t.reload_ticks,
        frozen_for: t.frozen_for,
        deep_frozen: t.deep_frozen,
        jumps: t.jumps,
        ddnet_flags: t.ddnet_flags,
        since_attack: t.since_attack,
    }
}

fn from_ts_event(e: ddai_tsworld::WorldEvent) -> WorldEvent {
    match e {
        ddai_tsworld::WorldEvent::HammerHit { from, to } => WorldEvent::HammerHit { from, to },
        ddai_tsworld::WorldEvent::HammerFire { from, hits } => WorldEvent::HammerFire { from, hits },
        ddai_tsworld::WorldEvent::Death { id, by } => WorldEvent::Death { id, by },
        ddai_tsworld::WorldEvent::Explosion { .. }
        | ddai_tsworld::WorldEvent::LaserHit { .. }
        | ddai_tsworld::WorldEvent::Freeze { .. } => WorldEvent::Other,
    }
}

impl PlanCollision for ddai_tsworld::Collision {
    fn identity(&self) -> u64 {
        (std::ptr::from_ref(self)) as u64
    }
    fn width(&self) -> i32 {
        self.width
    }
    fn height(&self) -> i32 {
        self.height
    }
    fn game_tile(&self, tx: i32, ty: i32) -> u8 {
        self.tiles[(ty * self.width + tx) as usize]
    }
    fn is_solid(&self, x: f64, y: f64) -> bool {
        ddai_tsworld::Collision::is_solid(self, x, y)
    }
    fn is_death(&self, x: f64, y: f64) -> bool {
        ddai_tsworld::Collision::is_death(self, x, y)
    }
    fn is_freeze(&self, x: f64, y: f64) -> bool {
        ddai_tsworld::Collision::is_freeze(self, x, y)
    }
    fn is_un_freeze(&self, x: f64, y: f64) -> bool {
        ddai_tsworld::Collision::is_un_freeze(self, x, y)
    }
    fn is_no_hook(&self, x: f64, y: f64) -> bool {
        ddai_tsworld::Collision::is_no_hook(self, x, y)
    }
    fn test_box(&self, pos: Vec2, size: Vec2) -> bool {
        ddai_tsworld::Collision::test_box(self, to_ts_vec2(pos), to_ts_vec2(size))
    }
    fn intersect_line(&self, pos0: Vec2, pos1: Vec2) -> LineHit {
        let h = ddai_tsworld::Collision::intersect_line(self, to_ts_vec2(pos0), to_ts_vec2(pos1));
        LineHit {
            collision: h.collision,
            out_pos: from_ts_vec2(h.out_pos),
            out_before_pos: from_ts_vec2(h.out_before_pos),
        }
    }
    fn intersect_line_hook(&self, pos0: Vec2, pos1: Vec2) -> LineHit {
        let h = ddai_tsworld::Collision::intersect_line_hook(self, to_ts_vec2(pos0), to_ts_vec2(pos1));
        LineHit {
            collision: h.collision,
            out_pos: from_ts_vec2(h.out_pos),
            out_before_pos: from_ts_vec2(h.out_before_pos),
        }
    }
    fn has_tele(&self) -> bool {
        ddai_tsworld::Collision::has_tele(self)
    }
    fn tele_at(&self, x: f64, y: f64) -> (i32, i32) {
        ddai_tsworld::Collision::tele_at(self, x, y)
    }
    fn tele_outs_for(&self, number: i32) -> Vec<Vec2> {
        ddai_tsworld::Collision::tele_outs_for(self, number)
            .iter()
            .map(|&v| from_ts_vec2(v))
            .collect()
    }
}

// Sanity check: this adapter's `LineHit::collision` bitmask must line up with
// `crate::plan_world`'s `CFLAG_*` constants -- both sides use DDNet's own convention
// (`CFLAG_SOLID = 1`, `CFLAG_DEATH = 2`, `CFLAG_NOHOOK = 4`), so no translation is needed; this
// `const` block just fails to compile if that convention ever drifts between the two crates.
const _: () = {
    assert!(CFLAG_SOLID == ddai_tsworld::tuning::CFLAG_SOLID);
    assert!(CFLAG_DEATH == ddai_tsworld::tuning::CFLAG_DEATH);
    assert!(CFLAG_NOHOOK == ddai_tsworld::tuning::CFLAG_NOHOOK);
};

impl PlanWorld for ddai_tsworld::SimWorld {
    type Collision = ddai_tsworld::Collision;
    type SavedState = ddai_tsworld::world::SimState;

    fn collision(&self) -> &Self::Collision {
        &self.collision
    }
    fn tick(&self) -> i64 {
        self.tick
    }

    fn get_tee(&self, id: i32) -> Option<TeeState> {
        ddai_tsworld::SimWorld::get_tee(self, id).map(from_ts_tee)
    }
    fn read_tee(&self, id: i32, out: &mut TeeState) -> bool {
        let mut buf = ddai_tsworld::types::blank_tee_state();
        if ddai_tsworld::SimWorld::read_tee(self, id, &mut buf) {
            *out = from_ts_tee(buf);
            true
        } else {
            false
        }
    }
    fn all_tees(&self) -> Vec<TeeState> {
        ddai_tsworld::SimWorld::all_tees(self)
            .into_iter()
            .map(from_ts_tee)
            .collect()
    }

    fn set_input(&mut self, id: i32, input: PlayerInput) {
        ddai_tsworld::SimWorld::set_input(self, id, to_ts_input(&input));
    }
    fn set_held_input(&mut self, id: i32, input: PlayerInput) {
        ddai_tsworld::SimWorld::set_held_input(self, id, to_ts_input(&input));
    }
    fn step(&mut self) -> Vec<WorldEvent> {
        ddai_tsworld::SimWorld::step(self)
            .into_iter()
            .map(from_ts_event)
            .collect()
    }

    fn save_state(&self) -> Self::SavedState {
        ddai_tsworld::SimWorld::save_state(self)
    }
    fn save_state_into(&self, into: &mut Self::SavedState) {
        ddai_tsworld::SimWorld::save_state_into(self, into);
    }
    fn restore_state(&mut self, state: &Self::SavedState) {
        ddai_tsworld::SimWorld::restore_state(self, state);
    }

    fn apply_tee_state(&mut self, id: i32, st: &TeeState) {
        ddai_tsworld::SimWorld::apply_tee_state(self, id, &to_ts_tee(st));
    }
    fn add_tee(&mut self, id: i32, pos: Vec2) {
        ddai_tsworld::SimWorld::add_tee(self, id, to_ts_vec2(pos));
    }
    fn remove_tee(&mut self, id: i32) {
        ddai_tsworld::SimWorld::remove_tee(self, id);
    }

    fn apply_force(&mut self, id: i32, force: Vec2) {
        ddai_tsworld::SimWorld::apply_force(self, id, to_ts_vec2(force));
    }
    fn unfreeze(&mut self, id: i32) {
        ddai_tsworld::SimWorld::unfreeze(self, id);
    }

    /// `new SimWorld(collision, {svHit:true, respawnDelayTicks:0, infiniteAmmo:true})`
    /// (`planner.ts:1873-1876`).
    fn new_scratch(&self) -> Self {
        ddai_tsworld::SimWorld::new(
            self.collision.clone(),
            ddai_tsworld::world::SimWorldOptions {
                respawn_delay_ticks: Some(0),
                infinite_ammo: Some(true),
                sv_hit: Some(true),
                all_weapons: None,
                no_weak_hook: None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_collision() -> ddai_tsworld::Collision {
        // 4x4, all air except a solid floor on the bottom row.
        let w = 4;
        let h = 4;
        let mut tiles = vec![0u8; (w * h) as usize];
        for x in 0..w {
            tiles[((h - 1) * w + x) as usize] = 1; // TILE_SOLID
        }
        ddai_tsworld::Collision::new(w, h, tiles, None, None, None)
    }

    #[test]
    fn round_trips_a_tee_through_apply_and_get() {
        let col = tiny_collision();
        let mut world = ddai_tsworld::SimWorld::new(col, ddai_tsworld::world::SimWorldOptions::default());
        <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut world, 0, Vec2 { x: 50.0, y: 50.0 });
        let mut st = crate::types::blank_tee_state();
        st.id = 0;
        st.alive = true;
        st.pos = Vec2 { x: 12.0, y: 34.0 };
        <ddai_tsworld::SimWorld as PlanWorld>::apply_tee_state(&mut world, 0, &st);
        let back = <ddai_tsworld::SimWorld as PlanWorld>::get_tee(&world, 0).unwrap();
        assert_eq!(back.pos, Vec2 { x: 12.0, y: 34.0 });
    }

    #[test]
    fn new_scratch_shares_the_collision_but_starts_with_no_tees() {
        let col = tiny_collision();
        let mut world = ddai_tsworld::SimWorld::new(col, ddai_tsworld::world::SimWorldOptions::default());
        <ddai_tsworld::SimWorld as PlanWorld>::add_tee(&mut world, 0, Vec2 { x: 50.0, y: 50.0 });
        let scratch = <ddai_tsworld::SimWorld as PlanWorld>::new_scratch(&world);
        assert!(<ddai_tsworld::SimWorld as PlanWorld>::get_tee(&scratch, 0).is_none());
        assert_eq!(scratch.collision.width, world.collision.width);
    }

    #[test]
    fn collision_identity_is_stable_and_distinguishes_different_collisions() {
        let a = tiny_collision();
        let b = tiny_collision();
        assert_eq!(PlanCollision::identity(&a), PlanCollision::identity(&a));
        assert_ne!(PlanCollision::identity(&a), PlanCollision::identity(&b));
    }
}
