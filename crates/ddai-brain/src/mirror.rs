//! X-flip ("mirror") helpers for [`crate::observation::Observation`] (task 7.3, acceptance
//! criterion 1's "mirror helpers"; acceptance criterion 8's mirror-symmetry test builds directly
//! on this). [`Action::mirror_x`](crate::action::Action::mirror_x) lives with `Action` itself
//! (`action.rs`) since it needs no map/width context; this module holds the map- and
//! character-state mirroring, which does.
//!
//! Mirroring is a **single reflection through the map's own vertical centerline**
//! (`x -> width_px - x`), applied consistently to the map's tiles *and* every character's
//! position/velocity/hook state — not just negating coordinates around each character's own
//! position. This matters for a real, left-right-*asymmetric* map: reflecting only the
//! characters (leaving the tiles alone) would have a mirrored fly "seeing" geometry that doesn't
//! correspond to any real, consistent world, which would make the acceptance criterion 8 mirror
//! test meaningless on anything but a hand-built symmetric fixture. Reflecting both keeps ray
//! casting against `Observation::map` and every position/velocity self-consistent, so the mirror
//! test genuinely exercises "does the encoder + connectome + decoder pipeline commute with an
//! X-flip", on any map, real or synthetic.
//!
//! **Scope of the mirrored map**: only the `game`/`front` tile layers are reflected (all the
//! ray-casting/tile-class encoder logic ever reads — see `ddai-fly`'s `encoder` module); `tele`/
//! `speedup`/`switch`/`tune` are dropped (`None`) in the mirrored copy rather than being
//! (incorrectly) carried over unreflected, since correctly mirroring a teleporter's *destination*
//! or a switch door's *direction* is out of scope for what any current consumer of a mirrored
//! `Observation` needs.

use std::sync::Arc;

use ddai_physics::map::MapData;

use crate::observation::{CharacterObservation, Observation};

/// Reverses each row of a tile layer in place order (a `width`-wide, `height`-tall row-major
/// grid), producing the X-mirrored layer. Only `.index` (tile class) needs to survive this for
/// every current consumer (see the module doc comment) — `.flags`/`.skip`/`.reserved` are carried
/// along unchanged, so a rotation-flag-sensitive reader (none of this crate's own code is) would
/// see the original tile's orientation at its new, mirrored position, not a corrected one.
fn mirror_tile_layer<T: Copy>(layer: &[T], width: usize, height: usize) -> Vec<T> {
    debug_assert_eq!(layer.len(), width * height);
    let mut out = layer.to_vec();
    for row in 0..height {
        out[row * width..(row + 1) * width].reverse();
    }
    out
}

/// X-mirrors a map's `game`/`front` tile layers (see the module doc comment for exactly what is
/// and isn't carried over). Public (not just used internally by [`Observation::mirror_x`])
/// because a caller building its own mirrored-world tests (e.g. `ddai-fly`'s encoder gradient
/// checks) may want the mirrored map without going through a whole `Observation`.
pub fn mirror_map_data(map: &MapData) -> MapData {
    let width = map.width as usize;
    let height = map.height as usize;
    MapData {
        width: map.width,
        height: map.height,
        game: mirror_tile_layer(&map.game, width, height),
        front: map.front.as_ref().map(|f| mirror_tile_layer(f, width, height)),
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: map.settings.clone(),
    }
}

impl CharacterObservation {
    /// X-mirrors this character's spatial fields through the vertical line `x = width_px / 2`
    /// (`width_px` is the map's width in pixels, `map.width * 32` — pass
    /// [`Observation::mirror_x`]'s own `width_px`, not half of it: the reflection formula is
    /// `x' = width_px - x`, matching [`mirror_map_data`]'s tile-column reversal exactly, so a
    /// position and the map it's read against stay consistent after mirroring). `id`/`team`/
    /// `hooked_player` (another character's *identity*, not a spatial quantity) and every
    /// non-spatial field (freeze/jump/weapon/...) are unchanged.
    pub fn mirror_x(&self, width_px: f32) -> Self {
        CharacterObservation {
            pos: ddai_physics::vmath::Vec2::new(width_px - self.pos.x, self.pos.y),
            vel: ddai_physics::vmath::Vec2::new(-self.vel.x, self.vel.y),
            hook_pos: ddai_physics::vmath::Vec2::new(width_px - self.hook_pos.x, self.hook_pos.y),
            direction: -self.direction,
            ..*self
        }
    }
}

impl Observation {
    /// X-mirrors the whole observation: the map ([`mirror_map_data`]) plus every character's
    /// spatial fields ([`CharacterObservation::mirror_x`]), through the same reflection axis
    /// (the map's own vertical centerline) for both — see the module doc comment for why both
    /// must move together. `tick`/`tuning`/`target_id` are unchanged (none has an X-orientation
    /// — `target_id` is an *identity* reference into `others`, same as `CharacterObservation::
    /// mirror_x` already leaves `id`/`team`/`hooked_player` unchanged for the same reason).
    pub fn mirror_x(&self) -> Observation {
        let width_px = self.map.width as f32 * 32.0;
        Observation {
            map: Arc::new(mirror_map_data(&self.map)),
            tick: self.tick,
            self_state: self.self_state.mirror_x(width_px),
            others: self.others.iter().map(|c| c.mirror_x(width_px)).collect(),
            target_id: self.target_id,
            tuning: self.tuning,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::map::{TILE_AIR, TILE_SOLID, Tile};
    use ddai_physics::vmath::Vec2;

    fn tile(index: u8) -> Tile {
        Tile {
            index,
            ..Default::default()
        }
    }

    fn asymmetric_map() -> MapData {
        // 4x1: solid wall only on the far left -- a genuinely asymmetric map, so a mirror test
        // against it can't pass by accident (a symmetric fixture would).
        MapData {
            width: 4,
            height: 1,
            game: vec![tile(TILE_SOLID), tile(TILE_AIR), tile(TILE_AIR), tile(TILE_AIR)],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn mirror_map_data_reverses_the_row() {
        let mirrored = mirror_map_data(&asymmetric_map());
        assert_eq!(
            mirrored.game.iter().map(|t| t.index).collect::<Vec<_>>(),
            vec![TILE_AIR, TILE_AIR, TILE_AIR, TILE_SOLID],
            "the solid tile must move from the leftmost to the rightmost column"
        );
    }

    #[test]
    fn mirror_map_data_drops_the_layers_it_does_not_correctly_mirror() {
        let mut map = asymmetric_map();
        map.tele = Some(vec![Default::default(); 4]);
        let mirrored = mirror_map_data(&map);
        assert!(mirrored.tele.is_none());
    }

    #[test]
    fn mirroring_twice_returns_the_original_map() {
        let map = asymmetric_map();
        let twice = mirror_map_data(&mirror_map_data(&map));
        assert_eq!(twice.game, map.game);
    }

    #[test]
    fn observation_mirror_x_flips_position_velocity_and_direction() {
        let map = Arc::new(asymmetric_map()); // width_px = 128
        let mut me = CharacterObservation::at_rest(0);
        me.pos = Vec2::new(10.0, 5.0);
        me.vel = Vec2::new(3.0, -1.0);
        me.direction = 1;
        let obs = Observation {
            map,
            tick: 7,
            self_state: me,
            others: vec![],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };

        let mirrored = obs.mirror_x();
        assert_eq!(mirrored.self_state.pos, Vec2::new(128.0 - 10.0, 5.0));
        assert_eq!(mirrored.self_state.vel, Vec2::new(-3.0, -1.0));
        assert_eq!(mirrored.self_state.direction, -1);
        assert_eq!(mirrored.tick, 7);

        // Mirroring twice returns to the original position (float-exact here: only a subtraction
        // is involved, no accumulated rounding).
        let back = mirrored.mirror_x();
        assert_eq!(back.self_state.pos, obs.self_state.pos);
    }

    #[test]
    fn observation_mirror_x_leaves_identity_and_non_spatial_fields_alone() {
        let map = Arc::new(asymmetric_map());
        let mut other = CharacterObservation::at_rest(3);
        other.team = 2;
        other.hooked_player = 0;
        other.jumps_left = 1;
        other.weapon = 4;
        let obs = Observation {
            map,
            tick: 0,
            self_state: CharacterObservation::at_rest(0),
            others: vec![other],
            target_id: None,
            tuning: ddai_physics::tuning::TuningParams::default(),
        };
        let mirrored = obs.mirror_x();
        let m = &mirrored.others[0];
        assert_eq!(m.id, 3);
        assert_eq!(m.team, 2);
        assert_eq!(m.hooked_player, 0);
        assert_eq!(m.jumps_left, 1);
        assert_eq!(m.weapon, 4);
    }
}
