//! Synthetic builders for tests (no demo bytes, no map bytes and no nicknames of anyone real).

use ddai_net::generated::objects;
use ddai_physics::map::{MapData, TILE_FREEZE, TILE_SOLID, Tile};

/// A wire `Character` at rest at pixel `(x, y)` reckoned at `tick`.
pub fn wire_character(tick: i32, x: i32, y: i32) -> objects::Character {
    objects::Character {
        tick,
        x,
        y,
        vel_x: 0,
        vel_y: 0,
        angle: 0,
        direction: 0,
        jumped: 0,
        hooked_player: -1,
        hook_state: 0,
        hook_tick: 0,
        hook_x: x,
        hook_y: y,
        hook_dx: 0,
        hook_dy: 0,
        player_flags: 0,
        health: 0,
        armor: 0,
        ammo_count: 0,
        weapon: 0,
        emote: 0,
        attack_tick: 0,
    }
}

pub fn player_info(id: i32) -> objects::PlayerInfo {
    objects::PlayerInfo {
        local: 0,
        client_id: id,
        team: 0,
        score: 0,
        latency: 0,
    }
}

pub fn client_info(name: &str) -> objects::ClientInfo {
    objects::ClientInfo {
        name: name.to_string(),
        clan: "synthetic-clan".to_string(),
        country: -1,
        skin: "default".to_string(),
        use_custom_color: 0,
        color_body: 0,
        color_feet: 0,
    }
}

/// A flat test arena, `w x h` tiles: solid floor on the bottom row and solid side walls; a pit of
/// freeze tiles in the floor from column `pit.0` (inclusive) to `pit.1` (exclusive). Tile size is
/// 32 px, so the floor surface is at `y = (h - 1) * 32`.
pub fn arena_with_pit(w: u32, h: u32, pit: (u32, u32)) -> MapData {
    let mut game = vec![Tile::default(); (w * h) as usize];
    for x in 0..w {
        let idx = if x >= pit.0 && x < pit.1 {
            TILE_FREEZE
        } else {
            TILE_SOLID
        };
        game[((h - 1) * w + x) as usize] = Tile {
            index: idx,
            ..Default::default()
        };
    }
    for y in 0..h {
        for x in [0, w - 1] {
            game[(y * w + x) as usize] = Tile {
                index: TILE_SOLID,
                ..Default::default()
            };
        }
    }
    MapData {
        width: w,
        height: h,
        game,
        front: None,
        tele: None,
        speedup: None,
        switch: None,
        tune: None,
        settings: Vec::new(),
    }
}

/// Like [`arena_with_pit`] but with a freeze ceiling across columns `ceil.0..ceil.1` on row 0.
pub fn arena_with_pit_and_ceiling(w: u32, h: u32, pit: (u32, u32), ceil: (u32, u32)) -> MapData {
    let mut m = arena_with_pit(w, h, pit);
    for x in ceil.0..ceil.1 {
        m.game[x as usize] = Tile {
            index: TILE_FREEZE,
            ..Default::default()
        };
    }
    m
}
