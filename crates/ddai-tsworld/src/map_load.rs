//! Literal port of `src/map/loadMap.ts`'s `loadMapCollision`/`readMapSettings` (TS,
//! `Wranked1/DDNet-AI`, GPL-3.0) — building a [`Collision`] from a real DDNet `.map` file.
//!
//! **Restructuring vs. TS:** TS's `loadMapCollision` does its own byte-level datafile parsing
//! (RLE tile-skip unpacking, per-layer offset tables — `src/map/datafile.ts`,
//! `src/map/loadMap.ts:108-243`) because the old bot had no other `.map` reader. This crate
//! reuses `ddai-map` (task 1.4) for that instead (per the task spec: "Depends on ... `ddai-map`
//! for reading `.map` files") — [`ddai_map::load_map`] already performs the *same* decoding
//! TS's `loadMapCollision` does (tile-skip unpacking only for the game layer, fixed-size records
//! for tele/speedup/front — see `ddai-map`'s own crate docs, cross-checked against DDNet's C++
//! loader independently of this crate), so building a [`Collision`] from its
//! [`ddai_map::LoadedMap`] output is equivalent to TS's own parsing, not a second independent
//! implementation of the datafile format. What *is* ported here field-for-field is the part
//! `ddai-map` has no equivalent for: turning that already-decoded data into `Collision`'s
//! constructor arguments, and `readMapSettings`'s `sv_no_weak_hook` parsing.
//!
//! **Known divergence (documented, not exercised):** `ddai-map` decodes the Settings blob as
//! UTF-8 (lossy); TS decodes it as `latin1` (`loadMap.ts:94`). These agree byte-for-byte for
//! every ASCII byte (`< 0x80`), which is all `sv_no_weak_hook <0|1>` and every other real map's
//! settings command in this crate's corpus ever uses — a setting containing a raw byte `>= 0x80`
//! would decode differently, but no map this crate loads has one.

use crate::collision::{Collision, CollisionExtras, SpeedupLayer, TeleLayer};

/// The result of loading a `.map` file the way `SimWorld` needs it: a ready [`Collision`] plus
/// the `sv_no_weak_hook` setting (`loadMapCollision`'s `LoadedMap.settings.noWeakHook`,
/// `loadMap.ts:79`), which callers pass into [`crate::world::SimWorldOptions::no_weak_hook`] (or
/// rely on [`crate::world::SimWorld::new`]'s own `collision.no_weak_hook` fallback).
#[derive(Debug, Clone)]
pub struct LoadedTsMap {
    pub collision: Collision,
    pub width: i32,
    pub height: i32,
    pub no_weak_hook: bool,
}

/// Reads a `.map` file's bytes into a [`LoadedTsMap`]. Thin wrapper around
/// [`ddai_map::load_map`] + [`collision_from_loaded_map`].
pub fn load_map_bytes(bytes: &[u8]) -> Result<LoadedTsMap, ddai_map::MapError> {
    let loaded = ddai_map::load_map(bytes)?;
    Ok(collision_from_loaded_map(&loaded))
}

/// `loadMapCollision`'s `Collision` construction (`loadMap.ts:108-242`), given `ddai-map`'s
/// already-decoded [`ddai_map::LoadedMap`] instead of TS's own raw datafile bytes.
pub fn collision_from_loaded_map(loaded: &ddai_map::LoadedMap) -> LoadedTsMap {
    let data = &loaded.data;
    let width = data.width as i32;
    let height = data.height as i32;

    let mut tiles = Vec::with_capacity(data.game.len());
    let mut tile_flags = Vec::with_capacity(data.game.len());
    for t in &data.game {
        tiles.push(t.index);
        tile_flags.push(t.flags);
    }

    let front = data.front.as_ref().map(|f| {
        let mut index = Vec::with_capacity(f.len());
        let mut flags = Vec::with_capacity(f.len());
        for t in f {
            index.push(t.index);
            flags.push(t.flags);
        }
        (index, flags)
    });

    let tele = data.tele.as_ref().map(|t| TeleLayer {
        types: t.iter().map(|x| x.kind).collect(),
        numbers: t.iter().map(|x| x.number).collect(),
    });

    let speedup = data.speedup.as_ref().map(|s| SpeedupLayer {
        force: s.iter().map(|x| x.force).collect(),
        max_speed: s.iter().map(|x| x.max_speed).collect(),
        angle: s.iter().map(|x| x.angle).collect(),
    });

    let no_weak_hook = no_weak_hook_from_settings(&data.settings);

    let collision = Collision::new(
        width,
        height,
        tiles,
        tele,
        speedup,
        Some(CollisionExtras {
            tile_flags: Some(tile_flags),
            front,
            no_weak_hook,
        }),
    );

    LoadedTsMap {
        collision,
        width,
        height,
        no_weak_hook,
    }
}

/// `readMapSettings` (`loadMap.ts:81-106`), given the settings lines `ddai-map` already
/// NUL-split for us (see this module's top-level doc comment).
fn no_weak_hook_from_settings(settings: &[String]) -> bool {
    let mut result = false;
    for line in settings {
        for command in line.split(';') {
            // `split_whitespace` already trims leading/trailing whitespace on its own (matches
            // TS's `command.trim().split(/\s+/)` for any input that isn't the empty string,
            // which no `sv_no_weak_hook` command in practice is).
            let words: Vec<&str> = command.split_whitespace().collect();
            if words.first().copied() != Some("sv_no_weak_hook") || words.len() < 2 {
                continue;
            }
            if let Some(value) = js_parse_int_leading(words[1]) {
                result = value.clamp(0, 1) == 1;
            }
            // `Number.parseInt` returning `NaN` (no leading numeric prefix) leaves `settings`
            // unchanged in TS (`Number.isFinite(NaN)` is `false`) — `js_parse_int_leading`
            // returning `None` here does the same (no assignment).
        }
    }
    result
}

/// `Number.parseInt(s, 10)`'s leading-integer-prefix behavior (`"1"` -> `1`, `"1abc"` -> `1`,
/// `"abc"` -> `NaN`), restricted to what `words[1]` (already whitespace-trimmed and
/// whitespace-split) can contain. Returns `None` for `NaN` (no valid leading sign+digits).
fn js_parse_int_leading(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut sign = 1i64;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        if bytes[i] == b'-' {
            sign = -1;
        }
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return None;
    }
    s[start..i].parse::<i64>().ok().map(|v| v * sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_int_leading_matches_js_parse_int() {
        assert_eq!(js_parse_int_leading("1"), Some(1));
        assert_eq!(js_parse_int_leading("0"), Some(0));
        assert_eq!(js_parse_int_leading("1abc"), Some(1));
        assert_eq!(js_parse_int_leading("-1"), Some(-1));
        assert_eq!(js_parse_int_leading("abc"), None);
        assert_eq!(js_parse_int_leading(""), None);
    }

    #[test]
    fn no_weak_hook_parses_semicolon_separated_commands() {
        let settings = vec!["sv_map foo;sv_no_weak_hook 1".to_string()];
        assert!(no_weak_hook_from_settings(&settings));
        let settings = vec!["sv_no_weak_hook 0".to_string()];
        assert!(!no_weak_hook_from_settings(&settings));
        let settings: Vec<String> = vec![];
        assert!(!no_weak_hook_from_settings(&settings));
        // Clamped: values above 1 also count as "true" (Math.min(1, Math.max(0, value)) === 1).
        let settings = vec!["sv_no_weak_hook 5".to_string()];
        assert!(no_weak_hook_from_settings(&settings));
        // Last matching command wins if it appears more than once.
        let settings = vec!["sv_no_weak_hook 1;sv_no_weak_hook 0".to_string()];
        assert!(!no_weak_hook_from_settings(&settings));
    }
}
