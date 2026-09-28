//! `MapScene`: a coarse, renderer-friendly classification of every tile in a map into a small
//! "kind" palette (acceptance criterion 1), built once per distinct map (keyed by sha256, see
//! `crate::live::map_cache`) and served compressed at `GET /api/map/<sha256>`.
//!
//! This is deliberately *not* a physics engine: it never reproduces the exact solidity/hook/
//! freeze rules `CCollision` implements (that is task 1.6's job, ported from the same DDNet
//! source this module cites). It answers a much narrower question a background renderer needs:
//! "what should this one cell look like, as a single flat color", picking exactly one [`Kind`]
//! per cell even where the real game layers several independent effects on top of each other.
//! See [`classify`] for the precedence this module uses to pick that one value, and
//! `docs/formats.md`'s new section for the citations backing each rule.

use ddai_physics::map::{self, MapData};

/// A single coarse "kind" for one map cell — the renderer-facing palette (acceptance criterion 1
/// lists exactly this set: "air, solid, nohook, freeze, deep freeze, undeep, unfreeze, death,
/// stopper, tele-in/out/checkpoint, speedup, switch/door, tune zone, spawn").
///
/// `#[repr(u8)]` and contiguous from 0 — this is exactly the wire value `/api/map/<sha256>`
/// sends per cell (docs/formats.md), so adding a new variant is a wire-format change (append
/// only; never renumber an existing one, existing clients decode by raw byte value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Air = 0,
    Solid = 1,
    NoHook = 2,
    Death = 3,
    DeepFreeze = 4,
    Freeze = 5,
    DeepUnfreeze = 6,
    Unfreeze = 7,
    TeleIn = 8,
    TeleOut = 9,
    TeleCheckpoint = 10,
    Speedup = 11,
    Switch = 12,
    TuneZone = 13,
    Stopper = 14,
    Spawn = 15,
}

impl Kind {
    pub const COUNT: usize = 16;

    pub fn from_u8(v: u8) -> Option<Kind> {
        Some(match v {
            0 => Kind::Air,
            1 => Kind::Solid,
            2 => Kind::NoHook,
            3 => Kind::Death,
            4 => Kind::DeepFreeze,
            5 => Kind::Freeze,
            6 => Kind::DeepUnfreeze,
            7 => Kind::Unfreeze,
            8 => Kind::TeleIn,
            9 => Kind::TeleOut,
            10 => Kind::TeleCheckpoint,
            11 => Kind::Speedup,
            12 => Kind::Switch,
            13 => Kind::TuneZone,
            14 => Kind::Stopper,
            15 => Kind::Spawn,
            _ => return None,
        })
    }
}

/// Game-layer tile indices for spawn markers (`ENTITY_OFFSET + ENTITY_SPAWN{,_RED,_BLUE}`,
/// `mapitems.h`: `ENTITY_OFFSET = 255 - 16*4 = 191`, `ENTITY_SPAWN = 1/2/3`) — these are outside
/// `ddai_physics::map`'s own `TILE_*` constants (which only cover the physics-relevant range),
/// since spawns are an entity-layer concept the physics collision code never reads at all.
const ENTITY_OFFSET: u8 = 191;
const ENTITY_SPAWN: u8 = ENTITY_OFFSET + 1;
const ENTITY_SPAWN_RED: u8 = ENTITY_OFFSET + 2;
const ENTITY_SPAWN_BLUE: u8 = ENTITY_OFFSET + 3;

/// Classifies one cell from its game-layer tile, front-layer tile (if any), and the special-layer
/// records at the same index. Ties are broken by a fixed precedence (documented at each arm
/// below and in `docs/formats.md`) since a real map can legitimately stack effects a flat
/// per-cell "kind" cannot show at once (e.g. a speedup tile inside a tune zone) — there is no
/// single "correct" order in DDNet itself (it draws every layer with its own transparency), this
/// is our own choice for a coarse background view.
///
/// `game`/`front` are the raw tile index bytes ([`ddai_physics::map::Tile::index`]) at this cell.
#[allow(clippy::too_many_arguments)]
fn classify_cell(
    game_index: u8,
    front_index: Option<u8>,
    tele_kind: Option<u8>,
    speedup_force: Option<u8>,
    switch_kind: Option<u8>,
    tune_number: Option<u8>,
) -> Kind {
    // 1. Solid/no-hook: game layer only (`CCollision::IsSolid`, `collision.cpp:605-608` —
    // `return Index == TILE_SOLID || Index == TILE_NOHOOK;`, checked directly against the source
    // again for review round 1, finding F7: `TILE_NOLASER` is NOT in that check at all, despite
    // sharing `GetTile()`'s raw `[TILE_SOLID, TILE_NOLASER]` return range with the two tiles that
    // ARE solid — it only ever blocks lasers (`CCollision::IsNoLaser`, unrelated to movement/hook)
    // and a player walks straight through it. The previous version of this classifier folded it
    // into `Solid` as what its own comment called "a harmless visual simplification" — it isn't
    // harmless: BlmapChill alone has 199 such tiles a player can walk through but this view drew
    // as an impassable wall. Falls through to `Air` below instead (a future, finer-grained view
    // could give it its own faint "blocks laser only" kind; not needed for this task's palette).
    if game_index == map::TILE_SOLID {
        return Kind::Solid;
    }
    if game_index == map::TILE_NOHOOK {
        return Kind::NoHook;
    }

    // 2. Death: exact match on EITHER layer (`character.cpp:1477-1484`, `GetCollisionAt`/
    // `GetFrontCollisionAt` both checked).
    if game_index == map::TILE_DEATH || front_index == Some(map::TILE_DEATH) {
        return Kind::Death;
    }

    // 3. Freeze family: exact match on EITHER layer (`character.cpp:1658-1677`,
    // `HandleTiles` — `m_TileIndex == TILE_FREEZE || m_TileFIndex == TILE_FREEZE`, and so on for
    // each of the other three). `TILE_LFREEZE`/`TILE_LUNFREEZE` (live freeze, `character.cpp:
    // 1672-1678`) look the same as plain freeze/unfreeze to a player watching this view, so they
    // fold into the same two visual kinds rather than adding a fifth "live frozen tile" kind the
    // acceptance criteria's palette doesn't ask for.
    let is = |index: u8| game_index == index || front_index == Some(index);
    if is(map::TILE_DFREEZE) {
        return Kind::DeepFreeze;
    }
    if is(map::TILE_FREEZE) || is(map::TILE_LFREEZE) {
        return Kind::Freeze;
    }
    if is(map::TILE_DUNFREEZE) {
        return Kind::DeepUnfreeze;
    }
    if is(map::TILE_UNFREEZE) || is(map::TILE_LUNFREEZE) {
        return Kind::Unfreeze;
    }

    // 4. Stopper: `TILE_STOP`/`TILE_STOPS`/`TILE_STOPA` on either layer (`collision.cpp:876-892`
    // checks both `m_pTiles` and `m_pFront` for all three). Direction (`TILE_STOP`'s rotation
    // flags) is not represented in this coarse per-cell palette — a future, finer-grained view
    // could read it straight from `MapData`'s own `Tile::flags`, already carried alongside.
    let is_stopper =
        |index: u8| matches!(index, i if i == map::TILE_STOP || i == map::TILE_STOPS || i == map::TILE_STOPA);
    if is_stopper(game_index) || front_index.is_some_and(is_stopper) {
        return Kind::Stopper;
    }

    // 5. Tele/speedup/switch/tune: these come from their own dedicated layers (independent grids
    // parallel to game/front, `ddai_physics::map::MapData`), not from the game/front tile index
    // at all — the game/front tile at this cell may well be `TILE_AIR` (visually walkable) while
    // the tele layer still marks it a teleporter. Checked in this order (arbitrary but fixed —
    // real maps essentially never stack more than one of these at the same cell).
    if let Some(kind) = tele_kind {
        if matches!(
            kind,
            k if k == map::TILE_TELEIN
                || k == map::TILE_TELEINEVIL
                || k == map::TILE_TELEINWEAPON
                || k == map::TILE_TELEINHOOK
        ) {
            return Kind::TeleIn;
        }
        if kind == map::TILE_TELEOUT {
            return Kind::TeleOut;
        }
        if matches!(
            kind,
            k if k == map::TILE_TELECHECK
                || k == map::TILE_TELECHECKOUT
                || k == map::TILE_TELECHECKIN
                || k == map::TILE_TELECHECKINEVIL
        ) {
            return Kind::TeleCheckpoint;
        }
    }
    if speedup_force.is_some_and(|f| f > 0) {
        return Kind::Speedup;
    }
    if switch_kind.is_some_and(|k| k != 0) {
        return Kind::Switch;
    }
    if tune_number.is_some_and(|n| n != 0) {
        return Kind::TuneZone;
    }

    // 6. Spawn markers: entity-layer indices on the game (or front) layer, outside the
    // physics-relevant `TILE_*` range entirely (see `ENTITY_SPAWN*` above).
    let is_spawn = |index: u8| matches!(index, ENTITY_SPAWN | ENTITY_SPAWN_RED | ENTITY_SPAWN_BLUE);
    if is_spawn(game_index) || front_index.is_some_and(is_spawn) {
        return Kind::Spawn;
    }

    Kind::Air
}

/// A classified map, ready to send to the client: one [`Kind`] byte per cell, row-major, same
/// `width`/`height` as the source [`MapData`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapScene {
    pub width: u32,
    pub height: u32,
    pub kinds: Vec<u8>,
}

impl MapScene {
    /// Builds a [`MapScene`] from a loaded map's tile grids (acceptance criterion 1).
    ///
    /// `data` is trusted to already satisfy [`MapData::validate`] (every present layer has
    /// exactly `width * height` cells) — both real callers (`ddai_map::load_map`,
    /// `ddai_trace::rawmap::read`) guarantee this themselves; this function still never panics on
    /// a layer that's merely *shorter* than expected (a defensively-missing cell reads as `Air`
    /// for that specific special-layer lookup) but does not attempt to repair a fundamentally
    /// malformed map — call [`MapData::validate`] first if `data` did not come from one of those
    /// two trusted readers.
    pub fn build(data: &MapData) -> MapScene {
        let cell_count = data.cell_count();
        let mut kinds = Vec::with_capacity(cell_count);
        for i in 0..cell_count {
            let game_index = data.game.get(i).map(|t| t.index).unwrap_or(map::TILE_AIR);
            let front_index = data.front.as_ref().and_then(|f| f.get(i)).map(|t| t.index);
            let tele_kind = data.tele.as_ref().and_then(|t| t.get(i)).map(|t| t.kind);
            let speedup_force = data.speedup.as_ref().and_then(|s| s.get(i)).map(|s| s.force);
            let switch_kind = data.switch.as_ref().and_then(|s| s.get(i)).map(|s| s.kind);
            let tune_number = data.tune.as_ref().and_then(|t| t.get(i)).map(|t| t.number);
            let kind = classify_cell(
                game_index,
                front_index,
                tele_kind,
                speedup_force,
                switch_kind,
                tune_number,
            );
            kinds.push(kind as u8);
        }
        MapScene {
            width: data.width,
            height: data.height,
            kinds,
        }
    }
}

/// The compressed wire payload `GET /api/map/<sha256>` serves (acceptance criterion 1: "returns
/// the scene compressed: width, height, kinds `u8[]`, zstd or deflate"). Deflate (via `flate2`'s
/// pure-Rust backend, the same choice `ddai-map` already made for the same reason: no C
/// dependency) rather than zstd — one fewer new dependency, and a kinds grid is simple enough
/// (long runs of the same byte on any real map) that deflate's ratio is already good; see
/// `docs/formats.md` for the measured ratio on `BlmapChill`.
///
/// Wire layout: `u32 width, u32 height, u32 deflated_len, deflated_len bytes of raw DEFLATE
/// (RFC 1951, no zlib/gzip wrapper) of `kinds``.
pub fn encode_compressed(scene: &MapScene) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::DeflateEncoder;
    use std::io::Write;

    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(&scene.kinds)
        .expect("writing to an in-memory Vec never fails");
    let deflated = encoder
        .finish()
        .expect("finishing an in-memory DeflateEncoder never fails");

    let mut out = Vec::with_capacity(12 + deflated.len());
    out.extend_from_slice(&scene.width.to_le_bytes());
    out.extend_from_slice(&scene.height.to_le_bytes());
    out.extend_from_slice(&(deflated.len() as u32).to_le_bytes());
    out.extend_from_slice(&deflated);
    out
}

/// The error returned by [`decode_compressed`] — used by this module's own round-trip tests and
/// available to callers that want to validate the wire format independently of this crate's own
/// encoder (e.g. a differential test against a hand-built payload).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error("payload shorter than the 12-byte header")]
    Truncated,
    #[error("declared width*height ({0}) overflows or is larger than the {1}-byte cap")]
    TooLarge(u64, usize),
    #[error("inflate failed: {0}")]
    Inflate(String),
    #[error("inflated length ({actual}) does not match width*height ({expected})")]
    LengthMismatch { expected: usize, actual: usize },
}

/// Hard cap on `width * height` this decoder will ever inflate into, independent of what the
/// payload's header claims — a corrupt or adversarial header must not be able to make this
/// allocate an unbounded amount of memory. Comfortably larger than any real DDNet map (the
/// largest maps in this corpus are on the order of a few hundred thousand cells).
const MAX_CELLS: usize = 64 * 1024 * 1024;

/// Inverse of [`encode_compressed`] — used by this crate's round-trip tests (acceptance criterion
/// 5: "frame encode/decode round trip + a golden byte fixture" also covers this map payload) and
/// by the client conceptually (the actual consumer is JS, see `assets/live.js`, but this Rust
/// decoder exists so the wire format has one authoritative, tested definition on the server side
/// too, not just "whatever `encode_compressed` happens to produce").
pub fn decode_compressed(bytes: &[u8]) -> Result<MapScene, DecodeError> {
    if bytes.len() < 12 {
        return Err(DecodeError::Truncated);
    }
    let width = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let deflated_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let cell_count = (width as u64) * (height as u64);
    if cell_count > MAX_CELLS as u64 {
        return Err(DecodeError::TooLarge(cell_count, MAX_CELLS));
    }
    let deflated = bytes.get(12..12 + deflated_len).ok_or(DecodeError::Truncated)?;

    use flate2::read::DeflateDecoder;
    use std::io::Read;
    let mut decoder = DeflateDecoder::new(deflated);
    // Bounded by `cell_count + 1`: reading one byte more than expected is how we notice the
    // stream claims *more* data than `width*height` without needing a separate length check on
    // the inflater itself (which `flate2` doesn't expose for `read_to_end` directly).
    let mut kinds = Vec::with_capacity(cell_count as usize);
    decoder
        .by_ref()
        .take(cell_count + 1)
        .read_to_end(&mut kinds)
        .map_err(|e| DecodeError::Inflate(e.to_string()))?;
    if kinds.len() != cell_count as usize {
        return Err(DecodeError::LengthMismatch {
            expected: cell_count as usize,
            actual: kinds.len(),
        });
    }
    Ok(MapScene { width, height, kinds })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_physics::map::{SpeedupTile, SwitchTile, TeleTile, Tile, TuneTile};

    fn tile(index: u8) -> Tile {
        Tile {
            index,
            flags: 0,
            skip: 0,
            reserved: 0,
        }
    }

    fn map_2x1(game: [u8; 2]) -> MapData {
        MapData {
            width: 2,
            height: 1,
            game: game.into_iter().map(tile).collect(),
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        }
    }

    #[test]
    fn classifies_every_kind_the_acceptance_criteria_names() {
        assert_eq!(classify_cell(map::TILE_AIR, None, None, None, None, None), Kind::Air);
        assert_eq!(
            classify_cell(map::TILE_SOLID, None, None, None, None, None),
            Kind::Solid
        );
        assert_eq!(
            classify_cell(map::TILE_NOHOOK, None, None, None, None, None),
            Kind::NoHook
        );
        assert_eq!(
            classify_cell(map::TILE_DEATH, None, None, None, None, None),
            Kind::Death
        );
        assert_eq!(
            classify_cell(map::TILE_DFREEZE, None, None, None, None, None),
            Kind::DeepFreeze
        );
        assert_eq!(
            classify_cell(map::TILE_FREEZE, None, None, None, None, None),
            Kind::Freeze
        );
        assert_eq!(
            classify_cell(map::TILE_DUNFREEZE, None, None, None, None, None),
            Kind::DeepUnfreeze
        );
        assert_eq!(
            classify_cell(map::TILE_UNFREEZE, None, None, None, None, None),
            Kind::Unfreeze
        );
        assert_eq!(
            classify_cell(map::TILE_AIR, None, Some(map::TILE_TELEIN), None, None, None),
            Kind::TeleIn
        );
        assert_eq!(
            classify_cell(map::TILE_AIR, None, Some(map::TILE_TELEOUT), None, None, None),
            Kind::TeleOut
        );
        assert_eq!(
            classify_cell(map::TILE_AIR, None, Some(map::TILE_TELECHECK), None, None, None),
            Kind::TeleCheckpoint
        );
        assert_eq!(
            classify_cell(map::TILE_AIR, None, None, Some(5), None, None),
            Kind::Speedup
        );
        assert_eq!(
            classify_cell(map::TILE_AIR, None, None, None, Some(1), None),
            Kind::Switch
        );
        assert_eq!(
            classify_cell(map::TILE_AIR, None, None, None, None, Some(1)),
            Kind::TuneZone
        );
        assert_eq!(
            classify_cell(map::TILE_STOP, None, None, None, None, None),
            Kind::Stopper
        );
        assert_eq!(classify_cell(ENTITY_SPAWN, None, None, None, None, None), Kind::Spawn);
        assert_eq!(
            classify_cell(ENTITY_SPAWN_RED, None, None, None, None, None),
            Kind::Spawn
        );
        assert_eq!(
            classify_cell(ENTITY_SPAWN_BLUE, None, None, None, None, None),
            Kind::Spawn
        );
    }

    #[test]
    fn front_layer_freeze_applies_even_over_an_air_game_tile() {
        // A very common real-map pattern: an open (walkable) game tile with a front-layer freeze
        // zone drawn over it (`character.cpp`'s `m_TileFIndex == TILE_FREEZE` check).
        assert_eq!(
            classify_cell(map::TILE_AIR, Some(map::TILE_FREEZE), None, None, None, None),
            Kind::Freeze
        );
    }

    #[test]
    fn solid_never_comes_from_the_front_layer() {
        // `CCollision::IsSolid` only ever reads the game layer for body solidity — the front
        // layer's `TILE_SOLID`/`TILE_NOHOOK` (if an editor even allows placing them there) must
        // not turn an otherwise-air game cell solid in this classifier either.
        assert_eq!(
            classify_cell(map::TILE_AIR, Some(map::TILE_SOLID), None, None, None, None),
            Kind::Air
        );
    }

    /// Regression test for review round 1, finding F7: `TILE_NOLASER` only blocks lasers
    /// (`CCollision::IsNoLaser`) — `CCollision::IsSolid` (`collision.cpp:605-608`) checks only
    /// `TILE_SOLID`/`TILE_NOHOOK`, so a player walks straight through a `TILE_NOLASER` cell. The
    /// previous version of this classifier drew it as `Solid` (an impassable-looking wall);
    /// confirmed on the real corpus this isn't a rare edge case: `BlmapChill` has 199
    /// game-layer `TILE_NOLASER` cells, `blmapV5_ddpp` has 28.
    #[test]
    fn nolaser_is_not_solid() {
        assert_eq!(
            classify_cell(map::TILE_NOLASER, None, None, None, None, None),
            Kind::Air
        );
    }

    #[test]
    fn solid_takes_precedence_over_death_and_freeze_on_the_same_cell() {
        // Not a realistic map (a cell can't simultaneously carry two different game-layer tile
        // indices), but demonstrates the precedence itself: solid is checked before front-layer
        // death, so a solid game tile is never re-classified by an (impossible in practice, but
        // not memory-unsafe) front freeze at the same index.
        assert_eq!(
            classify_cell(map::TILE_SOLID, Some(map::TILE_FREEZE), None, None, None, None),
            Kind::Solid
        );
    }

    #[test]
    fn build_classifies_a_small_real_map_row() {
        let data = map_2x1([map::TILE_SOLID, map::TILE_FREEZE]);
        let scene = MapScene::build(&data);
        assert_eq!(scene.width, 2);
        assert_eq!(scene.height, 1);
        assert_eq!(scene.kinds, vec![Kind::Solid as u8, Kind::Freeze as u8]);
    }

    #[test]
    fn build_reads_every_special_layer() {
        let mut data = map_2x1([map::TILE_AIR, map::TILE_AIR]);
        data.tele = Some(vec![
            TeleTile {
                number: 1,
                kind: map::TILE_TELEIN,
            },
            TeleTile::default(),
        ]);
        data.speedup = Some(vec![
            SpeedupTile::default(),
            SpeedupTile {
                force: 10,
                max_speed: 0,
                kind: map::TILE_SPEED_BOOST,
                angle: 0,
            },
        ]);
        let scene = MapScene::build(&data);
        assert_eq!(scene.kinds[0], Kind::TeleIn as u8);
        assert_eq!(scene.kinds[1], Kind::Speedup as u8);
    }

    #[test]
    fn build_handles_switch_and_tune_layers() {
        let mut data = map_2x1([map::TILE_AIR, map::TILE_AIR]);
        data.switch = Some(vec![
            SwitchTile {
                number: 1,
                kind: 1,
                flags: 0,
                delay: 0,
            },
            SwitchTile::default(),
        ]);
        data.tune = Some(vec![TuneTile::default(), TuneTile { number: 3, kind: 0 }]);
        let scene = MapScene::build(&data);
        assert_eq!(scene.kinds[0], Kind::Switch as u8);
        assert_eq!(scene.kinds[1], Kind::TuneZone as u8);
    }

    #[test]
    fn compressed_round_trip_preserves_every_field() {
        let data = map_2x1([map::TILE_SOLID, map::TILE_DEATH]);
        let scene = MapScene::build(&data);
        let bytes = encode_compressed(&scene);
        let decoded = decode_compressed(&bytes).expect("decode");
        assert_eq!(decoded, scene);
    }

    #[test]
    fn compressed_round_trip_on_a_larger_uniform_map_compresses_well() {
        // A uniform (all-air) 64x64 map — deflate should shrink this dramatically, demonstrating
        // the format actually compresses (not just round-trips).
        let data = MapData {
            width: 64,
            height: 64,
            game: vec![tile(map::TILE_AIR); 64 * 64],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: Vec::new(),
        };
        let scene = MapScene::build(&data);
        let bytes = encode_compressed(&scene);
        assert!(
            bytes.len() < scene.kinds.len() / 4,
            "expected meaningful compression on a uniform map: {} vs {} raw",
            bytes.len(),
            scene.kinds.len()
        );
        assert_eq!(decode_compressed(&bytes).expect("decode"), scene);
    }

    #[test]
    fn decode_rejects_a_header_claiming_an_absurd_cell_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&65535u32.to_le_bytes()); // width
        bytes.extend_from_slice(&65535u32.to_le_bytes()); // height — width*height way over MAX_CELLS
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(decode_compressed(&bytes), Err(DecodeError::TooLarge(_, _))));
    }

    #[test]
    fn decode_rejects_truncated_input() {
        assert_eq!(decode_compressed(&[1, 2, 3]), Err(DecodeError::Truncated));
    }

    #[test]
    fn decode_rejects_a_deflated_len_that_overruns_the_buffer() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1000u32.to_le_bytes()); // claims 1000 bytes follow; none do
        assert_eq!(decode_compressed(&bytes), Err(DecodeError::Truncated));
    }

    #[test]
    fn decode_rejects_inflated_length_mismatching_the_header() {
        // Valid deflate stream for 2 bytes, but header claims a 1x1 (1-cell) map.
        let scene = MapScene {
            width: 1,
            height: 2,
            kinds: vec![0, 0],
        };
        let mut bytes = encode_compressed(&scene);
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes()); // lie: height 2 -> 1
        assert!(matches!(
            decode_compressed(&bytes),
            Err(DecodeError::LengthMismatch { expected: 1, .. })
        ));
    }
}
