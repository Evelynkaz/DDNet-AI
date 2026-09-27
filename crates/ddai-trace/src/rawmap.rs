//! rawmap v1: a little-endian binary encoding of [`MapData`], precise enough for the C++ oracle
//! to read without any of DDNet's own datafile/zlib machinery. See `docs/formats.md` for the
//! exact byte layout this module implements.

use crate::io::{FormatError, Reader, Writer};
use ddai_physics::map::{MapData, SpeedupTile, SwitchTile, TeleTile, Tile, TuneTile};

const MAGIC: &[u8; 4] = b"RMP1";
const VERSION: u32 = 1;

const PRESENT_FRONT: u8 = 1 << 0;
const PRESENT_TELE: u8 = 1 << 1;
const PRESENT_SPEEDUP: u8 = 1 << 2;
const PRESENT_SWITCH: u8 = 1 << 3;
const PRESENT_TUNE: u8 = 1 << 4;

/// Serializes `map` to rawmap v1 bytes.
///
/// # Panics
///
/// Panics if `map.validate()` would fail (a present layer's length doesn't match
/// `width * height`) — this is a programmer error in whatever built the `MapData` (a recipe or
/// a hand-built test map), not a runtime condition callers should recover from.
pub fn write(map: &MapData) -> Vec<u8> {
    map.validate()
        .expect("MapData must be internally consistent before it can be serialized");

    let mut present = 0u8;
    if map.front.is_some() {
        present |= PRESENT_FRONT;
    }
    if map.tele.is_some() {
        present |= PRESENT_TELE;
    }
    if map.speedup.is_some() {
        present |= PRESENT_SPEEDUP;
    }
    if map.switch.is_some() {
        present |= PRESENT_SWITCH;
    }
    if map.tune.is_some() {
        present |= PRESENT_TUNE;
    }

    let mut w = Writer::new();
    w.bytes(MAGIC);
    w.u32(VERSION);
    w.u32(map.width);
    w.u32(map.height);
    w.u8(present);

    write_tiles(&mut w, &map.game);
    if let Some(front) = &map.front {
        write_tiles(&mut w, front);
    }
    if let Some(tele) = &map.tele {
        for t in tele {
            w.u8(t.number).u8(t.kind);
        }
    }
    if let Some(speedup) = &map.speedup {
        for s in speedup {
            w.u8(s.force)
                .u8(s.max_speed)
                .u8(s.kind)
                .u8(0 /* reserved, mirrors CSpeedupTile::m_MustBe0 */);
            w.i16(s.angle);
        }
    }
    if let Some(switch) = &map.switch {
        for s in switch {
            w.u8(s.number).u8(s.kind).u8(s.flags).u8(s.delay);
        }
    }
    if let Some(tune) = &map.tune {
        for t in tune {
            w.u8(t.number).u8(t.kind);
        }
    }

    w.u32(map.settings.len() as u32);
    for s in &map.settings {
        w.string32(s);
    }

    w.into_bytes()
}

fn write_tiles(w: &mut Writer, tiles: &[Tile]) {
    for t in tiles {
        w.u8(t.index).u8(t.flags).u8(t.skip).u8(t.reserved);
    }
}

/// Parses rawmap v1 bytes back into a [`MapData`]. The result always passes `validate()`.
pub fn read(bytes: &[u8]) -> Result<MapData, FormatError> {
    let mut r = Reader::new(bytes);
    r.expect_magic(MAGIC)?;
    let version = r.u32("version")?;
    if version != VERSION {
        return Err(FormatError::UnsupportedVersion {
            format: "rawmap",
            version,
        });
    }
    let width = r.u32("width")?;
    let height = r.u32("height")?;
    let present = r.u8("present-layer bitmask")?;
    let n = width as usize * height as usize;

    let game = read_tiles(&mut r, n)?;
    let front = if present & PRESENT_FRONT != 0 {
        Some(read_tiles(&mut r, n)?)
    } else {
        None
    };
    let tele = if present & PRESENT_TELE != 0 {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(TeleTile {
                number: r.u8("tele number")?,
                kind: r.u8("tele type")?,
            });
        }
        Some(v)
    } else {
        None
    };
    let speedup = if present & PRESENT_SPEEDUP != 0 {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let force = r.u8("speedup force")?;
            let max_speed = r.u8("speedup max speed")?;
            let kind = r.u8("speedup type")?;
            let _reserved = r.u8("speedup reserved")?;
            let angle = r.i16("speedup angle")?;
            v.push(SpeedupTile {
                force,
                max_speed,
                kind,
                angle,
            });
        }
        Some(v)
    } else {
        None
    };
    let switch = if present & PRESENT_SWITCH != 0 {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(SwitchTile {
                number: r.u8("switch number")?,
                kind: r.u8("switch type")?,
                flags: r.u8("switch flags")?,
                delay: r.u8("switch delay")?,
            });
        }
        Some(v)
    } else {
        None
    };
    let tune = if present & PRESENT_TUNE != 0 {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(TuneTile {
                number: r.u8("tune number")?,
                kind: r.u8("tune type")?,
            });
        }
        Some(v)
    } else {
        None
    };

    let settings_count = r.u32("settings count")?;
    let mut settings = Vec::with_capacity(settings_count as usize);
    for _ in 0..settings_count {
        settings.push(r.string32("setting")?);
    }
    r.expect_eof()?;

    let map = MapData {
        width,
        height,
        game,
        front,
        tele,
        speedup,
        switch,
        tune,
        settings,
    };
    map.validate().map_err(|_| FormatError::InvalidValue {
        context: "map layer length",
        value: -1,
    })?;
    Ok(map)
}

fn read_tiles(r: &mut Reader, n: usize) -> Result<Vec<Tile>, FormatError> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(Tile {
            index: r.u8("tile index")?,
            flags: r.u8("tile flags")?,
            skip: r.u8("tile skip")?,
            reserved: r.u8("tile reserved")?,
        });
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_map() -> MapData {
        MapData {
            width: 2,
            height: 2,
            game: vec![
                Tile {
                    index: 1,
                    flags: 0,
                    skip: 0,
                    reserved: 0,
                },
                Tile {
                    index: 0,
                    flags: 0,
                    skip: 0,
                    reserved: 0,
                },
                Tile {
                    index: 0,
                    flags: 0,
                    skip: 0,
                    reserved: 0,
                },
                Tile {
                    index: 1,
                    flags: 8,
                    skip: 0,
                    reserved: 0,
                },
            ],
            front: None,
            tele: None,
            speedup: None,
            switch: None,
            tune: None,
            settings: vec!["sv_foo 1".to_string()],
        }
    }

    #[test]
    fn round_trips_game_only_map() {
        let map = tiny_map();
        let bytes = write(&map);
        assert_eq!(read(&bytes).unwrap(), map);
    }

    #[test]
    fn round_trips_all_layers_present() {
        let mut map = tiny_map();
        map.front = Some(vec![Tile::default(); 4]);
        map.tele = Some(vec![TeleTile { number: 1, kind: 26 }; 4]);
        map.speedup = Some(vec![
            SpeedupTile {
                force: 5,
                max_speed: 0,
                kind: 28,
                angle: -90
            };
            4
        ]);
        map.switch = Some(vec![
            SwitchTile {
                number: 1,
                kind: 24,
                flags: 0,
                delay: 0
            };
            4
        ]);
        map.tune = Some(vec![TuneTile { number: 1, kind: 68 }; 4]);
        let bytes = write(&map);
        assert_eq!(read(&bytes).unwrap(), map);
    }

    #[test]
    fn negative_speedup_angle_round_trips() {
        let mut map = tiny_map();
        map.speedup = Some(vec![
            SpeedupTile {
                force: 1,
                max_speed: 0,
                kind: 28,
                angle: -1
            };
            4
        ]);
        let bytes = write(&map);
        assert_eq!(read(&bytes).unwrap().speedup.unwrap()[0].angle, -1);
    }

    #[test]
    fn write_is_deterministic() {
        let map = tiny_map();
        assert_eq!(write(&map), write(&map));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = write(&tiny_map());
        bytes[0] = b'Q';
        assert!(read(&bytes).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = write(&tiny_map());
        bytes[4] = 99; // version is the first LE u32 after the 4-byte magic
        assert!(matches!(
            read(&bytes),
            Err(FormatError::UnsupportedVersion {
                format: "rawmap",
                version: 99
            })
        ));
    }

    #[test]
    fn rejects_truncated_tile_data() {
        let bytes = write(&tiny_map());
        let truncated = &bytes[..bytes.len() - 2];
        assert!(read(truncated).is_err());
    }

    #[test]
    #[should_panic]
    fn write_panics_on_mismatched_layer_length() {
        let mut map = tiny_map();
        map.front = Some(vec![Tile::default(); 1]); // wrong length for a 2x2 map
        write(&map);
    }

    #[test]
    fn empty_settings_round_trip() {
        let mut map = tiny_map();
        map.settings.clear();
        let bytes = write(&map);
        assert_eq!(read(&bytes).unwrap().settings, Vec::<String>::new());
    }
}
