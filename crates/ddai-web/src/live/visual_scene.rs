//! The wire form of a map's *visual* scene (task 5.10): what the browser needs to draw the real map — every group and
//! layer in draw order, the images they use and the envelopes that animate them — extracted from the `.map` by
//! `ddai_map::scene` and sent once per map at `GET /api/map/<sha256>/scene`; embedded images are fetched one by one at
//! `GET /api/map/<sha256>/image/<n>` so a map with a dozen 4 MiB textures does not delay the first picture.
//!
//! Format (`docs/formats.md` §35.1), version 1, everything little-endian, the whole body raw-DEFLATE compressed (RFC 1951,
//! no zlib or gzip wrapper — the page inflates it with `DecompressionStream("deflate-raw")`):
//!
//! ```text
//! "DWSC" | u8 version = 1 | u8 0 | u16 0 | u32 json_len | json (UTF-8, padded with spaces to a multiple of 4) | blob
//! ```
//!
//! The JSON describes the scene and points into the blob by byte offset (always a multiple of 4, so the page can view the
//! blob as typed arrays): a tiles layer has `w*h*2` bytes of `(index, flags)` at `o` and `w*h*stride` bytes of per-cell
//! auxiliary numbers at `a`; a quads layer has `n` records of 104 bytes at `o` (10 `i32` positions, 16 colour bytes, 8 `i32`
//! texture coordinates, 4 `i32` envelope fields; the positions and texture coordinates are 22.10 fixed point).
//!
//! The server never lets a name from the map reach the filesystem: an external image whose name is not a plain file stem
//! (`http::ddnet_assets::is_safe_name`) is sent with an empty name and the page draws it as missing.

use std::sync::Arc;

use ddai_map::scene::{Image, Layer, TileLayer, VisualScene};
use flate2::Compression;
use flate2::write::DeflateEncoder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::Write;

pub const MAGIC: &[u8; 4] = b"DWSC";
pub const VERSION: u8 = 1;
/// Bytes of one quad record in the blob.
pub const QUAD_RECORD_BYTES: usize = 104;
/// Largest `.map` this module reads from disk (the same cap as `map_resolve`).
const MAX_MAP_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// One run of bytes in the blob, in the order they are laid out. The blob is never built in memory: [`write_scene`] first
/// lays the parts out (offsets only), writes the JSON that names them, then streams each part.
enum Part<'a> {
    Bytes(&'a [u8]),
    Quads(&'a [ddai_map::scene::Quad]),
}

impl Part<'_> {
    fn len(&self) -> usize {
        match self {
            Part::Bytes(b) => b.len(),
            Part::Quads(q) => q.len() * QUAD_RECORD_BYTES,
        }
    }
}

/// The offsets of the blob's parts (each 4-byte aligned) and the parts themselves.
#[derive(Default)]
struct Layout<'a> {
    parts: Vec<(usize, Part<'a>)>,
    end: usize,
}

impl<'a> Layout<'a> {
    /// Reserves `part` at the next 4-byte boundary and returns its offset.
    fn add(&mut self, part: Part<'a>) -> usize {
        let at = self.end.next_multiple_of(4);
        self.end = at + part.len();
        self.parts.push((at, part));
        at
    }
}

fn tile_layer_json<'a>(layout: &mut Layout<'a>, t: &'a TileLayer) -> Value {
    let o = layout.add(Part::Bytes(&t.tiles));
    let a = if t.aux.is_empty() {
        -1
    } else {
        layout.add(Part::Bytes(&t.aux)) as i64
    };
    json!({
        "k": "t",
        "r": t.role.as_str(),
        "w": t.width,
        "h": t.height,
        "d": u8::from(t.detail),
        "c": t.color,
        "ce": t.color_env,
        "co": t.color_env_offset,
        "i": t.image,
        "o": o,
        "a": a,
    })
}

fn quad_record(q: &ddai_map::scene::Quad, out: &mut Vec<u8>) {
    for p in q.points {
        out.extend_from_slice(&p[0].to_le_bytes());
        out.extend_from_slice(&p[1].to_le_bytes());
    }
    for c in q.colors {
        out.extend_from_slice(&c);
    }
    for t in q.texcoords {
        out.extend_from_slice(&t[0].to_le_bytes());
        out.extend_from_slice(&t[1].to_le_bytes());
    }
    for v in [q.pos_env, q.pos_env_offset, q.color_env, q.color_env_offset] {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Writes the scene as the uncompressed wire body (see the module docs) into `out`, part by part: the only buffers held
/// besides the scene are the JSON header and one small chunk of quad records.
pub fn write_scene<W: Write>(scene: &VisualScene, out: &mut W) -> std::io::Result<()> {
    let mut layout = Layout::default();
    let groups: Vec<Value> = scene
        .groups
        .iter()
        .map(|g| {
            let layers: Vec<Value> = g
                .layers
                .iter()
                .map(|l| match l {
                    Layer::Tiles(t) => tile_layer_json(&mut layout, t),
                    Layer::Quads(q) => {
                        let o = layout.add(Part::Quads(&q.quads));
                        json!({"k": "q", "d": u8::from(q.detail), "i": q.image, "n": q.quads.len(), "o": o})
                    }
                })
                .collect();
            json!({
                "ox": g.offset_x, "oy": g.offset_y, "px": g.parallax_x, "py": g.parallax_y,
                "clip": g.clip, "layers": layers,
            })
        })
        .collect();
    let images: Vec<Value> = scene
        .images
        .iter()
        .map(|i| {
            // An external image is looked up by name under the DDNet data directory: only a plain stem is ever named.
            let name = if i.external && !crate::http::ddnet_assets::is_safe_name(&i.name) {
                String::new()
            } else {
                i.name.clone()
            };
            json!({"n": name, "w": i.width, "h": i.height, "x": u8::from(i.external), "d": u8::from(i.rgba.is_some())})
        })
        .collect();
    let envelopes: Vec<Value> = scene
        .envelopes
        .iter()
        .map(|e| {
            let points: Vec<i64> = e
                .points
                .iter()
                .flat_map(|p| {
                    [
                        i64::from(p.time_ms),
                        i64::from(p.curve),
                        i64::from(p.values[0]),
                        i64::from(p.values[1]),
                        i64::from(p.values[2]),
                        i64::from(p.values[3]),
                    ]
                })
                .collect();
            json!({"c": e.channels, "p": points})
        })
        .collect();
    let game = scene
        .game_layer()
        .map(|g| json!({"w": g.width, "h": g.height}))
        .unwrap_or(Value::Null);
    let header = json!({
        "v": VERSION,
        "skipped": scene.skipped,
        "game": game,
        "images": images,
        "env": envelopes,
        "groups": groups,
    });
    let mut json_bytes = serde_json::to_vec(&header).expect("a JSON value serialises");
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    out.write_all(MAGIC)?;
    out.write_all(&[VERSION, 0, 0, 0])?;
    out.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
    out.write_all(&json_bytes)?;
    drop(json_bytes);

    let mut written = 0usize;
    let mut chunk: Vec<u8> = Vec::with_capacity(QUAD_RECORD_BYTES * QUAD_CHUNK);
    for (at, part) in &layout.parts {
        out.write_all(&[0u8; 3][..*at - written])?;
        match part {
            Part::Bytes(bytes) => out.write_all(bytes)?,
            Part::Quads(quads) => {
                for run in quads.chunks(QUAD_CHUNK) {
                    chunk.clear();
                    for quad in run {
                        quad_record(quad, &mut chunk);
                    }
                    out.write_all(&chunk)?;
                }
            }
        }
        written = at + part.len();
    }
    out.write_all(&[0u8; 3][..layout.end.next_multiple_of(4) - written])?;
    Ok(())
}

/// Quads turned into records at a time.
const QUAD_CHUNK: usize = 256;

/// The scene as the uncompressed wire body (tests; the server never holds this).
#[cfg(test)]
pub fn encode(scene: &VisualScene) -> Vec<u8> {
    let mut out = Vec::new();
    write_scene(scene, &mut out).expect("writing to a Vec never fails");
    out
}

/// The scene as the raw-DEFLATE compressed body of `GET /api/map/<sha256>/scene`, streamed through the encoder.
pub fn encode_compressed(scene: &VisualScene) -> Vec<u8> {
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
    write_scene(scene, &mut enc).expect("writing to an in-memory Vec never fails");
    enc.finish().expect("finishing an in-memory DeflateEncoder never fails")
}

/// An embedded image as `u32 width | u32 height | RGBA`, raw-DEFLATE compressed (`GET /api/map/<sha256>/image/<n>`);
/// `None` when the image is external or its pixels were unusable.
pub fn encode_image_compressed(image: &Image) -> Option<Vec<u8>> {
    let rgba = image.rgba.as_ref()?;
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(4));
    enc.write_all(&image.width.to_le_bytes())
        .and_then(|()| enc.write_all(&image.height.to_le_bytes()))
        .and_then(|()| enc.write_all(rgba))
        .expect("writing to an in-memory Vec never fails");
    Some(enc.finish().expect("finishing an in-memory DeflateEncoder never fails"))
}

/// Most compressed bytes (the scene and its images together) one map may cost the cache; a map past it is refused.
pub const MAX_ENTRY_BYTES: usize = 24 * 1024 * 1024;

/// Why a map has no visual scene for the page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualError {
    /// The map's layers and pictures do not fit the memory a build may use, or the result is too big to keep: the page
    /// draws the map from the coarse geometry instead. Never retried for the same map file.
    TooLarge,
    /// Anything else (the file is gone or changed, it is not a map); may pass.
    Failed(String),
}

impl std::fmt::Display for VisualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisualError::TooLarge => f.write_str("the map's layers and images are too large to draw"),
            VisualError::Failed(why) => f.write_str(why),
        }
    }
}

/// A map's wire bodies, kept (a few at a time, by their size: see `MapCache`). The decoded scene is dropped once these are
/// built: it is tens of MiB for a texture-heavy map, the bodies a few hundred KiB.
#[derive(Debug)]
pub struct VisualEntry {
    /// The compressed scene body.
    pub scene_deflated: Arc<Vec<u8>>,
    /// The compressed body of each embedded image (`None` for an external or broken one).
    images: Vec<Option<Arc<Vec<u8>>>>,
}

impl VisualEntry {
    /// Reads `path` (at most 256 MiB), checks it hashes to `sha256`, extracts the scene and encodes it. Runs on the
    /// blocking pool: it reads a whole map file and compresses.
    pub fn build(path: &std::path::Path, sha256: [u8; 32]) -> Result<VisualEntry, VisualError> {
        let failed = |why: String| VisualError::Failed(why);
        let meta = std::fs::metadata(path).map_err(|e| failed(format!("cannot read the map: {e}")))?;
        if meta.len() > MAX_MAP_FILE_BYTES {
            return Err(failed("the map file is too large".to_string()));
        }
        let bytes = std::fs::read(path).map_err(|e| failed(format!("cannot read the map: {e}")))?;
        let actual: [u8; 32] = Sha256::digest(&bytes).into();
        if actual != sha256 {
            return Err(failed("the map file changed on disk".to_string()));
        }
        let scene =
            ddai_map::extract_visual_scene(&bytes).map_err(|e| failed(format!("cannot read the map's layers: {e}")))?;
        drop(bytes);
        Self::from_scene(scene)
    }

    /// Encodes `scene` and drops it: the scene body is streamed into the encoder, then each embedded image is compressed
    /// and its pixels freed one by one. A scene the budget cut short, or one that compresses to more than
    /// [`MAX_ENTRY_BYTES`], is [`VisualError::TooLarge`].
    pub fn from_scene(mut scene: VisualScene) -> Result<VisualEntry, VisualError> {
        if scene.over_budget {
            return Err(VisualError::TooLarge);
        }
        let scene_deflated = encode_compressed(&scene);
        scene.groups = Vec::new();
        let mut total = scene_deflated.len();
        let mut images = Vec::with_capacity(scene.images.len());
        for image in &mut scene.images {
            let body = encode_image_compressed(image).map(Arc::new);
            image.rgba = None;
            total += body.as_ref().map_or(0, |b| b.len());
            if total > MAX_ENTRY_BYTES {
                return Err(VisualError::TooLarge);
            }
            images.push(body);
        }
        Ok(VisualEntry {
            scene_deflated: Arc::new(scene_deflated),
            images,
        })
    }

    /// Bytes this entry keeps (what the cache counts).
    pub fn size_bytes(&self) -> usize {
        self.scene_deflated.len() + self.images.iter().flatten().map(|b| b.len()).sum::<usize>()
    }

    /// The compressed body of embedded image `index`; `None` for an unknown index, an external image or a broken one.
    pub fn image_deflated(&self, index: usize) -> Option<Arc<Vec<u8>>> {
        self.images.get(index)?.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddai_map::testutil::{
        MapWriter, QuadSpec, TILESLAYERFLAG_GAME, TILESLAYERFLAG_TELE, TileLayerLook, TileLayerSpec, TilemapShape,
        encode_tile_skip,
    };
    use std::io::Read;

    /// A parsed wire body: the JSON header and the blob.
    struct Parsed {
        json: Value,
        blob: Vec<u8>,
    }

    fn parse(compressed: &[u8]) -> Parsed {
        let mut raw = Vec::new();
        flate2::read::DeflateDecoder::new(compressed)
            .read_to_end(&mut raw)
            .expect("raw deflate");
        parse_raw(&raw)
    }

    fn parse_raw(raw: &[u8]) -> Parsed {
        assert_eq!(&raw[..4], MAGIC);
        assert_eq!((raw[4], raw[5], raw[6], raw[7]), (VERSION, 0, 0, 0));
        let json_len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        assert_eq!(json_len % 4, 0, "the blob starts 4-byte aligned");
        Parsed {
            json: serde_json::from_slice(&raw[12..12 + json_len]).expect("json"),
            blob: raw[12 + json_len..].to_vec(),
        }
    }

    fn sample_scene() -> VisualScene {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let ext = w.add_image("grass_main", 1024, 1024, None, 1);
        let evil = w.add_image("../skins/greyfox", 64, 64, None, 1);
        let rgba: Vec<u8> = (0..2 * 2 * 4).map(|i| 200 - i as u8).collect();
        let emb = w.add_image("art", 2, 2, Some(&rgba), 2);
        let env = w.add_envelope(3, &[(0, 1, [0, 0, 0, 0]), (1000, 4, [10 << 10, 20 << 10, 90 << 10, 0])]);
        let q = QuadSpec {
            points: [[0, 0], [1024, 0], [0, 2048], [1024, 2048], [512, 1024]],
            colors: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            texcoords: [[0, 0], [1024, 0], [0, 1024], [1024, 1024]],
            pos_env: env,
            pos_env_offset: -17,
            color_env: -1,
            color_env_offset: 0,
        };
        let l0 = w.add_quad_layer(&[q, q], emb, true);
        let design: Vec<(u8, u8)> = (0..6).map(|i| (i as u8, (i % 4) as u8)).collect();
        let packed = encode_tile_skip(&design);
        w.add_tile_layer_look(
            &TileLayerSpec {
                shape: TilemapShape::Full,
                item_version: 4,
                width: 3,
                height: 2,
                flags: 0,
                data: &packed,
            },
            TileLayerLook {
                image: ext,
                color: [9, 8, 7, 6],
                ..Default::default()
            },
        );
        let game = encode_tile_skip(&[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (9, 0)]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 3,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game,
        });
        let mut tele = vec![0u8; 12];
        tele[2 * 4] = 5;
        tele[2 * 4 + 1] = 26;
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 2,
            flags: TILESLAYERFLAG_TELE,
            data: &tele,
        });
        let _ = (evil, l0);
        w.add_group_ext(0, 4, [3, -4], [50, 60], Some([1, 2, 3, 4]));
        ddai_map::extract_visual_scene(&w.finish()).expect("scene")
    }

    #[test]
    fn the_scene_round_trips_through_the_wire_format() {
        let scene = sample_scene();
        let p = parse(&encode_compressed(&scene));
        assert_eq!(p.json["v"], 1);
        assert_eq!(p.json["skipped"], 0);
        assert_eq!(p.json["game"], json!({"w": 3, "h": 2}));
        let g = &p.json["groups"][0];
        assert_eq!(
            (&g["ox"], &g["oy"], &g["px"], &g["py"]),
            (&json!(3), &json!(-4), &json!(50), &json!(60))
        );
        assert_eq!(g["clip"], json!([1, 2, 3, 4]));
        let layers = g["layers"].as_array().unwrap();
        assert_eq!(layers.len(), 4);

        // Quads: 104-byte records, offsets aligned, fields decode back.
        let q = &layers[0];
        assert_eq!(
            (&q["k"], &q["d"], &q["i"], &q["n"]),
            (&json!("q"), &json!(1), &json!(2), &json!(2))
        );
        let o = q["o"].as_u64().unwrap() as usize;
        assert_eq!(o % 4, 0);
        let rec = &p.blob[o..o + QUAD_RECORD_BYTES];
        let i32_at = |b: &[u8], off: usize| i32::from_le_bytes(b[off..off + 4].try_into().unwrap());
        assert_eq!((i32_at(rec, 16), i32_at(rec, 20)), (0, 2048)); // third corner
        assert_eq!((i32_at(rec, 32), i32_at(rec, 36)), (512, 1024)); // the pivot
        assert_eq!(&rec[40..56], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!((i32_at(rec, 56 + 4 * 7), i32_at(rec, 56 + 4 * 6)), (1024, 1024)); // last corner's texcoord
        assert_eq!(
            (i32_at(rec, 88), i32_at(rec, 92), i32_at(rec, 96), i32_at(rec, 100)),
            (0, -17, -1, 0)
        ); // pos env, offset, colour env
        assert_eq!(p.blob.len() % 4, 0);

        // The design tiles layer: tile-skip is gone, (index, flags) pairs are plain.
        let t = &layers[1];
        assert_eq!(
            (&t["k"], &t["r"], &t["w"], &t["h"], &t["i"]),
            (&json!("t"), &json!("visual"), &json!(3), &json!(2), &json!(0))
        );
        assert_eq!(t["c"], json!([9, 8, 7, 6]));
        let o = t["o"].as_u64().unwrap() as usize;
        assert_eq!(&p.blob[o..o + 12], &[0, 0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 1]);
        assert_eq!(t["a"], -1);

        // The game layer and the tele layer with its aux numbers.
        let game = &layers[2];
        assert_eq!(game["r"], "game");
        let o = game["o"].as_u64().unwrap() as usize;
        assert_eq!((p.blob[o], p.blob[o + 10]), (1, 9));
        let tele = &layers[3];
        let (o, a) = (
            tele["o"].as_u64().unwrap() as usize,
            tele["a"].as_u64().unwrap() as usize,
        );
        assert_eq!(p.blob[o + 4 * 2], 26);
        assert_eq!(p.blob[a + 4], 5);

        // Envelopes keep their raw fixed-point values.
        assert_eq!(
            p.json["env"][0],
            json!({"c": 3, "p": [0, 1, 0, 0, 0, 0, 1000, 4, 10240, 20480, 92160, 0]})
        );
    }

    #[test]
    fn external_names_that_are_not_plain_stems_are_never_sent() {
        let scene = sample_scene();
        let p = parse(&encode_compressed(&scene));
        let images = p.json["images"].as_array().unwrap();
        assert_eq!(
            images[0],
            json!({"n": "grass_main", "w": 1024, "h": 1024, "x": 1, "d": 0})
        );
        assert_eq!(images[1]["n"], "", "a path-like external name is dropped");
        assert_eq!(images[1]["x"], 1);
        assert_eq!(images[2], json!({"n": "art", "w": 2, "h": 2, "x": 0, "d": 1}));
        let text = serde_json::to_string(&p.json).unwrap();
        assert!(!text.contains("skins"), "{text}");
    }

    #[test]
    fn an_embedded_image_is_width_height_then_rgba_and_cached() {
        let entry = VisualEntry::from_scene(sample_scene()).expect("entry");
        assert!(entry.image_deflated(0).is_none(), "external");
        assert!(entry.image_deflated(1).is_none(), "external");
        assert!(entry.image_deflated(9).is_none(), "no such image");
        let body = entry.image_deflated(2).expect("embedded");
        let mut raw = Vec::new();
        flate2::read::DeflateDecoder::new(&body[..])
            .read_to_end(&mut raw)
            .unwrap();
        assert_eq!(u32::from_le_bytes(raw[0..4].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), 2);
        let expect: Vec<u8> = (0..16).map(|i| 200 - i as u8).collect();
        assert_eq!(&raw[8..], &expect[..]);
        assert!(Arc::ptr_eq(&body, &entry.image_deflated(2).unwrap()), "compressed once");
    }

    #[test]
    fn building_from_a_file_checks_the_hash() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_group_ext(0, 0, [0, 0], [100, 100], None);
        let bytes = w.finish();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.map");
        std::fs::write(&path, &bytes).unwrap();
        let sha: [u8; 32] = Sha256::digest(&bytes).into();
        assert!(VisualEntry::build(&path, sha).is_ok());
        assert!(
            VisualEntry::build(&path, [1; 32])
                .unwrap_err()
                .to_string()
                .contains("changed")
        );
        assert!(VisualEntry::build(&dir.path().join("nope.map"), sha).is_err());
        std::fs::write(&path, b"not a map").unwrap();
        let junk: [u8; 32] = Sha256::digest(b"not a map").into();
        assert!(
            VisualEntry::build(&path, junk)
                .unwrap_err()
                .to_string()
                .contains("layers")
        );
    }

    #[test]
    fn a_scene_the_budget_cut_short_is_refused_not_sent_in_part() {
        let mut scene = sample_scene();
        scene.over_budget = true;
        assert_eq!(VisualEntry::from_scene(scene).unwrap_err(), VisualError::TooLarge);
    }

    #[test]
    fn an_entry_keeps_only_compressed_bytes() {
        let entry = VisualEntry::from_scene(sample_scene()).expect("entry");
        let image = entry.image_deflated(2).unwrap();
        assert_eq!(entry.size_bytes(), entry.scene_deflated.len() + image.len());
        assert!(entry.size_bytes() < 2048, "{}", entry.size_bytes());
    }

    /// The streamed body is the same bytes a blob built in memory would give: every part starts 4-aligned and the body
    /// ends aligned too, whatever the part lengths (3 tele cells = 3 bytes of aux, odd).
    #[test]
    fn the_streamed_body_is_aligned_after_odd_sized_parts() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let game = encode_tile_skip(&[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 3,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game,
        });
        let mut tele = vec![0u8; 18]; // 3 x 3 cells: 9 aux bytes, an odd length
        tele[0] = 7;
        tele[1] = 1;
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 3,
            height: 3,
            flags: TILESLAYERFLAG_TELE,
            data: &tele,
        });
        w.add_group_ext(0, 2, [0, 0], [100, 100], None);
        let scene = ddai_map::extract_visual_scene(&w.finish()).expect("scene");
        let raw = encode(&scene);
        let p = parse_raw(&raw);
        assert_eq!(p.blob.len() % 4, 0);
        let layers = p.json["groups"][0]["layers"].as_array().unwrap();
        for l in layers {
            for key in ["o", "a"] {
                if let Some(at) = l[key].as_u64() {
                    assert_eq!(at % 4, 0, "{l}");
                }
            }
        }
        let tele = &layers[1];
        let (o, a) = (
            tele["o"].as_u64().unwrap() as usize,
            tele["a"].as_u64().unwrap() as usize,
        );
        assert_eq!((p.blob[o], p.blob[a]), (1, 7));
        assert_eq!(encode_compressed(&scene), {
            let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
            enc.write_all(&raw).unwrap();
            enc.finish().unwrap()
        });
    }

    /// A tiny map of six 8192 x 4096 design layers (each 64 MiB of tiles, a few hundred bytes on disk): refused as too
    /// large without ever holding more than the budget.
    #[test]
    fn a_hostile_map_of_huge_layers_is_refused_as_too_large() {
        let bytes = hostile_map();
        assert!(bytes.len() < 1024 * 1024, "{}", bytes.len());
        let scene = ddai_map::extract_visual_scene(&bytes).expect("scene");
        assert!(scene.over_budget);
        assert_eq!(VisualEntry::from_scene(scene).unwrap_err(), VisualError::TooLarge);
        if let Ok(path) = std::env::var("DDAI_WRITE_HOSTILE_MAP") {
            std::fs::write(path, &bytes).unwrap();
        }
    }

    fn hostile_map() -> Vec<u8> {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let game = encode_tile_skip(&[(1u8, 0u8); 100 * 100]);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 4,
            width: 100,
            height: 100,
            flags: TILESLAYERFLAG_GAME,
            data: &game,
        });
        // One tile-skip record repeats a tile up to 256 times: 8192 * 4096 / 256 records, all alike, deflate to nothing.
        let design = encode_tile_skip(&vec![(2u8, 0u8); 8192 * 4096]);
        for _ in 0..6 {
            w.add_tile_layer(&TileLayerSpec {
                shape: TilemapShape::Full,
                item_version: 4,
                width: 8192,
                height: 4096,
                flags: 0,
                data: &design,
            });
        }
        w.add_single_group_with_all_layers();
        w.finish()
    }
}
