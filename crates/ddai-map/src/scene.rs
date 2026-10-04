//! The *visual* scene of a `.map`: every group, tile layer and quad layer in the order DDNet's client draws them
//! (`game/map/map_renderer.cpp`, `render_layer.cpp`), the images they use (embedded pixels or the name of an external
//! `mapres` file) and the envelopes that animate colours and quad positions.
//!
//! [`crate::load_map`] keeps only what physics needs (the six winning gameplay layers). A renderer needs everything in
//! draw order, so this module walks the same validated datafile ([`crate::datafile::Datafile`]) a second time and keeps
//! the rest. It follows the client, not the server: where `CMapRenderer::Load` and `CRenderLayer*` skip something
//! ("just ignore invalid layers from rendering"), so does this reader — a layer whose item or data is unusable is
//! dropped and the rest of the scene is still returned. Only a datafile that cannot be parsed at all is an error.
//!
//! What is read, with the DDNet 20.1 source it mirrors (`mapitems.h` field offsets, in bytes, `i32` little-endian):
//! - group `CMapItemGroup`: offset x/y, parallax x/y, layer range, and (item version >= 2) the clip rectangle;
//! - tiles layer `CMapItemLayerTilemap`: size, flags, colour and its envelope, image index, and the data index of the
//!   role it plays (visual, game, front, tele, speedup, switch, tune — chosen by flags exactly like
//!   `CMapRenderer::GetLayerType`);
//! - quads layer `CMapItemLayerQuads` and its `CQuad` records (5 points, 4 vertex colours, 4 texture coordinates, two
//!   envelope references);
//! - image `CMapItemImage`: embedded images are always RGBA (`mapimages.cpp`), external ones are `mapres/<name>.png`;
//! - envelope `CMapItemEnvelope` and its `CEnvPoint` records (curve types step, linear, slow, fast, smooth; a bezier
//!   point is read as linear because the bezier handles live in a UUID item this reader does not parse).
//!
//! Not read: sound layers (`LAYERTYPE_SOUNDS`), the tune layer's contents beyond its numbers, group names.
//!
//! Every allocation is bounded by [`MAX_LAYER_TILES`], [`MAX_IMAGE_BYTES`] and the total budget [`MAX_SCENE_BYTES`]
//! (a crafted tiny file must not make a server allocate gigabytes), and nothing here panics on hostile input.
//!
//! Portions derived from DDNet (zlib license, Copyright (C) 2007-2014 Magnus Auvinen / Teeworlds and the DDNet
//! contributors). This is an altered version, not the original software.

use crate::datafile::Datafile;
use crate::error::MapError;
use crate::mapitems::{self, MAPITEMTYPE_GROUP, MAPITEMTYPE_LAYER};

const MAPITEMTYPE_IMAGE: u16 = 2;
const MAPITEMTYPE_ENVELOPE: u16 = 3;
const MAPITEMTYPE_ENVPOINTS: u16 = 6;

const LAYERTYPE_TILES: i32 = 2;
const LAYERTYPE_QUADS: i32 = 3;
const LAYERFLAG_DETAIL: i32 = 1;

/// `CMapItemEnvelope::VERSION_TEEWORLDS_BEZIER`: with one such envelope the points are `CEnvPointBezier_upstream`.
const ENVELOPE_VERSION_BEZIER: i32 = 3;
const ENVPOINT_BYTES: usize = 24;
const ENVPOINT_BEZIER_BYTES: usize = 88;
const MAX_ENV_POINTS: usize = 100_000;
const MAX_ENVELOPES: usize = 4096;

/// `sizeof(CQuad)`: 5 points (8 bytes), 4 colours (16), 4 texture coordinates (8), 4 envelope `i32`s.
const QUAD_BYTES: usize = 152;
/// Quads per layer this reader accepts (a real map has at most a few thousand).
const MAX_QUADS_PER_LAYER: usize = 1_000_000;

/// Most cells in one tile layer (the same cap as the physics loader, `loader::MAX_TILE_COUNT`).
pub const MAX_LAYER_TILES: i64 = 32 * 1024 * 1024;
/// Largest embedded image (bytes of RGBA) this reader decodes.
pub const MAX_IMAGE_BYTES: usize = 128 * 1024 * 1024;
/// Total bytes of tile data, quad data and image pixels one scene may hold *at its peak* (the decoded blob a layer is
/// unpacked from counts while it is alive); a layer or image that would pass it is dropped and
/// [`VisualScene::over_budget`] is set. 96 MiB: the web keeps no decoded scene, but a build must fit a 512 MiB unit.
pub const MAX_SCENE_BYTES: usize = 96 * 1024 * 1024;
/// Largest side of an embedded image.
const MAX_IMAGE_SIDE: i32 = 16384;
/// DDNet's `MAX_MAPIMAGES`.
const MAX_IMAGES: usize = 64;

/// What a tiles layer is (`TILESLAYERFLAG_*`, decided in the same order as `CMapRenderer::GetLayerType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileRole {
    /// An ordinary design layer, drawn with its image.
    Visual,
    Game,
    Front,
    Tele,
    Speedup,
    Switch,
    Tune,
}

impl TileRole {
    /// The word on the wire and in the docs.
    pub fn as_str(self) -> &'static str {
        match self {
            TileRole::Visual => "visual",
            TileRole::Game => "game",
            TileRole::Front => "front",
            TileRole::Tele => "tele",
            TileRole::Speedup => "speedup",
            TileRole::Switch => "switch",
            TileRole::Tune => "tune",
        }
    }

    /// Bytes of per-cell auxiliary data a layer of this role carries ([`TileLayer::aux`]).
    pub fn aux_stride(self) -> usize {
        match self {
            TileRole::Visual | TileRole::Game | TileRole::Front => 0,
            TileRole::Tele | TileRole::Tune => 1,
            TileRole::Switch => 2,
            TileRole::Speedup => 4,
        }
    }
}

/// One tiles layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayer {
    pub role: TileRole,
    pub detail: bool,
    pub width: u32,
    pub height: u32,
    /// `m_Color` (RGBA, 0..=255); physics roles always have white.
    pub color: [u8; 4],
    pub color_env: i32,
    pub color_env_offset: i32,
    /// Index into [`VisualScene::images`], or `-1` (flat colour, or the entities tileset for the entity roles).
    pub image: i32,
    /// `width * height` cells of `(index, flags)`. For the special roles the index is what the client draws with the
    /// entities tileset: the tele tile type, the speedup tile type, the switch tile type (its flags are the switch
    /// flags), the tune tile type. Cells with no tile are `(0, 0)`.
    pub tiles: Vec<u8>,
    /// Per-cell auxiliary numbers, [`TileRole::aux_stride`] bytes per cell, empty for the other roles:
    /// tele `[number]`, tune `[number]`, switch `[number, delay]`, speedup `[force, max_speed, angle_lo, angle_hi]`
    /// (the angle is a little-endian `i16`, degrees).
    pub aux: Vec<u8>,
}

/// One quad (`CQuad`): positions and texture coordinates in 22.10 fixed point as stored, colours 0..=255.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quad {
    /// Corners in the order the file stores them (top-left, top-right, bottom-left, bottom-right) and the pivot
    /// (`m_aPoints[4]`) the rotation of the position envelope turns about, as `[x, y]` pairs.
    pub points: [[i32; 2]; 5],
    pub colors: [[u8; 4]; 4],
    pub texcoords: [[i32; 2]; 4],
    pub pos_env: i32,
    pub pos_env_offset: i32,
    pub color_env: i32,
    pub color_env_offset: i32,
}

/// A quads layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadLayer {
    pub detail: bool,
    pub image: i32,
    pub quads: Vec<Quad>,
}

/// A layer of a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layer {
    Tiles(TileLayer),
    Quads(QuadLayer),
}

/// A group: the offset and parallax the camera applies to all its layers, an optional clip rectangle, and the layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub offset_x: i32,
    pub offset_y: i32,
    pub parallax_x: i32,
    pub parallax_y: i32,
    /// `[x, y, w, h]` in world units, when the group clips.
    pub clip: Option<[i32; 4]>,
    pub layers: Vec<Layer>,
}

/// An image the layers refer to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The file stem of an external image (`mapres/<name>.png`), or the embedded image's own name.
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub external: bool,
    /// `width * height * 4` bytes of RGBA for an embedded image whose data is usable; `None` otherwise (external, or broken).
    pub rgba: Option<Vec<u8>>,
}

/// One envelope point: time in milliseconds, curve type (`CURVETYPE_*`, 0 step, 1 linear, 2 slow, 3 fast, 4 smooth,
/// 5 bezier) and up to four values in 22.10 fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvPoint {
    pub time_ms: i32,
    pub curve: i32,
    pub values: [i32; 4],
}

/// An envelope: 1..=4 channels and its points in time order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Envelope {
    pub channels: u8,
    pub points: Vec<EnvPoint>,
}

/// Everything a renderer needs of a map, in draw order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VisualScene {
    pub groups: Vec<Group>,
    pub images: Vec<Image>,
    pub envelopes: Vec<Envelope>,
    /// Layers or images dropped because they were unusable or past the budget (a hint for the page, not an error).
    pub skipped: u32,
    /// `true` when something was dropped *because the budget was spent* (not because it was broken): the scene is
    /// incomplete, and a caller that would rather show nothing than a part of the map refuses it.
    pub over_budget: bool,
}

impl VisualScene {
    /// The cells of the game layer, `(width, height, tiles)` — the last game layer in draw order, like `CLayers`.
    pub fn game_layer(&self) -> Option<&TileLayer> {
        self.groups
            .iter()
            .flat_map(|g| g.layers.iter())
            .filter_map(|l| match l {
                Layer::Tiles(t) if t.role == TileRole::Game => Some(t),
                _ => None,
            })
            .next_back()
    }
}

fn i32_field(payload: &[u8], offset: usize, default: i32) -> i32 {
    mapitems::i32_at(payload, offset).unwrap_or(default)
}

/// `CDataFileReader::GetDataString`-like: the blob up to its first NUL, lossily decoded, at most 255 bytes.
fn data_string(df: &Datafile, index: i32) -> String {
    if index < 0 {
        return String::new();
    }
    match df.data_bounded(index as usize, 256) {
        Ok(bytes) => {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len()).min(255);
            String::from_utf8_lossy(&bytes[..end]).into_owned()
        }
        Err(_) => String::new(),
    }
}

/// `CMap::ExtractTiles` (map.cpp:278-313) for the visual `(index, flags)` pairs: a tile-skip stream expanded to
/// exactly `count` cells, or a plain `CTile` array; `None` when the blob is too short.
fn unpack_tiles(raw: &[u8], count: usize, skip_format: bool) -> Option<Vec<u8>> {
    let mut out = vec![0u8; count * 2];
    if skip_format {
        let mut dst = 0usize;
        let mut src = 0usize;
        while dst < count && src + 4 <= raw.len() {
            let (index, flags, skip) = (raw[src], raw[src + 1], raw[src + 2] as usize);
            src += 4;
            for _ in 0..=skip {
                if dst >= count {
                    break;
                }
                out[dst * 2] = index;
                out[dst * 2 + 1] = flags;
                dst += 1;
            }
        }
        // DDNet requires exactly `count` cells; a short stream leaves the rest empty here instead of dropping the layer.
        Some(out)
    } else {
        if raw.len() < count * 4 {
            return None;
        }
        for (i, chunk) in raw.as_chunks::<4>().0.iter().take(count).enumerate() {
            out[i * 2] = chunk[0];
            out[i * 2 + 1] = chunk[1];
        }
        Some(out)
    }
}

/// The scene budget still to spend, and whether a layer or image was refused for lack of it.
struct Budget {
    left: usize,
    refused: bool,
}

impl Budget {
    /// Spends `retained` bytes for good; `false` (and nothing spent, `refused` set) when `retained + transient` (the
    /// buffers alive together while the layer is unpacked) would pass what is left.
    fn spend(&mut self, retained: usize, transient: usize) -> bool {
        match retained.checked_add(transient) {
            Some(peak) if peak <= self.left => {
                self.left -= retained;
                true
            }
            _ => {
                self.refused = true;
                false
            }
        }
    }
}

fn role_of(flags: u32) -> TileRole {
    if flags & mapitems::TILESLAYERFLAG_GAME != 0 {
        TileRole::Game
    } else if flags & mapitems::TILESLAYERFLAG_FRONT != 0 {
        TileRole::Front
    } else if flags & mapitems::TILESLAYERFLAG_SWITCH != 0 {
        TileRole::Switch
    } else if flags & mapitems::TILESLAYERFLAG_TELE != 0 {
        TileRole::Tele
    } else if flags & mapitems::TILESLAYERFLAG_SPEEDUP != 0 {
        TileRole::Speedup
    } else if flags & mapitems::TILESLAYERFLAG_TUNE != 0 {
        TileRole::Tune
    } else {
        TileRole::Visual
    }
}

fn read_tile_layer(df: &Datafile, payload: &[u8], budget: &mut Budget) -> Option<TileLayer> {
    // Layers `parse_tilemap` rejects (a bad name, colour or size) are dropped, like `CRenderLayer::IsValid`.
    let tm = mapitems::parse_tilemap(payload, 0, 0).ok()?;
    let count_i64 = i64::from(tm.width) * i64::from(tm.height);
    if count_i64 <= 0 || count_i64 > MAX_LAYER_TILES {
        return None;
    }
    let count = count_i64 as usize;
    let role = role_of(tm.flags);
    let detail = i32_field(payload, 8, 0) & LAYERFLAG_DETAIL != 0;
    let skip_format = tm.item_version >= 4;

    let data_index = match role {
        TileRole::Visual | TileRole::Game => tm.data,
        TileRole::Front => tm.front,
        TileRole::Tele => tm.tele,
        TileRole::Speedup => tm.speedup,
        TileRole::Switch => tm.switch,
        TileRole::Tune => tm.tune,
    };
    if data_index < 0 || data_index as usize >= df.num_data() {
        return None;
    }
    let data_index = data_index as usize;
    let stride_aux = role.aux_stride();
    let record = match role {
        TileRole::Visual | TileRole::Game | TileRole::Front => 4,
        TileRole::Tele | TileRole::Tune => 2,
        TileRole::Switch => 4,
        TileRole::Speedup => 6,
    };
    let want = count.checked_mul(record)?;
    if !budget.spend(count * (2 + stride_aux), want) {
        return None;
    }
    let raw = df.data_bounded(data_index, want).ok()?;

    let (tiles, aux) = match role {
        TileRole::Visual | TileRole::Game | TileRole::Front => {
            // DDNet only tile-skip-decodes plain and game layers; a front layer is always a plain array.
            let skip = skip_format && role != TileRole::Front;
            (unpack_tiles(&raw, count, skip)?, Vec::new())
        }
        _ => {
            if raw.len() < want {
                return None;
            }
            let mut tiles = vec![0u8; count * 2];
            let mut aux = vec![0u8; count * stride_aux];
            for (i, rec) in raw.chunks_exact(record).take(count).enumerate() {
                match role {
                    // `CTeleTile { m_Number, m_Type }`
                    TileRole::Tele => {
                        tiles[i * 2] = rec[1];
                        aux[i] = rec[0];
                    }
                    // `CTuneTile { m_Number, m_Type }`
                    TileRole::Tune => {
                        tiles[i * 2] = rec[1];
                        aux[i] = rec[0];
                    }
                    // `CSwitchTile { m_Number, m_Type, m_Flags, m_Delay }`
                    TileRole::Switch => {
                        tiles[i * 2] = rec[1];
                        tiles[i * 2 + 1] = rec[2];
                        aux[i * 2] = rec[0];
                        aux[i * 2 + 1] = rec[3];
                    }
                    // `CSpeedupTile { m_Force, m_MaxSpeed, m_Type, m_MustBe0, m_Angle }`
                    TileRole::Speedup => {
                        tiles[i * 2] = rec[2];
                        aux[i * 4] = rec[0];
                        aux[i * 4 + 1] = rec[1];
                        aux[i * 4 + 2] = rec[4];
                        aux[i * 4 + 3] = rec[5];
                    }
                    _ => {}
                }
            }
            (tiles, aux)
        }
    };

    let physics = role != TileRole::Visual;
    let color = if physics {
        [255; 4]
    } else {
        let c = |off| i32_field(payload, off, 255).clamp(0, 255) as u8;
        [c(28), c(32), c(36), c(40)]
    };
    Some(TileLayer {
        role,
        detail,
        width: tm.width as u32,
        height: tm.height as u32,
        color,
        color_env: if physics { -1 } else { i32_field(payload, 44, -1) },
        color_env_offset: if physics { 0 } else { i32_field(payload, 48, 0) },
        image: if physics { -1 } else { i32_field(payload, 52, -1) },
        tiles,
        aux,
    })
}

fn read_quad_layer(df: &Datafile, payload: &[u8], budget: &mut Budget) -> Option<QuadLayer> {
    let num = i32_field(payload, 16, 0);
    let data_index = i32_field(payload, 20, -1);
    let image = i32_field(payload, 24, -1);
    let detail = i32_field(payload, 8, 0) & LAYERFLAG_DETAIL != 0;
    if num <= 0 || data_index < 0 || data_index as usize >= df.num_data() {
        return None;
    }
    let num = num as usize;
    if num > MAX_QUADS_PER_LAYER {
        return None;
    }
    // `CRenderLayerQuads::OnInit`: the blob must hold at least `m_NumQuads` records.
    let need = num.checked_mul(QUAD_BYTES)?;
    if !budget.spend(num * std::mem::size_of::<Quad>(), need) {
        return None;
    }
    let raw = df.data_bounded(data_index as usize, need).ok()?;
    if raw.len() < need {
        return None;
    }
    let mut quads = Vec::with_capacity(num);
    for rec in raw.as_chunks::<QUAD_BYTES>().0.iter().take(num) {
        let at = |o: usize| i32_field(rec, o, 0);
        let mut points = [[0i32; 2]; 5];
        for (p, point) in points.iter_mut().enumerate() {
            *point = [at(p * 8), at(p * 8 + 4)];
        }
        let mut colors = [[0u8; 4]; 4];
        for (c, color) in colors.iter_mut().enumerate() {
            for (k, v) in color.iter_mut().enumerate() {
                *v = at(40 + c * 16 + k * 4).clamp(0, 255) as u8;
            }
        }
        let mut texcoords = [[0i32; 2]; 4];
        for (t, tc) in texcoords.iter_mut().enumerate() {
            *tc = [at(104 + t * 8), at(104 + t * 8 + 4)];
        }
        quads.push(Quad {
            points,
            colors,
            texcoords,
            pos_env: at(136),
            pos_env_offset: at(140),
            color_env: at(144),
            color_env_offset: at(148),
        });
    }
    Some(QuadLayer { detail, image, quads })
}

fn read_images(df: &Datafile, budget: &mut Budget, skipped: &mut u32) -> Vec<Image> {
    let (start, num) = df.type_range(MAPITEMTYPE_IMAGE);
    let mut images = Vec::new();
    for i in 0..num.min(MAX_IMAGES) {
        let Ok(item) = df.item(start + i) else {
            images.push(Image {
                name: String::new(),
                width: 0,
                height: 0,
                external: false,
                rgba: None,
            });
            continue;
        };
        let p = item.payload;
        let version = i32_field(p, 0, 0);
        let width = i32_field(p, 4, 0);
        let height = i32_field(p, 8, 0);
        let external = i32_field(p, 12, 0) != 0;
        let name = data_string(df, i32_field(p, 16, -1));
        let data_index = i32_field(p, 20, -1);
        // `CMapItemImage_v2::m_MustBe1`: a version-2 image whose marker is not 1 is refused by the client.
        let valid_type = version <= 1 || i32_field(p, 24, 0) == 1;
        let mut image = Image {
            name,
            width: width.clamp(0, MAX_IMAGE_SIDE) as u32,
            height: height.clamp(0, MAX_IMAGE_SIDE) as u32,
            external,
            rgba: None,
        };
        if !external {
            let ok_size = width > 0 && height > 0 && width <= MAX_IMAGE_SIDE && height <= MAX_IMAGE_SIDE;
            let bytes = (width.max(0) as usize)
                .saturating_mul(height.max(0) as usize)
                .saturating_mul(4);
            if valid_type && ok_size && bytes <= MAX_IMAGE_BYTES && data_index >= 0 && budget.spend(bytes, 0) {
                match df.data_bounded(data_index as usize, bytes) {
                    Ok(raw) if raw.len() >= bytes => image.rgba = Some(raw),
                    _ => *skipped += 1,
                }
            } else {
                *skipped += 1;
            }
        }
        images.push(image);
    }
    images
}

fn read_envelopes(df: &Datafile) -> Vec<Envelope> {
    let (start, num) = df.type_range(MAPITEMTYPE_ENVELOPE);
    if num == 0 {
        return Vec::new();
    }
    let items: Vec<_> = (0..num.min(MAX_ENVELOPES))
        .filter_map(|i| df.item(start + i).ok())
        .collect();
    // With one upstream-Teeworlds envelope (version >= 3) the points are the 88-byte bezier records.
    let size = if items
        .iter()
        .any(|e| i32_field(e.payload, 0, 0) >= ENVELOPE_VERSION_BEZIER)
    {
        ENVPOINT_BEZIER_BYTES
    } else {
        ENVPOINT_BYTES
    };
    let (pstart, pnum) = df.type_range(MAPITEMTYPE_ENVPOINTS);
    let points_item = if pnum > 0 { df.item(pstart).ok() } else { None };
    let max_points = points_item
        .as_ref()
        .map_or(0, |it| (it.payload.len() / size).min(MAX_ENV_POINTS));
    items
        .iter()
        .map(|e| {
            let p = e.payload;
            let channels = i32_field(p, 4, 0).clamp(0, 4) as u8;
            let first = (i32_field(p, 8, 0).max(0) as usize).min(max_points);
            let count = (i32_field(p, 12, 0).max(0) as usize).min(max_points - first);
            let mut points = Vec::with_capacity(count);
            if let Some(it) = &points_item {
                for k in 0..count {
                    let o = (first + k) * size;
                    let at = |off: usize| i32_field(it.payload, o + off, 0);
                    points.push(EnvPoint {
                        time_ms: at(0),
                        curve: at(4),
                        values: [at(8), at(12), at(16), at(20)],
                    });
                }
            }
            Envelope { channels, points }
        })
        .collect()
}

/// Reads the visual scene of the `.map` file `bytes`.
///
/// Never panics. Returns an error only when the datafile container itself is invalid or the map has no groups; unusable
/// layers and images are dropped and counted in [`VisualScene::skipped`].
pub fn extract_visual_scene(bytes: &[u8]) -> Result<VisualScene, MapError> {
    extract_visual_scene_within(bytes, MAX_SCENE_BYTES)
}

/// [`extract_visual_scene`] with an explicit budget (bytes of decoded tiles, quads and pixels, peak included).
pub fn extract_visual_scene_within(bytes: &[u8], budget: usize) -> Result<VisualScene, MapError> {
    let df = Datafile::parse(bytes)?;
    let mut budget = Budget {
        left: budget,
        refused: false,
    };
    let mut skipped = 0u32;

    let images = read_images(&df, &mut budget, &mut skipped);
    let envelopes = read_envelopes(&df);

    let (layers_start, layers_num) = df.type_range(MAPITEMTYPE_LAYER);
    let (groups_start, groups_num) = df.type_range(MAPITEMTYPE_GROUP);
    if groups_num == 0 {
        return Err(MapError::NoGameLayer);
    }
    let mut groups = Vec::with_capacity(groups_num);
    for g in 0..groups_num {
        let item = df.item(groups_start + g)?;
        let p = item.payload;
        if p.len() < mapitems::SIZEOF_GROUP_V1 {
            return Err(MapError::InvalidGroup { group: g });
        }
        let version = i32_field(p, 0, 0);
        let start = i32_field(p, 20, 0);
        let num = i32_field(p, 24, 0);
        if start < 0 || num < 0 || (start as i64) + (num as i64) > layers_num as i64 {
            return Err(MapError::InvalidGroup { group: g });
        }
        let clip = (version >= 2 && i32_field(p, 28, 0) != 0).then(|| {
            [
                i32_field(p, 32, 0),
                i32_field(p, 36, 0),
                i32_field(p, 40, 0),
                i32_field(p, 44, 0),
            ]
        });
        let mut layers = Vec::with_capacity(num as usize);
        for l in 0..num as usize {
            let Ok(layer_item) = df.item(layers_start + start as usize + l) else {
                skipped += 1;
                continue;
            };
            let lp = layer_item.payload;
            if lp.len() < mapitems::SIZEOF_LAYER {
                skipped += 1;
                continue;
            }
            match i32_field(lp, 4, 0) {
                LAYERTYPE_TILES => match read_tile_layer(&df, lp, &mut budget) {
                    Some(t) => layers.push(Layer::Tiles(t)),
                    None => skipped += 1,
                },
                LAYERTYPE_QUADS => match read_quad_layer(&df, lp, &mut budget) {
                    Some(q) => layers.push(Layer::Quads(q)),
                    None => skipped += 1,
                },
                _ => {} // sound layers and anything newer are not drawn
            }
        }
        groups.push(Group {
            offset_x: i32_field(p, 4, 0),
            offset_y: i32_field(p, 8, 0),
            parallax_x: i32_field(p, 12, 100),
            parallax_y: i32_field(p, 16, 100),
            clip,
            layers,
        });
    }
    Ok(VisualScene {
        groups,
        images,
        envelopes,
        skipped,
        over_budget: budget.refused,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        MapWriter, QuadSpec, TILESLAYERFLAG_FRONT, TILESLAYERFLAG_GAME, TILESLAYERFLAG_SPEEDUP, TILESLAYERFLAG_SWITCH,
        TILESLAYERFLAG_TELE, TILESLAYERFLAG_TUNE, TileLayerLook, TileLayerSpec, TilemapShape, encode_tile_skip,
        game_layer_data,
    };

    fn tiles_spec<'a>(w: i32, h: i32, flags: u32, version: i32, data: &'a [u8]) -> TileLayerSpec<'a> {
        TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: version,
            width: w,
            height: h,
            flags,
            data,
        }
    }

    fn square_quad(x: i32, y: i32, size: i32, color: [i32; 4]) -> QuadSpec {
        let (x, y, s) = (x << 10, y << 10, size << 10);
        QuadSpec {
            points: [[x, y], [x + s, y], [x, y + s], [x + s, y + s], [x + s / 2, y + s / 2]],
            colors: [color; 4],
            texcoords: [[0, 0], [1024, 0], [0, 1024], [1024, 1024]],
            pos_env: -1,
            pos_env_offset: 0,
            color_env: -1,
            color_env_offset: 0,
        }
    }

    /// A map exercising every kind of layer in a known draw order:
    /// group 0 (parallax 0, clipped): a quad layer; group 1 (offset, parallax 50): a design tiles layer with an
    /// external image, a detail quad layer with an embedded image; group 2: game, front, tele, speedup, switch, tune.
    fn sample_map() -> Vec<u8> {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_info_item(None, None, None, None, &[]);
        let sky = w.add_image("grass_main", 1024, 1024, None, 1);
        assert_eq!(sky, 0);
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
        let emb = w.add_image("my art", 4, 4, Some(&rgba), 2);
        assert_eq!(emb, 1);
        let env_color = w.add_envelope(4, &[(0, 1, [1024, 1024, 1024, 0]), (2000, 4, [1024, 512, 0, 1024])]);
        let env_pos = w.add_envelope(3, &[(0, 1, [0, 0, 0, 0]), (1000, 1, [10 << 10, -5 << 10, 90 << 10, 0])]);

        let mut q = square_quad(0, 0, 100, [10, 20, 30, 255]);
        q.color_env = env_color;
        q.color_env_offset = 250;
        let l0 = w.add_quad_layer(&[q], -1, false);
        w.add_group_ext(l0, 1, [0, 0], [0, 0], Some([0, -64, 1000, 500]));

        let cells = 6 * 5;
        let design: Vec<(u8, u8)> = (0..cells).map(|i| if i % 7 == 0 { (65, 1) } else { (0, 0) }).collect();
        let packed = encode_tile_skip(&design);
        let l1 = w.add_tile_layer_look(
            &tiles_spec(6, 5, 0, 4, &packed),
            TileLayerLook {
                image: sky,
                color: [10, 20, 30, 200],
                color_env: env_color,
                color_env_offset: 40,
                ..Default::default()
            },
        );
        let mut moving = square_quad(5, 5, 10, [255, 255, 255, 255]);
        moving.pos_env = env_pos;
        w.add_quad_layer(&[moving, square_quad(20, 20, 10, [1, 2, 3, 4])], emb, true);
        w.add_group_ext(l1, 2, [-12, 7], [50, 60], None);

        let (gw, gh) = (6, 5);
        let mut game = vec![(0u8, 0u8); cells];
        game[0] = (1, 0);
        game[8] = (9, 0);
        let game_data = encode_tile_skip(&game);
        let g = w.add_tile_layer(&tiles_spec(gw, gh, TILESLAYERFLAG_GAME, 4, &game_data));
        let mut front = vec![0u8; cells * 4];
        front[4 * 3] = 2; // cell 3: TILE_DEATH
        w.add_tile_layer(&tiles_spec(gw, gh, TILESLAYERFLAG_FRONT, 3, &front));
        let mut tele = vec![0u8; cells * 2];
        tele[2 * 4] = 7; // number
        tele[2 * 4 + 1] = 26; // TILE_TELEIN
        w.add_tile_layer(&tiles_spec(gw, gh, TILESLAYERFLAG_TELE, 3, &tele));
        let mut speedup = vec![0u8; cells * 6];
        speedup[6 * 5..6 * 5 + 6].copy_from_slice(&[50, 200, 29, 0, 0x2c, 0x01]); // force, max, type, 0, angle 300
        w.add_tile_layer(&tiles_spec(gw, gh, TILESLAYERFLAG_SPEEDUP, 3, &speedup));
        let mut switch = vec![0u8; cells * 4];
        switch[4 * 6..4 * 6 + 4].copy_from_slice(&[3, 24, 4, 9]); // number, type, flags, delay
        w.add_tile_layer(&tiles_spec(gw, gh, TILESLAYERFLAG_SWITCH, 3, &switch));
        let mut tune = vec![0u8; cells * 2];
        tune[2 * 9] = 2;
        tune[2 * 9 + 1] = 68;
        w.add_tile_layer(&tiles_spec(gw, gh, TILESLAYERFLAG_TUNE, 3, &tune));
        w.add_group(g, 6);
        w.finish()
    }

    #[test]
    fn groups_and_layers_come_out_in_draw_order() {
        let scene = extract_visual_scene(&sample_map()).expect("scene");
        assert_eq!(scene.groups.len(), 3);
        assert_eq!(scene.skipped, 0);

        let g0 = &scene.groups[0];
        assert_eq!((g0.offset_x, g0.offset_y, g0.parallax_x, g0.parallax_y), (0, 0, 0, 0));
        assert_eq!(g0.clip, Some([0, -64, 1000, 500]));
        assert!(matches!(&g0.layers[..], [Layer::Quads(_)]));

        let g1 = &scene.groups[1];
        assert_eq!(
            (g1.offset_x, g1.offset_y, g1.parallax_x, g1.parallax_y),
            (-12, 7, 50, 60)
        );
        assert_eq!(g1.clip, None);
        let [Layer::Tiles(design), Layer::Quads(art)] = &g1.layers[..] else {
            panic!(
                "group 1 layers: {:?}",
                g1.layers
                    .iter()
                    .map(|l| matches!(l, Layer::Tiles(_)))
                    .collect::<Vec<_>>()
            );
        };
        assert_eq!(design.role, TileRole::Visual);
        assert_eq!((design.width, design.height), (6, 5));
        assert_eq!(design.image, 0);
        assert_eq!(design.color, [10, 20, 30, 200]);
        assert_eq!((design.color_env, design.color_env_offset), (0, 40));
        assert!(!design.detail);
        assert!(art.detail);
        assert_eq!(art.image, 1);
        assert_eq!(art.quads.len(), 2);

        let roles: Vec<TileRole> = scene.groups[2]
            .layers
            .iter()
            .map(|l| match l {
                Layer::Tiles(t) => t.role,
                Layer::Quads(_) => panic!("unexpected quads"),
            })
            .collect();
        assert_eq!(
            roles,
            [
                TileRole::Game,
                TileRole::Front,
                TileRole::Tele,
                TileRole::Speedup,
                TileRole::Switch,
                TileRole::Tune
            ]
        );
        assert_eq!(scene.game_layer().map(|g| (g.width, g.height)), Some((6, 5)));
    }

    #[test]
    fn tile_skip_streams_expand_and_flags_survive() {
        let scene = extract_visual_scene(&sample_map()).unwrap();
        let Layer::Tiles(design) = &scene.groups[1].layers[0] else {
            panic!()
        };
        assert_eq!(design.tiles.len(), 6 * 5 * 2);
        for i in 0..30 {
            let expect = if i % 7 == 0 { (65, 1) } else { (0, 0) };
            assert_eq!((design.tiles[i * 2], design.tiles[i * 2 + 1]), expect, "cell {i}");
        }
        let game = scene.game_layer().unwrap();
        assert_eq!((game.tiles[0], game.tiles[8 * 2]), (1, 9));
        let Layer::Tiles(front) = &scene.groups[2].layers[1] else {
            panic!()
        };
        assert_eq!(front.tiles[3 * 2], 2);
    }

    #[test]
    fn special_layers_are_normalised_to_a_tile_index_and_aux_numbers() {
        let scene = extract_visual_scene(&sample_map()).unwrap();
        let get = |role: TileRole| {
            scene.groups[2]
                .layers
                .iter()
                .find_map(|l| match l {
                    Layer::Tiles(t) if t.role == role => Some(t),
                    _ => None,
                })
                .unwrap()
        };
        let tele = get(TileRole::Tele);
        assert_eq!((tele.tiles[4 * 2], tele.aux[4]), (26, 7));
        assert_eq!(tele.aux.len(), 30);
        let speedup = get(TileRole::Speedup);
        assert_eq!(speedup.tiles[5 * 2], 29);
        assert_eq!(&speedup.aux[5 * 4..5 * 4 + 4], &[50, 200, 0x2c, 0x01]);
        assert_eq!(
            i16::from_le_bytes([speedup.aux[5 * 4 + 2], speedup.aux[5 * 4 + 3]]),
            300
        );
        let switch = get(TileRole::Switch);
        assert_eq!((switch.tiles[6 * 2], switch.tiles[6 * 2 + 1]), (24, 4));
        assert_eq!(&switch.aux[6 * 2..6 * 2 + 2], &[3, 9]);
        let tune = get(TileRole::Tune);
        assert_eq!((tune.tiles[9 * 2], tune.aux[9]), (68, 2));
        // Physics layers carry no look of their own.
        assert_eq!((tele.image, tele.color, tele.color_env), (-1, [255; 4], -1));
    }

    #[test]
    fn images_say_whether_they_are_embedded_or_external_and_keep_their_pixels() {
        let scene = extract_visual_scene(&sample_map()).unwrap();
        assert_eq!(scene.images.len(), 2);
        let ext = &scene.images[0];
        assert_eq!(
            (ext.name.as_str(), ext.external, ext.rgba.is_none()),
            ("grass_main", true, true)
        );
        assert_eq!((ext.width, ext.height), (1024, 1024));
        let emb = &scene.images[1];
        assert_eq!(
            (emb.name.as_str(), emb.external, emb.width, emb.height),
            ("my art", false, 4, 4)
        );
        let expect: Vec<u8> = (0..64).map(|i| i as u8).collect();
        assert_eq!(emb.rgba.as_deref(), Some(&expect[..]));
    }

    #[test]
    fn quads_keep_corners_colours_texcoords_and_envelope_references() {
        let scene = extract_visual_scene(&sample_map()).unwrap();
        let Layer::Quads(sky) = &scene.groups[0].layers[0] else {
            panic!()
        };
        let q = sky.quads[0];
        assert_eq!(q.points[0], [0, 0]);
        assert_eq!(q.points[3], [100 << 10, 100 << 10]);
        assert_eq!(q.points[4], [50 << 10, 50 << 10]);
        assert_eq!(q.colors[2], [10, 20, 30, 255]);
        assert_eq!(q.texcoords[3], [1024, 1024]);
        assert_eq!((q.color_env, q.color_env_offset, q.pos_env), (0, 250, -1));
        let Layer::Quads(art) = &scene.groups[1].layers[1] else {
            panic!()
        };
        assert_eq!(art.quads[0].pos_env, 1);
        assert_eq!(art.quads[1].colors[0], [1, 2, 3, 4]);
    }

    #[test]
    fn envelopes_are_split_by_their_point_ranges() {
        let scene = extract_visual_scene(&sample_map()).unwrap();
        assert_eq!(scene.envelopes.len(), 2);
        let color = &scene.envelopes[0];
        assert_eq!(color.channels, 4);
        assert_eq!(
            color.points,
            [
                EnvPoint {
                    time_ms: 0,
                    curve: 1,
                    values: [1024, 1024, 1024, 0]
                },
                EnvPoint {
                    time_ms: 2000,
                    curve: 4,
                    values: [1024, 512, 0, 1024]
                },
            ]
        );
        let pos = &scene.envelopes[1];
        assert_eq!(pos.channels, 3);
        assert_eq!(pos.points[1].values, [10 << 10, -5 << 10, 90 << 10, 0]);
    }

    /// A tiles layer whose size field claims billions of cells: dropped (counted), the rest of the map still there, and
    /// nothing close to that allocation is made.
    #[test]
    fn a_layer_that_claims_an_absurd_size_is_dropped_not_allocated() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let game = encode_tile_skip(&[(0u8, 0u8); 4]);
        let l0 = w.add_tile_layer(&tiles_spec(2, 2, TILESLAYERFLAG_GAME, 4, &game));
        let tiny = [0u8, 0, 255, 0]; // one tile-skip record covering 256 cells
        let l1 = w.add_tile_layer(&tiles_spec(30000, 30000, 0, 4, &tiny));
        let l2 = w.add_tile_layer(&tiles_spec(4000, 4000, 0, 3, &tiny)); // plain array far too short for its size
        assert_eq!((l0, l1, l2), (0, 1, 2));
        w.add_single_group_with_all_layers();
        let scene = extract_visual_scene(&w.finish()).expect("scene");
        assert_eq!(scene.skipped, 2);
        assert_eq!(scene.groups[0].layers.len(), 1);
    }

    #[test]
    fn broken_images_and_quad_layers_are_dropped_without_failing_the_map() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_image("too short", 64, 64, Some(&[1, 2, 3]), 1);
        w.add_image("huge", 16000, 16000, Some(&[0; 16]), 1);
        w.add_image("negative", -5, 8, Some(&[0; 16]), 1);
        w.add_image("fine", 1, 1, Some(&[1, 2, 3, 4]), 1);
        // A quad layer that says it has 5 quads but whose blob holds one record, and a good one.
        let q = square_quad(0, 0, 8, [255; 4]);
        w.add_quad_layer(&[q], 3, false);
        w.add_quad_layer_claiming(&[q], 5, 3, false);
        let game = encode_tile_skip(&[(0u8, 0u8); 4]);
        w.add_tile_layer(&tiles_spec(2, 2, TILESLAYERFLAG_GAME, 4, &game));
        w.add_single_group_with_all_layers();
        let scene = extract_visual_scene(&w.finish()).expect("scene");
        assert!(scene.images[0].rgba.is_none());
        assert!(scene.images[1].rgba.is_none());
        assert!(scene.images[2].rgba.is_none());
        assert_eq!(scene.images[3].rgba.as_deref(), Some(&[1u8, 2, 3, 4][..]));
        // Three images and the over-claiming quad layer are dropped; the good quad layer and the game layer stay.
        assert_eq!(scene.skipped, 4);
        assert_eq!(scene.groups[0].layers.len(), 2);
    }

    #[test]
    fn a_version_two_image_without_the_must_be_one_marker_is_not_drawn() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        // Hand-built item: version 2, the marker field is 0.
        let mut p = Vec::new();
        let idx = w.add_data_auto(b"x\0") as i32;
        let pixels = w.add_data_auto(&[9, 9, 9, 9]) as i32;
        for v in [2, 1, 1, 0, idx, pixels, 0] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        w.add_item(2, 0, &p);
        w.add_group_ext(0, 0, [0, 0], [100, 100], None);
        let scene = extract_visual_scene(&w.finish()).unwrap();
        assert!(scene.images[0].rgba.is_none());
        assert_eq!(scene.skipped, 1);
    }

    #[test]
    fn a_hostile_external_image_name_is_returned_verbatim_for_the_server_to_refuse() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_image("../skins/greyfox", 64, 64, None, 1);
        w.add_group_ext(0, 0, [0, 0], [100, 100], None);
        let scene = extract_visual_scene(&w.finish()).unwrap();
        assert_eq!(scene.images[0].name, "../skins/greyfox");
    }

    #[test]
    fn a_group_that_names_layers_that_do_not_exist_is_an_error() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_group_ext(5, 3, [0, 0], [100, 100], None);
        assert_eq!(
            extract_visual_scene(&w.finish()),
            Err(MapError::InvalidGroup { group: 0 })
        );
    }

    #[test]
    fn no_groups_and_not_a_datafile_are_errors() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        assert!(extract_visual_scene(&w.finish()).is_err());
        assert!(extract_visual_scene(b"").is_err());
        assert!(extract_visual_scene(&[0xAB; 4096]).is_err());
    }

    /// Every truncation and a sweep of single-byte corruptions of a valid map: an error or a scene, never a panic.
    #[test]
    fn truncated_and_corrupted_maps_never_panic() {
        let bytes = sample_map();
        for len in (0..bytes.len()).step_by(7) {
            let _ = extract_visual_scene(&bytes[..len]);
        }
        let mut seed = 0x9E3779B97F4A7C15u64;
        for _ in 0..2000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut copy = bytes.clone();
            let pos = (seed >> 33) as usize % copy.len();
            copy[pos] ^= (seed >> 8) as u8 | 1;
            let _ = extract_visual_scene(&copy);
        }
    }

    #[test]
    fn the_scene_budget_drops_layers_that_would_pass_it() {
        let mut budget = Budget {
            left: 100,
            refused: false,
        };
        assert!(budget.spend(60, 0));
        assert!(!budget.spend(60, 0));
        assert!(budget.refused);
        assert_eq!(budget.left, 40);
        assert!(!budget.spend(30, 20), "the peak counts, not only what stays");
        assert_eq!(budget.left, 40);
        assert!(budget.spend(30, 10));
        assert_eq!(budget.left, 10);
        assert!(!budget.spend(usize::MAX, 1), "no overflow");
    }

    /// A layer that fits only without the buffer it is unpacked from is refused, and the scene says why.
    #[test]
    fn a_scene_over_its_budget_says_so() {
        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        let game = encode_tile_skip(&[(0u8, 0u8); 4]);
        w.add_tile_layer(&tiles_spec(2, 2, TILESLAYERFLAG_GAME, 4, &game));
        let design = encode_tile_skip(&[(1u8, 0u8); 100 * 100]);
        w.add_tile_layer(&tiles_spec(100, 100, 0, 4, &design));
        w.add_single_group_with_all_layers();
        let bytes = w.finish();
        let whole = extract_visual_scene(&bytes).expect("scene");
        assert!(!whole.over_budget);
        assert_eq!(whole.groups[0].layers.len(), 2);
        // 100*100 cells need 20000 bytes kept and 40000 more to unpack from: 60000 at the peak, so 55000 refuses the layer.
        let tight = extract_visual_scene_within(&bytes, 55_000).expect("scene");
        assert!(tight.over_budget);
        assert_eq!(tight.skipped, 1);
        assert_eq!(tight.groups[0].layers.len(), 1);
    }

    #[test]
    fn game_layer_helper_data_is_all_air() {
        // The shared fixture helper builds an all-air game layer: a sanity check that this module's tile reader agrees.
        let raw = game_layer_data(3, 3);
        let tiles = unpack_tiles(&raw, 9, false).unwrap();
        assert!(tiles.iter().all(|&b| b == 0));
        assert!(unpack_tiles(&raw[..8], 9, false).is_none());
    }
}
