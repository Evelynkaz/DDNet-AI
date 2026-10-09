//! Synthetic `.map` and `.demo` bytes for end-to-end tests. Everything is generated here from
//! the crates' own encoders: no third-party map, demo or nickname ever enters the repository.

use ddai_demo::testutil::{build_prelude_bytes, write_tick_marker};
use ddai_map::testutil::{MapWriter, TILESLAYERFLAG_GAME, TileLayerSpec, TilemapShape};
use ddai_net::generated::objects;
use ddai_net::huffman::Huffman;
use ddai_net::intstr::str_to_ints;
use ddai_net::packer::{MAX_BYTES_PACKED, pack_ints};
use ddai_physics::map::{TILE_FREEZE, TILE_SOLID};

/// A valid v4 datafile: `w` x `h` tiles, solid floor on the last row with a freeze pit in columns
/// `pit.0..pit.1`, solid side walls.
pub fn map_bytes(w: i32, h: i32, pit: (i32, i32)) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 4) as usize];
    let mut set = |x: i32, y: i32, idx: u8| data[((y * w + x) * 4) as usize] = idx;
    for x in 0..w {
        set(
            x,
            h - 1,
            if x >= pit.0 && x < pit.1 {
                TILE_FREEZE
            } else {
                TILE_SOLID
            },
        );
    }
    for y in 0..h {
        set(0, y, TILE_SOLID);
        set(w - 1, y, TILE_SOLID);
    }
    let mut mw = MapWriter::new(4);
    mw.add_version_item(1);
    mw.add_tile_layer(&TileLayerSpec {
        shape: TilemapShape::Full,
        item_version: 3,
        width: w,
        height: h,
        flags: TILESLAYERFLAG_GAME,
        data: &data,
    });
    mw.add_single_group_with_all_layers();
    mw.finish()
}

/// One character in one synthetic snapshot.
#[derive(Debug, Clone)]
pub struct SynthChar {
    pub id: i32,
    pub name: String,
    pub clan: String,
    pub wire: objects::Character,
}

fn item(ty: i32, id: i32, data: &[i32]) -> Vec<i32> {
    let mut v = vec![(ty << 16) | id];
    v.extend_from_slice(data);
    v
}

fn character_ints(c: &objects::Character) -> Vec<i32> {
    vec![
        c.tick,
        c.x,
        c.y,
        c.vel_x,
        c.vel_y,
        c.angle,
        c.direction,
        c.jumped,
        c.hooked_player,
        c.hook_state,
        c.hook_tick,
        c.hook_x,
        c.hook_y,
        c.hook_dx,
        c.hook_dy,
        c.player_flags,
        c.health,
        c.armor,
        c.ammo_count,
        c.weapon,
        c.emote,
        c.attack_tick,
    ]
}

/// Raw `CSnapshot` ints (`[data_size, num_items, offsets.., items..]`).
fn snapshot_ints(chars: &[SynthChar]) -> Vec<i32> {
    let mut items: Vec<Vec<i32>> = Vec::new();
    for c in chars {
        items.push(item(objects::PlayerInfo::ID, c.id, &[0, c.id, 0, 0, 0]));
        let mut ci: Vec<i32> = str_to_ints(&c.name, 4);
        ci.extend(str_to_ints(&c.clan, 3));
        ci.push(-1); // country
        ci.extend(str_to_ints("skin-x", 6));
        ci.extend([0, 0, 0]);
        items.push(item(objects::ClientInfo::ID, c.id, &ci));
        items.push(item(objects::Character::ID, c.id, &character_ints(&c.wire)));
    }
    let mut offsets = Vec::new();
    let mut bytes = 0i32;
    for it in &items {
        offsets.push(bytes);
        bytes += (it.len() * 4) as i32;
    }
    let mut out = vec![bytes, items.len() as i32];
    out.extend(offsets);
    for it in items {
        out.extend(it);
    }
    out
}

/// Chunk writer that also handles payloads of 256 bytes and more (`size == 31` form).
fn write_big_chunk(out: &mut Vec<u8>, huffman: &Huffman, ty: u8, ints: &[i32]) {
    let mut raw = vec![0u8; ints.len() * MAX_BYTES_PACKED];
    let n = pack_ints(&mut raw, ints).expect("scratch buffer is large enough");
    let compressed = huffman.compress_vec(&raw[..n]);
    let size = compressed.len();
    if size < 30 {
        out.push(((ty & 0x3) << 5) | size as u8);
    } else if size < 256 {
        out.push(((ty & 0x3) << 5) | 30);
        out.push(size as u8);
    } else {
        out.push(((ty & 0x3) << 5) | 31);
        out.extend_from_slice(&(size as u16).to_le_bytes());
    }
    out.extend_from_slice(&compressed);
}

/// A version-5 client demo with one full snapshot per entry of `snapshots` (`(tick, characters)`).
/// With `embed` the map is embedded; without it the header still carries the map's size and crc
/// but no map bytes (like the 30 demos of the real archive that rely on the server's map).
pub fn demo_bytes(map: &[u8], embed: bool, snapshots: &[(i32, Vec<SynthChar>)]) -> Vec<u8> {
    demo_bytes_with_pre_inputs(map, embed, snapshots, &[])
}

/// The ints of an `Sv_PreInput` message chunk (the packed message bytes as little-endian words, the
/// way `CDemoRecorder::Write` stores any message).
fn pre_input_ints(m: &ddai_net::generated::messages::SvPreInput) -> Vec<i32> {
    use ddai_net::packer::Packer;
    use ddai_net::uuid::{MsgId, calculate_uuid, pack_msg_id};
    let mut buf = [0u8; 256];
    let mut packer = Packer::new(&mut buf);
    let uuid = calculate_uuid("preinput@netmsg.ddnet.org");
    pack_msg_id(&mut packer, MsgId::Ex { uuid, resolved: None }, false);
    ddai_net::generated::messages::encode_sv_pre_input(m, &mut packer);
    let mut bytes = packer.data().to_vec();
    bytes.resize(bytes.len().div_ceil(4) * 4, 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

/// [`demo_bytes`] plus `Sv_PreInput` messages: each `(tick, message)` is recorded in the chunk
/// stream of the snapshot with that tick (messages of other ticks are dropped).
pub fn demo_bytes_with_pre_inputs(
    map: &[u8],
    embed: bool,
    snapshots: &[(i32, Vec<SynthChar>)],
    pre_inputs: &[(i32, ddai_net::generated::messages::SvPreInput)],
) -> Vec<u8> {
    let loaded = ddai_map::load_map(map).expect("synthetic map is valid");
    let embedded = if embed { map.len() } else { 0 };
    let mut out = build_prelude_bytes(5, embedded as u32);
    if embed {
        let map_at = out.len() - map.len();
        out[map_at..].copy_from_slice(map);
    }
    // Header layout: marker 7 + version 1 + netversion 64 + map name 64, then size and crc (be).
    out[136..140].copy_from_slice(&(embedded as u32).to_be_bytes());
    out[140..144].copy_from_slice(&loaded.crc32.to_be_bytes());
    let huffman = Huffman::new();
    for (tick, chars) in snapshots {
        write_tick_marker(&mut out, *tick, true);
        for (_, m) in pre_inputs.iter().filter(|(t, _)| t == tick) {
            write_big_chunk(&mut out, &huffman, 2, &pre_input_ints(m));
        }
        write_big_chunk(&mut out, &huffman, 1, &snapshot_ints(chars));
    }
    out
}
