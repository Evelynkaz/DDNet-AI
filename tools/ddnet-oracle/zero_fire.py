#!/usr/bin/env python3
# Zeroes every tick's `fire` field in a scenario v2 file (docs/formats.md section 2), leaving
# every other byte -- including the map reference/sha256, tuning overrides, character list and
# every other input field -- untouched. Used by parity_check_b.sh (task 1.5, acceptance
# criterion 7's "arena recipe, fire never pressed" consistency check): Oracle B goes through the
# real `HandleWeapons()`/`FireWeapon()`, so an un-neutered `fire` press can apply real weapon
# damage/knockback that Oracle A's core-only tick never does, which would show up as a genuine
# (not merely `active_weapon`-field) core-state difference unrelated to what that check is
# supposed to isolate.
#
# This is DDNet-AI's own original file (GPL-3.0-only, like the rest of this repository) -- it
# does not read, write, or otherwise touch any DDNet source.
import struct, sys

# scenario v2 layout (docs/formats.md section 2):
# magic[4] version:u32 map_ref_tag:u8 map_ref_string:{u16,bytes} map_sha256[32] no_weak_hook:u8
# tuning_override_count:u32 { name:{u16,bytes} value_x100:i32 } * count
# character_count:u32 { id:u32 spawn_x:i32 spawn_y:i32 } * count
# tick_count:u32 inputs[t][c]: 11 x i32 (direction,target_x,target_y,aim_slot,jump,fire,hook,player_flags,wanted_weapon,next_weapon,prev_weapon)

def zero_fire(path_in, path_out):
    data = bytearray(open(path_in, 'rb').read())
    pos = 0
    assert bytes(data[0:4]) == b'SCN1'
    pos = 4
    version = struct.unpack_from('<I', data, pos)[0]; pos += 4
    assert version == 2
    tag = data[pos]; pos += 1
    slen = struct.unpack_from('<H', data, pos)[0]; pos += 2
    pos += slen
    pos += 32  # map_sha256
    pos += 1  # no_weak_hook
    ovc = struct.unpack_from('<I', data, pos)[0]; pos += 4
    for _ in range(ovc):
        nlen = struct.unpack_from('<H', data, pos)[0]; pos += 2
        pos += nlen
        pos += 4  # value_x100
    cc = struct.unpack_from('<I', data, pos)[0]; pos += 4
    pos += cc * (4 + 4 + 4)
    tick_count = struct.unpack_from('<I', data, pos)[0]; pos += 4
    for t in range(tick_count):
        for c in range(cc):
            # fields: direction,target_x,target_y,aim_slot,jump,fire,hook,player_flags,wanted_weapon,next_weapon,prev_weapon
            fire_off = pos + 4 * 5
            struct.pack_into('<i', data, fire_off, 0)
            pos += 44
    assert pos == len(data), (pos, len(data))
    open(path_out, 'wb').write(data)

zero_fire(sys.argv[1], sys.argv[2])
