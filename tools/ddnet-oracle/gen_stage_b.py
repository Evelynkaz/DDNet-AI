#!/usr/bin/env python3
"""Stage-B scenario crafter for Oracle B (task 1.6 stage B, docs/formats.md section 12.8).

Builds small hand-made maps (rawmap v1, docs/formats.md section 1) and random-but-seeded input
scripts (scenario v3, section 12.6) that exercise the mechanics the real-map corpus covers thinly:
laser/shotgun bounces (incl. tele-in-weapon tiles, telegun laser, old-laser, hit-disabled), ninja
dashes, rotating/opening/closing lights (also switch-gated), draggers (all six variants, solo,
teams, switch-gated), turrets with every freeze/explosive variant, and a mixed map.

Every scenario is replayed through the real Oracle B server (`--rawmap`/`--scenario`), twice, and
the two trace files must be byte-identical (the determinism check); a `SHA256SUMS` file with the
digests of every `.trb`, `.scn` and `.rawmap` in OUT_DIR (the corpus's traces *and* the inputs the
parity test replays; also those the real-map script wrote) is rewritten next to them.

Usage:
    gen_stage_b.py OUT_DIR [--oracle PATH] [--only PREFIX] [--no-determinism-check]

Output (per scenario NAME): NAME.rawmap, NAME.scn, NAME.trb, NAME.cov.json
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ENTITY_OFFSET = 255 - 16 * 4  # 191, mapitems.h

# --- mapitems.h constants (DDNet 20.1) ---------------------------------------------------------
E_SPAWN = 1
E_ARMOR_1 = 6
E_HEALTH_1 = 7
E_WEAPON_SHOTGUN = 8
E_WEAPON_GRENADE = 9
E_POWERUP_NINJA = 10
E_WEAPON_LASER = 11
E_LASER_FAST_CCW = 12
E_LASER_NORMAL_CCW = 13
E_LASER_SLOW_CCW = 14
E_LASER_STOP = 15
E_LASER_SLOW_CW = 16
E_LASER_NORMAL_CW = 17
E_LASER_FAST_CW = 18
E_LASER_SHORT = 19
E_LASER_MEDIUM = 20
E_LASER_LONG = 21
E_LASER_C_SLOW = 22
E_LASER_C_NORMAL = 23
E_LASER_C_FAST = 24
E_LASER_O_SLOW = 25
E_LASER_O_NORMAL = 26
E_LASER_O_FAST = 27
E_PLASMAE = 29
E_PLASMAF = 30
E_PLASMA = 31
E_PLASMAU = 32
E_DRAGGER_WEAK = 42
E_DRAGGER_NORMAL = 43
E_DRAGGER_STRONG = 44
E_DRAGGER_WEAK_NW = 45
E_DRAGGER_NORMAL_NW = 46
E_DRAGGER_STRONG_NW = 47

T_SOLID = 1
T_DEATH = 2
T_NOHOOK = 3
T_NOLASER = 4
T_FREEZE = 9
T_UNFREEZE = 11
T_TELEINWEAPON = 14  # tele layer kind
T_TELEIN = 26  # tele layer kind
T_TELEOUT = 27  # tele layer kind
T_SOLO_ENABLE = 21
T_SOLO_DISABLE = 22
T_SWITCHOPEN = 24  # switch layer kind
T_SWITCHCLOSE = 25  # switch layer kind
T_HIT_ENABLE = 19
T_HIT_DISABLE = 20
T_ALLOW_TELE_GUN = 98  # front layer
T_ALLOW_BLUE_TELE_GUN = 99
T_TELE_LASER_ENABLE = 128  # game/front: gives the toucher a telegun laser
T_TELE_LASER_DISABLE = 129
T_LFREEZE = 144

TILE = 32


class SplitMix64:
    """Same generator as crates/ddai-trace/src/prng.rs (docs/formats.md section 4)."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, n):
        if n == 0:
            return 0
        return (self.next_u64() * n) >> 64

    def range_incl(self, lo, hi):
        return lo + self.below(hi - lo + 1)

    def chance(self, num, den):
        return self.below(den) < num


class Map:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.game = [[0, 0] for _ in range(w * h)]  # [index, flags]
        self.front = None
        self.tele = None  # [number, kind]
        self.switch = None  # [number, kind, flags, delay]
        self.settings = []

    def _i(self, x, y):
        assert 0 <= x < self.w and 0 <= y < self.h, (x, y)
        return y * self.w + x

    def set(self, x, y, index, flags=0):
        self.game[self._i(x, y)] = [index, flags]

    def entity(self, x, y, entity, flags=0):
        self.set(x, y, ENTITY_OFFSET + entity, flags)

    def set_front(self, x, y, index, flags=0):
        if self.front is None:
            self.front = [[0, 0] for _ in range(self.w * self.h)]
        self.front[self._i(x, y)] = [index, flags]

    def set_tele(self, x, y, number, kind):
        if self.tele is None:
            self.tele = [[0, 0] for _ in range(self.w * self.h)]
        self.tele[self._i(x, y)] = [number, kind]

    def set_switch(self, x, y, number, kind, flags=0, delay=0):
        if self.switch is None:
            self.switch = [[0, 0, 0, 0] for _ in range(self.w * self.h)]
        self.switch[self._i(x, y)] = [number, kind, flags, delay]

    def switch_entity(self, x, y, number, entity, flags=0):
        self.set_switch(x, y, number, ENTITY_OFFSET + entity, flags, 0)

    def border(self):
        for x in range(self.w):
            self.set(x, 0, T_SOLID)
            self.set(x, self.h - 1, T_SOLID)
        for y in range(self.h):
            self.set(0, y, T_SOLID)
            self.set(self.w - 1, y, T_SOLID)

    def block(self, x0, y0, x1, y1, index=T_SOLID):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set(x, y, index)

    def free(self, x, y):
        i = self._i(x, y)
        return self.game[i][0] == 0 and (self.front is None or self.front[i][0] == 0)

    def to_bytes(self):
        present = 0
        if self.front is not None:
            present |= 0x01
        if self.tele is not None:
            present |= 0x02
        if self.switch is not None:
            present |= 0x08
        out = bytearray(b"RMP1")
        out += struct.pack("<IIIB", 1, self.w, self.h, present)
        for idx, flags in self.game:
            out += struct.pack("<BBBB", idx, flags, 0, 0)
        if self.front is not None:
            for idx, flags in self.front:
                out += struct.pack("<BBBB", idx, flags, 0, 0)
        if self.tele is not None:
            for number, kind in self.tele:
                out += struct.pack("<BB", number, kind)
        if self.switch is not None:
            for number, kind, flags, delay in self.switch:
                out += struct.pack("<BBBB", number, kind, flags, delay)
        out += struct.pack("<I", len(self.settings))
        for s in self.settings:
            b = s.encode()
            out += struct.pack("<I", len(b)) + b
        return bytes(out)


class InputDriver:
    """Random-but-seeded input script for one character (resolved scenario-v3 inputs)."""

    def __init__(self, rng, aim_bias=None):
        self.rng = rng
        self.direction = 0
        self.dir_left = 0
        self.aim = (100, 0)
        self.aim_left = 0
        self.fire_held = False
        self.fire_counter = 0
        self.fire_left = 0
        self.hook_left = 0
        self.hook = 0
        self.jump_left = 0
        self.weapon_left = 0
        self.wanted = 0
        self.wanted_hold = 0
        self.aim_bias = aim_bias or []

    def step(self, allow_kill=True, fire_rate=3):
        r = self.rng
        if self.dir_left <= 0:
            self.direction = r.range_incl(-1, 1)
            self.dir_left = r.range_incl(5, 40)
        self.dir_left -= 1
        if self.aim_left <= 0:
            if self.aim_bias and r.chance(1, 2):
                tx, ty = self.aim_bias[r.below(len(self.aim_bias))]
                self.aim = (tx + r.range_incl(-20, 20), ty + r.range_incl(-20, 20))
            else:
                self.aim = (r.range_incl(-900, 900), r.range_incl(-700, 700))
            if self.aim == (0, 0):
                self.aim = (1, 0)
            self.aim_left = r.range_incl(8, 30)
        self.aim_left -= 1
        # fire: hold for a while (full-auto weapons) or tap; the wire field is an edge counter.
        if self.fire_left <= 0:
            self.fire_held = r.chance(fire_rate, 4)
            self.fire_left = r.range_incl(3, 60)
            if r.chance(1, 5):
                self.fire_left = r.range_incl(1, 3)
        self.fire_left -= 1
        want = 1 if self.fire_held else 0
        if (self.fire_counter & 1) != want:
            self.fire_counter = (self.fire_counter + 1) & 0x3F
        if self.hook_left <= 0:
            self.hook = 1 if r.chance(1, 3) else 0
            self.hook_left = r.range_incl(3, 50)
        self.hook_left -= 1
        jump = 0
        if self.jump_left <= 0:
            jump = 1 if r.chance(1, 2) else 0
            self.jump_left = r.range_incl(1, 12)
        self.jump_left -= 1
        # `HandleWeaponSwitch` reads `m_Input.m_WantedWeapon` (the *previous* tick's applied input,
        # `character.cpp:446`) next to the latest one, so a request must be held for >= 2 ticks.
        if self.weapon_left <= 0:
            self.wanted = r.range_incl(1, 6) if r.chance(3, 5) else 0  # 1-based on the wire
            self.weapon_left = r.range_incl(20, 140)
            self.wanted_hold = r.range_incl(2, 5) if self.wanted else 0
        self.weapon_left -= 1
        wanted = self.wanted
        if self.wanted_hold > 0:
            self.wanted_hold -= 1
        else:
            wanted = 0
        kill = 1 if allow_kill and r.chance(1, 600 if allow_kill is True else int(allow_kill)) else 0
        return [self.direction, self.aim[0], self.aim[1], -1, jump, self.fire_counter, self.hook, 0, wanted, 0, 0, kill]


def write_scenario(path, rawmap_name, rawmap_bytes, chars, inputs, cfg_lines, seed, generator_id):
    out = bytearray(b"SCN1")
    out += struct.pack("<I", 3)
    out += struct.pack("<B", 1)
    name = rawmap_name.encode()
    out += struct.pack("<H", len(name)) + name
    out += hashlib.sha256(rawmap_bytes).digest()
    out += struct.pack("<B", 0)  # no_weak_hook (v3 uses cfg lines)
    out += struct.pack("<I", 0)  # tuning overrides
    out += struct.pack("<I", len(chars))
    for cid, x, y, team in chars:
        out += struct.pack("<Iiii", cid, x, y, team)
    ticks = len(inputs)
    out += struct.pack("<I", ticks)
    for tick_inputs in inputs:
        for row in tick_inputs:
            assert len(row) == 12
            out += struct.pack("<12i", *row)
    out += struct.pack("<I", len(cfg_lines))
    for line in cfg_lines:
        b = line.encode()
        out += struct.pack("<H", len(b)) + b
    g = generator_id.encode()
    out += struct.pack("<H", len(g)) + g
    out += struct.pack("<Q", seed)
    with open(path, "wb") as f:
        f.write(out)


# --- map templates -----------------------------------------------------------------------------


def free_cells(m, margin=2):
    return [(x, y) for y in range(margin, m.h - margin) for x in range(margin, m.w - margin) if m.free(x, y)]


def arena(w, h, rng, pillars=4, freeze=True):
    m = Map(w, h)
    m.border()
    for _ in range(pillars):
        px, py = rng.range_incl(4, w - 6), rng.range_incl(3, h - 5)
        m.block(px, py, px + rng.range_incl(0, 2), py + rng.range_incl(0, 2))
    if freeze:
        fx, fy = rng.range_incl(3, w - 8), h - 3
        m.block(fx, fy, fx + 4, fy, T_FREEZE)
    return m


def place_chars(m, rng, n, near=None, avoid_hazard=True):
    cells = [c for c in free_cells(m) if m.game[m._i(*c)][0] == 0]
    chars = []
    used = set()
    for cid in range(n):
        for attempt in range(200):
            if near and cid < len(near) and attempt == 0:
                x, y = near[cid]
            else:
                x, y = cells[rng.below(len(cells))]
            if (x, y) not in used and m.free(x, y):
                used.add((x, y))
                chars.append((cid, x * TILE + 16, y * TILE + 16, 0))
                break
    return chars


def add_weapon_pickups(m, rng, cells):
    """Weapon pickups next to the given cells so the spawning tee picks several up at tick 0."""
    for (x, y) in cells:
        for dx, ent in ((-1, E_WEAPON_SHOTGUN), (1, E_WEAPON_LASER)):
            if m.free(x + dx, y):
                m.entity(x + dx, y, ent)
        if m.free(x, y - 1) and rng.chance(1, 2):
            m.entity(x, y - 1, E_WEAPON_GRENADE)


def light_at(m, x, y, direction, speed_entity, length_marker, oc_marker=None, switch_no=0):
    """A light source at (x, y): `speed_entity` is E_LASER_STOP/CW/CCW..., `direction` 0..7 indexes
    gamecontroller.cpp's aSides (S, SE, E, NE, N, NW, W, SW), `length_marker` E_LASER_SHORT..LONG,
    `oc_marker` an optional C/O speed marker two cells out. Cells are written on the game layer
    (switch_no == 0) or the switch layer (switch_no > 0)."""
    offs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    dx, dy = offs[direction]
    put = (lambda cx, cy, ent: m.entity(cx, cy, ent)) if switch_no == 0 else (lambda cx, cy, ent: m.switch_entity(cx, cy, switch_no, ent))
    put(x, y, speed_entity)
    put(x + dx, y + dy, length_marker)
    if oc_marker is not None:
        put(x + 2 * dx, y + 2 * dy, oc_marker)


def switch_pair(m, x_open, y_open, x_close, y_close, number):
    m.set_switch(x_open, y_open, number, T_SWITCHOPEN)
    m.set_switch(x_close, y_close, number, T_SWITCHCLOSE)


# --- scenario recipes --------------------------------------------------------------------------


def recipe_lasers(seed, variant):
    rng = SplitMix64(seed)
    m = arena(40, 22, rng, pillars=5)
    n = rng.range_incl(2, 4)
    near = [(rng.range_incl(5, 34), rng.range_incl(4, 17)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    add_weapon_pickups(m, rng, near)
    chars = place_chars(m, rng, n, near=near)
    cfg = []
    if variant == 1:
        cfg += ["sv_old_laser 1"]
    if variant == 2:
        cfg += ["tune laser_bounce_num 5", "tune laser_reach 1400", "tune laser_bounce_cost 40"]
    if variant == 3:
        cfg += ["tune shotgun_strength 30", "tune laser_bounce_delay 40"]
    if variant == 4:
        m.settings.append("sv_hit 0")  # a map setting sticks; a --cfg line would be wiped (section 12.1)
    if variant == 5:
        cfg += ["sv_destroy_lasers_on_death 1"]
        chars = [(c[0], c[1], c[2], 3 if c[0] % 2 == 0 else 0) for c in chars]
    if variant == 6:
        chars = [(c[0], c[1], c[2], 5) for c in chars]
        cfg += ["tune laser_bounce_num 3"]
    return m, chars, cfg


def recipe_laser_tele(seed, variant):
    rng = SplitMix64(seed)
    m = arena(44, 24, rng, pillars=3, freeze=True)
    # tele-in-weapon tiles on walls and the floor, with 1..3 outs per number (RandomOr0 coverage)
    numbers = [1, 2]
    for _ in range(rng.range_incl(1, 2)):  # a tele-in-weapon number nobody leads out of (`TeleOuts` empty)
        x, y = rng.range_incl(3, 40), rng.range_incl(3, 19)
        if m.free(x, y):
            m.set_tele(x, y, 3, T_TELEINWEAPON)
    for number in numbers:
        for _ in range(rng.range_incl(1, 2)):
            x, y = rng.range_incl(3, 40), rng.range_incl(3, 19)
            if m.free(x, y):
                m.set_tele(x, y, number, T_TELEINWEAPON)
        for _ in range(number):  # number 1 -> one out, number 2 -> two outs
            x, y = rng.range_incl(3, 40), rng.range_incl(3, 19)
            if m.free(x, y):
                m.set_tele(x, y, number, T_TELEOUT)
    # telegun tiles on the front layer (allow tele gun / blue), and chars get a telegun laser tile
    for _ in range(6):
        x, y = rng.range_incl(2, 41), rng.range_incl(2, 21)
        m.set_front(x, y, T_ALLOW_TELE_GUN if rng.chance(2, 3) else T_ALLOW_BLUE_TELE_GUN)
    n = rng.range_incl(2, 3)
    near = [(rng.range_incl(5, 38), rng.range_incl(4, 18)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    add_weapon_pickups(m, rng, near)
    for (x, y) in near:
        if variant in (0, 1, 2) and m.free(x, y + 1):
            m.set(x, y + 1, T_TELE_LASER_ENABLE)  # underfoot: telegun laser for whoever stands there
    chars = place_chars(m, rng, n, near=near)
    cfg = []
    if variant == 1:
        cfg += ["sv_old_teleport_weapons 1"]
    if variant == 2:
        cfg += ["tune laser_bounce_num 6"]
    if variant == 3:
        cfg += ["sv_old_laser 1", "tune laser_bounce_num 4"]
    return m, chars, cfg


def recipe_hitoff(seed, variant):
    """Characters standing on `HIT_DISABLE`/`HIT_ENABLE` front tiles (the per-character
    `m_LaserHitDisabled`/`m_ShotgunHitDisabled` flags: a laser whose owner cannot hit others hits only
    its owner, `laser.cpp:54-57`, and `CInteractions` sets `NoHitOthers`), plus `sv_hit 0` as a *map
    setting* (a `--cfg` line would be wiped by `sv_ddrace_tune_reset`, docs/formats.md section 12.1)."""
    rng = SplitMix64(seed)
    m = arena(40, 22, rng, pillars=4)
    n = rng.range_incl(3, 4)
    near = [(rng.range_incl(5, 34), rng.range_incl(4, 17)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    add_weapon_pickups(m, rng, near)
    chars = place_chars(m, rng, n, near=near)
    for k, (_, px, py, _) in enumerate(chars):
        if k % 2 == 0:
            m.set_front(px // TILE, py // TILE, T_HIT_DISABLE)
        elif rng.chance(1, 2):
            m.set_front(px // TILE, py // TILE, T_HIT_ENABLE)
    if variant == 1:
        m.settings.append("sv_hit 0")
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3 if c[0] < 2 else 0) for c in chars]
    return m, chars, []


def recipe_solo(seed, variant):
    """Solo tees (`TILE_SOLO_ENABLE` under some spawns): `CInteractions::m_Solo` makes a solo owner's
    laser hit only itself, and lasers of others skip solo tees (`CanCollide`); with turrets and
    draggers around, solo targets are handled separately from their team (`gun.cpp:83`, `dragger.cpp:103`)."""
    rng = SplitMix64(seed)
    m = arena(40, 22, rng, pillars=3)
    if variant >= 1:
        for _ in range(2):
            x, y = rng.range_incl(3, 36), rng.range_incl(3, 18)
            if m.free(x, y):
                m.entity(x, y, E_PLASMA if variant == 1 else E_DRAGGER_NORMAL)
    n = rng.range_incl(3, 4)
    near = [(rng.range_incl(5, 34), rng.range_incl(4, 17)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    add_weapon_pickups(m, rng, near)
    chars = place_chars(m, rng, n, near=near)
    for k, (_, px, py, _) in enumerate(chars):
        if k % 2 == 0:
            m.set_front(px // TILE, py // TILE, T_SOLO_ENABLE)
    return m, chars, []


def recipe_deathlaser(seed, variant):
    """Lasers in flight when their owner dies (`kill` every ~120 ticks per tee): the default keeps them
    flying (and `NoHitOthers = sv_hit`), `sv_destroy_lasers_on_death` removes them; team members die
    out of teams (`RemoveEntitiesFromPlayer`)."""
    rng = SplitMix64(seed)
    m = arena(36, 18, rng, pillars=3)
    n = rng.range_incl(2, 4)
    near = [(rng.range_incl(4, 31), rng.range_incl(4, 13)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    add_weapon_pickups(m, rng, near)
    for (x, y) in near:
        m.entity(x, y - 2, E_SPAWN) if m.free(x, y - 2) else None
    chars = place_chars(m, rng, n, near=near)
    cfg = []
    if variant == 1:
        cfg += ["sv_destroy_lasers_on_death 1"]
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3) for c in chars]
    return m, chars, cfg


def recipe_telegun(seed, variant):
    """Telegun lasers: every wall cell carries a front-layer `ALLOW_TELE_GUN` (or the blue one), so a
    rifle shot's last bounce is on a tile that arms the teleport (`laser.cpp:225-253`: the front tile
    is read at the *collision* point, i.e. on the wall cell itself), and the tees stand on a
    `TELE_LASER_ENABLE` tile that gives them the telegun laser in the first place."""
    rng = SplitMix64(seed)
    m = arena(40, 20, rng, pillars=3, freeze=True)
    for x in range(m.w):
        for y in (0, m.h - 1):
            m.set_front(x, y, T_ALLOW_TELE_GUN if (variant != 0 or rng.chance(2, 3)) else T_ALLOW_BLUE_TELE_GUN)
    for y in range(m.h):
        for x in (0, m.w - 1):
            m.set_front(x, y, T_ALLOW_TELE_GUN if (variant != 0 or rng.chance(2, 3)) else T_ALLOW_BLUE_TELE_GUN)
    n = rng.range_incl(2, 3)
    near = [(rng.range_incl(5, 34), rng.range_incl(4, 15)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    add_weapon_pickups(m, rng, near)
    chars = place_chars(m, rng, n, near=near)
    for (_, px, py, _) in chars:
        m.set_front(px // TILE, py // TILE, T_TELE_LASER_ENABLE)  # the spawn cell itself
    cfg = []
    if variant == 1:
        cfg += ["sv_old_laser 1"]
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3) for c in chars]
        cfg += ["tune laser_bounce_num 3"]
    return m, chars, cfg


def recipe_ninja(seed, variant):
    rng = SplitMix64(seed)
    m = arena(40, 20, rng, pillars=4)
    n = rng.range_incl(2, 4)
    near = [(rng.range_incl(5, 34), rng.range_incl(4, 15)) for _ in range(n)]
    near = [c for c in near if m.free(*c)] or [(6, 6)]
    for (x, y) in near:
        if m.free(x + 1, y):
            m.entity(x + 1, y, E_POWERUP_NINJA)
        if m.free(x - 1, y):
            m.entity(x - 1, y, E_WEAPON_LASER)
    chars = place_chars(m, rng, n, near=near)
    cfg = []
    if variant == 1:
        chars = [(c[0], c[1], c[2], 3) for c in chars]
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3 if c[0] != 1 else 0) for c in chars]
        m.set_front(near[0][0], near[0][1] + 1, T_SOLO_ENABLE)
    if variant == 3:
        cfg += ["tune ground_elasticity_x 0.5", "tune ground_elasticity_y 0.5"]
    return m, chars, cfg


def recipe_lights(seed, variant):
    rng = SplitMix64(seed)
    m = arena(46, 26, rng, pillars=2, freeze=False)
    speeds = [E_LASER_STOP, E_LASER_SLOW_CW, E_LASER_NORMAL_CW, E_LASER_FAST_CW, E_LASER_SLOW_CCW, E_LASER_NORMAL_CCW, E_LASER_FAST_CCW]
    lengths = [E_LASER_SHORT, E_LASER_MEDIUM, E_LASER_LONG]
    ocs = [None, E_LASER_C_SLOW, E_LASER_C_NORMAL, E_LASER_C_FAST, E_LASER_O_SLOW, E_LASER_O_NORMAL, E_LASER_O_FAST]
    placed = 0
    for _ in range(40):
        if placed >= rng.range_incl(5, 8):
            break
        x, y = rng.range_incl(6, 39), rng.range_incl(6, 19)
        d = rng.below(8)
        offs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        cells = [(x, y), (x + offs[d][0], y + offs[d][1]), (x + 2 * offs[d][0], y + 2 * offs[d][1])]
        if not all(m.free(cx, cy) for cx, cy in cells):
            continue
        switch_no = 0
        if variant in (1, 2) and rng.chance(1, 2):
            switch_no = rng.range_incl(1, 2)
        light_at(m, x, y, d, speeds[rng.below(len(speeds))], lengths[rng.below(len(lengths))], ocs[rng.below(len(ocs))], switch_no)
        placed += 1
    if variant in (1, 2):
        switch_pair(m, 3, 3, 42, 3, 1)
        switch_pair(m, 3, 22, 42, 22, 2)
    n = rng.range_incl(2, 4)
    chars = place_chars(m, rng, n)
    cfg = []
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3 if c[0] == 0 else 0) for c in chars]
    if variant == 3:
        cfg += ["sv_freeze_delay 1"]
    return m, chars, cfg


def recipe_draggers(seed, variant):
    rng = SplitMix64(seed)
    m = arena(46, 26, rng, pillars=3, freeze=True)
    kinds = [E_DRAGGER_WEAK, E_DRAGGER_NORMAL, E_DRAGGER_STRONG, E_DRAGGER_WEAK_NW, E_DRAGGER_NORMAL_NW, E_DRAGGER_STRONG_NW]
    # a wall segment between some draggers and the tees, so IgnoreWalls matters
    m.block(20, 8, 20, 17)
    for i in range(rng.range_incl(4, 7)):
        x, y = rng.range_incl(4, 41), rng.range_incl(4, 21)
        if not m.free(x, y):
            continue
        ent = kinds[rng.below(len(kinds))]
        if variant in (1, 2) and rng.chance(1, 2):
            m.switch_entity(x, y, rng.range_incl(1, 2), ent)
        else:
            m.entity(x, y, ent)
    if variant in (1, 2):
        switch_pair(m, 3, 3, 42, 3, 1)
        switch_pair(m, 3, 22, 42, 22, 2)
    n = rng.range_incl(2, 4)
    chars = place_chars(m, rng, n)
    cfg = []
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3 if c[0] % 2 == 0 else 0) for c in chars]
    if variant == 3:
        cfg += ["sv_dragger_range 300"]
    if variant == 4:
        m.set_front(chars[0][1] // TILE, chars[0][2] // TILE + 1, T_SOLO_ENABLE)
        cfg += []
    if variant == 5:
        chars = [(c[0], c[1], c[2], 3) for c in chars]
    return m, chars, cfg


def recipe_turrets(seed, variant):
    rng = SplitMix64(seed)
    m = arena(46, 26, rng, pillars=3, freeze=True)
    kinds = [E_PLASMAE, E_PLASMAF, E_PLASMA, E_PLASMAU]
    for i in range(rng.range_incl(4, 8)):
        x, y = rng.range_incl(3, 42), rng.range_incl(3, 22)
        if not m.free(x, y):
            continue
        ent = kinds[rng.below(len(kinds))]
        if variant in (1, 2) and rng.chance(1, 2):
            m.switch_entity(x, y, rng.range_incl(1, 2), ent)
        else:
            m.entity(x, y, ent)
    if variant in (1, 2):
        switch_pair(m, 3, 3, 42, 3, 1)
        switch_pair(m, 3, 22, 42, 22, 2)
    n = rng.range_incl(2, 4)
    chars = place_chars(m, rng, n)
    cfg = []
    if variant == 2:
        chars = [(c[0], c[1], c[2], 3 if c[0] % 2 == 0 else 0) for c in chars]
    if variant == 3:
        cfg += ["sv_plasma_per_sec 20"]
    if variant == 4:
        cfg += ["sv_plasma_per_sec 1", "sv_plasma_range 400"]
    if variant == 5:
        m.set_front(chars[0][1] // TILE, chars[0][2] // TILE + 1, T_SOLO_ENABLE)
        chars = [(c[0], c[1], c[2], 3 if c[0] else 0) for c in chars]
    if variant == 6:
        cfg += ["tune explosion_strength 20"]
    return m, chars, cfg


def recipe_mixed(seed, variant):
    rng = SplitMix64(seed)
    m = arena(52, 30, rng, pillars=6, freeze=True)
    for ent in (E_PLASMAE, E_PLASMAF, E_PLASMA, E_PLASMAU, E_DRAGGER_NORMAL, E_DRAGGER_STRONG_NW, E_DRAGGER_WEAK):
        for _ in range(100):
            x, y = rng.range_incl(3, 48), rng.range_incl(3, 26)
            if m.free(x, y):
                m.entity(x, y, ent)
                break
    for _ in range(3):
        for _ in range(100):
            x, y = rng.range_incl(6, 45), rng.range_incl(6, 23)
            d = rng.below(8)
            offs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            cells = [(x, y), (x + offs[d][0], y + offs[d][1]), (x + 2 * offs[d][0], y + 2 * offs[d][1])]
            if all(m.free(cx, cy) for cx, cy in cells):
                light_at(m, x, y, d, [E_LASER_STOP, E_LASER_NORMAL_CW, E_LASER_FAST_CCW][rng.below(3)], E_LASER_MEDIUM, [None, E_LASER_C_NORMAL, E_LASER_O_SLOW][rng.below(3)])
                break
    near = [(rng.range_incl(5, 46), rng.range_incl(4, 25)) for _ in range(4)]
    near = [c for c in near if m.free(*c)]
    add_weapon_pickups(m, rng, near)
    for (x, y) in near[:2]:
        if m.free(x, y + 1):
            m.entity(x, y + 1, E_POWERUP_NINJA)
    n = rng.range_incl(3, 5)
    chars = place_chars(m, rng, n, near=near)
    cfg = []
    if variant == 1:
        chars = [(c[0], c[1], c[2], 3 if c[0] % 2 == 0 else 0) for c in chars]
    return m, chars, cfg


# (prefix, recipe, variants, seeds-per-variant, ticks, allow_kill). `calm` is the `mixed` map with
# the random /kill requests switched off: nobody dies, so the whole trace is respawn-free (the
# allocation tests can replay it end to end).
PLAN = [
    ("lasers", recipe_lasers, 7, 3, 1200, True),
    ("lasertele", recipe_laser_tele, 4, 4, 1200, True),
    ("ninja", recipe_ninja, 4, 4, 900, True),
    ("lights", recipe_lights, 4, 4, 1000, True),
    ("draggers", recipe_draggers, 6, 3, 1000, True),
    ("turrets", recipe_turrets, 7, 3, 1000, True),
    ("mixed", recipe_mixed, 2, 4, 1500, True),
    ("calm", recipe_mixed, 2, 2, 1500, False),
    ("telegun", recipe_telegun, 3, 4, 1000, True),
    ("hitoff", recipe_hitoff, 3, 4, 1000, True),
    ("solo", recipe_solo, 3, 4, 1000, True),
    ("deathlaser", recipe_deathlaser, 3, 4, 1200, 120),
]


def build_scenario(prefix, recipe, variant, k, ticks, seed_base, allow_kill=True):
    seed = seed_base + k
    m, chars, cfg = recipe(seed, variant)
    rng = SplitMix64(seed ^ 0xA5A5A5A5)
    drivers = []
    spawn_aims = [(c[1], c[2]) for c in chars]
    for c in chars:
        others = [(x - c[1], y - c[2]) for (_, x, y, _) in chars if (x, y) != (c[1], c[2])]
        drivers.append(InputDriver(SplitMix64(rng.next_u64()), aim_bias=others))
    inputs = [[d.step(allow_kill=allow_kill) for d in drivers] for _ in range(ticks)]
    name = f"{prefix}_v{variant}_s{seed}"
    return name, seed, m, chars, inputs, cfg


def run_oracle(oracle, name, seed, out_dir, rawmap_bytes, check_determinism):
    cov = os.path.join(out_dir, f"{name}.cov.json")
    trb = os.path.join(out_dir, f"{name}.trb")
    digests = []
    for attempt in range(2 if check_determinism else 1):
        with tempfile.TemporaryDirectory() as storage:
            cmd = [
                oracle,
                "--storage-dir", storage,
                "--rawmap", os.path.join(out_dir, f"{name}.rawmap"),
                "--scenario", os.path.join(out_dir, f"{name}.scn"),
                "--seed", str(seed),
                "--out", trb,
                "--coverage-out", cov,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                raise SystemExit(f"oracle failed for {name}")
        with open(trb, "rb") as f:
            digests.append(hashlib.sha256(f.read()).hexdigest())
    if check_determinism and digests[0] != digests[1]:
        raise SystemExit(f"NON-DETERMINISTIC trace {name}: {digests}")
    return digests[0]


def write_sha256sums(out_dir):
    """`SHA256SUMS` over every trace, scenario and rawmap in `out_dir`, sorted by file name."""
    lines = []
    for name in sorted(os.listdir(out_dir)):
        if name.endswith((".trb", ".scn", ".rawmap")):
            with open(os.path.join(out_dir, name), "rb") as f:
                lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {name}\n")
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        f.writelines(lines)


def scripted_row(direction=0, tx=100, ty=0, jump=0, fire=0, hook=0, wanted=0, kill=0):
    return [direction, tx, ty, -1, jump, fire, hook, 0, wanted, 0, 0, kill]


def teamkill_scenarios():
    """Review 1.6b F2: a team-3 owner is killed (the kill bit) around the tick its laser/shotgun shot
    bounces off the far wall. Death moves it to team 0, i.e. `RemoveEntitiesFromPlayer` removes the
    laser at once (`teams.cpp:497`); the map setting `sv_hit 0` makes a *dead* owner's laser able to hit
    the team-0 victim (`NoHitOthers = sv_hit`), so a laser that wrongly ticks once more pulls it.
    Yields `(name, seed, map, chars, inputs, cfg)` for both weapons and kill offsets 5..11."""
    for wanted, wname in ((3, "sg"), (5, "rf")):
        for off in range(5, 12):
            m = Map(40, 14)
            m.border()
            m.block(1, 9, 38, 9)  # floor
            m.block(30, 1, 30, 8)  # wall the shot bounces off
            m.settings.append("sv_hit 0")
            m.entity(4, 8, E_WEAPON_SHOTGUN if wanted == 3 else E_WEAPON_LASER)
            chars = [(0, 5 * TILE + 16, 8 * TILE + 16, 3), (1, 15 * TILE + 16, 8 * TILE + 16, 0)]
            fire_at = 40
            inputs = []
            for i in range(120):
                r0 = scripted_row(
                    wanted=wanted if 5 <= i < 12 else 0,
                    fire=1 if i == fire_at else (2 if i > fire_at else 0),
                    kill=1 if i == fire_at + off else 0,
                )
                inputs.append([r0, scripted_row(tx=-100)])
            yield f"teamkill_{wname}_off{off}", 50000 + off, m, chars, inputs, ["tune laser_reach 2000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--oracle", default=os.path.expanduser("~/aiddnet/build/oracle-b/build/ddai_oracle_server"))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-determinism-check", action="store_true")
    ap.add_argument("--seed-base", type=int, default=30000)
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    count = 0
    for pi, (prefix, recipe, variants, seeds, ticks, allow_kill) in enumerate(PLAN):
        if args.only and not prefix.startswith(args.only):
            continue
        for variant in range(variants):
            for k in range(seeds):
                name, seed, m, chars, inputs, cfg = build_scenario(
                    prefix, recipe, variant, k, ticks, args.seed_base + pi * 1000 + variant * 50, allow_kill
                )
                rawmap_bytes = m.to_bytes()
                with open(os.path.join(args.out_dir, f"{name}.rawmap"), "wb") as f:
                    f.write(rawmap_bytes)
                write_scenario(
                    os.path.join(args.out_dir, f"{name}.scn"),
                    os.path.join(args.out_dir, f"{name}.rawmap"),
                    rawmap_bytes,
                    chars,
                    inputs,
                    cfg,
                    seed,
                    "gen_stage_b.py/1",
                )
                digest = run_oracle(args.oracle, name, seed, args.out_dir, rawmap_bytes, not args.no_determinism_check)
                count += 1
                print(f"[gen_stage_b] {name}: ok {digest[:12]}", file=sys.stderr)
    if not args.only or "teamkill".startswith(args.only):
        for name, seed, m, chars, inputs, cfg in teamkill_scenarios():
            rawmap_bytes = m.to_bytes()
            with open(os.path.join(args.out_dir, f"{name}.rawmap"), "wb") as f:
                f.write(rawmap_bytes)
            write_scenario(
                os.path.join(args.out_dir, f"{name}.scn"),
                os.path.join(args.out_dir, f"{name}.rawmap"),
                rawmap_bytes,
                chars,
                inputs,
                cfg,
                seed,
                "gen_stage_b.py/1",
            )
            digest = run_oracle(args.oracle, name, seed, args.out_dir, rawmap_bytes, not args.no_determinism_check)
            count += 1
            print(f"[gen_stage_b] {name}: ok {digest[:12]}", file=sys.stderr)
    write_sha256sums(args.out_dir)
    print(f"[gen_stage_b] {count} scenarios", file=sys.stderr)


if __name__ == "__main__":
    main()
