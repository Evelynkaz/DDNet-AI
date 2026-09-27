#!/usr/bin/env python3
"""Generates Rust source for `crates/ddai-net/src/generated/` from DDNet's own protocol
description, `datasrc/network.py` + `datasrc/datatypes.py` (the same files DDNet's own
`datasrc/compile.py` reads to generate its C++ `protocol.h`/`protocol.cpp`).

Python 3 stdlib only (task 2.2b acceptance criterion 1) — no pip packages. The DDNet source tree
is never imported as a package; instead `datasrc/network.py` and `datasrc/datatypes.py` are loaded
directly off disk with `importlib`, exactly the two files DDNet's own generator needs (network.py's
only import is `datatypes`; datatypes.py has no non-stdlib imports either) — this reproduces
DDNet's own field lists/ordering/defaults exactly rather than re-parsing the Python source as text.

Usage:
    python3 tools/ddnet-protocol-gen/generate.py <path to DDNet src tree> [--out <dir>]

<path to DDNet src tree> is the directory containing `datasrc/network.py` (i.e. the repository
root of a DDNet checkout, NOT the `datasrc/` directory itself). `--out` defaults to
`crates/ddai-net/src/generated` relative to this script's repository.

Pinned to DDNet commit c9d208138f85755521f16a0096b6fe036c5c8698 ("20.1") — see docs/formats.md
and each generated file's header. Regenerating against a different commit's tree will change the
committed output; that is expected when DDNet's protocol changes and is exactly the point of this
tool (task 2.2b's goal: "keep up with protocol drift").

After generating, this script formats the output with `rustfmt` (if the `rustfmt` binary is on
PATH) so re-running it reproduces the committed files byte-for-byte; without `rustfmt` on PATH the
files are still valid Rust but may not byte-match the committed (rustfmt'd) versions.
"""
from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
import types
from pathlib import Path

PINNED_COMMIT = "c9d208138f85755521f16a0096b6fe036c5c8698"
PINNED_LABEL = "20.1"

HEADER = f"""\
// GENERATED — do not edit by hand.
//
// Produced by `tools/ddnet-protocol-gen/generate.py` from DDNet's own protocol description
// (`datasrc/network.py` + `datasrc/datatypes.py`), commit {PINNED_COMMIT} ("{PINNED_LABEL}").
// Regenerate with (from the repository root):
//
//   python3 tools/ddnet-protocol-gen/generate.py ~/aiddnet/build/ddnet-20.1/src
//
// Re-running against the same pinned commit's tree reproduces these files byte-for-byte (the
// script formats its own output with `rustfmt`). See `tools/ddnet-protocol-gen/README.md`.
"""

# --- Symbol table for the C++ constant expressions used as NetIntRange bounds / defaults in
# datasrc/network.py + datatypes.py. DDNet's own generator (datasrc/compile.py) just emits these
# expressions verbatim into C++ and lets the C++ compiler resolve them against real constants
# elsewhere in the DDNet source tree; since we generate Rust, we resolve them here instead, each
# one cited to its DDNet source location (all read at the pinned commit above).
SYMBOLS: dict[str, int] = {
    "MAX_CLIENTS": 128,  # src/engine/shared/protocol.h: `MAX_CLIENTS = 128`
    "NUM_WEAPONS": 6,  # datasrc/content.py: Weapons() lists hammer,gun,shotgun,grenade,laser,ninja
    "NUM_SOUNDS": 41,  # datasrc/content.py: 41 `container.sounds.Add(...)` calls
    "NUM_EMOTICONS": 16,  # datatypes.py: len(Emoticons)
    "TEAM_RED": 0,  # datatypes.py: Teams = ["ALL","SPECTATORS","RED","BLUE",...], Enum(start=-2)
    "TEAM_BLUE": 1,
    "TEAM_SPECTATORS": -1,
    "SPEC_FREEVIEW": -1,  # datasrc/datatypes.py RawHeader: `SPEC_FREEVIEW=-1, SPEC_FOLLOW=-2`
    "FLAG_MISSING": -3,  # datasrc/datatypes.py RawHeader: `FLAG_MISSING=-3, FLAG_ATSTAND, FLAG_TAKEN`
    "AUTHED_NO": 0,  # datatypes.py: Authed = ["NO","HELPER","MOD","ADMIN"], Enum(start=0)
    "AUTHED_ADMIN": 3,
    "SAVESTATE_PENDING": 0,  # datatypes.py: SaveStates = ["PENDING","DONE","FALLBACKFILE","WARNING","ERROR"]
    "SAVESTATE_ERROR": 4,
    "min_int": -2147483648,  # datasrc/compile.py: `min_int = 0x80000000`
    "max_int": 2147483647,  # datasrc/compile.py: `max_int = 0x7fffffff`
    "MIN_TICK": 0,  # src/engine/shared/protocol.h
    "MAX_TICK": 0x6FFFFFFF,
    "TuneZone::OVERRIDE_NONE": -1,  # src/engine/shared/protocol.h `namespace TuneZone`
    "TuneZone::NUM": 256,
    "FinishTime::NOT_FINISHED_TIMESCORE": -9999,  # src/engine/shared/protocol.h `namespace FinishTime`
    "FinishTime::NOT_FINISHED_MILLIS": -1,
    "FinishTime::UNSET": -2,
    "WEAPON_GAME": -3,  # datasrc/compile.py's generated header: WEAPON_GAME=-3, WEAPON_SELF=-2, WEAPON_WORLD=-1
    "WEAPON_SELF": -2,
    "WEAPON_WORLD": -1,
}


def resolve_expr(expr: str) -> int:
    """Resolves one bound/default expression exactly as it appears in datasrc/network.py
    (already stringified by `datatypes.NetVariable`/`NetIntRange`), e.g. "0", "-1", "MAX_CLIENTS",
    "MAX_CLIENTS-1", "TuneZone::NUM-1"."""
    expr = expr.strip()
    try:
        return int(expr)
    except ValueError:
        pass
    if expr in SYMBOLS:
        return SYMBOLS[expr]
    m = re.match(r"^(.+?)([+-]\d+)$", expr)
    if m and m.group(1) in SYMBOLS:
        return SYMBOLS[m.group(1)] + int(m.group(2))
    raise ValueError(f"cannot resolve symbolic bound/default {expr!r} — add it to SYMBOLS")


def load_datasrc(ddnet_src: Path) -> types.ModuleType:
    """Imports `datasrc/datatypes.py` then `datasrc/network.py` from the given DDNet source tree,
    exactly as `datasrc/compile.py` does (both run with `datasrc/` as the working directory /
    first sys.path entry so `network.py`'s `from datatypes import ...` resolves)."""
    datasrc_dir = ddnet_src / "datasrc"
    network_py = datasrc_dir / "network.py"
    if not network_py.is_file():
        raise SystemExit(f"error: {network_py} not found — is {ddnet_src} a DDNet source tree root?")

    sys.path.insert(0, str(datasrc_dir))
    try:
        # `datatypes` first (network.py imports names from it by `from datatypes import ...`,
        # which requires the module to be importable under exactly that name).
        importlib.import_module("datatypes")
        network = importlib.import_module("network")
    finally:
        sys.path.remove(str(datasrc_dir))
    return network


# --- Name conversion (DDNet's PascalCase-with-m_-prefix C++ convention -> idiomatic Rust). These
# only affect our own generated Rust identifiers, never the wire format, so any reasonable,
# stable, deterministic scheme is fine.


def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def rust_type_name(ddnet_name: str) -> str:
    """"Sv_TuneParams" -> "SvTuneParams", "DDNetCharacter" -> "DDNetCharacter"."""
    return ddnet_name.replace("_", "")


def rust_snake_name(ddnet_name: str) -> str:
    """"Sv_TuneParams" -> "sv_tune_params"."""
    return camel_to_snake(ddnet_name.replace("_", ""))


def rust_field_name(cpp_field: str) -> str:
    """"m_ClientId" -> "client_id", "m_pMessage" -> "message", "m_aName" -> "name"."""
    assert cpp_field.startswith("m_"), cpp_field
    rest = cpp_field[2:]
    if rest[:1] in ("p", "a") and rest[1:2].isupper():
        rest = rest[1:]
    snake = camel_to_snake(rest)
    # A handful of DDNet field names collide with Rust keywords once snake_cased.
    if snake in ("type", "move"):
        snake += "_"
    return snake


# --- Field classification. Every field type actually used in datasrc/network.py at the pinned
# commit is handled explicitly below; anything else raises, so a future protocol change that adds
# a new field kind fails loudly at generation time instead of silently emitting something wrong.


class Field:
    def __init__(self, var, array_len: int | None = None, is_int_string: bool = False):
        self.var = var
        self.name = rust_field_name(var.name if array_len is None else var.base_name if hasattr(var, "base_name") else var.name)
        self.array_len = array_len
        self.is_int_string = is_int_string
        self.has_range = hasattr(var, "min") and hasattr(var, "max")
        self.default = None if var.default is None else resolve_expr(var.default)


def object_fields(variables) -> list[Field]:
    """Flattens a NetObject's variable list into per-field descriptors, expanding NetArray /
    NetTwIntString into a single array-typed Rust field each (not N scalar fields — unlike DDNet's
    C++ struct layout, which really does declare N separate scalar members, a Rust `[i32; N]` (or,
    for NetTwIntString, a decoded `String`) is equivalent on the wire and much nicer to use)."""
    datatypes = sys.modules["datatypes"]
    out: list[Field] = []
    for v in variables:
        if isinstance(v, datatypes.NetTwIntString):
            out.append(Field(v, array_len=v.size, is_int_string=True))
        elif isinstance(v, datatypes.NetArray):
            out.append(Field(v, array_len=v.size))
        elif isinstance(v, (datatypes.NetIntAny, datatypes.NetIntRange, datatypes.NetBool, datatypes.NetTick, datatypes.NetTickStrict)):
            out.append(Field(v))
        else:
            raise ValueError(f"unhandled object field type {type(v).__name__} for {v.name!r}")
    return out


def message_fields(variables) -> list[tuple[str, object]]:
    """Returns (kind, var) pairs for a NetMessage's variables. `kind` is "int", "int_range" or
    "string" (with the exact SanitizeMode DDNet's generated `emit_unpack_msg` would use)."""
    datatypes = sys.modules["datatypes"]
    out = []
    for v in variables:
        if isinstance(v, datatypes.NetString):
            out.append(("string_sanitize", v))
        elif isinstance(v, datatypes.NetStringStrict):
            out.append(("string_sanitize_cc_ws", v))
        elif isinstance(v, datatypes.NetStringHalfStrict):
            out.append(("string_sanitize_cc", v))
        elif isinstance(v, (datatypes.NetBool, datatypes.NetTickStrict, datatypes.NetIntRange)):
            out.append(("int_range", v))
        elif isinstance(v, datatypes.NetIntAny):
            out.append(("int", v))
        else:
            raise ValueError(f"unhandled message field type {type(v).__name__} for {v.name!r}")
    return out


def rust_lit(n: int) -> str:
    return str(n)


I32_MIN = -2147483648
I32_MAX = 2147483647


# --- Emitters


def emit_enums(network) -> str:
    lines = [HEADER, "//! Enums and bitflags from `datasrc/datatypes.py`.", ""]
    for e in network.Enums:
        lines.append(f"/// `{e.name}_*` (`datasrc/datatypes.py`).")
        lines.append(f"pub mod {e.name.lower()} {{")
        for i, v in enumerate(e.values):
            lines.append(f"    pub const {v}: i32 = {e.start + i};")
        lines.append(f"    pub const NUM: usize = {len(e.values)};")
        lines.append("}")
        lines.append("")
    for f in network.Flags:
        lines.append(f"/// `{f.name}FLAG_*` bit flags (`datasrc/datatypes.py`).")
        lines.append(f"pub mod {f.name.lower()}flag {{")
        for i, v in enumerate(f.values):
            lines.append(f"    pub const {v}: i32 = 1 << {i};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def emit_object_struct(obj, all_objects, is_ex: bool, numeric_id: int | None) -> str:
    datatypes = sys.modules["datatypes"]
    name = rust_type_name(obj.name)
    variables = obj.members_from_this_and_parents(all_objects)
    fields = object_fields(variables)

    has_string_field = any(f.is_int_string for f in fields)
    out = []
    out.append(f"/// `{obj.struct_name}` (`{obj.enum_name}`).")
    if is_ex:
        out.append(f"/// UUID name: `{obj.ex}`.")
    derive = "#[derive(Debug, Clone, PartialEq, Eq)]" if has_string_field else "#[derive(Debug, Clone, Copy, PartialEq, Eq)]"
    out.append(derive)
    out.append(f"pub struct {name} {{")
    for f in fields:
        if f.is_int_string:
            ty = "String"
            out.append(
                f"    /// Decoded from {f.array_len} wire ints (see [`crate::intstr::ints_to_str`])."
            )
        elif f.array_len is not None:
            ty = f"[i32; {f.array_len}]"
        else:
            ty = "i32"
        out.append(f"    pub {f.name}: {ty},")
    out.append("}")
    out.append("")

    if numeric_id is not None:
        out.append(f"impl {name} {{")
        out.append(f"    /// `{obj.enum_name}` — the fixed, non-UUID snapshot object type id.")
        out.append(f"    pub const ID: i32 = {numeric_id};")
        size_ints = sum(f.array_len or 1 for f in fields)
        out.append(f"    /// Static size in `i32`s (`sizeof({obj.struct_name}) / 4`).")
        out.append(f"    pub const SIZE_INTS: usize = {size_ints};")
        out.append("}")
        out.append("")

    # decode(data: &[i32]) -> Option<(Self, corrections: u32)>
    out.append(f"impl {name} {{")
    out.append(
        "    /// Decodes from a snapshot item's raw `i32` data (`CNetObjHandler::SecureUnpackObj`'s"
    )
    out.append(
        "    /// per-type case): reads each field in order (missing trailing fields, for object"
    )
    out.append(
        "    /// types that declare a default, fall back to that default rather than failing —"
    )
    out.append(
        "    /// matches `GetUncompressedIntOrDefault`); returns `None` only if a field with no"
    )
    out.append(
        "    /// default ran out of data. Out-of-range fields are clamped, never rejected (matches"
    )
    out.append(
        "    /// `ClampInt`); the second element of the `Some` is how many fields were clamped."
    )
    # `hi >= I32_MAX`/`lo <= I32_MIN` bounds are always true for an `i32` and clippy correctly
    # flags comparing against them as useless — skip emitting that half of the clamp in that
    # case (DDNet's own `ClampInt` still "checks" it, harmlessly, since C++ has no such lint).
    def field_clamp_fires(f: Field) -> bool:
        if not f.has_range or f.array_len is not None:
            return False
        return resolve_expr(f.var.min) > I32_MIN or resolve_expr(f.var.max) < I32_MAX

    any_range_fires = any(field_clamp_fires(f) for f in fields)
    out.append("    pub fn decode(data: &[i32]) -> Option<(Self, u32)> {")
    out.append("        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();")
    out.append("        let mut unpacker = crate::packer::Unpacker::new(&bytes);")
    out.append(f"        let {'mut ' if any_range_fires else ''}corrections: u32 = 0;")
    for f in fields:
        if f.is_int_string:
            reads = ", ".join(
                "unpacker.get_uncompressed_int()" for _ in range(f.array_len)
            )
            out.append(f"        let __{f.name}_ints = [{reads}];")
            out.append(
                f"        let {f.name} = crate::intstr::ints_to_str(&__{f.name}_ints);"
            )
        elif f.array_len is not None:
            default = f.default if f.default is not None else 0
            elems = ", ".join(
                f"unpacker.get_uncompressed_int_or_default({rust_lit(default)})"
                for _ in range(f.array_len)
            )
            out.append(f"        let {f.name} = [{elems}];")
        else:
            lo = resolve_expr(f.var.min) if f.has_range else None
            hi = resolve_expr(f.var.max) if f.has_range else None
            check_lo = lo is not None and lo > I32_MIN
            check_hi = hi is not None and hi < I32_MAX
            binding = "let mut" if (check_lo or check_hi) else "let"
            if f.default is not None:
                out.append(
                    f"        {binding} {f.name} = unpacker.get_uncompressed_int_or_default({rust_lit(f.default)});"
                )
            else:
                out.append(f"        {binding} {f.name} = unpacker.get_uncompressed_int();")
            if check_lo:
                out.append(f"        if {f.name} < {rust_lit(lo)} {{ {f.name} = {rust_lit(lo)}; corrections += 1; }}")
                if check_hi:
                    out.append(f"        else if {f.name} > {rust_lit(hi)} {{ {f.name} = {rust_lit(hi)}; corrections += 1; }}")
            elif check_hi:
                out.append(f"        if {f.name} > {rust_lit(hi)} {{ {f.name} = {rust_lit(hi)}; corrections += 1; }}")
    out.append("        if unpacker.error() {")
    out.append("            return None;")
    out.append("        }")
    field_names = ", ".join(f.name for f in fields)
    out.append(f"        Some(({name} {{ {field_names} }}, corrections))")
    out.append("    }")
    out.append("}")
    out.append("")
    return "\n".join(out)


def emit_objects(network) -> str:
    lines = [
        HEADER,
        "//! Snapshot objects and events (`NETOBJTYPE_*`/`NETEVENTTYPE_*`) from `datasrc/network.py`.",
        "//!",
        "//! Non-UUID types 1..=20 are assigned ids in exactly the order DDNet's own",
        "//! `datasrc/compile.py` assigns them (list order, `ex is None` only) — this order is part",
        "//! of the wire format (it is the raw `int` stored in a snapshot item's key) and must never",
        "//! change independent of DDNet itself. UUID (`ex`) types carry no fixed numeric id on the",
        "//! wire at all (see `crate::snapshot`); [`EX_NAMES`] lists their UUID names in `network.py`",
        "//! declaration order for [`crate::uuid::UuidRegistry::from_names`].",
        "",
    ]
    non_extended = [o for o in network.Objects if o.ex is None]
    extended = [o for o in network.Objects if o.ex is not None]
    for i, obj in enumerate(non_extended, start=1):
        lines.append(emit_object_struct(obj, network.Objects, is_ex=False, numeric_id=i))
    for obj in extended:
        lines.append(emit_object_struct(obj, network.Objects, is_ex=True, numeric_id=None))

    lines.append("/// UUID names for every `ex` snapshot object/event, in `datasrc/network.py`")
    lines.append("/// declaration order (see the module docs).")
    lines.append("pub const EX_NAMES: &[&str] = &[")
    for obj in extended:
        lines.append(f'    "{obj.ex}",')
    lines.append("];")
    lines.append("")

    lines.append("/// Maps an ex object/event's UUID name to a decode function over its raw item")
    lines.append("/// data. `None` covers both \"unrecognised name\" and \"recognised but this")
    lines.append("/// particular item failed to decode\" — the caller (the snapshot/view layer)")
    lines.append("/// treats both the same way: the raw item is kept regardless (tolerant")
    lines.append("/// decoding), only the *typed* view is missing that one item.")
    lines.append(
        "pub fn decode_ex_by_name(name: &str, data: &[i32]) -> Option<(ExObject, u32)> {"
    )
    lines.append("    Some(match name {")
    for obj in extended:
        rname = rust_type_name(obj.name)
        lines.append(f'        "{obj.ex}" => {{ let (v, c) = {rname}::decode(data)?; (ExObject::{rname}(v), c) }}')
    lines.append("        _ => return None,")
    lines.append("    })")
    lines.append("}")
    lines.append("")

    lines.append("/// Every ex (UUID-typed) snapshot object/event, decoded.")
    lines.append("#[derive(Debug, Clone, Copy, PartialEq, Eq)]")
    lines.append("pub enum ExObject {")
    for obj in extended:
        lines.append(f"    {rust_type_name(obj.name)}({rust_type_name(obj.name)}),")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def emit_message_struct(msg) -> str:
    name = rust_type_name(msg.name)
    fields = message_fields(msg.variables)
    is_ex = msg.ex is not None

    out = []
    out.append(f"/// `{msg.struct_name}` (`{msg.enum_name}`).")
    if is_ex:
        out.append(f"/// UUID name: `{msg.ex}`.")
    out.append("#[derive(Debug, Clone, PartialEq, Eq)]")
    out.append(f"pub struct {name} {{")
    for kind, v in fields:
        ty = "String" if kind.startswith("string") else "i32"
        out.append(f"    pub {rust_field_name(v.name)}: {ty},")
    out.append("}")
    out.append("")

    snake = rust_snake_name(msg.name)
    # --- decode
    out.append(
        f"/// Decodes a `{name}` payload (mirrors `CNetObjHandler::SecureUnpackMsg`'s"
    )
    out.append(
        "/// per-type case): an out-of-range field REJECTS the whole message (`None`), unlike"
    )
    out.append("/// object decoding, which clamps — this matches DDNet exactly.")
    out.append(f"pub fn decode_{snake}(unpacker: &mut crate::packer::Unpacker) -> Option<{name}> {{")
    for kind, v in fields:
        fname = rust_field_name(v.name)
        default = None if v.default is None else resolve_expr(v.default)
        read = "unpacker.get_int()" if default is None else f"unpacker.get_int_or_default({rust_lit(default)})"
        if kind == "int":
            out.append(f"    let {fname} = {read};")
        elif kind == "int_range":
            out.append(f"    let {fname} = {read};")
            lo = resolve_expr(v.min)
            hi = resolve_expr(v.max)
            check_lo = lo > I32_MIN
            check_hi = hi < I32_MAX
            if check_lo and check_hi:
                # clippy::manual_range_contains
                out.append(f"    if !({rust_lit(lo)}..={rust_lit(hi)}).contains(&{fname}) {{ return None; }}")
            elif check_lo:
                out.append(f"    if {fname} < {rust_lit(lo)} {{ return None; }}")
            elif check_hi:
                out.append(f"    if {fname} > {rust_lit(hi)} {{ return None; }}")
        elif kind == "string_sanitize":
            out.append(f"    let {fname} = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE);")
        elif kind == "string_sanitize_cc":
            out.append(f"    let {fname} = unpacker.get_string(crate::packer::SanitizeMode::SANITIZE_CC);")
        elif kind == "string_sanitize_cc_ws":
            out.append(
                "    let "
                + fname
                + " = unpacker.get_string(crate::packer::SanitizeMode { sanitize: false, sanitize_cc: true, skip_start_whitespace: true });"
            )
    out.append("    if unpacker.error() {")
    out.append("        return None;")
    out.append("    }")
    field_names = ", ".join(rust_field_name(v.name) for _, v in fields)
    out.append(f"    Some({name} {{ {field_names} }})")
    out.append("}")
    out.append("")

    # --- encode. `Cl_Say`'s encoder is `pub(crate)` only (D-007: no public send path for chat) —
    # every other message's encoder is `pub`.
    visibility = "pub(crate)" if msg.name == "Cl_Say" else "pub"
    out.append(f"/// Encodes a `{name}` payload (message body only, no leading msg-id — see")
    out.append("/// `crate::message`).")
    if msg.name == "Cl_Say":
        out.append(
            "#[allow(dead_code)] // D-007: exists for completeness/tests only — never called by"
        )
        out.append("// non-test code; there is no public send path for chat.")
    msg_param = "msg" if fields else "_msg"
    packer_param = "packer" if fields else "_packer"
    out.append(
        f"{visibility} fn encode_{snake}({msg_param}: &{name}, {packer_param}: &mut crate::packer::Packer) {{"
    )
    for kind, v in fields:
        fname = rust_field_name(v.name)
        if kind in ("int", "int_range"):
            out.append(f"    packer.add_int(msg.{fname});")
        else:
            out.append(f"    packer.add_string(&msg.{fname}, 0, true);")
    out.append("}")
    out.append("")
    return "\n".join(out)


def emit_message_roundtrip_test(msg) -> str:
    """One `encode_*` -> `decode_*` round-trip test per message, with a sample value for every
    field that is always in range (the field's own lower bound if ranged, else 0/"abc") — proves
    every generated message (including `Cl_Say`, whose `pub(crate)`-only encoder would otherwise
    be unused dead code) actually round-trips end to end, and is itself a regression test against
    a future generator change breaking the field order or a validation bound."""
    name = rust_type_name(msg.name)
    snake = rust_snake_name(msg.name)
    fields = message_fields(msg.variables)
    args = []
    for kind, v in fields:
        fname = rust_field_name(v.name)
        if kind.startswith("string"):
            args.append(f'{fname}: "abc".to_string()')
        elif kind == "int_range":
            args.append(f"{fname}: {rust_lit(resolve_expr(v.min))}")
        else:
            args.append(f"{fname}: 0")
    out = []
    out.append(f"    #[test]")
    out.append(f"    fn roundtrip_{snake}() {{")
    out.append(f"        let msg = {name} {{ {', '.join(args)} }};")
    out.append("        let mut buf = [0u8; 4096];")
    out.append("        let mut packer = crate::packer::Packer::new(&mut buf);")
    out.append(f"        encode_{snake}(&msg, &mut packer);")
    out.append("        assert!(!packer.error());")
    out.append("        let mut unpacker = crate::packer::Unpacker::new(packer.data());")
    out.append(f"        assert_eq!(decode_{snake}(&mut unpacker), Some(msg));")
    out.append("    }")
    out.append("")
    return "\n".join(out)


def emit_messages(network) -> str:
    lines = [
        HEADER,
        "//! Game messages (`NETMSGTYPE_*`) from `datasrc/network.py`: `Sv_*` (server->client) and",
        "//! `Cl_*` (client->server).",
        "//!",
        "//! D-007 / task constraint: the bot must never send chat. `encode_cl_say` exists (for",
        "//! completeness and its own round-trip test) but is `pub(crate)`, not `pub` — there is no",
        "//! public send path for `Cl_Say` anywhere in this crate's API.",
        "",
    ]
    non_extended = [m for m in network.Messages if m.ex is None]
    extended = [m for m in network.Messages if m.ex is not None]
    for msg in non_extended + extended:
        lines.append(emit_message_struct(msg))

    lines.append("/// Non-UUID message ids, in `datasrc/network.py` declaration order starting at 1")
    lines.append("/// (`datasrc/compile.py`'s `create_enum_table([\"NETMSGTYPE_EX\", ...])`) — part of")
    lines.append("/// the wire format, must track DDNet exactly.")
    lines.append("pub mod id {")
    for i, msg in enumerate(non_extended, start=1):
        lines.append(f"    pub const {msg.enum_name}: i32 = {i};")
    lines.append("}")
    lines.append("")

    lines.append("/// UUID names for every `ex` game message, in `datasrc/network.py` declaration")
    lines.append("/// order (see `crate::uuid::UuidRegistry::from_names`).")
    lines.append("pub const EX_NAMES: &[&str] = &[")
    for msg in extended:
        lines.append(f'    "{msg.ex}",')
    lines.append("];")
    lines.append("")

    lines.append("/// Every non-UUID game message, decoded.")
    lines.append("#[derive(Debug, Clone, PartialEq, Eq)]")
    lines.append(
        "#[allow(clippy::large_enum_variant)] // mechanical: field counts vary a lot message to"
    )
    lines.append(
        "// message (e.g. `Sv_VoteOptionListAdd`'s 15 strings); this is a short-lived decode"
    )
    lines.append("// result, not something kept around in bulk, so boxing isn't worth the ergonomics cost.")
    lines.append("pub enum GameMsg {")
    for msg in non_extended:
        lines.append(f"    {rust_type_name(msg.name)}({rust_type_name(msg.name)}),")
    lines.append("}")
    lines.append("")
    lines.append("/// Decodes a non-UUID game message payload by its numbered id (`id::*`).")
    lines.append(
        "pub fn decode_game_msg(msg_id: i32, unpacker: &mut crate::packer::Unpacker) -> Option<GameMsg> {"
    )
    lines.append("    Some(match msg_id {")
    for msg in non_extended:
        rname = rust_type_name(msg.name)
        snake = rust_snake_name(msg.name)
        lines.append(f"        id::{msg.enum_name} => GameMsg::{rname}(decode_{snake}(unpacker)?),")
    lines.append("        _ => return None,")
    lines.append("    })")
    lines.append("}")
    lines.append("")

    lines.append("/// Every UUID (`ex`) game message, decoded.")
    lines.append("#[derive(Debug, Clone, PartialEq, Eq)]")
    lines.append("pub enum ExGameMsg {")
    for msg in extended:
        lines.append(f"    {rust_type_name(msg.name)}({rust_type_name(msg.name)}),")
    lines.append("}")
    lines.append("")
    lines.append("/// Decodes a UUID (`ex`) game message payload by its UUID name.")
    lines.append(
        "pub fn decode_ex_game_msg(name: &str, unpacker: &mut crate::packer::Unpacker) -> Option<ExGameMsg> {"
    )
    lines.append("    Some(match name {")
    for msg in extended:
        rname = rust_type_name(msg.name)
        snake = rust_snake_name(msg.name)
        lines.append(f'        "{msg.ex}" => ExGameMsg::{rname}(decode_{snake}(unpacker)?),')
    lines.append("        _ => return None,")
    lines.append("    })")
    lines.append("}")
    lines.append("")

    lines.append("#[cfg(test)]")
    lines.append("mod generated_roundtrip_tests {")
    lines.append("    use super::*;")
    lines.append("")
    for msg in non_extended + extended:
        lines.append(emit_message_roundtrip_test(msg))
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def emit_mod() -> str:
    return HEADER + """\
//! Rust code generated from DDNet 20.1's own protocol description
//! (`datasrc/network.py` + `datasrc/datatypes.py`) by `tools/ddnet-protocol-gen/generate.py`.
//! See that script and `tools/ddnet-protocol-gen/README.md` for how to regenerate.
//!
//! This module (and everything under it) contains no hand-written protocol logic — only
//! mechanical field lists, ids, and the encode/decode/validate functions `datasrc/compile.py`'s
//! own templates would produce, ported to Rust. Hand-written logic that *uses* this (message
//! dispatch, snapshot delta unpacking, the typed view API, ...) lives in the parent `ddai_net`
//! crate modules, never here.

pub mod enums;
pub mod objects;
pub mod messages;
"""


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ddnet_src", type=Path, help="path to the DDNet source tree root")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: crates/ddai-net/src/generated next to this script)")
    args = parser.parse_args()

    out_dir = args.out or (REPO_ROOT / "crates" / "ddai-net" / "src" / "generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    network = load_datasrc(args.ddnet_src)

    files = {
        "mod.rs": emit_mod(),
        "enums.rs": emit_enums(network),
        "objects.rs": emit_objects(network),
        "messages.rs": emit_messages(network),
    }
    for name, content in files.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path} ({len(content)} bytes)")

    try:
        subprocess.run(["rustfmt", "--version"], capture_output=True, check=True)
        have_rustfmt = True
    except (OSError, subprocess.CalledProcessError):
        have_rustfmt = False
    if have_rustfmt:
        # F5 (review round 1): rustfmt only *discovers* `rustfmt.toml` by walking up from the
        # input file's own directory — with `--out` pointing outside the repository (e.g. a temp
        # dir, exactly what a byte-for-byte regeneration check needs to do to compare against the
        # committed files without overwriting them first) it would never find this repo's
        # `rustfmt.toml` (`max_width = 120`) and silently fall back to rustfmt's own defaults
        # (`max_width = 100`), reformatting ~40 struct-literal signatures differently — passing
        # `--config-path` explicitly makes the output identical regardless of `--out`.
        rustfmt_toml = REPO_ROOT / "rustfmt.toml"
        extra_args = ["--config-path", str(rustfmt_toml)] if rustfmt_toml.is_file() else []
        for name in files:
            subprocess.run(["rustfmt", "--edition", "2024", *extra_args, str(out_dir / name)], check=True)
        print("formatted with rustfmt")
    else:
        print("warning: rustfmt not found on PATH — output is valid Rust but not rustfmt'd", file=sys.stderr)


if __name__ == "__main__":
    main()
