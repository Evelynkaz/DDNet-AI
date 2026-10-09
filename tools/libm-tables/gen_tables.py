#!/usr/bin/env python3
"""Generate crates/ddai-libm/src/tables.rs from the glibc 2.39 sources (task 5.5a).

Usage: gen_tables.py <glibc-2.39 source root> <output .rs>

Every table value is emitted as its exact IEEE-754 bit pattern (`u64`), so no float literal is ever
re-parsed or rounded; `tables.rs` turns them into `f64` with `f64::from_bits` in const context.
Expressions in the C sources that scale a hex-float by a power of two (`* N`, `/ N`, `* -2`, ...) are
evaluated with Python floats, which is exact for powers of two.
"""
import re
import struct
import sys

HEXF = r'[+-]?0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*[pP][+-]?\d+'


def bits(x: float) -> int:
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def region(text, start_marker, end_marker, start_from=0):
    """Text between the first `start_marker` after `start_from` and the next `end_marker`."""
    a = text.index(start_marker, start_from)
    b = text.index(end_marker, a + len(start_marker))
    return text[a + len(start_marker):b]


def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'//[^\n]*', '', s)
    return s


def eval_item(item, consts):
    """Evaluate one C initialiser item: hex floats combined with * and / and small integer factors."""
    item = item.strip()
    expr = re.sub(HEXF, lambda m: repr(float.fromhex(m.group(0))), item)
    for name, val in consts.items():
        expr = re.sub(r'\b' + re.escape(name) + r'\b', repr(float(val)), expr)
    assert re.fullmatch(r'[0-9eE+\-*/(). ]+', expr), expr
    return eval(expr)


def hexfloats(text, consts=None):
    """All comma-separated floating values of a C initialiser block, in order."""
    consts = consts or {}
    text = strip_comments(text)
    # drop preprocessor lines
    text = '\n'.join(l for l in text.splitlines() if not l.strip().startswith('#'))
    out = []
    for item in re.split(r'[,{}]', text):
        if item.strip():
            out.append(eval_item(item, consts))
    return out


def emit_u64s(name, values, per_line=4, doc=None):
    lines = []
    if doc:
        lines.append(f'/// {doc}')
    lines.append(f'pub(crate) const {name}: [u64; {len(values)}] = [')
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append('    ' + ' '.join(f'0x{v:016x},' for v in chunk))
    lines.append('];')
    return '\n'.join(lines)


def emit_f64s(name, values, doc=None):
    return emit_u64s(name, [bits(v) for v in values], doc=doc)


def main():
    root, out = sys.argv[1], sys.argv[2]
    d64 = root + '/sysdeps/ieee754/dbl-64/'
    f32 = root + '/sysdeps/ieee754/flt-32/'
    parts = []

    # ---- sinf / cosf (s_sincosf_data.c, TOINT_INTRINSICS == 0 branch) ----
    t = read(f32 + 's_sincosf_data.c')
    body = region(t, 'const sincos_t __sincosf_table[2] =', '/* Table with 4/PI')
    ents = body.split('#else')  # first '#else' per entry selects the !TOINT_INTRINSICS hpi_inv value
    # Robust approach: take, per table entry, the values after '#endif' and the one after '#else'.
    entries = re.findall(r'\{\s*\{[^}]*\},\s*#if TOINT_INTRINSICS\s*(' + HEXF + r'),\s*#else\s*(' + HEXF +
                         r')\s*,\s*#endif\s*(.*?)\n  \}', body, flags=re.S)
    assert len(entries) == 2, len(entries)
    sc = []
    for _, hpi_inv, rest in entries:
        vals = [float.fromhex(h) for h in re.findall(HEXF, rest)]
        assert len(vals) == 9, vals  # hpi, c0..c4, s1..s3
        sc.append([float.fromhex(hpi_inv)] + vals)
    # layout per entry: hpi_inv, hpi, c0, c1, c2, c3, c4, s1, s2, s3
    flat = [v for e in sc for v in e]
    parts.append(emit_f64s('SINCOSF_TABLE', flat,
                           doc='`__sincosf_table[2]`, 10 doubles per entry: hpi_inv (scaled by 2^24), hpi, c0..c4, s1..s3.'))
    inv = re.findall(r'0x[0-9a-f]+', region(t, 'const uint32_t __inv_pio4[24] =', '};'))
    inv = [int(x, 16) for x in inv]
    assert len(inv) == 24
    parts.append('/// `__inv_pio4`: 4/PI to 192 bits, 8 new bits per entry.\n'
                 'pub(crate) const INV_PIO4: [u32; 24] = [\n' +
                 '\n'.join('    ' + ' '.join(f'0x{v:08x},' for v in inv[i:i + 4]) for i in range(0, 24, 4)) + '\n];')

    # ---- exp2f / powf data (TOINT_INTRINSICS == 0) ----
    t = read(f32 + 'e_exp2f_data.c')
    tab = [int(x, 16) for x in re.findall(r'0x[0-9a-f]{16}', region(t, '.tab = {', '},'))]
    assert len(tab) == 32
    parts.append(emit_u64s('EXP2F_TAB', tab, doc='`__exp2f_data.tab`: uint(2^(i/32)) - (i << 47).'))
    N = 32
    shift_scaled = float.fromhex('0x1.8p+52') / N
    poly = hexfloats(region(t, '.poly = {', '},'))
    assert len(poly) == 3
    parts.append(emit_f64s('EXP2F_POLY', poly, doc='`__exp2f_data.poly`.'))
    parts.append(f'/// `__exp2f_data.shift_scaled` = 0x1.8p+52 / 32.\npub(crate) const EXP2F_SHIFT_SCALED: u64 = 0x{bits(shift_scaled):016x};')

    t = read(f32 + 'e_powf_log2_data.c')
    tabtxt = strip_comments(region(t, '.tab = {', '},\n  .poly'))
    tabv = hexfloats(tabtxt.replace('POWF_SCALE', '1.0'))
    assert len(tabv) == 32, len(tabv)
    parts.append(emit_f64s('POWF_LOG2_TAB', tabv, doc='`__powf_log2_data.tab`: (invc, logc) x 16 (POWF_SCALE == 1).'))
    pv = hexfloats(region(t, '.poly = {', '}\n};').replace('POWF_SCALE', '1.0'))
    assert len(pv) == 5, len(pv)
    parts.append(emit_f64s('POWF_LOG2_POLY', pv, doc='`__powf_log2_data.poly`.'))

    # ---- log (double) ----
    t = read(d64 + 'e_log_data.c')
    ln2hi = float.fromhex('0x1.62e42fefa3800p-1')
    ln2lo = float.fromhex('0x1.ef35793c76730p-45')
    assert '.ln2hi = 0x1.62e42fefa3800p-1' in t and '.ln2lo = 0x1.ef35793c76730p-45' in t
    poly1 = hexfloats(region(t, '.poly1 = {', '},'))
    assert len(poly1) == 11, len(poly1)
    poly = hexfloats(region(t, '.poly = {', '},'))
    assert len(poly) == 5, len(poly)
    tabtxt = region(t, '.tab = {', '},\n#ifndef __FP_FAST_FMA')
    tabv = hexfloats(tabtxt)
    assert len(tabv) == 256, len(tabv)
    parts.append(f'/// `__log_data.ln2hi`.\npub(crate) const LOG_LN2HI: u64 = 0x{bits(ln2hi):016x};')
    parts.append(f'/// `__log_data.ln2lo`.\npub(crate) const LOG_LN2LO: u64 = 0x{bits(ln2lo):016x};')
    parts.append(emit_f64s('LOG_POLY1', poly1, doc='`__log_data.poly1` (B[0..11]).'))
    parts.append(emit_f64s('LOG_POLY', poly, doc='`__log_data.poly` (A[0..5]).'))
    parts.append(emit_f64s('LOG_TAB', tabv, doc='`__log_data.tab`: (invc, logc) x 128.'))

    # ---- exp data (for pow) ----
    t = read(d64 + 'e_exp_data.c')
    N = 128
    invln2N = float.fromhex('0x1.71547652b82fep0') * N
    negln2hiN = float.fromhex('-0x1.62e42fefa0000p-8')
    negln2loN = float.fromhex('-0x1.cf79abc9e3b3ap-47')
    assert '.negln2hiN = -0x1.62e42fefa0000p-8' in t and '.negln2loN = -0x1.cf79abc9e3b3ap-47' in t
    expoly = hexfloats(region(t, '.poly = {', '},'))
    assert len(expoly) == 4, len(expoly)
    tabv = [int(x, 16) for x in re.findall(r'0x[0-9a-f]+', strip_comments(region(t, '.tab = {', '},\n};')))]
    tabv = [x for x in tabv]
    assert len(tabv) == 256, len(tabv)
    parts.append(f'/// `__exp_data.invln2N`.\npub(crate) const EXP_INVLN2N: u64 = 0x{bits(invln2N):016x};')
    parts.append(f'/// `__exp_data.negln2hiN`.\npub(crate) const EXP_NEGLN2HIN: u64 = 0x{bits(negln2hiN):016x};')
    parts.append(f'/// `__exp_data.negln2loN`.\npub(crate) const EXP_NEGLN2LON: u64 = 0x{bits(negln2loN):016x};')
    parts.append(f'/// `__exp_data.shift` = 0x1.8p52.\npub(crate) const EXP_SHIFT: u64 = 0x{bits(float.fromhex("0x1.8p52")):016x};')
    parts.append(emit_f64s('EXP_POLY', expoly, doc='`__exp_data.poly` (C2..C5).'))
    parts.append(emit_u64s('EXP_TAB', tabv, doc='`__exp_data.tab`: (tail bits, scale bits - (k << 45)) x 128.'))

    # ---- pow log data ----
    t = read(d64 + 'e_pow_log_data.c')
    plpoly = hexfloats(region(t, '.poly = {', '},'))
    assert len(plpoly) == 7, len(plpoly)
    tabtxt = region(t, '.tab = {', '#endif\n},')
    rows = re.findall(r'A\(\s*(' + HEXF + r')\s*,\s*(' + HEXF + r')\s*,\s*(' + HEXF + r')\s*\)', tabtxt)
    assert len(rows) == 128, len(rows)
    flat = []
    for a, b, c in rows:
        flat += [float.fromhex(a), float.fromhex(b), float.fromhex(c)]
    parts.append(emit_f64s('POW_LOG_POLY', plpoly, doc='`__pow_log_data.poly` (A[0..7], pre-scaled).'))
    parts.append(emit_f64s('POW_LOG_TAB', flat, doc='`__pow_log_data.tab`: (invc, logc, logctail) x 128 (the unused pad is dropped).'))

    # ---- atan2 (IBM) ----
    t = read(d64 + 'uatan.tbl')
    be = region(t, 'cij[241][7] = {', '#else')
    words = re.findall(r'0[xX]([0-9A-Fa-f]{8}),\s*0[xX]([0-9A-Fa-f]{8})', be)
    assert len(words) == 241 * 7, len(words)
    cij = [(int(h, 16) << 32) | int(l, 16) for h, l in words]
    parts.append(emit_u64s('ATAN2_CIJ', cij, per_line=7, doc='`cij[241][7]` of uatan.tbl, row-major.'))
    t = read(d64 + 'atnat2.h')
    be = t[:t.index('#else')]
    consts = {}
    for name, hi, lo in re.findall(r'/\*\*/\s*(\w+)\s*=\s*\{\{0x([0-9a-f]{8}),\s*0x([0-9a-f]{8})\}', be):
        consts[name] = (int(hi, 16) << 32) | int(lo, 16)
    # simple single-value ones only (ud[] etc. are not needed)
    for nm in ['d3', 'd5', 'd7', 'd9', 'd11', 'd13', 'inv16', 'opi', 'opi1', 'mopi', 'hpi', 'hpi1', 'mhpi', 'qpi',
               'mqpi', 'tqpi', 'mtqpi', 'two500', 'twom500']:
        assert nm in consts, nm
        parts.append(f'pub(crate) const ATAN2_{nm.upper()}: u64 = 0x{consts[nm]:016x};')

    header = ('// @generated by tools/libm-tables/gen_tables.py from the glibc 2.39 sources; do not edit by hand.\n'
              '//\n'
              '// Data tables of the glibc math functions ported in this crate (see the header of each function\'s\n'
              '// module for its licence): sysdeps/ieee754/flt-32/{s_sincosf_data.c,e_exp2f_data.c,e_powf_log2_data.c}\n'
              '// and sysdeps/ieee754/dbl-64/{e_log_data.c,e_exp_data.c,e_pow_log_data.c,uatan.tbl,atnat2.h}.\n'
              '// Values are the exact IEEE-754 bit patterns the C compiler produced.\n\n')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(header + '\n\n'.join(parts) + '\n')


if __name__ == '__main__':
    main()
