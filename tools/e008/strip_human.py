#!/usr/bin/env python3
"""Writes a copy of an experiment config without its [human] section (offline evaluation on the teacher sets only:
the human corpus is gigabytes and not needed for the hook-by-state tables).
  tools/e008/strip_human.py <in.toml> <out.toml>
"""
import re, sys
s = open(sys.argv[1]).read()
i = s.find('\n[human]')
if i >= 0:
    rest = s[i + 1:]
    m = re.search(r'\n\[(?!human)', rest)
    s = s[:i + 1] + (rest[m.start() + 1:] if m else '')
open(sys.argv[2], 'w').write(s)
