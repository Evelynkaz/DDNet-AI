#!/usr/bin/env python3
"""Removes the first occurrence of a name from the queue file and appends it to <queue>.history."""
import sys
q, n = sys.argv[1], sys.argv[2]
lines = open(q).read().split('\n')
for i, l in enumerate(lines):
    if l.strip() == n:
        lines.pop(i)
        break
open(q, 'w').write('\n'.join(lines))
open(q + '.history', 'a').write(n + '\n')
