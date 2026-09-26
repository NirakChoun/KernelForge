#!/usr/bin/env python3
"""Counts SASS opcodes per kernel from `cuobjdump -sass <binary> | c++filt` on stdin.

Writes CSV to stdout: function,opcode,count. Opcodes keep their modifiers
(LDG.E.128 vs LDG.E) because width and cache hints matter for the analysis.
Counts are static instruction counts, not executed counts.
"""
import collections
import csv
import re
import sys

func_re = re.compile(r"^\s*Function : (.+?)\s*$")
inst_re = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")

counts = collections.OrderedDict()
func = None
for line in sys.stdin:
    m = func_re.match(line)
    if m:
        func = m.group(1)
        counts.setdefault(func, collections.Counter())
        continue
    if func is None:
        continue
    m = inst_re.search(line)
    if m:
        counts[func][m.group(1)] += 1

w = csv.writer(sys.stdout)
w.writerow(["function", "opcode", "count"])
for f, c in counts.items():
    for op, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        w.writerow([f, op, n])
