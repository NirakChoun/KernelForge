#!/usr/bin/env python3
"""Prints a Markdown table from a results CSV, so docs quote CSV values verbatim.

usage: .venv/bin/python scripts/md_table.py <csv> <col,col,...> [--where "pandas query"] [--sort col,...]
"""
import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("cols")
p.add_argument("--where")
p.add_argument("--sort")
a = p.parse_args()
df = pd.read_csv(a.csv, dtype=str, keep_default_na=False)
if a.where:
    num = pd.read_csv(a.csv)
    df = df[num.eval(a.where).values]
if a.sort:
    num = pd.read_csv(a.csv).loc[df.index]
    df = df.loc[num.sort_values(a.sort.split(",")).index]
cols = a.cols.split(",")
print("| " + " | ".join(cols) + " |")
print("|" + "|".join("---" for _ in cols) + "|")
for _, r in df[cols].iterrows():
    print("| " + " | ".join(r.values) + " |")
