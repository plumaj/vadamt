#!/usr/bin/env python3
# compute_exclusivity.py
# Post-process variant family statistics (from summary.csv) to compute Gini or entropy.

import pandas as pd
import numpy as np
import argparse

def gini(x):
    x = np.sort(np.array(x))
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return np.nan
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def entropy(x):
    x = np.array(x, dtype=float)
    if x.sum() == 0:
        return np.nan
    p = x / x.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def parse_counts(s):
    if isinstance(s, str):
        return [int(x) for x in s.split(",") if x.strip().isdigit()]
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(map(int, s))
    return []

def main():
    parser = argparse.ArgumentParser(description="Compute exclusivity metrics (Gini / Entropy) per variant family.")
    parser.add_argument("--input", required=True, help="Path to summary CSV file (with family_id).")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--metric", choices=["gini", "entropy"], default="gini")
    parser.add_argument("--families", nargs="*", type=int, help="Optional family IDs to include.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    metric_func = gini if args.metric == "gini" else entropy

    rows = []
    for _, row in df.iterrows():
        fid = row.get("family_id")
        if args.families and fid not in args.families:
            continue
        counts = parse_counts(row.get("dim_counts", ""))
        if not counts:
            continue
        val = metric_func(counts)
        rows.append({"family_id": fid, args.metric: val})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
