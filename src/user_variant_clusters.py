#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from math import log2
import json


# -----------------------------
# Loading
# -----------------------------

def load_json_robust(path):
    try:
        return pd.read_json(path, lines=False)
    except ValueError:
        return pd.read_json(path, lines=True)


def load_variant_families(summary_csv):
    df = pd.read_csv(summary_csv)
    families = {}
    for _, row in df.iterrows():
        fam_id = int(row["family_id"])
        tokens = row["variants"].split(",")
        families[fam_id] = set(tokens)
    return families


# -----------------------------
# Tokenization (simple fallback)
# -----------------------------

def simple_tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()


# -----------------------------
# Build User × Family Matrix
# -----------------------------

def build_user_family_matrix(df, families, text_field, user_field):
    user_families = defaultdict(set)

    for user, texts in df.groupby(user_field)[text_field]:
        joined = " ".join(map(str, texts))
        toks = set(simple_tokenize(joined))

        for fam_id, fam_tokens in families.items():
            if toks & fam_tokens:
                user_families[user].add(fam_id)

    users = list(user_families.keys())
    fam_ids = sorted(families.keys())

    matrix = np.zeros((len(users), len(fam_ids)), dtype=np.uint8)

    fam_index = {f: i for i, f in enumerate(fam_ids)}

    for row_i, user in enumerate(users):
        for fam in user_families[user]:
            col_j = fam_index[fam]
            matrix[row_i, col_j] = 1

    return users, fam_ids, matrix


# -----------------------------
# Entropy Selection
# -----------------------------

def family_entropy(col):
    p = col.mean()
    if p == 0 or p == 1:
        return 0.0
    return -(p * log2(p) + (1 - p) * log2(1 - p))


def select_top_families(matrix, fam_ids, top_k=50):
    entropies = np.array([family_entropy(matrix[:, i]) for i in range(matrix.shape[1])])
    top_idx = np.argsort(-entropies)[:top_k]
    return matrix[:, top_idx], [fam_ids[i] for i in top_idx]


# -----------------------------
# Fast Combination Search
# -----------------------------

def find_informative_combinations(matrix, fam_ids, min_users=100, max_comb_size=2):
    results = []
    n_users = matrix.shape[0]

    # Single families
    col_sums = matrix.sum(axis=0)
    for i, count in enumerate(col_sums):
        if count >= min_users:
            results.append({
                "families": [fam_ids[i]],
                "user_count": int(count)
            })

    # Pairwise combinations
    if max_comb_size >= 2:
        for i, j in combinations(range(matrix.shape[1]), 2):
            both = np.sum(matrix[:, i] & matrix[:, j])
            if both >= min_users:
                results.append({
                    "families": [fam_ids[i], fam_ids[j]],
                    "user_count": int(both)
                })

    results.sort(key=lambda x: -x["user_count"])
    return results


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--user_field", default="user_id")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--min_users", type=int, default=100)
    parser.add_argument("--max_comb_size", type=int, default=4)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    print("Loading JSON...")
    df = load_json_robust(args.input)

    print("Loading variant families...")
    families = load_variant_families(args.summary_csv)

    print("Building user-family matrix...")
    users, fam_ids, matrix = build_user_family_matrix(
        df,
        families,
        args.text_field,
        args.user_field
    )

    print("Initial families:", len(fam_ids))

    matrix, fam_ids = select_top_families(matrix, fam_ids, top_k=args.top_k)

    print("Families after entropy selection:", len(fam_ids))

    print("Searching for informative combinations...")
    combos = find_informative_combinations(
        matrix,
        fam_ids,
        min_users=args.min_users,
        max_comb_size=args.max_comb_size
    )

    print("Combinations found:", len(combos))

    print("Writing output...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combos, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
