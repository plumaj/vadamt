#!/usr/bin/env python3
# Analyse variant families using a unified config.
# Jaccard is computed over the *configured dimension* (e.g., user_id, region, date bucket).
# If no dimension is set in config, Jaccard is marked N/A and set to 1.0.

import argparse, json, os, time
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import FastText
from utils import tokenize, setup_logging
import csv

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads("".join(line for line in f if not line.strip().startswith("//")))

def jaccard(a, b):
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def connected_components(edges, vocab, token_counts):
    seen, families = set(), []
    for w in vocab:
        if w in seen: continue
        stack = [w]; comp = []
        seen.add(w)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in edges.get(x, ()):
                if y not in seen:
                    seen.add(y); stack.append(y)
        if len(comp) >= 2:
            families.append(sorted(comp, key=lambda t: (-token_counts[t], t)))
    families.sort(key=len, reverse=True)
    if families: families = families[1:]  # drop largest
    return families

def family_overlap_allpairs(comp, token_dimension_sets):
    """
    Compute pairwise Jaccard across *all* token pairs in a family.
    Returns:
      family_mean : float  (mean Jaccard over all pairs)
      per_token_mean : dict[token -> mean Jaccard to others in family]
    """
    members = list(comp)
    n = len(members)
    if n <= 1:
        return 1.0, {t: 1.0 for t in members}

    # sum of pairwise J for whole family, and per-token sums
    total_sum = 0.0
    count_pairs = 0
    per_token_sum = {t: 0.0 for t in members}

    for i in range(n):
        Ai = token_dimension_sets[members[i]]
        for j in range(i + 1, n):
            Aj = token_dimension_sets[members[j]]
            val = jaccard(Ai, Aj)
            total_sum += val
            count_pairs += 1
            per_token_sum[members[i]] += val
            per_token_sum[members[j]] += val

    family_mean = total_sum / count_pairs if count_pairs else 1.0
    per_token_mean = {t: (per_token_sum[t] / (n - 1)) for t in members}
    return family_mean, per_token_mean

# ----- OPEN (your first block) -----
def build_families_open(model, token_counts, token_dimension_sets, cfg, dimension_set: bool):
    TH = cfg.get("open_TH", 0.75)
    TOPN = cfg.get("open_TOPN", 30)
    vocab = [w for w, c in token_counts.items() if c >= cfg.get("min_count", 10) and w in model.wv]
    vec = {w: model.wv[w] for w in vocab}

    from collections import defaultdict as _dd
    edges = _dd(set)
    for w in tqdm(vocab, desc="Linking (open)"):
        for nb, score in model.wv.most_similar(w, topn=TOPN):
            if nb in vec and score >= TH:
                edges[w].add(nb); edges[nb].add(w)

    families = connected_components(edges, vocab, token_counts)

    scored = []
    for comp in families:
        total = sum(token_counts[t] for t in comp)
        if dimension_set:
            fam_j_mean, _ = family_overlap_allpairs(comp, token_dimension_sets)
        else:
            fam_j_mean = 1.0  # N/A when no dimension
        scored.append((total, fam_j_mean, comp))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored, families, TH

# ----- STRICT (your second block) -----
def build_families_strict(model, token_counts, token_dimension_sets, cfg, dimension_set: bool):
    TOPN = cfg.get("strict_TOPN", 100)
    TH = cfg.get("strict_TH", 0.73)
    SNN_MIN = cfg.get("SNN_MIN", 2)
    DEGREE_CAP = cfg.get("DEGREE_CAP", 200)
    MIN_LEN = cfg.get("MIN_LEN", 3)
    MIN_USERS = cfg.get("MIN_USERS", 3)  # interpreted as MIN_dimension_MEMBERS
    MAX_FREQ_RATIO = cfg.get("MAX_FREQ_RATIO", 25)

    stopwords = set()
    spath = cfg.get("stopwords_path", "")
    if spath:
        try:
            with open(spath, "r", encoding="utf-8") as f:
                stopwords = set(w.strip().lower() for w in f)
        except FileNotFoundError:
            stopwords = set()

    vocab = [
        w for w, c in token_counts.items()
        if c >= cfg.get("min_count", 10) and w in model.wv and len(w) >= MIN_LEN
        and w not in stopwords and (not dimension_set or len(token_dimension_sets[w]) >= MIN_USERS)
    ]

    toplists = {w: dict(model.wv.most_similar(w, topn=TOPN)) for w in tqdm(vocab, desc="Building toplists")}
    from collections import defaultdict as _dd
    edges = _dd(set)

    for w, nbs in tqdm(toplists.items(), desc="Linking (strict)"):
        for nb, score in nbs.items():
            if score < TH or nb not in toplists: continue
            if w not in toplists[nb]: continue                 # mutual NN
            if len(set(nbs).intersection(toplists[nb])) < SNN_MIN: continue  # shared NNs
            a, b = token_counts[w], token_counts[nb]
            if max(a, b) / max(1, min(a, b)) > MAX_FREQ_RATIO: continue      # freq ratio
            if len(edges[w]) >= DEGREE_CAP or len(edges[nb]) >= DEGREE_CAP: continue
            edges[w].add(nb); edges[nb].add(w)

    families = connected_components(edges, vocab, token_counts)

    scored = []
    for comp in families:
        total = sum(token_counts[t] for t in comp)
        if dimension_set:
            fam_j_mean, _ = family_overlap_allpairs(comp, token_dimension_sets)
        else:
            fam_j_mean = 1.0
        scored.append((total, fam_j_mean, comp))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored, families, TH

def main():
    parser = argparse.ArgumentParser(description="Analyze variant families (dimension-agnostic Jaccard).")
    parser.add_argument("--input", required=True, help="Path to JSON (array or JSONL; pandas can infer).")
    parser.add_argument("--text_field", default="text", help="Field containing text. Default: text")
    parser.add_argument("--model", required=True, help="Path to FastText .model")
    parser.add_argument("--method", choices=["open", "strict"], default="open", help="Family construction.")
    parser.add_argument("--config", default="config/settings.json", help="Path to unified config (settings.jsonc).")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    import logging
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)
    lowercase = cfg.get("lowercase", True)
    dimension = cfg.get("dimension") or ""    # generic dimension (user_id, region, month, …)
    dimension_set = bool(dimension)

    logging.info("Loading model: %s", args.model)
    model = FastText.load(args.model)

    logging.info("Reading input via pandas: %s", args.input)
    try:
        df = pd.read_json(args.input, encoding="utf-8", lines=False)
    except ValueError:
        df = pd.read_json(args.input, encoding="utf-8", lines=True)

    if args.text_field not in df.columns:
        raise ValueError(f"Input must contain column '{args.text_field}'")

    token_counts = Counter()
    token_dimension_sets = defaultdict(set)  # <-- generic dimension sets

    if dimension_set:
        if dimension not in df.columns:
            raise ValueError(f"dimension '{dimension}' not in input columns.")
        logging.info("Rebuilding token stats aggregated by dimension='%s'.", dimension)
        for axval, texts in df.groupby(dimension)[args.text_field]:
            joined = " ".join(map(str, texts))
            toks = tokenize(joined, lowercase=lowercase)
            token_counts.update(toks)
            for t in set(toks):
                token_dimension_sets[t].add(axval)
    else:
        logging.info("No dimension in config; building token stats from individual records.")
        for raw in df[args.text_field].astype(str):
            toks = tokenize(raw, lowercase=lowercase)
            token_counts.update(toks)
        # token_dimension_sets stays empty; Jaccard marked N/A.

    # Build families
    if args.method == "open":
        scored, families, TH = build_families_open(model, token_counts, token_dimension_sets, cfg, dimension_set)
    else:
        scored, families, TH = build_families_strict(model, token_counts, token_dimension_sets, cfg, dimension_set)

    method_tag = args.method
    variant_file = os.path.join(args.output_dir, f"variant_families_{method_tag}.txt")
    summary_file = os.path.join(args.output_dir, f"summary_{method_tag}.txt")

    # Write ALL families (no truncation)
    with open(variant_file, "w", encoding="utf-8") as f:
        if dimension_set:
            f.write(f"# Jaccard is computed as MEAN pairwise *dimension* overlap across all token pairs in each family (dimension='{dimension}').\n")
            f.write("# Each token line shows its MEAN overlap with all other tokens in the family.\n\n")
        else:
            f.write("# No dimension specified; Jaccard over dimension overlap is not applicable. Values set to 1.0 (N/A).\n\n")

        f.write(f"Discovered {len(families)} variant families (τ={TH}).\n\n")

        for total, fam_j_mean, comp in scored:
            f.write(f"=== Family (total {total:,}) | Jaccard(mean pairwise dimension overlap)={fam_j_mean:.3f} | size={len(comp)} ===\n")

            # compute per-token means for display
            if dimension_set:
                _, per_token_mean = family_overlap_allpairs(comp, token_dimension_sets)
            else:
                per_token_mean = {t: 1.0 for t in comp}

            for t in comp:
                members = len(token_dimension_sets[t]) if dimension_set else 0
                f.write(
                    f"  • {t:<15} freq={token_counts[t]:<6} "
                    f"dimension_members={members:<4} dimension_overlap_mean={per_token_mean[t]:.3f}\n"
                )
            f.write("\n")


    # Summary (also echo to console)
    avg_size = (sum(len(c) for c in families) / max(1, len(families))) if families else 0.0
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Total families: {len(families)}\n")
        f.write(f"Average family size: {avg_size:.2f}\n")

    print(f"\n=== SUMMARY ({method_tag}) ===")
    print(f"Total families: {len(families)}")
    print(f"Average family size: {avg_size:.2f}")
    print(f"Full results: {variant_file}")
    print(f"Summary file: {summary_file}\n")

    summary_csv = os.path.join(args.output_dir, f"summary_{method_tag}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["family_id", "size", "total_freq", "dimension_coverage", "variants"])
        for fid, (total, fam_j_mean, comp) in enumerate(scored, start=1):
            if dimension_set:
                coverage = len(set().union(*(token_dimension_sets[t] for t in comp)))
            else:
                coverage = 0
            writer.writerow([fid, len(comp), total, coverage, ",".join(comp)])

    print(f"Summary CSV file: {summary_csv}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
