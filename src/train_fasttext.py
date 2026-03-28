#!/usr/bin/env python3
# Train FastText with a single unified config.
# If config["dimension"] is set, aggregate texts by that dimension.
# Otherwise, train on individual records.

import argparse
import json
import os
import time
from collections import defaultdict, Counter
from typing import List

import pandas as pd
from gensim.models import FastText
from utils import tokenize, setup_logging

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads("".join(line for line in f if not line.strip().startswith("//")))

def main():
    parser = argparse.ArgumentParser(description="Train FastText (optionally aggregated by dimension) using a unified config.")
    parser.add_argument("--input", required=True, help="Path to JSON file (array or JSONL; pandas can infer).")
    parser.add_argument("--text_field", default="text", help="Field containing text. Default: text")
    parser.add_argument("--config", default="config/settings.json", help="Path to unified config (settings.jsonc).")
    parser.add_argument("--output", required=True, help="Where to save the .model")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    import logging

    t0 = time.time()
    logging.info("Reading input via pandas: %s", args.input)
    # pandas can read JSON arrays; for JSONL set lines=True
    try:
        df = pd.read_json(args.input, encoding="utf-8", lines=False)
    except ValueError:
        df = pd.read_json(args.input, encoding="utf-8", lines=True)

    cfg = load_config(args.config)
    lowercase = cfg.get("lowercase", True)
    dimension = cfg.get("dimension") or ""  # treat None as empty

    if args.text_field not in df.columns:
        raise ValueError(f"Input must contain column '{args.text_field}'")

    corpus: List[List[str]] = []
    token_counts = Counter()          # side-effect sanity stats
    token_user_sets = defaultdict(set)

    if dimension:
        if dimension not in df.columns:
            raise ValueError(f"dimension '{dimension}' not in input columns.")
        logging.info("Aggregating by dimension='%s' and tokenizing (lowercase=%s)...", dimension, lowercase)
        for uid, texts in df.groupby(dimension)[args.text_field]:
            joined = " ".join(map(str, texts))
            toks = tokenize(joined, lowercase=lowercase)
            if not toks:
                continue
            corpus.append(toks)
            token_counts.update(toks)
            for t in set(toks):
                token_user_sets[t].add(uid)
        logging.info("Prepared %d dimension-documents for training.", len(corpus))
    else:
        logging.info("No dimension specified. Training on individual records (lowercase=%s)...", lowercase)
        for raw in df[args.text_field].astype(str):
            toks = tokenize(raw, lowercase=lowercase)
            if not toks:
                continue
            corpus.append(toks)
            token_counts.update(toks)
        logging.info("Prepared %d documents for training.", len(corpus))

    if not corpus:
        raise RuntimeError("Empty corpus after preparation. Check input and config.")

    logging.info("Initializing FastText with config...")
    model = FastText(
        sentences=corpus,
        vector_size=cfg.get("vector_size", 100),
        window=cfg.get("window", 5),
        min_count=cfg.get("min_count", 10),
        sg=cfg.get("sg", 1),
        workers=cfg.get("workers", 12),
        min_n=cfg.get("min_n", 3),
        max_n=cfg.get("max_n", 7),
        epochs=cfg.get("epochs", 10),
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    logging.info("Saving model to %s", args.output)
    model.save(args.output)
    logging.info("Done in %.2fs", time.time() - t0)

if __name__ == "__main__":
    main()
