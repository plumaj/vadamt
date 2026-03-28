#!/usr/bin/env python3
"""
utils.py
Shared utilities for variation-ngram CLI tools.
"""
from __future__ import annotations

import re
import json
import sys
from typing import Dict, Iterable, Iterator, List, Tuple, Any, DefaultDict, Set
from collections import defaultdict, Counter
import logging

# ---------------- Logging ----------------
def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a simple, readable format.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

# ---------------- Data loading ----------------
def load_json_records(path: str) -> List[dict]:
    """
    Load JSON input from either:
    - a JSON array: [ {...}, {...}, ... ]
    - or JSON Lines: one JSON object per line.
    Returns a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.strip().startswith("["):
            # JSON array
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be a list of objects.")
            return data
        else:
            # JSON Lines
            recs = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recs.append(json.loads(line))
            return recs

def tokenize(text: str, lowercase: bool = True):
    """
    Tokenization identical to notebook version (Unicode-friendly regex).

    - Matches sequences of letters incl. accented (A–Z, À–ÿ)
    - Keeps apostrophes inside words
    - Lowercases by default

    Parameters
    ----------
    text : str
        Input text
    lowercase : bool
        Whether to lowercase text (default True)
    """
    text = str(text)
    if lowercase:
        text = text.lower()
    # Find letter sequences including accented characters and apostrophes
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text)


def group_by_axis(records: List[dict], axis_field: str) -> Dict[str, List[str]]:
    """
    Group raw texts by an axis (e.g., 'user', 'date').
    Returns a dict: axis_value -> list of raw texts
    """
    groups: DefaultDict[str, List[str]] = defaultdict(list)
    for r in records:
        key = str(r.get(axis_field, "UNKNOWN"))
        groups[key].append(str(r.get("__TEXT__", "")))
    return groups

# ---------------- Similarity helpers ----------------
def char_ngrams(s: str, n_min: int = 3, n_max: int = 5) -> Set[str]:
    """
    Character n-grams for Jaccard. Keeps raw string (no normalization).
    """
    grams: Set[str] = set()
    L = len(s)
    for n in range(n_min, n_max + 1):
        if n <= 0:
            continue
        for i in range(0, max(0, L - n + 1)):
            grams.add(s[i:i+n])
    return grams

def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Jaccard similarity for sets.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ---------------- Union-Find for clustering ----------------
class UnionFind:
    def __init__(self):
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
