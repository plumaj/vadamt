# variation-ngram

Command-line tools to train a FastText model (gensim) and analyze lexical variation "families" derived from text corpora.  
The design reproduces the original Jupyter workflow — with optional dimension-based aggregation and full pairwise overlap analysis.

---

## Overview

The pipeline consists of two stages:

1. **Training** – builds a subword FastText model on a corpus (optionally aggregated by an *dimension*, e.g., `user_id` or `region`).  
2. **Analysis** – extracts clusters (“variant families”) of similar tokens and computes their *pairwise overlap* across the same dimension.

Both scripts read all parameters (including tokenizer settings, dimension, and thresholds) from a **single unified config** file:  
`config/settings.jsonc`.

---

## Tokenization

Tokenization uses a Unicode-aware regex identical to the original notebook:

```python
re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())
```

- Keeps accented letters and apostrophes.  
- Removes punctuation.  
- Lowercasing can be toggled in the config (`"lowercase": true/false`).

---

## Configuration

All settings live in **`config/settings.jsonc`**, for example:

```jsonc
{
  "lowercase": true,
  "dimension": "user_id",            // set to "" or null to disable dimension aggregation
  "vector_size": 100,
  "window": 5,
  "min_count": 10,
  "epochs": 10,
  "sg": 1,
  "workers": 12,
  "min_n": 3,
  "max_n": 7,
  "open_TOPN": 30,
  "open_TH": 0.75,
  "strict_TOPN": 100,
  "strict_TH": 0.73,
  "SNN_MIN": 2,
  "DEGREE_CAP": 200,
  "MIN_LEN": 3,
  "MIN_USERS": 3,
  "MAX_FREQ_RATIO": 25,
  "stopwords_path": "stopwords.txt"
}
```

- **`dimension`** — if set, training aggregates all texts per dimension value (e.g., per user).  
  Analysis then measures overlap between dimension values (user overlap, region overlap, etc.).  
  If left empty, training runs on individual records and overlap is marked “N/A”.  
- **`lowercase`** — controls whether text is lowercased before tokenization.  
- The remaining fields control FastText hyperparameters and variant-family thresholds.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -r requirements.txt
```

---

## Training

Train a FastText model on your corpus:

```bash
python src/train_fasttext.py \
  --input data/example.json \
  --text_field text \
  --config config/settings.jsonc \
  --output models/fasttext_model.model
```

- If `dimension` is set in the config, training aggregates all texts per dimension value (e.g., per user).  
- If no dimension is set, each record is treated independently.

---

## Analysis

Extract variant families and compute pairwise overlaps:

```bash
python src/analyze_variation.py \
  --input data/example.json \
  --model models/fasttext_model.model \
  --text_field text \
  --method open \
  --config config/settings.jsonc \
  --output_dir results/
```

Outputs:

- `results/variant_families_<method>.txt` – detailed families with full pairwise overlap stats  
- `results/summary_<method>.txt` – short summary (number of families, average family size)
- `results/summary_<method>.csv` - detailed table for exclusivity score script

Optional: compute exclusivity scores

```bash
  python src/compute_exclusivity.py \
  --input results/summary_open.csv \
  --output results/exclusivity_open.csv \
  --metric gini
```

or, for selected families:

```bash
python src/compute_exclusivity.py \
  --input results/summary_open.csv \
  --output results/exclusivity_open_subset.csv \
  --metric entropy \
  --families 12 45 97
```

---

## Methods

- **`open`** – links tokens using cosine similarity on FastText embeddings (threshold τ = 0.75, top-30 neighbours).  
- **`strict`** – adds mutual/shared-neighbour filtering, frequency-ratio guard, and degree cap for more conservative clusters.

---

## Pairwise Overlap

For each family:

- All token pairs are compared across the configured dimension (e.g., user overlap).  
- The **mean Jaccard overlap** across all pairs defines the family’s cohesion.  
- Each token line also shows its **mean overlap** with the rest of the family.

If no dimension is configured, overlap is marked “N/A” and set to 1.0.

---

## Example Output

```text
# Jaccard is computed as MEAN pairwise *dimension* overlap across all token pairs in each family (dimension='user_id').
# Each token line shows its MEAN overlap with all other tokens in the family.

=== Family (total 512) | Jaccard(mean pairwise dimension overlap)=0.432 | size=8 ===
  • moien          freq=143    dimension_members=12   dimension_overlap_mean=0.48
  • moineen        freq=94     dimension_members=8    dimension_overlap_mean=0.41
  • mojeen         freq=63     dimension_members=7    dimension_overlap_mean=0.36
  • ...
```
