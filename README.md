# Fine-Tuning Sentence Transformers for Financial Document Retrieval

Fine-tuning a sentence-transformer bi-encoder for intra-document passage retrieval on financial reports, comparing two hard-negative mining strategies.

## Overview

Given a question about a financial document and a set of candidate passages from the same document, the model retrieves the passage that answers the question. The base model is `all-MiniLM-L12-v2`, fine-tuned on FinQA with triplet loss.

The core research question: **can token-swap negatives provide a useful learning signal compared to the established BM25 baseline?** Two strategies are compared:

- **BM25 hard negatives** — lexically similar non-positive passages from the same document
- **Token-swap negatives** — positives in which the most informative token (highest TF-IDF score within the document) has been replaced with a similarly informative token from another positive, producing a semantically misleading variant

Both are benchmarked against a zero-shot baseline. Each strategy is trained over 3 random seeds; results are reported as mean ± std.

## Results

| Model              | Recall@1            | Recall@3            | Recall@5            | MRR                 |
|--------------------|---------------------|---------------------|---------------------|---------------------|
| Zero-shot baseline | 0.7188              | 0.8639              | 0.8934              | 0.8015              |
| BM25 negatives     | 0.7596 (± 0.0079)   | 0.8919 (± 0.0129)   | 0.9267 (± 0.0092)   | 0.8341 (± 0.0072)   |
| Swap negatives     | 0.6765 (± 0.0057)   | 0.8337 (± 0.0035)   | 0.8972 (± 0.0086)   | 0.7744 (± 0.0045)   |

See `analysis.ipynb` for full plots and per-query breakdown.

**Summary:** BM25 hard negatives improved over the zero-shot baseline across all metrics. The proposed token-swap strategy underperformed the baseline on stricter metrics (Recall@1, Recall@3, MRR) and only matched it on Recall@5. A likely cause is that the swapped tokens shifted meaning too subtly to provide a clear training signal, while sometimes producing passages that remained semantically valid — suggesting that surface-level token perturbations are not sufficient to generate informative negatives in this domain.

## Project structure

```
.
├── data_preparation.py     # Build corpus and query files from raw FinQA
├── negative_mining.py      # Generate BM25 and token-swap hard negatives
├── train_and_evaluate.py   # Fine-tune model and evaluate on test set
├── analysis.ipynb          # Results tables and plots across seeds
├── source_data/            # Raw FinQA splits (train/dev/test.json)
└── data/                   # Generated artifacts
    ├── corpus_text.jsonl
    ├── train_queries_text.jsonl
    ├── dev_queries_text.jsonl
    ├── test_queries_text.jsonl
    ├── top{1..5}_train_hard_negatives_bm25.jsonl
    ├── train_swap_negatives.jsonl
    └── trained/            # One subfolder per seed/strategy
```

## Method

### Data preparation (`data_preparation.py`)

Builds a passage corpus from FinQA documents, keeping only text passages (table rows are dropped) and filtering passages exceeding the model's 512-token limit. Queries are linked to their text positives only.

### Negative mining (`negative_mining.py`)

- **BM25**: for each query, the top-5 BM25-scored non-positive passages from the query's document are saved as hard negatives (one file per rank).
- **Token-swap**: for each positive, identifies the query-overlapping word with the highest TF-IDF score inside its document. This word is replaced with a similarly informative word taken from another positive. The result is a passage that looks correct on the surface but contains a misleading semantic shift.

A sanity check verifies that no mined negative accidentally matches a positive.

### Training and evaluation (`train_and_evaluate.py`)

Triplet loss with cosine distance. Each epoch re-samples positives and negatives without replacement, so every example is seen before any is repeated. The best checkpoint is selected by Dev Recall@5 and evaluated on the test set with Recall@1/3/5 and MRR.

All evaluation is **intra-document**: the candidate set for a query is restricted to passages from the same document, matching the practical retrieval setting.

## Usage

```bash
# 1. Prepare data
python data_preparation.py

# 2. Mine hard negatives
python negative_mining.py

# 3. Train one model (edit SEED and HARD_NEGATIVES_PATH at the top of the script)
python train_and_evaluate.py

# 4. Analyse results across seeds
jupyter notebook analysis.ipynb
```

To compare strategies, run step 3 multiple times — once per (strategy × seed) combination — then run the notebook.

## Requirements

- Python 3.13
- `sentence-transformers`
- `rank_bm25`
- `scikit-learn`
- `torch`
- `numpy`, `pandas`, `matplotlib`, `tqdm`

Install via `pip install -r requirements.txt`.

## Dataset

This project uses the FinQA dataset:

- **Paper:** Chen et al., *FinQA: A Dataset of Numerical Reasoning over Financial Data* (EMNLP 2021)
- **Repository:** [github.com/czyssrs/FinQA](https://github.com/czyssrs/FinQA)

The dataset is not distributed with this repository due to licensing and size considerations.

### Download

Clone the FinQA repository and copy the official data splits:

```bash
git clone https://github.com/czyssrs/FinQA.git
```

Place the following files in `source_data/`:

```
source_data/
├── train.json
├── dev.json
└── test.json
```

The preprocessing script expects the original FinQA file format without modifications.

## Reproducibility

All experiments reported in this repository were conducted using:

- FinQA official train/dev/test splits
- `sentence-transformers/all-MiniLM-L12-v2`
- Random seeds: 21, 42, 63

Generated artifacts in `data/` can be reproduced by running:

```bash
python data_preparation.py
python negative_mining.py
```
