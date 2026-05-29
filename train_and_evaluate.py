"""
train_and_evaluate.py

Fine-tunes a MiniLM sentence-transformer for intra-document passage retrieval
on FinQA, using triplet loss with configurable hard negatives.

Pipeline:
    1. Load corpus and queries (train / dev / test)
    2. Load hard negatives for the chosen strategy (BM25 top-k or token-swap)
    3. Train with triplet loss (query, positive, hard_negative)
       - Each epoch re-samples positives and negatives without replacement
       - Best checkpoint by Dev Recall@5 is kept
    4. Evaluate on test set: Recall@1/3/5 and MRR
    5. Save per-query analysis (improved vs regressed vs unresolved)

The script trains ONE model for ONE seed/strategy. To average over seeds,
run it multiple times and call aggregate_seed_results().
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, InputExample, losses


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus_text.jsonl"
TRAIN_QUERIES_PATH = DATA_DIR / "train_queries_text.jsonl"
DEV_QUERIES_PATH = DATA_DIR / "dev_queries_text.jsonl"
TEST_QUERIES_PATH = DATA_DIR / "test_queries_text.jsonl"

# Choose the hard-negative source for this training run:
HARD_NEGATIVES_PATH = DATA_DIR / "top5_train_hard_negatives_bm25.jsonl"
# Alternative: DATA_DIR / "top1_train_hard_negatives_bm25.jsonl"

# Output directory for this run (one folder per seed/strategy)
OUT_DIR = DATA_DIR / "trained" / "bm25_seed63"

# Base model and training hyperparameters
MODEL_NAME = "all-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 512
SEED = 63
LEARNING_RATE = 5e-6
TRIPLET_MARGIN = 0.3
EPOCHS = 2
BATCH_SIZE = 16
WARMUP_RATIO = 0.1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> dict:
    """Reads corpus JSONL into a dict: doc_id -> text."""
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["doc_id"]] = obj["text"]
    return corpus


def load_queries(path: Path) -> list:
    """Reads query JSONL into a list of dicts."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def group_corpus_by_base_id(corpus: dict) -> dict:
    """Groups passages by their base_id: base_id -> [(doc_id, text), ...]."""
    by_base = defaultdict(list)
    for doc_id, text in corpus.items():
        if "::" not in doc_id:
            continue
        base_id = doc_id.split("::")[0]
        by_base[base_id].append((doc_id, text))
    return dict(by_base)


def load_hard_negatives(path: Path) -> dict:
    """Reads hard-negative JSONL into a dict: query_id -> [negative_text, ...]."""
    by_qid = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            qid = record.get("query_id", "").strip()
            neg_text = record.get("negative_text", "").strip()
            if qid and neg_text:
                by_qid[qid].append(neg_text)
    return dict(by_qid)


# ---------------------------------------------------------------------------
# Training example builder
#   For each query we sample one positive and one negative per epoch.
#   We use small "queues" so that across epochs each positive/negative is
#   visited before any is repeated (sampling without replacement).
# ---------------------------------------------------------------------------

class SamplingQueues:
    """Holds per-query shuffled queues for sampling without replacement."""

    def __init__(self):
        self.positives = {}
        self.in_doc_negatives = {}
        self.hard_negatives = {}

    def clear(self):
        self.positives.clear()
        self.in_doc_negatives.clear()
        self.hard_negatives.clear()


def draw_without_replacement(queue_dict: dict, qid: str, pool: list, n: int, rng: random.Random) -> list:
    """
    Draws n elements from pool without replacement, persisting state across calls.
    When the queue is exhausted, the pool is reshuffled and we start over.
    """
    drawn = []
    for _ in range(n):
        if qid not in queue_dict or not queue_dict[qid]:
            queue_dict[qid] = list(pool)
            rng.shuffle(queue_dict[qid])
        drawn.append(queue_dict[qid].pop())
    return drawn


def build_triplet_examples(
    queries: list,
    corpus: dict,
    corpus_by_base: dict,
    hard_negatives_by_qid: dict,
    queues: SamplingQueues,
    epoch: int,
    seed: int,
) -> list:
    """
    Builds InputExamples for triplet loss: [query, positive, negative].

    The negative is either a hard negative (if available for this query) or
    a random in-document non-positive passage.
    """
    rng = random.Random(seed + epoch)
    examples = []

    for query in queries:
        qid = query["query_id"]
        query_text = query["query"]
        positive_ids = query["positives"]

        base_id = positive_ids[0].split("::")[0]
        all_doc_passages = corpus_by_base.get(base_id, [])
        if not all_doc_passages:
            continue

        # In-document negatives = passages of the same document that are NOT positive
        positive_set = set(positive_ids)
        in_doc_negative_ids = [
            doc_id for doc_id, _ in all_doc_passages
            if doc_id not in positive_set
        ]
        if not in_doc_negative_ids:
            continue

        # 1. Sample a positive
        pos_id = draw_without_replacement(queues.positives, qid, positive_ids, 1, rng)[0]
        pos_text = corpus.get(pos_id)
        if pos_text is None:
            continue

        # 2. Sample an in-doc negative as fallback
        neg_id = draw_without_replacement(queues.in_doc_negatives, qid, in_doc_negative_ids, 1, rng)[0]
        neg_text = corpus[neg_id]

        # 3. If a hard negative is available, replace the fallback
        hard_neg_pool = hard_negatives_by_qid.get(qid)
        if hard_neg_pool:
            hard_neg_text = draw_without_replacement(queues.hard_negatives, qid, hard_neg_pool, 1, rng)[0]
            if hard_neg_text and hard_neg_text != neg_text:
                neg_text = hard_neg_text

        examples.append(InputExample(texts=[query_text, pos_text, neg_text]))

    return examples


# ---------------------------------------------------------------------------
# Evaluation
#   All metrics are computed PER DOCUMENT: we only rank passages of the same
#   document as the query, because that is the retrieval setting.
# ---------------------------------------------------------------------------

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-12)


@torch.inference_mode()
def encode_query_and_passages(model, query_text: str, passage_texts: list, batch_size: int):
    """Encodes one query and its candidate passages, returns L2-normalized arrays."""
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    query_embedding = l2_normalize(query_embedding.astype(np.float32))

    passage_embeddings = model.encode(passage_texts, batch_size=batch_size, convert_to_numpy=True)
    passage_embeddings = l2_normalize(passage_embeddings.astype(np.float32))

    return query_embedding, passage_embeddings


def rank_passages_for_query(model, query: dict, corpus_by_base: dict, batch_size: int):
    """
    Returns (doc_ids, ranked_indices) for one query — passages restricted to
    the query's document and sorted by similarity (descending).
    Returns (None, None) if the document is not in the corpus.
    """
    positives = query["positives"]
    base_id = positives[0].split("::")[0]
    doc_passages = corpus_by_base.get(base_id, [])
    if not doc_passages:
        return None, None

    doc_ids = [doc_id for doc_id, _ in doc_passages]
    texts = [text for _, text in doc_passages]

    query_emb, passage_emb = encode_query_and_passages(model, query["query"], texts, batch_size)
    similarities = (query_emb @ passage_emb.T)[0]
    ranked_indices = np.argsort(-similarities)

    return doc_ids, ranked_indices


@torch.inference_mode()
def eval_recall_at_k(model, queries: list, corpus_by_base: dict, k: int, batch_size: int = 64) -> float:
    """Recall@K: fraction of queries whose top-K retrieved set contains a positive."""
    n_correct = 0
    n_evaluated = 0

    for query in tqdm(queries, desc=f"Eval Recall@{k}"):
        doc_ids, ranked_indices = rank_passages_for_query(model, query, corpus_by_base, batch_size)
        if doc_ids is None:
            continue

        n_evaluated += 1
        top_k_ids = {doc_ids[i] for i in ranked_indices[:k]}
        if top_k_ids & set(query["positives"]):
            n_correct += 1

    return n_correct / n_evaluated if n_evaluated else 0.0


@torch.inference_mode()
def eval_mrr(model, queries: list, corpus_by_base: dict, batch_size: int = 64) -> float:
    """Mean Reciprocal Rank — averages 1 / rank_of_first_positive across queries."""
    reciprocal_ranks = []

    for query in tqdm(queries, desc="Eval MRR"):
        doc_ids, ranked_indices = rank_passages_for_query(model, query, corpus_by_base, batch_size)
        if doc_ids is None:
            continue

        positive_set = set(query["positives"])
        for rank, idx in enumerate(ranked_indices, start=1):
            if doc_ids[idx] in positive_set:
                reciprocal_ranks.append(1.0 / rank)
                break

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


@torch.inference_mode()
def eval_recall_per_query(model, queries: list, corpus_by_base: dict, k: int, batch_size: int = 64) -> dict:
    """Returns {query_id: True/False} — whether each query was recalled within top-K."""
    results = {}
    for query in tqdm(queries, desc=f"Per-query Recall@{k}"):
        doc_ids, ranked_indices = rank_passages_for_query(model, query, corpus_by_base, batch_size)
        if doc_ids is None:
            continue
        top_k_ids = {doc_ids[i] for i in ranked_indices[:k]}
        results[query["query_id"]] = bool(top_k_ids & set(query["positives"]))
    return results


@torch.inference_mode()
def collect_best_positive_ranks(model, queries: list, corpus_by_base: dict, batch_size: int = 64) -> dict:
    """Returns {query_id: best_rank} — the rank of the highest-ranked positive."""
    best_ranks = {}
    for query in tqdm(queries, desc="Collecting positive ranks"):
        doc_ids, ranked_indices = rank_passages_for_query(model, query, corpus_by_base, batch_size)
        if doc_ids is None:
            continue
        positive_set = set(query["positives"])
        for rank, idx in enumerate(ranked_indices, start=1):
            if doc_ids[idx] in positive_set:
                best_ranks[query["query_id"]] = rank
                break
    return best_ranks


# ---------------------------------------------------------------------------
# Per-query analysis (baseline vs trained)
# ---------------------------------------------------------------------------

def categorize_queries(baseline_recall: dict, trained_recall: dict) -> dict:
    """Splits queries into already_easy / improved / regressed / unresolved buckets."""
    categories = {"already_easy": [], "improved": [], "regressed": [], "unresolved": []}

    for qid, was_recalled in baseline_recall.items():
        is_recalled = trained_recall.get(qid, False)

        if was_recalled and is_recalled:
            categories["already_easy"].append(qid)
        elif not was_recalled and is_recalled:
            categories["improved"].append(qid)
        elif was_recalled and not is_recalled:
            categories["regressed"].append(qid)
        else:
            categories["unresolved"].append(qid)

    return categories


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def flatten_numeric_metrics(obj, prefix: str = "") -> dict:
    """Flattens a nested results dict into {dotted_key: value} for numeric values only."""
    flat = {}
    if not isinstance(obj, dict):
        return flat

    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[full_key] = value
        elif isinstance(value, dict):
            flat.update(flatten_numeric_metrics(value, full_key))
    return flat


def aggregate_seed_results(seed_dirs: list, out_path: Path):
    """
    Aggregates eval_results.json from multiple seed directories.
    Computes mean and std for each metric across seeds.
    """
    all_evals = []
    for d in seed_dirs:
        eval_file = d / "eval_results.json"
        if not eval_file.exists():
            print(f"  Skipping (not found): {eval_file}")
            continue
        with open(eval_file, encoding="utf-8") as f:
            all_evals.append(json.load(f))

    if len(all_evals) < 2:
        raise ValueError(f"Need at least 2 seed results, found {len(all_evals)}")

    # Only keep metric keys (skip hyperparameters like seed, learning_rate)
    metric_prefixes = ("MRR", "Dev.", "Test.")
    values_by_key = defaultdict(list)
    for e in all_evals:
        flat = flatten_numeric_metrics(e)
        for key, value in flat.items():
            if key == "MRR" or any(key.startswith(p) for p in metric_prefixes):
                values_by_key[key].append(value)

    aggregated = {}
    for key, values in sorted(values_by_key.items()):
        arr = np.array(values)
        aggregated[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    result = {
        "n_seeds": len(all_evals),
        "seed_dirs": [str(d) for d in seed_dirs],
        "metrics": aggregated,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Mean ± Std across seeds:")
    for key, stats in aggregated.items():
        print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nWritten: {out_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)

    print("=== Loading data ===")
    corpus = load_corpus(CORPUS_PATH)
    corpus_by_base = group_corpus_by_base_id(corpus)
    train_queries = load_queries(TRAIN_QUERIES_PATH)
    dev_queries = load_queries(DEV_QUERIES_PATH)
    test_queries = load_queries(TEST_QUERIES_PATH)
    hard_negatives = load_hard_negatives(HARD_NEGATIVES_PATH)

    print(f"  Corpus passages: {len(corpus)}")
    print(f"  Train / Dev / Test queries: {len(train_queries)} / {len(dev_queries)} / {len(test_queries)}")
    print(f"  Queries with hard negatives: {len(hard_negatives)}")

    print("\n=== Loading base model ===")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("\n=== Baseline (zero-shot) test ranks ===")
    baseline_test_recall = eval_recall_per_query(model, test_queries, corpus_by_base, k=5)
    baseline_ranks = collect_best_positive_ranks(model, test_queries, corpus_by_base)
    with open(OUT_DIR / "baseline_ranks.json", "w", encoding="utf-8") as f:
        json.dump(baseline_ranks, f)

    print("\n=== Training ===")
    train_loss_fn = losses.TripletLoss(
        model=model,
        triplet_margin=TRIPLET_MARGIN,
        distance_metric=losses.TripletDistanceMetric.COSINE,
    )

    queues = SamplingQueues()
    best_dev_recall = -1.0
    eval_results = {}

    for epoch in range(1, EPOCHS + 1):
        examples = build_triplet_examples(
            train_queries, corpus, corpus_by_base, hard_negatives,
            queues=queues, epoch=epoch, seed=SEED,
        )

        train_dataloader = DataLoader(
            examples,
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=True,
            collate_fn=model.smart_batching_collate,
        )
        warmup_steps = math.ceil(len(train_dataloader) * WARMUP_RATIO)

        model.fit(
            train_objectives=[(train_dataloader, train_loss_fn)],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": LEARNING_RATE},
            show_progress_bar=True,
        )

        dev_recall = eval_recall_at_k(model, dev_queries, corpus_by_base, k=5)
        eval_results[f"Recall@5_epoch{epoch}"] = dev_recall
        print(f"Epoch {epoch}: Dev Recall@5 = {dev_recall:.4f}")

        if dev_recall > best_dev_recall:
            best_dev_recall = dev_recall
            model.save(str(OUT_DIR))
            print(f"  New best — saved to {OUT_DIR}")

    print(f"\nBest Dev Recall@5: {best_dev_recall:.4f}")

    print("\n=== Final test evaluation ===")
    best_model = SentenceTransformer(str(OUT_DIR))
    best_model.max_seq_length = MAX_SEQ_LENGTH

    test_results = {}
    for k in [1, 3, 5]:
        recall_k = eval_recall_at_k(best_model, test_queries, corpus_by_base, k=k)
        test_results[f"Recall@{k}"] = recall_k
        print(f"  Test Recall@{k}: {recall_k:.4f}")

    test_mrr = eval_mrr(best_model, test_queries, corpus_by_base)
    print(f"  Test MRR: {test_mrr:.4f}")

    print("\n=== Per-query analysis ===")
    trained_test_recall = eval_recall_per_query(best_model, test_queries, corpus_by_base, k=5)
    trained_ranks = collect_best_positive_ranks(best_model, test_queries, corpus_by_base)
    categories = categorize_queries(baseline_test_recall, trained_test_recall)

    total = sum(len(v) for v in categories.values())
    for name, qids in categories.items():
        pct = 100 * len(qids) / total if total else 0
        print(f"  {name:15s}: {len(qids):4d}  ({pct:.1f}%)")

    # Save everything
    results_summary = {
        "MRR": test_mrr,
        "Dev": eval_results,
        "Test": test_results,
        "max_seq_length": MAX_SEQ_LENGTH,
        "triplet_margin": TRIPLET_MARGIN,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "seed": SEED,
    }
    with open(OUT_DIR / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    query_analysis = dict(categories)
    query_analysis["baseline_best_rank"] = baseline_ranks
    query_analysis["trained_best_rank"] = trained_ranks
    with open(OUT_DIR / "query_analysis.json", "w", encoding="utf-8") as f:
        json.dump(query_analysis, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
