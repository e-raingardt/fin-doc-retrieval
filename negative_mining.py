"""
negative_mining.py

Generates two types of hard negatives for training:

    1. BM25 Hard Negatives
       For each query, the lexically most similar passages from the SAME document
       (excluding the positives) are selected via BM25. Five files are written,
       one per rank (top1 ... top5).

    2. Token-Swap Negatives
       For each positive, the most informative token (highest TF-IDF score within
       its document) is replaced with a similarly informative token from another
       positive. The resulting passage looks correct on the surface but contains
       a misleading semantic shift.

A final sanity check verifies that no negative accidentally matches a positive.

Input:  data/corpus_text.jsonl
        data/train_queries_text.jsonl
Output: data/top1_train_hard_negatives_bm25.jsonl ... top5
        data/train_swap_negatives.jsonl
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus_text.jsonl"
TRAIN_QUERIES_PATH = DATA_DIR / "train_queries_text.jsonl"

BM25_TOP_K = 5
BM25_OUT_PATHS = [
    DATA_DIR / f"top{rank}_train_hard_negatives_bm25.jsonl"
    for rank in range(1, BM25_TOP_K + 1)
]

SWAP_OUT_PATH = DATA_DIR / "train_swap_negatives.jsonl"
SWAP_TFIDF_RADIUS = 1.0   # only swap with replacement words of similar TF-IDF score
SWAP_MIN_WORD_LEN = 4
SWAP_RANDOM_SEED = 42

# Filler / scale words that should not be used as informative tokens
EXCLUDE_WORDS = {
    "million", "millions",
    "billion", "billions",
    "thousand", "thousands",
}


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> dict:
    """Reads corpus JSONL into a dict: doc_id -> text."""
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["doc_id"]] = obj.get("text", "")
    return corpus


def load_queries(path: Path) -> list:
    """Reads query JSONL into a list of dicts."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def group_corpus_by_base_id(corpus: dict) -> dict:
    """Groups passages by their base_id (document): base_id -> [(doc_id, text), ...]."""
    by_base = defaultdict(list)
    for doc_id, text in corpus.items():
        if "::" not in doc_id:
            continue
        base_id = doc_id.split("::")[0]
        by_base[base_id].append((doc_id, text))
    return dict(by_base)


def write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


# ---------------------------------------------------------------------------
# Part 1: BM25 Hard Negatives
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> list:
    """Basic whitespace + lowercase tokenizer used by BM25."""
    return text.lower().split()


def mine_bm25_top_negatives(query: str, positives: list, doc_list: list, k: int) -> list:
    """
    Returns up to k (doc_id, text) tuples — the BM25-highest-ranked passages
    from doc_list that are not in positives, sorted by score descending.
    """
    positives_set = set(positives)
    candidates = [
        (doc_id, text)
        for doc_id, text in doc_list
        if doc_id not in positives_set
    ]
    if not candidates:
        return []

    tokenized_docs = [simple_tokenize(text) for _, text in candidates]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(simple_tokenize(query))

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted_indices[:k]

    return [candidates[i] for i in top_indices]


def mine_bm25_negatives_for_all_queries(queries: list, corpus_by_base: dict, k: int) -> list:
    """
    For each query, mines top-k BM25 hard negatives from its document.

    Returns a list of length k, where entry rank contains all records for
    that rank (one record per query).
    """
    mined_by_rank = [[] for _ in range(k)]
    skipped_no_doc = 0
    skipped_no_candidate = 0

    for query in queries:
        query_id = query["query_id"]
        query_text = query["query"]
        positives = query["positives"]

        base_id = positives[0].split("::")[0]
        doc_list = corpus_by_base.get(base_id, [])
        if not doc_list:
            skipped_no_doc += 1
            continue

        top_negatives = mine_bm25_top_negatives(query_text, positives, doc_list, k=k)
        if not top_negatives:
            skipped_no_candidate += 1
            continue

        for rank, (neg_doc_id, neg_text) in enumerate(top_negatives):
            mined_by_rank[rank].append({
                "query_id": query_id,
                "query": query_text,
                "negative_doc_id": neg_doc_id,
                "negative_text": neg_text,
            })

    print(f"  Skipped (document not in corpus): {skipped_no_doc}")
    print(f"  Skipped (no negative candidates): {skipped_no_candidate}")
    for rank in range(k):
        print(f"  Top{rank + 1}: {len(mined_by_rank[rank])} negatives")

    return mined_by_rank


# ---------------------------------------------------------------------------
# Part 2: Token-Swap Negatives
# ---------------------------------------------------------------------------

def informative_tokenize(text: str) -> list:
    """
    Tokenizer for TF-IDF: lowercase, no stopwords, no scale words,
    no very short words. Returns a list (with repetitions).
    """
    if not text:
        return []
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [
        t for t in raw_tokens
        if t not in ENGLISH_STOP_WORDS
        and t not in EXCLUDE_WORDS
        and len(t) >= SWAP_MIN_WORD_LEN
    ]


def compute_idf_per_document(corpus_by_base: dict) -> dict:
    """
    Computes IDF for each base_id (each document is its own mini-corpus).
    Returns: base_id -> {word: idf_score}.
    """
    idf_by_base = {}
    for base_id, passages in corpus_by_base.items():
        n_passages = len(passages)
        doc_frequency = defaultdict(int)
        for _, text in passages:
            for word in set(informative_tokenize(text)):
                doc_frequency[word] += 1
        idf_by_base[base_id] = {
            word: math.log((n_passages + 1) / (freq + 1))
            for word, freq in doc_frequency.items()
        }
    return idf_by_base


def compute_tfidf_score(text: str, word: str, idf_map: dict) -> float:
    """TF-IDF score of a single word inside one passage."""
    term_counts = Counter(informative_tokenize(text))
    if word not in term_counts or word not in idf_map:
        return 0.0
    tf = term_counts[word]
    tf_scaled = 1.0 + math.log(tf) if tf >= 1 else 1.0
    return tf_scaled * idf_map[word]


def find_top_word_per_positive(queries: list, corpus: dict, corpus_by_base: dict, idf_by_base: dict) -> list:
    """
    For every (query, positive) pair, finds the most informative word in the
    positive passage (highest TF-IDF, considering only words that also appear
    in the query). Returns a list of dicts.
    """
    results = []

    for query in queries:
        query_id = query["query_id"]
        query_words = set(re.findall(r"[a-z0-9]+", query["query"].lower()))

        for doc_id in query["positives"]:
            passage_text = corpus.get(doc_id, "")
            if not passage_text:
                continue

            base_id = doc_id.split("::")[0]
            idf_map = idf_by_base.get(base_id, {})

            # Candidate words: in query AND in passage (the overlap)
            passage_words = set(informative_tokenize(passage_text))
            overlap_words = query_words & passage_words

            best_word = None
            best_score = -1.0
            for word in overlap_words:
                if re.search(r"[0-9]", word):
                    continue
                score = compute_tfidf_score(passage_text, word, idf_map)
                if score > best_score:
                    best_word = word
                    best_score = score

            if best_word is None or best_score <= 0:
                continue

            results.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "top_word": best_word,
                "tfidf": best_score,
            })

    return results


def replace_whole_word(text: str, old_word: str, new_word: str) -> str:
    """Replaces all whole-word occurrences of old_word with new_word (case-insensitive)."""
    pattern = r"\b" + re.escape(old_word) + r"\b"
    return re.sub(pattern, new_word, text, flags=re.IGNORECASE)


def build_swap_negatives(top_words_per_positive: list, queries: list, corpus: dict) -> list:
    """
    Creates one swap-negative per positive:
      - Take the top_word of THIS positive
      - Find another positive with a different top_word and similar TF-IDF score
      - Replace top_word in the passage with the other one
    """
    random.seed(SWAP_RANDOM_SEED)
    query_by_id = {q["query_id"]: q for q in queries}

    swap_negatives = []

    for i, entry in enumerate(top_words_per_positive):
        query_id = entry["query_id"]
        doc_id = entry["doc_id"]
        own_top_word = entry["top_word"]
        own_tfidf = entry["tfidf"]

        passage_text = corpus.get(doc_id, "")
        if not passage_text:
            continue

        # Sanity: the top word must actually appear as a whole word
        word_pattern = r"\b" + re.escape(own_top_word) + r"\b"
        if not re.search(word_pattern, passage_text, re.IGNORECASE):
            continue

        # Find replacement candidates from other positives
        candidates = []
        for j, other in enumerate(top_words_per_positive):
            if j == i:
                continue
            if other["top_word"] == own_top_word:
                continue
            if abs(other["tfidf"] - own_tfidf) > SWAP_TFIDF_RADIUS:
                continue
            candidates.append(other["top_word"])

        if not candidates:
            continue

        replacement_word = random.choice(candidates)
        swapped_text = replace_whole_word(passage_text, own_top_word, replacement_word)

        query_text = query_by_id.get(query_id, {}).get("query", "")
        swap_negatives.append({
            "query_id": query_id,
            "query": query_text,
            "negative_doc_id": doc_id,
            "negative_text": swapped_text,
        })

    return swap_negatives


# ---------------------------------------------------------------------------
# Part 3: Sanity check
# ---------------------------------------------------------------------------

def check_bm25_negatives(bm25_paths: list, query_to_positives: dict):
    """Verifies that no mined negative_doc_id is a positive for its query."""
    print("\n--- BM25 negatives check ---")
    for path in bm25_paths:
        if not path.exists():
            print(f"  Missing: {path}")
            continue

        violations = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                qid = record["query_id"]
                neg_id = record["negative_doc_id"]
                if neg_id in query_to_positives.get(qid, set()):
                    violations += 1
        print(f"  {path.name}: {violations} violations")


def check_swap_negatives(swap_path: Path, query_to_positives: dict, corpus: dict):
    """Verifies that no swap negative text matches a positive text exactly."""
    print("\n--- Swap negatives check ---")
    if not swap_path.exists():
        print(f"  Missing: {swap_path}")
        return

    # Map each query to the set of texts of its positives
    query_to_positive_texts = {}
    for qid, pos_ids in query_to_positives.items():
        query_to_positive_texts[qid] = {
            corpus.get(pid, "") for pid in pos_ids
        }

    violations = 0
    with open(swap_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            qid = record["query_id"]
            neg_text = record["negative_text"]
            if neg_text in query_to_positive_texts.get(qid, set()):
                violations += 1

    print(f"  {swap_path.name}: {violations} violations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Loading data ===")
    corpus = load_corpus(CORPUS_PATH)
    queries = load_queries(TRAIN_QUERIES_PATH)
    corpus_by_base = group_corpus_by_base_id(corpus)
    print(f"  Corpus passages: {len(corpus)}")
    print(f"  Train queries:   {len(queries)}")
    print(f"  Documents:       {len(corpus_by_base)}")

    print("\n=== Mining BM25 hard negatives ===")
    mined_by_rank = mine_bm25_negatives_for_all_queries(queries, corpus_by_base, k=BM25_TOP_K)
    for rank, out_path in enumerate(BM25_OUT_PATHS):
        write_jsonl(mined_by_rank[rank], out_path)
        print(f"  Written: {out_path}")

    print("\n=== Building token-swap negatives ===")
    idf_by_base = compute_idf_per_document(corpus_by_base)
    top_words = find_top_word_per_positive(queries, corpus, corpus_by_base, idf_by_base)
    print(f"  Positives with informative top-word: {len(top_words)}")

    swap_negatives = build_swap_negatives(top_words, queries, corpus)
    write_jsonl(swap_negatives, SWAP_OUT_PATH)
    print(f"  Swap negatives created: {len(swap_negatives)}")
    print(f"  Written: {SWAP_OUT_PATH}")

    print("\n=== Sanity checks ===")
    query_to_positives = {q["query_id"]: set(q["positives"]) for q in queries}
    check_bm25_negatives(BM25_OUT_PATHS, query_to_positives)
    check_swap_negatives(SWAP_OUT_PATH, query_to_positives, corpus)

    print("\nDone.")


if __name__ == "__main__":
    main()
