"""
data_preparation.py

Builds the retrieval corpus and query files from the raw FinQA dataset.

Pipeline:
    1. Build full corpus  (text passages + linearized table rows)
    2. Build query files  (train / dev / test)
    3. Filter corpus      (text-only, max 512 tokens, used base_ids only)
    4. Filter queries     (text-only positives, fix broken IDs, remove edge cases)

Input:  source_data/train.json, dev.json, test.json
Output: data/corpus_text.jsonl
        data/train_queries_text.jsonl
        data/dev_queries_text.jsonl
        data/test_queries_text.jsonl
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE_DIR = Path("source_data")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = OUT_DIR / "corpus_text.jsonl"
TRAIN_QUERIES_PATH = OUT_DIR / "train_queries_text.jsonl"
DEV_QUERIES_PATH = OUT_DIR / "dev_queries_text.jsonl"
TEST_QUERIES_PATH = OUT_DIR / "test_queries_text.jsonl"

SPLITS = ["train.json", "dev.json", "test.json"]
QUERY_OUT_PATHS = [TRAIN_QUERIES_PATH, DEV_QUERIES_PATH, TEST_QUERIES_PATH]

# Tokenizer limit for MiniLM
MODEL_NAME = "all-MiniLM-L12-v2"
MAX_TOKENS = 512

# One known broken ID in the dataset: text index -1 means the last passage
BROKEN_ID_FIXES = {
    "RE/2010/page_120.pdf::text_-1": "RE/2010/page_120.pdf::text_23",
}


# ---------------------------------------------------------------------------
# Step 1: Build corpus (text passages only)
# ---------------------------------------------------------------------------

def build_corpus(source_dir: Path) -> dict:
    """
    Reads all FinQA splits and returns a corpus dict: doc_id -> text.

    doc_id format:
        <base_id>::text_<i>   for text passages
    """
    corpus = {}

    for split_filename in SPLITS:
        split_path = source_dir / split_filename

        with open(split_path, encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            base_id = entry["id"].split("-")[0]

            # Text passages come from pre_text + post_text
            all_text_lines = entry.get("pre_text", []) + entry.get("post_text", [])
            for i, line in enumerate(all_text_lines):
                text = line.strip().lower()
                is_dots_only = text and set(text) <= {"."}
                if not text or is_dots_only:
                    continue
                doc_id = f"{base_id}::text_{i}"
                if doc_id not in corpus:
                    corpus[doc_id] = text

    return corpus


# ---------------------------------------------------------------------------
# Step 2: Build query files (text-only positives)
# ---------------------------------------------------------------------------

def build_query_file(source_path: Path, out_path: Path, needed_doc_ids: set):
    """
    Reads one FinQA split and writes a query JSONL file.

    Only keeps queries that have at least one text positive (not table).
    Collects all used text doc_ids in needed_doc_ids (used later to trim the corpus).

    Each output line: {"query_id": ..., "query": ..., "positives": [...]}
    """
    with open(source_path, encoding="utf-8") as f:
        entries = json.load(f)

    records = []
    for entry in entries:
        query_id = entry["id"]
        query_text = entry["qa"]["question"].strip().lower()
        base_id = query_id.split("-")[0]
        gold_indices = entry["qa"].get("gold_inds", {})

        text_positives = []
        for index_key in gold_indices.keys():
            is_text_passage = isinstance(index_key, str) and index_key.startswith("text_")
            if not is_text_passage:
                continue
            doc_id = f"{base_id}::{index_key}"
            text_positives.append(doc_id)
            needed_doc_ids.add(doc_id)

        if not text_positives:
            continue

        records.append({
            "query_id": query_id,
            "query": query_text,
            "positives": text_positives,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f)
            f.write("\n")

    print(f"  {out_path.name}: {len(records)} queries written")


# ---------------------------------------------------------------------------
# Step 3: Filter corpus by token limit and used documents
# ---------------------------------------------------------------------------

def filter_corpus_by_token_limit(corpus: dict, tokenizer, max_tokens: int) -> dict:
    """Removes passages that exceed the model's token limit."""
    filtered = {}
    removed_count = 0

    for doc_id, text in corpus.items():
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_tokens:
            removed_count += 1
            continue
        filtered[doc_id] = text

    print(f"  Removed {removed_count} passages exceeding {max_tokens} tokens")
    return filtered


def filter_corpus_to_used_base_ids(corpus: dict, needed_doc_ids: set) -> dict:
    """
    Keeps only passages whose base_id appears in at least one query.
    This retains both positives and their in-document negatives.
    """
    needed_base_ids = {doc_id.split("::")[0] for doc_id in needed_doc_ids}

    filtered = {}
    for doc_id, text in corpus.items():
        base_id = doc_id.split("::")[0]
        if base_id in needed_base_ids:
            filtered[doc_id] = text

    print(f"  Corpus trimmed to {len(filtered)} passages across {len(needed_base_ids)} documents")
    return filtered


# ---------------------------------------------------------------------------
# Step 4: Clean query files
# ---------------------------------------------------------------------------

def build_corpus_index(corpus: dict) -> dict:
    """Groups corpus passage IDs by their base_id: base_id -> set of doc_ids."""
    index = defaultdict(set)
    for doc_id in corpus:
        base_id = doc_id.split("::")[0]
        index[base_id].add(doc_id)
    return dict(index)


def clean_query_file(query_path: Path, valid_doc_ids: set, corpus_index: dict):
    """
    Cleans one query file in place:
      - Fixes known broken passage IDs
      - Removes positives no longer present in the corpus
      - Removes queries with no valid positives left
      - Removes queries where all passages are positive (no negatives available)
    """
    with open(query_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    cleaned = []
    stats = {"fixed_ids": 0, "removed_positives": 0, "dropped_no_positives": 0, "dropped_no_negatives": 0}

    for record in records:
        valid_positives = []

        for pos_id in record["positives"]:
            # Apply known ID fix
            if pos_id in BROKEN_ID_FIXES:
                pos_id = BROKEN_ID_FIXES[pos_id]
                stats["fixed_ids"] += 1

            if pos_id in valid_doc_ids:
                valid_positives.append(pos_id)
            else:
                stats["removed_positives"] += 1

        if not valid_positives:
            stats["dropped_no_positives"] += 1
            continue

        # Check that at least one non-positive passage exists (needed as negative)
        base_id = valid_positives[0].split("::")[0]
        all_passage_ids = corpus_index.get(base_id, set())
        has_at_least_one_negative = bool(all_passage_ids - set(valid_positives))

        if not has_at_least_one_negative:
            stats["dropped_no_negatives"] += 1
            continue

        record["positives"] = valid_positives
        cleaned.append(record)

    with open(query_path, "w", encoding="utf-8") as f:
        for record in cleaned:
            json.dump(record, f)
            f.write("\n")

    print(
        f"  {query_path.name}: {len(records)} -> {len(cleaned)} queries "
        f"| fixed_ids={stats['fixed_ids']}, removed_positives={stats['removed_positives']}, "
        f"dropped_no_positives={stats['dropped_no_positives']}, dropped_no_negatives={stats['dropped_no_negatives']}"
    )


def write_corpus(corpus: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            json.dump({"doc_id": doc_id, "text": text}, f)
            f.write("\n")
    print(f"  Corpus written: {len(corpus)} passages -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Step 1: Build corpus ===")
    corpus = build_corpus(SOURCE_DIR)
    print(f"  Total passages: {len(corpus)}")

    print("\n=== Step 2: Build query files ===")
    needed_doc_ids = set()
    for split_filename, query_out_path in zip(SPLITS, QUERY_OUT_PATHS):
        build_query_file(SOURCE_DIR / split_filename, query_out_path, needed_doc_ids)

    print("\n=== Step 3: Filter corpus ===")
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    corpus = filter_corpus_by_token_limit(corpus, tokenizer, MAX_TOKENS)
    corpus = filter_corpus_to_used_base_ids(corpus, needed_doc_ids)
    write_corpus(corpus, CORPUS_PATH)

    print("\n=== Step 4: Clean query files ===")
    valid_doc_ids = set(corpus.keys())
    corpus_index = build_corpus_index(corpus)
    for query_path in QUERY_OUT_PATHS:
        clean_query_file(query_path, valid_doc_ids, corpus_index)

    print("\nDone.")


if __name__ == "__main__":
    main()