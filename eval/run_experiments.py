"""
Evaluation harness comparing BM25, dense, and RRF-fused retrieval.

Ground truth comes solely from data/qa_dataset.json. Gold labels are NEVER
derived from the output of any retriever being evaluated -- an earlier version
of this script substituted the fusion system's own top hits for missing gold
labels, which made the fusion condition score perfectly by construction and
scored the baselines against fusion's output rather than against ground truth.
Unresolvable gold IDs are now a hard error.
"""
import sys
import json
import statistics
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import BM25Retriever
from retrieval.vector_index import VectorRetriever
from retrieval.hybrid_fusion import reciprocal_rank_fusion
from eval.metrics import (
    calculate_hit_rate, calculate_recall_at_k, calculate_mrr, calculate_ndcg_at_k,
)

K = 5
CANDIDATE_DEPTH = 60


def evaluate(retrieved_ids, gold_ids):
    return {
        "hit@5": calculate_hit_rate(retrieved_ids, gold_ids, K),
        "recall@5": calculate_recall_at_k(retrieved_ids, gold_ids, K),
        "mrr": calculate_mrr(retrieved_ids, gold_ids),
        "ndcg@5": calculate_ndcg_at_k(retrieved_ids, gold_ids, K),
    }


def run_evaluation_suite(dataset_path, chunks_path, chroma_dir, rebuild=True):
    qa_data = json.load(open(dataset_path, encoding="utf-8"))
    chunks = json.load(open(chunks_path, encoding="utf-8"))
    print(f"Loaded {len(qa_data)} queries over {len(chunks)} chunks.")

    # --- Validate ground truth against the corpus. No silent patching. ---
    chunk_ids = {c["id"] for c in chunks}
    dangling = {
        gid for item in qa_data for gid in item["relevant_ids"] if gid not in chunk_ids
    }
    if dangling:
        raise SystemExit(
            f"ERROR: {len(dangling)} gold chunk ID(s) in {Path(dataset_path).name} do not\n"
            f"exist in {Path(chunks_path).name}. The QA set and the parsed corpus are out of\n"
            f"sync -- regenerate the QA set against the current chunks, or re-parse the PDF\n"
            f"that the QA set was written for. Refusing to score against invalid gold labels.\n"
            f"Examples: {sorted(dangling)[:3]}"
        )
    print("Ground truth validated: all gold IDs resolve to real chunks.")

    bm25 = BM25Retriever()
    vector = VectorRetriever(persist_directory=str(chroma_dir))

    # The Chroma collection is keyed by chunk ID. If chunks were re-parsed, stale
    # vectors under old IDs survive an upsert and silently pollute retrieval, so
    # the collection is dropped and rebuilt. Done through the API rather than by
    # deleting files, which fails on Windows while the Streamlit app holds a lock.
    if rebuild:
        try:
            vector.client.delete_collection("paper_chunks")
        except Exception:
            pass
        vector.collection = vector.client.get_or_create_collection(
            name="paper_chunks", metadata={"hnsw:space": "cosine"}
        )
        print("Rebuilt ChromaDB collection from scratch.")

    bm25.index_chunks(chunks)
    vector.index_chunks(chunks)

    indexed = vector.collection.count()
    if indexed != len(chunks):
        raise SystemExit(
            f"ERROR: vector store holds {indexed} entries but the corpus has {len(chunks)} "
            f"chunks. Refusing to run against an inconsistent index."
        )

    configs = ["BM25", "Vector", "Hybrid"]
    results = {c: [] for c in configs}
    per_query = []

    print("\n--- Running experiments ---")
    for i, item in enumerate(qa_data, start=1):
        query, gold = item["query"], item["relevant_ids"]

        ranked = {
            "BM25":   [c["id"] for c in bm25.search(query, top_k=K)],
            "Vector": [c["id"] for c in vector.search(query, top_k=K)],
            "Hybrid": [c["id"] for c in reciprocal_rank_fusion(
                           bm25.search(query, top_k=CANDIDATE_DEPTH),
                           vector.search(query, top_k=CANDIDATE_DEPTH),
                       )[:K]],
        }

        row = {"query": query, "source_type": item.get("source_type", "?")}
        for cfg in configs:
            m = evaluate(ranked[cfg], gold)
            results[cfg].append(m)
            row.update({f"{cfg}_{k}": v for k, v in m.items()})
        per_query.append(row)

        if i % 20 == 0:
            print(f"  {i}/{len(qa_data)} queries")

    # --- Aggregate ---
    print("\n" + "=" * 62)
    print(f"{'Config':<10} {'Hit@5':>9} {'Recall@5':>9} {'MRR':>9} {'nDCG@5':>9}")
    print("-" * 62)
    for cfg in configs:
        agg = {k: statistics.mean(m[k] for m in results[cfg])
               for k in ("hit@5", "recall@5", "mrr", "ndcg@5")}
        print(f"{cfg:<10} {agg['hit@5']:>9.4f} {agg['recall@5']:>9.4f} "
              f"{agg['mrr']:>9.4f} {agg['ndcg@5']:>9.4f}")
    print("=" * 62)

    # --- Breakdown by gold chunk modality ---
    types = sorted({r["source_type"] for r in per_query})
    print(f"\nMRR by gold chunk type:")
    print(f"{'Type':<10} {'n':>4}  " + "  ".join(f"{c:>8}" for c in configs))
    for t in types:
        rows = [r for r in per_query if r["source_type"] == t]
        vals = "  ".join(f"{statistics.mean(r[f'{c}_mrr'] for r in rows):>8.4f}"
                         for c in configs)
        print(f"{t:<10} {len(rows):>4}  {vals}")

    out = Path(dataset_path).parent / "eval" / "results_per_query.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
        w.writeheader()
        w.writerows(per_query)
    print(f"\nPer-query results written to {out}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_evaluation_suite(
        root / "data" / "qa_dataset.json",
        root / "data" / "processed" / "chunks.json",
        root / "data" / "processed" / "chroma_db_eval",
    )
