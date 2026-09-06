"""
Evaluation metrics for retrieval.

Note on naming: with a single gold chunk per query, "recall@k" as computed here
is equivalent to hit rate / success@k. It is reported as hit_rate to avoid
overstating what is measured.
"""
import math
from typing import List


def calculate_hit_rate(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, else 0.0."""
    return 1.0 if set(retrieved_ids[:k]) & set(relevant_ids) else 0.0


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of gold chunks retrieved in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal rank of the first relevant chunk; 0.0 if none retrieved."""
    rel = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """nDCG@k with binary relevance."""
    rel = set(relevant_ids)
    dcg = sum(1.0 / math.log2(i + 1)
              for i, rid in enumerate(retrieved_ids[:k], start=1) if rid in rel)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_correctness(prediction: str, ground_truth: str) -> float:
    raise NotImplementedError("Answer-correctness evaluation is not yet implemented.")


def evaluate_groundedness(prediction: str, context: str) -> float:
    raise NotImplementedError("Groundedness evaluation is not yet implemented.")
