"""
Paired significance testing over per-query retrieval results.

Reads data/eval/results_per_query.csv (produced by run_experiments.py) and
reports, for each metric, the mean paired difference between Hybrid and each
baseline with a bootstrap 95% CI and a sign-flip permutation p-value.

Queries are paired (every config sees every query), so paired tests are the
correct choice; unpaired comparisons would overstate the variance.
"""
import csv, random, statistics as st
from pathlib import Path

N_RESAMPLES = 10000
SEED = 0
ROOT = Path(__file__).resolve().parent.parent


def paired_test(a, b, n=N_RESAMPLES):
    d = [x - y for x, y in zip(a, b)]
    obs = st.mean(d)
    means = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(n))
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    cnt = sum(1 for _ in range(n)
              if abs(st.mean([x if random.random() < 0.5 else -x for x in d])) >= abs(obs))
    return obs, lo, hi, (cnt + 1) / (n + 1)


def main():
    random.seed(SEED)
    rows = list(csv.DictReader(open(ROOT / "data/eval/results_per_query.csv", encoding="utf-8")))
    print(f"n = {len(rows)} paired queries, {N_RESAMPLES} resamples, seed {SEED}\n")

    def col(cfg, m):
        return [float(r[f"{cfg}_{m}"]) for r in rows]

    for m in ["mrr", "ndcg@5", "hit@5", "recall@5"]:
        for base in ["BM25", "Vector"]:
            o, lo, hi, p = paired_test(col("Hybrid", m), col(base, m))
            sig = "*" if (hi < 0 or lo > 0) else " "
            print(f"{m:>9}  Hybrid - {base:<7} {o:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f} {sig}")
        print()


if __name__ == "__main__":
    main()
