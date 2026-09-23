"""Evaluate the structure-only baseline on every regime.

    python scripts/run_baseline.py
    python scripts/run_baseline.py --regimes sameAs closeMatch

Nothing is trained: the bags of neighbours of the UCPGx are scored with the code that scores
the embedding tables of the models (``src/benchmark/baseline.py``), on the same splits and
queries. Each regime writes ``logs/<regime>/baseline/result.json``, then ``logs/results.json``
is rebuilt, so that ``make_tables.py`` shows the baseline next to the models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from run_benchmark import LOGS, REGIMES, ROOT, index, run_dir_of

sys.path.insert(0, str(ROOT / "code"))
from src import benchmark                                       # noqa: E402
from src.benchmark.baseline import Bags, evaluate               # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regimes", nargs="+", default=REGIMES)
    parser.add_argument("--csls-k", type=int, default=3)
    parser.add_argument("--tol-k", type=int, default=10, help="top-k of the decision F1")
    args = parser.parse_args()

    base = benchmark.load(ROOT / "Data" / "pgxlod_csv", cache_dir=ROOT / "Data" / "cache")
    bags = Bags(base["triples"], base["num_ucpgx"])
    print(f"{bags.num_ucpgx:,} UCPGx, {bags.num_pairs:,} distinct (predicate, component) pairs, "
          f"{bags.size.mean():.2f} pairs per bag on average")

    for regime in args.regimes:
        started = time.time()
        out = evaluate(bags, base, regime, csls_k=args.csls_k, tol_k=args.tol_k)
        out["minutes"] = round((time.time() - started) / 60.0, 1)
        dest = run_dir_of("baseline", regime)
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        cos, csls, clf = out["final"]["cosine"], out["final"]["csls"], out["final"]["classification"]
        print(f"{regime:16s} filtered MRR cos={cos['filtered']['MRR']:.4f} "
              f"csls={csls['filtered']['MRR']:.4f} | multiref MRR cos={cos['multiref']['MRR']:.4f} "
              f"csls={csls['multiref']['MRR']:.4f} | F1={clf['F1']:.4f} ({out['minutes']} min)",
              flush=True)

    print(f"{index()} results indexed in {LOGS / 'results.json'}")


if __name__ == "__main__":
    main()
