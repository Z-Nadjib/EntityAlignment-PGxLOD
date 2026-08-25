"""Build the result tables from ``logs/results.json``.

    python scripts/make_tables.py
    python scripts/make_tables.py --out ../logs/tables.md

One table per metric, regimes in rows and models in columns, with the rank of each column
on the mean over the regimes as the last row. The structure-only baseline
(``scripts/run_baseline.py``), when it has been run, is the last column; it is a reference
and is left out of the rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"

REGIMES = ["broadMatch/pgx", "closeMatch", "sameAs", "broadMatch/onco", "relatedMatch",
           "related"]
MODELS = ["gcnalign", "alinet", "rrea", "mraea", "kecg", "bootea", "jape", "naea"]
BASELINE = "baseline"
NAMES = {"gcnalign": "GCN-Align", "alinet": "AliNet", "rrea": "RREA", "mraea": "MRAEA",
         "kecg": "KECG", "bootea": "BootEA", "jape": "JAPE", "naea": "NAEA",
         BASELINE: "Baseline"}

# (title, block, key)
METRICS = [("filtered MRR", "filtered", "MRR"),
           ("filtered Hit@1", "filtered", "Hit@1"),
           ("filtered Hit@10", "filtered", "Hit@10"),
           ("multiref MRR", "multiref", "MRR"),
           ("top-10 decision F1", "classification", "F1")]

NOTES = {
    "filtered MRR": "Per test pair, the other known partners of the query masked.",
    "filtered Hit@1": "Share of the test pairs whose partner comes first.",
    "filtered Hit@10": "Share of the test pairs whose partner is in the first ten.",
    "multiref MRR": "Per query rather than per pair: the position of its first test partner, "
                    "only the training ones masked. Stricter by construction.",
    "top-10 decision F1": "Yes or no decision on the cosine score of the best candidate, a "
                          "query counting as correct when one of its test partners is in its "
                          "top ten (`clf_tol_k = 10`), its train partners masked. The "
                          "threshold is chosen on the validation set and applied unchanged "
                          "to the test split, with disjoint negatives on each side. Scored "
                          "at cosine in every table.",
}


def load_results(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found, run scripts/run_benchmark.py first")
    return {(r["model"], r["regime"]): r for r in json.loads(path.read_text())}


def value(entry, block: str, key: str, score: str = "csls"):
    """One metric of a result. The F1 block is scored at cosine whatever ``score`` is."""
    if entry is None:
        return None
    source = entry["final"] if block == "classification" else entry["final"][score]
    holder = source.get(block)
    if not holder or key not in holder:
        return None
    return float(holder[key])


def fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def table(header, rows) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(lines)


def metric_section(title, block, key, results, score) -> str:
    columns = MODELS + ([BASELINE] if any(m == BASELINE for m, _ in results) else [])
    header = ["regime"] + [NAMES.get(m, m) for m in columns]
    values = {m: [] for m in columns}
    rows = []
    for regime in REGIMES:
        row = [regime]
        for model in columns:
            v = value(results.get((model, regime)), block, key, score)
            values[model].append(v)
            row.append(fmt(v))
        rows.append(row)

    # The mean is taken over the regimes where every column has a value: a model missing a
    # regime would otherwise be ranked on an easier basis than the others.
    common = [i for i, _ in enumerate(REGIMES)
              if all(values[m][i] is not None for m in MODELS)]
    means = {m: (sum(values[m][i] for i in common) / len(common)
                 if common and all(values[m][i] is not None for i in common) else None)
             for m in columns}
    order = sorted([m for m in MODELS if means[m] is not None], key=lambda m: -means[m])
    rank = {m: i + 1 for i, m in enumerate(order)}
    rows.append(["**rank**"] + [str(rank.get(m, "-")) for m in columns])
    rows.append(["*mean*"] + [fmt(means[m]) for m in columns])

    basis = ("the six regimes" if len(common) == len(REGIMES)
             else "regimes " + ", ".join(f"`{REGIMES[i]}`" for i in common))
    podium = ", ".join(f"**{i + 1}. {NAMES.get(m, m)}** ({means[m]:.4f})"
                       for i, m in enumerate(order[:3]))
    return "\n".join([f"### {title}", "", NOTES[title], "", table(header, rows), "",
                      f"Rank and mean over {basis}, that is the ones where every column is "
                      f"filled. Podium: {podium}."])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(LOGS / "results.json"))
    parser.add_argument("--score", default="csls", choices=["cosine", "csls"])
    parser.add_argument("--out", default=None, help="write to this file instead of stdout")
    args = parser.parse_args()

    results = load_results(Path(args.results))
    missing = [(m, r) for r in REGIMES for m in MODELS if (m, r) not in results]
    runs = sum(1 for m, _ in results if m in MODELS)

    body = ["# Entity alignment on PGxLOD: the tables", "",
            f"Ranks scored at {args.score.upper()}, the setting of the configurations of "
            f"this repository; the F1 at cosine. Test split read once, {runs} runs "
            f"out of {len(MODELS) * len(REGIMES)}.", ""]
    if missing:
        body += ["Missing runs: " + ", ".join(f"`{m}`/`{r}`" for m, r in missing) + "", ""]
    for title, block, key in METRICS:
        body += [metric_section(title, block, key, results, args.score), "", "---", ""]

    text = "\n".join(body)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
