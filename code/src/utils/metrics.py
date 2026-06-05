"""Formatting of the metric blocks for the training log.

The metrics themselves are computed by :mod:`src.benchmark.scoring` (tie-aware ranks,
filtered and multiref blocks, top-k decision F1).
"""
from __future__ import annotations


def format_metrics(res: dict) -> str:
    """One line per metric block."""
    lines = []
    for name, m in res.items():
        if not isinstance(m, dict) or "MRR" not in m:
            continue
        parts = [f"MRR={m['MRR']:.4f}"]
        parts += [f"{k}={v:.4f}" for k, v in m.items() if k.startswith("Hit@")]
        parts.append(f"MR={m['MeanRank']:.1f}")
        parts.append(f"n={m['n']}")
        lines.append(f"[{name:>8}] " + " ".join(parts))
    return "\n".join(lines)
