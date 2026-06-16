"""The graph, the six regimes and their leak-free splits, built once and cached.

The graph and its flat verbalisation come from :mod:`.graph`, the regime splits and the
validation carved out of the train set from :mod:`.partition`. This module assembles them
and caches the result: the benchmark runs one process per (model, regime) pair, and reading
the CSV again for each run is wasted work.

The CSV directory holds the PGxLOD extraction (``entities_*.csv``, ``ucpgx_components.csv``,
``ucpgx_links_unique.csv``). It is not part of this repository; ``data.csv_dir`` in the
config points at it, and a symlink under ``Data/`` is enough.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .graph import encode, load_graph, verbalize
from .partition import carve_validation, leakage, make_split

# (relation, family, intra-cluster split), from the strictest relation to the loosest.
# intra_cluster marks the regimes a single cluster occupies at more than 88%: oncology
# through Neoplasms, relatedMatch and related. There the cut runs inside the cluster,
# between groups of identical verbalisation.
REGIMES = [("broadMatch", "pgx", False), ("closeMatch", None, False),
           ("sameAs", None, False),
           ("broadMatch", "onco", True), ("relatedMatch", None, True),
           ("related", None, True)]

DATA_SEED = 2024        # split, validation and query sampling (the benchmark itself)
VAL_RATIO = 0.25        # share of the train groups held out for epoch selection
SPLIT_RATIO = 0.3       # share of the pairs kept for training


def regime_tag(relation: str, family: str | None) -> str:
    return relation + (f"/{family}" if family else "")


def regime_names() -> list[str]:
    return [regime_tag(r, f) for r, f, _ in REGIMES]


def build(csv_dir, seed: int = DATA_SEED, val_ratio: float = VAL_RATIO,
          split_ratio: float = SPLIT_RATIO, logger=None) -> dict:
    """Load the graph and build one split per regime."""
    g = load_graph(Path(csv_dir), logger=logger)
    uris = g.ucpgx
    id_of = {u: i for i, u in enumerate(uris)}
    verbalisation = {u: verbalize(g, u) for u in uris}
    num_nodes, num_relations, triples = encode(g)

    splits, validations = {}, {}
    for relation, family, intra in REGIMES:
        if not g.links.get(relation):
            if logger:
                logger.info(f"regime {relation} absent from the CSV, skipped")
            continue
        sp = make_split(g, relation, family=family, split_ratio=split_ratio, seed=seed,
                        anti_leak=True, merge_keys=verbalisation, intra_cluster=intra)
        sp["stats"]["leak_ratio"] = leakage(sp, lambda u: verbalisation[u])["leak_ratio"]
        tag = regime_tag(relation, family)
        splits[tag] = sp
        validations[tag] = carve_validation(sp, val_ratio, seed)

    return {"uris": uris, "id_of": id_of, "verbalisation": verbalisation,
            "num_nodes": num_nodes, "num_ucpgx": len(uris),
            "num_relations": num_relations, "triples": triples,
            "splits": splits, "validations": validations,
            "seed": seed, "val_ratio": val_ratio, "split_ratio": split_ratio}


def load(csv_dir, seed: int = DATA_SEED, val_ratio: float = VAL_RATIO,
         split_ratio: float = SPLIT_RATIO, cache_dir=None, rebuild: bool = False,
         logger=None) -> dict:
    """Return :func:`build`, from the cache when it is there."""
    if cache_dir is None:
        return build(csv_dir, seed, val_ratio, split_ratio, logger)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"splits_s{seed}_v{val_ratio}_t{split_ratio}.pkl"
    if path.exists() and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)
    data = build(csv_dir, seed, val_ratio, split_ratio, logger)
    tmp = path.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    tmp.rename(path)
    return data


def describe(base: dict) -> str:
    """One line per regime: sizes, split mode and measured leakage."""
    lines = []
    for tag, sp in base["splits"].items():
        s = sp["stats"]
        mode = "intra-cluster" if s["intra_cluster"] else "by cluster"
        lines.append(f"  {tag:16s} train={s['train_pairs']:>7,} test={s['test_pairs']:>7,} "
                     f"({s['pair_test_ratio']:.0%}) queries={s['test_queries']:>6,} "
                     f"{mode:<13} leakage={s['leak_ratio']:.2%}")
    return "\n".join(lines)
