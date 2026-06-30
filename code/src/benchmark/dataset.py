"""The split of one regime, turned into the :class:`~src.data.PGxLOD` container.

``train_pairs`` holds the fitting pairs, that is the train split minus its validation share,
and ``val_pairs`` the validation pairs. The test split never enters the container: it is
read once, after training, by :func:`.scoring.final_evaluation`.
"""
from __future__ import annotations

import numpy as np

from ..data import PGxLOD


def _pairs(gold: dict, id_of: dict, oriented: bool) -> np.ndarray:
    """A ``query -> partners`` mapping to a (P, 2) array of indices.

    An oriented relation keeps its direction. A symmetric one is written both ways by
    ``make_split``, so only one occurrence per pair is kept.
    """
    seen, out = set(), []
    for a, partners in gold.items():
        ia = id_of.get(a)
        if ia is None:
            continue
        for b in partners:
            ib = id_of.get(b)
            if ib is None:
                continue
            if not oriented:
                key = (ia, ib) if ia <= ib else (ib, ia)
                if key in seen:
                    continue
                seen.add(key)
            out.append((ia, ib))
    return np.asarray(out, dtype=np.int64) if out else np.zeros((0, 2), np.int64)


def build_dataset(base: dict, regime: str) -> PGxLOD:
    """Build the dataset of one regime."""
    if regime not in base["validations"]:
        raise ValueError(f"unknown regime {regime!r}, known: {list(base['validations'])}")
    validation = base["validations"][regime]
    id_of = base["id_of"]
    oriented = bool(validation["oriented"])

    fit = _pairs(validation["gold_train"], id_of, oriented)
    val = _pairs(validation["gold_test"], id_of, oriented)
    candidates = np.arange(base["num_ucpgx"], dtype=np.int64)

    return PGxLOD(
        num_entities=base["num_nodes"],
        num_relations=base["num_relations"],
        triples=base["triples"],
        train_pairs=fit,
        val_pairs=val,
        candidate_ids=candidates,
        # the entities a BootEA / NAEA bootstrapping round may pair up: every UCPGx without
        # a fitting label.
        unlabeled_ids=np.setdiff1d(candidates, fit.reshape(-1)),
    )
