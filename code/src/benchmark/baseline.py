"""The structure-only baseline: bags of neighbours compared by cosine, nothing learned.

The bag of neighbours of a UCPGx is the set of its (predicate, component) pairs, that is
the edges of the graph that leave it. Two UCPGx are scored by the cosine of their bags,
``|A & B| / sqrt(|A| |B|)``: the more pairs they share, the closer they are.

The bags go through the scoring of the embedding tables (:mod:`.scoring`): ranks at cosine
and at CSLS, decision F1 at cosine, on the same splits and the same queries as the models.
The shared pairs are counted on integers, through the inverted index ``pair -> UCPGx``, so
two candidates with the same bag get exactly the same score and their tie is averaged as
for any other scorer.
"""
from __future__ import annotations

import numpy as np
import torch

from .scoring import TEST_QUERIES, Scorer, classification, evaluate_embeddings


class Bags:
    """The bag of neighbours of every UCPGx, indexed both ways: unit -> pairs, pair -> units."""

    def __init__(self, triples: np.ndarray, num_ucpgx: int):
        triples = np.asarray(triples, dtype=np.int64)
        triples = triples[triples[:, 0] < num_ucpgx]
        _, pair = np.unique(triples[:, 1:], axis=0, return_inverse=True)
        cells = np.unique(np.stack([triples[:, 0], pair.reshape(-1)], 1), axis=0)
        unit, pair = cells[:, 0], cells[:, 1]           # one (unit, pair) cell per bag entry

        self.num_ucpgx = int(num_ucpgx)
        self.num_pairs = int(pair.max()) + 1 if len(pair) else 0
        self.size = np.bincount(unit, minlength=self.num_ucpgx)
        self.norm = np.sqrt(self.size.astype(np.float64))
        # cells are sorted by unit: the pairs of unit u are unit_pairs[unit_ptr[u]:unit_ptr[u+1]]
        self.unit_ptr = np.concatenate([[0], np.cumsum(self.size)])
        self.unit_pairs = pair
        order = np.argsort(pair, kind="stable")
        self.pair_units = unit[order]
        self.pair_ptr = np.concatenate([[0], np.cumsum(np.bincount(pair, minlength=self.num_pairs))])

    def shared(self, u: int) -> np.ndarray:
        """Number of pairs that every UCPGx shares with ``u``."""
        pairs = self.unit_pairs[self.unit_ptr[u]:self.unit_ptr[u + 1]]
        if len(pairs) == 0:
            return np.zeros(self.num_ucpgx, dtype=np.int64)
        members = np.concatenate([self.pair_units[self.pair_ptr[p]:self.pair_ptr[p + 1]]
                                  for p in pairs])
        return np.bincount(members, minlength=self.num_ucpgx)

    def cosine(self, idx) -> np.ndarray:
        """``(b, N)`` cosine between the bags of ``idx`` and the bag of every UCPGx."""
        out = np.zeros((len(idx), self.num_ucpgx), dtype=np.float32)
        for r, u in enumerate(idx):
            denom = self.norm[u] * self.norm
            np.divide(self.shared(int(u)), denom, out=out[r], where=denom > 0, casting="unsafe")
        return out

    def scorer(self, index, metric: str = "cosine", csls_k: int = 3) -> "BagScorer":
        return BagScorer(self, index, metric=metric, csls_k=csls_k)


class BagScorer(Scorer):
    """:class:`.scoring.Scorer` on the bags of neighbours instead of an embedding table."""

    def __init__(self, bags: Bags, query_index, metric: str = "cosine", csls_k: int = 3,
                 chunk: int = 256):
        self.bags = bags
        self.metric = metric
        self.k = int(csls_k)
        self.device = torch.device("cpu")
        self.r_t = self._r_t(query_index, chunk) if metric == "csls" else None

    def _similarities(self, q: torch.Tensor, without_self: bool) -> torch.Tensor:
        cos = torch.from_numpy(self.bags.cosine(q.tolist()))
        if without_self:
            cos[torch.arange(len(q)), q] = float("-inf")
        return cos


def evaluate(bags: Bags, base: dict, regime: str, csls_k: int = 3, tol_k: int = 10) -> dict:
    """The test evaluation of the baseline on one regime, laid out as the ``result.json``
    of a model run (ranks at cosine and at CSLS, decision F1 at cosine)."""
    split = base["splits"][regime]
    final = {metric: evaluate_embeddings(bags, base, split, metric=metric, csls_k=csls_k,
                                         max_queries=TEST_QUERIES)
             for metric in ("cosine", "csls")}
    final["classification"] = classification(bags, base, regime, tol_k)
    return {"model": "baseline", "regime": regime, "data_seed": int(base["seed"]),
            "csls_k": int(csls_k), "split": split["stats"],
            "pairs": {"test": int(split["stats"]["test_pairs"])}, "final": final}
