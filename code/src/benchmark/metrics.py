"""Tie-aware ranking metrics with several references per query.

The rank averages over ties ::

    rank(g) = 1 + #{c : sim(c) > sim(g)} + #{c : sim(c) == sim(g), c != g} / 2

Counting only the strictly better candidates would give a perfect rank to a constant scorer,
whose MRR would then be 1.00.

:func:`evaluate` returns two blocks:

* ``filtered``  per pair (q, g), the other known partners of q masked;
* ``multiref``  per query, position of its first test partner, only its train partners
  masked.
"""
from __future__ import annotations

import numpy as np


def _rank(sim_row: np.ndarray, g: int, keep: np.ndarray) -> float:
    """Average rank of column ``g`` among the candidates where ``keep`` is true."""
    sg = sim_row[g]
    above = int(((sim_row > sg) & keep).sum())
    ties = int(((sim_row == sg) & keep).sum())
    if keep[g]:
        ties -= 1                       # do not count itself
    return above + 1 + max(ties, 0) / 2.0


def filtered_ranks_for_query(sim_row, test_gold_cols, ignore_mask) -> list[float]:
    """Filtered rank of every test partner of a query."""
    keep = ~ignore_mask
    return [_rank(sim_row, g, keep) for g in test_gold_cols]


def multiref_rank_for_query(sim_row, test_gold_cols, ignore_mask) -> float:
    """Average position of the first right answer among the test partners of a query.

    With ``a`` candidates scored strictly above the best partner, and a block of ties at its
    score holding ``m`` partners and ``t`` other candidates, the first right answer is on
    average at position ``a + (m + t + 1) / (m + 1)``. With a single partner this is
    ``_rank``. Taking the minimum of the ``_rank`` of each partner would count the other
    tied partners as errors: two partners tied at the top would get 1.5 instead of 1.
    """
    keep = ~ignore_mask
    gold = set(int(g) for g in test_gold_cols)
    best = max(sim_row[g] for g in gold)
    at_best = [g for g in gold if sim_row[g] == best]
    m = len(at_best)
    above = int(((sim_row > best) & keep).sum())
    t = int(((sim_row == best) & keep).sum()) - sum(1 for g in at_best if keep[g])
    return above + (m + t + 1) / (m + 1)


def aggregate(ranks, hits_at=(1, 5, 10)) -> dict:
    """MRR, Hit@k and MeanRank over a list of ranks.

    Hit@k is ``rank <= k`` on the average rank, the expectation of Hit@k under ties, and
    consistent with the MRR averaged the same way.
    """
    if len(ranks) == 0:
        return {"MRR": 0.0, **{f"Hit@{k}": 0.0 for k in hits_at}, "MeanRank": 0.0, "n": 0}
    r = np.asarray(ranks, dtype=np.float64)
    out = {"MRR": float((1.0 / r).mean())}
    for k in hits_at:
        out[f"Hit@{k}"] = float((r <= k).mean())
    out["MeanRank"] = float(r.mean())
    out["n"] = int(len(r))
    return out


def best_f1(scores, is_pos, correct) -> tuple:
    """Sweep a threshold over the top-1 scores and return the best (F1, P, R, threshold)."""
    scores = np.asarray(scores, dtype=np.float64)
    is_pos = np.asarray(is_pos, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-scores)
    s, co = scores[order], correct[order]
    P = int(is_pos.sum())
    tp = fp = 0
    best = (0.0, 0.0, 0.0, 1.0)
    for i in range(len(s)):
        if co[i]:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[0]:
            best = (f1, prec, rec, float(s[i]))
    return best


def evaluate(score_fn, uris, id_of, split, hits_at=(1, 5, 10), chunk=256) -> dict:
    """The ``filtered`` and ``multiref`` blocks of any scorer against a split.

    ``score_fn(query_indices) -> (b, N)`` gives the similarity of a batch of queries to
    every candidate, in the index order of ``id_of``. ``uris`` maps index -> URI, and
    ``split`` is a dict made by :func:`.partition.make_split`.
    """
    N = len(uris)
    gold_full = split["gold_full"]
    gold_train = split["gold_train"]
    gold_test = split["gold_test"]
    equiv = split.get("equiv", {})
    queries = [id_of[q] for q in split["test_queries"] if q in id_of]

    def cols(uri_set):
        return [id_of[u] for u in uri_set if u in id_of]

    filt_ranks, multi_ranks = [], []
    for start in range(0, len(queries), chunk):
        idx = queries[start:start + chunk]
        sims = score_fn(idx)
        for r, qi in enumerate(idx):
            row = sims[r]
            q_uri = uris[qi]
            tg = cols(gold_test.get(q_uri, ()))
            if not tg:
                continue
            twins = cols(equiv.get(q_uri, ()))
            # filtered masks every known partner, multiref only the train ones
            full = set(cols(gold_full.get(q_uri, ()))) | {qi} | set(twins)
            m_full = np.zeros(N, dtype=bool); m_full[list(full)] = True
            train = set(cols(gold_train.get(q_uri, ()))) | {qi} | set(twins)
            m_train = np.zeros(N, dtype=bool); m_train[list(train)] = True
            filt_ranks.extend(filtered_ranks_for_query(row, tg, m_full))
            multi_ranks.append(multiref_rank_for_query(row, tg, m_train))

    return {"filtered": aggregate(filt_ranks, hits_at),
            "multiref": aggregate(multi_ranks, hits_at)}
