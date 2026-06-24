"""Embedding tables scored with the tie-aware metrics of :mod:`.metrics`.

:func:`.metrics.evaluate` takes any scorer honouring ``score_fn(idx) -> (b, N)``; the
embedding table of every model is wrapped in such a scorer.

* the rank handles ties, ``1 + #{>} + #{=}/2``. Counting only the strictly better ranked
  candidates gives a perfect rank to a constant scorer;
* the candidates are the UCPGx and the query itself is masked.

The ranks are reported at cosine and at CSLS, the score the configurations tune.

The top-k decision F1 chooses its threshold on the validation set, with negatives drawn
from one half of the partner-less UCPGx, and applies it unchanged to the test split, with
negatives drawn from the other half. Two rules keep it a measure of the split it is read on:

* a query is correct only when one of its partners *of that split* is in its top-k, and its
  partners known before that split (fitting pairs for validation, train pairs for the test)
  are masked, as in ``multiref``;
* the decision score is the cosine, whatever the ranking score. CSLS computes ``r_T`` over the
  positive queries only, so it lowers the score of their partners and not the one of the
  negatives' neighbours: on CSLS the positives score below the negatives and no threshold
  separates them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M

VALIDATION_QUERIES = 1500       # queries kept per validation pass
TEST_QUERIES = 3000             # queries kept for the final test pass
QUERY_SEED = 2024               # query sampling


class Scorer:
    """``score_fn(idx) -> (b, N)`` over the UCPGx, by cosine or CSLS.

    CSLS (Lample et al., 2018) corrects hubness: ``2.cos - r_T(c) - r_S(q)`` where
    ``r_T(c)`` is the mean similarity of candidate ``c`` to its ``k`` nearest queries and
    ``r_S(q)`` the mean similarity of the query to its ``k`` nearest candidates. ``r_T`` is
    computed once over the queries of the regime; ``r_S`` is computed row by row, so the
    scorer stays usable on queries outside that set, which the P/R/F1 block needs for its
    negatives.

    Queries and candidates live in the same table, so an entity is both. Its similarity to
    itself (1 with a single table) is left out of both neighbourhoods: kept in, it would
    raise ``r_T`` of exactly the candidates that are also queries, that is the partners of
    the other queries in a symmetric regime, and push the right answers down.
    """

    def __init__(self, table: torch.Tensor, query_index, metric: str = "cosine",
                 csls_k: int = 3, chunk: int = 1024):
        self.metric = metric
        self.k = int(csls_k)
        self.device = table.device
        self.queries = F.normalize(table.detach().float(), dim=-1)
        self.candidates = self.queries
        self.r_t = self._r_t(query_index, chunk) if metric == "csls" else None

    def _r_t(self, query_index, chunk: int) -> torch.Tensor:
        """Mean similarity of every candidate to its ``k`` nearest queries."""
        q = torch.as_tensor(np.asarray(query_index, dtype=np.int64), device=self.device)
        k = min(self.k, len(q) - 1)
        top = None
        for start in range(0, len(q), chunk):
            cos = self._similarities(q[start:start + chunk], without_self=True)
            top = cos if top is None else torch.cat([top, cos], 0)
            top = top.topk(min(k, top.shape[0]), dim=0).values
        return top.mean(dim=0)

    def _similarities(self, q: torch.Tensor, without_self: bool) -> torch.Tensor:
        cos = self.queries[q] @ self.candidates.t()
        if without_self:
            cos[torch.arange(len(q), device=self.device), q] = float("-inf")
        return cos

    def __call__(self, idx) -> np.ndarray:
        q = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
        cos = self._similarities(q, without_self=False)
        if self.r_t is None:
            return cos.cpu().numpy()
        others = cos.clone()
        others[torch.arange(len(q), device=self.device), q] = float("-inf")
        r_s = others.topk(min(self.k, cos.shape[1] - 1), dim=1).values.mean(dim=1, keepdim=True)
        return (2.0 * cos - self.r_t.unsqueeze(0) - r_s).cpu().numpy()


def sample_queries(split: dict, max_queries: int, seed: int = QUERY_SEED) -> list:
    """At most ``max_queries`` queries of ``split``, drawn with a fixed seed.

    The draw depends only on the split, so every model of a regime is judged on the same
    queries.
    """
    queries = split["test_queries"]
    if len(queries) <= max_queries:
        return list(queries)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(queries, max_queries, replace=False))


def _scorer(z, base: dict, split: dict, metric: str, csls_k: int, max_queries: int):
    """The sampled queries of ``split`` and the scorer built on them.

    ``z`` is the embedding table of a model, or any object with a
    ``scorer(index, metric, csls_k)`` method, such as the bags of neighbours of the
    structure-only baseline (:mod:`.baseline`).
    """
    queries = sample_queries(split, max_queries)
    index = [base["id_of"][u] for u in queries if u in base["id_of"]]
    if isinstance(z, torch.Tensor):
        scorer = Scorer(z[:base["num_ucpgx"]], index, metric=metric, csls_k=csls_k)
    else:
        scorer = z.scorer(index, metric=metric, csls_k=csls_k)
    return queries, index, scorer


def evaluate_embeddings(z, base: dict, split: dict, metric: str = "cosine",
                        csls_k: int = 3, max_queries: int = TEST_QUERIES,
                        chunk: int = 256) -> dict:
    """The ranking blocks (filtered, multiref) for one table on one split."""
    sub = dict(split)
    sub["test_queries"], index, scorer = _scorer(z, base, split, metric, csls_k, max_queries)
    res = M.evaluate(scorer, base["uris"], base["id_of"], sub, hits_at=(1, 5, 10),
                     chunk=chunk)
    res["splittable"] = bool(split["stats"]["splittable"])
    res["intra_cluster"] = bool(split["stats"].get("intra_cluster"))
    res["leak_ratio"] = split["stats"].get("leak_ratio")
    res["n_queries"] = len(index)
    return res


# --------------------------------------------------------------------------- #
#  Top-k decision F1, threshold chosen on validation
# --------------------------------------------------------------------------- #
def negative_pools(base: dict, split: dict, seed: int = QUERY_SEED):
    """The UCPGx without any partner, split in two disjoint halves (validation, test)."""
    has_partner = set(split["gold_full"])
    pool = np.array([i for i, u in enumerate(base["uris"]) if u not in has_partner],
                    dtype=np.int64)
    pool = np.random.default_rng(seed).permutation(pool)
    half = len(pool) // 2
    return pool[:half], pool[half:]


def decision_scores(scorer, base: dict, split: dict, queries: list, negatives: np.ndarray,
                    tol_k: int = 10, seed: int = QUERY_SEED, chunk: int = 256):
    """Top-1 score and correctness of the top-k decision, for positives and negatives.

    A positive is correct when one of its partners in ``split["gold_test"]`` is in its top
    ``tol_k``; its partners in ``split["gold_train"]``, known before that split, are masked,
    as in ``multiref``. The query itself and its ``sameAs`` twins that are not partners are
    masked. A negative is never correct. As many negatives as positives are drawn from
    ``negatives``.
    """
    uris, id_of = base["uris"], base["id_of"]
    equiv = split.get("equiv", {})
    gold, known = split["gold_test"], split["gold_train"]
    pos = [id_of[q] for q in queries if q in id_of]
    rng = np.random.default_rng(seed)
    neg = list(rng.choice(negatives, min(len(pos), len(negatives)), replace=False))
    todo = pos + neg
    k = min(max(1, tol_k), len(uris))
    scores, is_pos, correct = [], [], []
    for start in range(0, len(todo), chunk):
        idx = todo[start:start + chunk]
        sims = scorer(idx)
        for r, qi in enumerate(idx):
            positive = start + r < len(pos)
            row = sims[r].copy()
            row[qi] = -np.inf
            partners = {id_of[u] for u in gold.get(uris[qi], ()) if u in id_of} if positive else set()
            if positive:
                for u in known.get(uris[qi], ()):
                    ui = id_of.get(u)
                    if ui is not None and ui not in partners:
                        row[ui] = -np.inf
            for t in equiv.get(uris[qi], ()):
                ti = id_of.get(t)
                if ti is not None and ti not in partners:
                    row[ti] = -np.inf
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            scores.append(float(row[top[0]]))
            is_pos.append(positive)
            correct.append(bool(partners) and any(int(t) in partners for t in top))
    return np.asarray(scores), np.asarray(is_pos), np.asarray(correct)


def f1_at(scores, is_pos, correct, threshold: float) -> dict:
    """Precision, recall and F1 of the decision ``score >= threshold``."""
    decided = scores >= threshold
    tp = int((decided & correct).sum())
    fp = int((decided & ~correct).sum())
    n_pos = int(is_pos.sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_pos if n_pos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Precision": prec, "Recall": rec, "F1": f1}


def classification(z, base: dict, regime: str, tol_k: int = 10) -> dict:
    """Top-k decision F1 on the test split, at the threshold that is best on validation.

    The decision score is the cosine (see the module docstring). On validation a positive
    is judged on its validation partners, its fitting partners masked; on the test, on its
    test partners, its train partners masked.
    """
    split, validation = base["splits"][regime], base["validations"][regime]
    neg_val, neg_test = negative_pools(base, split)

    v_queries, _, v_scorer = _scorer(z, base, validation, "cosine", 0, TEST_QUERIES)
    v = decision_scores(v_scorer, base, validation, v_queries, neg_val, tol_k)
    val_f1, _, _, threshold = M.best_f1(*v)

    t_queries, _, t_scorer = _scorer(z, base, split, "cosine", 0, TEST_QUERIES)
    t = decision_scores(t_scorer, base, split, t_queries, neg_test, tol_k)
    out = f1_at(*t, threshold)
    out.update({"score": "cosine", "threshold": threshold, "threshold_from": "validation",
                "validation_F1": val_f1, "tol_k": int(tol_k),
                "n_pos": int(t[1].sum()), "n_neg": int((~t[1]).sum())})
    return out


class ValidationEvaluator:
    """What the trainers evaluate against during training.

    Every trainer calls :meth:`evaluate` at the same point of its loop, with the embedding
    table of the current epoch. It measures on the validation set and keeps a copy of the
    best table, so the stopping epoch is chosen on validation and the test split is only
    read once, by :func:`final_evaluation`, on that best table.
    """

    def __init__(self, base: dict, regime: str, select: str = "filtered",
                 metric: str = "cosine", csls_k: int = 3,
                 max_queries: int = VALIDATION_QUERIES):
        self.base = base
        self.regime = regime
        self.split = base["validations"][regime]
        self.select = select if select in ("filtered", "multiref") else "filtered"
        self.metric = metric
        self.csls_k = csls_k
        self.max_queries = max_queries
        self.best_score = -1.0
        self.best_table = None

    def evaluate(self, z: torch.Tensor, hits_at=(1, 5, 10)) -> dict:
        res = evaluate_embeddings(z, self.base, self.split, metric=self.metric,
                                  csls_k=self.csls_k, max_queries=self.max_queries)
        blocks = {"filtered": res["filtered"], "multiref": res["multiref"]}
        score = blocks[self.select]["MRR"]
        if score > self.best_score:
            self.best_score = score
            self.best_table = z[:self.base["num_ucpgx"]].detach().clone().cpu()
        return blocks

def final_evaluation(evaluator: ValidationEvaluator, cfg, dataset, run_dir, history: dict,
                     device, logger, minutes: float = None) -> dict:
    """Read the test split once, ranks at cosine and at CSLS, F1 at cosine, and write
    ``result.json``."""
    base, regime = evaluator.base, evaluator.regime
    split = base["splits"][regime]
    if evaluator.best_table is None:
        raise RuntimeError("no epoch was evaluated, check eval.every and train.epochs")
    z = evaluator.best_table.to(device)
    csls_k = int(cfg.eval.get("csls_k", 3))
    clf_tol_k = int(cfg.eval.get("clf_tol_k", 10))

    logger.info("Final evaluation on the test split, read once")
    final = {}
    for metric in ("cosine", "csls"):
        res = evaluate_embeddings(z, base, split, metric=metric, csls_k=csls_k,
                                  max_queries=TEST_QUERIES)
        final[metric] = res
        f, m = res["filtered"], res["multiref"]
        logger.info(f"  [{metric:>6}] filtMRR={f['MRR']:.4f} H@1={f['Hit@1']:.4f} "
                    f"H@10={f['Hit@10']:.4f} n={f['n']} | mrefMRR={m['MRR']:.4f} "
                    f"H@1={m['Hit@1']:.4f}")
    c = final["classification"] = classification(z, base, regime, clf_tol_k)
    logger.info(f"  [cosine] F1={c['F1']:.4f} P={c['Precision']:.4f} R={c['Recall']:.4f} "
                f"(threshold {c['threshold']:.3f} from validation, "
                f"validation F1={c['validation_F1']:.4f})")

    out = {"model": str(cfg.experiment.get("model")), "regime": regime,
           "config": Path(cfg._config_path).name,
           "seed": int(cfg.experiment.seed),
           "data_seed": int(base["seed"]),
           "selection": {"block": evaluator.select, "on": "validation carved from train",
                         "best_validation_MRR": evaluator.best_score,
                         "best_epoch": history.get("best_epoch")},
           "minutes": None if minutes is None else round(minutes, 1),
           "split": split["stats"],
           "pairs": {"fit": int(len(dataset.train_pairs)),
                     "validation": int(len(dataset.val_pairs)),
                     "test": int(split["stats"]["test_pairs"])},
           "final": final}
    path = Path(run_dir) / "result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.save(Path(run_dir) / "embeddings_ucpgx.npy",
            evaluator.best_table.numpy())
    logger.info(f"Wrote {path}")
    return out
