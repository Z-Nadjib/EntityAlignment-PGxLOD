"""Leak-free train/test partition of one regime, and the validation carved out of train.

Three sources of leakage are handled here.

1. Transitivity. ``A sameAs B`` and ``B sameAs C`` in train give ``A sameAs C``. The cut is
   therefore made by connected component: a whole group goes to one side.
2. Textual twins. 37% of the UCPGx link the same entities and produce the same
   verbalisation. The components are computed over ``relation | sameAs | identical
   signature``, not over the relation alone.
3. Giant cluster. On ``broadMatch`` the oncology family forms one component holding 92% of
   the pairs. The components are therefore computed per family, and a cluster that would
   overshoot the target by more than ``tolerance`` is refused.

In the regimes a single group occupies (onco 92%, relatedMatch 95.5%, related 88.2%), the cut
runs inside the group:

* oriented relation: the query nodes are partitioned. Every test query is unseen, and what
  stays visible on both sides is the popularity of the target;
* symmetric relation: the pairs are partitioned, both directions staying on the same side.

In both cases the partition works on groups of identical signature, never on single nodes:
separating two textual twins would bring the leakage back to 92.6%.

``carve_validation`` takes the validation out of train with the same grouping key as the
split; otherwise the intra-cluster mode would send a single block to validation.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .graph import Graph, pair_family


def connected_components(pairs) -> dict:
    """Union-find with path compression: node -> component id."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {x: find(x) for x in list(parent)}


def make_split(g: Graph, relation: str, split_ratio: float = 0.3, seed: int = 2024,
               anti_leak: bool = True, merge_keys: dict | None = None,
               family: str | None = None, tolerance: float = 0.10,
               intra_cluster: bool = False, max_cluster_for_split: float = 0.5) -> dict:
    """The gold split of one relation, optionally restricted to one family.

    Returns ::

        { "relation", "oriented", "family",
          "gold_full":  q -> set(every known partner, undirected, for the filtering),
          "gold_train": q -> set(train partners),
          "gold_test":  q -> set(test partners),
          "test_queries": [ ... ],
          "equiv": q -> set(sameAs twins),
          "stats": {...} }

    ``anti_leak`` merges, for the components of the split only, the pairs of the relation,
    those of ``sameAs`` and the nodes of identical signature. The default signature is the
    set of components; ``merge_keys`` gives another one, the flat verbalisation here.

    ``family`` restricts the components to the pairs of that family. On the whole
    ``broadMatch`` the oncology cluster holds 92% of the pairs and no cut is possible;
    restricted to ``pgx``, the largest cluster weighs 1.8%.

    Shuffled clusters are stacked until ``split_ratio`` of the pairs are in train, and a
    cluster that would overshoot the target by more than ``tolerance`` is refused: a cluster
    larger than the target always goes to test.
    """
    oriented = g.oriented.get(relation, False)
    all_pairs = list(g.links.get(relation, []))
    pairs = ([p for p in all_pairs if pair_family(g, p[0], p[1]) == family]
             if family else all_pairs)

    # merge edges, used for the components only
    split_pairs = list(pairs)
    if anti_leak:
        if relation != "sameAs":
            split_pairs += list(g.links.get("sameAs", []))
        # identical signature: the members of a group are chained together
        if merge_keys is None:
            merge_keys = {u: frozenset(g.comp.get(u, ())) for u in g.ucpgx}
        by_key: dict = defaultdict(list)
        for u, k in merge_keys.items():
            by_key[k].append(u)
        for members in by_key.values():
            for i in range(1, len(members)):
                split_pairs.append((members[0], members[i]))

    comp_of = connected_components(split_pairs)
    # a node never seen in split_pairs is its own component
    for a, _ in pairs:
        comp_of.setdefault(a, a)

    # pairs per cluster, to balance the assignment in pairs
    pairs_per_cluster: dict = defaultdict(int)
    for a, b in pairs:
        pairs_per_cluster[comp_of[a]] += 1

    clusters = sorted(pairs_per_cluster)
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    target_train_pairs = split_ratio * len(pairs)
    ceiling = target_train_pairs * (1.0 + tolerance)
    train_clusters, acc = set(), 0
    for c in clusters:
        if acc >= target_train_pairs:
            break
        if acc + pairs_per_cluster[c] > ceiling and acc > 0:
            continue                             # overshoot refused
        train_clusters.add(c)
        acc += pairs_per_cluster[c]

    # intra-cluster cut, for the regimes a single group occupies. The partition key is a
    # group of identical signature, never a single node.
    big = {c for c, n in pairs_per_cluster.items()
           if n > max_cluster_for_split * len(pairs)} if intra_cluster else set()
    sig = merge_keys if merge_keys is not None else {}
    intra_train_keys: set = set()
    if big:
        rng2 = np.random.default_rng(seed + 1)
        if oriented:
            # key = signature of the query, else the query itself
            keys = sorted({sig.get(a, a) for a, b in pairs if comp_of.get(a) in big},
                          key=str)
        else:
            # key = canonical pair of signatures, both directions together
            keys = sorted({tuple(sorted((str(sig.get(a, a)), str(sig.get(b, b)))))
                           for a, b in pairs if comp_of.get(a) in big})
        perm = rng2.permutation(len(keys))
        k = int(round(split_ratio * len(keys)))
        intra_train_keys = {keys[i] for i in perm[:k]}

    def intra_key(a, b):
        return (sig.get(a, a) if oriented
                else tuple(sorted((str(sig.get(a, a)), str(sig.get(b, b))))))

    def is_train(a, b):
        """Side of a pair: by cluster, or inside the cluster for the large blocks."""
        if comp_of.get(a) in big:
            return intra_key(a, b) in intra_train_keys
        return comp_of.get(a) in train_clusters

    gold_full = defaultdict(set)
    gold_train = defaultdict(set)
    gold_test = defaultdict(set)
    for a, b in all_pairs:                       # filtering: every known partner
        gold_full[a].add(b)
        gold_full[b].add(a)                      # undirected for the filtering
    for a, b in pairs:
        bucket = gold_train if is_train(a, b) else gold_test
        bucket[a].add(b)
        if not oriented:
            bucket[b].add(a)

    test_queries = sorted(gold_test.keys())

    # sameAs twins, masked when the target relation is not sameAs
    equiv: dict = {}
    if relation != "sameAs":
        sa_comp = connected_components(list(g.links.get("sameAs", [])))
        groups = defaultdict(set)
        for n, r in sa_comp.items():
            groups[r].add(n)
        equiv = {n: (groups[r] - {n}) for n, r in sa_comp.items()}

    train_pairs = int(sum(len(v) for v in gold_train.values()) / (1 if oriented else 2))
    test_pairs = int(sum(len(v) for v in gold_test.values()) / (1 if oriented else 2))
    biggest = max(pairs_per_cluster.values()) if pairs_per_cluster else 0
    max_share = biggest / max(1, len(pairs))
    stats = {
        "relation": relation, "family": family, "oriented": oriented,
        "gold_pairs": len(pairs), "clusters": len(clusters),
        "train_clusters": len(train_clusters),
        "test_clusters": len(clusters) - len(train_clusters),
        "test_queries": len(test_queries),
        "train_pairs": train_pairs, "test_pairs": test_pairs,
        "pair_test_ratio": round(test_pairs / max(1, train_pairs + test_pairs), 3),
        "max_cluster_share": round(max_share, 3),
        # a cluster holding more than half of the pairs makes the cut by cluster
        # arbitrary; the cut then runs inside it, and the leakage is measured
        "splittable": bool(max_share <= max_cluster_for_split),
        "intra_cluster": bool(big),
        "intra_cluster_mode": ("queries" if oriented else "pairs") if big else None,
        "anti_leak": bool(anti_leak),
    }

    return {"relation": relation, "family": family, "oriented": oriented,
            "gold_full": dict(gold_full), "gold_train": dict(gold_train),
            "gold_test": dict(gold_test), "test_queries": test_queries,
            "equiv": equiv, "stats": stats,
            # cluster of every node, to cut train again without touching the test
            "comp_of": comp_of, "train_clusters": train_clusters,
            # what carve_validation needs to replay the same grouping: in the
            # intra-cluster mode comp_of puts everything in one block
            "_intra": bool(big), "_sig": sig if big else None}


def carve_validation(split: dict, val_ratio: float = 0.25, seed: int = 7) -> dict:
    """Take a validation out of train, with the same anti-leak guarantees.

    ``val_ratio`` of the train groups are held out; the model fits on the rest, the best
    state is chosen on the validation, and the test is read once.

    Returns a split whose ``gold_train`` holds the fitting groups and ``gold_test`` the
    validation groups. ``gold_full``, used for the filtering, is unchanged.
    """
    # The grouping key must be the one of the split itself: in the intra-cluster mode,
    # comp_of puts every pair in a single component, and grouping on it would send the
    # whole block to validation.
    comp_of = split["comp_of"]
    intra, sig = split.get("_intra"), split.get("_sig") or {}
    oriented = split.get("oriented", False)

    def group_of(a, b):
        if not intra:
            return comp_of.get(a, a)
        if oriented:
            return ("q", sig.get(a, a))
        return ("p", tuple(sorted((str(sig.get(a, a)), str(sig.get(b, b))))))

    per_cluster: dict = defaultdict(int)
    for a, bs in split["gold_train"].items():
        for b in bs:
            per_cluster[group_of(a, b)] += 1
    clusters = sorted(per_cluster)
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    # same overshoot refusal as in make_split: without it a large group takes the whole
    # validation budget
    total = sum(per_cluster.values())
    target = val_ratio * total
    ceiling = target * 1.10
    val_clusters, acc = set(), 0
    for c in clusters:
        if acc >= target:
            break
        if acc + per_cluster[c] > ceiling and acc > 0:
            continue
        val_clusters.add(c)
        acc += per_cluster[c]

    fit, val = defaultdict(set), defaultdict(set)
    for a, bs in split["gold_train"].items():
        for b in bs:
            (val if group_of(a, b) in val_clusters else fit)[a].add(b)
    out = dict(split)
    out["gold_train"] = dict(fit)
    out["gold_test"] = dict(val)
    out["test_queries"] = sorted(val.keys())
    out["is_validation"] = True
    return out


def leakage(split: dict, verbalize_fn) -> dict:
    """Share of the test pairs whose pair of verbalisations already appears in train.

    A non-regression check: it must be zero with ``anti_leak=True``.
    """
    train_texts = set()
    for a, bs in split["gold_train"].items():
        ta = verbalize_fn(a)
        for b in bs:
            train_texts.add((ta, verbalize_fn(b)))
    tot = leak = 0
    for a, bs in split["gold_test"].items():
        ta = verbalize_fn(a)
        for b in bs:
            tot += 1
            if (ta, verbalize_fn(b)) in train_texts:
                leak += 1
    return {"test_pairs": tot, "leaked": leak,
            "leak_ratio": (leak / tot) if tot else 0.0}
