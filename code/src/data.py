"""The PGxLOD graph as the models consume it, plus the model-specific graph structures.

PGxLOD is a single graph: 50,435 UCPGx (pharmacogenomic relationship) nodes linked to
33,747 components (drugs, genetic factors, phenotypes) by 13 predicates. The task is
entity resolution inside that graph: for a UCPGx, rank the other UCPGx and find the ones
that denote the same (or a broader, a related, ...) relationship. There is one embedding
table of ``num_entities`` rows, and a pair ``(a, b)`` is read directly in it.

The container is filled by :func:`src.benchmark.build_dataset` from the leak-free split of
one regime. The models never read the disk.

Ids: UCPGx hold ``0..U-1`` in the order of the graph, components ``U..N-1``. Row ``i`` of
any embedding table is therefore the UCPGx ``i``, and the candidates of the ranking are
``0..U-1``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Container
# --------------------------------------------------------------------------- #
@dataclass
class PGxLOD:
    """Everything the models and trainers need for one regime.

    ``train_pairs`` are the fitting pairs, the only labels a model sees. ``val_pairs`` are
    the validation pairs, kept for reference: the epoch selection goes through
    ``evaluation``. The test split is not in here.
    """

    num_entities: int                  # N, UCPGx + components
    num_relations: int                 # R, component predicates
    triples: np.ndarray                # (M, 3) UCPGx -> component edges, one copy
    train_pairs: np.ndarray            # (S, 2) fitting pairs
    val_pairs: np.ndarray              # (V, 2) validation pairs
    candidate_ids: np.ndarray          # UCPGx ids, the ranking candidates
    unlabeled_ids: np.ndarray          # UCPGx outside the fitting pairs, BootEA / NAEA bootstrap pool
    evaluation: object = None          # src.benchmark.scoring.ValidationEvaluator

    @property
    def entities(self) -> np.ndarray:
        return np.arange(self.num_entities, dtype=np.int64)

    def summary(self) -> dict:
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "triples": len(self.triples),
            "train_pairs": len(self.train_pairs),
            "val_pairs": len(self.val_pairs),
            "candidates": len(self.candidate_ids),
            "unlabeled_pool": len(self.unlabeled_ids),
        }


# --------------------------------------------------------------------------- #
#  GCN-Align : functionality-weighted, symmetrically-normalised adjacency
# --------------------------------------------------------------------------- #
def build_gcnalign_adj(data: PGxLOD, self_loop_weight: float = 1.0):
    """Build GCN-Align's structure adjacency (Wang et al., EMNLP 2018).

    Each relation r gets a *functionality* ``fun(r)=#heads/#triples`` and inverse
    functionality ``ifun(r)=#tails/#triples``. For a triple ``(h, r, t)`` the edge
    weights accumulate ``M[h,t] += max(ifun(r), 0.3)`` and ``M[t,h] += max(fun(r), 0.3)``.
    Self-loops (weight ``self_loop_weight``; 0 disables) are added, then the matrix
    is symmetrically normalised ``D^{-1/2} (M+wI) D^{-1/2}``.
    """
    triples = data.triples
    cnt, head, tail = {}, {}, {}
    for h, r, t in triples:
        r = int(r)
        if r not in cnt:
            cnt[r] = 0; head[r] = set(); tail[r] = set()
        cnt[r] += 1; head[r].add(int(h)); tail[r].add(int(t))
    r2f = {r: len(head[r]) / cnt[r] for r in cnt}      # functionality
    r2if = {r: len(tail[r]) / cnt[r] for r in cnt}     # inverse functionality

    N = data.num_entities
    M = {}
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        M[(h, t)] = M.get((h, t), 0.0) + max(r2if[r], 0.3)
        M[(t, h)] = M.get((t, h), 0.0) + max(r2f[r], 0.3)
    if self_loop_weight:
        for i in range(N):
            M[(i, i)] = M.get((i, i), 0.0) + self_loop_weight

    rows = np.fromiter((i for (i, _j) in M), dtype=np.int64, count=len(M))
    cols = np.fromiter((j for (_i, j) in M), dtype=np.int64, count=len(M))
    vals = np.fromiter(M.values(), dtype=np.float64, count=len(M))
    deg = np.zeros(N, dtype=np.float64)
    np.add.at(deg, rows, vals)                          # row sums
    inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    norm_vals = (inv_sqrt[rows] * vals * inv_sqrt[cols]).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(norm_vals), (N, N)
    ).coalesce()


# --------------------------------------------------------------------------- #
#  KECG : undirected graph edges for the GAT
# --------------------------------------------------------------------------- #
def build_kecg_graph(data: PGxLOD):
    """Edges for KECG's GAT (relation types ignored), undirected, with self-loops.

    Returns ``edge_index`` ``(2, E) = [dst, src]`` (attention aggregates ``src`` into ``dst``).
    """
    N = data.num_entities
    seen = set()
    for h, _r, t in data.triples:
        h, t = int(h), int(t)
        if h != t:
            seen.add((h, t)); seen.add((t, h))
    for i in range(N):
        seen.add((i, i))
    e = np.fromiter((x for edge in seen for x in edge), dtype=np.int64, count=2 * len(seen))
    e = e.reshape(-1, 2)
    return torch.from_numpy(e.T.copy())


# --------------------------------------------------------------------------- #
#  AliNet : 1-hop normalised adjacency + capped 2-hop edges
# --------------------------------------------------------------------------- #
def build_alinet_graph(data: PGxLOD, max_two_hop: int = 10, seed: int = 0):
    """Graph structures AliNet needs (relation types ignored).

    Returns ``adj1`` (sparse N x N symmetric-normalised adjacency with self-loops,
    for 1-hop GCN) and ``two_hop`` ``[dst, src]`` (capped, sampled 2-hop edges,
    excluding 1-hop neighbours and self, for the attention aggregation).
    """
    rng = random.Random(seed)
    N = data.num_entities
    nbrs = [set() for _ in range(N)]
    for h, _r, t in data.triples:
        h, t = int(h), int(t)
        if h != t:
            nbrs[h].add(t); nbrs[t].add(h)

    rows, cols = [], []
    for i in range(N):
        rows.append(i); cols.append(i)
        for j in nbrs[i]:
            rows.append(i); cols.append(j)
    rows = np.asarray(rows, dtype=np.int64); cols = np.asarray(cols, dtype=np.int64)
    deg = np.asarray([len(nbrs[i]) + 1 for i in range(N)], dtype=np.float64)
    inv_sqrt = 1.0 / np.sqrt(deg)
    vals = (inv_sqrt[rows] * inv_sqrt[cols]).astype(np.float32)
    adj1 = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals), (N, N)).coalesce()

    dst, src = [], []
    for i in range(N):
        one = nbrs[i]
        if not one:
            continue
        two = set()
        for j in sorted(one)[:64]:
            two.update(sorted(nbrs[j])[:64])
        two.discard(i); two -= one
        if not two:
            continue
        two = sorted(two)
        if len(two) > max_two_hop:
            two = rng.sample(two, max_two_hop)
        for k in two:
            dst.append(i); src.append(k)
    two_hop = torch.tensor([dst, src], dtype=torch.long) if dst else torch.zeros((2, 0), dtype=torch.long)
    return adj1, two_hop


# --------------------------------------------------------------------------- #
#  MRAEA / RREA : meta-relation-aware sparse graph structures
# --------------------------------------------------------------------------- #
def build_mraea_graph(data: PGxLOD):
    """Build the sparse structures MRAEA/RREA's attention layer consumes.

      * directed edges: triple ``(h, r, t)`` -> edge ``(h, t)`` with relation ``r``
        and inverse edge ``(t, h)`` with relation ``r + R`` (table size ``2R``);
      * ``edge_rel`` sparse ``(E, 2R)`` mapping each unique edge to its relation(s),
        value ``1/#relations``;
      * ``rel_adj`` sparse ``(N, 2R)`` node->relation incidence;
      * ``ent_adj`` sparse ``(N, N)`` neighbour incidence + self-loops.
    """
    N = data.num_entities
    R = data.num_relations

    edge_rels: dict[tuple, list] = {}
    rel_pairs = set()
    for h, r, t in data.triples:
        h, r, t = int(h), int(r), int(t)
        edge_rels.setdefault((h, t), []).append(r)
        edge_rels.setdefault((t, h), []).append(R + r)
        rel_pairs.add((t, r))
        rel_pairs.add((h, R + r))

    edges = sorted(edge_rels.keys())
    eid = {e: i for i, e in enumerate(edges)}
    adj_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    r_rows, r_cols, r_vals = [], [], []
    for e in edges:
        rels = edge_rels[e]
        w = 1.0 / len(rels)
        for rid in rels:
            r_rows.append(eid[e]); r_cols.append(rid); r_vals.append(w)
    edge_rel_index = torch.tensor([r_rows, r_cols], dtype=torch.long)
    edge_rel_val = torch.tensor(r_vals, dtype=torch.float32)

    rel_adj_index = torch.tensor(sorted(rel_pairs), dtype=torch.long).t().contiguous()
    ent_pairs = set(edges) | {(i, i) for i in range(N)}
    ent_adj_index = torch.tensor(sorted(ent_pairs), dtype=torch.long).t().contiguous()

    return {
        "num_nodes": N, "num_rels": 2 * R, "num_edges": len(edges),
        "adj_index": adj_index,
        "edge_rel_index": edge_rel_index, "edge_rel_val": edge_rel_val,
        "rel_adj_index": rel_adj_index, "ent_adj_index": ent_adj_index,
    }


# --------------------------------------------------------------------------- #
#  Negative sampling helpers
# --------------------------------------------------------------------------- #
class TripleSampler:
    """Mini-batch iterator over a (mutable) set of triples with corrupted negatives.

    A negative replaces the head or the tail with a uniformly random entity, or, once
    :meth:`set_candidates` has been given a table, with one of that entity's nearest
    neighbours (the eps-truncated sampling of BootEA).

    Layout of a batch: ``neg`` is ``pos.repeat(neg, 1)`` corrupted, so row ``j * B + i`` of
    ``neg`` is a corruption of row ``i`` of ``pos``.
    """

    def __init__(self, data: PGxLOD, device, neg: int = 5):
        self.device = device
        self.neg = neg
        self.num_entities = data.num_entities
        self.cand = None
        self.set_triples(data.triples)

    def set_triples(self, triples):
        if isinstance(triples, np.ndarray):
            triples = torch.from_numpy(triples)
        self.triples = triples.to(self.device).long()

    def set_candidates(self, cand):
        """``cand`` : (N, C) LongTensor of nearest neighbours per entity, or None."""
        self.cand = cand.to(self.device) if cand is not None else None

    def __len__(self):
        return len(self.triples)

    def _replacement(self, ent_ids):
        if self.cand is None:
            return torch.randint(self.num_entities, ent_ids.shape, device=self.device)
        cand = self.cand[ent_ids]                                    # (n, C)
        sel = torch.randint(cand.shape[1], (cand.shape[0], 1), device=self.device)
        return torch.gather(cand, 1, sel).squeeze(1)

    def batches(self, batch_size: int, shuffle: bool = True):
        M = len(self.triples)
        order = torch.randperm(M, device=self.device) if shuffle else torch.arange(M, device=self.device)
        for s in range(0, M, batch_size):
            pos = self.triples[order[s:s + batch_size]]              # (B,3)
            B = pos.shape[0]
            neg = pos.repeat(self.neg, 1)                            # (B*n,3)
            corrupt_head = torch.rand(B * self.neg, device=self.device) < 0.5
            orig = torch.where(corrupt_head, neg[:, 0], neg[:, 2])
            repl = self._replacement(orig)
            neg[:, 0] = torch.where(corrupt_head, repl, neg[:, 0])
            neg[:, 2] = torch.where(~corrupt_head, repl, neg[:, 2])
            yield pos, neg


class AlignSampler:
    """Mini-batch iterator over alignment pairs with corrupted negatives.

    For a positive pair ``(q, t)`` the right side is corrupted (``neg_r``) and the left
    side too (``neg_l``), giving negatives in both directions. ``set_pairs`` lets a trainer
    inject bootstrapped pairs.

    **Hard negatives.** Once ``set_hard_negatives`` has been called, negatives are drawn
    from each pair's pre-computed nearest candidates instead of uniformly. A candidate that
    equals the gold side is replaced by a random entity.
    """

    def __init__(self, data: PGxLOD, device, neg: int = 5, pairs=None):
        self.device = device
        self.neg = neg
        self.num_entities = data.num_entities
        self.hard_r = None          # (S, C) nearest candidates of the left side
        self.hard_l = None          # (S, C) nearest candidates of the right side
        self.set_pairs(data.train_pairs if pairs is None else pairs)

    def set_pairs(self, pairs):
        if isinstance(pairs, np.ndarray):
            pairs = torch.from_numpy(pairs)
        self.pairs = pairs.to(self.device).long()
        self.hard_r = self.hard_l = None    # invalidate stale hard-negative tables

    def set_hard_negatives(self, hard_r, hard_l):
        """``hard_r``/``hard_l``: (S, C) LongTensors aligned with ``self.pairs``."""
        self.hard_r = hard_r.to(self.device)
        self.hard_l = hard_l.to(self.device)

    def __len__(self):
        return len(self.pairs)

    def _sample_hard(self, table, rows, n, gold):
        """Pick ``n`` candidates per row from ``table[rows]``, avoiding ``gold``."""
        cand = table[rows]                                       # (B, C)
        sel = torch.randint(cand.shape[1], (cand.shape[0], n), device=self.device)
        out = torch.gather(cand, 1, sel).reshape(-1)            # (B*n,)
        clash = out == gold
        if clash.any():
            out[clash] = torch.randint(self.num_entities, (int(clash.sum()),), device=self.device)
        return out

    def batches(self, batch_size: int, shuffle: bool = True):
        S = len(self.pairs)
        order = torch.randperm(S, device=self.device) if shuffle else torch.arange(S, device=self.device)
        for s in range(0, S, batch_size):
            idx = order[s:s + batch_size]
            pos = self.pairs[idx]                                # (B,2)
            B = pos.shape[0]
            n = self.neg
            e1 = pos[:, 0].repeat_interleave(n)
            e2 = pos[:, 1].repeat_interleave(n)
            if self.hard_r is not None and self.hard_l is not None:
                neg_r = self._sample_hard(self.hard_r, idx, n, e2)
                neg_l = self._sample_hard(self.hard_l, idx, n, e1)
            else:
                neg_r = torch.randint(self.num_entities, (B * n,), device=self.device)
                neg_l = torch.randint(self.num_entities, (B * n,), device=self.device)
            yield pos, (e1, e2, neg_l, neg_r)


# --------------------------------------------------------------------------- #
#  Bootstrapping : CSLS mutual nearest neighbours over the unlabelled pool
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mutual_nearest(z: torch.Tensor, pool: torch.Tensor, k: int, chunk: int = 4096):
    """CSLS mutual nearest neighbours inside ``pool``, an entity never matched to itself.

    ``z`` holds the (N, d) embeddings; ``pool`` the entity ids to match among themselves.

    Returns ``(left, right, cos)``: entity ids of the mutual pairs and their cosine. The
    similarity matrix is never materialised: two passes over row blocks, the first for the
    CSLS neighbourhoods, the second for the row and column arg-maxima.
    """
    zl = torch.nn.functional.normalize(z[pool], dim=-1)
    zr = zl
    P = pool.numel()
    if P < 2:
        empty = pool.new_empty(0)
        return empty, empty, zl.new_empty(0)
    k = min(k, P - 1)
    dev = zl.device

    def block(s):
        cos = zl[s:s + chunk] @ zr.t()
        rows = torch.arange(s, s + cos.shape[0], device=dev)
        cos[rows - s, rows] = float("-inf")                    # never the entity itself
        return cos

    r_s = torch.empty(P, device=dev)
    top_t = None
    for s in range(0, P, chunk):
        cos = block(s)
        r_s[s:s + cos.shape[0]] = cos.topk(k, dim=1).values.mean(1)
        top_t = cos.topk(k, dim=0).values if top_t is None else \
            torch.cat([top_t, cos], 0).topk(k, dim=0).values
    r_t = top_t.mean(0)

    best_r = torch.empty(P, dtype=torch.long, device=dev)
    best_cos = torch.empty(P, device=dev)
    col_val = torch.full((P,), float("-inf"), device=dev)
    col_arg = torch.zeros(P, dtype=torch.long, device=dev)
    for s in range(0, P, chunk):
        cos = block(s)
        csls = 2 * cos - r_t.unsqueeze(0) - r_s[s:s + cos.shape[0]].unsqueeze(1)
        v, a = csls.max(dim=1)
        best_r[s:s + cos.shape[0]] = a
        best_cos[s:s + cos.shape[0]] = cos.gather(1, a.unsqueeze(1)).squeeze(1)
        cv, ca = csls.max(dim=0)
        better = cv > col_val
        col_val = torch.where(better, cv, col_val)
        col_arg = torch.where(better, ca + s, col_arg)

    rows = torch.arange(P, device=dev)
    mutual = col_arg[best_r] == rows
    li, ri = rows[mutual], best_r[mutual]
    return pool[li], pool[ri], best_cos[mutual]


# --------------------------------------------------------------------------- #
#  BootEA : alignment by swapping
# --------------------------------------------------------------------------- #
class Swapper:
    """Generate BootEA *aligned triples* by swapping labelled entities.

    For a labelled pair ``(a, b)`` we substitute ``b`` for ``a`` in each of
    ``a``'s triples and vice-versa, e.g. ``(a, r, t) -> (b, r, t)``. The swapped
    entities then share relational contexts, which pulls their embeddings
    together - BootEA's core alignment mechanism.

    Per-entity triple lists are precomputed once; :meth:`generate` is then cheap
    enough to be re-run every bootstrapping round on the (growing) labelled set.
    """

    def __init__(self, data: PGxLOD):
        self.triples = data.triples
        n = data.num_entities
        self.as_head = [[] for _ in range(n)]
        self.as_tail = [[] for _ in range(n)]
        for i, (h, _r, t) in enumerate(self.triples):
            self.as_head[int(h)].append(i)
            self.as_tail[int(t)].append(i)

    def generate(self, labeled_pairs, cap_per_role: int = 100) -> np.ndarray:
        """Return an ``(K, 3)`` array of swapped (h, r, t) triples for the pairs."""
        T = self.triples
        cap = cap_per_role if cap_per_role and cap_per_role > 0 else None   # 0 = every triple
        out = []
        for a, b in labeled_pairs:
            a, b = int(a), int(b)
            for src, dst in ((a, b), (b, a)):          # dst takes src's place
                for ti in self.as_head[src][:cap]:
                    out.append((dst, T[ti, 1], T[ti, 2]))
                for ti in self.as_tail[src][:cap]:
                    out.append((T[ti, 0], T[ti, 1], dst))
        if not out:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(out, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  NAEA helper : padded neighbour tensors for the attention aggregator
# --------------------------------------------------------------------------- #
def build_neighbors(data: PGxLOD, max_neighbors: int, seed: int = 0):
    """Padded neighbour tensors for NAEA's neighbourhood-aware attention.

    For every entity ``e`` we collect its neighbours from all triples:
      * out-edge ``(e, r, t)`` -> neighbour ``t`` via ``r``, sign = -1 (msg ``t - r``)
      * in-edge  ``(h, r, e)`` -> neighbour ``h`` via ``r``, sign = +1 (msg ``h + r``)
    Entities with > ``max_neighbors`` neighbours are sub-sampled; fewer are
    zero-padded and masked. Returns ``(neigh_ent, neigh_rel, neigh_sign, mask)``
    each ``(num_entities, max_neighbors)``.
    """
    rng = random.Random(seed)
    N = data.num_entities
    adj: list = [[] for _ in range(N)]
    for h, r, t in data.triples:
        adj[int(h)].append((int(r), int(t), -1))   # out-edge: e=h, message t - r
        adj[int(t)].append((int(r), int(h), +1))   # in-edge:  e=t, message h + r

    K = max_neighbors
    neigh_ent = np.zeros((N, K), dtype=np.int64)
    neigh_rel = np.zeros((N, K), dtype=np.int64)
    neigh_sign = np.zeros((N, K), dtype=np.float32)
    mask = np.zeros((N, K), dtype=bool)
    for e in range(N):
        nbrs = adj[e]
        if len(nbrs) > K:
            nbrs = rng.sample(nbrs, K)
        for j, (r, ne, s) in enumerate(nbrs):
            neigh_rel[e, j] = r
            neigh_ent[e, j] = ne
            neigh_sign[e, j] = s
            mask[e, j] = True
    return (torch.from_numpy(neigh_ent), torch.from_numpy(neigh_rel),
            torch.from_numpy(neigh_sign), torch.from_numpy(mask))


# --------------------------------------------------------------------------- #
#  JAPE helper : merged-seed graph (SE structure channel, no attributes)
# --------------------------------------------------------------------------- #
def build_jape_merged(data: PGxLOD):
    """Merge the entities of each training pair into a single id (JAPE's SE bridge).

    Aligned TRAIN pairs share an id so TransE propagates the alignment to the other
    entities; validation and test entities stay separate. Returns
    ``(merged_triples, merge_map)`` where ``merge_map`` (size ``num_entities``) sends every
    entity to its canonical id (the smallest of its merged group) and ``merged_triples``
    is ``data.triples`` rewritten through it. At evaluation the embedding of entity ``e``
    is read as ``ent_emb[merge_map[e]]``.
    """
    N = data.num_entities
    parent = np.arange(N, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in data.train_pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            hi, lo = (ra, rb) if ra > rb else (rb, ra)
            parent[hi] = lo

    merge_map = np.array([find(i) for i in range(N)], dtype=np.int64)
    triples = data.triples.copy()
    triples[:, 0] = merge_map[triples[:, 0]]
    triples[:, 2] = merge_map[triples[:, 2]]
    return triples, merge_map
