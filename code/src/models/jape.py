"""JAPE on PGxLOD - Joint Attribute-Preserving Embedding (Sun et al., ISWC 2017).

The paper that introduced DBP15K. Two channels:

* **Structure Embedding (SE)** - a TransE energy ``f(h, r, t) = ||h + r - t||``
  over the triples in one space. Seed alignments are encoded by giving the
  entities of a TRAIN pair the **same id** (merged), so TransE propagates the
  alignment to the other entities. Margin loss + corrupted
  negatives; entity embeddings L2-normalised.
* **Attribute Embedding (AE)** - refines SE with an attribute bag.

PGxLOD adaptation: there are **no attributes**, so only the **SE channel** is used
(this is the legitimate structure-only JAPE). The merged-seed graph is built in
``data.build_jape_merged``; at eval the entity table is read through the merge map
(``z = ent_emb[merge_map]``) so the standard PGxLOD multi-reference evaluator scores
JAPE on the same filtered / multiref / classification metrics as every other model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class JAPE(nn.Module):
    """SE channel: plain TransE tables trained on the merged-pair graph.

    ``squared`` and ``normalize_relations`` switch to the official energy
    (``||h + r - t||^2``, both tables L2-normalised at every pass).
    """

    def __init__(self, num_entities, num_relations, embed_dim=200,
                 init="xavier", normalize_embeddings=True, normalize_relations=False,
                 squared=False):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.normalize_relations = normalize_relations
        self.squared = squared
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        if init == "trunc_normal":             # official: truncated_normal(1/sqrt(dim))
            std = 1.0 / math.sqrt(embed_dim)
            nn.init.trunc_normal_(self.ent_emb.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.trunc_normal_(self.rel_emb.weight, std=std, a=-2 * std, b=2 * std)
        elif init == "xavier":
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            nn.init.normal_(self.ent_emb.weight, std=0.02)
            nn.init.normal_(self.rel_emb.weight, std=0.02)

    def _ent(self, idx):
        e = self.ent_emb(idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    def triple_score(self, triples):
        """``||h + r - t||_2`` (or its square)."""
        h = self._ent(triples[:, 0])
        r = self.rel_emb(triples[:, 1])
        if self.normalize_relations:
            r = F.normalize(r, dim=-1)
        t = self._ent(triples[:, 2])
        d = h + r - t
        return (d * d).sum(-1) if self.squared else torch.norm(d, p=2, dim=-1)

    def encode(self, idx):
        return self._ent(idx)

    @torch.no_grad()
    def forward_all(self, merge_map: torch.Tensor) -> torch.Tensor:
        """Full (num_entities, dim) table read through the seed-merge map, for eval."""
        w = self.ent_emb.weight[merge_map]
        return F.normalize(w, dim=-1) if self.normalize_embeddings else w


def jape_se_loss(model, pos, neg, margin):
    """TransE margin-ranking loss for SE (negatives = corrupted triples)."""
    pos_s = model.triple_score(pos)
    neg_s = model.triple_score(neg)
    pos_s = pos_s.repeat(neg.shape[0] // pos.shape[0]) if neg.shape[0] != pos.shape[0] else pos_s
    return F.relu(margin + pos_s - neg_s).mean()
