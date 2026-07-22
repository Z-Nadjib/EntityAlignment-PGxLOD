"""NAEA on PGxLOD - Neighbourhood-Aware Entity Alignment (Zhu et al., IJCAI 2019).

NAEA learns entity/relation embeddings with two complementary signals:

1. **Relation-level (knowledge) representation** - a TransE energy
   ``f(h, r, t) = ||h + r - t||`` trained with a margin ranking loss.

2. **Neighbourhood-aware attentional representation** - for an entity ``e`` we
   aggregate its neighbours ``(r_k, e_j)`` with GAT-style attention. Each
   neighbour contributes a *translation-consistent message* ``m = e_j + sign*r_k``
   (``sign=+1`` for in-edges, ``-1`` for out-edges, so ``m`` reconstructs ``e``)::

        alpha_k = softmax_k( LeakyReLU( a^T [ W e || W m_k ] ) )
        e_hat    = sigma( sum_k alpha_k * W m_k )

   The representation used for **alignment** and **evaluation** is the joint
   vector ``z = e + e_hat`` (self + neighbourhood), L2-normalised.

Single embedding table over ``num_entities``. The alignment loss is the BootEA-style
limit-based objective. Bootstrapping (in the trainer) adds confident pseudo-pairs.
Hard negatives are off in the configs: in a many-to-many gold, the nearest neighbour of an
entity is often one of its true partners.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NAEA(nn.Module):
    """TransE tables + neighbour-aware attention encoder (joint z = e + e_hat)."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        neigh_ent: torch.Tensor,
        neigh_rel: torch.Tensor,
        neigh_sign: torch.Tensor,
        neigh_mask: torch.Tensor,
        attn_heads: int = 1,
        attn_dropout: float = 0.0,
        init: str = "xavier",
        normalize_embeddings: bool = True,
        neighbor_message: str = "trans",
    ):
        super().__init__()
        self.dim = embed_dim
        self.normalize_embeddings = normalize_embeddings
        self.neighbor_message = neighbor_message
        self.attn_dropout = attn_dropout

        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)

        self.heads = max(1, attn_heads)
        self.W = nn.Linear(embed_dim, embed_dim * self.heads, bias=False)
        self.a_self = nn.Parameter(torch.zeros(self.heads, embed_dim))
        self.a_neigh = nn.Parameter(torch.zeros(self.heads, embed_dim))
        self.leaky = nn.LeakyReLU(0.2)

        self.register_buffer("neigh_ent", neigh_ent)
        self.register_buffer("neigh_rel", neigh_rel)
        self.register_buffer("neigh_sign", neigh_sign)
        self.register_buffer("neigh_mask", neigh_mask)

        self._init_params(init)

    def _init_params(self, init: str):
        if init == "xavier":
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            nn.init.normal_(self.ent_emb.weight, std=0.02)
            nn.init.normal_(self.rel_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_self)
        nn.init.xavier_uniform_(self.a_neigh)

    # ----- relation-level (TransE) energy ----- #
    def _ent(self, idx):
        e = self.ent_emb(idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    def _rel(self, idx):
        r = self.rel_emb(idx)
        return F.normalize(r, dim=-1) if self.normalize_embeddings else r

    def transe_score(self, triples: torch.Tensor) -> torch.Tensor:
        """``||h + r - t||_2`` for a batch of (h, r, t). Lower = more plausible."""
        h = self._ent(triples[:, 0])
        r = self._rel(triples[:, 1])
        t = self._ent(triples[:, 2])
        return torch.norm(h + r - t, p=2, dim=-1)

    # ----- neighbourhood-aware attentional encoder ----- #
    def encode(self, ent_idx: torch.Tensor) -> torch.Tensor:
        """Joint neighbour-aware representation ``z = e + e_hat`` for ``ent_idx``."""
        flat = ent_idx.reshape(-1)
        B = flat.shape[0]
        H, d = self.heads, self.dim

        e = self.ent_emb(flat)                                   # (B, d)
        ne = self.neigh_ent[flat]                                # (B, K)
        nr = self.neigh_rel[flat]                                # (B, K)
        sg = self.neigh_sign[flat].unsqueeze(-1)                 # (B, K, 1)
        msk = self.neigh_mask[flat]                              # (B, K)

        ent_n = self.ent_emb(ne)                                 # (B, K, d)
        rel_n = self.rel_emb(nr)                                 # (B, K, d)
        msg = ent_n + sg * rel_n if self.neighbor_message == "trans" else ent_n

        Wself = self.W(e).view(B, H, d)                          # (B, H, d)
        Wmsg = self.W(msg).view(B, -1, H, d)                     # (B, K, H, d)

        logit_self = (Wself * self.a_self).sum(-1).unsqueeze(1)  # (B, 1, H)
        logit_neigh = (Wmsg * self.a_neigh).sum(-1)             # (B, K, H)
        logit = self.leaky(logit_self + logit_neigh)            # (B, K, H)
        logit = logit.masked_fill(~msk.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(logit, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)                # isolated entities
        if self.attn_dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.attn_dropout)

        neigh_repr = (alpha.unsqueeze(-1) * Wmsg).sum(1)        # (B, H, d)
        neigh_repr = torch.tanh(neigh_repr).mean(1)            # (B, d)

        z = e + neigh_repr
        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)
        return z.view(*ent_idx.shape, d)

    @torch.no_grad()
    def encode_all(self, ent_ids: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        self.eval()
        return torch.cat([self.encode(ent_ids[s:s + chunk]) for s in range(0, len(ent_ids), chunk)], 0)

    @torch.no_grad()
    def forward_all(self, chunk: int = 4096) -> torch.Tensor:
        """Full (num_entities, dim) neighbour-aware representation for the evaluator."""
        ids = torch.arange(self.ent_emb.num_embeddings, device=self.ent_emb.weight.device)
        return self.encode_all(ids, chunk=chunk)


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #
def margin_ranking_loss(pos_score, neg_score, margin: float) -> torch.Tensor:
    """``mean( relu(margin + pos - neg) )`` where *lower score = better*.

    ``neg_score`` follows the layout of ``TripleSampler``: row ``j * B + i`` is a
    corruption of positive ``i``, so the (n, B) view, transposed, pairs every negative
    with its own positive.
    """
    B = pos_score.shape[0]
    neg_score = neg_score.view(-1, B).t()                   # (B, n)
    return F.relu(margin + pos_score.unsqueeze(1) - neg_score).mean()


def alignment_loss(model: NAEA, e1, e2, neg_l, neg_r,
                   pos_margin: float, neg_margin: float, neg_weight: float = 1.0) -> torch.Tensor:
    """Limit-based (absolute-margin) alignment loss over the joint ``z`` (BootEA-style)."""
    z1 = model.encode(e1)
    z2 = model.encode(e2)
    zr = model.encode(neg_r)
    zl = model.encode(neg_l)
    d_pos = torch.norm(z1 - z2, p=2, dim=-1)
    d_neg_r = torch.norm(z1 - zr, p=2, dim=-1)
    d_neg_l = torch.norm(zl - z2, p=2, dim=-1)
    l_pos = F.relu(d_pos - pos_margin).mean()
    l_neg = 0.5 * (F.relu(neg_margin - d_neg_r).mean() + F.relu(neg_margin - d_neg_l).mean())
    return l_pos + neg_weight * l_neg
