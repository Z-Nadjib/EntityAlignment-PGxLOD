"""AliNet - Gated Multi-hop Neighborhood Aggregation (Sun et al., AAAI 2020) for PGxLOD.

Full-batch GNN combining a 1-hop GCN pass, a 2-hop attention pass, and a learned
gate; linear propagation (no ReLU), JK-concat of layer outputs, L2-normalised.
A relation-aware (TransE-style) loss anchors every entity structurally, which matters
on PGxLOD where most entities are in no training pair.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_softmax(scores, index, n):
    """Numerically-stable softmax of edge scores grouped by destination node.

    ``scores`` (E,) are attention logits, ``index`` (E,) the group id (dst node)
    of each edge, ``n`` the number of groups. The per-group max is subtracted
    before exp (stability); nodes with no edge keep -inf max, mapped to 0.
    """
    mx = scores.new_full((n,), float("-inf")).index_reduce_(0, index, scores, "amax", include_self=True)
    mx = torch.nan_to_num(mx, neginf=0.0)
    s = (scores - mx[index]).exp()
    denom = torch.zeros(n, device=scores.device, dtype=scores.dtype).index_add_(0, index, s)
    return s / (denom[index] + 1e-16)


class AliNetLayer(nn.Module):
    """1-hop GCN + 2-hop attention, combined by a gate. Linear (no ReLU)."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.a1 = nn.Parameter(torch.zeros(out_dim))
        self.a2 = nn.Parameter(torch.zeros(out_dim))
        self.gate = nn.Linear(out_dim, out_dim)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.a1.view(1, -1)); nn.init.xavier_uniform_(self.a2.view(1, -1))

    def forward(self, h, adj1, e_dst, e_src):
        n = h.shape[0]
        g1 = torch.sparse.mm(adj1, self.W1(h))
        wh = self.W2(h)
        if e_dst.numel() > 0:
            s1 = (wh * self.a1).sum(-1); s2 = (wh * self.a2).sum(-1)
            score = self.leaky(s1[e_dst] + s2[e_src])
            alpha = scatter_softmax(score, e_dst, n)
            if self.dropout > 0 and self.training:
                alpha = F.dropout(alpha, p=self.dropout)
            g2 = torch.zeros_like(g1)
            chunk = 500_000
            for s in range(0, e_src.shape[0], chunk):
                sl = slice(s, s + chunk)
                g2 = g2.index_add(0, e_dst[sl], alpha[sl].unsqueeze(-1) * wh[e_src[sl]])
        else:
            g2 = torch.zeros_like(g1)
        gate = torch.sigmoid(self.gate(g1))
        return gate * g1 + (1.0 - gate) * g2


class AliNet(nn.Module):
    """Stack of gated 1-hop/2-hop layers; forward_all() gives JK-concat embeddings."""

    def __init__(self, num_entities, adj1, two_hop, embed_dim=300,
                 layer_dims=(300, 300), dropout=0.0, normalize_embeddings=True,
                 num_relations=0):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.feat_dropout = dropout
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        nn.init.normal_(self.ent_emb.weight, std=1.0)          # unit-scale (not xavier!)
        with torch.no_grad():
            self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data, dim=-1)

        dims = [embed_dim] + list(layer_dims)
        self.layers = nn.ModuleList(
            [AliNetLayer(dims[i], dims[i + 1], dropout=dropout) for i in range(len(layer_dims))])
        self.register_buffer("_adj1", adj1)
        self.register_buffer("e_dst", two_hop[0])
        self.register_buffer("e_src", two_hop[1])
        self.out_dim = sum(layer_dims)
        self.rel_emb = nn.Embedding(num_relations, self.out_dim) if num_relations else None
        if self.rel_emb is not None:
            nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward_all(self):
        h = self.ent_emb.weight
        reps = []
        for layer in self.layers:
            if self.feat_dropout > 0 and self.training:
                h = F.dropout(h, p=self.feat_dropout)
            h = layer(h, self._adj1, self.e_dst, self.e_src)
            reps.append(h)
        z = torch.cat(reps, dim=-1)
        return F.normalize(z, dim=-1) if self.normalize_embeddings else z


def alinet_align_loss(z, e1, e2, neg_l, neg_r, margin):
    """Margin-ranking alignment loss on L2 distance (both corruption sides)."""
    d_pos = torch.norm(z[e1] - z[e2], p=2, dim=-1)
    d_neg_r = torch.norm(z[e1] - z[neg_r], p=2, dim=-1)
    d_neg_l = torch.norm(z[neg_l] - z[e2], p=2, dim=-1)
    return 0.5 * (F.relu(margin + d_pos - d_neg_r).mean() + F.relu(margin + d_pos - d_neg_l).mean())


def alinet_relation_loss(z, rel_emb, pos, neg_t, margin):
    """Relation-aware (TransE) loss ``||z_h + r - z_t||`` - anchors every entity."""
    h = z[pos[:, 0]]; r = rel_emb(pos[:, 1]); t = z[pos[:, 2]]
    pos_s = torch.norm(h + r - t, p=2, dim=-1)
    neg_s = torch.norm(h + r - z[neg_t], p=2, dim=-1)
    return F.relu(margin + pos_s - neg_s).mean()
