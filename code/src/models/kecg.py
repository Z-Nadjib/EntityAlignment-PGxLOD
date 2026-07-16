"""KECG - Semi-supervised EA via Knowledge Embedding + Cross-graph (Li et al., EMNLP 2019).

Two models sharing the entity table: a Cross-Graph multi-head **diagonal GAT**
(shared over the combined graph, pulls seed pairs together) and a Knowledge-
Embedding **TransE** energy on the triples. Training alternates the two. Faithful
port of the official code (THU-KEG/KECG); the KE loss keeps its sum of the
normalised error vector, as in ``run.py`` of that repository.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_softmax(scores, index, n):
    """Numerically-stable softmax of edge scores grouped by destination node.

    Same helper as in ``alinet.py``: per-group max subtracted before exp;
    groups without edges keep a -inf max, mapped to 0 by nan_to_num.
    """
    mx = scores.new_full((n,), float("-inf")).index_reduce_(0, index, scores, "amax", include_self=True)
    mx = torch.nan_to_num(mx, neginf=0.0)
    s = (scores - mx[index]).exp()
    denom = torch.zeros(n, device=scores.device, dtype=scores.dtype).index_add_(0, index, s)
    return s / (denom[index] + 1e-16)


def scatter_add(src, index, n):
    """Sum rows of ``src`` (E, d) into ``n`` buckets given by ``index`` (E,)."""
    out = torch.zeros((n, src.shape[1]), device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


class DiagGATLayer(nn.Module):
    """Multi-head GAT with a diagonal (per-head element-wise) transform; heads
    averaged; linear propagation; KECG attention = softmax(-LeakyReLU(.))."""

    def __init__(self, dim, n_heads, attn_dropout=0.0, combine="mean"):
        super().__init__()
        self.n_heads = n_heads
        self.combine = combine
        self.w = nn.Parameter(torch.ones(n_heads, dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, dim))
        self.a_src = nn.Parameter(torch.zeros(n_heads, dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.attn_dropout = attn_dropout
        self.out_dim = dim * n_heads if combine == "concat" else dim
        nn.init.xavier_uniform_(self.a_dst); nn.init.xavier_uniform_(self.a_src)

    def forward(self, h, e_dst, e_src):
        n = h.shape[0]
        outs = []
        for hd in range(self.n_heads):
            g = h * self.w[hd]
            sd = (g * self.a_dst[hd]).sum(-1)
            ss = (g * self.a_src[hd]).sum(-1)
            score = -self.leaky(sd[e_dst] + ss[e_src])
            alpha = scatter_softmax(score, e_dst, n)
            if self.attn_dropout > 0 and self.training:
                alpha = F.dropout(alpha, p=self.attn_dropout)
            outs.append(scatter_add(alpha.unsqueeze(-1) * g[e_src], e_dst, n))
        if self.combine == "concat":
            return torch.cat(outs, dim=-1)
        return torch.stack(outs, 0).mean(0)


class KECG(nn.Module):
    """Diagonal-GAT encoder + relation table shared by the CG and KE objectives."""

    def __init__(self, num_entities, num_relations, edge_index, embed_dim=128,
                 n_layers=2, n_heads=2, attn_dropout=0.0, normalize_embeddings=False,
                 instance_normalization=False):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.instance_normalization = instance_normalization
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        nn.init.normal_(self.ent_emb.weight, std=1.0 / math.sqrt(num_entities))
        nn.init.xavier_uniform_(self.rel_emb.weight)
        if self.instance_normalization:
            self.norm = nn.InstanceNorm1d(embed_dim, momentum=0.0, affine=True)
        self.layers = nn.ModuleList(
            [DiagGATLayer(embed_dim, n_heads, attn_dropout, combine="mean") for _ in range(n_layers)])
        self.register_buffer("e_dst", edge_index[0])
        self.register_buffer("e_src", edge_index[1])
        self.out_dim = embed_dim

    def forward_all(self):
        h = self.ent_emb.weight
        if self.instance_normalization:
            h = self.norm(h.t().unsqueeze(0)).squeeze(0).t()
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h, self.e_dst, self.e_src)
            if i < last:
                h = F.elu(h)
        return F.normalize(h, dim=-1) if self.normalize_embeddings else h


def kecg_cg_loss(z, e1, e2, neg_l, neg_r, margin):
    """Cross-graph triplet margin loss on GAT embeddings (both directions)."""
    d_pos = torch.norm(z[e1] - z[e2], p=2, dim=-1)
    d_neg_r = torch.norm(z[e1] - z[neg_r], p=2, dim=-1)
    d_neg_l = torch.norm(z[neg_l] - z[e2], p=2, dim=-1)
    return 0.5 * (F.relu(margin + d_pos - d_neg_r).mean() + F.relu(margin + d_pos - d_neg_l).mean())


def kecg_ke_loss(z, rel_emb, pos, neg, margin):
    """TransE margin-ranking on the GAT outputs (replicates the repo's normalised-error sum)."""
    if neg.shape[0] != pos.shape[0]:
        pos = pos.repeat(neg.shape[0] // pos.shape[0], 1)
    x_pos = F.normalize(z[pos[:, 0]] + rel_emb(pos[:, 1]) - z[pos[:, 2]], p=2, dim=-1)
    x_neg = F.normalize(z[neg[:, 0]] + rel_emb(neg[:, 1]) - z[neg[:, 2]], p=2, dim=-1)
    y = torch.ones(x_pos.size(0), 1, device=z.device)
    return F.margin_ranking_loss(x_pos.sum(1).view(-1, 1), x_neg.sum(1).view(-1, 1), y, margin=margin)
