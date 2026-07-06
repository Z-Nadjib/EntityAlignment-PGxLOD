"""The GCN-Align model (structure channel) for PGxLOD entity alignment.

GCN-Align (Wang, Lv, Lan, Zhang, Li - "Cross-lingual Knowledge Graph Alignment
via Graph Convolutional Networks", EMNLP 2018) embeds entities with a **2-layer
GCN whose weights are shared across the graph(s)**, so structurally-equivalent
entities get similar embeddings. The adjacency is functionality-weighted
(``data.build_gcnalign_adj``). Alignment is trained with a **margin-based loss on
L1 distance** between seed-pair embeddings, negatives drawn from all entities.

On PGxLOD the same SE model runs over the single graph. Deviation from the
official code: the layers carry a weight matrix (the official SE layers use
``transform=False``, i.e. ``A X`` only); ``n_layers``, ``activation`` and
``self_loop_weight`` are set in the configs, which use a single layer without self-loops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """A GCN layer ``H' = act(A_hat H W)`` with the (pre-normalised) sparse adjacency A_hat."""

    def __init__(self, in_dim, out_dim, act=True, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.act = act

    def forward(self, h, adj):
        h = torch.sparse.mm(adj, h @ self.weight)
        if self.bias is not None:
            h = h + self.bias
        return F.relu(h) if self.act else h


class GCNAlign(nn.Module):
    """Shared GCN encoder over the fixed adjacency; one embedding row per entity."""

    def __init__(self, num_entities, adj, embed_dim=300, n_layers=2,
                 dropout=0.0, activation=False, normalize_embeddings=True):
        super().__init__()
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        # shared GCN layers; final layer is linear (no activation), like GCN-Align SE
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GraphConvolution(embed_dim, embed_dim, act=(activation and i < n_layers - 1)))
        self.register_buffer("_adj", adj)
        self.out_dim = embed_dim

    def forward_all(self):
        h = self.ent_emb.weight
        for layer in self.layers:
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout)
            h = layer(h, self._adj)
        # L2-normalise so the L1 margin loss operates on a bounded scale (else the
        # margin is trivially satisfied by the huge raw distances => no learning)
        return F.normalize(h, dim=-1) if self.normalize_embeddings else h


def gcnalign_loss(z, e1, e2, neg_l, neg_r, margin):
    """Margin-based alignment loss on **L1** distance (both corruption directions).

    ``[ ||z_e1 - z_e2||_1 + gamma - ||z_e1 - z_negR||_1 ]_+``  (+ symmetric left side).
    Index tensors share length (``e1``/``e2`` repeated ``k`` times to match the negatives).
    """
    d_pos = torch.norm(z[e1] - z[e2], p=1, dim=-1)
    d_neg_r = torch.norm(z[e1] - z[neg_r], p=1, dim=-1)
    d_neg_l = torch.norm(z[neg_l] - z[e2], p=1, dim=-1)
    return 0.5 * (F.relu(margin + d_pos - d_neg_r).mean() + F.relu(margin + d_pos - d_neg_l).mean())
