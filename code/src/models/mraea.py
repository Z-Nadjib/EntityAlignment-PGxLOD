"""MRAEA - Meta Relation Aware Entity Alignment (Mao et al., WSDM 2020) for PGxLOD.

Reference: https://dl.acm.org/doi/10.1145/3336191.3371804 ; Keras: github.com/MaoXinn/MRAEA.
Plain-torch port of the official Keras code. RREA's predecessor:
same sparse graph (``data.build_mraea_graph``) and same L1 margin loss, but a
**meta-relation-aware** additive attention (relation embedding + self + neighbour,
leakyrelu) instead of RREA's Householder reflection.

Initial feature: ``h0 = relu([ mean(neigh+self ent_emb) || mean(rel emb) ])``;
``depth`` shared GAT steps; edge logit ``a_r*rel(i,j) + a_s*h_i + a_n*h_j`` softmaxed
over neighbours; JK-concat of all steps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rrea import rrea_align_loss as mraea_align_loss   # identical L1 margin loss


class MRAEA(nn.Module):
    """Meta-relation-aware GAT encoder; forward() returns all-entity embeddings."""

    def __init__(self, graph: dict, node_hidden=100, rel_hidden=100,
                 depth=2, attn_heads=2, dropout=0.3):
        super().__init__()
        self.N = graph["num_nodes"]
        self.R = graph["num_rels"]
        self.depth = depth
        self.heads = attn_heads
        self.dropout = dropout
        self.ent_F = node_hidden + rel_hidden
        self.out_dim = self.ent_F * (depth + 1)

        self.ent_emb = nn.Embedding(self.N, node_hidden)
        self.rel_emb = nn.Embedding(self.R, rel_hidden)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.k_self = nn.Parameter(torch.empty(attn_heads, self.ent_F, 1))
        self.k_neigh = nn.Parameter(torch.empty(attn_heads, self.ent_F, 1))
        self.k_rel = nn.Parameter(torch.empty(attn_heads, rel_hidden, 1))
        for p in (self.k_self, self.k_neigh, self.k_rel):
            nn.init.xavier_uniform_(p)

        self.register_buffer("adj_index", graph["adj_index"])
        self.register_buffer("edge_rel_index", graph["edge_rel_index"])
        self.register_buffer("edge_rel_val", graph["edge_rel_val"])
        ei = graph["ent_adj_index"]
        self.register_buffer("ent_adj_index", ei)
        self.register_buffer("ent_adj_val", self._row_uniform(ei[0], self.N))
        ri = graph["rel_adj_index"]
        self.register_buffer("rel_adj_index", ri)
        self.register_buffer("rel_adj_val", self._row_uniform(ri[0], self.N))

    @staticmethod
    def _row_uniform(rows, n):
        deg = torch.zeros(n).index_add_(0, rows, torch.ones(rows.numel()))
        return 1.0 / deg[rows].clamp(min=1.0)

    def _sp(self, index, val, shape):
        return torch.sparse_coo_tensor(index, val, shape, device=index.device).coalesce()

    @staticmethod
    def _segment_softmax(att, src, n):
        """Softmax of edge logits within each source-node group (edge order kept).

        Replaces ``torch.sparse.softmax`` over an (N, N) matrix, whose autograd
        densifies and runs out of memory at PGxLOD scale."""
        gmax = torch.full((n,), float("-inf"), device=att.device)
        gmax = gmax.scatter_reduce(0, src, att, reduce="amax", include_self=True)
        att = (att - gmax[src]).exp()
        gsum = torch.zeros(n, device=att.device).index_add_(0, src, att)
        return att / gsum[src].clamp_min(1e-12)

    def forward(self):
        ent_emb, rel_emb = self.ent_emb.weight, self.rel_emb.weight
        N, R2, E = self.N, self.R, self.adj_index.size(1)
        ent_adj = self._sp(self.ent_adj_index, self.ent_adj_val, (N, N))
        rel_adj = self._sp(self.rel_adj_index, self.rel_adj_val, (N, R2))
        edge_rel = self._sp(self.edge_rel_index, self.edge_rel_val, (E, R2))

        ent_features = torch.sparse.mm(ent_adj, ent_emb)
        rel_features = torch.sparse.mm(rel_adj, rel_emb)
        features = F.relu(torch.cat([ent_features, rel_features], dim=-1))
        outputs = [features]

        src, dst = self.adj_index[0], self.adj_index[1]
        for _ in range(self.depth):
            head_feats = []
            for head in range(self.heads):
                rel_proj = rel_emb @ self.k_rel[head]
                rel_score = torch.sparse.mm(edge_rel, rel_proj).squeeze(-1)     # (E,)
                a_self = (features @ self.k_self[head]).squeeze(-1)             # (N,)
                a_neigh = (features @ self.k_neigh[head]).squeeze(-1)           # (N,)
                att_val = F.leaky_relu(rel_score + a_self[src] + a_neigh[dst])  # (E,)
                att = self._segment_softmax(att_val, src, N)                    # softmax over src groups
                agg = torch.zeros(N, self.ent_F, device=features.device)
                agg.index_add_(0, src, features[dst] * att.unsqueeze(-1))       # aggregate neighbours
                head_feats.append(agg)
            features = F.relu(torch.stack(head_feats, 0).mean(0))
            outputs.append(features)

        out = torch.cat(outputs, dim=-1)
        return F.dropout(out, p=self.dropout, training=self.training)
