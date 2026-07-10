"""RREA - Relational Reflection Entity Alignment (Mao et al., CIKM 2020) for PGxLOD.

Reference: https://arxiv.org/abs/2008.07962 ; official Keras: github.com/MaoXinn/RREA.
Plain-torch port of the official Keras code.

Key idea - **relational reflection**: when node ``i`` aggregates a neighbour ``j``
via relation ``r``, the neighbour is reflected across the hyperplane orthogonal to
the unit relation vector ``r_hat`` (Householder, norm/orthogonality-preserving):

    reflect_r(h_j) = h_j - 2 (h_j * r_hat) r_hat

Edge attention logit = ``a * [ h_i || reflect_r(h_j) || r_hat ]`` (softmax over
neighbours, no leakyrelu); the node aggregates the *reflected* neighbours. A shared
``depth``-layer encoder (JK-concat) is run on an entity-based and a relation-based
initial feature, concatenated. Sparse graph = ``data.build_mraea_graph``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RREA(nn.Module):
    """Relational-reflection GAT encoder; forward() returns all-entity embeddings."""

    def __init__(self, graph: dict, node_hidden=100, depth=2, attn_heads=1, dropout=0.3):
        super().__init__()
        self.N = graph["num_nodes"]
        self.R = graph["num_rels"]
        self.depth = depth
        self.heads = attn_heads
        self.dropout = dropout
        self.F = node_hidden
        self.out_dim = 2 * node_hidden * (depth + 1)

        self.ent_emb = nn.Embedding(self.N, node_hidden)
        self.rel_emb = nn.Embedding(self.R, node_hidden)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.attn = nn.Parameter(torch.empty(depth, attn_heads, 3 * node_hidden, 1))
        nn.init.xavier_uniform_(self.attn)

        self.register_buffer("adj_index", graph["adj_index"])            # (2, E)
        self.register_buffer("edge_rel_index", graph["edge_rel_index"])  # (2, M)
        self.register_buffer("edge_rel_val", graph["edge_rel_val"])      # (M,)
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
        """Softmax of edge logits within each source-node group (edge order kept)."""
        gmax = torch.full((n,), float("-inf"), device=att.device)
        gmax = gmax.scatter_reduce(0, src, att, reduce="amax", include_self=True)
        att = (att - gmax[src]).exp()
        gsum = torch.zeros(n, device=att.device).index_add_(0, src, att)
        return att / gsum[src].clamp_min(1e-12)

    def _encode(self, features, rel_emb, edge_rel):
        src, dst = self.adj_index[0], self.adj_index[1]
        features = F.relu(features)
        outputs = [features]
        for l in range(self.depth):
            head_feats = []
            for head in range(self.heads):
                rels = torch.sparse.mm(edge_rel, rel_emb)          # (E, F) per-edge relation
                rels = F.normalize(rels, dim=1)                    # unit relation vector r_hat
                neigh = features[dst]
                slf = features[src]
                bias = (neigh * rels).sum(1, keepdim=True) * rels  # (h_j*r_hat) r_hat
                neigh = neigh - 2.0 * bias                         # reflection
                logit = (torch.cat([slf, neigh, rels], dim=-1)
                         @ self.attn[l, head]).squeeze(-1)
                att = self._segment_softmax(logit, src, self.N)
                agg = torch.zeros(self.N, self.F, device=features.device)
                agg.index_add_(0, src, neigh * att.unsqueeze(-1))
                head_feats.append(agg)
            features = F.relu(torch.stack(head_feats, 0).mean(0))
            outputs.append(features)
        return torch.cat(outputs, dim=-1)                          # (N, F*(depth+1))

    def forward(self):
        ent_emb, rel_emb = self.ent_emb.weight, self.rel_emb.weight
        N, R2, E = self.N, self.R, self.adj_index.size(1)
        ent_adj = self._sp(self.ent_adj_index, self.ent_adj_val, (N, N))
        rel_adj = self._sp(self.rel_adj_index, self.rel_adj_val, (N, R2))
        edge_rel = self._sp(self.edge_rel_index, self.edge_rel_val, (E, R2))

        ent_feature = torch.sparse.mm(ent_adj, ent_emb)
        rel_feature = torch.sparse.mm(rel_adj, rel_emb)
        out = torch.cat([self._encode(ent_feature, rel_emb, edge_rel),
                         self._encode(rel_feature, rel_emb, edge_rel)], dim=-1)
        return F.dropout(out, p=self.dropout, training=self.training)


def rrea_align_loss(emb, quad, gamma):
    """L1 margin loss (MRAEA/RREA): relu(g + d(l,r) - d(l,r-)) + relu(g + d(l,r) - d(l-,r)).

    ``quad`` = ``[B, 4]`` indices ``[l, r, neg_for_l, neg_for_r]``; L1 distances on
    the (unnormalised) embeddings.
    """
    l = emb[quad[:, 0]]; r = emb[quad[:, 1]]
    nl = emb[quad[:, 2]]; nr = emb[quad[:, 3]]

    def d(a, b):
        return (a - b).abs().sum(-1)

    pos = d(l, r)
    loss = F.relu(gamma + pos - d(l, nr)) + F.relu(gamma + pos - d(nl, r))
    return loss.sum() / l.size(0)
