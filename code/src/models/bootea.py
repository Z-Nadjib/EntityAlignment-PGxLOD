"""BootEA on PGxLOD - Bootstrapping Entity Alignment with KG Embedding.

BootEA (Sun, Hu, Zhang, Qu - "Bootstrapping Entity Alignment with Knowledge
Graph Embedding", IJCAI 2018, https://www.ijcai.org/proceedings/2018/0611.pdf)
learns alignment-oriented KG embeddings:

1. **Alignment-oriented embedding (AlignE).** A TransE energy
   ``f(h, r, t) = ||h + r - t||`` trained with a **limit-based** objective
   (absolute margins) rather than a relative margin:

       O_e = sum_{tau in D+} [ f(tau) - gamma1 ]+  +  mu * sum_{tau' in D-} [ gamma2 - f(tau') ]+

   Entity embeddings are constrained to the unit sphere (L2-normalised);
   relation embeddings are free. Negatives use **eps-truncated** sampling
   (corrupt with one of the entity's nearest same-KG neighbours).

2. **Alignment by swapping.** For a labelled pair ``(e1, e2)`` BootEA generates
   *aligned triples* by swapping ``e1`` and ``e2`` in each other's triples
   (handled in the trainer / data module). Because the swapped entities then
   share relational contexts, their embeddings are pulled together - BootEA's
   core alignment mechanism, complemented by a limit-based pull on labelled pairs.

3. **Bootstrapping.** An editable, recomputed mutual 1-to-1 matching over the
   unlabelled pool proposes new alignments each round (in the trainer).

The representation used for **alignment** and **evaluation** is simply the
(L2-normalised) entity embedding - no neighbourhood aggregation. Queries and
candidates are read out of the single entity table.

PGxLOD caveat: *hard-negative* mining for the alignment loss is DISABLED in the
configs, because in a many-to-many gold the nearest neighbour of an entity is
often one of its true partners - mining it as a "hard negative" pushes apart real
positives. eps-truncated *triple* negatives stay on.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BootEA(nn.Module):
    """AlignE embedding tables (TransE energy); the trainer adds swapping/bootstrap.

    ``squared`` and ``normalize_relations`` switch to the official energy: the released
    code scores ``||h + r - t||^2`` with both tables L2-normalised at every pass, and its
    margins (0.01 / 2.0) refer to that squared energy.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        init: str = "xavier",
        normalize_embeddings: bool = True,
        normalize_relations: bool = False,
        squared: bool = False,
    ):
        super().__init__()
        self.dim = embed_dim
        self.out_dim = embed_dim
        self.normalize_embeddings = normalize_embeddings
        self.normalize_relations = normalize_relations
        self.squared = squared
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        self._init_params(init)

    def _init_params(self, init: str):
        if init == "trunc_normal":             # official: truncated_normal(1/sqrt(dim))
            std = 1.0 / math.sqrt(self.dim)
            nn.init.trunc_normal_(self.ent_emb.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.trunc_normal_(self.rel_emb.weight, std=std, a=-2 * std, b=2 * std)
        elif init == "xavier":
            nn.init.xavier_uniform_(self.ent_emb.weight)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            nn.init.normal_(self.ent_emb.weight, std=0.02)
            nn.init.normal_(self.rel_emb.weight, std=0.02)

    # ------------------------------------------------------------------ #
    #  Embeddings
    # ------------------------------------------------------------------ #
    def _ent(self, idx):
        e = self.ent_emb(idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    def _rel(self, idx):
        r = self.rel_emb(idx)
        return F.normalize(r, dim=-1) if self.normalize_relations else r

    def triple_score(self, triples: torch.Tensor) -> torch.Tensor:
        """``||h + r - t||`` (or its square) for a batch of (h, r, t). Lower = more plausible."""
        h = self._ent(triples[:, 0])
        r = self._rel(triples[:, 1])
        t = self._ent(triples[:, 2])
        d = h + r - t
        return (d * d).sum(-1) if self.squared else torch.norm(d, p=2, dim=-1)

    def encode(self, ent_idx: torch.Tensor) -> torch.Tensor:
        """Representation used for alignment / evaluation: the (normalised) entity embedding."""
        e = self.ent_emb(ent_idx)
        return F.normalize(e, dim=-1) if self.normalize_embeddings else e

    @torch.no_grad()
    def encode_all(self, ent_ids: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        self.eval()
        outs = [self.encode(ent_ids[s:s + chunk]) for s in range(0, len(ent_ids), chunk)]
        return torch.cat(outs, 0)

    def forward_all(self) -> torch.Tensor:
        """Full (num_entities, dim) representation for the PGxLOD evaluator."""
        w = self.ent_emb.weight
        return F.normalize(w, dim=-1) if self.normalize_embeddings else w


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #
def limit_based_triple_loss(pos_score, neg_score, pos_margin: float,
                            neg_margin: float, neg_weight: float) -> torch.Tensor:
    """AlignE limit-based objective (absolute margins).

      * positives pulled **below** ``pos_margin`` : ``[ f(tau) - gamma1 ]+``
      * negatives pushed **above** ``neg_margin`` : ``[ gamma2 - f(tau') ]+``
    """
    l_pos = F.relu(pos_score - pos_margin).mean()
    l_neg = F.relu(neg_margin - neg_score).mean()
    return l_pos + neg_weight * l_neg


def alignment_loss(model: BootEA, e1, e2, neg_l, neg_r,
                   pos_margin: float, neg_margin: float, neg_weight: float = 1.0) -> torch.Tensor:
    """**Limit-based** (absolute-margin) contrastive alignment loss - the Hit@1 driver.

    With L2-normalised ``z`` the distance ``d in [0, 2]``, so a *relative* margin
    can never saturate and collapses the space (especially with hard negatives).
    The limit-based objective instead sets absolute targets that **saturate**:

      * pull positives **below** ``pos_margin``  : ``[ d(z_e1,z_e2) - gamma1 ]+``
      * push negatives **above** ``neg_margin``  : ``[ gamma2 - d(z_e1,z_negR) ]+`` (+ left side)

    Once a negative is far enough (``d >= gamma2``) its gradient is zero -> no runaway
    repulsion, even for nearest-neighbour (hard) negatives.
    """
    z1 = model.encode(e1)
    z2 = model.encode(e2)
    zr = model.encode(neg_r)
    zl = model.encode(neg_l)
    d_pos = torch.norm(z1 - z2, p=2, dim=-1)                     # (B*neg,)
    d_neg_r = torch.norm(z1 - zr, p=2, dim=-1)
    d_neg_l = torch.norm(zl - z2, p=2, dim=-1)
    l_pos = F.relu(d_pos - pos_margin).mean()
    l_neg = 0.5 * (F.relu(neg_margin - d_neg_r).mean() + F.relu(neg_margin - d_neg_l).mean())
    return l_pos + neg_weight * l_neg
