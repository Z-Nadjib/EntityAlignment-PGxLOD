"""Entity-alignment models for PGxLOD.

One file per model. Eight models are implemented:
GCN-Align, RREA, MRAEA (GNN / graph attention), AliNet, KECG (multi-hop / GAT),
BootEA, NAEA and JAPE (TransE family).
Each file exposes the model class plus its loss function(s).
"""
from .gcnalign import GCNAlign, GraphConvolution, gcnalign_loss
from .rrea import RREA, rrea_align_loss
from .mraea import MRAEA, mraea_align_loss
from .alinet import AliNet, alinet_align_loss, alinet_relation_loss
from .kecg import KECG, kecg_cg_loss, kecg_ke_loss
from .bootea import BootEA, alignment_loss, limit_based_triple_loss
from .naea import NAEA, margin_ranking_loss
from .jape import JAPE, jape_se_loss

__all__ = ["GCNAlign", "GraphConvolution", "gcnalign_loss", "RREA", "rrea_align_loss",
           "MRAEA", "mraea_align_loss", "AliNet", "alinet_align_loss", "alinet_relation_loss",
           "KECG", "kecg_cg_loss", "kecg_ke_loss",
           "BootEA", "alignment_loss", "limit_based_triple_loss",
           "NAEA", "margin_ranking_loss", "JAPE", "jape_se_loss"]
