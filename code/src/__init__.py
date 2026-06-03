"""EA-PGxLOD - entity alignment models on the PGxLOD pharmacogenomic graph.

PGxLOD is a single knowledge graph. The task is entity resolution inside it: for a UCPGx
(pharmacogenomic relationship) node, find the other UCPGx linked to it by ``owl:sameAs``,
``skos:closeMatch``, ``broadMatch``, ``relatedMatch`` or ``related``. Eight entity-alignment
models are trained on the leak-free splits of six regimes and evaluated on the same
tie-aware metrics.

Package layout
--------------
main          : CLI entry point, one model on one regime.
benchmark/    : the graph, the six regimes and their splits, validation and test scoring.
data          : the dataset container, model-specific graph structures, negative sampling.
trainer       : one trainer per model on a shared training loop.
models/       : one file per model (GCN-Align, RREA, MRAEA, AliNet, KECG, BootEA, NAEA,
                JAPE).
utils/        : config loading, logging, metric formatting, plotting.
"""
