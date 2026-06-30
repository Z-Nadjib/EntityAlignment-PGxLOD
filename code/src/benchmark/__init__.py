"""The PGxLOD entity-resolution benchmark: one graph, six regimes, leak-free splits.

* one graph and one embedding table: the task is to find, for a UCPGx, the other UCPGx it
  is linked to by the relation of the regime (``sameAs``, ``closeMatch``, ``broadMatch``,
  ``relatedMatch``, ``related``);
* the split is made by connected component, by family for ``broadMatch``, with the groups
  of identical verbalisation chained together, so that no test pair can be read off a
  train pair (the leakage is measured at every build);
* a validation set is carved out of the train split and chooses the stopping epoch; the
  test split is read once, at the end;
* ranks average over ties.

``src.main`` builds the dataset here, hands :class:`ValidationEvaluator` to the trainer and
calls :func:`final_evaluation` once the training is over.
"""
from .dataset import build_dataset
from .scoring import ValidationEvaluator, evaluate_embeddings, final_evaluation
from .splits import DATA_SEED, REGIMES, SPLIT_RATIO, VAL_RATIO, describe, load, regime_names, regime_tag

__all__ = ["DATA_SEED", "REGIMES", "SPLIT_RATIO", "VAL_RATIO", "describe", "load",
           "regime_names", "regime_tag", "build_dataset", "ValidationEvaluator",
           "evaluate_embeddings", "final_evaluation"]
