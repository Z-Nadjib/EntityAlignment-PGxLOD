"""Command-line entry point: train one model on one regime of the PGxLOD benchmark.

A single YAML config drives the run; the model is chosen by ``experiment.model`` (or
``--model``) and the regime by ``data.regime`` (or ``--regime``). ``main`` builds the
regime's dataset (:mod:`src.benchmark`), wires it to the matching trainer, chooses the
stopping epoch on the validation set and reads the test split once at the end. All
artefacts (log, checkpoints, metrics CSV, plots, ``result.json``) go to a timestamped run
directory under ``logging.output_dir`` (``experiments/``).

    python -m src.main --config ../configs/gcnalign_broadmatch.yaml --regime broadMatch/pgx
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import benchmark
from .data import build_alinet_graph, build_gcnalign_adj, build_kecg_graph, build_mraea_graph, build_neighbors
from .models.gcnalign import GCNAlign
from .models.rrea import RREA
from .models.mraea import MRAEA
from .models.alinet import AliNet
from .models.kecg import KECG
from .models.bootea import BootEA
from .models.naea import NAEA
from .models.jape import JAPE
from .trainer import (AliNetTrainer, BootEATrainer, GCNAlignTrainer, JAPETrainer,
                      KECGTrainer, MRAEATrainer, NAEATrainer, RREATrainer)
from .utils.config import load_config, make_run_dir
from .utils.logger import get_logger


def set_seed(seed: int):
    """Seed every RNG used in the pipeline (python, numpy, torch CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train an EA model on one regime of PGxLOD.")
    p.add_argument("--config", required=True, help="path to the YAML config")
    p.add_argument("--project-root", default=None, help="project root (default: parent of configs/)")
    p.add_argument("--name", default=None, help="override experiment.name (run-dir prefix)")
    p.add_argument("--regime", default=None,
                   help="regime to train on, one of: " + ", ".join(benchmark.regime_names()))
    p.add_argument("--csv-dir", default=None, help="override data.csv_dir")
    p.add_argument("--seed", type=int, default=None, help="override experiment.seed (model seed)")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--device", default=None, help="override experiment.device (cuda | cpu)")
    p.add_argument("--model", default=None, choices=["gcnalign", "rrea", "mraea", "alinet", "kecg", "bootea", "naea", "jape"], help="override experiment.model")
    return p.parse_args()


def build_gcnalign(cfg, data, device):
    adj = build_gcnalign_adj(data, self_loop_weight=cfg.model.get("self_loop_weight", 1.0))
    return GCNAlign(
        num_entities=data.num_entities, adj=adj.to(device),
        embed_dim=cfg.model.embed_dim, n_layers=cfg.model.n_layers,
        dropout=cfg.model.get("dropout", 0.0), activation=cfg.model.get("activation", False),
        normalize_embeddings=cfg.model.get("normalize_embeddings", True),
    ).to(device)


def build_rrea(cfg, data, device):
    graph = build_mraea_graph(data)
    m = cfg.model
    return RREA(graph, node_hidden=m.get("node_hidden", 100), depth=m.get("depth", 2),
                attn_heads=m.get("attn_heads", 1), dropout=m.get("dropout", 0.3)).to(device)


def build_mraea(cfg, data, device):
    graph = build_mraea_graph(data)
    m = cfg.model
    return MRAEA(graph, node_hidden=m.get("node_hidden", 100), rel_hidden=m.get("rel_hidden", 100),
                 depth=m.get("depth", 2), attn_heads=m.get("attn_heads", 2),
                 dropout=m.get("dropout", 0.3)).to(device)


def build_alinet(cfg, data, device):
    m = cfg.model
    adj1, two_hop = build_alinet_graph(data, max_two_hop=m.get("max_two_hop", 10), seed=cfg.experiment.seed)
    return AliNet(num_entities=data.num_entities, adj1=adj1.to(device), two_hop=two_hop.to(device),
                  embed_dim=m.get("embed_dim", 300), layer_dims=tuple(m.get("layer_dims", [300, 300])),
                  dropout=m.get("dropout", 0.0), normalize_embeddings=m.get("normalize_embeddings", True),
                  num_relations=data.num_relations).to(device)


def build_kecg(cfg, data, device):
    m = cfg.model
    edge_index = build_kecg_graph(data)
    return KECG(num_entities=data.num_entities, num_relations=data.num_relations,
                edge_index=edge_index.to(device), embed_dim=m.get("embed_dim", 128),
                n_layers=m.get("n_layers", 2), n_heads=m.get("n_heads", 2),
                attn_dropout=m.get("attn_dropout", 0.0),
                normalize_embeddings=m.get("normalize_embeddings", False),
                instance_normalization=m.get("instance_normalization", False)).to(device)


def build_bootea(cfg, data, device):
    m = cfg.model
    return BootEA(num_entities=data.num_entities, num_relations=data.num_relations,
                  embed_dim=m.get("embed_dim", 300), init=m.get("init", "xavier"),
                  normalize_embeddings=m.get("normalize_embeddings", True),
                  normalize_relations=m.get("normalize_relations", False),
                  squared=m.get("squared", False)).to(device)


def build_jape(cfg, data, device):
    """JAPE SE channel: TransE over the merged-seed graph (or with alignment-link triples)."""
    m = cfg.model
    link = str(cfg.train.get("merge", "all")).lower() == "link"
    return JAPE(num_entities=data.num_entities, num_relations=data.num_relations + int(link),
                embed_dim=m.get("embed_dim", 200), init=m.get("init", "xavier"),
                normalize_embeddings=m.get("normalize_embeddings", True),
                normalize_relations=m.get("normalize_relations", False),
                squared=m.get("squared", False)).to(device)


def build_naea(cfg, data, device):
    """NAEA: shared embeddings + neighbourhood-aware attention + padded neighbour tensors."""
    m = cfg.model
    ne, nr, ns, nm = build_neighbors(data, m.get("max_neighbors", 50), seed=cfg.experiment.seed)
    return NAEA(num_entities=data.num_entities, num_relations=data.num_relations,
                embed_dim=m.get("embed_dim", 200),
                neigh_ent=ne.to(device), neigh_rel=nr.to(device),
                neigh_sign=ns.to(device), neigh_mask=nm.to(device),
                attn_heads=m.get("attn_heads", 1), attn_dropout=m.get("attn_dropout", 0.0),
                init=m.get("init", "xavier"), normalize_embeddings=m.get("normalize_embeddings", True),
                neighbor_message=m.get("neighbor_message", "trans")).to(device)


def build_data(cfg, logger):
    """Dataset and validation evaluator of the regime, see ``src/benchmark``.

    The graph and the six splits are built once and cached; the regime picks which split
    this run trains on. ``data.seed`` drives the split, the validation carving and the
    query sampling. It is the seed of the benchmark, not of the model, and stays the same
    across model seeds so that every run is judged on the same split.
    """
    root = Path(cfg._project_root)
    regime = str(cfg.data.get("regime", "")).strip()
    if not regime:
        raise ValueError("data.regime (or --regime) is required, one of: "
                         + ", ".join(benchmark.regime_names()))
    base = benchmark.load(root / cfg.data.get("csv_dir", "Data/pgxlod_csv"),
                          seed=int(cfg.data.get("seed", benchmark.DATA_SEED)),
                          val_ratio=float(cfg.data.get("val_ratio", benchmark.VAL_RATIO)),
                          split_ratio=float(cfg.data.get("split_ratio", benchmark.SPLIT_RATIO)),
                          cache_dir=root / cfg.data.get("cache_dir", "Data/cache"),
                          logger=logger)
    logger.info(f"Regime {regime} | graph: {base['num_nodes']:,} nodes, "
                f"{len(base['triples']):,} triples, {base['num_ucpgx']:,} UCPGx candidates")
    for line in benchmark.describe(base).splitlines():
        logger.info(line)
    data = benchmark.build_dataset(base, regime)
    data.evaluation = benchmark.ValidationEvaluator(
        base, regime, select=str(cfg.eval.get("select", "filtered")),
        metric=str(cfg.eval.get("metric", "cosine")),
        csls_k=int(cfg.eval.get("csls_k", 3)))
    return data


def main():
    args = parse_args()
    cfg = load_config(args.config, project_root=args.project_root)

    if args.regime:
        cfg.data.regime = args.regime
    if args.csv_dir:
        cfg.data.csv_dir = args.csv_dir
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.epochs:
        cfg.train.epochs = args.epochs
        cfg.train.epoch_per_turn = args.epochs        # RREA / MRAEA read epoch_per_turn
    if args.device:
        cfg.experiment.device = args.device
    if args.model:
        cfg.experiment.model = args.model
    model_name = str(cfg.experiment.get("model", "gcnalign")).lower()
    regime_slug = str(cfg.data.get("regime", "")).replace("/", "-")
    cfg.experiment.name = args.name if args.name else f"{model_name}_{regime_slug}"

    set_seed(cfg.experiment.seed)
    run_dir = make_run_dir(cfg)
    logger = get_logger(cfg, run_dir)

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Model: {model_name} | seed {cfg.experiment.seed} | device: {device}")

    data = build_data(cfg, logger)
    logger.info(f"Data: {data.summary()}")

    # model dispatch: each branch builds the model-specific graph structures
    # (build_* above) and pairs it with its trainer class from src.trainer
    builders = {"gcnalign": (build_gcnalign, GCNAlignTrainer), "rrea": (build_rrea, RREATrainer),
                "mraea": (build_mraea, MRAEATrainer), "alinet": (build_alinet, AliNetTrainer),
                "kecg": (build_kecg, KECGTrainer), "bootea": (build_bootea, BootEATrainer),
                "naea": (build_naea, NAEATrainer),
                "jape": (build_jape, JAPETrainer)}
    if model_name not in builders:
        raise ValueError(f"unknown model {model_name!r}, known: {sorted(builders)}")
    build, Trainer = builders[model_name]
    model = build(cfg, data, device)
    trainer = Trainer(cfg, data, model, run_dir, logger)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    started = time.time()
    history = trainer.fit()
    benchmark.final_evaluation(data.evaluation, cfg, data, run_dir, history, device,
                               logger, minutes=(time.time() - started) / 60.0)
    logger.info(f"Finished. Best validation MRR={history['best_mrr']:.4f} @ epoch {history['best_epoch']}.")
    return history


if __name__ == "__main__":
    main()
