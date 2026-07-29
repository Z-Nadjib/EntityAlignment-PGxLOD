"""Training loops of the eight entity-alignment models.

Every trainer follows the same contract:

  * constructor ``(cfg, data, model, run_dir, logger)``;
  * ``fit()`` runs the loop and returns ``{best_mrr, best_epoch, metric_hist, loss_hist}``;
  * identical artefacts per run dir: ``training.txt``, ``loss.csv``, ``metrics.csv``,
    ``model_best.pt`` / ``model.pt``, ``embeddings.pt``, plots.

What is shared lives in :class:`BaseTrainer`: the epoch loop, the evaluation on the
validation set (``data.evaluation``), the best-epoch bookkeeping, early stopping, the CSV
and the plots. A model trainer only states how an epoch is trained, which table is
evaluated and what happens between epochs (bootstrapping, negative refreshes, ...).

The block that drives best-epoch selection and early stopping is the one of the
evaluator (``eval.select`` in the config), so the trainer and the evaluator can never
disagree on the best epoch.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
import torch.nn.functional as F                                            # noqa: E402

from .data import (PGxLOD, AlignSampler, Swapper, TripleSampler, build_jape_merged,  # noqa: E402
                   mutual_nearest)
from .models.alinet import alinet_align_loss, alinet_relation_loss        # noqa: E402
from .models.bootea import alignment_loss, limit_based_triple_loss        # noqa: E402
from .models.gcnalign import gcnalign_loss                                # noqa: E402
from .models.jape import jape_se_loss                                     # noqa: E402
from .models.kecg import kecg_cg_loss, kecg_ke_loss                       # noqa: E402
from .models.naea import alignment_loss as naea_alignment_loss, margin_ranking_loss  # noqa: E402
from .models.rrea import rrea_align_loss                                  # noqa: E402
from .utils.metrics import format_metrics                                 # noqa: E402
from .utils.plotting import plot_loss_curves, plot_metric_curves, set_modern_dark_style  # noqa: E402

try:
    from tqdm import tqdm
except Exception:                                    # pragma: no cover
    def tqdm(x, **k):
        return x

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD,
              "adagrad": torch.optim.Adagrad, "rmsprop": torch.optim.RMSprop}


# =========================================================================== #
#  Shared loop
# =========================================================================== #
class BaseTrainer:
    """Epoch loop, validation, best-epoch bookkeeping, early stopping, artefacts."""

    name = "model"
    loss_keys = ("loss",)

    def __init__(self, cfg, data: PGxLOD, model, run_dir: Path, logger):
        self.cfg = cfg
        self.data = data
        self.model = model
        self.run_dir = Path(run_dir)
        self.log = logger
        self.device = next(model.parameters()).device
        self.select = data.evaluation.select
        self.loss_hist, self.metric_hist = [], []
        self.best_mrr, self.best_epoch, self.no_improve = -1.0, -1, 0

    # ---- to be provided by each model ------------------------------------ #
    def train_epoch(self, epoch: int) -> dict:
        raise NotImplementedError

    def embeddings(self) -> torch.Tensor:
        """The table the evaluator ranks (model in eval mode, no grad)."""
        raise NotImplementedError

    def embeddings_to_save(self) -> dict:
        return {"entity_repr": self.embeddings().detach().cpu()}

    def before_epoch(self, epoch: int):
        """Hook run before training epoch ``epoch``."""

    def after_epoch(self, epoch: int) -> str:
        """Hook run after training epoch ``epoch``; returns a note for the log, or ''."""
        return ""

    def total_epochs(self) -> int:
        return int(self.cfg.train.epochs)

    def describe(self) -> str:
        return ""

    # ---- shared machinery --------------------------------------------------- #
    def make_optimizer(self, default: str = "adam"):
        c = self.cfg.train
        name = str(c.get("optimizer", default)).lower()
        kwargs = {"lr": c.lr, "weight_decay": c.get("weight_decay", 0.0)}
        if name == "rmsprop":                         # Keras-matched RMSprop (rho=0.9, eps=1e-7)
            kwargs.update(alpha=c.get("rms_alpha", 0.9), eps=c.get("rms_eps", 1e-7))
        return OPTIMIZERS.get(name, OPTIMIZERS[default])(self.model.parameters(), **kwargs)

    def clip(self):
        c = self.cfg.train
        if c.get("grad_clip", 0) and c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        return self.data.evaluation.evaluate(self.embeddings())

    def save_checkpoint(self, name, epoch, res=None):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "config": self.cfg.to_plain(), "metrics": res}, self.run_dir / name)

    def _append_csv(self, name, row, header_order=None):
        path = self.run_dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_curves(self):
        set_modern_dark_style()
        plots = self.cfg.logging.plots
        if self.loss_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_loss_curves(self.loss_hist, ax=ax, keys=self.loss_keys)
            fig.tight_layout(); fig.savefig(self.run_dir / plots.loss_curve); plt.close(fig)
        if self.metric_hist:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_metric_curves(self.metric_hist, ax=ax)
            fig.tight_layout(); fig.savefig(self.run_dir / plots.ranking_metrics); plt.close(fig)

    def log_evaluation(self, epoch: int, res: dict) -> bool:
        """Log, record and compare one validation result; True when patience runs out."""
        cfg = self.cfg
        for line in format_metrics(res).splitlines():
            self.log.info("           " + line)
        row = {"epoch": epoch}
        for blk in ("filtered", "multiref"):
            for kk, vv in res[blk].items():
                if kk != "n":
                    row[f"{blk}_{kk}"] = vv
        self._append_csv(cfg.logging.metrics_csv, row)
        self.metric_hist.append({
            "epoch": epoch,
            "filt_MRR": res["filtered"]["MRR"], "filt_Hit@1": res["filtered"]["Hit@1"],
            "multi_MRR": res["multiref"]["MRR"], "multi_Hit@1": res["multiref"]["Hit@1"]})
        self.plot_curves()
        mrr = res[self.select]["MRR"]
        if mrr > self.best_mrr:
            self.best_mrr, self.best_epoch, self.no_improve = mrr, epoch, 0
            if cfg.logging.save_best:
                self.save_checkpoint("model_best.pt", epoch, res)
                self.log.info(f"           -> new best {self.select} MRR={mrr:.4f} (saved model_best.pt)")
        else:
            self.no_improve += 1
        patience = cfg.train.get("early_stop_patience", 0)
        if patience and self.no_improve >= patience:
            self.log.info(f"           early stop (best={self.best_mrr:.4f} @ {self.best_epoch}).")
            return True
        return False

    def fit(self):
        cfg = self.cfg
        total = self.total_epochs()
        self.log.info(f"Run directory: {self.run_dir}")
        self.log.info(f"{self.name} | device={self.device} | {self.data.summary()} {self.describe()}")
        t0 = time.time()
        epoch = 0
        for epoch in tqdm(range(1, total + 1), desc=self.name, ncols=100):
            self.before_epoch(epoch)
            losses = self.train_epoch(epoch)
            losses["epoch"] = epoch
            self.loss_hist.append(losses)
            self._append_csv(cfg.logging.loss_csv, losses, ["epoch", *self.loss_keys])
            note = self.after_epoch(epoch)
            evaluate = epoch % cfg.eval.every == 0 or epoch == total
            if evaluate or note:
                parts = " ".join(f"{k}={losses[k]:.4f}" for k in self.loss_keys)
                self.log.info(f"epoch {epoch:>4}/{total} | {parts}{' | ' + note if note else ''}")
            if evaluate and self.log_evaluation(epoch, self.evaluate()):
                break
        if cfg.logging.save_last:
            self.save_checkpoint(cfg.logging.checkpoint_name, epoch)
        with torch.no_grad():
            self.model.eval()
            torch.save(self.embeddings_to_save(), self.run_dir / cfg.logging.embeddings_name)
        self.plot_curves()
        self.log.info(f"Done in {(time.time() - t0) / 60:.1f} min. "
                      f"Best {self.select} MRR={self.best_mrr:.4f} @ epoch {self.best_epoch}.")
        return {"best_mrr": self.best_mrr, "best_epoch": self.best_epoch,
                "metric_hist": self.metric_hist, "loss_hist": self.loss_hist}

    # ---- helpers shared by the bootstrapping models ------------------------ #
    def pseudo_pairs(self, z, bs) -> np.ndarray:
        """Confident CSLS mutual pairs of the unlabelled pool, one-to-one, best first."""
        pool = torch.as_tensor(self.data.unlabeled_ids, device=self.device)
        left, right, conf = mutual_nearest(z, pool, k=self.cfg.eval.csls_k)
        keep = left < right                          # a mutual pair shows up once per side
        left, right, conf = left[keep], right[keep], conf[keep]
        keep = conf >= bs.threshold
        left, right, conf = left[keep], right[keep], conf[keep]
        order = torch.argsort(conf, descending=True)
        used, pairs = set(), []
        for a, b in zip(left[order].tolist(), right[order].tolist()):
            if a in used or b in used:
                continue
            used.update((a, b))
            pairs.append((a, b))
            if len(pairs) >= bs.max_add:
                break
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    @torch.no_grad()
    def nearest_others(self, z_query, query_ids, z_all, C, chunk=1024):
        """The ``C`` nearest entities of each query, the query itself left out."""
        out = torch.empty((z_query.shape[0], C), dtype=torch.long, device=self.device)
        for s in range(0, z_query.shape[0], chunk):
            sim = z_query[s:s + chunk] @ z_all.t()
            rows = torch.arange(sim.shape[0], device=self.device)
            sim[rows, query_ids[s:s + chunk]] = float("-inf")
            out[s:s + chunk] = sim.topk(C, dim=1).indices
        return out


# =========================================================================== #
#  GCN-Align (shared GCN, structure channel SE)
# =========================================================================== #
class GCNAlignTrainer(BaseTrainer):
    """GCN-Align (Wang et al., EMNLP 2018).

    Full-batch: each epoch the GCN encodes all entities once, then a margin-based L1
    alignment loss is applied to the training pairs with random negatives on both sides.
    """

    name = "GCN-Align"

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.seed_l = torch.from_numpy(data.train_pairs[:, 0]).to(self.device)
        self.seed_r = torch.from_numpy(data.train_pairs[:, 1]).to(self.device)
        self.optimizer = self.make_optimizer()

    def train_epoch(self, epoch):
        self.model.train()
        c = self.cfg.train
        z = self.model.forward_all()
        k = c.k
        S = self.seed_l.shape[0]
        # The loss is chunked over the pairs (each expands into k negatives) so the large
        # regimes do not run out of memory; one backward per chunk into the single encoded
        # z (retain_graph), then one optimizer step.
        bs = int(c.get("align_batch", 20000))
        total_exp = max(1, S * k)
        nchunks = (S + bs - 1) // bs
        self.optimizer.zero_grad()
        total_loss = 0.0
        for ci, s in enumerate(range(0, S, bs)):
            e1 = self.seed_l[s:s + bs].repeat_interleave(k)
            e2 = self.seed_r[s:s + bs].repeat_interleave(k)
            n = e1.shape[0]
            neg_r = torch.randint(self.data.num_entities, (n,), device=self.device)
            neg_l = torch.randint(self.data.num_entities, (n,), device=self.device)
            loss = gcnalign_loss(z, e1, e2, neg_l, neg_r, c.margin) * (n / total_exp)
            loss.backward(retain_graph=(ci < nchunks - 1))
            total_loss += float(loss.item())
        self.clip()
        self.optimizer.step()
        return {"loss": total_loss}

    def embeddings(self):
        return self.model.forward_all()

    def embeddings_to_save(self):
        return {"entity_repr": self.model.forward_all().cpu(),
                "ent_emb": self.model.ent_emb.weight.detach().cpu()}


# =========================================================================== #
#  RREA / MRAEA (relation-aware GAT; optional turn-based CSLS bootstrap)
# =========================================================================== #
class RREATrainer(BaseTrainer):
    """RREA (Mao et al., CIKM 2020) and MRAEA (Mao et al., WSDM 2020).

    Full-batch graph attention with an L1 margin loss. With ``turns > 1`` the training
    runs in turns of ``epoch_per_turn`` epochs, and between two turns the CSLS mutual
    nearest neighbours between the left and right entities of the validation pairs are
    added to the training pairs, as in the official RREA code (``rest_set_1`` /
    ``rest_set_2`` built from ``dev_pair``). Matched entities leave the pool.
    """

    name = "RREA"

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.name = type(model).__name__
        self.train_pairs = torch.from_numpy(data.train_pairs).long().to(self.device)
        self.rest_left = torch.from_numpy(np.unique(data.val_pairs[:, 0])).long().to(self.device)
        self.rest_right = torch.from_numpy(np.unique(data.val_pairs[:, 1])).long().to(self.device)
        self.optimizer = self.make_optimizer()
        self.gamma = cfg.train.get("gamma", 3.0)
        self.csls_k = cfg.train.get("csls_k", 10)
        self.turns = cfg.train.get("turns", 1)
        self.ept = cfg.train.get("epoch_per_turn", cfg.train.get("epochs", 400))

    def total_epochs(self):
        return self.turns * self.ept

    def describe(self):
        return f"turns={self.turns} x {self.ept}"

    def train_epoch(self, epoch):
        self.model.train()
        pos = self.train_pairs
        neg = torch.randint(self.model.N, pos.shape, device=self.device)
        loss = rrea_align_loss(self.model(), torch.cat([pos, neg], dim=-1), self.gamma)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return {"loss": float(loss.item())}

    def after_epoch(self, epoch):
        if self.turns > 1 and epoch % self.ept == 0 and epoch < self.total_epochs():
            return self.bootstrap()
        return ""

    @torch.no_grad()
    def bootstrap(self):
        """CSLS mutual nearest neighbours between the two pools become training pairs."""
        if self.rest_left.numel() == 0 or self.rest_right.numel() == 0:
            return "bootstrap: empty pool"
        self.model.eval()
        emb = self.model()
        zl = F.normalize(emb[self.rest_left], dim=-1)
        zr = F.normalize(emb[self.rest_right], dim=-1)
        sim = zl @ zr.t()
        k = min(self.csls_k, sim.size(0), sim.size(1))
        rl = sim.topk(k, dim=1).values.mean(1)
        rr = sim.topk(k, dim=0).values.mean(0)
        csls = 2 * sim - rl[:, None] - rr[None, :]
        a, b = csls.argmax(dim=1), csls.argmax(dim=0)
        rows = torch.arange(self.rest_left.numel(), device=self.device)
        mutual = b[a] == rows
        li, ri = rows[mutual], a[mutual]
        if li.numel() == 0:
            return "bootstrap: 0 pairs"
        new = torch.stack([self.rest_left[li], self.rest_right[ri]], dim=1)
        self.train_pairs = torch.cat([self.train_pairs, new], dim=0)
        keep_l = torch.ones(self.rest_left.numel(), dtype=torch.bool, device=self.device); keep_l[li] = False
        keep_r = torch.ones(self.rest_right.numel(), dtype=torch.bool, device=self.device); keep_r[ri] = False
        self.rest_left, self.rest_right = self.rest_left[keep_l], self.rest_right[keep_r]
        return (f"bootstrap: +{new.size(0)} mutual pairs "
                f"(train={self.train_pairs.size(0)}, pool={self.rest_left.numel()})")

    def embeddings(self):
        return self.model()


MRAEATrainer = RREATrainer          # same graph, loss and loop; only the encoder differs


# =========================================================================== #
#  AliNet (gated multi-hop GNN + relation-aware TransE anchor)
# =========================================================================== #
class AliNetTrainer(BaseTrainer):
    """AliNet (Sun et al., AAAI 2020).

    Full-batch: encodes all entities once per epoch, then a margin-ranking alignment loss
    (random negatives on both sides) plus a relation-aware TransE loss on a sampled
    triple batch that anchors every entity structurally.
    """

    name = "AliNet"
    loss_keys = ("loss", "align", "rel")

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.seed_l = torch.from_numpy(data.train_pairs[:, 0]).to(self.device)
        self.seed_r = torch.from_numpy(data.train_pairs[:, 1]).to(self.device)
        self.triples = torch.from_numpy(data.triples).to(self.device)
        self.optimizer = self.make_optimizer()

    def describe(self):
        return f"2hop_edges={self.model.e_src.numel()}"

    def train_epoch(self, epoch):
        self.model.train()
        c = self.cfg.train
        N = self.data.num_entities
        z = self.model.forward_all()
        n = c.get("neg_samples", 5)
        e1 = self.seed_l.repeat_interleave(n)
        e2 = self.seed_r.repeat_interleave(n)
        B = e1.shape[0]
        neg_r = torch.randint(N, (B,), device=self.device)
        neg_l = torch.randint(N, (B,), device=self.device)
        margin = c.get("align_margin", 1.0)

        # Alignment loss accumulated over chunks of pairs: the gather z[e1] is (B, dim) and
        # does not fit at once on the largest regime. Chunk means weighted by chunk / B give
        # the full-batch gradient; align_chunk_size absent or 0 is a single chunk.
        self.optimizer.zero_grad()
        chunk = c.get("align_chunk_size", 0) or B
        align_val = 0.0
        for s in range(0, B, chunk):
            cs = min(s + chunk, B) - s
            la = alinet_align_loss(z, e1[s:s + chunk], e2[s:s + chunk],
                                   neg_l[s:s + chunk], neg_r[s:s + chunk], margin)
            (la * (cs / B)).backward(retain_graph=True)
            align_val += float(la.item()) * (cs / B)

        rel_val, rel_weight = 0.0, 0.0
        rcfg = c.get("relation", None)
        if rcfg and rcfg.get("enabled", True) and self.model.rel_emb is not None:
            rel_weight = rcfg.get("weight", 1.0)
            M_ = self.triples.shape[0]
            bs = min(rcfg.get("batch_size", 20000), M_)
            pos = self.triples[torch.randint(M_, (bs,), device=self.device)]
            neg_t = torch.randint(N, (bs,), device=self.device)
            rel = alinet_relation_loss(z, self.model.rel_emb, pos, neg_t, rcfg.get("margin", 1.0))
            rel_val = float(rel.item())
            (rel_weight * rel).backward()

        self.clip()
        self.optimizer.step()
        return {"loss": align_val + rel_weight * rel_val, "align": align_val, "rel": rel_val}

    def embeddings(self):
        return self.model.forward_all()


# =========================================================================== #
#  KECG (alternating Cross-Graph GAT / Knowledge-Embedding TransE)
# =========================================================================== #
class KECGTrainer(BaseTrainer):
    """KECG (Li et al., EMNLP 2019).

    Alternates per epoch: even = Cross-Graph triplet margin loss on the GAT output (hard
    negatives = nearest other training entities, refreshed every ``update_num``); odd =
    Knowledge-Embedding TransE on the triples.

    The official code takes the nearest rows of the training column and drops the first
    one, which is the entity itself only under a one-to-one alignment. On PGxLOD an entity
    has several partners, so the hard negatives are searched among distinct entities and
    never include the anchor or one of its training partners.
    """

    name = "KECG"

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.seed_l = torch.from_numpy(data.train_pairs[:, 0]).to(self.device)
        self.seed_r = torch.from_numpy(data.train_pairs[:, 1]).to(self.device)
        self.triples = torch.from_numpy(data.triples).to(self.device)
        self.hard_r = self.hard_l = self.nns_pick = None
        self.optimizer = self.make_optimizer(default="adagrad")
        self.use_nns = cfg.train.get("use_nns", True)
        self.update_num = cfg.train.get("update_num", 5)
        # training partners of every entity, both ways: they are never hard negatives
        self.partners: dict = {}
        for a, b in data.train_pairs.tolist():
            self.partners.setdefault(a, set()).add(b)
            self.partners.setdefault(b, set()).add(a)
        self._masks: dict = {}

    @torch.no_grad()
    def refresh_nns(self):
        """For each training pair side, the k nearest OTHER training entities.

        ``cg_sample_size`` restricts the search (and the CG loss) to a random subset of the
        pairs: on ``related`` the full cdist would not fit. The subset (``nns_pick``) is
        shared with ``train_epoch`` so e1/e2 and the hard negatives stay aligned until the
        next refresh."""
        k = self.cfg.train.get("k_cg", 25)
        samp = self.cfg.train.get("cg_sample_size", 0)
        self.model.eval()
        z = self.model.forward_all()
        ns = self.seed_l.numel()
        full = not (samp and samp < ns)
        if full:
            self.nns_pick = torch.arange(ns, device=self.device)
        else:
            self.nns_pick = torch.randperm(ns, device=self.device)[:samp]
        sl, sr = self.seed_l[self.nns_pick], self.seed_r[self.nns_pick]
        # (samp, k) entity ids, aligned with nns_pick
        self.hard_r = self._nearest(z, sr, sl, self.seed_r, k, "r" if full else None)
        self.hard_l = self._nearest(z, sl, sr, self.seed_l, k, "l" if full else None)

    def _nearest(self, z, anchor, other, side, k, cache_key=None):
        """The ``k`` entities of ``side`` nearest to each ``anchor``, the anchor itself and
        the training partners of ``other`` left out.

        The search runs over the distinct entities of the column: an entity with several
        partners appears several times in it, and its own copies, at distance 0, would
        otherwise fill its nearest neighbours. A row with fewer than ``k`` admissible
        entities is completed with random ones."""
        pool = torch.unique(side)
        dist = torch.cdist(z[anchor], z[pool])
        mask = self._masks.get(cache_key) if cache_key else None
        if mask is None:
            col = {e: j for j, e in enumerate(pool.tolist())}
            rows, cols = [], []
            for i, (a, o) in enumerate(zip(anchor.tolist(), other.tolist())):
                for e in {a} | self.partners.get(o, set()):
                    j = col.get(e)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
            mask = (torch.tensor(rows, dtype=torch.long, device=self.device),
                    torch.tensor(cols, dtype=torch.long, device=self.device))
            if cache_key:
                self._masks[cache_key] = mask
        dist[mask] = float("inf")
        vals, idx = dist.topk(min(k, pool.numel()), largest=False)
        out = pool[idx]
        if out.shape[1] < k:
            pad = torch.full((out.shape[0], k - out.shape[1]), float("inf"), device=self.device)
            vals = torch.cat([vals, pad], dim=1)
            out = torch.cat([out, torch.zeros_like(pad, dtype=torch.long)], dim=1)
        blocked = torch.isinf(vals)
        if blocked.any():
            out[blocked] = torch.randint(self.data.num_entities, (int(blocked.sum()),),
                                         device=self.device)
        return out

    def before_epoch(self, epoch):
        if (self.use_nns and epoch % 2 == 0
                and (self.hard_r is None or epoch % (2 * self.update_num) == 0)):
            self.refresh_nns()

    def train_epoch(self, epoch):
        self.model.train()
        c = self.cfg.train
        N = self.data.num_entities
        if epoch % 2 == 0:                                       # Cross-Graph
            z = self.model.forward_all()
            n = c.get("k_cg", 25)
            samp = c.get("cg_sample_size", 0)
            if self.hard_r is not None:                          # hard negatives
                sl, sr = self.seed_l[self.nns_pick], self.seed_r[self.nns_pick]
                e1 = sl.repeat_interleave(n); e2 = sr.repeat_interleave(n)
                neg_r = self.hard_r.reshape(-1); neg_l = self.hard_l.reshape(-1)
            else:                                                # random negatives
                if samp and samp < self.seed_l.numel():
                    pick = torch.randperm(self.seed_l.numel(), device=self.device)[:samp]
                    sl, sr = self.seed_l[pick], self.seed_r[pick]
                else:
                    sl, sr = self.seed_l, self.seed_r
                e1 = sl.repeat_interleave(n); e2 = sr.repeat_interleave(n)
                neg_r = torch.randint(N, (e1.shape[0],), device=self.device)
                neg_l = torch.randint(N, (e1.shape[0],), device=self.device)
            loss = kecg_cg_loss(z, e1, e2, neg_l, neg_r, c.get("margin_cg", 3.0))
        else:                                                    # Knowledge Embedding
            bs = min(c.get("ke_batch_size", 50000), self.triples.shape[0])
            pos = self.triples[torch.randint(self.triples.shape[0], (bs,), device=self.device)]
            pos_r = pos.repeat(c.get("k_ke", 2), 1)
            neg = pos_r.clone()
            ch = torch.rand(pos_r.shape[0], device=self.device) < 0.5
            rnd = torch.randint(N, (pos_r.shape[0],), device=self.device)
            neg[:, 0] = torch.where(ch, rnd, neg[:, 0]); neg[:, 2] = torch.where(~ch, rnd, neg[:, 2])
            z = self.model.forward_all()
            loss = kecg_ke_loss(z, self.model.rel_emb, pos_r, neg, c.get("margin_ke", 3.0))
        self.optimizer.zero_grad(); loss.backward()
        self.clip()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def embeddings(self):
        return self.model.forward_all()


# =========================================================================== #
#  BootEA (AlignE + swapping + editable bootstrapping)
# =========================================================================== #
class BootEATrainer(BaseTrainer):
    """BootEA (Sun et al., IJCAI 2018).

      * **AlignE**: limit-based TransE loss over a (mutable) triple set with
        eps-truncated negatives (nearest neighbours).
      * **Alignment by swapping**: the training pairs generate aligned triples.
      * **Alignment loss**: limit-based contrastive objective on the training pairs.
      * **Editable bootstrapping**: a recomputed CSLS mutual one-to-one matching over the
        unlabelled pool feeds a pseudo alignment term, replaced at each round.
    """

    name = "BootEA"
    loss_keys = ("loss", "kge", "align", "pseudo")

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.triple_sampler = TripleSampler(data, self.device, neg=cfg.train.neg_samples)
        self.swapper = Swapper(data)
        self.rebuild_triples(data.train_pairs)
        self.align_sampler = AlignSampler(data, self.device, neg=cfg.train.neg_samples)
        self.pseudo_sampler = None
        self.optimizer = self.make_optimizer()
        self.scheduler = None
        if str(cfg.train.get("lr_schedule", "none")).lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.02)

    def rebuild_triples(self, pairs):
        """Set the triple set to the graph plus the triples swapped from ``pairs``."""
        triples = self.data.triples
        sw = self.cfg.train.swapping
        if sw.enabled and len(pairs):
            swapped = self.swapper.generate(pairs, cap_per_role=sw.cap_per_role)
            if len(swapped):
                triples = np.concatenate([triples, swapped], axis=0)
        self.triple_sampler.set_triples(triples)

    def describe(self):
        return f"triples(+swap)={len(self.triple_sampler)}"

    @torch.no_grad()
    def refresh_truncated_candidates(self):
        """eps-truncated negatives: the nearest neighbours of every entity."""
        C = self.cfg.train.eps_truncated.num_candidates
        self.model.eval()
        ids = torch.arange(self.data.num_entities, device=self.device)
        z = F.normalize(self.model.encode_all(ids), dim=-1)
        self.triple_sampler.set_candidates(self.nearest_others(z, ids, z, C))

    @torch.no_grad()
    def refresh_hard_negatives(self):
        C = self.cfg.train.hard_negatives.num_candidates
        self.model.eval()
        pairs = self.align_sampler.pairs
        z = F.normalize(self.model.encode_all(torch.arange(self.data.num_entities, device=self.device)), dim=-1)
        hard_r = self.nearest_others(z[pairs[:, 0]], pairs[:, 0], z, C)
        hard_l = self.nearest_others(z[pairs[:, 1]], pairs[:, 1], z, C)
        self.align_sampler.set_hard_negatives(hard_r, hard_l)

    def train_epoch(self, epoch):
        self.model.train()
        c = self.cfg.train
        align_batches = list(self.align_sampler.batches(c.align_batch_size))
        n_align = max(1, len(align_batches))
        pseudo_batches = (list(self.pseudo_sampler.batches(c.align_batch_size))
                          if self.pseudo_sampler is not None else [])
        n_pseudo = len(pseudo_batches)
        pw = c.bootstrap.pseudo_weight

        tot = {"loss": 0.0, "kge": 0.0, "align": 0.0, "pseudo": 0.0}
        steps = 0
        for i, (pos, neg) in enumerate(self.triple_sampler.batches(c.batch_size)):
            self.optimizer.zero_grad()
            ps, ns = self.model.triple_score(pos), self.model.triple_score(neg)
            if str(c.get("kge_reduction", "mean")) == "sum":      # official AlignE reduction
                kge = F.relu(ps - c.pos_margin_kge).sum() + c.neg_weight_kge * F.relu(c.neg_margin_kge - ns).sum()
            else:
                kge = limit_based_triple_loss(ps, ns, c.pos_margin_kge, c.neg_margin_kge, c.neg_weight_kge)
            align = torch.zeros((), device=self.device)
            if c.align_loss_weight > 0:
                _p, (e1, e2, nl, nr) = align_batches[i % n_align]
                align = alignment_loss(self.model, e1, e2, nl, nr,
                                       c.align_pos_margin, c.align_neg_margin, c.align_neg_weight)
            loss = kge + c.align_loss_weight * align
            pseudo_val = 0.0
            if n_pseudo and pw > 0:
                _q, (pe1, pe2, pnl, pnr) = pseudo_batches[i % n_pseudo]
                pseudo = alignment_loss(self.model, pe1, pe2, pnl, pnr,
                                        c.align_pos_margin, c.align_neg_margin, c.align_neg_weight)
                loss = loss + pw * pseudo
                pseudo_val = pseudo.item()
            loss.backward()
            self.clip()
            self.optimizer.step()
            tot["kge"] += kge.item(); tot["align"] += align.item()
            tot["pseudo"] += pseudo_val; tot["loss"] += loss.item()
            steps += 1
        return {k: v / max(1, steps) for k, v in tot.items()}

    @torch.no_grad()
    def bootstrap(self) -> int:
        self.model.eval()
        pairs = self.pseudo_pairs(self.model.forward_all(), self.cfg.train.bootstrap)
        if len(pairs) == 0:
            self.pseudo_sampler = None
            return 0
        if self.pseudo_sampler is None:
            self.pseudo_sampler = AlignSampler(self.data, self.device,
                                               neg=self.cfg.train.neg_samples, pairs=pairs)
        else:
            self.pseudo_sampler.set_pairs(pairs)             # replaced, never accumulated
        if self.cfg.train.swapping.get("swap_pseudo", False):
            self.rebuild_triples(np.concatenate([self.data.train_pairs, pairs], axis=0))
        return len(pairs)

    def after_epoch(self, epoch):
        c = self.cfg.train
        boot, eps, hn = c.bootstrap, c.eps_truncated, c.hard_negatives
        notes = []
        if hn.enabled and epoch >= hn.start_epoch and (epoch - hn.start_epoch) % hn.refresh_every == 0:
            self.refresh_hard_negatives(); notes.append("hard-neg refreshed")
        if eps.enabled and epoch >= eps.start_epoch and (epoch - eps.start_epoch) % eps.refresh_every == 0:
            self.refresh_truncated_candidates(); notes.append("eps-trunc refreshed")
        if boot.enabled and epoch >= boot.start_epoch and (epoch - boot.start_epoch) % boot.every == 0:
            notes.append(f"bootstrap: {self.bootstrap()} pseudo-pairs (beta={boot.pseudo_weight})")
        if self.scheduler is not None:
            self.scheduler.step()
        return " | ".join(notes)

    def embeddings(self):
        return self.model.forward_all()

    def embeddings_to_save(self):
        return {"ent_emb": self.model.ent_emb.weight.detach().cpu(),
                "rel_emb": self.model.rel_emb.weight.detach().cpu()}


# =========================================================================== #
#  NAEA (TransE + neighbourhood-aware attention + CSLS bootstrap)
# =========================================================================== #
class NAEATrainer(BaseTrainer):
    """NAEA (Zhu et al., IJCAI 2019).

    Joint objective: a TransE margin-ranking loss over the triples + a limit-based
    contrastive alignment loss over the neighbour-aware embeddings ``z = e + e_hat`` on the
    training (and bootstrapped) pairs.
    """

    name = "NAEA"
    loss_keys = ("loss", "kge", "align", "pseudo")

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.triple_sampler = TripleSampler(data, self.device, neg=cfg.train.neg_samples)
        self.align_sampler = AlignSampler(data, self.device, neg=cfg.train.neg_samples)
        self.pseudo_sampler = None
        self.optimizer = self.make_optimizer()

    @torch.no_grad()
    def refresh_hard_negatives(self):
        C = self.cfg.train.hard_negatives.num_candidates
        self.model.eval()
        pairs = self.align_sampler.pairs
        z = F.normalize(self.embeddings(), dim=-1)
        self.align_sampler.set_hard_negatives(self.nearest_others(z[pairs[:, 0]], pairs[:, 0], z, C),
                                              self.nearest_others(z[pairs[:, 1]], pairs[:, 1], z, C))

    def train_epoch(self, epoch):
        self.model.train()
        c = self.cfg.train
        align_batches = list(self.align_sampler.batches(c.align_batch_size))
        n_align = max(1, len(align_batches))
        pseudo_batches = (list(self.pseudo_sampler.batches(c.align_batch_size))
                          if self.pseudo_sampler is not None else [])
        n_pseudo = len(pseudo_batches)
        pw = c.bootstrap.pseudo_weight

        tot = {"loss": 0.0, "kge": 0.0, "align": 0.0, "pseudo": 0.0}
        steps = 0
        for i, (pos, neg) in enumerate(self.triple_sampler.batches(c.batch_size)):
            self.optimizer.zero_grad()
            kge = margin_ranking_loss(self.model.transe_score(pos), self.model.transe_score(neg), c.margin_kge)
            _p, (e1, e2, nl, nr) = align_batches[i % n_align]
            align = naea_alignment_loss(self.model, e1, e2, nl, nr,
                                        c.align_pos_margin, c.align_neg_margin, c.align_neg_weight)
            loss = kge + c.align_loss_weight * align
            pseudo_val = 0.0
            if n_pseudo:
                _q, (pe1, pe2, pnl, pnr) = pseudo_batches[i % n_pseudo]
                pseudo = naea_alignment_loss(self.model, pe1, pe2, pnl, pnr,
                                             c.align_pos_margin, c.align_neg_margin, c.align_neg_weight)
                loss = loss + pw * pseudo
                pseudo_val = pseudo.item()
            loss.backward()
            self.clip()
            self.optimizer.step()
            tot["kge"] += kge.item(); tot["align"] += align.item()
            tot["pseudo"] += pseudo_val; tot["loss"] += loss.item()
            steps += 1
        return {k: v / max(1, steps) for k, v in tot.items()}

    @torch.no_grad()
    def bootstrap(self) -> int:
        self.model.eval()
        pairs = self.pseudo_pairs(self.embeddings(), self.cfg.train.bootstrap)
        if len(pairs) == 0:
            self.pseudo_sampler = None
            return 0
        if self.pseudo_sampler is None:
            self.pseudo_sampler = AlignSampler(self.data, self.device,
                                               neg=self.cfg.train.neg_samples, pairs=pairs)
        else:
            self.pseudo_sampler.set_pairs(pairs)
        return len(pairs)

    def after_epoch(self, epoch):
        c = self.cfg.train
        boot, hn = c.bootstrap, c.hard_negatives
        notes = []
        if hn.enabled and epoch >= hn.start_epoch and (epoch - hn.start_epoch) % hn.refresh_every == 0:
            self.refresh_hard_negatives(); notes.append("hard-neg refreshed")
        if boot.enabled and epoch >= boot.start_epoch and (epoch - boot.start_epoch) % boot.every == 0:
            notes.append(f"bootstrap: {self.bootstrap()} pseudo-pairs (beta={boot.pseudo_weight})")
        return " | ".join(notes)

    def embeddings(self):
        return self.model.forward_all(chunk=self.cfg.model.get("encode_chunk", 4096))

    def embeddings_to_save(self):
        return {"ent_emb": self.model.ent_emb.weight.detach().cpu(),
                "rel_emb": self.model.rel_emb.weight.detach().cpu()}


# =========================================================================== #
#  JAPE (SE structure channel: TransE on the merged-pair graph)
# =========================================================================== #
class JAPETrainer(BaseTrainer):
    """JAPE (Sun et al., ISWC 2017), structure channel only (PGxLOD has no attributes).

    ``train.merge`` sets how the training pairs enter the graph:

    * ``all``  (JAPE): the entities of a training pair share an id. The merge is
      transitive, so on a relation that is not an equivalence whole clusters collapse
      onto one vector;
    * ``link``: no merge; each training pair becomes a triple ``(a, r_align, b)`` with a
      dedicated relation, which TransE learns like any other.

    ``train.loss`` is ``margin`` (TransE margin ranking) or ``official`` (Eq. 1 of the
    paper, ``sum f(pos) - alpha * sum f(neg)`` with two alternating optimiser steps).
    Evaluation reads the entity table through the merge map (identity for ``link``).
    """

    name = "JAPE"

    def __init__(self, cfg, data, model, run_dir, logger):
        super().__init__(cfg, data, model, run_dir, logger)
        self.merge = str(cfg.train.get("merge", "all")).lower()
        if self.merge == "link":
            pairs = data.train_pairs
            link = np.stack([pairs[:, 0], np.full(len(pairs), data.num_relations), pairs[:, 1]], 1)
            triples = np.concatenate([data.triples, link], 0)
            merge_map = np.arange(data.num_entities, dtype=np.int64)
        else:
            triples, merge_map = build_jape_merged(data)
        self.triples = torch.from_numpy(triples).to(self.device)
        self.merge_map = torch.from_numpy(merge_map).to(self.device)
        self.cano = torch.unique(self.triples[:, [0, 2]].reshape(-1))
        self.loss = str(cfg.train.get("loss", "margin")).lower()
        self.optimizer = self.make_optimizer()
        if self.loss == "official":
            self.opt_neg = self.make_optimizer()

    def describe(self):
        return f"merge={self.merge} loss={self.loss} canonical={len(self.cano)}"

    def train_epoch(self, epoch):
        self.model.train()
        c = self.cfg.train
        order = torch.randperm(self.triples.shape[0], device=self.device)
        tot, steps = 0.0, 0
        for s in range(0, self.triples.shape[0], c.batch_size):
            pos = self.triples[order[s:s + c.batch_size]]
            pos_r = pos.repeat(c.neg_samples, 1)
            neg = pos_r.clone()
            ch = torch.rand(pos_r.shape[0], device=self.device) < 0.5
            rnd = self.cano[torch.randint(len(self.cano), (pos_r.shape[0],), device=self.device)]
            neg[:, 0] = torch.where(ch, rnd, neg[:, 0])
            neg[:, 2] = torch.where(~ch, rnd, neg[:, 2])
            if self.loss == "official":
                self.optimizer.zero_grad()
                lp = self.model.triple_score(pos).sum()
                lp.backward(); self.clip(); self.optimizer.step()
                self.opt_neg.zero_grad()
                ln = -c.get("alpha", 0.1) * self.model.triple_score(neg).sum()
                ln.backward(); self.clip(); self.opt_neg.step()
                loss_val = float(lp.item() + ln.item()) / len(pos)
            else:
                loss = jape_se_loss(self.model, pos, neg, c.margin)
                self.optimizer.zero_grad(); loss.backward()
                self.clip()
                self.optimizer.step()
                loss_val = loss.item()
            tot += loss_val; steps += 1
        return {"loss": tot / max(1, steps)}

    def embeddings(self):
        return self.model.forward_all(self.merge_map)

    def embeddings_to_save(self):
        return {"ent_emb": self.model.ent_emb.weight.detach().cpu(),
                "rel_emb": self.model.rel_emb.weight.detach().cpu(),
                "merge_map": self.merge_map.cpu()}
