<div align="center">

<img src="docs/assets/banner.svg" width="100%" alt="EntityAlignment-PGxLOD"/>

<br/>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=500&size=18&duration=3000&pause=900&color=2DD4BF&center=true&vCenter=true&width=760&lines=Eight+entity+alignment+models+in+one+PyTorch+codebase;Six+alignment+tasks+on+the+PGxLOD+knowledge+graph;One+YAML+config%2C+one+command%2C+one+result.json;Benchmark+runner%2C+baseline+and+result+tables+included" alt="Eight entity alignment models in one PyTorch codebase"/>

<p>
  <img src="https://img.shields.io/badge/models-8-F5B700?style=for-the-badge&labelColor=0B2A33" alt="8 models"/>
  <img src="https://img.shields.io/badge/alignment%20tasks-6-3B82F6?style=for-the-badge&labelColor=0B2A33" alt="6 alignment tasks"/>
  <img src="https://img.shields.io/badge/knowledge%20graph-PGxLOD-2DD4BF?style=for-the-badge&labelColor=0B2A33" alt="PGxLOD"/>
</p>
<p>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.9-2DD4BF?style=flat-square&logo=python&logoColor=white&labelColor=0B2A33" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EF4444?style=flat-square&logo=pytorch&logoColor=white&labelColor=0B2A33" alt="PyTorch 2.0+"/>
  <img src="https://img.shields.io/badge/GPU-CUDA-22C55E?style=flat-square&logo=nvidia&logoColor=white&labelColor=0B2A33" alt="CUDA"/>
  <img src="https://img.shields.io/badge/configs-YAML-A78BFA?style=flat-square&labelColor=0B2A33" alt="YAML configs"/>
  <img src="https://img.shields.io/badge/license-MIT-22C55E?style=flat-square&labelColor=0B2A33" alt="MIT license"/>
</p>

<p>
  <a href="#installation"><b>Installation</b></a> &nbsp;|&nbsp;
  <a href="#data"><b>Data</b></a> &nbsp;|&nbsp;
  <a href="#usage"><b>Usage</b></a> &nbsp;|&nbsp;
  <a href="#configuration"><b>Configuration</b></a> &nbsp;|&nbsp;
  <a href="#outputs"><b>Outputs</b></a> &nbsp;|&nbsp;
  <a href="#adding-a-model"><b>Adding a model</b></a> &nbsp;|&nbsp;
  <a href="#citation"><b>Citation</b></a>
</p>

</div>

<br/>

**EntityAlignment-PGxLOD** trains and evaluates entity alignment models on
[PGxLOD](https://doi.org/10.1186/s12859-019-2693-9), a pharmacogenomic knowledge graph. Given a
relation such as `owl:sameAs` or `skos:broadMatch`, a model learns from known pairs of
pharmacogenomic knowledge units (PGxKU) and must find the partners of unseen units among all the
units of the graph. The repository contains the eight models, the data pipeline, the evaluation
code, the configurations and the logs of every run.

<p align="center">
  <img src="docs/assets/features.svg" width="100%" alt="Eight models in one codebase, six ready-made alignment tasks, cached splits, ranking and decision metrics, one YAML config per run, campaign runner and tables included"/>
</p>

## How it works

The graph is aligned with itself: the known pairs of a relation are split into training (anchor)
pairs and test pairs; a model embeds every unit from its neighbourhood, and each test query is
ranked against all units.

<p align="center">
  <img src="docs/assets/task.svg" width="100%" alt="The graph, an alignment model trained on anchor pairs, the embedding space, and the ranking of the candidates of a query"/>
</p>

<br/>

## Installation

Requirements: Python 3.9 or later, PyTorch 2.0 or later, and a CUDA GPU for the models (the
baseline runs on CPU).

```bash
git clone https://github.com/Z-Nadjib/EntityAlignment-PGxLOD.git
cd EntityAlignment-PGxLOD
python3 -m venv .venv
.venv/bin/pip install -e code
```

The package installs its dependencies (`torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `PyYAML`)
and an `ea-pgxlod-train` command, equivalent to `python -m src.main`.

## Data

The code reads a CSV extraction of PGxLOD. Point `Data/pgxlod_csv` to the directory that holds it:

```bash
ln -s /path/to/pgxlod_csv Data/pgxlod_csv
```

| File | Columns | Content |
|---|---|---|
| `entities_drug.csv`, `entities_genetic.csv`, `entities_phenotype.csv` | `uri, name, type` | the components and their PGxO class |
| `ucpgx_components.csv` | `ucpgx, predicate, object` | the edges linking each unit to its components |
| `ucpgx_links_unique.csv` | `a, relation, b, oriente` | the known alignments, the targets to recover |

On the first run, the graph and the splits of the six tasks are built and cached under
`Data/cache/`; later runs read the cache. The location of both
directories is set by `data.csv_dir` and `data.cache_dir` in the configuration.

<br/>

## Usage

<p align="center">
  <img src="docs/assets/pipeline.svg" width="100%" alt="From the PGxLOD CSV to the result tables: graph and splits, training, run directory, results, tables"/>
</p>

All commands run from `code/`.

### Train one model on one task

```bash
python -m src.main --config ../configs/rrea.yaml --regime closeMatch --device cuda:0
```

| Option | Description |
|---|---|
| `--config` | YAML configuration file (required) |
| `--regime` | alignment task, see the table below |
| `--model` | overrides `experiment.model`: `gcnalign`, `alinet`, `rrea`, `mraea`, `kecg`, `bootea`, `jape`, `naea` |
| `--device` | `cuda`, `cuda:1`, `cpu`, ... |
| `--epochs` | overrides `train.epochs` |
| `--seed` | overrides the model seed `experiment.seed` |
| `--csv-dir` | overrides `data.csv_dir` |
| `--name` | prefix of the run directory |

| `--regime` | Relation | Configuration file |
|---|---|---|
| `sameAs` | `owl:sameAs` | `configs/<model>.yaml` |
| `closeMatch` | `skos:closeMatch` | `configs/<model>.yaml` |
| `relatedMatch` | `skos:relatedMatch` | `configs/<model>.yaml` |
| `broadMatch/pgx` | `skos:broadMatch`, pharmacogenomic family | `configs/<model>_broadmatch.yaml` |
| `broadMatch/onco` | `skos:broadMatch`, oncology family | `configs/<model>_broadmatch.yaml` |
| `related` | `skos:related` | `configs/<model>_related.yaml` |

### Run the whole benchmark

```bash
python scripts/run_benchmark.py --devices cuda:0 cuda:1                  # 8 models x 6 tasks
python scripts/run_benchmark.py --tasks rrea:closeMatch kecg:related     # selected runs only
```

Runs are distributed over the given devices, one at a time per device. A run already collected
under `logs/` is skipped, so a campaign can be stopped and resumed; `--force` reruns it,
`--index-only` only rebuilds `logs/results.json`.

### Structure-only baseline

```bash
python scripts/run_baseline.py
```

A reference without learning: two units are scored by the cosine of their bags of
(predicate, component) pairs. It runs on CPU in a few minutes and writes its results next to the
models'.

### Result tables

```bash
python scripts/make_tables.py --out ../logs/tables.md      # --score cosine | csls
```

<br/>

## Models

<p align="center">
  <img src="docs/assets/models.svg" width="100%" alt="BootEA, NAEA and JAPE (translation), GCN-Align and AliNet (graph neural networks), MRAEA, RREA and KECG (relation-aware attention), and the structure-only baseline"/>
</p>

Each model lives in `code/src/models/<model>.py`, with its loss, and its training loop in
`code/src/trainer.py`. The encoders and losses follow the official code of each paper; the few
adaptations to PGxLOD are documented in the docstrings and in [`code/README.md`](code/README.md).
The implementations were first checked against the published DBP15K results in
[EntityAlignment-Nexus](https://github.com/Z-Nadjib/EntityAlignment-Nexus).

<br/>

## Configuration

One YAML file drives a run. Each model has three configurations, one per task family, and every
section can be overridden from the command line or edited directly.

```yaml
# excerpt of configs/rrea.yaml
experiment: {name: rrea, model: rrea, seed: 2026, device: cuda}
data:       {csv_dir: Data/pgxlod_csv, cache_dir: Data/cache}
model:      {node_hidden: 100, depth: 2, attn_heads: 1, dropout: 0.3}
train:      {epochs: 400, lr: 0.005, optimizer: rmsprop, gamma: 3.0, early_stop_patience: 10}
eval:       {every: 10, metric: csls, csls_k: 10, select: filtered, clf_tol_k: 10}
logging:    {output_dir: experiments, save_best: true}
```

| Section | Purpose |
|---|---|
| `experiment` | model name, model seed and device |
| `data` | CSV directory, cache directory, and optionally the split seed `data.seed` (2024) |
| `model` | architecture of the encoder |
| `train` | optimiser, learning rate, margins, negative sampling, early stopping |
| `eval` | validation frequency, ranking score (`cosine` or `csls`), block used to select the epoch (`filtered` or `multiref`), top-k of the decision |
| `logging` | run directory, checkpoints, CSV files and plots |

<br/>

## Outputs

Each run writes a timestamped directory under `experiments/`; `run_benchmark.py` then copies its
text artefacts under `logs/`.

```
experiments/rrea_closeMatch_20260929-052009/      logs/
|-- result.json         final test metrics        |-- closeMatch/
|-- config_used.yaml    exact configuration       |   |-- rrea/
|-- training.txt        training log              |   |   |-- result.json
|-- metrics.csv         validation curve          |   |   |-- training.txt
|-- loss.csv            loss curve                |   |   `-- ...
|-- loss_curve.png                                |   `-- baseline/
|-- ranking_metrics.png                           |-- results.json   index of all runs
|-- model_best.pt       best checkpoint           `-- resultat.md    result tables
|-- embeddings.pt
`-- embeddings_ucpgx.npy
```

`result.json` holds the selected epoch, the split statistics and, under `final`, the test metrics:

```
final.csls.filtered       MRR, Hit@1, Hit@5, Hit@10, MeanRank   one rank per test pair
final.csls.multiref       MRR, Hit@1, Hit@5, Hit@10, MeanRank   one rank per query
final.cosine.*            the same blocks with the cosine similarity
final.classification      Precision, Recall, F1, threshold      top-10 decision
```

<br/>

## Benchmark results

Mean over the six tasks of the runs stored in `logs/` (rankings with CSLS, decision with the
cosine similarity). Per-task tables: [`logs/resultat.md`](logs/resultat.md).

<p align="center">
  <img src="docs/assets/results.svg" width="100%" alt="Mean filtered MRR, multiref MRR and decision F1 of the eight models and the baseline"/>
</p>

<br/>

## Adding a model

1. Write the encoder and its loss in `code/src/models/<name>.py`.
2. Add a trainer to `code/src/trainer.py`, subclassing `BaseTrainer` and implementing
   `train_epoch(epoch)` and `embeddings()`; the shared loop handles validation, early stopping,
   checkpoints, curves and the final evaluation.
3. Register a builder for the model in the `builders` dictionary of `code/src/main.py`, and add
   its name to the `--model` choices.
4. Add `configs/<name>.yaml`, `configs/<name>_broadmatch.yaml` and `configs/<name>_related.yaml`,
   and the name to `MODELS` in `code/scripts/run_benchmark.py` and to `MODELS` and `NAMES` in
   `code/scripts/make_tables.py`.

<br/>

## Project structure

```
EntityAlignment-PGxLOD
|-- code
|   |-- src
|   |   |-- benchmark     graph loading, splits, metrics, scoring, baseline
|   |   |-- models        one file per model
|   |   |-- utils         configuration, logging, plotting
|   |   |-- data.py       dataset container and model-specific graph structures
|   |   |-- trainer.py    shared training loop, one trainer per model
|   |   `-- main.py       command-line entry point
|   |-- scripts           run_benchmark.py, run_baseline.py, make_tables.py
|   `-- pyproject.toml
|-- configs               24 YAML files, three per model
|-- docs/assets           figures of this page
`-- logs                  results and training logs of every run
```

<br/>

## Citation

If you use this code, please cite:

```bibtex
@misc{zahaf2026pgxlod,
  author       = {Zahaf, Nadjib},
  title        = {{EntityAlignment-PGxLOD}: Code and Configurations of the Benchmark},
  year         = {2026},
  howpublished = {\url{https://github.com/Z-Nadjib/EntityAlignment-PGxLOD}}
}
```

The benchmark is described in *Matching Pharmacogenomic Knowledge Units in PGxLOD: a Benchmark of
Entity Alignment Methods*, N. B. Zahaf, A. Coulet, J. Legrand and P. Monnin (submitted to
SWAT4HCLS).

## Acknowledgements

Experiments were run on a compute node of the Data Center for Education of CentraleSupélec Metz
and on the [Grid'5000](https://www.grid5000.fr) testbed. PGxLOD is described in
[Monnin et al., BMC Bioinformatics, 2019](https://doi.org/10.1186/s12859-019-2693-9).

## License

Released under the [MIT License](code/LICENSE).

<div align="center">
<br/>
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A3433,50:0B2A33,100:2DD4BF&height=110&section=footer&animation=twinkling" width="100%" alt=""/>
</div>
