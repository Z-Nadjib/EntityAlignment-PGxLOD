# Entity Alignment on PGxLOD

Eight entity-alignment models (GCN-Align, KECG, AliNet, MRAEA, RREA, BootEA, JAPE,
NAEA) applied to **PGxLOD**, a pharmacogenomic knowledge graph, to find the UCPGx
(pharmacogenomic relationship) nodes that denote the same relationship, or a broader or a
related one, under different URIs.

## The task

PGxLOD is a single graph: 50,435 UCPGx linked to 33,747 components (drugs, genetic
factors, phenotypes) by 13 predicates, 131,238 edges. The models embed that graph in one
table, and for a UCPGx query they rank the other 50,434 UCPGx. The gold links between UCPGx
are never edges of the graph: they are the labels to rediscover.

Six regimes, from the strictest relation to the loosest:

| regime | split | train pairs | test pairs | leakage |
|---|---|---:|---:|---:|
| `broadMatch/pgx` | by cluster | 3,744 | 8,599 | 0% |
| `closeMatch` | by cluster | 2,738 | 6,296 | 0% |
| `sameAs` | by cluster | 19,942 | 46,531 | 0% |
| `broadMatch/onco` | intra-cluster | 25,039 | 47,737 | 0% |
| `relatedMatch` | intra-cluster | 2,801 | 6,289 | 0% |
| `related` | intra-cluster | 471,647 | 762,046 | 0% |

* **Split by cluster.** Pairs are grouped by connected component over the relation,
  `sameAs` and the groups of identical verbalisation (UCPGx linking the same entities),
  and a whole group goes to one side. No test pair can be read off a train pair.
* **Intra-cluster.** In the last three regimes one cluster holds more than 88% of the
  pairs, so the cut runs inside it, between groups of identical verbalisation.
* **Validation.** 25% of the train groups are held out with the same grouping. The
  stopping epoch is chosen on them; the model only trains on the rest.
* **Test.** Read once, after training, on the best validation epoch. Ranks average over
  ties and are reported at cosine and at CSLS: `filtered` (per test pair, the other known
  partners masked) and `multiref` (per query, position of its first test partner). A
  top-10 decision F1, on the cosine score, counts a query as correct when one of its test
  partners is in its top ten, its train partners masked; its threshold is chosen on the
  validation set and applied unchanged to the test split (the partner-less negatives are
  split in two disjoint halves, one per side).

## Data

The graph is read from the CSV of the PGxLOD extraction (`entities_*.csv`,
`ucpgx_components.csv`, `ucpgx_links_unique.csv`). `data.csv_dir` points at them; a symlink
is enough:

```bash
ln -s /path/to/csv Data/pgxlod_csv
```

The six splits are built once and cached under `Data/cache/` (a few seconds).

## Layout

```
code/
  src/
    main.py          one model on one regime
    benchmark/       graph, regimes, splits, validation and test scoring, baseline
    data.py          dataset container, model graph structures, samplers, bootstrapping
    trainer.py       shared training loop + one trainer per model
    models/          one file per model
    utils/           config, logger, metric formatting, plotting
  scripts/
    run_benchmark.py every model on every regime, one run at a time per GPU
    run_baseline.py  the structure-only baseline on every regime (CPU, nothing trained)
    make_tables.py   markdown tables from logs/results.json
configs/             <model>.yaml, <model>_broadmatch.yaml, <model>_related.yaml
experiments/         timestamped run dirs (checkpoints, embeddings), not versioned
logs/<regime>/<model>/  the text artefacts of each run, and logs/results.json
```

`configs/<model>_broadmatch.yaml` serves the two broadMatch regimes,
`configs/<model>_related.yaml` serves `related`, `configs/<model>.yaml` the three others.
`experiment.seed` is the model seed (2026); `data.seed` is the split seed (2024 by default)
and does not change with the model seed.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e code
```

## Run

```bash
cd code

# one run
python -m src.main --config ../configs/gcnalign_broadmatch.yaml --regime broadMatch/pgx --device cuda:0

# the whole benchmark (48 runs), then the tables
python scripts/run_benchmark.py --devices cuda:0 cuda:1
python scripts/run_baseline.py
python scripts/make_tables.py --out ../logs/tables.md
```

## The structure-only baseline

A reference that learns nothing (`src/benchmark/baseline.py`). The bag of neighbours of a
UCPGx is the set of its (predicate, component) pairs, and two UCPGx are scored by the cosine
of their bags, counted exactly on integers. The bags go through the scoring of the embedding
tables: same splits, same queries, ranks at cosine and at CSLS, decision F1 at cosine. The
results land in `logs/<regime>/baseline/result.json`, and the tables show them as a last
column, left out of the rank.

## Notes on the models

* **Faithfulness.** The encoders and losses follow the official code of each paper. The
  deviations are listed in the model docstrings: GCN-Align layers carry a weight matrix,
  the KECG knowledge-embedding loss keeps the odd sum of the normalised error vector of the
  official `run.py`, JAPE only has its structure channel (PGxLOD has no attributes), and
  the KECG hard negatives are searched among distinct entities, never the anchor or one of
  its training partners (the official search assumes a one-to-one alignment).
* **Bootstrapping.** RREA/MRAEA with `turns > 1` follow the official RREA code: between two
  turns, the CSLS mutual nearest neighbours between the left and right entities of the
  validation pairs are added to the training pairs. BootEA and NAEA pair up entities of the
  UCPGx outside the training pairs, above a confidence threshold, and never pair an entity
  with itself.
