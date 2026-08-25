"""Run the benchmark: every model on every regime.

    python scripts/run_benchmark.py --devices cuda:0 cuda:1
    python scripts/run_benchmark.py --tasks kecg:related bootea:related

One process per run, one run at a time per device. A run whose result is already in
``logs/`` is skipped, so the campaign can be relaunched without losing anything.

Each run writes to ``experiments/<model>_<regime>_<stamp>/``; its text artefacts are then
copied under ``logs/<regime>/<model>/`` and the index ``logs/results.json`` is rebuilt.
Checkpoints and embeddings stay in ``experiments/``.

The configuration of a (model, regime) pair is ``configs/<model>_broadmatch.yaml`` for the
two broadMatch regimes, ``configs/<model>_related.yaml`` for related and
``configs/<model>.yaml`` for the others.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
LOGS = ROOT / "logs"

# From fastest to slowest.
MODELS = ["gcnalign", "kecg", "alinet", "mraea", "rrea", "bootea", "jape", "naea"]
REGIMES = ["broadMatch/pgx", "closeMatch", "sameAs", "broadMatch/onco", "relatedMatch",
           "related"]
CONFIG_GROUP = {"broadMatch": "_broadmatch", "related": "_related"}

# Text artefacts kept next to the results; checkpoints and embeddings stay in experiments/.
ARTEFACTS = ["result.json", "training.txt", "metrics.csv", "loss.csv", "config_used.yaml",
             "loss_curve.png", "ranking_metrics.png"]

PRINT_LOCK = threading.Lock()


def config_for(model: str, regime: str) -> Path:
    """The configuration of a run, see the module docstring."""
    group = CONFIG_GROUP.get(regime.split("/")[0], "")
    return ROOT / "configs" / f"{model}{group}.yaml"


def run_dir_of(model: str, regime: str) -> Path:
    return LOGS / regime.replace("/", "-") / model


def collect(model: str, regime: str) -> str:
    """Copy the artefacts of the newest matching run under logs/<regime>/<model>/."""
    prefix = f"{model}_{regime.replace('/', '-')}_"
    for run in sorted((ROOT / "experiments").glob(prefix + "*"), reverse=True):
        if not (run / "result.json").exists():
            continue
        dest = run_dir_of(model, regime)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ARTEFACTS:
            if (run / name).exists():
                shutil.copy(run / name, dest / name)
        return "ok"
    return "finished without result.json"


def index() -> int:
    """Rebuild logs/results.json from the collected runs."""
    results = []
    for path in sorted(LOGS.glob("*/*/result.json")):
        results.append(json.loads(path.read_text()))
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    return len(results)


def one_run(model: str, regime: str, device: str, timeout: float, force: bool,
            seed: int | None) -> str:
    dest = run_dir_of(model, regime)
    if (dest / "result.json").exists() and not force:
        return "already done"
    log = dest.with_suffix(".console.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.main", "--config", str(config_for(model, regime)),
           "--model", model, "--regime", regime, "--device", device]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    started = time.time()
    with open(log, "w") as handle:
        try:
            done = subprocess.run(cmd, cwd=CODE, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=dict(os.environ))
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout / 60:.0f} min"
    if done.returncode != 0:
        return f"failed with code {done.returncode}, see {log.name}"
    state = collect(model, regime)
    return f"{state} ({(time.time() - started) / 60:.1f} min)" if state == "ok" else state


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--regimes", nargs="+", default=REGIMES)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="explicit list of model:regime pairs instead of the full grid")
    parser.add_argument("--timeout", type=float, default=9000.0, help="seconds per run")
    parser.add_argument("--force", action="store_true", help="rerun runs already collected")
    parser.add_argument("--seed", type=int, default=None,
                        help="model seed, overrides experiment.seed of the configs (2026)")
    parser.add_argument("--index-only", action="store_true",
                        help="rebuild results.json and exit")
    args = parser.parse_args()

    if args.index_only:
        print(f"{index()} results indexed in {LOGS / 'results.json'}")
        return

    pending: queue.Queue = queue.Queue()
    if args.tasks:
        for task in args.tasks:
            model, regime = task.split(":", 1)
            pending.put((model, regime))
    else:
        for regime in args.regimes:
            for model in args.models:
                pending.put((model, regime))
    total, done = pending.qsize(), [0]

    # Build the split cache once, before the workers start: two runs launched together on
    # an empty cache would otherwise both build it.
    subprocess.run([sys.executable, "-c",
                    "from pathlib import Path; from src import benchmark; "
                    f"benchmark.load(Path({str(ROOT / 'Data' / 'pgxlod_csv')!r}), "
                    f"cache_dir=Path({str(ROOT / 'Data' / 'cache')!r}))"],
                   cwd=CODE, check=True)

    def worker(device: str):
        while True:
            try:
                model, regime = pending.get_nowait()
            except queue.Empty:
                return
            with PRINT_LOCK:
                print(f"[{time.strftime('%H:%M:%S')}] {device} -> {model} {regime}", flush=True)
            state = one_run(model, regime, device, args.timeout, args.force, args.seed)
            with PRINT_LOCK:
                done[0] += 1
                print(f"[{time.strftime('%H:%M:%S')}] {device} <- {model} {regime}: {state}"
                      f"   ({done[0]}/{total})", flush=True)
                index()
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(device,)) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"campaign over, {index()} results indexed")


if __name__ == "__main__":
    main()
