"""
Dump each run's test-split metrics to `results_*/metrics.json` in the canonical
schema (analysis/_common.py:compute_metrics), so aggregation is mechanical.

This re-runs the eval the train script already does, but writes machine-readable
JSON instead of the human-readable `loss_history.txt` tail. It does NOT touch any
trained artifact other than creating `metrics.json`.

Usage (run inside the `dl_new` conda env):
    # one run:
    python analysis/eval_to_json.py train_model_v3/call/results_vol_surface
    # all 12 (+ any _don / ablation dirs found under train_model_v3):
    python analysis/eval_to_json.py --all
    # smoke test on a subsample:
    python analysis/eval_to_json.py train_model_v3/call/results_vol_surface --max-contracts 20000

Verification gate: on call/results_vol_surface this must print
R2(price)~=0.999851 and R2(log,T>1day)~=0.992481 (matches loss_history.txt).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from _common import (PROJECT_ROOT, compute_metrics, load_run, load_test_split,
                     model_predict_logv, resolve_h5_path)


def eval_one(results_dir: Path, max_contracts=None, stable_log=False, device="cpu"):
    config, arch, model = load_run(results_dir, use_stable_log=stable_log, device=device)
    h5_path = resolve_h5_path(config)
    data = load_test_split(h5_path, config["branch_key"], max_contracts=max_contracts)

    t0 = time.time()
    log_pred, _sigma = model_predict_logv(model, data, device=device)
    elapsed = time.time() - t0

    metrics = compute_metrics(
        log_pred, data["target_v_log"], data["T"], data["log_m"], n_seconds=elapsed,
    )
    metrics["arch"] = arch
    metrics["option_type"] = config["option_type"]
    metrics["branch_key"] = config["branch_key"]
    metrics["results_dir"] = str(results_dir.relative_to(PROJECT_ROOT))
    metrics["stable_log"] = stable_log
    metrics["max_contracts"] = max_contracts
    return metrics


def discover_runs():
    runs = []
    for d in sorted((PROJECT_ROOT / "train_model_v3").glob("*/results_*")):
        if (d / "config.json").exists() and (d / "best_model.pth").exists():
            runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser(description="Dump per-run metrics.json")
    ap.add_argument("results_dir", nargs="?", help="a single results_* dir")
    ap.add_argument("--all", action="store_true", help="sweep all runs under train_model_v3/")
    ap.add_argument("--max-contracts", type=int, default=None)
    ap.add_argument("--stable-log", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    targets = discover_runs() if args.all else [Path(args.results_dir).resolve()]
    if not targets:
        raise SystemExit("No runs found. Pass a results_* dir or --all.")

    for d in targets:
        print(f"\n=== {d} ===")
        m = eval_one(d, max_contracts=args.max_contracts,
                     stable_log=args.stable_log, device=device)
        # Don't overwrite the canonical full-split metrics.json with a subsample
        # or with the eval-only stable-log pricer's numbers.
        out_name = "metrics"
        if args.stable_log:
            out_name += "_stable"
        if args.max_contracts is not None:
            out_name += "_smoke"
        out_name += ".json"
        (d / out_name).write_text(json.dumps(m, indent=2))
        print(f"  arch={m['arch']}  n={m['n_test']}")
        print(f"  R2(price)        = {m['r2_price']:.6f}")
        print(f"  R2(log, T>1day)  = {m['r2_log_filtered']:.6f}   <-- headline")
        print(f"  R2(log, full)    = {m['r2_log_full']:.6f}")
        print(f"  RMSE(log)={m['rmse_log']:.6f}  RMSE(price)={m['rmse_price']:.6f}  "
              f"MAE(price)={m['mae_price']:.6f}")
        print(f"  wrote {out_name}")


if __name__ == "__main__":
    main()
