"""
Dump each run's test-split metrics to `results_*/metrics.json` in the canonical
schema (analysis/_common.py:compute_metrics), so aggregation is mechanical.

This re-runs the eval the train script already does, but writes machine-readable
JSON instead of the human-readable `loss_history.txt` tail. It does NOT touch any
trained artifact other than creating `metrics.json`.

Every metrics.json carries the provenance of the sample it was computed on
(`dataset_version`, `schema_version`, `h5_path`, `h5_sha256`, `seed`), and this
script REFUSES to write one when the run's config.json disagrees with the HDF5
actually being read -- that mismatch is how a v3 number reaches a v4 table.

Dataset identity comes from `wrds_data_2020-2025/DATASET_MANIFEST.json` -- which
versions exist, what their files hash to, and which one is canonical. No hash and
no canonical version is hard-coded here. `dataset_version` (the SAMPLE) and
`schema_version` (the tensor layout) are separate fields and neither is inferred
from the other: v5 is a different sample carrying the same schema "v4".

Usage (run inside the `dl_new` conda env):
    # one run (deliberate, explicit -- any dataset version):
    python analysis/eval_to_json.py train_model_v3/call/results_vol_surface_v5
    # the canonical sweep (the ONLY safe form of --all):
    python analysis/eval_to_json.py --all --dataset-version v5
    # smoke test on a subsample:
    python analysis/eval_to_json.py train_model_v3/call/results_vol_surface --max-contracts 20000

Verification gate: on call/results_vol_surface this must print
R2(price)~=0.999851 and R2(log,T>1day)~=0.992481 (matches loss_history.txt).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import torch

from _common import (PROJECT_ROOT, canonical_version, compute_metrics,
                     h5_attr, load_manifest, load_run, load_test_split,
                     model_predict_logv, resolve_h5_path)
from aggregate_results import is_smoke, warn_smoke

CANONICAL_SEED = 42


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def h5_schema_version(h5_path) -> str:
    """`schema_version` attr as a str; "" when absent (that IS the v3 signature)."""
    with h5py.File(h5_path, "r") as f:
        v = f.attrs.get("schema_version", "")
    return v.decode() if isinstance(v, bytes) else str(v)


def _prov_block(config: dict) -> dict:
    """The 8 non-VIX scripts write `data_provenance`, the 4 VIX ones `vix_provenance`."""
    return config.get("data_provenance") or config.get("vix_provenance") or {}


def _config_field(config: dict, key: str) -> str:
    """A provenance field the training scripts write BOTH at cfg top level and inside
    the nested provenance block. Either alone is accepted (v3/v4-era configs only have
    one); a disagreement is not resolvable here and stops the pipeline.
    """
    top = str(config.get(key, ""))
    nested = str(_prov_block(config).get(key, ""))
    if top and nested and top != nested:
        raise SystemExit(
            f"PROVENANCE MISMATCH: config top-level {key}={top!r} but "
            f"data_provenance/vix_provenance says {nested!r}.\n"
            "  The run's config.json is internally inconsistent; do not evaluate it."
        )
    return top or nested


def config_schema_version(config: dict) -> str:
    """schema_version (TENSOR LAYOUT) the training run recorded. "" for v3-era."""
    return _config_field(config, "schema_version")


def config_dataset_version(config: dict) -> str:
    """dataset_version (SAMPLE) the training run recorded, "" if it recorded none.

    Deliberately NOT derived from schema_version: v5 files carry schema "v4".
    """
    return _config_field(config, "dataset_version")


def version_by_hash(sha256: str, manifest_path=None):
    """-> (dataset_version, option_type) for a hash the manifest knows, else None.

    Hashes identify a sample exactly, so this is the primary identity source and
    the reason no hash is hard-coded in this file.
    """
    if not sha256:
        return None
    for ver, e in load_manifest(manifest_path).get("datasets", {}).items():
        for opt, rec in (e.get("files") or {}).items():
            if rec.get("sha256") == sha256:
                return str(e.get("dataset_version", ver)), opt
    return None


def resolve_dataset_version(sha256, declared_dataset, schema, manifest_path=None):
    """Identify the SAMPLE. -> (dataset_version, matched_manifest_hash: bool).

    Order: manifest hash match (exact) > an explicitly recorded/on-file
    `dataset_version` > the legacy shim below. The shim exists only because every
    v3/v4 artifact predates the `dataset_version` attr; it is NOT general
    inference of one axis from the other, and it never yields "v5" -- a v5 file
    always states its own dataset_version.
    """
    hit = version_by_hash(sha256, manifest_path)
    if hit:
        return hit[0], True
    if declared_dataset:
        return declared_dataset, False
    return ("v4" if schema == "v4" else "v3"), False


def run_provenance(config: dict, h5_path: Path, rehash: bool = False) -> dict:
    """Provenance stamp for one run, refusing config/HDF5 disagreement.

    Hashing a 8-13 GB HDF5 takes minutes, so by default we REUSE the hash the
    training run already computed and stored in config["h5_sha256"] -- it was
    taken from this same file at train time, and the cheap checks (on-file
    schema_version, and the canonical-hash comparison below) already catch the
    realistic failure mode of resolve_h5_path() silently falling back to the v3
    file. `--rehash` recomputes and hard-fails on any drift; use it once before
    submission, not on every eval.
    """
    declared, on_file = config_schema_version(config), h5_schema_version(h5_path)
    if declared != on_file:
        raise SystemExit(
            f"REFUSING TO WRITE metrics.json: provenance mismatch.\n"
            f"  config.json says schema_version={declared or '(none => v3)'!r}\n"
            f"  but the HDF5 actually being evaluated is {h5_path}\n"
            f"  which carries schema_version={on_file or '(none => v3)'!r}.\n"
            "  The config's h5_path probably does not exist here and a different sample\n"
            "  was resolved. Point it at the right HDF5 instead."
        )
    declared_ds = config_dataset_version(config)
    on_file_ds = h5_attr(h5_path, "dataset_version") or ""
    if declared_ds and on_file_ds and declared_ds != on_file_ds:
        raise SystemExit(
            f"REFUSING TO WRITE metrics.json: dataset (SAMPLE) mismatch.\n"
            f"  config.json says dataset_version={declared_ds!r}\n"
            f"  but {h5_path} carries dataset_version={on_file_ds!r}.\n"
            "  Same schema does not mean same sample (v5 is schema v4)."
        )

    cached = config.get("h5_sha256")
    if rehash or not cached:
        actual = sha256_of(h5_path)
        if cached and actual != cached:
            raise SystemExit(
                f"REFUSING TO WRITE metrics.json: {h5_path} hashes to\n  {actual}\n"
                f"but config.json recorded\n  {cached}\n"
                "  The file changed since training; the run's results are not reproducible."
            )
        cached = actual

    version, hash_known = resolve_dataset_version(
        cached, declared_ds or on_file_ds, on_file)
    # canonical_dataset = "this IS the manifest's file for its own dataset version"
    # (a smoke build of the same schema is not). v3 predates the manifest's hashes
    # entirely, so it cannot be checked and stays true, as before.
    canonical = hash_known or version == "v3"
    if not canonical:
        print(f"  WARNING: NOT a manifest-listed sample (sha256={cached[:12]}...) —")
        print("    likely a smoke build. Stamped canonical_dataset=false; it will be")
        print("    excluded from aggregate_results.py tables.")
    return {
        "seed": config.get("seed"),
        "dataset_version": version,          # the SAMPLE
        "schema_version": on_file or "v3 (attr absent)",   # the TENSOR LAYOUT
        "is_canonical_version": version == canonical_version(),
        "h5_path": str(h5_path),
        "h5_sha256": cached,
        "canonical_dataset": canonical,
    }


def eval_one(results_dir: Path, max_contracts=None, stable_log=False, device="cpu",
             rehash=False):
    config, arch, model = load_run(results_dir, use_stable_log=stable_log, device=device)
    h5_path = resolve_h5_path(config)
    prov = run_provenance(config, h5_path, rehash=rehash)
    data = load_test_split(h5_path, config["branch_key"], max_contracts=max_contracts)

    t0 = time.time()
    log_pred, _sigma = model_predict_logv(model, data, device=device)
    elapsed = time.time() - t0

    metrics = compute_metrics(
        log_pred, data["target_v_log"], data["T"], data["log_m"], n_seconds=elapsed,
        # v4/v5 only; absent in v3 -> no liquidity strata reported. Reporting
        # strata only: no row is dropped for liquidity anywhere.
        best_bid=data.get("best_bid"), best_offer=data.get("best_offer"),
    )
    metrics["arch"] = arch
    metrics["option_type"] = config["option_type"]
    metrics["branch_key"] = config["branch_key"]
    metrics["results_dir"] = str(results_dir.relative_to(PROJECT_ROOT))
    metrics["stable_log"] = stable_log
    metrics["max_contracts"] = max_contracts
    metrics.update(prov)
    return metrics


def discover_runs(dataset_version):
    """Evaluable runs under train_model_v3/ trained on the SAMPLE `dataset_version`.

    Selection is by recorded provenance, never by directory name: the run's
    recorded HDF5 sha256 is looked up in the manifest (exact), falling back to the
    config's own `dataset_version`, then the legacy schema shim. `results_*_v4`
    and `results_*_v5` are told apart by what they were trained on, not by what
    someone named the folder. 1-batch `*_smoke` dirs are skipped so a smoke dir
    never even gets a metrics.json written into it.
    """
    runs, skipped_smoke, skipped_version = [], [], []
    for d in sorted((PROJECT_ROOT / "train_model_v3").glob("*/results_*")):
        if not ((d / "config.json").exists() and (d / "best_model.pth").exists()):
            continue
        if is_smoke(d):
            skipped_smoke.append(d)
            continue
        cfg = json.loads((d / "config.json").read_text())
        got, _ = resolve_dataset_version(cfg.get("h5_sha256"),
                                         config_dataset_version(cfg),
                                         config_schema_version(cfg))
        if got != dataset_version:
            skipped_version.append((d, got))
            continue
        runs.append(d)
    warn_smoke(skipped_smoke)
    if skipped_version:
        print(f"WARNING: {len(skipped_version)} run(s) excluded — not dataset-version "
              f"{dataset_version} (this is the point of --dataset-version):")
        for d, got in skipped_version:
            print(f"  - {d.relative_to(PROJECT_ROOT)}  (dataset_version={got})")
    return runs


def main():
    ap = argparse.ArgumentParser(description="Dump per-run metrics.json")
    ap.add_argument("results_dir", nargs="?", help="a single results_* dir")
    ap.add_argument("--all", action="store_true", help="sweep all runs under train_model_v3/")
    ap.add_argument("--dataset-version", choices=("v3", "v4", "v5"),
                    help="REQUIRED with --all: only sweep runs trained on this dataset "
                         "SAMPLE (resolved via DATASET_MANIFEST.json from the recorded "
                         "hash, not from dir names or schema_version). A single "
                         "explicitly-named dir needs no selector — naming it is already "
                         "a deliberate act.")
    ap.add_argument("--max-contracts", type=int, default=None)
    ap.add_argument("--stable-log", action="store_true")
    ap.add_argument("--rehash", action="store_true",
                    help="recompute the HDF5 sha256 instead of reusing config.json's "
                         "(minutes on a 10 GB file; run once before submission)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if args.all:
        if not args.dataset_version:
            raise SystemExit(
                "--all requires --dataset-version {v3,v4,v5}. Without it the sweep also "
                "discovers the historical v3/v4 runs and mixes samples in one table.\n"
                "  The canonical sample is %s (per DATASET_MANIFEST.json)."
                % canonical_version()
            )
        targets = discover_runs(args.dataset_version)
    else:
        if not args.results_dir:
            raise SystemExit("Pass a results_* dir, or --all --dataset-version v4.")
        targets = [Path(args.results_dir).resolve()]
    if not targets:
        raise SystemExit("No runs found. Pass a results_* dir or --all.")

    for d in targets:
        print(f"\n=== {d} ===")
        m = eval_one(d, max_contracts=args.max_contracts,
                     stable_log=args.stable_log, device=device, rehash=args.rehash)
        # Don't overwrite the canonical full-split metrics.json with a subsample
        # or with the eval-only stable-log pricer's numbers.
        out_name = "metrics"
        if args.stable_log:
            out_name += "_stable"
        if args.max_contracts is not None:
            out_name += "_smoke"
        out_name += ".json"
        (d / out_name).write_text(json.dumps(m, indent=2))
        print(f"  arch={m['arch']}  n={m['n_test']}  seed={m['seed']}  "
              f"dataset={m['dataset_version']}  sha256={str(m['h5_sha256'])[:12]}...")
        print(f"  R2(price)        = {m['r2_price']:.6f}")
        print(f"  R2(log, T>1day)  = {m['r2_log_filtered']:.6f}   <-- headline")
        print(f"  R2(log, full)    = {m['r2_log_full']:.6f}")
        print(f"  RMSE(log)={m['rmse_log']:.6f}  RMSE(price)={m['rmse_price']:.6f}  "
              f"MAE(price)={m['mae_price']:.6f}")
        print(f"  wrote {out_name}")


if __name__ == "__main__":
    main()
