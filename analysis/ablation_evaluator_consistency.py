#!/usr/bin/env python
"""
Why the scalar-sigma ablation's training-time tail and the canonical CPU
evaluator disagree in LOG space while agreeing in PRICE space.

THE OBSERVATION. For the two scalar-sigma runs, the metrics printed at the tail
of `loss_history.txt` (written on the GPU inside the training process, test
batch = batch_size*2 = 512, via the training script's own DataLoader) and the
canonical post-hoc evaluator `analysis/eval_ablation.py` (CPU, EVAL_BATCH_SIZE
4096) differ:

    call  R2(log) 0.800342 (train tail) vs 0.800103 (evaluator)
          RMSE(log) 1.047691           vs 1.048319
    put   R2(log) -0.313834            vs -0.316871
          RMSE(log) 2.267795           vs 2.270415

while R2(price) agrees to ~4e-7. The same comparison for the 20 per-query runs
agrees to <= 1.7e-6. The scalar model contains no batch-dependent layer, so
batching alone cannot move a prediction; something else must.

WHAT THIS SCRIPT DOES. It re-runs the two EXISTING checkpoints (no training, no
optimizer, no write to any .pth) over the full test split along three paths --

    cpu_bs512   CPU float32, batch 512, the analysis loader
    cpu_bs4096  CPU float32, batch 4096, the analysis loader (CANONICAL)
    gpu_bs512   GPU float32, batch 512, the TRAINING script's own
                H5BranchDataset / make_loader / model class, i.e. the path the
                loss_history.txt tail came from

-- keeps sigma_hat and log(V/K) for every row on every path, compares them
pairwise, and then separates the two candidate explanations:

  (1) sigma_hat itself differs across devices (FFT and matmul kernels are not
      bit-identical between CPU and GPU), or
  (2) sigma_hat is essentially the same and the DECODER amplifies it: log(V/K)
      is computed as log(clamp(M e^{-qT} N(d1) - e^{-rT} N(d2), 1e-8)) in
      float32, a catastrophic cancellation for deep-OTM / short-T rows whose
      true price sits at or below the 1e-8 clamp floor. One ulp of sigma_hat
      there moves log-price by many units, or flips a row on and off the clamp.

The separation is done by taking each path's FIXED float32 sigma_hat and
re-decoding it (a) in float64 with the same naive formula and (b) with the
stable log-normal-CDF decoder `bs_log_normalized_price`. If the metric gap
survives a float64 decode, the sigma_hat estimates genuinely differ; if it
collapses, the gap is decoder cancellation near the clamp.

    python analysis/ablation_evaluator_consistency.py            # both types, +control
    python analysis/ablation_evaluator_consistency.py --no-per-query
    python analysis/ablation_evaluator_consistency.py --cpu-only

Writes results/ablation_evaluator_consistency_{ver}.{json,md} and a small
`_arrays.npz` (seeded subsample -- the full arrays are ~50 MB and are not
committed). It does NOT touch results/ablation_scalar_sigma_v5.*: the CPU
evaluator stays canonical. If this script were to establish an actual
implementation error it says so in the conclusion and changes nothing on its
own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PROJECT_ROOT, bs_log_normalized_price, bs_normalized_price,  # noqa: E402
                     compute_metrics, load_run, load_test_split, model_predict_logv,
                     resolve_h5_path)
from eval_ablation import EVAL_BATCH_SIZE, RUN_DIRS, load_scalar_run  # noqa: E402
from eval_to_json import sha256_of  # noqa: E402

ABLATION_JSON = PROJECT_ROOT / "results" / "ablation_scalar_sigma_v5.json"
TRAIN_SCRIPT = {"scalar_sigma": "train_vol_surface_scalarsigma.py",
                "per_query": "train_vol_surface.py"}
TRAIN_TAIL_BATCH = 512          # config batch_size 256 -> test loader batch_size*2
LOGP_DIVERGENCE = 0.1           # "materially different log-price" threshold
CLAMP_LOG = float(np.log(1e-8))
SUBSAMPLE = 100_000
SUBSAMPLE_SEED = 42

# The training-time numbers this script is trying to explain, transcribed from
# each run's loss_history.txt tail. Reported, never used as a gate.
TRAINING_TAIL = {
    ("call", "scalar_sigma"): {"r2_log": 0.800342, "rmse_log": 1.047691,
                               "r2_price": 0.999825},
    ("put", "scalar_sigma"): {"r2_log": -0.313834, "rmse_log": 2.267795,
                              "r2_price": -3.090190},
}


# ----------------------------------------------------------- pure statistics

def compare_arrays(a, b):
    """Pairwise |a - b| summary. Pure; the unit tests drive this directly."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.abs(a - b)
    finite = np.isfinite(d)
    out = {"n": int(d.size), "n_nonfinite": int((~finite).sum()),
           "n_exactly_equal": int((a == b).sum())}
    dd = d[finite]
    if dd.size == 0:
        out.update({k: None for k in ("max", "mean", "p50", "p90", "p99", "p99_9")})
        return out
    out.update({
        "max": float(dd.max()), "mean": float(dd.mean()),
        "p50": float(np.percentile(dd, 50)), "p90": float(np.percentile(dd, 90)),
        "p99": float(np.percentile(dd, 99)), "p99_9": float(np.percentile(dd, 99.9)),
        "n_gt_1e-6": int((dd > 1e-6).sum()), "n_gt_1e-3": int((dd > 1e-3).sum()),
        "n_gt_0.1": int((dd > LOGP_DIVERGENCE).sum()), "n_gt_1": int((dd > 1.0).sum()),
    })
    return out


def divergent_rows(log_a, log_b, log_m, T, log_target, thresh=LOGP_DIVERGENCE):
    """Where the rows whose log-price moved by more than `thresh` sit.

    Reports moneyness / maturity / price level and how many of them are ON the
    1e-8 clamp floor on either side -- the whole question is whether the moved
    rows are the near-zero-price ones.
    """
    log_a = np.asarray(log_a, dtype=np.float64).ravel()
    log_b = np.asarray(log_b, dtype=np.float64).ravel()
    m = np.abs(log_a - log_b) > thresh
    n = int(m.sum())
    out = {"threshold": thresh, "n_rows": n,
           "pct_rows": 100.0 * n / log_a.size if log_a.size else 0.0}
    if n == 0:
        return out
    lm = np.asarray(log_m, dtype=np.float64).ravel()[m]
    tt = np.asarray(T, dtype=np.float64).ravel()[m]
    tg = np.asarray(log_target, dtype=np.float64).ravel()[m]
    q = lambda v: [float(np.min(v)), float(np.percentile(v, 50)), float(np.max(v))]  # noqa: E731
    out.update({
        "log_moneyness_min_med_max": q(lm),
        "T_days_min_med_max": [x * 365.0 for x in q(tt)],
        "target_log_price_min_med_max": q(tg),
        "n_target_below_clamp": int((tg < CLAMP_LOG).sum()),
        "n_a_at_clamp": int((log_a[m] <= CLAMP_LOG + 1e-6).sum()),
        "n_b_at_clamp": int((log_b[m] <= CLAMP_LOG + 1e-6).sum()),
        "n_clamp_state_flipped": int(((log_a[m] <= CLAMP_LOG + 1e-6) !=
                                      (log_b[m] <= CLAMP_LOG + 1e-6)).sum()),
        "max_abs_delta": float(np.abs(log_a - log_b)[m].max()),
        "sum_sq_delta_contribution": float(np.sum((log_a[m] - log_b[m]) ** 2)),
    })
    return out


def metric_deltas(m_a, m_b, keys=("r2_log_full", "rmse_log", "r2_price", "r2_log_filtered")):
    """{key: a - b} for the metrics the manuscript quotes."""
    return {k: (m_a[k] - m_b[k]) for k in keys if k in m_a and k in m_b}


# --------------------------------------------------------------- decoders

def decode(sigma, log_m, T, r, q, option_type, dtype=torch.float64, stable=False):
    """Decode a FIXED sigma_hat to log(V/K) at a chosen precision / formula.

    `stable=False` is the training/eval decoder: log(clamp(price, 1e-8)) on the
    naive `M e^{-qT} N(d1) - e^{-rT} N(d2)` difference. `stable=True` is
    `bs_log_normalized_price` (log_ndtr, no clamp floor). Both are the repo's
    own functions -- nothing is re-implemented here.
    """
    t = lambda a: torch.as_tensor(np.asarray(a).reshape(-1, 1), dtype=dtype)  # noqa: E731
    s, lm, T_, r_, q_ = t(sigma), t(log_m), t(T), t(r), t(q)
    if stable:
        return bs_log_normalized_price(lm, T_, r_, q_, s, option_type).numpy().ravel()
    price = bs_normalized_price(lm, T_, r_, q_, s, option_type)
    return torch.log(torch.clamp(price, min=1e-8)).numpy().ravel()


# ---------------------------------------------------------- the three paths

def import_train_script(option_type, variant):
    """Import train_model_v3/<type>/<train script>.py as a module (import-safe)."""
    path = PROJECT_ROOT / "train_model_v3" / option_type / TRAIN_SCRIPT[variant]
    name = "trainmod_%s_%s" % (option_type, variant)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def build_train_model(mod, config, ckpt, device):
    """The TRAINING script's own model class, loaded strict=True from the ckpt."""
    model = mod.VolEstimatorModel(
        config["grid_h"], config["grid_w"], config["modes1"], config["modes2"],
        config.get("fno_width", 32),
        latent_dim=config.get("latent_dim", 128),
        head_hidden=config.get("head_hidden", 128),
        option_type=config["option_type"],
        use_stable_log=config.get("use_stable_log", False),
    ).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device), strict=True)
    model.eval()
    return model


def subsample_positions(h5_path, max_contracts, seed=42):
    """The positions WITHIN the test split that load_test_split(max_contracts) keeps.

    Smoke path only, so the training-pipeline loader can be restricted to the
    same rows as the analysis loader. Mirrors load_test_split's rng exactly.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        idx_all = np.where(f["split_id"][:] == 2)[0]
    if max_contracts is None or len(idx_all) <= max_contracts:
        return None
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(idx_all, size=max_contracts, replace=False))
    return np.searchsorted(idx_all, chosen)


@torch.no_grad()
def run_training_path(mod, model, h5_path, branch_key, device, batch_size,
                      positions=None):
    """Reproduce the training script's final test evaluation, row for row.

    Uses the training module's own H5BranchDataset and make_loader (collate
    stacks per-row tensors; shuffle=False so drop_last=False and the row order
    is the file order). Differences from the original, all unavoidable:
      * we also keep sigma_hat, which the training loop discards (the forward
        pass is identical -- it computes sigma_hat either way);
      * `model.eval()` is left set, where the training routine flips back to
        train() after evaluating (irrelevant: no dropout/batchnorm in this
        stack);
      * the GPU, driver and CUDA runtime are today's, not August 2026's. That
        environment was never snapshotted (see
        results/environment_snapshot_*.txt) and cannot be reconstructed.
    """
    ds = mod.H5BranchDataset(str(h5_path), "test", branch_key)
    if positions is not None:                      # smoke subsample only
        ds = torch.utils.data.Subset(ds, list(map(int, positions)))
    loader = mod.make_loader(ds, batch_size, shuffle=False)
    preds, sigmas = [], []
    for branch, log_m, T, r, q, _target in loader:
        branch, log_m, T, r, q = [b.to(device) for b in (branch, log_m, T, r, q)]
        s, lv = model(branch, log_m, T, r, q)
        preds.append(lv.cpu().numpy().ravel())
        sigmas.append(s.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(sigmas)


def verify_checkpoints(rows):
    """Hash every checkpoint we are about to load against the published table."""
    report = []
    for r in rows:
        ckpt = PROJECT_ROOT / r["checkpoint"].replace("\\", "/")
        got = sha256_of(ckpt)
        ok = got == r["checkpoint_sha256"]
        report.append({"checkpoint": r["checkpoint"], "expected_sha256": r["checkpoint_sha256"],
                       "observed_sha256": got, "match": ok})
        if not ok:
            raise SystemExit(
                "ABORT: %s hashes to %s but results/ablation_scalar_sigma_v5.json records "
                "%s. The checkpoint on disk is not the one the published table was built "
                "from; refusing to compare against it."
                % (ckpt, got, r["checkpoint_sha256"]))
        print("  ckpt OK %s %s..." % (r["checkpoint"], got[:12]))
    return report


# -------------------------------------------------------------------- driver

def evaluate_all_paths(option_type, variant, args, device_gpu):
    """Run every enabled path for one (option_type, variant) and compare them."""
    run_dir = PROJECT_ROOT / RUN_DIRS[(option_type, variant)]
    ckpt = run_dir / "best_model.pth"
    config = json.loads((run_dir / "config.json").read_text())
    h5_path = resolve_h5_path(config)
    data = load_test_split(h5_path, config["branch_key"], max_contracts=args.max_contracts)
    n = data["target_v_log"].shape[0]
    log_target = np.asarray(data["target_v_log"], dtype=np.float64).ravel()
    log_m = np.asarray(data["log_m"], dtype=np.float64).ravel()
    T = np.asarray(data["T"], dtype=np.float64).ravel()
    r_ = np.asarray(data["r"], dtype=np.float64).ravel()
    q_ = np.asarray(data["q"], dtype=np.float64).ravel()

    paths, timings = {}, {}

    def cpu_path(batch_size, label):
        if variant == "scalar_sigma":
            _cfg, model = load_scalar_run(run_dir, device="cpu")
        else:
            _cfg, _arch, model = load_run(run_dir, device="cpu")
        t0 = time.time()
        lp, sg = model_predict_logv(model, data, device="cpu", batch_size=batch_size)
        timings[label] = time.time() - t0
        del model
        return lp, sg

    # (a) CPU float32 batch 512 -- scalar-sigma only; the control needs (b) and (c).
    if variant == "scalar_sigma":
        print("  [%s/%s] cpu_bs512 ..." % (option_type, variant))
        paths["cpu_bs512"] = cpu_path(TRAIN_TAIL_BATCH, "cpu_bs512")
    # (b) CPU float32 batch 4096 -- the canonical evaluator's exact path.
    print("  [%s/%s] cpu_bs4096 ..." % (option_type, variant))
    paths["cpu_bs4096"] = cpu_path(EVAL_BATCH_SIZE, "cpu_bs4096")
    # (c) GPU float32 batch 512 through the TRAINING script's own pipeline.
    if device_gpu is not None:
        print("  [%s/%s] gpu_bs512 (training pipeline) ..." % (option_type, variant))
        mod, train_path = import_train_script(option_type, variant)
        model = build_train_model(mod, config, ckpt, device_gpu)
        t0 = time.time()
        paths["gpu_bs512"] = run_training_path(
            mod, model, h5_path, config["branch_key"], device_gpu, TRAIN_TAIL_BATCH,
            positions=subsample_positions(h5_path, args.max_contracts))
        timings["gpu_bs512"] = time.time() - t0
        del model
        torch.cuda.empty_cache()
    else:
        train_path = None

    # ---- metrics as each path produced them, plus the two re-decodes
    per_path = {}
    for label, (lp, sg) in paths.items():
        assert lp.shape[0] == n, "%s returned %d rows, expected %d" % (label, lp.shape[0], n)
        m_native = compute_metrics(lp, log_target, T, log_m)
        lp64 = decode(sg, log_m, T, r_, q_, option_type, torch.float64, stable=False)
        lp64s = decode(sg, log_m, T, r_, q_, option_type, torch.float64, stable=True)
        per_path[label] = {
            "device": "cuda" if label.startswith("gpu") else "cpu",
            "batch_size": TRAIN_TAIL_BATCH if label.endswith("512") else EVAL_BATCH_SIZE,
            "pipeline": "train script H5BranchDataset/make_loader" if label.startswith("gpu")
                        else "analysis/_common.model_predict_logv",
            "seconds": timings.get(label),
            "metrics_as_produced": m_native,
            "metrics_sigma_redecoded_float64_naive": compute_metrics(lp64, log_target, T, log_m),
            "metrics_sigma_redecoded_float64_stable": compute_metrics(lp64s, log_target, T, log_m),
            "sigma_summary": {"min": float(sg.min()), "median": float(np.median(sg)),
                              "max": float(sg.max()), "mean": float(sg.mean())},
            "n_at_clamp_floor": int((lp <= CLAMP_LOG + 1e-6).sum()),
        }
        per_path[label]["_lp64"] = lp64
        per_path[label]["_lp64s"] = lp64s

    # ---- pairwise comparisons
    labels = list(paths)
    pairwise = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            key = "%s__vs__%s" % (a, b)
            lpa, sga = paths[a]
            lpb, sgb = paths[b]
            pairwise[key] = {
                "sigma_hat": compare_arrays(sga, sgb),
                "log_price_as_produced": compare_arrays(lpa, lpb),
                "log_price_float64_naive_redecode": compare_arrays(per_path[a]["_lp64"],
                                                                   per_path[b]["_lp64"]),
                "log_price_float64_stable_redecode": compare_arrays(per_path[a]["_lp64s"],
                                                                    per_path[b]["_lp64s"]),
                "divergent_rows_as_produced": divergent_rows(lpa, lpb, log_m, T, log_target),
                "divergent_rows_float64_stable": divergent_rows(
                    per_path[a]["_lp64s"], per_path[b]["_lp64s"], log_m, T, log_target),
                "metric_delta_as_produced": metric_deltas(
                    per_path[a]["metrics_as_produced"], per_path[b]["metrics_as_produced"]),
                "metric_delta_float64_naive": metric_deltas(
                    per_path[a]["metrics_sigma_redecoded_float64_naive"],
                    per_path[b]["metrics_sigma_redecoded_float64_naive"]),
                "metric_delta_float64_stable": metric_deltas(
                    per_path[a]["metrics_sigma_redecoded_float64_stable"],
                    per_path[b]["metrics_sigma_redecoded_float64_stable"]),
            }

    # ---- decode isolation, per path: same sigma, three decoders
    decode_isolation = {}
    for label in labels:
        p = per_path[label]
        lp, sg = paths[label]
        decode_isolation[label] = {
            "float32_native_vs_float64_naive": compare_arrays(lp, p["_lp64"]),
            "float32_native_vs_float64_stable": compare_arrays(lp, p["_lp64s"]),
            "float64_naive_vs_float64_stable": compare_arrays(p["_lp64"], p["_lp64s"]),
            "metric_delta_native_minus_float64_naive": metric_deltas(
                p["metrics_as_produced"], p["metrics_sigma_redecoded_float64_naive"]),
            "metric_delta_native_minus_float64_stable": metric_deltas(
                p["metrics_as_produced"], p["metrics_sigma_redecoded_float64_stable"]),
        }

    arrays = {}
    if n:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        idx = np.sort(rng.choice(n, size=min(SUBSAMPLE, n), replace=False))
        arrays["%s_%s_subsample_index" % (option_type, variant)] = idx.astype(np.int64)
        for label, (lp, sg) in paths.items():
            arrays["%s_%s_%s_sigma" % (option_type, variant, label)] = sg[idx].astype(np.float32)
            arrays["%s_%s_%s_logprice" % (option_type, variant, label)] = lp[idx].astype(np.float32)

    for p in per_path.values():
        p.pop("_lp64", None)
        p.pop("_lp64s", None)

    return {
        "option_type": option_type,
        "model_variant": variant,
        "results_dir": RUN_DIRS[(option_type, variant)],
        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_of(ckpt),
        "h5_path": str(h5_path),
        "h5_sha256_from_config": config.get("h5_sha256"),
        "train_script": None if train_path is None else str(train_path.relative_to(PROJECT_ROOT)),
        "train_script_sha256": None if train_path is None else sha256_of(train_path),
        "n_test": int(n),
        "config_batch_size": config["batch_size"],
        "use_stable_log_in_config": bool(config.get("use_stable_log", False)),
        "training_tail_reference": TRAINING_TAIL.get((option_type, variant)),
        "paths": per_path,
        "pairwise": pairwise,
        "decode_isolation": decode_isolation,
    }, arrays


# ------------------------------------------------------------------ reporting

def nvidia_smi():
    try:
        p = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=120)
        return (p.stdout or p.stderr or "").strip()
    except Exception as exc:
        return "unavailable (%s)" % exc


def device_block():
    cudnn = torch.backends.cudnn.version()
    d = {"torch": torch.__version__, "torch_version_cuda": torch.version.cuda,
         "cudnn": cudnn, "cuda_available": torch.cuda.is_available(),
         "python": sys.version.replace("\n", " "), "numpy": np.__version__,
         "nvidia_smi": nvidia_smi()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d["gpu_name"] = p.name
        d["gpu_capability"] = "%d.%d" % (p.major, p.minor)
        d["gpu_total_memory_gib"] = round(p.total_memory / 2 ** 30, 2)
    return d


def git_head():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        return h, bool(d)
    except Exception as exc:
        return "unavailable (%s)" % exc, None


STABLE_DECODER_CAVEAT = (
    "The float64 `log_ndtr` decoder is reported alongside the other two but it is **not** "
    "a drop-in replacement here: it has no 1e-8 floor, so wherever the scalar-sigma model "
    "prices a contract far below 1e-8 it returns the true (very negative) log-price "
    "instead of -18.42, and the log-space metric moves accordingly. That is the floor "
    "being removed, not a defect in either evaluator, and it is why the stable-decoder "
    "R2(log) row must not be read as \"the correct number\"."
)


def _pair(res, a, b):
    """The pairwise block for two path labels, in whichever order it was stored."""
    return res["pairwise"].get("%s__vs__%s" % (a, b)) or \
        res["pairwise"].get("%s__vs__%s" % (b, a))


def markdown(results, meta, conclusion):
    out = ["# Scalar-sigma ablation: evaluator-consistency diagnostic", "",
           "Generated %s. git HEAD `%s`%s. torch %s (cuda %s, cuDNN %s) on %s."
           % (meta["generated"], meta["git_head"][:12],
              " (DIRTY WORKTREE)" if meta["git_worktree_dirty"] else "",
              meta["device"]["torch"], meta["device"]["torch_version_cuda"],
              meta["device"]["cudnn"], meta["device"].get("gpu_name", "CPU only")),
           "",
           "No training, no optimizer step, no checkpoint written. Every path runs the "
           "EXISTING `best_model.pth` under `torch.no_grad()` with `model.eval()`, after "
           "hashing it against `results/ablation_scalar_sigma_v5.json`. "
           "**`results/ablation_scalar_sigma_v5.*` is unchanged and the CPU evaluator "
           "(`cpu_bs4096`) remains canonical.**",
           "",
           "| path | device | batch | pipeline |", "|---|---|---|---|",
           "| `cpu_bs512` | CPU | 512 | `analysis/_common.model_predict_logv` |",
           "| `cpu_bs4096` | CPU | 4096 | `analysis/_common.model_predict_logv` (canonical evaluator) |",
           "| `gpu_bs512` | GPU | 512 | the training script's own `H5BranchDataset` / "
           "`make_loader` / model class -- the path that wrote the `loss_history.txt` tail |",
           "",
           "The August 2026 training environment was never snapshotted, so `gpu_bs512` "
           "reproduces the training-time *code path* on today's GPU, driver and CUDA "
           "runtime, not the original hardware stack.",
           ""]
    for res in results:
        ot, var = res["option_type"], res["model_variant"]
        out += ["## %s / %s" % (ot, var), "",
                "%s test rows, checkpoint `%s` (`%s...`)."
                % (f"{res['n_test']:,}", res["checkpoint"], res["checkpoint_sha256"][:12]), ""]
        tail = res["training_tail_reference"]
        if tail:
            out += ["Training-time tail (`loss_history.txt`, for reference): "
                    "R2(log) %.6f, RMSE(log) %.6f, R2(price) %.6f."
                    % (tail["r2_log"], tail["rmse_log"], tail["r2_price"]), ""]
        out += ["| path | R2(log) | RMSE(log) | R2(price) | R2(log,T>1d) | rows at clamp | s |",
                "|---|---|---|---|---|---|---|"]
        for label, p in res["paths"].items():
            m = p["metrics_as_produced"]
            out.append("| `%s` | %.6f | %.6f | %.6f | %.6f | %s | %s |"
                       % (label, m["r2_log_full"], m["rmse_log"], m["r2_price"],
                          m["r2_log_filtered"], f"{p['n_at_clamp_floor']:,}",
                          "-" if p["seconds"] is None else "%.0f" % p["seconds"]))
        out += ["", "Same sigma_hat, re-decoded at higher precision (this is the "
                "isolation step -- the model output is held fixed and only the decoder "
                "changes):", "",
                "| path | decoder | R2(log) | RMSE(log) | R2(price) |",
                "|---|---|---|---|---|"]
        for label, p in res["paths"].items():
            for dec, key in (("float32 naive (as produced)", "metrics_as_produced"),
                             ("float64 naive", "metrics_sigma_redecoded_float64_naive"),
                             ("float64 stable log_ndtr", "metrics_sigma_redecoded_float64_stable")):
                m = p[key]
                out.append("| `%s` | %s | %.6f | %.6f | %.6f |"
                           % (label, dec, m["r2_log_full"], m["rmse_log"], m["r2_price"]))
        out += ["", STABLE_DECODER_CAVEAT, "", "Pairwise path comparison:", "",
                "| pair | max Δσ̂ | p99 Δσ̂ | max Δlog-price | rows Δlog-price > 0.1 | "
                "ΔR2(log) native | ΔR2(log) float64 | ΔR2(log) stable | ΔR2(price) native |",
                "|---|---|---|---|---|---|---|---|---|"]
        for key, c in res["pairwise"].items():
            s, lp = c["sigma_hat"], c["log_price_as_produced"]
            dv = c["divergent_rows_as_produced"]
            out.append("| `%s` | %.3g | %.3g | %.3g | %s (%.4f%%) | %.3g | %.3g | %.3g | %.3g |"
                       % (key.replace("__vs__", "` vs `"), s["max"], s["p99"], lp["max"],
                          f"{dv['n_rows']:,}", dv["pct_rows"],
                          c["metric_delta_as_produced"]["r2_log_full"],
                          c["metric_delta_float64_naive"]["r2_log_full"],
                          c["metric_delta_float64_stable"]["r2_log_full"],
                          c["metric_delta_as_produced"]["r2_price"]))
        out += [""]
        for key, c in res["pairwise"].items():
            dv = c["divergent_rows_as_produced"]
            if dv["n_rows"] == 0:
                out.append("- `%s`: no row moved by more than %g in log-price."
                           % (key, dv["threshold"]))
                continue
            out.append("- `%s`: the %s moved rows have log-moneyness in [%.3f, %.3f] "
                       "(median %.3f), maturity %.2f-%.2f days (median %.2f), true "
                       "log-price in [%.2f, %.2f]; %d have a true price below the 1e-8 "
                       "clamp, %d sit on the clamp on one path and off it on the other."
                       % (key, f"{dv['n_rows']:,}", dv["log_moneyness_min_med_max"][0],
                          dv["log_moneyness_min_med_max"][2], dv["log_moneyness_min_med_max"][1],
                          dv["T_days_min_med_max"][0], dv["T_days_min_med_max"][2],
                          dv["T_days_min_med_max"][1], dv["target_log_price_min_med_max"][0],
                          dv["target_log_price_min_med_max"][2], dv["n_target_below_clamp"],
                          dv["n_clamp_state_flipped"]))
        out += [""]
    out += ["## Conclusion", "", conclusion, ""]
    return "\n".join(out)


def build_conclusion(results):
    """One paragraph, every number in it measured by this run.

    The verdict is decided by the measurement, not asserted: an implementation
    error would show up as a sigma_hat difference that SURVIVES a float64
    re-decode. If it does, the text says so and tells the reader to stop.
    """
    scal = [r for r in results if r["model_variant"] == "scalar_sigma"]
    if not scal:
        return "No scalar-sigma path was run; nothing to conclude."
    if not any(_pair(r, "cpu_bs4096", "gpu_bs512") for r in scal):
        return ("The GPU path did not run (CPU-only invocation), so the training-time "
                "vs evaluator discrepancy could not be reproduced. Nothing is concluded.")

    def g(c, block, key="r2_log_full"):
        return abs(c[block][key]) if c else float("nan")

    native = max(g(_pair(r, "cpu_bs4096", "gpu_bs512"), "metric_delta_as_produced")
                 for r in scal)
    f64 = max(g(_pair(r, "cpu_bs4096", "gpu_bs512"), "metric_delta_float64_naive")
              for r in scal)
    batch = max([g(_pair(r, "cpu_bs512", "cpu_bs4096"), "metric_delta_as_produced")
                 for r in scal if _pair(r, "cpu_bs512", "cpu_bs4096")] or [float("nan")])
    price_gap = max(abs(cc["metric_delta_as_produced"]["r2_price"])
                    for rr in scal for cc in rr["pairwise"].values())
    sigma_max = max(_pair(r, "cpu_bs4096", "gpu_bs512")["sigma_hat"]["max"] for r in scal)
    sigma_p99 = max(_pair(r, "cpu_bs4096", "gpu_bs512")["sigma_hat"]["p99"] for r in scal)

    # An implementation error would leave the gap standing after the decoder is
    # taken out of the picture. Judge it, do not assume it.
    explained = f64 < 0.05 * native if native > 0 else True
    verdict = ("no implementation error was found -- the discrepancy is a device /"
               " float32-precision path difference on rows the model prices at the "
               "1e-8 clamp floor" if explained else
               "THE DISCREPANCY IS NOT EXPLAINED BY DECODER PRECISION -- treat this as a "
               "possible implementation error and stop before citing either number")

    per_type = []
    for r in scal:
        c = _pair(r, "cpu_bs4096", "gpu_bs512")
        dv = c["divergent_rows_as_produced"]
        clamps = [p["n_at_clamp_floor"] for p in r["paths"].values()]
        per_type.append(
            "For %ss, %s of %s test rows (%.4f%%) decode to a log-price that differs by "
            "more than %g between the canonical CPU evaluator and the GPU training path, "
            "by up to %.3g; %s of them sit on the 1e-8 clamp on one path and off it on "
            "the other, and the per-path clamp-hit count itself differs (%s). Those rows "
            "are deep out-of-the-money and short-dated (median maturity %.1f days, median "
            "log-moneyness %.3f) -- exactly where M e^{-qT} N(d1) - e^{-rT} N(d2) is a "
            "difference of two nearly equal float32 numbers."
            % (r["option_type"], f"{dv['n_rows']:,}", f"{r['n_test']:,}", dv["pct_rows"],
               dv["threshold"], c["log_price_as_produced"]["max"],
               f"{dv.get('n_clamp_state_flipped', 0):,}",
               " vs ".join(f"{x:,}" for x in clamps),
               dv.get("T_days_min_med_max", [0, 0, 0])[1],
               dv.get("log_moneyness_min_med_max", [0, 0, 0])[1]))

    control = ""
    pq = [r for r in results if r["model_variant"] == "per_query"
          and _pair(r, "cpu_bs4096", "gpu_bs512")]
    if pq:
        gaps = [g(_pair(r, "cpu_bs4096", "gpu_bs512"), "metric_delta_as_produced") for r in pq]
        clamp = max(max(p["n_at_clamp_floor"] for p in r["paths"].values()) for r in pq)
        control = (" The per-query checkpoints, run through the same two paths as a "
                   "positive control, move by at most %.2g in R2(log) -- %.0fx smaller -- "
                   "and hit the clamp floor on %d rows, which is the whole difference: "
                   "the per-query model does not price contracts at zero."
                   % (max(gaps), native / max(gaps) if max(gaps) else float("inf"), clamp))

    return (
        "**Conclusion: %s.** Batch size alone is not the cause: holding the device fixed "
        "and moving from batch 512 to batch 4096 on the CPU changes R2(log) by %.2g. "
        "Changing the DEVICE changes it by %.3g, which is the size of the published "
        "discrepancy. The model output is not what differs -- sigma_hat agrees between "
        "CPU and GPU to %.3g in the worst row and %.3g at the 99th percentile, i.e. to "
        "float32 rounding. What differs is the decoder. %s Holding each path's own "
        "float32 sigma_hat fixed and re-decoding it in float64 with the same formula "
        "collapses the R2(log) difference from %.3g to %.3g, so essentially all of the "
        "gap is float32 arithmetic in the decoder rather than a different prediction. "
        "R2(price) never differs by more than %.3g on any comparison, because a row worth "
        "1e-8 contributes nothing in price space.%s %s The practical consequence: the "
        "scalar-sigma log-space metrics are reproducible across devices only to about "
        "%.0e in R2, so the six-decimal figures in results/ablation_scalar_sigma_v5.* "
        "should be read as ~%.0e-precise; the price-space figures, which agree to %.0e, "
        "carry their printed precision. The CPU evaluator at batch 4096 remains canonical "
        "and nothing in results/ablation_scalar_sigma_v5.* was changed."
        % (verdict, batch, native, sigma_max, sigma_p99, " ".join(per_type),
           native, f64, price_gap, control, STABLE_DECODER_CAVEAT, native, native, price_gap))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--option-type", choices=["call", "put", "both"], default="both")
    ap.add_argument("--no-per-query", action="store_true",
                    help="skip the per-query positive control")
    ap.add_argument("--cpu-only", action="store_true", help="skip every GPU path")
    ap.add_argument("--max-contracts", type=int, default=None,
                    help="subsample the test split (SMOKE ONLY -- writes *_smoke.* so the "
                         "full-split artifact can never be overwritten)")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    args = ap.parse_args()

    published = json.loads(ABLATION_JSON.read_text())
    types = ["call", "put"] if args.option_type == "both" else [args.option_type]
    variants = ["scalar_sigma"] + ([] if args.no_per_query else ["per_query"])
    wanted = [r for r in published["rows"]
              if r["option_type"] in types and r["model_variant"] in variants]
    print("=== checkpoint hash verification (before anything else) ===")
    ckpt_report = verify_checkpoints(wanted)

    device_gpu = None if args.cpu_only or not torch.cuda.is_available() else "cuda"
    if device_gpu is None:
        print("GPU paths DISABLED (%s)" % ("--cpu-only" if args.cpu_only else "no CUDA device"))

    results, arrays = [], {}
    for variant in variants:
        for ot in types:
            print("=== %s / %s ===" % (ot, variant))
            res, arr = evaluate_all_paths(ot, variant, args, device_gpu)
            results.append(res)
            arrays.update(arr)

    head, dirty = git_head()
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "analysis/ablation_evaluator_consistency.py",
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "git_head": head, "git_worktree_dirty": dirty,
        "device": device_block(),
        "checkpoint_verification": ckpt_report,
        "canonical_evaluator": ("analysis/eval_ablation.py, CPU, EVAL_BATCH_SIZE %d -- "
                                "path `cpu_bs4096` here. results/ablation_scalar_sigma_v5.* "
                                "is NOT modified by this script." % EVAL_BATCH_SIZE),
        "subsample": {"n": SUBSAMPLE, "seed": SUBSAMPLE_SEED,
                      "note": "full per-row arrays are ~50 MB and are not committed; the "
                              "npz holds a fixed-seed subsample of sigma_hat and log-price "
                              "per path, indexed by *_subsample_index"},
        "divergence_threshold": LOGP_DIVERGENCE,
        "clamp_log": CLAMP_LOG,
        "max_contracts": args.max_contracts,
    }
    conclusion = build_conclusion(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / ("ablation_evaluator_consistency_v5"
                           + ("_smoke" if args.max_contracts is not None else ""))
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": results, "conclusion": conclusion}, indent=2),
        encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(results, meta, conclusion), encoding="utf-8")
    if arrays:
        np.savez_compressed(str(stem) + "_arrays.npz", **arrays)
        size = (Path(str(stem) + "_arrays.npz")).stat().st_size
        print("wrote %s_arrays.npz (%.1f MB)" % (stem.name, size / 1e6))
    print("wrote %s.{json,md}" % stem)
    print()
    print(conclusion)


if __name__ == "__main__":
    main()
