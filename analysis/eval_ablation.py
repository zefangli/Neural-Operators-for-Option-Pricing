"""
Scalar-sigma ABLATION table: the canonical per-query vol_surface runs vs their
scalar-sigma counterparts (sigma_hat from the market-state latent alone).

Deliberately a SEPARATE table from T1, and deliberately unable to reach T1:
this script writes NOTHING into any `results_*/` run directory. `aggregate_results.py`
globs `train_model_v3/*/results_*/metrics.json`, so dropping a metrics.json into
`results_vol_surface_scalarsigma_v5/` would make an ablation row glob-discoverable
and would collide with the real call/branch_u/A cell. All output lives in
results/ablation_scalar_sigma_<ver>.{json,csv,md}.

Runs are identified by the explicit `model_variant` config field (per_query vs
scalar_sigma), never by directory name -- a directory can be renamed or copied.

WHICH NUMBERS ARE CANONICAL (applies to this script, eval_to_json.py and
aggregate_results.py alike): the independent evaluator's are. They are computed
on the FULL test split, at a fixed evaluation batch size, under torch.no_grad()
with the model in eval(), from the saved `best_model.pth` -- deterministic and
reproducible. The metrics printed at the tail of a run's `loss_history.txt` are
training-time, preliminary numbers (different batching, computed inside the
training loop) and differ slightly for the same run. Cite the evaluator's
numbers in the manuscript; never a loss_history.txt tail.

Provenance is enforced, not merely recorded: every run must be seed 42, on the
same dataset version / HDF5 sha256 / quote filter, with the same test-row count
per option type (same gates as the training provenance block and
aggregate_results.validate_canonical). A subsample (`--max-contracts`) writes to
`*_smoke` filenames and can never overwrite the canonical table.

Usage (inside the `dl_new` conda env, CPU is fine):
    python analysis/eval_ablation.py [--max-contracts N]
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import (PROJECT_ROOT, VolEstimatorModelA, compute_metrics,
                     load_run, load_test_split, manifest_entry,
                     model_predict_logv, resolve_h5_path)
from eval_to_json import CANONICAL_SEED, config_dataset_version, run_provenance, sha256_of

EVAL_BATCH_SIZE = 4096   # fixed, so the canonical numbers are batching-independent

# (option_type, variant) -> results dir. The four runs this table compares.
RUN_DIRS = {
    ("call", "per_query"):   "train_model_v3/call/results_vol_surface_v5",
    ("put",  "per_query"):   "train_model_v3/put/results_vol_surface_v5",
    ("call", "scalar_sigma"): "train_model_v3/call/results_vol_surface_scalarsigma_v5",
    ("put",  "scalar_sigma"): "train_model_v3/put/results_vol_surface_scalarsigma_v5",
}


class ScalarVolHead(nn.Module):
    """Ablation head: latent -> sigma_hat. Mirrors train_vol_surface_scalarsigma.py.

    Same depth/width as the per-query head minus the 2 query inputs, so
    `head.net.0` is Linear(latent_dim, hidden). state_dict prefix `head.net.*`.
    """
    def __init__(self, latent_dim=128, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent):
        return F.softplus(self.net(latent))


class ScalarSigmaModel(VolEstimatorModelA):
    """Same encoder/pricer as model A; sigma_hat ignores the per-contract query."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.head = ScalarVolHead(latent_dim=kw.get("latent_dim", 128),
                                  hidden=kw.get("head_hidden", 128))

    def sigma(self, branch, log_moneyness, T_years):
        return self.head(self.encoder(branch))


def load_scalar_run(results_dir: Path, device="cpu"):
    config = json.loads((results_dir / "config.json").read_text())
    model = ScalarSigmaModel(
        grid_h=config["grid_h"], grid_w=config["grid_w"],
        modes1=config["modes1"], modes2=config["modes2"],
        width=config.get("fno_width", 32), option_type=config["option_type"],
        latent_dim=config.get("latent_dim", 128),
        head_hidden=config.get("head_hidden", 128),
    ).to(device)
    model.load_state_dict(torch.load(str(results_dir / "best_model.pth"),
                                     map_location=device), strict=True)
    model.eval()
    return config, model


def _quote_filter(dataset_version):
    """The manifest's quote-filter description for a sample ("" if unknown/smoke)."""
    try:
        return str(manifest_entry(dataset_version).get("quote_filter") or "")
    except SystemExit:
        return ""


def validate_provenance(rows):
    """Refuse to write an ablation table whose four runs are not comparable.

    Same class of gate as aggregate_results.validate_canonical: off-seed, mixed
    sample, mixed HDF5 hash, mixed quote filter, non-canonical (e.g. smoke-built)
    HDF5, or a differing test-row count within one option type all HARD-FAIL. A
    silently-proceeding comparison is how a v4 scalar-sigma row lands next to a
    v5 per-query row and gets called an ablation.
    """
    errs = []
    seeds = {r.get("seed") for r in rows}
    if seeds != {CANONICAL_SEED}:
        errs.append(f"OFF-SEED run(s): seeds={sorted(map(str, seeds))}, want {CANONICAL_SEED} only")
    vers = {r.get("dataset_version") for r in rows}
    if len(vers) != 1:
        errs.append(f"MIXED dataset_version across the runs: {sorted(map(str, vers))}")
    qfs = {r.get("quote_filter") for r in rows}
    if len(qfs) != 1:
        errs.append(f"MIXED quote_filter across the runs: {sorted(map(str, qfs))}")
    noncanon = [r["results_dir"] for r in rows if r.get("canonical_dataset") is False]
    if noncanon:
        errs.append("NON-CANONICAL HDF5 (sha256 not in the manifest, e.g. a smoke build): "
                    + ", ".join(sorted(noncanon)))
    for opt in ("call", "put"):
        sub = [r for r in rows if r.get("option_type") == opt]
        hashes = {r.get("h5_sha256") for r in sub}
        if len(hashes) > 1:
            errs.append(f"MIXED {opt} HDF5 sha256: "
                        + ", ".join(sorted(str(h)[:12] + "..." for h in hashes)))
        ns = {r.get("n_test") for r in sub}
        if len(ns) > 1:
            errs.append(f"MIXED {opt} test-row count across variants: {sorted(map(str, ns))}")
    if errs:
        raise SystemExit("REFUSING TO WRITE the ablation table:\n  " + "\n  ".join(errs))


def output_stem(out_dir, ver, max_contracts):
    """Canonical stem, or a `_smoke` one for a subsample.

    `--max-contracts` is a dev-sized subsample; it must never overwrite the
    full-test-split table that the manuscript cites.
    """
    suffix = "_smoke" if max_contracts is not None else ""
    return Path(out_dir) / f"ablation_scalar_sigma_{ver}{suffix}"


def eval_one(option_type, variant, max_contracts=None, device="cpu"):
    d = PROJECT_ROOT / RUN_DIRS[(option_type, variant)]
    if variant == "scalar_sigma":
        config, model = load_scalar_run(d, device=device)
    else:
        config, _arch, model = load_run(d, device=device)

    # The point of the new provenance fields: identify the architecture by FIELD.
    got = config.get("model_variant")
    if got != variant:
        raise SystemExit(
            f"REFUSING to build the ablation table: {d} has model_variant={got!r}, "
            f"expected {variant!r}. Do not infer architecture from the dir name."
        )
    if config["option_type"] != option_type:
        raise SystemExit(f"{d}: option_type={config['option_type']!r}, want {option_type!r}")

    h5_path = resolve_h5_path(config)
    # Same gate eval_to_json.py runs before writing any metrics.json: config vs
    # on-file schema_version / dataset_version / sha256. Reused, not re-invented.
    prov = run_provenance(config, h5_path)

    data = load_test_split(h5_path, config["branch_key"], max_contracts=max_contracts)
    log_pred, _sigma = model_predict_logv(model, data, device=device,
                                          batch_size=EVAL_BATCH_SIZE)
    m = compute_metrics(log_pred, data["target_v_log"], data["T"], data["log_m"])
    ckpt = d / "best_model.pth"
    return {
        "option_type": option_type,
        "model_variant": variant,
        "sigma_conditioning": config.get("sigma_conditioning"),
        "results_dir": RUN_DIRS[(option_type, variant)],
        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_of(ckpt),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "device": device,
        "seed": config.get("seed"),
        "dataset_version": prov["dataset_version"] or config_dataset_version(config),
        "h5_sha256": prov["h5_sha256"],
        "canonical_dataset": prov["canonical_dataset"],
        "quote_filter": _quote_filter(prov["dataset_version"]),
        "n_test": m["n_test"],
        "r2_price": m["r2_price"],
        "r2_log_filtered": m["r2_log_filtered"],
        "rmse_log": m["rmse_log"],
        "rmse_price": m["rmse_price"],
    }


HEADLINE = (
    "**Headline: the per-query architecture substantially outperforms the scalar-sigma "
    "control, with the larger degradation on puts.** Removing the per-contract "
    "conditioning (sigma_hat from the market-state latent alone, i.e. flat-vol "
    "Black-Scholes with no smile/skew) drops call R2(log,T>1d) from {c_pq_log:.6f} to "
    "{c_sc_log:.6f} and call R2(price) from {c_pq_p:.6f} to {c_sc_p:.6f}; on puts it "
    "drops R2(log,T>1d) from {p_pq_log:.6f} to {p_sc_log:.6f} and R2(price) from "
    "{p_pq_p:.6f} to {p_sc_p:.6f} -- i.e. **negative**, worse than predicting the "
    "mean price. Seeds: 42 only; the effect size is large relative to the seed-to-seed "
    "dispersion reported in results/replication_seeds_{ver}.md.\n\n"
    "This table is NOT part of T1 (results/table_T1_{ver}.md) and must never be "
    "merged into it: it reports a restricted scalar-sigma control architecture, not a "
    "candidate model.\n\n"
    "Numbers here come from the independent evaluator (full test split, fixed "
    "evaluation batch size) and are the canonical ones; training-tail numbers in "
    "loss_history.txt are preliminary and are not cited."
)


def main():
    ap = argparse.ArgumentParser(description="Scalar-sigma ablation comparison table")
    ap.add_argument("--max-contracts", type=int, default=None,
                    help="subsample the test split (smoke only -- not for the manuscript)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    rows = []
    for option_type in ("call", "put"):
        for variant in ("per_query", "scalar_sigma"):
            print(f"=== {option_type} / {variant} ===")
            rows.append(eval_one(option_type, variant, args.max_contracts, device))
            r = rows[-1]
            print(f"  R2(price)={r['r2_price']:.6f}  R2(log,T>1d)={r['r2_log_filtered']:.6f}")

    validate_provenance(rows)
    ver = rows[0]["dataset_version"]

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stem = output_stem(out_dir, ver, args.max_contracts)
    if args.max_contracts is not None:
        print(f"SUBSAMPLE (--max-contracts {args.max_contracts}): writing {stem.name}.* — "
              "the canonical full-test-split table is NOT overwritten.")
    stem.with_suffix(".json").write_text(json.dumps(
        {"dataset_version": ver, "max_contracts": args.max_contracts,
         "eval_batch_size": EVAL_BATCH_SIZE, "device": device,
         "canonical_numbers": "This independent, full-test-split evaluation is canonical; "
                              "loss_history.txt tail numbers are preliminary.",
         "note": "Ablation only. Never merge into T1.", "rows": rows}, indent=2))

    keys = list(rows[0])
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    by = {(r["option_type"], r["model_variant"]): r for r in rows}
    md = [HEADLINE.format(
        ver=ver,
        c_pq_log=by[("call", "per_query")]["r2_log_filtered"],
        c_sc_log=by[("call", "scalar_sigma")]["r2_log_filtered"],
        c_pq_p=by[("call", "per_query")]["r2_price"],
        c_sc_p=by[("call", "scalar_sigma")]["r2_price"],
        p_pq_log=by[("put", "per_query")]["r2_log_filtered"],
        p_sc_log=by[("put", "scalar_sigma")]["r2_log_filtered"],
        p_pq_p=by[("put", "per_query")]["r2_price"],
        p_sc_p=by[("put", "scalar_sigma")]["r2_price"],
    ), "",
        "| option | variant | sigma conditioning | R2(price) | R2(log,T>1d) | RMSE(log) | RMSE(price) |",
        "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {option_type} | {model_variant} | {sigma_conditioning} | "
                  "{r2_price:.6f} | {r2_log_filtered:.6f} | {rmse_log:.6f} | "
                  "{rmse_price:.6f} |".format(**r))
    stem.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"\nwrote results/{stem.name}.{{json,csv,md}} "
          "(nothing written into any run dir -- invisible to aggregate_results.py)")
    print(stem.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
