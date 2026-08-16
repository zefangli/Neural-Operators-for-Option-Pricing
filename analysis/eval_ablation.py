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
                     load_run, load_test_split, model_predict_logv,
                     resolve_h5_path)
from eval_to_json import config_dataset_version

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

    data = load_test_split(resolve_h5_path(config), config["branch_key"],
                           max_contracts=max_contracts)
    log_pred, _sigma = model_predict_logv(model, data, device=device)
    m = compute_metrics(log_pred, data["target_v_log"], data["T"], data["log_m"])
    return {
        "option_type": option_type,
        "model_variant": variant,
        "sigma_conditioning": config.get("sigma_conditioning"),
        "results_dir": RUN_DIRS[(option_type, variant)],
        "seed": config.get("seed"),
        "dataset_version": config_dataset_version(config),
        "n_test": m["n_test"],
        "r2_price": m["r2_price"],
        "r2_log_filtered": m["r2_log_filtered"],
        "rmse_log": m["rmse_log"],
        "rmse_price": m["rmse_price"],
    }


HEADLINE = (
    "**Headline: the per-query architecture decisively outperforms the scalar-sigma "
    "ablation, and the gap is catastrophic on puts.** Removing the per-contract "
    "conditioning (sigma_hat from the market-state latent alone, i.e. flat-vol "
    "Black-Scholes with no smile/skew) drops call R2(log,T>1d) from {c_pq_log:.6f} to "
    "{c_sc_log:.6f} and call R2(price) from {c_pq_p:.6f} to {c_sc_p:.6f}; on puts it "
    "drops R2(log,T>1d) from {p_pq_log:.6f} to {p_sc_log:.6f} and R2(price) from "
    "{p_pq_p:.6f} to {p_sc_p:.6f} -- i.e. **negative**, worse than predicting the "
    "mean price. Seeds: 42 only; the gap needs no replication to be decisive.\n\n"
    "This table is NOT part of T1 (results/table_T1_{ver}.md) and must never be "
    "merged into it: it reports a deliberately crippled architecture."
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

    vers = {r["dataset_version"] for r in rows}
    if len(vers) != 1:
        raise SystemExit(f"MIXED dataset versions across the four runs: {sorted(vers)}")
    ver = vers.pop()

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f"ablation_scalar_sigma_{ver}"
    stem.with_suffix(".json").write_text(json.dumps(
        {"dataset_version": ver, "max_contracts": args.max_contracts,
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
    print(f"\nwrote results/ablation_scalar_sigma_{ver}.{{json,csv,md}} "
          "(nothing written into any run dir -- invisible to aggregate_results.py)")
    print(stem.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
