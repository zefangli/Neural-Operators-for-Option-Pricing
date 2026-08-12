"""
Substantiate the "exact greeks" claim: autograd through the Black-Scholes decoder
vs the closed-form greeks, at the model's predicted sigma_hat, on sampled test
contracts. Expect agreement to ~1e-7 (fp32: ~1e-6).

Conventions (normalized price c = V/K, M = S/K = exp(log_moneyness)):
  c_call = M e^{-qT} N(d1) - e^{-rT} N(d2)
  delta_M := dc/dM = e^{-qT} N(d1)            (call),  -e^{-qT} N(-d1)   (put)
  vega    := dc/dsigma = M e^{-qT} n(d1) sqrt(T)        (same for call/put)
We differentiate c w.r.t. log_moneyness and recover dc/dM = (dc/dlog_m)/M, and
w.r.t. sigma directly. sigma is fixed at the model's sigma_hat (greeks are BS
greeks AT sigma_hat, not through the encoder).

Usage:  python analysis/greeks_check.py [results_dir] [--n 2000]
Default results_dir = train_model_v3/call/results_vol_surface.
Writes results/greeks_check.json (+ arrays for figure F6).
Cheap (a few k contracts, one forward + autograd) -> safe to run on CPU.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from _common import (PROJECT_ROOT, bs_normalized_price, load_run,
                     load_test_split, resolve_h5_path)


def _norm_pdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf_t(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def closed_form_greeks(log_m, T, r, q, sigma, option_type):
    M = torch.exp(log_m)
    sqrt_T = torch.sqrt(torch.clamp(T, min=1e-10))
    sig = torch.clamp(sigma, min=1e-10)
    d1 = (log_m + (r - q + 0.5 * sig ** 2) * T) / (sig * sqrt_T)
    if option_type == "call":
        delta_M = torch.exp(-q * T) * _norm_cdf_t(d1)
    else:
        delta_M = -torch.exp(-q * T) * _norm_cdf_t(-d1)
    vega = M * torch.exp(-q * T) * _norm_pdf(d1) * sqrt_T
    return delta_M, vega


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?",
                    default=str(PROJECT_ROOT / "train_model_v3/call/results_vol_surface"))
    ap.add_argument("--n", type=int, default=2000, help="number of sampled test contracts")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    config, arch, model = load_run(results_dir, device="cpu")
    h5_path = resolve_h5_path(config)
    data = load_test_split(h5_path, config["branch_key"], max_contracts=args.n)
    option_type = config["option_type"]

    branch = torch.as_tensor(data["branch"], dtype=torch.float32)
    log_m0 = torch.as_tensor(data["log_m"], dtype=torch.float32)
    T = torch.as_tensor(data["T"], dtype=torch.float32)

    # sigma_hat is produced by the (fp32) network, then the greeks comparison is
    # done in float64 so autograd-vs-closed-form agreement is limited by the BS
    # math, not fp32 round-off. This lets us honestly report ~1e-7 agreement.
    with torch.no_grad():
        sigma_hat = model.sigma(branch, log_m0, T).detach().to(torch.float64)

    log_m0 = log_m0.to(torch.float64)
    T = T.to(torch.float64)
    r = torch.as_tensor(data["r"], dtype=torch.float64)
    q = torch.as_tensor(data["q"], dtype=torch.float64)

    # Autograd through BS at sigma_hat. Differentiate the scalar sum of prices;
    # because each contract's price depends only on its own row, the per-row grad
    # equals d(price_i)/d(input_i). NOTE: this checks that autograd through the BS
    # decoder equals the analytic BS greek AT the fixed sigma_hat -- both come from
    # the same BS formula, so it validates the differentiable-decoder claim, not an
    # independent oracle, and NOT the total (sticky-strike) delta that would include
    # d sigma_hat/dS * vega.
    log_m = log_m0.clone().requires_grad_(True)
    sigma = sigma_hat.clone().requires_grad_(True)
    price = bs_normalized_price(log_m, T, r, q, sigma, option_type)
    (g_logm,) = torch.autograd.grad(price.sum(), log_m, create_graph=False, retain_graph=True)
    (g_sigma,) = torch.autograd.grad(price.sum(), sigma)

    M = torch.exp(log_m0)
    delta_auto = (g_logm / M).detach()            # dc/dM
    vega_auto = g_sigma.detach()
    delta_cf, vega_cf = closed_form_greeks(log_m0, T, r, q, sigma_hat, option_type)

    def stats(auto, cf):
        a = auto.numpy().ravel()
        c = cf.numpy().ravel()
        absdiff = np.abs(a - c)
        denom = np.maximum(np.abs(c), 1e-8)
        return {
            "max_abs_diff": float(absdiff.max()),
            "median_abs_diff": float(np.median(absdiff)),
            "max_rel_diff": float((absdiff / denom).max()),
        }

    out = {
        "results_dir": str(results_dir.relative_to(PROJECT_ROOT)),
        "arch": arch,
        "option_type": option_type,
        "n_sampled": int(branch.shape[0]),
        "delta": stats(delta_auto, delta_cf),
        "vega": stats(vega_auto, vega_cf),
    }
    OUT = PROJECT_ROOT / "results"
    OUT.mkdir(exist_ok=True)
    (OUT / "greeks_check.json").write_text(json.dumps(out, indent=2))
    # Arrays for figure F6 (autograd vs closed-form scatter).
    np.savez(OUT / "greeks_check_arrays.npz",
             delta_auto=delta_auto.numpy().ravel(), delta_cf=delta_cf.numpy().ravel(),
             vega_auto=vega_auto.numpy().ravel(), vega_cf=vega_cf.numpy().ravel())
    print(json.dumps(out, indent=2))
    if out["delta"]["max_abs_diff"] > 1e-6 or out["vega"]["max_abs_diff"] > 1e-6:
        print("WARNING: greeks disagreement exceeds 1e-6 (float64 claim is ~1e-7). Investigate.")


if __name__ == "__main__":
    main()
