"""Black-Scholes decoder contract tests (`analysis/_common.py`).

Run either way:
    conda run -n dl_new python -m pytest tests/ -q
    conda run -n dl_new python tests/test_bs_pricer.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from _common import bs_log_normalized_price, bs_normalized_price  # noqa: E402

torch.manual_seed(0)


def _grid(n=400, seed=0):
    """A well-conditioned sample: T in [0.05, 2], sigma in [0.08, 0.6], |log_m| < 0.4."""
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)
    return dict(log_m=u(-0.4, 0.4), T=u(0.05, 2.0), r=u(0.0, 0.06),
                q=u(0.0, 0.03), sigma=u(0.08, 0.6))


def test_put_call_parity():
    """C - P = M*exp(-qT) - exp(-rT) in normalized V/K units."""
    g = _grid()
    a = (g["log_m"], g["T"], g["r"], g["q"], g["sigma"])
    c = bs_normalized_price(*a, "call")
    p = bs_normalized_price(*a, "put")
    parity = torch.exp(g["log_m"] - g["q"] * g["T"]) - torch.exp(-g["r"] * g["T"])
    err = (c - p - parity).abs().max().item()
    assert err < 1e-12, "put-call parity broken by %.3e" % err


def test_monotone_in_sigma():
    """Vega > 0: price is strictly increasing in sigma for T > 0."""
    g = _grid(200)
    sig = torch.linspace(0.05, 1.5, 40, dtype=torch.float64)
    for otype in ("call", "put"):
        prices = torch.stack([
            bs_normalized_price(g["log_m"], g["T"], g["r"], g["q"],
                                torch.full_like(g["log_m"], float(s)), otype)
            for s in sig
        ])                                        # (n_sigma, n_contracts)
        d = prices[1:] - prices[:-1]
        assert d.min().item() > -1e-14, "%s not monotone in sigma (min diff %.3e)" % (
            otype, d.min().item())
        assert d.max().item() > 1e-6, "%s price does not respond to sigma at all" % otype


def test_deep_itm_otm_limits():
    """Deep OTM -> 0; deep ITM -> the discounted-forward intrinsic."""
    T = torch.tensor([0.5], dtype=torch.float64)
    r = torch.tensor([0.03], dtype=torch.float64)
    q = torch.tensor([0.01], dtype=torch.float64)
    s = torch.tensor([0.20], dtype=torch.float64)
    deep, shallow = torch.tensor([4.0], dtype=torch.float64), torch.tensor([-4.0], dtype=torch.float64)
    fwd_deep = torch.exp(deep - q * T)
    fwd_shal = torch.exp(shallow - q * T)
    disc = torch.exp(-r * T)

    assert bs_normalized_price(shallow, T, r, q, s, "call").item() < 1e-12
    assert abs(bs_normalized_price(deep, T, r, q, s, "call").item()
               - (fwd_deep - disc).item()) < 1e-10
    assert bs_normalized_price(deep, T, r, q, s, "put").item() < 1e-12
    assert abs(bs_normalized_price(shallow, T, r, q, s, "put").item()
               - (disc - fwd_shal).item()) < 1e-10


def test_T_to_zero_converges_to_intrinsic():
    """As T -> 0 the price -> max(M-1,0) / max(1-M,0)."""
    log_m = torch.tensor([-0.20, -0.02, 0.02, 0.20], dtype=torch.float64)
    z = torch.zeros_like(log_m)
    s = torch.full_like(log_m, 0.20)
    for otype, intr in (("call", torch.clamp(torch.exp(log_m) - 1.0, min=0.0)),
                        ("put", torch.clamp(1.0 - torch.exp(log_m), min=0.0))):
        prev = float("inf")
        for T in (1e-2, 1e-3, 1e-4, 1e-6):
            e = (bs_normalized_price(log_m, torch.full_like(log_m, T), z, z, s, otype)
                 - intr).abs().max().item()
            assert e <= prev + 1e-15, "%s: error not shrinking as T->0" % otype
            prev = e
        assert prev < 1e-6, "%s: T->0 limit off intrinsic by %.3e" % (otype, prev)


def test_T_exactly_zero_is_sigma_degenerate():
    """The load-bearing fact behind the R2(log) ceiling: at T == 0 the decoder
    returns intrinsic for EVERY sigma, so those rows carry no learnable signal.
    See analysis/diagnostics.py."""
    log_m = torch.tensor([-0.20, -0.05, 0.05, 0.20], dtype=torch.float64)
    z = torch.zeros_like(log_m)
    T0 = torch.zeros_like(log_m)
    for otype, intr in (("call", torch.clamp(torch.exp(log_m) - 1.0, min=0.0)),
                        ("put", torch.clamp(1.0 - torch.exp(log_m), min=0.0))):
        px = torch.stack([
            bs_normalized_price(log_m, T0, z, z, torch.full_like(log_m, float(s)), otype)
            for s in np.linspace(0.01, 3.0, 30)
        ])
        spread = (px.max(0).values - px.min(0).values).max().item()
        assert spread < 1e-9, "%s: T=0 price still moves with sigma by %.3e" % (otype, spread)
        assert (px - intr).abs().max().item() < 1e-9


def test_stable_log_matches_plain_log():
    """bs_log_normalized_price == log(bs_normalized_price) where both are well
    conditioned -- the invariant --stable-log relies on.

    "Well conditioned" has to be stated as a price-magnitude cut, not a parameter
    box: once the normalized price drops below ~1e-6 the plain formula is two
    nearly-equal terms subtracting, and it is the PLAIN one that is wrong."""
    g = _grid(500, seed=7)
    for otype in ("call", "put"):
        price = bs_normalized_price(g["log_m"], g["T"], g["r"], g["q"], g["sigma"], otype)
        stable = bs_log_normalized_price(g["log_m"], g["T"], g["r"], g["q"],
                                         g["sigma"], otype)
        ok = price > 1e-6
        assert ok.float().mean().item() > 0.8, "%s: sample too degenerate to test" % otype
        err = (torch.log(price[ok]) - stable[ok]).abs().max().item()
        assert err < 1e-9, "%s: stable vs plain log disagree by %.3e" % (otype, err)


def test_stable_log_beats_plain_log_when_price_underflows():
    """The converse, and the reason --stable-log exists: for a deep-OTM
    short-dated contract the plain path loses the price entirely to cancellation
    (and then to the 1e-8 clamp), while log_ndtr still resolves it."""
    log_m = torch.tensor([-0.5], dtype=torch.float64)     # deep OTM call
    T = torch.tensor([0.02], dtype=torch.float64)
    z = torch.zeros_like(log_m)
    s = torch.tensor([0.10], dtype=torch.float64)
    plain = bs_normalized_price(log_m, T, z, z, s, "call").item()
    stable = bs_log_normalized_price(log_m, T, z, z, s, "call").item()
    assert plain < 1e-30, "sample is not actually underflowing (plain=%.3e)" % plain
    assert stable < -100.0, "stable log should still resolve the tiny price"
    assert np.log(max(plain, 1e-8)) > stable + 50.0, (
        "the 1e-8 clamp should be dramatically above the true log price here")


def test_price_stays_inside_static_bounds():
    """Model output can never violate the bounds analysis/data_quality.py checks
    the DATA against -- so every violation there is a data defect, not a model one."""
    g = _grid(500, seed=3)
    fwd = torch.exp(g["log_m"] - g["q"] * g["T"])
    disc = torch.exp(-g["r"] * g["T"])
    c = bs_normalized_price(g["log_m"], g["T"], g["r"], g["q"], g["sigma"], "call")
    p = bs_normalized_price(g["log_m"], g["T"], g["r"], g["q"], g["sigma"], "put")
    assert (c >= torch.clamp(fwd - disc, min=0.0) - 1e-12).all()
    assert (c <= fwd + 1e-12).all()
    assert (p >= torch.clamp(disc - fwd, min=0.0) - 1e-12).all()
    assert (p <= disc + 1e-12).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS %s" % f.__name__)
    print("%d/%d passed" % (len(fns), len(fns)))
