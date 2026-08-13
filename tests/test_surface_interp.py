"""The 11x17 IV-surface layout contract, plus a bilinear lookup on a synthetic
surface with a known analytic sigma(days, delta).

Layout, from `wrds_data_2020-2025/pre_process_1_data.py::_build_vol_surface_lazy`:
the raw rows are sorted by ["date", "cp_flag", "days", "delta"] and then collected
into one list per (date, cp_flag). Sorting on `days` first makes the flat 187-vector
**days-major, delta-minor**, i.e.

    flat[i * 17 + j] == sigma(days[i], delta[j])   ->   reshape(11, 17)[i, j]

which is exactly what `FNO_MarketEncoder.forward` assumes when it does
`x.view(-1, 1, grid_h, grid_w)` with (grid_h, grid_w) = (11, 17). Get this
transposed and the FNO reads the surface sideways with no error raised anywhere,
so it is worth an assert.

The interpolation helper lives here on purpose: it is used only by this test.

Run either way:
    conda run -n dl_new python -m pytest tests/ -q
    conda run -n dl_new python tests/test_surface_interp.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

N_DAYS, N_DELTA, FLAT_DIM = 11, 17, 187
# OptionMetrics standardised axes: 11 tenors x 17 deltas (call-delta convention).
DAYS_AXIS = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730], float)
DELTA_AXIS = np.arange(10, 95, 5, dtype=float)          # 10, 15, ..., 90  -> 17 points

assert DAYS_AXIS.size == N_DAYS and DELTA_AXIS.size == N_DELTA


def sigma_true(days, delta):
    """A smooth analytic surface: term structure + a delta smile."""
    days, delta = np.asarray(days, float), np.asarray(delta, float)
    return 0.18 + 0.03 * np.log(days / 30.0) + 8e-5 * (delta - 50.0) ** 2


def build_surface():
    """(11, 17) grid of sigma_true, and its days-major flattening."""
    grid = sigma_true(DAYS_AXIS[:, None], DELTA_AXIS[None, :])
    return grid, grid.reshape(-1)


def bilinear(grid, days, delta, days_axis=DAYS_AXIS, delta_axis=DELTA_AXIS):
    """Bilinear lookup on the (days, delta) grid, clamped at the edges."""
    def _w(axis, v):
        i = np.clip(np.searchsorted(axis, v) - 1, 0, axis.size - 2)
        t = (v - axis[i]) / (axis[i + 1] - axis[i])
        return i, np.clip(t, 0.0, 1.0)

    i, ti = _w(days_axis, np.asarray(days, float))
    j, tj = _w(delta_axis, np.asarray(delta, float))
    return ((1 - ti) * (1 - tj) * grid[i, j] + ti * (1 - tj) * grid[i + 1, j]
            + (1 - ti) * tj * grid[i, j + 1] + ti * tj * grid[i + 1, j + 1])


# ------------------------------------------------------------------ tests


def test_flatten_reshape_roundtrip_is_days_major():
    grid, flat = build_surface()
    assert flat.shape == (FLAT_DIM,)
    for i in range(N_DAYS):
        for j in range(N_DELTA):
            assert flat[i * N_DELTA + j] == grid[i, j], "flatten is not days-major"
    assert np.array_equal(flat.reshape(N_DAYS, N_DELTA), grid)
    # the transposed reading must be detectably different (guards the assert above)
    assert not np.allclose(flat.reshape(N_DELTA, N_DAYS).T, grid)


def test_row_blocks_share_a_tenor():
    """Consequence of days-major: each contiguous 17-block is ONE tenor across
    all deltas. If the layout were delta-major the block would be one delta
    across tenors, which the smile/term-structure signs below would flip."""
    _, flat = build_surface()
    for i, d in enumerate(DAYS_AXIS):
        block = flat[i * N_DELTA:(i + 1) * N_DELTA]
        assert np.allclose(block, sigma_true(d, DELTA_AXIS))
        # smile within a tenor: minimum at delta = 50, i.e. the middle of the block
        assert int(np.argmin(block)) == int(np.argmin(np.abs(DELTA_AXIS - 50.0)))
    # term structure across tenors at fixed delta: monotone increasing in days
    col = flat.reshape(N_DAYS, N_DELTA)[:, 8]
    assert np.all(np.diff(col) > 0)


def test_lookup_exact_at_grid_nodes():
    grid, _ = build_surface()
    for i, d in enumerate(DAYS_AXIS):
        for j, k in enumerate(DELTA_AXIS):
            got = float(bilinear(grid, d, k))
            assert abs(got - grid[i, j]) < 1e-12, "node lookup off by %.3e" % abs(got - grid[i, j])


def test_lookup_recovers_analytic_between_nodes():
    """Bilinear error is bounded by the surface curvature over a cell. The worst
    cell here is 10->30 days, where the tenor grid is coarsest relative to the
    log term structure: ~2.4% relative, everything else far below."""
    grid, _ = build_surface()
    rng = np.random.default_rng(0)
    d = rng.uniform(DAYS_AXIS[0], DAYS_AXIS[-1], 2000)
    k = rng.uniform(DELTA_AXIS[0], DELTA_AXIS[-1], 2000)
    rel = np.abs(bilinear(grid, d, k) - sigma_true(d, k)) / sigma_true(d, k)
    assert rel.max() < 3e-2, "max relative interpolation error %.4f" % rel.max()
    assert np.median(rel) < 2e-3, "median relative error %.5f" % np.median(rel)
    # the term-structure error is concentrated in the short-tenor cell; past 60
    # days only the ~3e-3 delta-smile floor remains (see the log-days test)
    assert rel[d > 60].max() < 5e-3, "long-tenor error %.5f" % rel[d > 60].max()
    assert rel[d < 30].max() > 5 * rel[d > 60].max()


def test_interpolating_in_log_days_is_much_more_accurate():
    """Directly useful for any surface baseline: the tenor axis is geometric
    (10,30,60,...,730), so interpolate in log(days), not days. Same 11x17 grid,
    ~8x less error. What is left (~3e-3) is the delta-smile curvature, which no
    change of tenor variable can remove: 8e-5 * h^2/8 * 2 over a 5-wide delta
    cell / 0.18 ~= 2.8e-3."""
    grid, _ = build_surface()
    rng = np.random.default_rng(0)
    d = rng.uniform(DAYS_AXIS[0], DAYS_AXIS[-1], 2000)
    k = rng.uniform(DELTA_AXIS[0], DELTA_AXIS[-1], 2000)
    lin = np.abs(bilinear(grid, d, k) - sigma_true(d, k)) / sigma_true(d, k)
    log = np.abs(bilinear(grid, np.log(d), k, days_axis=np.log(DAYS_AXIS))
                 - sigma_true(d, k)) / sigma_true(d, k)
    assert log.max() < 5e-3, "log-days max relative error %.5f" % log.max()
    assert log.max() < lin.max() / 5.0, "log-days gained only %.1fx" % (lin.max() / log.max())


def test_lookup_exact_for_a_bilinear_surface():
    """On a function that IS bilinear in the grid coordinates the interpolant
    must be exact -- this separates interpolation-scheme bugs from curvature."""
    x = (DAYS_AXIS - DAYS_AXIS[0]) / (DAYS_AXIS[-1] - DAYS_AXIS[0])
    y = (DELTA_AXIS - DELTA_AXIS[0]) / (DELTA_AXIS[-1] - DELTA_AXIS[0])
    # bilinear inside every cell only if built per-cell; use a piecewise-linear
    # separable form, which bilinear interpolation reproduces exactly.
    grid = 0.2 + 0.1 * x[:, None] + 0.05 * y[None, :] + 0.3 * x[:, None] * y[None, :]
    rng = np.random.default_rng(1)
    dd = rng.uniform(DAYS_AXIS[0], DAYS_AXIS[-1], 500)
    kk = rng.uniform(DELTA_AXIS[0], DELTA_AXIS[-1], 500)
    # exact reference: interpolate x, y linearly within the same cell
    def lin(axis, v, vals):
        i = np.clip(np.searchsorted(axis, v) - 1, 0, axis.size - 2)
        t = (v - axis[i]) / (axis[i + 1] - axis[i])
        return i, vals[i] + t * (vals[i + 1] - vals[i])

    _, xv = lin(DAYS_AXIS, dd, x)
    _, yv = lin(DELTA_AXIS, kk, y)
    want = 0.2 + 0.1 * xv + 0.05 * yv + 0.3 * xv * yv
    err = np.abs(bilinear(grid, dd, kk) - want).max()
    assert err < 1e-12, "bilinear interpolant not exact on a bilinear surface (%.3e)" % err


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS %s" % f.__name__)
    print("%d/%d passed" % (len(fns), len(fns)))
