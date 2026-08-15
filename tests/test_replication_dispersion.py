"""Replication table must never show a fake zero for unmeasured dispersion.

Regression test: `write_replication` used to run over ALL cells (every branch),
and assigned `std=0.0` whenever a cell had fewer than 2 seeds. Once seed 43/44
vol_surface replications land, every spot_history/vix_history cell still has
exactly one seed -- a naive std=0.0 there would misrepresent "we never measured
dispersion for this cell" as "dispersion is measured to be zero". Fixed by (a)
restricting the replication table to vol_surface cells only -- the only ones
with a deliberate multi-seed plan -- and (b) using None/"N/A" rather than 0.0
whenever a surviving cell still has fewer than 2 seeds.

Run: pytest tests/test_replication_dispersion.py  or  python tests/test_replication_dispersion.py
"""
import csv
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import aggregate_results as agg  # noqa: E402


def _row(option, branch, arch, seed, r2_price):
    """A minimal synthetic metrics.json record -- only the fields write_replication reads."""
    return {"option_type": option, "branch_key": branch, "arch": arch, "seed": seed,
            "r2_price": r2_price, "r2_log_filtered": r2_price - 0.01,
            "r2_log_full": r2_price - 0.01, "rmse_log": 0.5}


def _synthetic_rows():
    rows = []
    # 4 vol_surface cells, real 3-seed replication (42/43/44), distinct values -> real std
    for option in ("call", "put"):
        for arch in ("A", "B"):
            base = 0.99 if option == "call" else 0.98
            rows += [_row(option, "branch_u", arch, s, base + i * 0.001)
                     for i, s in enumerate((42, 43, 44))]
    # spot_history / vix_history: single seed only, exactly the current real state
    rows.append(_row("call", "spot_history", "A", 42, 0.92))
    rows.append(_row("put", "vix_history", "B", 42, 0.92))
    return rows


def test_single_seed_branches_excluded_entirely():
    rows = _synthetic_rows()
    with tempfile.TemporaryDirectory() as td:
        orig = agg.OUT_DIR
        agg.OUT_DIR = Path(td)
        try:
            out = agg.write_replication(rows, "testver")
        finally:
            agg.OUT_DIR = orig

    branches = {r["branch_key"] for r in out}
    assert branches == {"branch_u"}, f"non-vol_surface branch leaked into replication table: {branches}"
    assert len(out) == 4, f"expected exactly 4 vol_surface cells, got {len(out)}"


def test_multiseed_vol_surface_cells_get_real_mean_and_std():
    rows = _synthetic_rows()
    with tempfile.TemporaryDirectory() as td:
        orig = agg.OUT_DIR
        agg.OUT_DIR = Path(td)
        try:
            out = agg.write_replication(rows, "testver")
        finally:
            agg.OUT_DIR = orig

    call_a = next(r for r in out if r["option_type"] == "call" and r["arch"] == "A")
    vals = [0.990, 0.991, 0.992]
    assert call_a["n_seeds"] == 3
    assert call_a["seeds"] == "42,43,44"
    assert abs(call_a["r2_price_mean"] - statistics.mean(vals)) < 1e-9
    assert abs(call_a["r2_price_std"] - statistics.stdev(vals)) < 1e-9
    assert call_a["r2_price_std"] > 0


def test_single_seed_vol_surface_cell_gets_na_not_zero():
    """Defense in depth: even a vol_surface cell with <2 seeds must not show 0.0."""
    rows = [_row("call", "branch_u", "A", 42, 0.99)]  # only one seed -- edge case
    with tempfile.TemporaryDirectory() as td:
        orig = agg.OUT_DIR
        agg.OUT_DIR = Path(td)
        try:
            out = agg.write_replication(rows, "testver")
            md_text = (Path(td) / "replication_seeds_testver.md").read_text()
            csv_rows = list(csv.DictReader(open(Path(td) / "replication_seeds_testver.csv")))
        finally:
            agg.OUT_DIR = orig

    assert out[0]["r2_price_std"] is None, "single-seed cell must not get a numeric std"
    assert "N/A" in md_text, "markdown must show N/A, not a fake 0.000000, for unmeasured dispersion"
    assert "0.000000" not in md_text
    # CSV: DictWriter renders None as an empty string, never the literal "0.0"
    assert csv_rows[0]["r2_price_std"] == ""


if __name__ == "__main__":
    ok = True
    for fn in (test_single_seed_branches_excluded_entirely,
               test_multiseed_vol_surface_cells_get_real_mean_and_std,
               test_single_seed_vol_surface_cell_gets_na_not_zero):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - direct-run reporting
            print(f"FAIL {fn.__name__}: {exc}")
            ok = False
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
