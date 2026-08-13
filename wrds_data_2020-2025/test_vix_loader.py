"""
assert-based end-to-end VIX validation (no framework, no data written).

Run after a smoke build (phase 1 + phase 2, see SMOKE_REPORT_v4.md):
    conda run -n dl_new python test_vix_loader.py

What it checks, on top of pre_process_1_data_v4.py --self-test:
* HDF5 attrs identify the source as CBOE VIX (series, SHA-256) -- not an
  ambiguous "volatility index" label.
* Every vix_history row for the first few dates matches a directly-calculated
  raw 21-day normalized window from the CSV, within float32 tolerance.
* Every vix_level matches the raw VIX close for that date, within float32
  tolerance, and the level range is plausible for SPX VIX (a VVIX-tainted
  build would sit far above 82.69, the 2015-2025 SPX VIX max).
"""

import hashlib
import sys
from datetime import date
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import h5py  # noqa: E402

import pre_process_1_data_v4 as p1  # noqa: E402

H5_CALL = SCRIPT_DIR / "smoke_v4" / "deeponet_tensors_call_v4_smoke.h5"
H5_PUT = SCRIPT_DIR / "smoke_v4" / "deeponet_tensors_put_v4_smoke.h5"
VIX_CSV = SCRIPT_DIR / p1.VIX_CSV


def _check_h5(h5_path: Path, vix_map: dict, closes: dict) -> None:
    assert h5_path.exists(), f"missing smoke HDF5 (run phase 1+2 smoke build first): {h5_path}"
    with h5py.File(h5_path, "r") as f:
        dates = [d.decode() for d in f["date"][:]]
        attrs = dict(f.attrs)
        history = f["vix_history"][:]
        level = f["vix_level"][:]

    # HDF5 attrs must say CBOE VIX, with the exact file hash
    src = attrs.get("phase1_vix_source_series", "")
    assert "CBOE VIX" in src, f"attrs must identify CBOE VIX, got {src!r}"
    expected_sha = hashlib.sha256(VIX_CSV.read_bytes()).hexdigest()
    assert attrs.get("phase1_vix_sha256", "") == expected_sha, "attr SHA-256 mismatch"
    assert attrs.get("phase1_vix_source_columns", "") == "date,vixo,vixh,vixl,vix"
    assert attrs.get("phase1_vix_column_mapping", "") == "vixo->open, vixh->high, vixl->low, vix->close"

    # vix_level must be genuine SPX VIX (max ever in 2015-2025: 82.69) -- a
    # VVIX-tainted build would land near 125.7
    assert float(level.min()) > 0.0
    assert float(level.max()) < 90.0, f"vix_level max {level.max():.2f} is not SPX VIX"

    # cross-check against direct recomputation for the first 3 distinct dates
    checked = 0
    seen = []
    for d in dates:
        if d not in seen:
            seen.append(d)
        if len(seen) == 3:
            break
    for d in seen:
        day = date.fromisoformat(d)
        assert day in vix_map, f"date {d} missing from VIX lookback map"
        direct = np.asarray(vix_map[day], dtype=np.float32)
        assert direct.shape == (84,), direct.shape
        for i, d0 in enumerate(dates):
            if d0 != d:
                continue
            assert np.allclose(history[i], direct, rtol=1e-5, atol=1e-6), (h5_path, d, i)
            assert np.allclose(level[i], closes[day], rtol=1e-5, atol=1e-5), (h5_path, d, i)
            checked += 1
    assert checked > 0, "no rows cross-checked"
    print(f"  {h5_path.name}: {checked} rows cross-checked against direct recompute")


def main() -> None:
    p1._vix_self_test()  # CSV-level checks (schema, OHLC, reference, window builder)

    daily = p1._load_vix_daily(None, None)
    vix_map, _ = p1._build_ohlc_lookback_map(daily, p1.SPOT_LOOKBACK_DAYS)
    closes = dict(zip(daily["date"].to_list(), daily["close"].to_list()))
    for h5 in (H5_CALL, H5_PUT):
        _check_h5(h5, vix_map, closes)
    print("test_vix_loader OK: attrs + vix_history/vix_level match direct recompute")


if __name__ == "__main__":
    main()
