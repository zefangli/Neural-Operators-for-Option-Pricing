"""
Phase 3: inspect parquet and HDF5 outputs from preprocessing.

Run from this directory:
    conda activate dl_new
    python pre_process_3_print_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_PATH = SCRIPT_DIR / "deeponet_training_data.parquet"
PARQUET_PARTS_DIR = SCRIPT_DIR / "deeponet_training_data_parts"
H5_CALL = SCRIPT_DIR / "deeponet_tensors_call.h5"
H5_PUT = SCRIPT_DIR / "deeponet_tensors_put.h5"
VOL_SURFACE_DIM = 187  # 11 tenors x 17 deltas per cp_flag


def _parquet_inputs() -> list[Path]:
    if PARQUET_PARTS_DIR.exists():
        paths = sorted(PARQUET_PARTS_DIR.glob("*.parquet"))
        if paths:
            return paths
    return [PARQUET_PATH] if PARQUET_PATH.exists() else []


def inspect_parquet() -> pl.DataFrame | None:
    paths = _parquet_inputs()
    label = PARQUET_PARTS_DIR.name if len(paths) > 1 else PARQUET_PATH.name

    print("=" * 80)
    print(f"PARQUET: {label}")
    print("=" * 80)

    if not paths:
        print("File not found. Run pre_process_1_data.py first.")
        return None

    lf = pl.scan_parquet([str(p) for p in paths])
    summary = lf.select(
        pl.len().alias("rows"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ).collect()
    sample = lf.head(1).collect()

    print(f"Rows: {summary['rows'][0]}, columns: {len(sample.columns)}")
    print(f"Date range: {summary['min_date'][0]} .. {summary['max_date'][0]}")

    print("\nRows by cp_flag:")
    print(lf.group_by("cp_flag").len().sort("cp_flag").collect())

    print("\nColumn dtypes:")
    for name, dtype in zip(sample.columns, sample.dtypes):
        print(f"  {name:28} {dtype}")

    if "log_normalized_price" in sample.columns:
        print("\nlog_normalized_price quantiles:")
        print(
            lf.select(
                pl.col("log_normalized_price").quantile(0.01).alias("p01"),
                pl.col("log_normalized_price").quantile(0.50).alias("p50"),
                pl.col("log_normalized_price").quantile(0.99).alias("p99"),
            ).collect()
        )

    if "vol_surface_vector" in sample.columns:
        lens_summary = lf.select(
            pl.col("vol_surface_vector").list.len().min().alias("min_len"),
            pl.col("vol_surface_vector").list.len().max().alias("max_len"),
            (pl.col("vol_surface_vector").list.len() != VOL_SURFACE_DIM)
            .sum()
            .alias("bad"),
        ).collect()
        print(
            "\nvol_surface_vector lengths: "
            f"min={lens_summary['min_len'][0]}, max={lens_summary['max_len'][0]}"
        )
        bad = lens_summary["bad"][0]
        if bad:
            print(f"  WARNING: {bad} rows != {VOL_SURFACE_DIM}")

    for col, expected in [
        ("spot_history_tensor", (21, 5)),
        ("vix_history_tensor", (21, 4)),
    ]:
        if col not in sample.columns:
            continue
        values = sample[col].head(1).to_list()
        if values and values[0] is not None:
            print(f"\n{col}: nested shape example {len(values[0])} x {len(values[0][0])}")

    print("\nNull counts (key columns):")
    key_cols = [
        "r",
        "q",
        "log_moneyness",
        "log_normalized_price",
        "vol_surface_vector",
        "spot_history_tensor",
        "vix_history_tensor",
    ]
    present = [c for c in key_cols if c in sample.columns]
    print(lf.select(present).null_count().collect())

    return sample


def inspect_hdf5(path: Path, label: str) -> None:
    print("\n" + "=" * 80)
    print(f"HDF5 ({label}): {path.name}")
    print("=" * 80)

    if not path.exists():
        print("File not found.")
        return

    with h5py.File(path, "r") as f:
        print("Attributes:", dict(f.attrs))
        print("\nDatasets:")
        for key in f.keys():
            ds = f[key]
            print(f"  {key:22} shape={ds.shape} dtype={ds.dtype}")

        split_id = f["split_id"][:]
        for sid, name in [(0, "train"), (1, "val"), (2, "test")]:
            print(f"  split {name}: {(split_id == sid).sum()} rows")

        dates = f["date"][:]
        unique_dates = np.unique(dates)
        if len(unique_dates) >= 2:
            print(f"  dates: {unique_dates[0].decode()} .. {unique_dates[-1].decode()}")

        u = f["branch_u"]
        print(f"\nbranch_u: shape={u.shape}, sample row mean={u[0].mean():.4f}")

        if "spot_history" in f:
            sh = f["spot_history"]
            print(f"spot_history: shape={sh.shape}, last-day close (idx -5) sample={sh[0, -5]:.4f}")

        if "vix_history" in f:
            vh = f["vix_history"]
            print(f"vix_history: shape={vh.shape}, last-day close (idx -1) sample={vh[0, -1]:.4f}")

        ty = f["trunk_y"][:5]
        print("\ntrunk_y [log_moneyness, T_years, r, q] (first 5):")
        for i, row in enumerate(ty):
            print(f"  {i}: {row}")

        tv = f["target_v_log"][:5]
        print("\ntarget_v_log (first 5):", tv.ravel())


def main() -> None:
    inspect_parquet()
    inspect_hdf5(H5_CALL, "call")
    inspect_hdf5(H5_PUT, "put")
    print("\nInspection complete.")


if __name__ == "__main__":
    main()
