"""
Phase 2: parquet parts or deeponet_training_data.parquet -> deeponet_tensors_{call,put}.h5

Run from this directory:
    conda activate dl_new
    python pre_process_2_hdf5.py --type call
    python pre_process_2_hdf5.py --type put
"""

from __future__ import annotations

import argparse
import builtins
import functools
from pathlib import Path

import h5py
import numpy as np
import polars as pl

builtins.print = functools.partial(builtins.print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PARQUET = "deeponet_training_data.parquet"
INPUT_PARTS_DIR = "deeponet_training_data_parts"
VOL_SURFACE_DIM = 187  # 11 tenors x 17 deltas per cp_flag
BATCH_ROWS = 100_000

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

CP_FLAG_MAP = {"call": "C", "put": "P"}
OUTPUT_MAP = {"call": "deeponet_tensors_call.h5", "put": "deeponet_tensors_put.h5"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-type HDF5 tensors")
    parser.add_argument(
        "--type",
        choices=["call", "put"],
        required=True,
        help="Option type to export",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DIR / INPUT_PARQUET),
        help="Path to parquet file or parquet-parts directory from phase 1",
    )
    return parser.parse_args()


def _resolve_input_paths(input_path: str) -> list[Path]:
    requested = Path(input_path)
    parts_dir = SCRIPT_DIR / INPUT_PARTS_DIR
    default_input = (SCRIPT_DIR / INPUT_PARQUET).resolve()

    if requested.resolve() == default_input and parts_dir.exists():
        paths = sorted(parts_dir.glob("*.parquet"))
    elif requested.is_dir():
        paths = sorted(requested.glob("*.parquet"))
    elif requested.exists():
        paths = [requested]
    else:
        paths = sorted(parts_dir.glob("*.parquet")) if parts_dir.exists() else []

    if not paths:
        raise FileNotFoundError(
            f"No parquet input found at {requested} or {SCRIPT_DIR / INPUT_PARTS_DIR}"
        )
    return paths


def _flatten_nested(series: pl.Series) -> np.ndarray:
    rows = series.to_list()
    return np.vstack([np.asarray(r, dtype=np.float32).reshape(-1) for r in rows])


def _split_lookup(unique_dates: np.ndarray) -> tuple[dict, int, int]:
    n_dates = len(unique_dates)
    train_end = int(TRAIN_FRAC * n_dates)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n_dates)

    date_to_split: dict = {}
    for i, d in enumerate(unique_dates):
        if i < train_end:
            date_to_split[d] = 0
        elif i < val_end:
            date_to_split[d] = 1
        else:
            date_to_split[d] = 2

    return date_to_split, train_end, val_end


def _collect_metadata(
    input_paths: list[Path],
    cp_flag: str,
    required: list[str],
) -> tuple[np.ndarray, dict[Path, int], int, int]:
    all_dates: list[np.ndarray] = []
    row_counts: dict[Path, int] = {}

    print("Scanning parquet metadata...")
    for path in input_paths:
        df_dates = (
            pl.scan_parquet(str(path))
            .filter(pl.col("cp_flag") == cp_flag)
            .drop_nulls(subset=required)
            .filter(pl.col("vol_surface_vector").list.len() == VOL_SURFACE_DIM)
            .select("date")
            .collect()
        )
        row_counts[path] = df_dates.height
        if df_dates.height:
            all_dates.append(df_dates["date"].to_numpy())
        print(f"  {path.name}: {df_dates.height} rows")

    if not all_dates:
        return np.array([]), row_counts, 0, 0

    unique_dates = np.sort(np.unique(np.concatenate(all_dates)))
    _, train_end, val_end = _split_lookup(unique_dates)
    return unique_dates, row_counts, train_end, val_end


def _create_datasets(f: h5py.File, total_rows: int) -> None:
    chunk_rows = min(8192, max(1, total_rows))
    f.create_dataset(
        "branch_u",
        shape=(total_rows, VOL_SURFACE_DIM),
        dtype=np.float32,
        chunks=(chunk_rows, VOL_SURFACE_DIM),
    )
    f.create_dataset(
        "spot_history",
        shape=(total_rows, 105),
        dtype=np.float32,
        chunks=(chunk_rows, 105),
    )
    f.create_dataset(
        "vix_history",
        shape=(total_rows, 84),
        dtype=np.float32,
        chunks=(chunk_rows, 84),
    )
    f.create_dataset(
        "trunk_y",
        shape=(total_rows, 4),
        dtype=np.float32,
        chunks=(chunk_rows, 4),
    )
    f.create_dataset(
        "target_v_log",
        shape=(total_rows, 1),
        dtype=np.float32,
        chunks=(chunk_rows, 1),
    )
    f.create_dataset(
        "moneyness",
        shape=(total_rows, 1),
        dtype=np.float32,
        chunks=(chunk_rows, 1),
    )
    f.create_dataset(
        "normalized_price",
        shape=(total_rows, 1),
        dtype=np.float32,
        chunks=(chunk_rows, 1),
    )
    f.create_dataset("date", shape=(total_rows,), dtype="S10", chunks=(chunk_rows,))
    f.create_dataset("split_id", shape=(total_rows,), dtype=np.uint8, chunks=(chunk_rows,))


def _prepare_chunk(df: pl.DataFrame) -> tuple[np.ndarray, ...]:
    branch_u = np.vstack(df["vol_surface_vector"].to_list()).astype(np.float32)
    assert branch_u.shape[1] == VOL_SURFACE_DIM

    spot_history = _flatten_nested(df["spot_history_tensor"])
    vix_history = _flatten_nested(df["vix_history_tensor"])
    trunk_y = df.select(["log_moneyness", "T_years", "r", "q"]).to_numpy().astype(
        np.float32
    )
    target_v_log = df.select(["log_normalized_price"]).to_numpy().astype(np.float32)
    moneyness = df.select(["moneyness"]).to_numpy().astype(np.float32)
    normalized_price = df.select(["normalized_price"]).to_numpy().astype(np.float32)
    dates_np = df["date"].to_numpy()

    return (
        branch_u,
        spot_history,
        vix_history,
        trunk_y,
        target_v_log,
        moneyness,
        normalized_price,
        dates_np,
    )


def _iter_batches(df: pl.DataFrame, batch_rows: int = BATCH_ROWS):
    for start in range(0, df.height, batch_rows):
        yield df.slice(start, batch_rows)


def build_hdf5(option_type: str, input_path: str) -> None:
    cp_flag = CP_FLAG_MAP[option_type]
    output_path = SCRIPT_DIR / OUTPUT_MAP[option_type]
    input_paths = _resolve_input_paths(input_path)

    print(f"Loading {len(input_paths)} parquet input file(s) (cp_flag={cp_flag})...")
    required = [
        "log_moneyness",
        "T_years",
        "r",
        "q",
        "log_normalized_price",
        "vol_surface_vector",
        "spot_history_tensor",
        "vix_history_tensor",
        "moneyness",
        "normalized_price",
        "date",
    ]

    unique_dates, row_counts, train_end, val_end = _collect_metadata(
        input_paths, cp_flag, required
    )
    total_expected = sum(row_counts.values())
    if total_expected == 0:
        raise RuntimeError(f"No rows for type={option_type}")

    date_to_split, _, _ = _split_lookup(unique_dates)

    if output_path.exists():
        output_path.unlink()

    print(f"Writing {output_path} ({total_expected} rows expected)...")
    total_rows = 0
    split_counts = {0: 0, 1: 0, 2: 0}

    with h5py.File(output_path, "w") as f:
        f.attrs["cp_flag"] = cp_flag
        f.attrs["vol_surface_dim"] = VOL_SURFACE_DIM
        f.attrs["spot_history_flat_dim"] = 105
        f.attrs["vix_history_flat_dim"] = 84
        _create_datasets(f, total_expected)

        for path in input_paths:
            if row_counts.get(path, 0) == 0:
                continue
            print(f"  Processing {path.name} ...")
            df_part = pl.read_parquet(path).filter(pl.col("cp_flag") == cp_flag)
            df_part = df_part.drop_nulls(subset=required)
            df_part = df_part.filter(
                pl.col("vol_surface_vector").list.len() == VOL_SURFACE_DIM
            )
            if df_part.height == 0:
                continue

            part_rows = 0
            for df in _iter_batches(df_part):
                (
                    branch_u,
                    spot_history,
                    vix_history,
                    trunk_y,
                    target_v_log,
                    moneyness,
                    normalized_price,
                    dates_np,
                ) = _prepare_chunk(df)

                split_id = np.array([date_to_split[d] for d in dates_np], dtype=np.uint8)
                date_str = np.array([np.datetime_as_string(d, unit="D") for d in dates_np])

                start = total_rows
                end = start + df.height
                f["branch_u"][start:end] = branch_u
                f["spot_history"][start:end] = spot_history
                f["vix_history"][start:end] = vix_history
                f["trunk_y"][start:end] = trunk_y
                f["target_v_log"][start:end] = target_v_log
                f["moneyness"][start:end] = moneyness
                f["normalized_price"][start:end] = normalized_price
                f["date"][start:end] = date_str.astype("S10")
                f["split_id"][start:end] = split_id

                total_rows += df.height
                part_rows += df.height
                for sid in split_counts:
                    split_counts[sid] += int((split_id == sid).sum())
                print(f"      batch appended; part rows so far: {part_rows}")

            print(f"    -> appended {part_rows} rows")

            del df_part

    if total_rows != total_expected:
        raise RuntimeError(
            f"Wrote {total_rows} rows, expected {total_expected}; output may be incomplete."
        )

    for sid, label in [(0, "train"), (1, "val"), (2, "test")]:
        print(f"  {label}: {split_counts[sid]} rows")

    if len(unique_dates):
        print(f"  {len(unique_dates)} unique dates: {unique_dates[0]} .. {unique_dates[-1]}")
        if train_end > 0:
            print(f"  Last train date: {unique_dates[train_end - 1]}")
        if val_end > train_end:
            print(f"  Last val date: {unique_dates[val_end - 1]}")

    print(f"Valid samples: {total_rows}")
    print("HDF5 export complete.")


if __name__ == "__main__":
    args = _parse_args()
    build_hdf5(args.type, args.input)
