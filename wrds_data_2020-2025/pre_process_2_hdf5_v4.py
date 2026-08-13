"""
Phase 2 (v4/v5): v4 parquet parts -> deeponet_tensors_{call,put}_{v4,v5}.h5

  --dataset-version v4  (default) unfiltered robustness sample
  --dataset-version v5  canonical: adds the zero-tolerance static no-arbitrage
                        midpoint filter (see _bounds_keep_expr). schema_version
                        stays "v4" -- the tensor schema is identical; only the
                        row SAMPLE differs, which is what dataset_version tracks.

Evolved copy of pre_process_2_hdf5.py.

BACKWARD COMPATIBLE with the v3 layout: branch_u (n,187), spot_history (n,105),
vix_history (n,84), trunk_y (n,4) = [log_moneyness, T_years, r, q], target_v_log
(n,1), moneyness (n,1), normalized_price (n,1), date (n,)S10, split_id (n,)uint8
all keep their name, shape, dtype and row alignment, so every existing
train_model_v3 script reads a v4 file unchanged.

v4 adds row-aligned datasets ALONGSIDE those (trunk_y is deliberately NOT widened):
  identity : symbol (S24), root (S8), optionid (i8), am_settlement (u8),
             exercise_style (S2)
  maturity : T_calendar (n,1), T_settlement (n,1)   -- trunk_y[:,1] carries the
             active basis; both are always written so results stay traceable
  observed : impl_volatility (n,1), market_delta (n,1)
  quotes   : best_bid, best_offer, spread_norm, half_spread_norm, strike (n,1)
  market   : vix_level (n,1)  -- absolute VIX close (vix_history is normalized)

Nulls in the optional columns become NaN (float), -1 (optionid), 255
(am_settlement), b"" (strings). They are NOT in `required`, so adding them can
never drop a row relative to v3.

The maturity cut can be re-made here without re-scanning the 8.6 GB CSV, as long
as phase 1 was run with --min-maturity-days -1.

Run from this directory:
    conda activate dl_new
    python pre_process_2_hdf5_v4.py --self-test
    python pre_process_2_hdf5_v4.py --type call --input smoke_v4/parts_diag \
        --output smoke_v4/deeponet_tensors_call_v4_smoke.h5
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import functools
import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl

builtins.print = functools.partial(builtins.print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PARTS_DIR = "deeponet_training_data_parts_v4"
CONFIG_JSON = "pipeline_config_v4.json"
VOL_SURFACE_DIM = 187
BATCH_ROWS = 100_000
DAY_COUNT = 365.0

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

CP_FLAG_MAP = {"call": "C", "put": "P"}
DATASET_VERSIONS = ("v4", "v5")
# v5 = v4 rows minus the static no-arbitrage midpoint violations.
QUOTE_FILTER = {"v4": None, "v5": "static_bounds_midpoint"}
BOUNDS_DOC = (
    "normalized V/K units; fwd = moneyness*exp(-q*T), disc = exp(-r*T); "
    "call midpoint must lie in [max(fwd - disc, 0), fwd], "
    "put midpoint in [max(disc - fwd, 0), disc]; inclusive, zero tolerance. "
    "Evaluated in float64 on the parquet AND on the float32-rounded values, so "
    "no row can flip sides in the export cast."
)


def _output_name(option_type: str, dataset_version: str) -> str:
    return f"deeponet_tensors_{option_type}_{dataset_version}.h5"

# v3 datasets required to be non-null; keeping this list identical to v3 is what
# guarantees the v4 row set is the v3 row set minus the maturity cut.
REQUIRED = [
    "log_moneyness", "T_years", "r", "q", "log_normalized_price",
    "vol_surface_vector", "spot_history_tensor", "vix_history_tensor",
    "moneyness", "normalized_price", "date",
]

# v4 float extras: (hdf5 name, parquet column). Nulls -> NaN.
EXTRA_F32 = [
    ("T_calendar", "T_calendar"),
    ("T_settlement", "T_settlement"),
    ("impl_volatility", "obs_impl_volatility"),
    ("market_delta", "obs_delta"),
    ("best_bid", "best_bid"),
    ("best_offer", "best_offer"),
    ("spread_norm", "spread_norm"),
    ("half_spread_norm", "half_spread_norm"),
    ("strike", "strike"),
    ("vix_level", "vix_level"),
]
STR_COLS = [("symbol", "symbol", "S24"), ("root", "root", "S8"),
            ("exercise_style", "exercise_style", "S2")]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export per-type v4 HDF5 tensors")
    p.add_argument("--type", choices=["call", "put"], help="Option type to export")
    p.add_argument("--input", type=str, default=str(SCRIPT_DIR / INPUT_PARTS_DIR),
                   help="Parquet file or parquet-parts directory from phase 1 v4")
    p.add_argument("--output", type=str, default=None,
                   help="Output .h5 path (default deeponet_tensors_{type}_{dataset_version}.h5). "
                        "Must contain the dataset version marker -- v3/v4 files are irreplaceable.")
    p.add_argument("--dataset-version", choices=DATASET_VERSIONS, default="v4",
                   help="v4 = unfiltered; v5 = canonical, static no-arbitrage midpoint filter.")
    p.add_argument("--lineage-h5", type=str, default=None,
                   help="v4 HDF5 this export is the filtered counterpart of; its sha256 is "
                        "recorded as lineage (default: the same-type v4 file, if present).")
    p.add_argument("--min-maturity-days", type=float, default=1.0,
                   help="Keep rows with T_years*365 > this (default 1.0 = paper rule). "
                        "-1 disables (diagnostic only; T<=0 rows are not priceable).")
    p.add_argument("--t-basis", choices=["calendar", "settlement"], default=None,
                   help="Override which maturity lands in trunk_y[:,1]. "
                        "Default: whatever phase 1 wrote into T_years.")
    p.add_argument("--self-test", action="store_true", help="Run the assert-based self-check and exit.")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_input_paths(input_path: str) -> list[Path]:
    requested = Path(input_path)
    if not requested.is_absolute():
        requested = SCRIPT_DIR / requested
    if requested.is_dir():
        paths = sorted(requested.glob("*.parquet"))
    elif requested.exists():
        paths = [requested]
    else:
        paths = []
    if not paths:
        raise FileNotFoundError(f"No parquet input found at {requested}")
    return paths


def _load_phase1_config(input_paths: list[Path]) -> dict:
    for parent in {p.parent for p in input_paths}:
        cfg_path = parent / CONFIG_JSON
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg.pop("attrition", None)  # per-chunk detail, not useful as an h5 attr
            return cfg
    return {}


def _bounds_keep_expr(cp_flag: str) -> pl.Expr:
    """Static no-arbitrage bounds on the quoted midpoint, in V/K units.

    Evaluated twice: once on the float64 parquet values (the honest arithmetic)
    and once on the float32-rounded values (what actually lands in the HDF5).
    A row must satisfy both, so the export cast cannot resurrect a violation.
    """
    def keep(round32: bool) -> pl.Expr:
        def c(name: str) -> pl.Expr:
            e = pl.col(name).cast(pl.Float32) if round32 else pl.col(name)
            return e.cast(pl.Float64)
        M, T, r, q, v = (c(x) for x in ("moneyness", "T_years", "r", "q", "normalized_price"))
        fwd, disc = M * (-q * T).exp(), (-r * T).exp()
        lo, hi = ((fwd - disc, fwd) if cp_flag == "C" else (disc - fwd, disc))
        return (v >= pl.max_horizontal(lo, pl.lit(0.0))) & (v <= hi)
    return keep(False) & keep(True)


def _select(lf: pl.LazyFrame, cp_flag: str, min_days: float, t_basis: str | None,
            quote_filter: bool = False):
    lf = (
        lf.filter(pl.col("cp_flag") == cp_flag)
        .drop_nulls(subset=REQUIRED)
        .filter(pl.col("vol_surface_vector").list.len() == VOL_SURFACE_DIM)
    )
    if t_basis is not None:
        lf = lf.with_columns(
            pl.col("T_settlement" if t_basis == "settlement" else "T_calendar").alias("T_years")
        )
    if min_days >= 0.0:
        lf = lf.filter(pl.col("T_years") * DAY_COUNT > min_days)
    if quote_filter:
        lf = lf.filter(_bounds_keep_expr(cp_flag))
    return lf


def _flatten_nested(series: pl.Series) -> np.ndarray:
    return np.vstack([np.asarray(r, dtype=np.float32).reshape(-1) for r in series.to_list()])


def _split_lookup(unique_dates: np.ndarray) -> tuple[dict, int, int]:
    n_dates = len(unique_dates)
    train_end = int(TRAIN_FRAC * n_dates)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n_dates)
    date_to_split = {
        d: (0 if i < train_end else 1 if i < val_end else 2) for i, d in enumerate(unique_dates)
    }
    return date_to_split, train_end, val_end


def _collect_metadata(input_paths, cp_flag, min_days, t_basis, quote_filter: bool):
    """One scan: post-filter row counts + unique dates, and (when filtering) the
    pre-filter row counts per split, scored against the same date->split map."""
    kept_dates, pre_dates, row_counts = [], [], {}
    print("Scanning parquet metadata...")
    for path in input_paths:
        cols = ["date"]
        lf = _select(pl.scan_parquet(str(path)), cp_flag, min_days, t_basis)
        if quote_filter:
            lf = lf.with_columns(_bounds_keep_expr(cp_flag).alias("_keep"))
            cols.append("_keep")
        df = lf.select(cols).collect()
        d = df["date"].to_numpy()
        keep = df["_keep"].to_numpy() if quote_filter else np.ones(df.height, bool)
        row_counts[path] = int(keep.sum())
        if df.height:
            pre_dates.append(d)
            kept_dates.append(d[keep])
        print(f"  {path.name}: {row_counts[path]} rows"
              + (f" (of {df.height}, dropped {df.height - row_counts[path]})" if quote_filter else ""))
    if not kept_dates:
        return np.array([]), row_counts, 0, 0, {}
    unique_dates = np.sort(np.unique(np.concatenate(kept_dates)))
    date_to_split, train_end, val_end = _split_lookup(unique_dates)
    pre_counts = {0: 0, 1: 0, 2: 0}
    for d in pre_dates:
        for sid in np.fromiter((date_to_split.get(x, 255) for x in d), dtype=np.uint8, count=len(d)):
            if sid in pre_counts:
                pre_counts[sid] += 1
    return unique_dates, row_counts, train_end, val_end, pre_counts


def _create_datasets(f: h5py.File, n: int) -> None:
    c = min(8192, max(1, n))
    # --- v3 layout, byte-for-byte compatible ---
    f.create_dataset("branch_u", (n, VOL_SURFACE_DIM), np.float32, chunks=(c, VOL_SURFACE_DIM))
    f.create_dataset("spot_history", (n, 105), np.float32, chunks=(c, 105))
    f.create_dataset("vix_history", (n, 84), np.float32, chunks=(c, 84))
    f.create_dataset("trunk_y", (n, 4), np.float32, chunks=(c, 4))
    for name in ("target_v_log", "moneyness", "normalized_price"):
        f.create_dataset(name, (n, 1), np.float32, chunks=(c, 1))
    f.create_dataset("date", (n,), "S10", chunks=(c,))
    f.create_dataset("split_id", (n,), np.uint8, chunks=(c,))
    # --- v4 additions ---
    for name, _ in EXTRA_F32:
        f.create_dataset(name, (n, 1), np.float32, chunks=(c, 1))
    for name, _, dt in STR_COLS:
        f.create_dataset(name, (n,), dt, chunks=(c,))
    f.create_dataset("optionid", (n,), np.int64, chunks=(c,))
    f.create_dataset("am_settlement", (n,), np.uint8, chunks=(c,))  # 1=AM, 0=PM, 255=unknown


def _prepare_chunk(df: pl.DataFrame) -> dict:
    branch_u = np.vstack(df["vol_surface_vector"].to_list()).astype(np.float32)
    assert branch_u.shape[1] == VOL_SURFACE_DIM
    out = {
        "branch_u": branch_u,
        "spot_history": _flatten_nested(df["spot_history_tensor"]),
        "vix_history": _flatten_nested(df["vix_history_tensor"]),
        "trunk_y": df.select(["log_moneyness", "T_years", "r", "q"]).to_numpy().astype(np.float32),
        "target_v_log": df.select(["log_normalized_price"]).to_numpy().astype(np.float32),
        "moneyness": df.select(["moneyness"]).to_numpy().astype(np.float32),
        "normalized_price": df.select(["normalized_price"]).to_numpy().astype(np.float32),
    }
    for name, col in EXTRA_F32:
        out[name] = df.select(pl.col(col).cast(pl.Float32)).to_numpy().astype(np.float32)
    for name, col, dt in STR_COLS:
        out[name] = df[col].fill_null("").to_numpy().astype(dt)
    out["optionid"] = df["optionid"].fill_null(-1).to_numpy().astype(np.int64)
    out["am_settlement"] = df["am_settlement"].fill_null(255).to_numpy().astype(np.uint8)
    return out


def build_hdf5(option_type: str, input_path: str, output: str | None,
               min_maturity_days: float, t_basis: str | None,
               dataset_version: str = "v4", lineage_h5: str | None = None) -> None:
    cp_flag = CP_FLAG_MAP[option_type]
    quote_filter = QUOTE_FILTER[dataset_version]
    out_path = Path(output) if output else SCRIPT_DIR / _output_name(option_type, dataset_version)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path
    # Trust boundary: never let this script land on a v3 artifact, and never let a
    # v5 (filtered) export land on a v4 (unfiltered) file -- different samples.
    assert dataset_version in out_path.name, \
        f"output must carry the '{dataset_version}' marker, got {out_path.name}"
    if dataset_version == "v5":
        assert "v4" not in out_path.name, f"v5 export must not target a v4 name: {out_path.name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lineage = {}
    if dataset_version == "v5":
        src = Path(lineage_h5) if lineage_h5 else SCRIPT_DIR / _output_name(option_type, "v4")
        if not src.is_absolute():
            src = SCRIPT_DIR / src
        if src.exists():
            print(f"Hashing lineage source {src.name} ...")
            lineage = {"lineage_v4_h5": src.name, "lineage_v4_h5_sha256": _sha256(src)}
            print(f"  {lineage['lineage_v4_h5_sha256']}")

    input_paths = _resolve_input_paths(input_path)
    cfg = _load_phase1_config(input_paths)
    print(f"Loading {len(input_paths)} parquet input file(s) (cp_flag={cp_flag})...")
    print(f"Maturity cut at export: min_maturity_days={min_maturity_days}, "
          f"t_basis={t_basis or cfg.get('t_basis', 'inherit')}")

    print(f"dataset_version={dataset_version}, quote_filter={quote_filter or 'none'}")

    unique_dates, row_counts, train_end, val_end, pre_counts = _collect_metadata(
        input_paths, cp_flag, min_maturity_days, t_basis, bool(quote_filter)
    )
    total_expected = sum(row_counts.values())
    if total_expected == 0:
        raise RuntimeError(f"No rows for type={option_type}")
    date_to_split, _, _ = _split_lookup(unique_dates)

    if out_path.exists():
        out_path.unlink()

    print(f"Writing {out_path} ({total_expected} rows expected)...")
    total_rows = 0
    split_counts = {0: 0, 1: 0, 2: 0}

    with h5py.File(out_path, "w") as f:
        f.attrs["cp_flag"] = cp_flag
        f.attrs["vol_surface_dim"] = VOL_SURFACE_DIM
        f.attrs["spot_history_flat_dim"] = 105
        f.attrs["vix_history_flat_dim"] = 84
        f.attrs["schema_version"] = "v4"  # tensor schema; NOT the sample identity
        f.attrs["dataset_version"] = dataset_version
        f.attrs["quote_filter"] = quote_filter or "none"
        if quote_filter:
            f.attrs["quote_filter_tolerance"] = 0.0
            f.attrs["quote_filter_bounds"] = BOUNDS_DOC
            f.attrs["rows_prefilter_total"] = int(sum(pre_counts.values()))
            f.attrs["rows_prefilter_by_split"] = json.dumps(
                {n: int(pre_counts[s]) for s, n in ((0, "train"), (1, "val"), (2, "test"))})
            for k, v in lineage.items():
                f.attrs[k] = v
            f.attrs["lineage_parquet_parts"] = f"{INPUT_PARTS_DIR} ({len(input_paths)} parts)"
        f.attrs["export_min_maturity_days"] = min_maturity_days
        f.attrs["export_t_basis"] = t_basis or cfg.get("t_basis", "unknown")
        f.attrs["trunk_y_columns"] = "log_moneyness,T_years,r,q"
        f.attrs["am_settlement_coding"] = "1=AM (SOQ open) settled, 0=PM (close) settled, 255=unknown"
        for k, v in cfg.items():
            f.attrs[f"phase1_{k}"] = "null" if v is None else v
        _create_datasets(f, total_expected)

        for path in input_paths:
            if row_counts.get(path, 0) == 0:
                continue
            print(f"  Processing {path.name} ...")
            df_part = _select(pl.scan_parquet(str(path)), cp_flag, min_maturity_days,
                              t_basis, bool(quote_filter)).collect()
            if df_part.height == 0:
                continue
            part_rows = 0
            for start_i in range(0, df_part.height, BATCH_ROWS):
                df = df_part.slice(start_i, BATCH_ROWS)
                arrays = _prepare_chunk(df)
                dates_np = df["date"].to_numpy()
                split_id = np.array([date_to_split[d] for d in dates_np], dtype=np.uint8)
                date_str = np.array([np.datetime_as_string(d, unit="D") for d in dates_np])

                s, e = total_rows, total_rows + df.height
                for name, arr in arrays.items():
                    f[name][s:e] = arr
                f["date"][s:e] = date_str.astype("S10")
                f["split_id"][s:e] = split_id

                total_rows += df.height
                part_rows += df.height
                for sid in split_counts:
                    split_counts[sid] += int((split_id == sid).sum())
            print(f"    -> appended {part_rows} rows")
            del df_part

    if total_rows != total_expected:
        raise RuntimeError(f"Wrote {total_rows} rows, expected {total_expected}; output incomplete.")

    with h5py.File(out_path, "a") as f:
        f.attrs["rows_total"] = int(total_rows)
        f.attrs["rows_by_split"] = json.dumps(
            {n: int(split_counts[s]) for s, n in ((0, "train"), (1, "val"), (2, "test"))})

    for sid, label in [(0, "train"), (1, "val"), (2, "test")]:
        pre = pre_counts.get(sid)
        extra = ""
        if quote_filter and pre:
            extra = f"  (pre-filter {pre}, dropped {pre - split_counts[sid]} = " \
                    f"{100.0 * (pre - split_counts[sid]) / pre:.3f}%)"
        print(f"  {label}: {split_counts[sid]} rows{extra}")
    if quote_filter:
        pre_tot = sum(pre_counts.values())
        print(f"  quote filter: {pre_tot} -> {total_rows} rows "
              f"({100.0 * (pre_tot - total_rows) / pre_tot:.4f}% removed)")
    if len(unique_dates):
        print(f"  {len(unique_dates)} unique dates: {unique_dates[0]} .. {unique_dates[-1]}")
        if train_end > 0:
            print(f"  Last train date: {unique_dates[train_end - 1]}")
        if val_end > train_end:
            print(f"  Last val date: {unique_dates[val_end - 1]}")
    print(f"Valid samples: {total_rows}")
    print("HDF5 v4 export complete.")


def _self_test() -> None:
    """assert-based check of the export-time maturity cut and the null->sentinel
    encoding. Synthetic frame, writes nothing."""
    lf = pl.DataFrame(
        {
            "cp_flag": ["C", "C", "C", "P"],
            "T_calendar": [0.0, 1.0 / DAY_COUNT, 30.0 / DAY_COUNT, 30.0 / DAY_COUNT],
            "T_settlement": [-6.5 / (24 * DAY_COUNT), 17.5 / (24 * DAY_COUNT),
                             (30 * 24 - 6.5) / (24 * DAY_COUNT), 30.0 / DAY_COUNT],
            "T_years": [0.0, 1.0 / DAY_COUNT, 30.0 / DAY_COUNT, 30.0 / DAY_COUNT],
            "vol_surface_vector": [[0.2] * VOL_SURFACE_DIM] * 4,
            **{c: [1.0, 1.0, 1.0, 1.0] for c in
               ["log_moneyness", "r", "q", "log_normalized_price", "moneyness", "normalized_price"]},
            "spot_history_tensor": [[[1.0] * 5] * 21] * 4,
            "vix_history_tensor": [[[1.0] * 4] * 21] * 4,
            "date": ["2020-01-02"] * 4,
        }
    ).lazy()

    # the paper rule (T > 1 day, strict) drops both the expiry-day and the 1-day row
    assert _select(lf, "C", 1.0, None).collect().height == 1
    # T > 0 keeps the 1-day row but never the expiry-day row -- no epsilon repair
    assert _select(lf, "C", 0.0, None).collect().height == 2
    # switching basis re-cuts without a rescan: the 1-day AM row holds 17.5h, so at a
    # 0.9-day threshold it survives on the calendar basis but not on the settlement one
    assert _select(lf, "C", 0.9, "calendar").collect().height == 2
    assert _select(lf, "C", 0.9, "settlement").collect().height == 1
    assert _select(lf, "C", -1.0, None).collect().height == 3  # filter disabled

    # null -> sentinel encoding
    df = pl.DataFrame({"optionid": [None, 7], "am_settlement": [None, 1],
                       "symbol": [None, "SPXW 200320C3000000"],
                       "obs_impl_volatility": [None, 0.21]})
    assert df["optionid"].fill_null(-1).to_numpy().astype(np.int64).tolist() == [-1, 7]
    assert df["am_settlement"].fill_null(255).to_numpy().astype(np.uint8).tolist() == [255, 1]
    assert df["symbol"].fill_null("").to_numpy().astype("S24").tolist() == [b"", b"SPXW 200320C3000000"]
    iv = df.select(pl.col("obs_impl_volatility").cast(pl.Float32)).to_numpy()
    assert np.isnan(iv[0, 0]) and abs(iv[1, 0] - 0.21) < 1e-6

    # --- static no-arbitrage band arithmetic ---
    # V/K units, S/K = 1.05, T = 0.5y, r = 3%, q = 1.5%
    M, T, r, q = 1.05, 0.5, 0.03, 0.015
    fwd, disc = M * np.exp(-q * T), np.exp(-r * T)
    c_lo, c_hi = max(fwd - disc, 0.0), fwd
    p_lo, p_hi = max(disc - fwd, 0.0), disc
    # put-call parity: the two lower bounds differ by exactly fwd - disc
    assert abs((c_lo - p_lo) - (fwd - disc)) < 1e-12
    # a parity-consistent pair sits strictly inside both bands
    call_v, put_v = 0.09, 0.09 - (fwd - disc)
    assert c_lo < call_v < c_hi and p_lo < put_v < p_hi

    def _band(cp, m, t, rr, qq, v):
        df = pl.DataFrame({"cp_flag": [cp], "moneyness": [m], "T_years": [t],
                           "r": [rr], "q": [qq], "normalized_price": [v]})
        return bool(df.select(_bounds_keep_expr(cp)).to_series()[0])

    assert _band("C", M, T, r, q, call_v) and _band("P", M, T, r, q, put_v)
    # bounds are inclusive, but the float32 leg moves the edge by ~1e-7 in V/K
    # units, so probe just inside it rather than exactly on it
    assert _band("C", M, T, r, q, c_lo + 1e-6) and _band("C", M, T, r, q, c_hi - 1e-6)
    assert not _band("C", M, T, r, q, c_hi + 1e-4)   # above the forward: arbitrage
    assert not _band("C", M, T, r, q, c_lo - 1e-4)   # below intrinsic: arbitrage
    assert _band("P", M, T, r, q, p_hi - 1e-6) and not _band("P", M, T, r, q, p_hi + 1e-4)
    assert not _band("P", M, T, r, q, -1e-6)          # negative midpoint
    # deep-ITM call: lower bound is the discounted intrinsic, not zero
    assert not _band("C", 2.0, T, r, q, 0.5)
    # and the filter really is a subset of the unfiltered select
    lf2 = pl.DataFrame({"cp_flag": ["C", "C"], "moneyness": [1.0, 1.0],
                        "T_years": [0.5, 0.5], "r": [0.03, 0.03], "q": [0.0, 0.0],
                        "normalized_price": [0.05, 5.0]}).lazy()
    assert lf2.filter(_bounds_keep_expr("C")).collect().height == 1

    print("self-test OK: export maturity cut + null sentinels + no-arbitrage band")


if __name__ == "__main__":
    args = _parse_args()
    if args.self_test:
        _self_test()
    elif not args.type:
        raise SystemExit("--type is required (or use --self-test)")
    else:
        build_hdf5(args.type, args.input, args.output, args.min_maturity_days, args.t_basis,
                   args.dataset_version, args.lineage_h5)
