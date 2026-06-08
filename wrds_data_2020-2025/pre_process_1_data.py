"""
Phase 1: WRDS 2015-2025 option data -> deeponet_training_data_parts/

Processes year-by-year internally to keep memory low and each chunk fast.
Writes deeponet_training_data_parts/deeponet_training_data_YYYY.parquet (skips
existing years unless --force).

Run from this directory:
    conda activate dl_new
    python pre_process_1_data.py
    python pre_process_1_data.py --date-start 2020-01-01 --date-end 2021-12-31
    python pre_process_1_data.py --force
"""

from __future__ import annotations

import argparse
import builtins
import functools
from pathlib import Path

import polars as pl

builtins.print = functools.partial(builtins.print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent

OPTION_PRICE_CSV = "optionPrice_2015_2025.csv"
VOL_SURFACE_CSV = "volatilitySurface_2015_2025.csv"
INTEREST_RATE_CSV = "interestRate_2015_2025.csv"
SPX_PRICE_CSV = "spx_price_2015_2025.csv"
VIX_CSV = "vix_2015_2025.csv"
DIVIDEND_CSV = "spx_dividendYield_2015_2025.csv"

OUTPUT_PARQUET = "deeponet_training_data.parquet"
PARTS_DIR = "deeponet_training_data_parts"

SECID = 108105
SPOT_LOOKBACK_DAYS = 21
VOL_SURFACE_DIM = 187  # 11 tenors x 17 deltas per cp_flag
LOG_EPS = 1e-8

DATE_FMT = "%Y-%m-%d"


def _csv(name: str) -> str:
    return str(SCRIPT_DIR / name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deeponet_training_data.parquet")
    parser.add_argument(
        "--date-start",
        type=str,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD) for smoke runs",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD) for smoke runs",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=SPOT_LOOKBACK_DAYS,
        help=f"Trading-day OHLC lookback window (default {SPOT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing yearly parquet part files instead of resuming.",
    )
    return parser.parse_args()


def _date_filter(lf: pl.LazyFrame, col: str, start: str | None, end: str | None) -> pl.LazyFrame:
    if start:
        lf = lf.filter(pl.col(col) >= pl.lit(start).str.to_date(DATE_FMT))
    if end:
        lf = lf.filter(pl.col(col) <= pl.lit(end).str.to_date(DATE_FMT))
    return lf


def _build_vol_surface_lazy(start: str | None, end: str | None) -> pl.LazyFrame:
    lf = (
        pl.scan_csv(_csv(VOL_SURFACE_CSV))
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("cp_flag"),
            pl.col("days"),
            pl.col("delta"),
            pl.col("impl_volatility"),
        )
    )
    lf = _date_filter(lf, "date", start, end)
    return (
        lf.sort(["date", "cp_flag", "days", "delta"])
        .group_by(["date", "cp_flag"])
        .agg(pl.col("impl_volatility").alias("vol_surface_vector"))
    )


def _build_options_lazy(start: str | None, end: str | None) -> pl.LazyFrame:
    lf = (
        pl.scan_csv(_csv(OPTION_PRICE_CSV))
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("exdate").str.to_date(DATE_FMT),
            pl.col("cp_flag"),
            pl.col("strike_price"),
            pl.col("best_bid"),
            pl.col("best_offer"),
            pl.col("volume"),
        )
    )
    lf = _date_filter(lf, "date", start, end)
    return lf.filter(
        pl.col("cp_flag").is_in(["C", "P"]) & (pl.col("volume") > 0)
    ).with_columns(
        (pl.col("strike_price") / 1000.0).alias("strike"),
        ((pl.col("best_bid") + pl.col("best_offer")) / 2.0).alias("mid_price"),
        (pl.col("exdate") - pl.col("date"))
        .dt.total_days()
        .cast(pl.Int32)
        .alias("days_to_expiry"),
    )


def _load_spx_daily(start: str | None, end: str | None) -> pl.DataFrame:
    lf = (
        pl.scan_csv(_csv(SPX_PRICE_CSV))
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
        )
    )
    lf = _date_filter(lf, "date", start, end)
    return lf.sort("date").collect()


def _load_vix_daily(start: str | None, end: str | None) -> pl.DataFrame:
    lf = (
        pl.scan_csv(_csv(VIX_CSV))
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
        )
    )
    lf = _date_filter(lf, "date", start, end)
    return lf.sort("date").collect()


def _build_ohlcv_lookback_map(
    daily: pl.DataFrame,
    lookback_days: int,
    price_cols: list[str],
    volume_col: str | None = None,
) -> tuple[dict, object | None]:
    if daily.height < lookback_days:
        return {}, None

    dates = daily["date"].to_list()
    tensor_map: dict = {}

    for i in range(lookback_days - 1, len(dates)):
        window = daily.slice(i - lookback_days + 1, lookback_days)
        d = dates[i]
        last_close = float(window[-1, "close"])
        if last_close <= 0:
            continue

        rows_out: list[float] = []
        last_vol = float(window[-1, volume_col]) if volume_col else 1.0
        if volume_col and last_vol <= 0:
            last_vol = 1.0

        for row in window.iter_rows(named=True):
            for c in price_cols:
                rows_out.append(float(row[c]) / last_close)
            if volume_col:
                rows_out.append(float(row[volume_col]) / last_vol)

        tensor_map[d] = rows_out

    first_valid = dates[lookback_days - 1]
    return tensor_map, first_valid


def _build_ohlc_lookback_map(
    daily: pl.DataFrame,
    lookback_days: int,
) -> tuple[dict, object | None]:
    return _build_ohlcv_lookback_map(
        daily, lookback_days, ["open", "high", "low", "close"], volume_col=None
    )


def _attach_lookback(
    df: pl.DataFrame,
    tensor_map: dict,
    out_col: str,
    nested_shape: tuple[int, int],
) -> pl.DataFrame:
    n_rows, n_cols = nested_shape
    expected = n_rows * n_cols

    def _to_nested(flat: list[float] | None) -> list[list[float]] | None:
        if flat is None or len(flat) != expected:
            return None
        return [flat[i * n_cols : (i + 1) * n_cols] for i in range(n_rows)]

    nested = [_to_nested(tensor_map.get(d)) for d in df["date"].to_list()]
    return df.with_columns(pl.Series(out_col, nested))


def _validate_vol_surface(df: pl.DataFrame) -> pl.DataFrame:
    n_bad = df.filter(
        pl.col("vol_surface_vector").list.len() != VOL_SURFACE_DIM
    ).height
    if n_bad > 0:
        print(
            f"WARNING: {n_bad} rows with vol_surface_vector length != {VOL_SURFACE_DIM}; dropping."
        )
        sample = (
            df.filter(pl.col("vol_surface_vector").list.len() != VOL_SURFACE_DIM)
            .select("date", "cp_flag", pl.col("vol_surface_vector").list.len())
            .head(10)
        )
        print(sample)
        df = df.filter(pl.col("vol_surface_vector").list.len() == VOL_SURFACE_DIM)
    return df


def _collect_chunk(date_start: str | None, date_end: str | None) -> pl.DataFrame:
    """Run the join pipeline for a single date window and return the result."""
    q_spot = (
        pl.scan_csv(_csv(SPX_PRICE_CSV))
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("close").alias("spot"),
        )
    )
    q_spot = _date_filter(q_spot, "date", date_start, date_end)

    q_div = (
        pl.scan_csv(_csv(DIVIDEND_CSV))
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("expiration").str.to_date(DATE_FMT).alias("exdate"),
            (pl.col("rate") / 100.0).alias("q"),
        )
    )
    q_div = _date_filter(q_div, "date", date_start, date_end)

    q_rate = (
        pl.scan_csv(_csv(INTEREST_RATE_CSV))
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("days"),
            (pl.col("rate") / 100.0).alias("r"),
        )
    )
    q_rate = _date_filter(q_rate, "date", date_start, date_end)

    q_vol = _build_vol_surface_lazy(date_start, date_end)
    q_opt = _build_options_lazy(date_start, date_end)

    q_opt = q_opt.join(q_spot, on="date", how="inner")
    q_opt = q_opt.with_columns(
        (pl.col("spot") / pl.col("strike")).alias("moneyness"),
        (pl.col("days_to_expiry") / 365.0).alias("T_years"),
        (pl.col("mid_price") / pl.col("strike")).alias("normalized_price"),
    )

    q_opt = q_opt.join(q_div, on=["date", "exdate"], how="left")
    q_opt = q_opt.with_columns(pl.col("q").fill_null(0.0))

    q_opt = q_opt.sort(["date", "days_to_expiry"])
    q_rate = q_rate.sort(["date", "days"])
    q_opt = q_opt.join_asof(
        q_rate,
        left_on="days_to_expiry",
        right_on="days",
        by="date",
        strategy="nearest",
    )

    q_opt = q_opt.join(q_vol, on=["date", "cp_flag"], how="inner")
    q_opt = q_opt.drop(["strike_price", "best_bid", "best_offer"])

    return q_opt.collect(engine="streaming")


def build_dataset(
    date_start: str | None = None,
    date_end: str | None = None,
    lookback_days: int = SPOT_LOOKBACK_DAYS,
    force: bool = False,
) -> None:
    # ---- Phase A: load small reference tables ----
    print("Building SPX / VIX lookback tensors...")
    spx_daily = _load_spx_daily(date_start, date_end)
    vix_daily = _load_vix_daily(date_start, date_end)

    spx_map, spx_first = _build_ohlcv_lookback_map(
        spx_daily,
        lookback_days,
        ["open", "high", "low", "close"],
        volume_col="volume",
    )
    vix_map, vix_first = _build_ohlc_lookback_map(vix_daily, lookback_days)

    if spx_first is None or vix_first is None:
        raise RuntimeError(
            f"Not enough SPX/VIX history for lookback window ({lookback_days} trading days)."
        )

    min_valid_date = max(spx_first, vix_first)
    print(f"Lookback warmup: dropping rows with date < {min_valid_date}")

    # ---- Phase B: process options in year-by-year chunks ----
    if date_start and date_end:
        # Single chunk when smoke-testing
        year_ranges = [(date_start, date_end)]
    else:
        default_start = "2015-01-01"
        default_end = "2025-08-29"
        s = int((date_start or default_start)[:4])
        e = int((date_end or default_end)[:4])
        year_ranges = []
        for y in range(s, e + 1):
            ys = f"{y}-01-01"
            ye = f"{y}-12-31"
            if y == s and date_start and date_start[:4] != str(s):
                ys = date_start
            if y == e and date_end and date_end[:4] != str(e):
                ye = date_end
            year_ranges.append((ys, ye))

    parts_dir = SCRIPT_DIR / PARTS_DIR
    parts_dir.mkdir(exist_ok=True)
    part_paths: list[Path] = []
    existing_parts = sorted(parts_dir.glob("deeponet_training_data_*.parquet"))
    if existing_parts and not force:
        print(
            f"Found {len(existing_parts)} existing parquet part file(s); "
            "resuming and skipping completed years. Use --force to rebuild them."
        )
        part_paths.extend(existing_parts)

    if force:
        for stale_part in existing_parts:
            stale_part.unlink()

    for ys, ye in year_ranges:
        part_path = parts_dir / f"deeponet_training_data_{ys[:4]}.parquet"
        if part_path.exists() and not force:
            print(f"  Skipping {ys} .. {ye}; {part_path.name} already exists")
            continue

        print(f"  Processing {ys} .. {ye} ...")
        df_chunk = _collect_chunk(ys, ye)
        print(f"    -> {df_chunk.height} rows after joins")
        if df_chunk.height == 0:
            continue

        df_chunk = _validate_vol_surface(df_chunk)
        print(f"    -> {df_chunk.height} rows after vol surface validation")
        if df_chunk.height == 0:
            continue

        df_chunk = df_chunk.with_columns(
            pl.col("moneyness").clip(lower_bound=LOG_EPS).log().alias("log_moneyness"),
            pl.col("normalized_price")
            .clip(lower_bound=LOG_EPS)
            .log()
            .alias("log_normalized_price"),
        )

        df_chunk = df_chunk.filter(pl.col("date") >= min_valid_date)
        df_chunk = _attach_lookback(
            df_chunk, spx_map, "spot_history_tensor", nested_shape=(lookback_days, 5)
        )
        df_chunk = _attach_lookback(
            df_chunk, vix_map, "vix_history_tensor", nested_shape=(lookback_days, 4)
        )
        df_chunk = df_chunk.filter(
            pl.col("spot_history_tensor").is_not_null()
            & pl.col("vix_history_tensor").is_not_null()
        )
        df_chunk = df_chunk.sort(["date", "cp_flag", "days_to_expiry", "strike"])

        if df_chunk.height == 0:
            continue

        df_chunk = df_chunk.with_row_index("row_id")

        print(f"    -> writing part {part_path.name} ({df_chunk.height} rows)")
        tmp_path = part_path.with_suffix(".parquet.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        df_chunk.write_parquet(tmp_path)
        tmp_path.replace(part_path)
        part_paths.append(part_path)

        del df_chunk

    if not part_paths:
        raise RuntimeError("No data after processing all year chunks.")

    print(f"\nDone. Wrote {len(part_paths)} parquet part files to {parts_dir}.")
    print(
        "Use deeponet_training_data_parts/ as the parquet dataset for Phase 2 "
        "and inspection."
    )


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        date_start=args.date_start,
        date_end=args.date_end,
        lookback_days=args.lookback_days,
        force=args.force,
    )
