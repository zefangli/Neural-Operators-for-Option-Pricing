"""
Phase 1 (v4): WRDS 2015-2025 option data -> deeponet_training_data_parts_v4/

Evolved copy of pre_process_1_data.py. Same joins, same filters, but it RETAINS
fields v3 threw away and computes a settlement-aware time to expiry.

What v4 adds over v3
--------------------
* Contract identity:  symbol, root, optionid, am_settlement, exercise_style.
  v3 dropped these, which collapsed AM-settled SPX and PM-settled SPXW contracts
  that share (date, strike, expiry) into byte-identical model inputs.
* Observed quantities: impl_volatility (OptionMetrics IV), delta (market delta),
  best_bid, best_offer, and derived spread_norm / half_spread_norm (= (ask-bid)/K
  and half of it, normalized by strike to match normalized_price = mid/K).
* vix_level: the absolute VIX close on the trade date. vix_history is normalized
  by its own last close, so the level is destroyed there.
* Corrected VIX source: vix_2015_2025.csv is the Cboe wide-format SPX VIX file
  (vixo/vixh/vixl/vix + VXN/VXD). v3 mislabeled VVIX under this filename;
  v4 reads only the VIX columns, records full provenance, and refuses to load
  any other layout (see _validate_vix_schema / _vix_lazy).
* T_calendar (the v3 definition) AND T_settlement (below). `T_years` carries
  whichever basis --t-basis selects; both are always written out.

Settlement-aware maturity  (--t-basis settlement, the default)
--------------------------------------------------------------
ASSUMED CONVENTION -- override with the CLI flags, do not hardcode elsewhere:
  * OptionMetrics IvyDB quotes are end-of-day snapshots; we place the quote at
    16:00 ET on `date`                                    (--quote-time-et)
  * am_settlement == 1 (AM-settled, the SPX monthly / "third Friday" series)
    settles against the Special Opening Quotation, i.e. the opening print of the
    expiry date, 09:30 ET                                 (--am-settle-time-et)
  * am_settlement == 0 (PM-settled, the SPXW weeklies/EOM) settles against the
    closing index level on the expiry date, 16:00 ET      (--pm-settle-time-et)
Then
    T_settlement = (calendar_days * 24h + settle_hour - quote_hour) / (24 * 365)
Under the default times this means an AM-settled contract observed on its own
expiry date has T = -6.5h, i.e. it had ALREADY settled hours before the quote
was taken, and a PM-settled one has T = 0 exactly. Neither is repaired with an
epsilon -- both are removed by the minimum-maturity filter. Nothing here
manufactures optionality; it only makes the v3 `days/365` less wrong away from
expiry (an AM-settled contract is short 6.5 hours of the calendar tenor).

Minimum maturity
----------------
--min-maturity-days (default 1.0) keeps rows with T_active_days > threshold,
matching the project's headline metric R^2(log, T > 1/365). Set 0.0 to keep
everything with strictly positive maturity. T <= 0 is ALWAYS dropped.

Run from this directory:
    conda activate dl_new
    python pre_process_1_data_v4.py --self-test
    python pre_process_1_data_v4.py --date-start 2019-11-01 --date-end 2020-02-29 \
        --parts-dir smoke_v4/parts
    python pre_process_1_data_v4.py            # full rebuild (gated, see report)
"""

from __future__ import annotations

import argparse
import builtins
import functools
import hashlib
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

builtins.print = functools.partial(builtins.print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent

OPTION_PRICE_CSV = "optionPrice_2015_2025.csv"
VOL_SURFACE_CSV = "volatilitySurface_2015_2025.csv"
INTEREST_RATE_CSV = "interestRate_2015_2025.csv"
SPX_PRICE_CSV = "spx_price_2015_2025.csv"
VIX_CSV = "vix_2015_2025.csv"
DIVIDEND_CSV = "spx_dividendYield_2015_2025.csv"

PARTS_DIR = "deeponet_training_data_parts_v4"  # NEW path; v3 parts are untouched
CONFIG_JSON = "pipeline_config_v4.json"

SECID = 108105
SPOT_LOOKBACK_DAYS = 21
VOL_SURFACE_DIM = 187  # 11 tenors x 17 deltas per cp_flag
LOG_EPS = 1e-8
DAY_COUNT = 365.0

# Settlement convention defaults (ET, 24h). See module docstring.
QUOTE_TIME_ET = "16:00"
AM_SETTLE_TIME_ET = "09:30"
PM_SETTLE_TIME_ET = "16:00"

DATE_FMT = "%Y-%m-%d"

# Only the columns we actually read; pinned so schema inference on the 8.6 GB CSV
# cannot flip a mostly-empty column to Null and blow up mid-scan.
OPTION_SCHEMA = {
    "secid": pl.Int64,
    "date": pl.String,
    "symbol": pl.String,
    "exdate": pl.String,
    "cp_flag": pl.String,
    "strike_price": pl.Float64,
    "best_bid": pl.Float64,
    "best_offer": pl.Float64,
    "volume": pl.Int64,
    "open_interest": pl.Int64,
    "impl_volatility": pl.Float64,
    "delta": pl.Float64,
    "optionid": pl.Int64,
    "am_settlement": pl.Int64,
    "exercise_style": pl.String,
}


def _csv(name: str) -> str:
    return str(SCRIPT_DIR / name)


def _hours(hhmm: str) -> float:
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v4 parquet parts (identity + quotes retained)")
    p.add_argument("--date-start", type=str, default=None,
                   help="Optional inclusive start date (YYYY-MM-DD) for smoke runs")
    p.add_argument("--date-end", type=str, default=None,
                   help="Optional inclusive end date (YYYY-MM-DD) for smoke runs")
    p.add_argument("--lookback-days", type=int, default=SPOT_LOOKBACK_DAYS,
                   help=f"Trading-day OHLC lookback window (default {SPOT_LOOKBACK_DAYS})")
    p.add_argument("--chunk-months", type=int, default=12,
                   help="Months of trade dates per parquet part (default 12 = one part per "
                        "year, the v3 behaviour). Part size sets phase 2's peak memory "
                        "(~3.4-4.3 GB per 176k-row part); use 3 for the full rebuild.")
    p.add_argument("--parts-dir", type=str, default=PARTS_DIR,
                   help=f"Output directory for parquet parts (default {PARTS_DIR})")
    p.add_argument("--t-basis", choices=["calendar", "settlement"], default="settlement",
                   help="Which maturity goes into T_years. 'calendar' = v3 days/365.")
    p.add_argument("--min-maturity-days", type=float, default=1.0,
                   help="Keep rows with T_active_days > this (default 1.0 = paper rule). "
                        "0.0 keeps everything with strictly positive maturity. "
                        "-1 disables the filter entirely (DIAGNOSTIC ONLY: keeps T<=0 rows, "
                        "which are not priceable; phase 2 must then cut the paper sample).")
    p.add_argument("--quote-time-et", type=str, default=QUOTE_TIME_ET,
                   help=f"Assumed OptionMetrics quote time, ET (default {QUOTE_TIME_ET})")
    p.add_argument("--am-settle-time-et", type=str, default=AM_SETTLE_TIME_ET,
                   help=f"AM (SPX) settlement time, ET (default {AM_SETTLE_TIME_ET})")
    p.add_argument("--pm-settle-time-et", type=str, default=PM_SETTLE_TIME_ET,
                   help=f"PM (SPXW) settlement time, ET (default {PM_SETTLE_TIME_ET})")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing yearly parquet part files instead of resuming.")
    p.add_argument("--self-test", action="store_true",
                   help="Run the assert-based self-check and exit (touches no data).")
    return p.parse_args()


def _chunk_ranges(start: str, end: str, months: int) -> list[tuple[str, str]]:
    """Contiguous, non-overlapping [start, end] windows of at most `months` months.

    Part size is the memory knob for the WHOLE pipeline, and phase 2 is the
    binding constraint: it loads one part at a time and measured 3.4-4.3 GB of
    private commit on a 176k-row part. Phase 1 itself peaked at 3.4 GB for the
    same chunk (the 8.6 GB CSV is memory-mapped, so its working set looks huge
    but is reclaimable page cache, not commit). On a 15.7 GB box a 12-month part
    (~1.2M rows) is the risky end -- use --chunk-months 3 for the full rebuild.
    """
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out: list[tuple[str, str]] = []
    while s <= e:
        y, m = divmod(s.year * 12 + s.month - 1 + months, 12)
        nxt = date(y, m + 1, 1)
        out.append((s.isoformat(), min(nxt - timedelta(days=1), e).isoformat()))
        s = nxt
    return out


def _date_filter(lf: pl.LazyFrame, col: str, start: str | None, end: str | None) -> pl.LazyFrame:
    if start:
        lf = lf.filter(pl.col(col) >= pl.lit(start).str.to_date(DATE_FMT))
    if end:
        lf = lf.filter(pl.col(col) <= pl.lit(end).str.to_date(DATE_FMT))
    return lf


# --------------------------------------------------------------------------- #
# settlement-aware maturity
# --------------------------------------------------------------------------- #
def settlement_t_expr(
    quote_time: str = QUOTE_TIME_ET,
    am_time: str = AM_SETTLE_TIME_ET,
    pm_time: str = PM_SETTLE_TIME_ET,
) -> pl.Expr:
    """T in years from the assumed quote time to the assumed settlement time.

    Needs columns `days_to_expiry` (calendar) and `is_am_settled` (bool).
    Can be negative (AM-settled contract quoted at the close of its expiry date
    has already settled); callers must filter, never clamp.
    """
    offset_h = (
        pl.when(pl.col("is_am_settled"))
        .then(pl.lit(_hours(am_time) - _hours(quote_time)))
        .otherwise(pl.lit(_hours(pm_time) - _hours(quote_time)))
    )
    return (pl.col("days_to_expiry") * 24.0 + offset_h) / (24.0 * DAY_COUNT)


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
        pl.scan_csv(_csv(OPTION_PRICE_CSV), schema_overrides=OPTION_SCHEMA)
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("exdate").str.to_date(DATE_FMT),
            pl.col("cp_flag"),
            pl.col("strike_price"),
            pl.col("best_bid"),
            pl.col("best_offer"),
            pl.col("volume"),
            # --- v4 retained fields ---
            pl.col("symbol"),
            pl.col("optionid"),
            pl.col("am_settlement"),
            pl.col("exercise_style"),
            pl.col("impl_volatility").alias("obs_impl_volatility"),
            pl.col("delta").alias("obs_delta"),
            pl.col("open_interest"),
        )
    )
    lf = _date_filter(lf, "date", start, end)
    return lf.filter(
        pl.col("cp_flag").is_in(["C", "P"]) & (pl.col("volume") > 0)
    ).with_columns(
        (pl.col("strike_price") / 1000.0).alias("strike"),
        ((pl.col("best_bid") + pl.col("best_offer")) / 2.0).alias("mid_price"),
        (pl.col("exdate") - pl.col("date")).dt.total_days().cast(pl.Int32).alias("days_to_expiry"),
        # "SPX 200320C3000000" / "SPXW 200320C3000000" -> root
        pl.col("symbol").str.split(" ").list.first().alias("root"),
        # am_settlement is the authority; if OptionMetrics left it null, fall back
        # to the root (plain SPX == AM-settled monthly, SPXW == PM-settled).
        pl.col("am_settlement").eq(1)
        .fill_null(pl.col("symbol").str.split(" ").list.first() == "SPX")
        .alias("is_am_settled"),
    )


def _load_spx_daily(start: str | None, end: str | None) -> pl.DataFrame:
    lf = (
        pl.scan_csv(_csv(SPX_PRICE_CSV))
        .filter(pl.col("secid") == SECID)
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"),
        )
    )
    lf = _date_filter(lf, "date", start, end)
    return lf.sort("date").collect()


def _validate_vix_schema(csv_path: str | None = None) -> None:
    """Fail fast if the VIX CSV is not the Cboe wide format we rely on.

    Requires the SPX VIX columns AND the VXN/VXD columns: if vxno/vxdo ever
    disappear (file stripped, or it reverts to the old WRDS/VVIX layout with
    secid/open/high/low/close/ticker), the data read as "VIX" could silently be
    something else. No fallback, no guessing.
    """
    path = Path(csv_path) if csv_path else _csv(VIX_CSV)
    cols = set(pl.scan_csv(str(path)).collect_schema().names())
    missing = ({"date", "vixo", "vixh", "vixl", "vix"} | {"vxno", "vxdo"}) - cols
    if missing:
        raise RuntimeError(
            f"{path.name} is not the expected Cboe wide VIX file: missing columns "
            f"{sorted(missing)}. Expected header "
            "date,vixo,vixh,vixl,vix,vxno,vxnh,vxnl,vxn,vxdo,vxdh,vxdl,vxd. "
            "Refusing to load (no fallback to vvix_2015_2025.csv or the WRDS format)."
        )


def _vix_lazy() -> pl.LazyFrame:
    """Standardized lazy VIX frame: date, open, high, low, close (genuine Cboe VIX).

    Renames vixo/vixh/vixl/vix -> open/high/low/close and drops rows with no VIX
    print (the Cboe file leaves OHLC empty on 3 holidays -> null). VXN/VXD
    columns are never selected. Keeps only strictly positive closes.
    """
    _validate_vix_schema()
    return (
        pl.scan_csv(_csv(VIX_CSV))
        .select(
            pl.col("date").str.to_date(DATE_FMT),
            pl.col("vixo").alias("open"),
            pl.col("vixh").alias("high"),
            pl.col("vixl").alias("low"),
            pl.col("vix").alias("close"),
        )
        .filter(pl.col("close").is_not_null() & (pl.col("close") > 0.0))
    )


def _load_vix_daily(start: str | None, end: str | None) -> pl.DataFrame:
    lf = _vix_lazy()
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


def _build_ohlc_lookback_map(daily: pl.DataFrame, lookback_days: int) -> tuple[dict, object | None]:
    return _build_ohlcv_lookback_map(
        daily, lookback_days, ["open", "high", "low", "close"], volume_col=None
    )


def _attach_lookback(
    df: pl.DataFrame, tensor_map: dict, out_col: str, nested_shape: tuple[int, int]
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
    n_bad = df.filter(pl.col("vol_surface_vector").list.len() != VOL_SURFACE_DIM).height
    if n_bad > 0:
        print(f"WARNING: {n_bad} rows with vol_surface_vector length != {VOL_SURFACE_DIM}; dropping.")
        df = df.filter(pl.col("vol_surface_vector").list.len() == VOL_SURFACE_DIM)
    return df


def _collect_chunk(date_start: str | None, date_end: str | None, cfg: dict) -> pl.DataFrame:
    """Run the join pipeline for a single date window and return the result."""
    q_spot = (
        pl.scan_csv(_csv(SPX_PRICE_CSV))
        .filter(pl.col("secid") == SECID)
        .select(pl.col("date").str.to_date(DATE_FMT), pl.col("close").alias("spot"))
    )
    q_spot = _date_filter(q_spot, "date", date_start, date_end)

    # v4: absolute SPX VIX level on the trade date (vix_history is normalized by
    # its own last close, which destroys the level). Same loader as vix_history,
    # so it is guaranteed to be genuine CBOE VIX -- v3 mislabeled VVIX here; the
    # corrected Cboe wide file is read via _vix_lazy (see its docstring).
    q_vix_level = _vix_lazy().select(pl.col("date"), pl.col("close").alias("vix_level"))
    q_vix_level = _date_filter(q_vix_level, "date", date_start, date_end)

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

    q_rate = pl.scan_csv(_csv(INTEREST_RATE_CSV)).select(
        pl.col("date").str.to_date(DATE_FMT), pl.col("days"), (pl.col("rate") / 100.0).alias("r")
    )
    q_rate = _date_filter(q_rate, "date", date_start, date_end)

    q_vol = _build_vol_surface_lazy(date_start, date_end)
    q_opt = _build_options_lazy(date_start, date_end)

    q_opt = q_opt.join(q_spot, on="date", how="inner")
    q_opt = q_opt.join(q_vix_level, on="date", how="left")
    q_opt = q_opt.with_columns(
        (pl.col("spot") / pl.col("strike")).alias("moneyness"),
        (pl.col("days_to_expiry") / DAY_COUNT).alias("T_calendar"),
        settlement_t_expr(
            cfg["quote_time_et"], cfg["am_settle_time_et"], cfg["pm_settle_time_et"]
        ).alias("T_settlement"),
        (pl.col("mid_price") / pl.col("strike")).alias("normalized_price"),
        ((pl.col("best_offer") - pl.col("best_bid")) / pl.col("strike")).alias("spread_norm"),
    )
    q_opt = q_opt.with_columns((pl.col("spread_norm") / 2.0).alias("half_spread_norm"))

    q_opt = q_opt.join(q_div, on=["date", "exdate"], how="left")
    q_opt = q_opt.with_columns(pl.col("q").fill_null(0.0))

    q_opt = q_opt.sort(["date", "days_to_expiry"])
    q_rate = q_rate.sort(["date", "days"])
    q_opt = q_opt.join_asof(
        q_rate, left_on="days_to_expiry", right_on="days", by="date", strategy="nearest"
    )

    q_opt = q_opt.join(q_vol, on=["date", "cp_flag"], how="inner")
    q_opt = q_opt.drop("strike_price")

    return q_opt.collect(engine="streaming")


def build_dataset(
    date_start: str | None = None,
    date_end: str | None = None,
    lookback_days: int = SPOT_LOOKBACK_DAYS,
    chunk_months: int = 12,
    parts_dir: str = PARTS_DIR,
    t_basis: str = "settlement",
    min_maturity_days: float = 1.0,
    quote_time_et: str = QUOTE_TIME_ET,
    am_settle_time_et: str = AM_SETTLE_TIME_ET,
    pm_settle_time_et: str = PM_SETTLE_TIME_ET,
    force: bool = False,
) -> None:
    parts_path = SCRIPT_DIR / parts_dir
    # Trust boundary: the v3 parquet/HDF5 artifacts are irreplaceable.
    assert "v4" in parts_path.name or "v4" in str(parts_path), (
        f"refusing to write outside a v4-named directory: {parts_path}"
    )
    assert min_maturity_days >= 0.0 or min_maturity_days == -1.0, (
        "--min-maturity-days must be >= 0, or exactly -1 for the diagnostic unfiltered build"
    )
    if min_maturity_days == -1.0:
        print("WARNING: maturity filter DISABLED (diagnostic build). T <= 0 rows are kept and "
              "are NOT priceable; cut the paper sample in phase 2 before training.")

    # provenance, not cosmetics: full-source record of the corrected Cboe VIX
    # file, mirrored verbatim into the HDF5 attrs by phase 2. v3 shipped VVIX
    # under this filename; v4 reads the genuine Cboe wide file (see _vix_lazy).
    vix_full = _load_vix_daily(None, None)
    cfg = {
        "pipeline_version": "v4",
        "t_basis": t_basis,
        "min_maturity_days": min_maturity_days,
        "quote_time_et": quote_time_et,
        "am_settle_time_et": am_settle_time_et,
        "pm_settle_time_et": pm_settle_time_et,
        "day_count": DAY_COUNT,
        "lookback_days": lookback_days,
        "chunk_months": chunk_months,
        "date_start": date_start,
        "date_end": date_end,
        "vix_source_file": VIX_CSV,
        "vix_source_series": "CBOE VIX (S&P 500 volatility index)",
        "vix_source_columns": "date,vixo,vixh,vixl,vix",
        "vix_column_mapping": "vixo->open, vixh->high, vixl->low, vix->close",
        "vix_sha256": hashlib.sha256(Path(_csv(VIX_CSV)).read_bytes()).hexdigest(),
        # Both counts, so the record is unambiguous: the source file has 2,709
        # data rows, 3 of which are holidays with an empty VIX print and are
        # dropped by _vix_lazy(), leaving 2,706 usable. Recording only the
        # filtered count made the file look like it disagreed with its own sha.
        "vix_raw_row_count": pl.scan_csv(_csv(VIX_CSV)).select(pl.len()).collect().item(),
        "vix_usable_row_count": vix_full.height,
        "vix_date_range": f"{vix_full['date'].min().isoformat()} .. "
                          f"{vix_full['date'].max().isoformat()}",
        "settlement_convention": (
            "OptionMetrics EOD quote at {q} ET; am_settlement==1 settles at the SOQ "
            "(open, {a} ET) of exdate; am_settlement==0 settles at the close ({p} ET) "
            "of exdate. T_settlement = (cal_days*24 + settle_h - quote_h)/(24*365)."
        ).format(q=quote_time_et, a=am_settle_time_et, p=pm_settle_time_et),
    }

    print("Building SPX / VIX lookback tensors...")
    spx_daily = _load_spx_daily(date_start, date_end)
    vix_daily = _load_vix_daily(date_start, date_end)

    spx_map, spx_first = _build_ohlcv_lookback_map(
        spx_daily, lookback_days, ["open", "high", "low", "close"], volume_col="volume"
    )
    vix_map, vix_first = _build_ohlc_lookback_map(vix_daily, lookback_days)

    if spx_first is None or vix_first is None:
        raise RuntimeError(
            f"Not enough SPX/VIX history for lookback window ({lookback_days} trading days)."
        )

    min_valid_date = max(spx_first, vix_first)
    print(f"Lookback warmup: dropping rows with date < {min_valid_date}")
    print(f"Maturity: basis={t_basis}, min_maturity_days={min_maturity_days}")
    print(f"          {cfg['settlement_convention']}")

    chunk_ranges = _chunk_ranges(date_start or "2015-01-01", date_end or "2025-08-29", chunk_months)
    print(f"Processing in {len(chunk_ranges)} chunk(s) of <= {chunk_months} month(s)")

    parts_path.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    existing_parts = sorted(parts_path.glob("deeponet_training_data_*.parquet"))
    if existing_parts and not force:
        print(f"Found {len(existing_parts)} existing part file(s); resuming. Use --force to rebuild.")
        part_paths.extend(existing_parts)
    if force:
        for stale_part in existing_parts:
            stale_part.unlink()

    attrition: list[dict] = []

    for ys, ye in chunk_ranges:
        part_path = parts_path / f"deeponet_training_data_{ys}.parquet"
        if part_path.exists() and not force:
            print(f"  Skipping {ys} .. {ye}; {part_path.name} already exists")
            continue

        print(f"  Processing {ys} .. {ye} ...")
        df_chunk = _collect_chunk(ys, ye, cfg)
        stage = {"chunk": ys, "after_joins": df_chunk.height}
        print(f"    -> {df_chunk.height} rows after joins")
        if df_chunk.height == 0:
            continue

        df_chunk = _validate_vol_surface(df_chunk)
        stage["after_vol_surface"] = df_chunk.height
        print(f"    -> {df_chunk.height} rows after vol surface validation")
        if df_chunk.height == 0:
            continue

        # T_years is the ACTIVE basis; both bases are always retained alongside.
        df_chunk = df_chunk.with_columns(
            pl.col("T_settlement" if t_basis == "settlement" else "T_calendar").alias("T_years")
        )
        stage["T_le_0"] = df_chunk.filter(pl.col("T_years") <= 0.0).height
        stage["T_0_to_min"] = df_chunk.filter(
            (pl.col("T_years") > 0.0) & (pl.col("T_years") * DAY_COUNT <= min_maturity_days)
        ).height
        if min_maturity_days >= 0.0:
            df_chunk = df_chunk.filter(pl.col("T_years") * DAY_COUNT > min_maturity_days)
        stage["after_min_maturity"] = df_chunk.height
        print(f"    -> {df_chunk.height} rows after min-maturity filter "
              f"(T<=0: {stage['T_le_0']}, 0<T<=min: {stage['T_0_to_min']})")
        if df_chunk.height == 0:
            continue

        df_chunk = df_chunk.with_columns(
            pl.col("moneyness").clip(lower_bound=LOG_EPS).log().alias("log_moneyness"),
            pl.col("normalized_price").clip(lower_bound=LOG_EPS).log().alias("log_normalized_price"),
        )

        df_chunk = df_chunk.filter(pl.col("date") >= min_valid_date)
        stage["after_lookback_warmup"] = df_chunk.height
        df_chunk = _attach_lookback(df_chunk, spx_map, "spot_history_tensor", (lookback_days, 5))
        df_chunk = _attach_lookback(df_chunk, vix_map, "vix_history_tensor", (lookback_days, 4))
        df_chunk = df_chunk.filter(
            pl.col("spot_history_tensor").is_not_null() & pl.col("vix_history_tensor").is_not_null()
        )
        stage["after_history"] = df_chunk.height
        df_chunk = df_chunk.sort(["date", "cp_flag", "days_to_expiry", "strike"])
        attrition.append(stage)

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

    cfg["attrition"] = attrition
    (parts_path / CONFIG_JSON).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {len(part_paths)} parquet part files to {parts_path}.")
    print(f"Provenance written to {parts_path / CONFIG_JSON} (phase 2 copies it into HDF5 attrs).")


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _vix_self_test() -> None:
    """assert-based checks of the Cboe VIX loader against the real file (read-only).

    Covers: unique sorted dates; finite positive OHLC after the empty/zero-row
    filter; OHLC range consistency; the 2015-01-02 reference row; VXN/VXD never
    entering the loader output (by construction); the 21-day normalized window
    vs a hand-built calculation; and rejection of a legacy VVIX-format file.
    """
    raw = pl.scan_csv(_csv(VIX_CSV)).select(pl.col("date").str.to_date(DATE_FMT)).collect()
    assert raw.height == 2709, raw.height  # 2015-01-02 .. 2025-08-29, one row per trade day
    assert raw["date"].is_sorted(), "VIX dates must be sorted ascending"
    assert raw["date"].n_unique() == raw.height, "VIX dates must be unique"

    daily = _load_vix_daily(None, None)
    assert daily.height == 2706, daily.height  # 2709 - 3 holidays with no VIX print
    ohlc = daily.select(["open", "high", "low", "close"])
    assert np.isfinite(ohlc.to_numpy()).all(), "VIX OHLC must be finite"
    assert (ohlc > 0).to_numpy().all(), "VIX OHLC must be strictly positive"

    # low <= min(open, close) and high >= max(open, close) on every row
    assert (daily["low"] <= daily["open"]).all() and (daily["low"] <= daily["close"]).all()
    assert (daily["high"] >= daily["open"]).all() and (daily["high"] >= daily["close"]).all()

    # reference: 2015-01-02 VIX OHLC = [17.76, 20.14, 17.05, 17.79] exactly
    ref = daily.filter(pl.col("date") == date(2015, 1, 2))
    assert ref.height == 1
    assert ref.select(["open", "high", "low", "close"]).row(0) == (17.76, 20.14, 17.05, 17.79)

    # by construction, VXN/VXD never enter the loader output
    loader_cols = set(daily.columns)
    assert loader_cols == {"date", "open", "high", "low", "close"}, loader_cols
    assert loader_cols.isdisjoint(
        {"vxno", "vxnh", "vxnl", "vxn", "vxdo", "vxdh", "vxdl", "vxd"}
    )

    # 21-day normalized VIX window: builder output vs hand-computed from raw rows
    vix_map, _ = _build_ohlc_lookback_map(daily, SPOT_LOOKBACK_DAYS)
    probe = date(2020, 1, 2)  # deep in the series, lookback window fully available
    assert probe in vix_map
    window = daily.filter(pl.col("date") <= probe).tail(SPOT_LOOKBACK_DAYS)
    last_close = float(window["close"].to_list()[-1])
    manual = [float(r[c]) / last_close
              for r in window.iter_rows(named=True)
              for c in ("open", "high", "low", "close")]
    flat = vix_map[probe]
    assert len(flat) == SPOT_LOOKBACK_DAYS * 4
    for a, b in zip(flat, manual):
        assert abs(a - b) <= 1e-6 * max(1.0, abs(b)), (a, b)

    # a legacy VVIX-format file (WRDS columns) must be rejected, not misread
    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "legacy_vvix.csv"
        legacy.write_text(
            "secid,date,open,high,low,close,ticker\n"
            "152892,2015-01-02,17.76,20.14,17.05,17.79,VVIX\n"
        )
        try:
            _validate_vix_schema(str(legacy))
        except RuntimeError as e:
            assert "missing columns" in str(e), e
        else:
            raise AssertionError("legacy VVIX-format file must be rejected")

    print("self-test OK: CBOE VIX loader (schema, OHLC, reference row, window)")


def _self_test() -> None:
    """assert-based check of the VIX loader (real file) plus the two non-trivial
    pieces: settlement-aware T and the duplicate-input-key collapse (synthetic)."""
    _vix_self_test()
    # ---- settlement-aware maturity -------------------------------------- #
    df = pl.DataFrame(
        {
            "days_to_expiry": [0, 0, 30, 30, 1, 1],
            "is_am_settled": [True, False, True, False, True, False],
        }
    ).with_columns(settlement_t_expr().alias("T"))
    t = df["T"].to_list()
    hour = 1.0 / (24.0 * DAY_COUNT)
    # AM contract quoted at 16:00 on its expiry date settled 6.5h EARLIER -> negative
    assert abs(t[0] - (-6.5 * hour)) < 1e-12, t[0]
    # PM contract quoted at the close of its expiry date settles at that instant -> 0
    assert abs(t[1] - 0.0) < 1e-12, t[1]
    # away from expiry, AM is short 6.5h of the calendar tenor, PM matches it exactly
    assert abs(t[2] - (30.0 / DAY_COUNT - 6.5 * hour)) < 1e-12, t[2]
    assert abs(t[3] - 30.0 / DAY_COUNT) < 1e-12, t[3]
    # nothing is clamped to an epsilon: expiry-day rows stay <= 0
    assert t[0] < 0.0 and t[1] <= 0.0
    # a 1-calendar-day AM contract holds only 17.5h, a PM one exactly 24h; the
    # default filter is strict (> 1 day) so neither survives the paper sample
    assert t[4] * DAY_COUNT < 1.0 and abs(t[5] * DAY_COUNT - 1.0) < 1e-12, (t[4], t[5])
    # configurable, not a magic constant
    alt = pl.DataFrame({"days_to_expiry": [0], "is_am_settled": [False]}).with_columns(
        settlement_t_expr(quote_time="15:00", pm_time="16:15").alias("T")
    )["T"][0]
    assert abs(alt - 1.25 * hour) < 1e-12, alt

    # ---- duplicate model-input key collapse ------------------------------ #
    # The model sees (branch_u, spot_history, vix_history) -- all constant within a
    # trade date -- plus trunk_y = [log_moneyness, T, r, q]. So the identity the
    # model can actually resolve is this key:
    key_cal = ["date", "log_moneyness", "T_calendar", "r", "q"]
    key_set = ["date", "log_moneyness", "T_settlement", "r", "q"]
    dup = pl.DataFrame(
        {
            "date": ["2020-01-02"] * 4,
            "symbol": ["SPX 200117C3000000", "SPXW 200117C3000000",
                       "SPX 200117C3100000", "SPXW 200117C3100000"],
            "log_moneyness": [0.01, 0.01, -0.02, -0.02],
            "days_to_expiry": [15, 15, 15, 15],
            "is_am_settled": [True, False, True, False],
            "r": [0.015] * 4,
            "q": [0.019] * 4,
            "normalized_price": [0.031, 0.030, 0.011, 0.012],  # genuinely different prices
        }
    ).with_columns(
        (pl.col("days_to_expiry") / DAY_COUNT).alias("T_calendar"),
        settlement_t_expr().alias("T_settlement"),
    )

    def _dup_rows(frame: pl.DataFrame, key: list[str]) -> int:
        return frame.filter(pl.len().over(key) > 1).height

    assert _dup_rows(dup, key_cal) == 4, "v3 calendar-T key must collide AM vs PM"
    assert _dup_rows(dup, key_set) == 0, "settlement-aware T must separate AM from PM"
    # and the collision was not benign: the colliding rows carry different prices
    collided = dup.group_by(key_cal).agg(pl.col("normalized_price").n_unique().alias("n"))
    assert collided["n"].max() > 1

    # ---- chunk ranges: no gaps, no overlaps, exact coverage --------------- #
    for months in (1, 3, 6, 12):
        rs = _chunk_ranges("2015-01-01", "2025-08-29", months)
        assert rs[0][0] == "2015-01-01" and rs[-1][1] == "2025-08-29", rs[:1] + rs[-1:]
        for (_, a), (b, _) in zip(rs, rs[1:]):
            assert date.fromisoformat(b) - date.fromisoformat(a) == timedelta(days=1), (a, b)
    assert len(_chunk_ranges("2015-01-01", "2025-08-29", 12)) == 11
    assert _chunk_ranges("2019-11-01", "2020-02-29", 12) == [("2019-11-01", "2020-02-29")]

    print("self-test OK: settlement maturity + duplicate-key collapse + chunking")


if __name__ == "__main__":
    args = _parse_args()
    if args.self_test:
        _self_test()
    else:
        build_dataset(
            date_start=args.date_start,
            date_end=args.date_end,
            lookback_days=args.lookback_days,
            chunk_months=args.chunk_months,
            parts_dir=args.parts_dir,
            t_basis=args.t_basis,
            min_maturity_days=args.min_maturity_days,
            quote_time_et=args.quote_time_et,
            am_settle_time_et=args.am_settle_time_et,
            pm_settle_time_et=args.pm_settle_time_et,
            force=args.force,
        )
