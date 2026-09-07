# Learning Implied Volatility, Not Prices

A Neural-Operator Encoder with a Differentiable Black–Scholes Decoder for SPX Option Pricing.

## The idea

Analytical pricing formulas such as Black–Scholes are exact once the implied volatility is
known, but most neural option pricers throw that structure away and learn a direct, opaque
map from market state and contract terms straight to price.

This project instead learns a **per-contract implied volatility** σ̂ and feeds it through the
closed-form Black–Scholes formula, so a plain data loss on `log(V/K)` back-propagates through
the analytical pricer — there is no learned surrogate pricer, no PDE/arbitrage loss term, just
MSE on the log-normalized price after an exact analytical decoder.

A 2-D Fourier Neural Operator (FNO) encodes the market state from one of three candidate
inputs — the contemporaneous implied-volatility surface, SPX price history, or VIX history —
into a latent vector. A per-query head (either an MLP or a DeepONet branch–trunk readout) then
maps `(latent, log-moneyness, maturity)` to σ̂ for that specific contract, which is what lets
the model reproduce the volatility smile/skew instead of fitting one scalar per market state.

## Headline results

On quote-quality-filtered SPX options (OptionMetrics via WRDS, 2015–2025), split 80/10/10 by
trade date:

| | call | put |
|---|---|---|
| R²(price), vol-surface branch | 0.9999–1.0000 | 0.9926–0.9938 |
| R²(log, T > 1 day), vol-surface branch | 0.990–0.991 | 0.978–0.981 |
| R²(log, T > 1 day), spot/VIX-history branches | 0.90–0.94 | 0.90–0.92 |

Two results this table can't show on its own:

- **Per-query conditioning has the largest effect.** Collapsing σ̂ to one scalar per market state
  instead of one per contract drops put R²(log, T>1 day) from 0.978 to −0.317 and put R²(price) from
  0.993 to −3.090; calls move the same way (0.991 → 0.800).
- **Greeks come from the same analytical decoder.** Autograd through the decoder agrees with
  closed-form Black–Scholes at the predicted σ̂ to a maximum absolute discrepancy of ~2×10⁻¹⁴
  (delta) and ~2×10⁻¹⁵ (vega) — a derivative-consistency check at fixed σ̂, not a claim about the total system's sensitivity.

Two qualifications apply throughout: the vol-surface result is same-day cross-sectional
reconstruction conditional on an observed surface, not forecasting, and the branch comparison
is between whole input–encoder configurations, not isolated inputs. Full numbers, hedges, and
methodology are in [`paper/main.tex`](paper/main.tex); the underlying per-run metrics are in
[`results/`](results/) (see `table_T1_v5.md` for the complete architecture × branch grid).

## Repo layout

```
train_model_v3/     training scripts (call/ and put/ mirrors) + per-run results
                     (config.json, best_model.pth, loss curves) for each
                     branch × head-architecture combination
wrds_data_2020-2025/ preprocessing scripts + dataset manifest/validation reports
                     (the raw OptionMetrics/WRDS data itself is not in this repo)
analysis/            CPU-only evaluation, diagnostics, and figure-generation scripts
results/             aggregated metrics, figures, and tables backing the paper
paper/               the manuscript (LaTeX) and its figures
tests/               correctness checks for the pricer, provenance, and eval pipeline
```

Only the canonical `best_model.pth` checkpoint (the one used for evaluation) ships per run;
intermediate phase-1/final checkpoints are reproducible from the training scripts and each
run's `config.json` + seed. Raw SPX options data requires a WRDS/OptionMetrics subscription —
see `wrds_data_2020-2025/README.txt` for the exact tables and build steps.

## Setup

```bash
conda create -n dl_new python=3.11
conda activate dl_new
pip install torch polars numpy h5py matplotlib
```

There is no pinned `requirements.txt`/`environment.yml` in this repo yet — the above is the
dependency set the code actually imports. Every result in the paper was produced under Python
3.11.15 with torch 2.10.0 (built against CUDA 13.0), numpy 2.3.5, h5py 3.16.0, matplotlib 3.10.8,
and polars 1.38.1; the full table, including GPU and driver versions, is in the reproducibility and
environment appendix of [`paper/main.tex`](paper/main.tex).

## Quickstart

```bash
# Build the dataset from raw WRDS exports (needs wrds_data_2020-2025/*.csv, not included)
python wrds_data_2020-2025/pre_process_1_data_v4.py --chunk-months 3
python wrds_data_2020-2025/pre_process_2_hdf5_v4.py --type call
python wrds_data_2020-2025/pre_process_2_hdf5_v4.py --type put

# Train the vol-surface branch (MLP head)
python train_model_v3/call/train_vol_surface.py

# Evaluate an existing checkpoint only (loads best_model.pth, writes nothing)
python train_model_v3/call/train_vol_surface.py --eval-only
```

Every `train_*.py` script under `train_model_v3/{call,put}/` is self-contained and follows the
same pattern for the other branches (`train_spot_history.py`, `train_vix_history.py`) and the
DeepONet head variant (`*_don.py`).

## Status

Research code accompanying a manuscript in preparation. Not currently licensed for reuse —
open an issue or contact the author if you'd like to use any of this.

## Citation

```
Zefang Li. "Learning Implied Volatility, Not Prices: A Neural-Operator Encoder with a
Differentiable Black-Scholes Decoder for SPX Option Pricing." Manuscript in preparation.
```
