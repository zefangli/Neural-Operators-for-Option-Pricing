WRDS:
product: optionm_all 
data: Get Data/OptionMetrics/Ivy DB US/Options
 - Option Prices
 - Volatility Sruface

using:
optionPrice
volatilitySurface
interestRate
spx_dividendYield
spx_price

Security: 108105 SPX
European call options

Range: 2024-08-29 - 2025-08-29


pre-processing 1: pre_process_1_data.py
pre-processed data file: deeponet_training_data.parquet
pre-processing 2: pre_process_2_hdf5.py
pre-processed data file: deeponet_tensors.h5



--------------------------------------------------------
citation: 
Wharton Research Data Services. "WRDS" wrds.wharton.upenn.edu, accessed 2026-02-15. 