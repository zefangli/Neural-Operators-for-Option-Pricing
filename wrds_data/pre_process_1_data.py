import polars as pl

def build_dataset():
    # 1. SETUP LAZY SCANNERS
    q_spot = pl.scan_csv('spx_price.csv').select([
        pl.col('date').str.to_date("%Y-%m-%d"),
        pl.col('close').alias('spot')
    ])

    q_div = pl.scan_csv('spx_dividendYield.csv').select([
        pl.col('date').str.to_date("%Y-%m-%d"),
        pl.col('expiration').str.to_date("%Y-%m-%d").alias('exdate'),
        (pl.col('rate') / 100.0).alias('q') 
    ])

    q_rate = pl.scan_csv('interestRate.csv').select([
        pl.col('date').str.to_date("%Y-%m-%d"),
        pl.col('days'),
        (pl.col('rate') / 100.0).alias('r') 
    ])

    q_vol = pl.scan_csv('volatilitySurface.csv').select([
        pl.col('date').str.to_date("%Y-%m-%d"),
        pl.col('days'),
        pl.col('delta'),
        pl.col('impl_volatility')
    ]).sort(['date', 'days', 'delta']).group_by('date').agg([
        pl.col('impl_volatility').alias('vol_surface_vector') 
    ])

    # OPTION PRICE SCANNER (Greeks commented out)
    q_opt = pl.scan_csv('optionPrice.csv').select([
        pl.col('date').str.to_date("%Y-%m-%d"),
        pl.col('exdate').str.to_date("%Y-%m-%d"),
        pl.col('cp_flag'),
        pl.col('strike_price'),
        pl.col('best_bid'),
        pl.col('best_offer'),
        pl.col('open_interest'),
        # pl.col('delta').alias('market_delta'),
        # pl.col('gamma').alias('market_gamma'),
        # pl.col('vega').alias('market_vega')
    ])

    # 2. FILTER & CALCULATE
    q_opt = q_opt.filter(
        (pl.col('cp_flag') == 'C') &
        (pl.col('open_interest') > 100) &
        (pl.col('best_bid') > 0)
    )

    q_opt = q_opt.with_columns([
        (pl.col('strike_price') / 1000.0).alias('strike'),
        ((pl.col('best_bid') + pl.col('best_offer')) / 2.0).alias('mid_price'),
        (pl.col('exdate') - pl.col('date')).dt.total_days().cast(pl.Int32).alias('days_to_expiry')
    ])

    # 3. ALIGN AND MERGE
    q_opt = q_opt.join(q_spot, on='date', how='inner')

    # Using simple Moneyness (S/K) instead of Log-Moneyness
    q_opt = q_opt.with_columns([
        (pl.col('spot') / pl.col('strike')).alias('moneyness'),
        (pl.col('days_to_expiry') / 365.0).alias('T_years'),
        (pl.col('mid_price') / pl.col('strike')).alias('normalized_price') 
    ])

    q_opt = q_opt.join(q_div, on=['date', 'exdate'], how='left')
    q_opt = q_opt.with_columns(pl.col('q').fill_null(0.0))

    q_opt = q_opt.sort(['date', 'days_to_expiry'])
    q_rate = q_rate.sort(['date', 'days'])
    
    q_opt = q_opt.join_asof(
        q_rate, left_on='days_to_expiry', right_on='days', by='date', strategy='nearest'
    )

    q_opt = q_opt.join(q_vol, on='date', how='inner')
    q_opt = q_opt.drop(['strike_price', 'best_bid', 'best_offer', 'days'])

    # 4. EXECUTE & SAVE
    print("Executing computational graph...")
    df_final = q_opt.collect()
    print(f"Dataset compiled! Rows remaining: {df_final.height}")
    df_final.write_parquet('deeponet_training_data.parquet')
    print("Saved to deeponet_training_data.parquet")

if __name__ == "__main__":
    build_dataset()