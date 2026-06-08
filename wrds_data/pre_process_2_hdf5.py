import polars as pl
import numpy as np
import h5py

def build_hdf5_dataset():
    print("Loading Parquet file...")
    df = pl.read_parquet('deeponet_training_data.parquet')
    
    #  drop_nulls checking 'moneyness' 
    df = df.drop_nulls(subset=['moneyness', 'T_years', 'r', 'q', 'mid_price', 'vol_surface_vector'])
    
    N = df.height
    print(f"Total valid samples (N): {N}")

    # --- Core Inputs & Targets ---
    U = np.vstack(df['vol_surface_vector'].to_list()).astype(np.float32)
    # UPDATED: Trunk input now uses 'moneyness'
    Y = df.select(['moneyness', 'T_years', 'r', 'q']).to_numpy().astype(np.float32)
    V = df.select(['mid_price']).to_numpy().astype(np.float32)
    V_norm = df.select(['normalized_price']).to_numpy().astype(np.float32)

    # --- Process Greeks and Masks (COMMENTED OUT) ---
    # print("Processing Greeks and generating masks...")
    
    # Extract raw Greek columns
    # delta_raw = df['market_delta'].to_numpy().astype(np.float32).reshape(-1, 1)
    # gamma_raw = df['market_gamma'].to_numpy().astype(np.float32).reshape(-1, 1)
    # vega_raw = df['market_vega'].to_numpy().astype(np.float32).reshape(-1, 1)

    # Create Masks (1.0 if valid, 0.0 if NaN)
    # mask_delta = (~np.isnan(delta_raw)).astype(np.float32)
    # mask_gamma = (~np.isnan(gamma_raw)).astype(np.float32)
    # mask_vega  = (~np.isnan(vega_raw)).astype(np.float32)

    # Fill NaNs with 0.0 to prevent PyTorch NaN gradients
    # delta_clean = np.nan_to_num(delta_raw, nan=0.0)
    # gamma_clean = np.nan_to_num(gamma_raw, nan=0.0)
    # vega_clean  = np.nan_to_num(vega_raw, nan=0.0)

    # --- Save to HDF5 ---
    print("Saving to deeponet_tensors.h5...")
    with h5py.File('deeponet_tensors.h5', 'w') as f:
        # Core data
        f.create_dataset('branch_u', data=U)
        f.create_dataset('trunk_y', data=Y)
        f.create_dataset('target_v', data=V)
        f.create_dataset('target_v_normalized', data=V_norm)
        
        # Greeks Data (COMMENTED OUT)
        # f.create_dataset('market_delta', data=delta_clean)
        # f.create_dataset('market_gamma', data=gamma_clean)
        # f.create_dataset('market_vega',  data=vega_clean)
        
        # Greeks Masks (COMMENTED OUT)
        # f.create_dataset('mask_delta', data=mask_delta)
        # f.create_dataset('mask_gamma', data=mask_gamma)
        # f.create_dataset('mask_vega',  data=mask_vega)
        
    print("HDF5 file successfully created!")

if __name__ == "__main__":
    build_hdf5_dataset()