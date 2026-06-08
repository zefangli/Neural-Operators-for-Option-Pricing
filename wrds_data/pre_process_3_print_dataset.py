import polars as pl
import numpy as np
import h5py

def inspect_parquet():
    """Inspect the Parquet file (output of File 1)"""
    print("=" * 80)
    print("INSPECTING PARQUET FILE: deeponet_training_data.parquet")
    print("=" * 80)
    
    # Load the parquet file
    df = pl.read_parquet('deeponet_training_data.parquet')
    
    # Basic info
    print(f"\nTotal rows: {df.height}")
    print(f"Total columns: {df.width}")
    print(f"\nColumn names and types:")
    print("-" * 40)
    for col_name, col_type in zip(df.columns, df.dtypes):
        print(f"  {col_name:25} : {col_type}")
    
    # Print first 100 rows (excluding vol_surface_vector for readability)
    print("\n" + "=" * 80)
    print("FIRST 100 ROWS (excluding vol_surface_vector):")
    print("=" * 80)
    
    # Select columns to display (exclude the large vector column)
    display_cols = [col for col in df.columns if col != 'vol_surface_vector']
    
    # Configure Polars to show all columns and rows
    with pl.Config(
        tbl_cols=20,           # Show up to 20 columns
        tbl_rows=100,          # Show up to 100 rows
        tbl_width_chars=200,   # Wide table
        fmt_str_lengths=50     # String length limit
    ):
        print(df.select(display_cols).head(100))
    
    # Show vol_surface_vector info separately
    print("\n" + "=" * 80)
    print("VOL_SURFACE_VECTOR INFO (first 5 rows):")
    print("=" * 80)
    vol_vectors = df['vol_surface_vector'].head(5).to_list()
    for i, vec in enumerate(vol_vectors):
        print(f"Row {i}: length={len(vec)}, first 5 values={vec[:5]}, last 5 values={vec[-5:]}")
    
    # Statistics for key columns
    print("\n" + "=" * 80)
    print("STATISTICS FOR KEY COLUMNS:")
    print("=" * 80)
    stats_cols = ['spot', 'strike', 'moneyness', 'T_years', 'r', 'q', 'mid_price', 'normalized_price']
    available_stats_cols = [col for col in stats_cols if col in df.columns]
    print(df.select(available_stats_cols).describe())
    
    # Check for nulls
    print("\n" + "=" * 80)
    print("NULL VALUE COUNTS:")
    print("=" * 80)
    null_counts = df.null_count()
    print(null_counts)
    
    return df


def inspect_hdf5():
    """Inspect the HDF5 file (output of File 2)"""
    print("\n" + "=" * 80)
    print("INSPECTING HDF5 FILE: deeponet_tensors.h5")
    print("=" * 80)
    
    try:
        with h5py.File('deeponet_tensors.h5', 'r') as f:
            print("\nDatasets in HDF5 file:")
            print("-" * 40)
            for key in f.keys():
                dataset = f[key]
                print(f"  {key:25} : shape={dataset.shape}, dtype={dataset.dtype}")
            
            # Print first 100 rows of trunk_y (moneyness, T_years, r, q)
            print("\n" + "-" * 80)
            print("TRUNK_Y (first 100 rows) - [moneyness, T_years, r, q]:")
            print("-" * 80)
            trunk_y = f['trunk_y'][:100]
            print(f"{'Row':<6} {'Moneyness':<12} {'T_years':<12} {'r':<12} {'q':<12}")
            print("-" * 54)
            for i, row in enumerate(trunk_y):
                print(f"{i:<6} {row[0]:<12.6f} {row[1]:<12.6f} {row[2]:<12.6f} {row[3]:<12.6f}")
            
            # Print first 100 target values
            print("\n" + "-" * 80)
            print("TARGET VALUES (first 100 rows):")
            print("-" * 80)
            target_v = f['target_v'][:100]
            target_v_norm = f['target_v_normalized'][:100]
            print(f"{'Row':<6} {'Mid Price':<15} {'Normalized Price':<15}")
            print("-" * 36)
            for i in range(100):
                print(f"{i:<6} {target_v[i][0]:<15.4f} {target_v_norm[i][0]:<15.6f}")
            
            # Branch input info
            print("\n" + "-" * 80)
            print("BRANCH_U (vol surface) - first 5 rows:")
            print("-" * 80)
            branch_u = f['branch_u'][:5]
            for i, vec in enumerate(branch_u):
                print(f"Row {i}: length={len(vec)}, min={vec.min():.4f}, max={vec.max():.4f}, mean={vec.mean():.4f}")
                print(f"        first 5: {vec[:5]}")
                print(f"        last 5:  {vec[-5:]}")
                
    except FileNotFoundError:
        print("HDF5 file not found. Run File 2 first to generate it.")


def print_aligned_data():
    """Print aligned data showing how r and q were matched"""
    print("\n" + "=" * 80)
    print("ALIGNED DATA CHECK (Interest Rate & Dividend Yield)")
    print("=" * 80)
    
    df = pl.read_parquet('deeponet_training_data.parquet')
    
    # Select relevant columns to show alignment
    alignment_cols = ['date', 'exdate', 'days_to_expiry', 'spot', 'strike', 'moneyness', 'r', 'q', 'mid_price']
    available_cols = [col for col in alignment_cols if col in df.columns]
    
    print("\nFirst 100 rows showing date, expiry, and matched r/q values:")
    print("-" * 100)
    
    with pl.Config(tbl_cols=15, tbl_rows=100, tbl_width_chars=200):
        print(df.select(available_cols).head(100))
    
    # Show unique dates and their rate/yield ranges
    print("\n" + "-" * 80)
    print("SUMMARY BY DATE (first 10 dates):")
    print("-" * 80)
    
    date_summary = df.group_by('date').agg([
        pl.count().alias('num_options'),
        pl.col('r').min().alias('r_min'),
        pl.col('r').max().alias('r_max'),
        pl.col('q').min().alias('q_min'),
        pl.col('q').max().alias('q_max'),
        pl.col('days_to_expiry').min().alias('dte_min'),
        pl.col('days_to_expiry').max().alias('dte_max'),
    ]).sort('date').head(10)
    
    print(date_summary)


if __name__ == "__main__":
    # Run all inspections
    df = inspect_parquet()
    print("\n\n")
    inspect_hdf5()
    print("\n\n")
    print_aligned_data()