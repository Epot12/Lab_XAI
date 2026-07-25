import pandas as pd
import os

def generate_lagged_dataset(
    input_csv_path: str,
    output_csv_path: str,
    time_col: str = 'FM_data_ephemeris_time', 
    feature_col: str = 'FM_data_F10_7_index',
    lag_months: int = 1,
    drop_na: bool = True,
    csv_sep: str = ';' 
) -> pd.DataFrame:
    
    print(f"=== PREPARING DATASET WITH LAG ({feature_col} of -{lag_months} Month/s) ===")

    if os.path.exists(output_csv_path):
        print(f"File {output_csv_path} already exists. Loading...")
        return pd.read_csv(output_csv_path, sep=csv_sep)

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Error: File {input_csv_path} does not exist.")
        
    df = pd.read_csv(input_csv_path, sep=csv_sep)
    df.columns = df.columns.str.strip() 

    if time_col not in df.columns:
        raise KeyError(f"Impossible finding '{time_col}'.")

    # Copying
    df_feature = df[[time_col, feature_col]].copy()
    
    # Fixing time
    if pd.api.types.is_numeric_dtype(df[time_col]):
        print("Found numeric time (Ephemeris Time in seconds). Calculating mathematical offset...")
        # 1 avg month = 30,4368 days = 2.629.740 secs
        offset_seconds = lag_months * 30.4368 * 24 * 60 * 60
        df_feature[time_col] = df_feature[time_col] + offset_seconds
    else:
        print("Textual time detected. Converting to Datetime....")
        df[time_col] = pd.to_datetime(df[time_col])
        df_feature[time_col] = pd.to_datetime(df_feature[time_col]) + pd.DateOffset(months=lag_months)

    # chronological order 
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df_feature = df_feature.sort_values(by=time_col).reset_index(drop=True)

    lagged_col_name = f'{feature_col}_lagged'
    df_feature = df_feature.rename(columns={feature_col: lagged_col_name})

    # Asynchronous Join (searches for the nearest historical measurement)
    df_merged = pd.merge_asof(
        df,
        df_feature,
        on=time_col,
        direction='backward'
    )

    df_merged[feature_col] = df_merged[lagged_col_name]
    df_merged = df_merged.drop(columns=[lagged_col_name])

    # cleaning
    if drop_na:
        df_merged = df_merged.dropna(subset=[feature_col])
        print(f"Rows without history removed. Final dataset size: {df_merged.shape}")

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df_merged.to_csv(output_csv_path, sep=csv_sep, index=False)
    print(f"New dataset successfully saved in: {output_csv_path}\n")
    
    return df_merged