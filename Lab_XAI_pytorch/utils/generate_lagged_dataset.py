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
    
    print(f"=== PREPARAZIONE DATASET CON LAG ({feature_col} A -{lag_months} MESE/I) ===")

    if os.path.exists(output_csv_path):
        print(f"⚡ Il file {output_csv_path} esiste già. Caricamento in corso...")
        return pd.read_csv(output_csv_path, sep=csv_sep)

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Errore: Il file {input_csv_path} non esiste.")
        
    df = pd.read_csv(input_csv_path, sep=csv_sep)
    df.columns = df.columns.str.strip() # Rimuove spazi invisibili

    if time_col not in df.columns:
        raise KeyError(f"Impossibile trovare '{time_col}'.")

    # 1. Copia isolata della colonna target e del tempo
    df_feature = df[[time_col, feature_col]].copy()
    
    # 2. GESTIONE INTELLIGENTE DEL TEMPO (FIX)
    if pd.api.types.is_numeric_dtype(df[time_col]):
        print("⏱️ Rilevato tempo numerico (Ephemeris Time in secondi). Calcolo offset matematico...")
        # 1 mese medio = 30.4368 giorni = 2.629.740 secondi
        offset_seconds = lag_months * 30.4368 * 24 * 60 * 60
        df_feature[time_col] = df_feature[time_col] + offset_seconds
    else:
        print("📅 Rilevato tempo testuale. Conversione in Datetime in corso...")
        df[time_col] = pd.to_datetime(df[time_col])
        df_feature[time_col] = pd.to_datetime(df_feature[time_col]) + pd.DateOffset(months=lag_months)

    # 3. Ordinamento cronologico (Fondamentale per il merge_asof)
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df_feature = df_feature.sort_values(by=time_col).reset_index(drop=True)

    lagged_col_name = f'{feature_col}_lagged'
    df_feature = df_feature.rename(columns={feature_col: lagged_col_name})

    # 4. Unione Asincrona (cerca la misurazione storica più vicina)
    df_merged = pd.merge_asof(
        df,
        df_feature,
        on=time_col,
        direction='backward'
    )

    df_merged[feature_col] = df_merged[lagged_col_name]
    df_merged = df_merged.drop(columns=[lagged_col_name])

    # 5. Pulizia
    if drop_na:
        df_merged = df_merged.dropna(subset=[feature_col])
        print(f"Righe senza storico eliminate. Dimensione finale dataset: {df_merged.shape}")

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df_merged.to_csv(output_csv_path, sep=csv_sep, index=False)
    print(f"✅ Nuovo dataset salvato con successo in: {output_csv_path}\n")
    
    return df_merged