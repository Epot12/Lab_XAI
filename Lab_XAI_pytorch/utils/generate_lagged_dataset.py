import pandas as pd
import os

def generate_lagged_dataset(
    input_csv_path: str,
    output_csv_path: str,
    time_col: str = 'FM_data_ephemeris_time', # <-- AGGIORNATO
    feature_col: str = 'FM_data_F10_7_index',
    lag_months: int = 1,
    drop_na: bool = True,
    csv_sep: str = ';' # <-- NUOVO PARAMETRO PER IL SEPARATORE
) -> pd.DataFrame:
    """
    Applica un ritardo temporale (lag) a una colonna specifica del dataset.
    """
    print(f"=== PREPARAZIONE DATASET CON LAG ({feature_col} A -{lag_months} MESE/I) ===")

    if os.path.exists(output_csv_path):
        print(f"⚡ Il file {output_csv_path} esiste già.")
        print("Saltando la rigenerazione. Caricamento del dataset esistente in corso...")
        # FIX: Aggiunto sep=csv_sep
        return pd.read_csv(output_csv_path, sep=csv_sep, parse_dates=[time_col])

    # 1. Caricamento del dataset originale
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Errore: Il file {input_csv_path} non esiste.")
        
    # FIX: Aggiunto sep=csv_sep
    df = pd.read_csv(input_csv_path, sep=csv_sep)

    # 2. Conversione in formato datetime e ordinamento cronologico
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # 3. Estrazione della sola colonna target con i relativi timestamp
    df_feature = df[[time_col, feature_col]].copy()

    # 4. Spostiamo avanti il tempo in base ai mesi specificati
    df_feature[time_col] = df_feature[time_col] + pd.DateOffset(months=lag_months)
    lagged_col_name = f'{feature_col}_lagged'
    df_feature = df_feature.rename(columns={feature_col: lagged_col_name})

    df_feature = df_feature.sort_values(by=time_col).reset_index(drop=True)

    # 5. Unione asincrona (merge_asof) per cercare il valore lagato più vicino
    df_merged = pd.merge_asof(
        df,
        df_feature,
        on=time_col,
        direction='backward'
    )

    # 6. Sostituzione della colonna originale con quella ritardata
    df_merged[feature_col] = df_merged[lagged_col_name]
    df_merged = df_merged.drop(columns=[lagged_col_name])

    # 7. Rimozione delle righe senza dati storici (se richiesto)
    if drop_na:
        df_merged = df_merged.dropna(subset=[feature_col])
        print(f"Righe senza storico eliminate. Dimensione finale dataset: {df_merged.shape}")

    # 8. Creazione cartella di destinazione (se non esiste) e salvataggio CSV
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # FIX: Salviamo il nuovo file mantenendo lo stesso separatore originale (;)
    df_merged.to_csv(output_csv_path, sep=csv_sep, index=False)
    print(f"✅ Nuovo dataset salvato con successo in: {output_csv_path}\n")
    
    return df_merged