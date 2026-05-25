import numpy as np

def create_dataset_windows(data, target_col, time_col, lookback=12):
    """
    Crea finestre in cui X contiene il flusso e il mese dei passati 'lookback' mesi,
    e y contiene il flusso del mese successivo.
    """
    X_flux, X_time, y = [], [], []
    for i in range(len(data) - lookback):
        # Finestra storica dei passati W mesi fino al mese corrente
        X_flux.append(data[target_col].iloc[i:(i + lookback)].values)
        X_time.append(data[time_col].iloc[i:(i + lookback)].values)
        # Target: mese successivo
        y.append(data[target_col].iloc[i + lookback])
        
    return np.array(X_flux), np.array(X_time), np.array(y)

import os
import torch
import pickle
from datetime import datetime

def save_lstm_and_scalers(model, scalers_dict, save_dir="saved_lstm", model_name="lstm_model.pth", scaler_name="lstm_scalers.pkl"):
    """
    Salva i pesi di un modello PyTorch (LSTM) e i relativi scaler su disco.
    
    Parametri:
    - model: Il modello PyTorch addestrato.
    - scalers_dict (dict): Dizionario contenente gli scaler da salvare.
    - save_dir (str): Il percorso della cartella dove salvare i file.
    - model_name (str): Il nome del file per i pesi del modello.
    - scaler_name (str): Il nome del file per il salvataggio degli scaler.
    """
    # 1. Creazione della cartella (se non esiste)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_base, model_ext = os.path.splitext(model_name)
    scaler_base, scaler_ext = os.path.splitext(scaler_name)
    
    model_name_ts = f"{model_base}_{timestamp}{model_ext}"
    scaler_name_ts = f"{scaler_base}_{timestamp}{scaler_ext}"

    model_path = os.path.join(save_dir, model_name_ts)
    scaler_path = os.path.join(save_dir, scaler_name_ts)
    # 2. Salvataggio dei pesi del modello
    torch.save(model.state_dict(), model_path)
    
    # 3. Salvataggio degli scaler tramite pickle
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers_dict, f)
        
    print(f"✅ Operazione completata con successo!")
    print(f"   - Pesi modello salvati in: {model_path}")
    print(f"   - Scaler salvati in:       {scaler_path}")

