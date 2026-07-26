import numpy as np

def create_dataset_windows(data, target_col, time_col, lookback=12):
    """
    Creates windows where X contains the flow and the month for the past 'lookback' months,
    and y contains the flow for the following month.
    """
    X_flux, X_time, y = [], [], []
    for i in range(len(data) - lookback):
        # Historical window covering the past W months up to the current month.
        X_flux.append(data[target_col].iloc[i:(i + lookback)].values)
        X_time.append(data[time_col].iloc[i:(i + lookback)].values)
        # Target: following month
        y.append(data[target_col].iloc[i + lookback])
        
    return np.array(X_flux), np.array(X_time), np.array(y)

import os
import torch
import pickle
from datetime import datetime

def save_lstm_and_scalers(model, scalers_dict, save_dir="saved_lstm", model_name="lstm_model.pth", scaler_name="lstm_scalers.pkl"):
    """
    Saves PyTorch model (LSTM) weights and associated scalers to disk.

    Parameters:
    - model: The trained PyTorch model.
    - scalers_dict (dict): Dictionary containing the scalers to be saved.
    - save_dir (str): Path to the directory where files will be saved.
    - model_name (str): Filename for the model weights.
    - scaler_name (str): Filename for saving the scalers.
    """
    # creation of the folder
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_base, model_ext = os.path.splitext(model_name)
    scaler_base, scaler_ext = os.path.splitext(scaler_name)
    
    model_name_ts = f"{model_base}_{timestamp}{model_ext}"
    scaler_name_ts = f"{scaler_base}_{timestamp}{scaler_ext}"

    model_path = os.path.join(save_dir, model_name_ts)
    scaler_path = os.path.join(save_dir, scaler_name_ts)
    # saving model weights
    torch.save(model.state_dict(), model_path)
    
    # saving scalers
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers_dict, f)
        
    print(f"Operation successfully completed!")
    print(f"   - Model weights saved in: {model_path}")
    print(f"   - Scalers saved in:       {scaler_path}")

