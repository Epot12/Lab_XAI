import shap
import torch
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

def explain_with_shap(model, weights_path, scaler_path, X_background_raw, X_test_raw, bg_limit=500, test_limit=None):
    """
    Performs SHAP analysis on a trained PyTorch model and generates the summary plot.
    
    Parameters:
    - model: The PyTorch model class instance (e.g., NeurNet(input_dim=...))
    - weights_path: Path to the model weights file (e.g. 'exp_2/model_fold_0.pt')
    - scaler_path: Path to the scaler saved for that fold (e.g. 'scaler_chrono_0.save')
    - X_background_raw: Numpy array of training data (used to define the SHAP background)
    - X_test_raw: Numpy array of data to be explained (e.g., the test set or an outlier sample)
    - bg_limit: Maximum number of background samples to use 
    - test_limit: Maximum number of test samples to explain (None to explain them all)
    
    Returns:
    - shap_values: The calculated SHAP values ​​(useful for making other graphs later)
    - X_test_scaled: Normalized test data
    """
    print(f"\n=== STARTING SHAP PIPELINE ===")
    print(f"Model: {weights_path} | Scaler: {scaler_path}")
    
    # Setup Device
    device = torch.device("cpu")
    print(f"Forced device for SHAP stability: {device}")

    # Sampling data
    if bg_limit and len(X_background_raw) > bg_limit:
        # Takes a random sample for the background if the data is too large
        idx_bg = np.random.choice(len(X_background_raw), bg_limit, replace=False)
        X_bg_sampled = X_background_raw[idx_bg]
    else:
        X_bg_sampled = X_background_raw

    if test_limit and len(X_test_raw) > test_limit:
        X_test_sampled = X_test_raw[:test_limit]
    else:
        X_test_sampled = X_test_raw

    # Loading Scaler and normalization
    try:
        scaler = load(scaler_path)
    except FileNotFoundError:
        raise Exception(f"Error: Scaler not found in {scaler_path}")
        
    X_bg_scaled = scaler.transform(X_bg_sampled)
    X_test_scaled = scaler.transform(X_test_sampled)

    # Converting in PyTorch tensors
    tensor_bg = torch.tensor(X_bg_scaled, dtype=torch.float32).to(device)
    tensor_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Loading model weights
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        raise Exception(f"Error: Model not found in {weights_path}")
        
    model.to(device)
    model.eval()  # IMPORTANT

    print(f"SHAP calculation (Background: {len(X_bg_sampled)} samples, Test: {len(X_test_sampled)} samples)...")
    
    # Starting Explainer and Calculation
    explainer = shap.DeepExplainer(model, tensor_bg)
    shap_values = explainer.shap_values(tensor_test, check_additivity=False)

    # SHAP for PyTorch sometimes returns a list (one per class), in regressors we take the first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    print("SHAP calculation completed.")
    
    return shap_values, X_test_scaled, explainer



def extract_feature_names(dataset_path="Data/MARSIS_historical_dataset.csv", keep_flux=False):
    """
    Quickly extracts feature names for SHAP analysis by reading only the file header.
    
    Parameters:
    - dataset_path: path to the original CSV file.
    - keep_flux: If True, it keeps the 'FM_data_F10_7_index' variable among the names. If False, it excludes it.
    
    Returns:
    - A list of strings containing the exact names of the features seen by the model.
    """
    # reading only the first line of the original file to be quick
    df_temp = pd.read_csv(dataset_path, sep=";", nrows=1)

    # adding the two fake features for engineering
    df_temp['FM_data_solar_longitude_cos'] = 0.0
    df_temp['FM_data_solar_longitude_sin'] = 0.0

    # defining the columns that should always be deleted
    columns_to_delete = [
        'FM_data_ephemeris_time', 'FM_data_frequency',
        'FM_data_median_corrected_echo_power', 'FM_data_orbit_number',
        'FM_data_peak_corrected_echo_power', 'FM_data_peak_distorted_echo_power',
        'FM_data_peak_simulated_echo_power', 'FM_data_solar_longitude'
    ]

    # Logic for solar flux: if we do not want to keep it, we add it to the delete list
    if not keep_flux:
        columns_to_delete.append('FM_data_F10_7_index')

    # dropping and saving the list of remaining names
    feature_names = df_temp.drop(columns=columns_to_delete, errors='ignore').columns.tolist()

    return feature_names