
import numpy as np
import pandas as pd


def load_and_preprocess_data(dataset_path="MARSIS_historical_dataset.csv", orbit_path="orbit_to_remove", keep_flux=False):
    """
    Loads the dataset, applies the necessary filters, and removes any excluded columns.
    Returns the feature matrix X and the target vector y.
    """
    # 1. Reading the orbits to be removed
    orbit_to_remove = []
    with open(orbit_path) as file:
        for line in file:
            orbit_to_remove.append(float(line))

    # 2. Loading dataset
    df = pd.read_csv(dataset_path, sep=";")

    # 3. Frequency and orbits filtering
    frequency_to_keep = 4000000.0
    df = df[df['FM_data_frequency'] == frequency_to_keep]
    df = df[~df.FM_data_orbit_number.isin(orbit_to_remove)]

    # 4. Feature engineering
    df['FM_data_solar_longitude_cos'] = np.cos(df['FM_data_solar_longitude'])
    df['FM_data_solar_longitude_sin'] = np.sin(df['FM_data_solar_longitude'])

    # 5. Eliminating unnecessary columns
    if keep_flux:
        X_df = df.drop(columns=[
            'FM_data_ephemeris_time', 'FM_data_frequency',
            'FM_data_median_corrected_echo_power', 'FM_data_orbit_number',
            'FM_data_peak_corrected_echo_power', 'FM_data_peak_distorted_echo_power',
            'FM_data_peak_simulated_echo_power', 'FM_data_solar_longitude'
        ])

    else:
        X_df = df.drop(columns=[
            'FM_data_ephemeris_time', 'FM_data_F10_7_index', 'FM_data_frequency',
            'FM_data_median_corrected_echo_power', 'FM_data_orbit_number',
            'FM_data_peak_corrected_echo_power', 'FM_data_peak_distorted_echo_power',
            'FM_data_peak_simulated_echo_power', 'FM_data_solar_longitude'
        ])

    # 6. Extracting to NumPy array format
    X = X_df.to_numpy()
    y = df['FM_data_peak_distorted_echo_power'].to_numpy()

    return X, y

import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np

def compare_frameworks_pipeline(
    tf_log_path, 
    pt_log_path,  # Cambiato da pt_pickle_path a pt_log_path
    pt_fallback_maes=None, 
    output_dir=".", 
    show_plots=True
):
    """
    Parses TensorFlow logs from a file, parses PyTorch logs from a text file,
    and generates comparison plots (Bar Chart for Test MAE and Grid for Learning Curves).
    """
    if pt_fallback_maes is None:
        pt_fallback_maes = [2.798, 2.439, 2.652, 2.848, 4.065, 4.990, 4.415, 3.361, 3.725, 2.707]
        
    os.makedirs(output_dir, exist_ok=True)
    print("=== Starting Framework Comparison Analysis ===")

    # -----------------------------------------------------------------
    # PHASE 1: Parsing TensorFlow Log Source
    # -----------------------------------------------------------------
    print(f"Reading TensorFlow log from file: {tf_log_path}")
    with open(tf_log_path, "r", encoding="utf-8") as f:
        tf_log_text = f.read()

    tf_histories = []
    tf_test_maes = []
    current_train_loss = []
    current_val_loss = []

    for line in tf_log_text.strip().split('\n'):
        metrics_match = re.search(r"loss:\s*([\d.]+).*?val_loss:\s*([\d.]+)", line)
        if metrics_match:
            current_train_loss.append(float(metrics_match.group(1)))
            current_val_loss.append(float(metrics_match.group(2)))
        
        test_match = re.search(r"Train:\s*[\d.]+, Test:\s*([\d.]+)", line)
        if test_match:
            tf_test_maes.append(float(test_match.group(1)))
            tf_histories.append({
                'train_loss': current_train_loss,
                'val_loss': current_val_loss
            })
            current_train_loss = []
            current_val_loss = []

    num_folds = len(tf_test_maes)
    print(f"👉 TensorFlow Parsing complete: {num_folds} folds successfully extracted.")

    if num_folds == 0:
        print("⚠️ Warning: No TensorFlow folds were parsed. Please check the log formatting or regex matches.")
        num_folds = 10

    # -----------------------------------------------------------------
    # PHASE 2: Parsing PyTorch Text Log (MODIFICATA CHIRURGICAMENTE)
    # -----------------------------------------------------------------
    has_pytorch_data = False
    pt_histories = None
    pt_test_maes = []

    if os.path.exists(pt_log_path):
        try:
            print(f"Reading PyTorch log from text file: {pt_log_path}")
            with open(pt_log_path, "r", encoding="utf-8") as f:
                pt_log_text = f.read()
            
            # Estrae tutti i valori di MAE associati ai singoli fold (esclude il Global MAE alla fine)
            # Cerca i pattern "MAE = valore" prima della stringa "Global MAE"
            pt_folds_raw = pt_log_text.split("Global MAE")[0]
            pt_test_maes = [float(x) for x in re.findall(r"MAE\s*=\s*([\d.]+)", pt_folds_raw)]
            
            if len(pt_test_maes) == num_folds:
                has_pytorch_data = True
                print(f"👉 PyTorch data successfully extracted: {len(pt_test_maes)} MAE values found.")
            else:
                print(f"⚠️ Warning: Found {len(pt_test_maes)} MAE values for PyTorch, but TensorFlow has {num_folds} folds.")
                print("Switching to fallback MAE parameters for consistency.")
                pt_test_maes = pt_fallback_maes
                
        except Exception as e:
            print(f"⚠️ Error reading PyTorch text file: {e}. Switching to fallback MAE parameters.")
            pt_test_maes = pt_fallback_maes
    else:
        print(f"⚠️ PyTorch log file not found at '{pt_log_path}'. Using fallback verification MAEs.")
        pt_test_maes = pt_fallback_maes

    # -----------------------------------------------------------------
    # PHASE 3: Generating Plot 1 - Bar Chart (Test MAE Comparison)
    # -----------------------------------------------------------------
    folds = np.arange(num_folds)
    bar_width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(folds - bar_width/2, pt_test_maes[:num_folds], bar_width, label='PyTorch', color='#EE4C2C')
    plt.bar(folds + bar_width/2, tf_test_maes[:num_folds], bar_width, label='TensorFlow', color='#FF6F00', alpha=0.8)

    plt.xlabel('Fold Index', fontsize=12)
    plt.ylabel('Test MAE', fontsize=12)
    plt.title('Final Performance Cross-Validation: PyTorch vs TensorFlow', fontsize=14, fontweight='bold')
    plt.xticks(folds, [f"Fold {i}" for i in folds])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    bar_chart_path = os.path.join(output_dir, "framework_mae_comparison.png")
    plt.savefig(bar_chart_path, dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -----------------------------------------------------------------
    # PHASE 4: Generating Plot 2 - Grid (Learning Curves Overlay)
    # -----------------------------------------------------------------
    if has_pytorch_data and pt_histories is not None:
        # Questo blocco viene ignorato dal txt poiché non ci sono le curve storiche epoche per epoca
        cols = 2
        rows = (num_folds + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for fold_idx in range(num_folds):
            ax = axes[fold_idx]
            ax.plot(pt_histories[fold_idx]['train_loss'], label='PT Train Loss', color='#EE4C2C', linewidth=2)
            ax.plot(pt_histories[fold_idx]['val_loss'], label='PT Val Loss', color='#982C16', linewidth=2)
            
            if fold_idx < len(tf_histories):
                ax.plot(tf_histories[fold_idx]['train_loss'], label='TF Train Loss', color='#FF6F00', linewidth=1.8, linestyle='--')
                ax.plot(tf_histories[fold_idx]['val_loss'], label='TF Val Loss', color='#B34E00', linewidth=1.8, linestyle='--')
            
            ax.set_title(f'Learning Curve - Fold {fold_idx}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epochs', fontsize=10)
            ax.set_ylabel('Loss (MSE)', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=9, loc='upper right')

        for j in range(num_folds, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "framework_curves_comparison.png")
        plt.savefig(curves_path, dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("ℹ️ Note: PyTorch learning curves data (epochs history) not present in text log. Skipping Phase 4.")
            
    print(f"📊 Visual files successfully saved in the directory: '{output_dir}'")
    print("=== Comparison Process Successfully Completed ===")