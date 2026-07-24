
import numpy as np
import pandas as pd
from datetime import datetime


def load_and_preprocess_data(dataset_path="MARSIS_historical_dataset.csv", orbit_path="orbit_to_remove", keep_flux=False):
    """
    Loads the dataset, applies the necessary filters, and removes any excluded columns.
    Returns the feature matrix X and the target vector y.
    """
    # Reading the orbits to be removed
    orbit_to_remove = []
    with open(orbit_path) as file:
        for line in file:
            orbit_to_remove.append(float(line))

    # Loading dataset
    df = pd.read_csv(dataset_path, sep=";")

    # Frequency and orbits filtering
    frequency_to_keep = 4000000.0
    df = df[df['FM_data_frequency'] == frequency_to_keep]
    df = df[~df.FM_data_orbit_number.isin(orbit_to_remove)]

    # Feature engineering
    df['FM_data_solar_longitude_cos'] = np.cos(df['FM_data_solar_longitude'])
    df['FM_data_solar_longitude_sin'] = np.sin(df['FM_data_solar_longitude'])

    # Eliminating unnecessary columns
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

    # Extracting to NumPy array format
    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy()
    y = df['FM_data_peak_distorted_echo_power'].to_numpy()

    return X, y, feature_names

import os
import re
import matplotlib.pyplot as plt

def parse_generic_log(file_path):
    """
    It automatically detects the log file format (TF or PT) and extracts the final MAE test scores for each fold.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    maes = []
    
    # FORMAT 1: PyTorch (MAE)
    if "MAE =" in log_text or "Global MAE" in log_text:
        # Isolates the individual folds before the overall summary.
        folds_raw = log_text.split("Global MAE")[0] if "Global MAE" in log_text else log_text
        maes = [float(x) for x in re.findall(r"MAE\s*=\s*([\d.]+)", folds_raw)]
        print(f"-> PyTorch format detected. {len(maes)} MAE values extracted.")
        
    # FORMAT 2: TensorFlow / Keras (Train, Test)
    elif "Test:" in log_text:
        maes = [float(x) for x in re.findall(r"Train:\s*[\d.]+,\s*Test:\s*([\d.]+)", log_text)]
        print(f"-> TensorFlow format detected. {len(maes)} MAE values extracted.")
        
    return maes



def compare_experiments_pipeline(
    exp1_log_path, 
    exp2_log_path, 
    exp1_label="Esperimento 1",
    exp2_label="Esperimento 2",
    output_dir=".", 
    show_plots=True
):
    """
    Universal pipeline for comparing MAE test results from any two experiments
    (PT vs. PT, TF vs. TF, PT vs. TF) starting from text-based log files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=== Starting Experiment Comparison Analysis ===")

    # log parsing
    print(f"Analysis of {exp1_label}...")
    maes_exp1 = parse_generic_log(exp1_log_path)
    
    print(f"Analysis of {exp2_label}...")
    maes_exp2 = parse_generic_log(exp2_log_path)

    # validation of extracted data
    num_folds = max(len(maes_exp1), len(maes_exp2))
    if num_folds == 0:
        print("Error: Unable to extract valid data from either file. Check the formats.")
        return

    if len(maes_exp1) != len(maes_exp2):
        print(f"Warning: Number of folds mismatched ({len(maes_exp1)} vs {len(maes_exp2)}).")
        # Aligns the lengths by filling missing values ​​with NaN
        while len(maes_exp1) < num_folds: maes_exp1.append(np.nan)
        while len(maes_exp2) < num_folds: maes_exp2.append(np.nan)

    # -----------------------------------------------------------------
    # Generating a comparative bar chart
    # -----------------------------------------------------------------
    folds = np.arange(num_folds)
    bar_width = 0.35

    plt.figure(figsize=(12, 5))
    # Custom color to distinguish experiments
    plt.bar(folds - bar_width/2, maes_exp1, bar_width, label=exp1_label, color='#2C3E50')
    plt.bar(folds + bar_width/2, maes_exp2, bar_width, label=exp2_label, color='#16A085', alpha=0.9)

    plt.xlabel('Fold Index', fontsize=12)
    plt.ylabel('Test MAE', fontsize=12)
    plt.title(f'Performance Cross-Validation: {exp1_label} vs {exp2_label}', fontsize=14, fontweight='bold')
    plt.xticks(folds, [f"Fold {i}" for i in folds])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Creating base name
    filename_base = f"comparison_{date_str}_{exp1_label}_vs_{exp2_label}".replace(" ", "_").lower()
    
    # building the two paths
    chart_path_png = os.path.join(output_dir, f"{filename_base}.png")
    chart_path_pdf = os.path.join(output_dir, f"{filename_base}.pdf")
    
    # saving
    plt.savefig(chart_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(chart_path_pdf, format='pdf', bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
            
    print(f"Plots saved in:\n   - {chart_path_png}\n   - {chart_path_pdf}")
    print("=== Comparison Process Completed ===")

import io
import base64

def fig_to_html_element(fig):
    """
    Saves the image to memory, converts it to a Base64 string, and returns an isolated HTML div block.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Closes the shape to prevent it from duplicating underneath.
    return f'<div style="flex: 1; min-width: 45%; padding: 5px;"><img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto;"/></div>'