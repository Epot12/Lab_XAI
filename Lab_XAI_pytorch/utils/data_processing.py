
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
import numpy as np

def parse_generic_log(file_path):
    """
    Rileva automaticamente il formato del file log (TF o PT) 
    e ne estrae i Test MAE finali di ogni fold.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ File non trovato: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    maes = []
    
    # FORMATO DI TIPO 1: PyTorch (MAE = X.XXX)
    if "MAE =" in log_text or "Global MAE" in log_text:
        # Isola i singoli fold prima del riepilogo globale
        folds_raw = log_text.split("Global MAE")[0] if "Global MAE" in log_text else log_text
        maes = [float(x) for x in re.findall(r"MAE\s*=\s*([\d.]+)", folds_raw)]
        print(f"-> Rilevato formato PyTorch. Estratti {len(maes)} valori MAE.")
        
    # FORMATO DI TIPO 2: TensorFlow / Keras (Train: X.X, Test: X.XXX)
    elif "Test:" in log_text:
        maes = [float(x) for x in re.findall(r"Train:\s*[\d.]+,\s*Test:\s*([\d.]+)", log_text)]
        print(f"-> Rilevato formato TensorFlow. Estratti {len(maes)} valori MAE.")
        
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
    Pipeline universale per confrontare i Test MAE di due esperimenti qualsiasi
    (PT vs PT, TF vs TF, PT vs TF) partendo dai file di log testuali.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=== Starting Experiment Comparison Analysis ===")

    # Fase 1: Parsing dei due log
    print(f"Analisi {exp1_label}...")
    maes_exp1 = parse_generic_log(exp1_log_path)
    
    print(f"Analisi {exp2_label}...")
    maes_exp2 = parse_generic_log(exp2_log_path)

    # Validazione dei dati estratti
    num_folds = max(len(maes_exp1), len(maes_exp2))
    if num_folds == 0:
        print("❌ Errore: Impossibile estrarre dati validi da entrambi i file. Verifica i formati.")
        return

    # Se un esperimento ha meno fold dell'altro, pareggia con zeri o taglia per evitare crash nel grafico
    if len(maes_exp1) != len(maes_exp2):
        print(f"⚠️ Attenzione: Numero di fold disallineato ({len(maes_exp1)} vs {len(maes_exp2)}).")
        # Allinea le lunghezze riempiendo di NaN i valori mancanti
        while len(maes_exp1) < num_folds: maes_exp1.append(np.nan)
        while len(maes_exp2) < num_folds: maes_exp2.append(np.nan)

    # -----------------------------------------------------------------
    # GENERAZIONE GRAFICO A BARRE COMPARATIVO
    # -----------------------------------------------------------------
    folds = np.arange(num_folds)
    bar_width = 0.35

    plt.figure(figsize=(12, 5))
    # Colore personalizzato per distinguere gli esperimenti
    plt.bar(folds - bar_width/2, maes_exp1, bar_width, label=exp1_label, color='#2C3E50')
    plt.bar(folds + bar_width/2, maes_exp2, bar_width, label=exp2_label, color='#16A085', alpha=0.9)

    plt.xlabel('Fold Index', fontsize=12)
    plt.ylabel('Test MAE', fontsize=12)
    plt.title(f'Performance Cross-Validation: {exp1_label} vs {exp2_label}', fontsize=14, fontweight='bold')
    plt.xticks(folds, [f"Fold {i}" for i in folds])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    date_str = datetime.now().strftime("%Y%m%d")
    # Creiamo il nome base pulito (senza estensione)
    filename_base = f"comparison_{date_str}_{exp1_label}_vs_{exp2_label}".replace(" ", "_").lower()
    
    # Costruiamo i due percorsi separati
    chart_path_png = os.path.join(output_dir, f"{filename_base}.png")
    chart_path_pdf = os.path.join(output_dir, f"{filename_base}.pdf")
    
    # Salviamo in PNG e poi in PDF
    plt.savefig(chart_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(chart_path_pdf, format='pdf', bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
            
    print(f"📊 Grafici salvati in:\n   - {chart_path_png}\n   - {chart_path_pdf}")
    print("=== Processo di Confronto Completato ===")

import io
import base64
from IPython.display import display, HTML

def fig_to_html_element(fig):
    """
    Salva la figura in memoria, la converte in stringa Base64 
    e restituisce un blocco div HTML isolato.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Chiude la figura per evitare che si duplichi sotto
    return f'<div style="flex: 1; min-width: 45%; padding: 5px;"><img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto;"/></div>'