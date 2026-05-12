
import numpy as np
import pandas as pd


def load_and_preprocess_data(dataset_path="MARSIS_historical_dataset.csv", orbit_path="orbit_to_remove"):
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