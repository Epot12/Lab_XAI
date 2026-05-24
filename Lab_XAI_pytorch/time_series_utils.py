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

