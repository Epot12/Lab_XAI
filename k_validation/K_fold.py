
import time
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Importing utils
from utils.train_pytorch_model import *


def k_fold_val(X, y, model_class, n_splits=10, epochs=1000, patience=10, exp_name="exp_1"):
    input_dim = X.shape[1]

    # setting KFold
    kf = KFold(n_splits=n_splits, shuffle=False)

    fold = -1
    y_all_pred = np.zeros(y.shape)
    all_histories = []
    f = open(f"MAE_{exp_name}.txt", "w")
    t_start_global = time.time()

    for train_index, test_index in kf.split(X):
        fold += 1
        t_fold_start = time.time()
        print(f"\n--- Starting FOLD {fold} ---")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        f.write('TRAIN: ' + str(train_index) + '\n')
        f.write('TEST: ' + str(test_index) + '\n')
        dump(scaler, f'scaler_chrono_{exp_name}_{fold}.save')

        # training neural network
        model, device, history = train_pytorch_model(X_train, y_train, input_dim, model_class,
            epochs=epochs, patience=patience)
        
        all_histories.append(history)

        # Saving the model (Standard PyTorch .pth format)
        torch.save(model.state_dict(), f'model_{exp_name}_{fold}.pth')

        # Overall evaluation on the fold
        model.eval()
        criterion = torch.nn.MSELoss()
        
        # Helper function for block inference and not saturating VRAM
        def predict_in_batches(X_data, batch_size=2048):
            preds = []
            # dividing the data into blocks (batches)
            for i in range(0, len(X_data), batch_size):
                # moving only one small block at a time onto the GPU
                X_chunk = torch.tensor(X_data[i:i+batch_size], dtype=torch.float32).to(device)
                with torch.no_grad():
                    chunk_pred = model(X_chunk)
                # reporting immediately the result on the CPU and free up the GPU
                preds.append(chunk_pred.cpu())
            
            # concatenating all the calculated blocks
            return torch.cat(preds, dim=0)

        with torch.no_grad():
            # calculating predictions using safe blocks
            train_pred_cpu = predict_in_batches(X_train)
            test_pred_cpu = predict_in_batches(X_test)
            
            # bringing targets onto the GPU using tensor format
            y_train_cpu = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test_cpu = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            # calculating the MSE directly on the CPU
            train_mse = criterion(train_pred_cpu, y_train_cpu).item()
            test_mse = criterion(test_pred_cpu, y_test_cpu).item()

            # Converts predictions to NumPy format for sklearn
            y_pred_np = test_pred_cpu.numpy().flatten()


