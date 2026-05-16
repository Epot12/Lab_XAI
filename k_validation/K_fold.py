import os
import time
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Importing utils
from utils.train_pytorch_model import *


def k_fold_val(X, y, model_class, save_dir, n_splits=10, epochs=1000, patience=10, batch_size=32, exp_name="exp_1"):
    input_dim = X.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=False)

    fold = -1
    y_all_pred = np.zeros(y.shape)
    all_histories = []
    
    # saving logs in experiment folder
    f = open(os.path.join(save_dir, f"MAE_{exp_name}.txt"), "w")
    t_start_global = time.time()

    for train_index, test_index in kf.split(X):
        fold += 1
        t_fold_start = time.time()
        print(f"\n--- Starting FOLD {fold} ---")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        f.write('TRAIN: ' + str(train_index) + '\n')
        f.write('TEST: ' + str(test_index) + '\n')
        
        # saving scaler in the folder
        dump(scaler, os.path.join(save_dir, f'scaler_chrono_{exp_name}_{fold}.save'))

        model, device, history = train_pytorch_model(X_train, y_train, input_dim, model_class,
            epochs=epochs, patience=patience, batch_size=batch_size)
        
        all_histories.append(history)

        # saving model in the folder
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_{exp_name}_{fold}.pth'))

        model.eval()
        criterion = torch.nn.MSELoss()
        
        def predict_in_batches(X_data, eval_batch_size=2048):
            preds = []
            for i in range(0, len(X_data), eval_batch_size):
                X_chunk = torch.tensor(X_data[i:i+eval_batch_size], dtype=torch.float32).to(device)
                with torch.no_grad():
                    chunk_pred = model(X_chunk)
                preds.append(chunk_pred.cpu())
            return torch.cat(preds, dim=0)

        with torch.no_grad():
            train_pred_cpu = predict_in_batches(X_train)
            test_pred_cpu = predict_in_batches(X_test)
            
            y_train_cpu = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test_cpu = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            train_mse = criterion(train_pred_cpu, y_train_cpu).item()
            test_mse = criterion(test_pred_cpu, y_test_cpu).item()
            y_pred_np = test_pred_cpu.numpy().flatten()

        print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))
        t_fold_end = time.time()

        f.write(f'MAE = {mean_absolute_error(y_test, y_pred_np)}\n')
        f.write(f'MAPE = {mean_absolute_percentage_error(y_test, y_pred_np)}\n')
        f.write(f'MSE = {mean_squared_error(y_test, y_pred_np)}\n')
        f.write(f'Execution time for Fold {fold} = {t_fold_end - t_fold_start}\n')

        y_all_pred[test_index] = y_pred_np

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    t_end_global = time.time()
    total_duration = t_end_global - t_start_global

    global_mae = mean_absolute_error(y, y_all_pred)
    global_mape = mean_absolute_percentage_error(y, y_all_pred)
    global_mse = mean_squared_error(y, y_all_pred)

    f.write(f'\nGlobal MAE = {global_mae}')
    f.write(f'\nGlobal MAPE = {global_mape}')
    f.write(f'\nGlobal MSE = {global_mse}')
    f.write(f'\nTotal execution time = {total_duration}')
    f.close()

    # saving global predictions in the folder
    np.savetxt(os.path.join(save_dir, f"y_pred_{exp_name}.txt"), y_all_pred)

    print(f"\nExecution '{exp_name}' successfully completed in {total_duration:.2f}s!")

    return {
        "MAE": global_mae,
        "MAPE": global_mape,
        "MSE": global_mse,
        "predictions": y_all_pred,
        "execution_time": total_duration,
        "histories": all_histories
    }