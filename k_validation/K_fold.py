
import time
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Importing utils
from utils import train_pytorch_model


def k_fold_val(X, y, model_class, n_splits=10, epochs=1000, patience=10, exp_name="exp_1"):
    input_dim = X.shape[1]

    # setting KFold
    kf = KFold(n_splits=n_splits, shuffle=False)

    fold = -1
    y_all_pred = np.zeros(y.shape)
    f = open(f"MAE_{exp_name}.txt", "w")
    t1 = time.time()

    for train_index, test_index in kf.split(X):
        fold += 1
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
        model, device = train_pytorch_model(X_train, y_train, input_dim, model_class,
            epochs=epochs, patience=patience)

        # Saving the model (Standard PyTorch .pth format)
        torch.save(model.state_dict(), f'model_{exp_name}_{fold}.pth')

        # Overall evaluation on the fold
        model.eval()
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            # Move the complete fold data to the device for final evaluation
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            train_pred = model(X_train_t)
            test_pred = model(X_test_t)

            train_mse = criterion(train_pred, y_train_t).item()
            test_mse = criterion(test_pred, y_test_t).item()

            # Converts predictions to NumPy format for sklearn
            y_pred_np = test_pred.cpu().numpy().flatten()

        print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))
        t2 = time.time()

        # Saving metrics
        f.write(f'MAE = {mean_absolute_error(y_test, y_pred_np)}\n')
        f.write(f'MAPE = {mean_absolute_percentage_error(y_test, y_pred_np)}\n')
        f.write(f'MSE = {mean_squared_error(y_test, y_pred_np)}\n')
        f.write(f'Execution time = {t2 - t1}\n')

        # adding global predictions
        for i in range(len(test_index)):
            y_all_pred[test_index[i]] = y_pred_np[i]

    # global metrics OOF (Out Of Fold)
    t3 = time.time()

    # saving metrics in variables
    global_mae = mean_absolute_error(y, y_all_pred)
    global_mape = mean_absolute_percentage_error(y, y_all_pred)
    global_mse = mean_squared_error(y, y_all_pred)

    # using variables to write in log files
    f.write(f'\nGlobal MAE = {global_mae}')
    f.write(f'\nGlobal MAPE = {global_mape}')
    f.write(f'\nGlobal MSE = {global_mse}')
    f.write(f'\nTime = {t3 - t1}')
    f.close()

    np.savetxt(f"y_pred_{exp_name}.txt", y_all_pred)

    print(f"\nExecution '{exp_name}' successfully completed!")

    return {
        "MAE": global_mae,
        "MAPE": global_mape,
        "MSE": global_mse,
        "predictions": y_all_pred,
        "execution_time": t3 - t1
    }


