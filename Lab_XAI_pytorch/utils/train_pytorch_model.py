import torch
import copy
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def train_pytorch_model(X_train, y_train, input_dim, model_class, epochs=1000, patience=10, batch_size=32, **model_kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_dim, **model_kwargs).to(device)

    # Keras (original code) with validation_split=0.1 takes the last 10% of the training data
    val_split_idx = int(len(X_train) * 0.9)
    X_tr_sub, y_tr_sub = X_train[:val_split_idx], y_train[:val_split_idx]
    X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]

    # Converting to PyTorch Tensors
    X_tr_t = torch.tensor(X_tr_sub, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_sub, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # Creating DataLoader for batch training (Keras uses batch_size=32 by default)
    dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())  # Default LR 0.001

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    history = {'train_loss': [], 'val_loss': []}
    pbar = tqdm(range(epochs), desc="Training", leave=False)

    for epoch in pbar:
        model.train()  # Training Mode 
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_train_loss = np.mean(batch_losses)

        # Validation and Early Stopping
        model.eval()  # Inference Mode 
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        pbar.set_postfix({'Train MSE': f'{epoch_train_loss:.4f}', 'Val MSE': f'{val_loss:.4f}'})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, device, history

import os
import matplotlib.pyplot as plt
from datetime import datetime

def plot_and_save_history(history, save_dir="plots"):
    """
    Generates a learning curve plot (Training and Validation Loss)
    and saves it in high-quality PNG and PDF formats.
    
    Parameters:
    - history: The dictionary returned by the train_pytorch_model function
    - save_dir: The folder where the plots will be saved (created automatically if it does not exist)
    """
    # Creating destination folder
    os.makedirs(save_dir, exist_ok=True)
    
    # Generation of the unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # plotting settings
    plt.figure(figsize=(10, 6), dpi=300)
    
    # lines style
    plt.plot(history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Formatting axes, title, and legend
    plt.title('Learning Curve', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    # Saving
    base_filepath = os.path.join(save_dir, f"training_history_{timestamp}")
    
    # PNG
    png_path = f"{base_filepath}.png"
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    # PDF
    pdf_path = f"{base_filepath}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"\nPlots successfully saved to:")
    print(f"- {png_path}")
    print(f"- {pdf_path}")
    
    plt.show()
    
    plt.close()