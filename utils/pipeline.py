import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from k_validation import *

def pipeline(X, y, model_class, n_splits=10, epochs=1000, patience=10, batch_size=32, exp_name="exp_1"):
    """
    Esegue l'intero ciclo di vita dell'esperimento: Setup Directory -> Training -> Plotting -> Saving.
    """
    print(f"=== Starting Experiment Pipeline: {exp_name} ===")
    
    # ---------------------------------------------------------
    # FASE 0: Creazione Cartella Esperimento
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Es: experiments/exp_1_20260516_130500
    save_dir = os.path.join("experiments", f"{exp_name}_{timestamp}") 
    plots_dir = os.path.join(save_dir, "Learning_Curves")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Created isolated environment at: {save_dir}")

    # ---------------------------------------------------------
    # FASE 1: Addestramento e Inferenza
    # ---------------------------------------------------------
    k_fold_dict = k_fold_val(
        X=X, y=y, model_class=model_class, save_dir=save_dir, 
        n_splits=n_splits, epochs=epochs, patience=patience, 
        batch_size=batch_size, exp_name=exp_name
    )

    # ---------------------------------------------------------
    # FASE 2: Generazione e Salvataggio Plot
    # ---------------------------------------------------------
    print("Generating Learning Curves...")
    fold_histories = k_fold_dict["histories"]
    num_folds = len(fold_histories)
    cols = 2
    rows = (num_folds + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for target_fold, history in enumerate(fold_histories):
        train_loss = history['train_loss']
        val_loss = history['val_loss']

        # Salvataggio file individuali (Invisibili a schermo)
        fig_save = plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss (MSE)', color='blue', linewidth=2)
        plt.plot(val_loss, label='Validation Loss (MSE)', color='orange', linewidth=2)
        plt.title(f'Learning Curve - Fold {target_fold}', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Salviamo direttamente nella sottocartella dei plot di QUESTO esperimento
        plt.savefig(os.path.join(plots_dir, f"fold_{target_fold}.png"), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, f"fold_{target_fold}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig_save) 

        # Plot sulla griglia del notebook
        ax = axes[target_fold]
        ax.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
        ax.plot(val_loss, label='Validation Loss', color='orange', linewidth=2)
        ax.set_title(f'Learning Curve - Fold {target_fold}', fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    for j in range(num_folds, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # FASE 3: Salvataggio Pickle finale
    # ---------------------------------------------------------
    file_name_pick = os.path.join(save_dir, f"results_dict.pkl")
    with open(file_name_pick, "wb") as f:
        pickle.dump(k_fold_dict, f)

    print(f"Experiment completely saved in: {os.path.abspath(save_dir)}")
    return k_fold_dict, save_dir