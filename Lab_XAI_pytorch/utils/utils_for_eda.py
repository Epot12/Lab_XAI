import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_target_distribution(y, save_path=None):
    """
    Advanced statistical analysis of the radar target.
    Evaluates skewness and kurtosis, and tests the log-normality hypothesis.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Chart 1: original distribution
    sns.histplot(y, kde=True, ax=axes[0], color="#2C3E50", bins=100)
    axes[0].set_title(f"Target Distribution (Original)\nSkewness: {stats.skew(y):.2f} | Kurtosis: {stats.kurtosis(y):.2f}", fontsize=12)
    axes[0].set_xlabel("Peak Distorted Echo Power")
    
    # Chart 2: logarithmic scale 
    # Avoid log(0) by adding a small constant if necessary
    y_log = np.log10(y - np.min(y) + 1) if np.min(y) <= 0 else np.log10(y)
    
    sns.histplot(y_log, kde=True, ax=axes[1], color="#16A085", bins=100)
    axes[1].set_title(f"Target Distribution (Log10 Scale)\nSkewness: {stats.skew(y_log):.2f} | Kurtosis: {stats.kurtosis(y_log):.2f}", fontsize=12)
    axes[1].set_xlabel("Log10(Peak Distorted Echo Power)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_sequential_dependency(y, max_lag=50, save_path=None):
    """
    Calcola e plotta l'autocorrelazione del target per dimostrare
    la natura sequenziale/temporale dei dati lungo le orbite.
    """
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
        acf.append(corr)
        
    plt.figure(figsize=(10, 4))
    plt.stem(range(max_lag + 1), acf, basefmt=" ", linefmt="#2980B9", markerfmt="o")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.axhline(1.96 / np.sqrt(len(y)), color="red", linestyle="--", alpha=0.5, label="95% Confidence Interval")
    plt.axhline(-1.96 / np.sqrt(len(y)), color="red", linestyle="--", alpha=0.5)
    
    plt.title("Autocorrelation Function (ACF) of Radar Echo Power", fontsize=13, fontweight='bold')
    plt.xlabel("Lag (Distanza in campioni temporali)")
    plt.ylabel("Autocorrelation Coefficient")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hierarchical_correlation(df_features, save_path=None):
    """
    Genera una heatmap di correlazione accoppiata a un clustering gerarchico (Dendrogramma).
    Identifica gruppi di feature fisiche ridondanti.
    """
    # Calcolo della correlazione di Spearman (gestisce relazioni non lineari monotoniche)
    corr_matrix = df_features.corr(method='spearman')
    
    # Generazione del Clustermap
    g = sns.clustermap(
        corr_matrix, 
        cmap="coolwarm", 
        vmin=-1, vmax=1, 
        annot=True, 
        fmt=".2f",
        figsize=(12, 10),
        cbar_kws={'label': 'Spearman Correlation'},
        annot_kws={"size": 8}
    )
    
    g.ax_heatmap.set_title("Hierarchical Feature Correlation (Spearman)", fontsize=14, fontweight='bold', pad=20)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bivariate_trends(df, feature_x, target_y="TARGET_power", sample_size=5000, save_path=None):
    """
    Plotta la relazione tra una feature e il target usando un campionamento condizionato 
    e un trend non parametrico per evidenziare comportamenti non lineari fisici.
    """
    # Campionamento per non intasare la visualizzazione se il dataset ha milioni di righe
    df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 6))
    # Scatter plot ad alta densità con trasparenza
    sns.scatterplot(data=df_sampled, x=feature_x, y=target_y, alpha=0.3, color="#34495E", edgecolor=None)
    
    # Linea di trend non lineare (regolarizzata tramite una regressione polinomiale locale o GAM locale)
    sns.regplot(data=df_sampled, x=feature_x, y=target_y, scatter=False, color="#E74C3C", order=3, 
                label="Non-linear Trend (Order 3)")
    
    plt.title(f"Physical Interaction: {feature_x} vs Target", fontsize=13, fontweight='bold')
    plt.xlabel(feature_x)
    plt.ylabel("Peak Distorted Echo Power")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



