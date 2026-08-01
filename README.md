# Project work di Explainable Artificial Intelligence (XAI)

Questo Project Work si concentra sull'utilizzo di tecniche di Explainable Artificial Intelligence (XAI) nel contesto delle Space Observations. Nello specifico, le osservazioni considerate in questo progetto provengono dal radar MARSIS installato sulla sonda della missione ESA (European Space Agency) Mars Express, volta all'analisi di Marte, con lo scopo di individuare elementi adatti alla vita umana, come la presenza di acqua. A questo proposito, il radar viene usato per l'invio di onde elettromagnetiche che, una volta riflesse dalla superficie marziana, se analizzate permettano di comprenderne la composizione. 

## Organizzazione del repository
- **Lab_XAI_REPORT.pdf:** relazione sul progetto.
- **Lab_XAI_pytorch:** cartella contenente main (dove si trovano gli esperimenti sull'MLP), flux_pred_main.ipynb (predizione indice del flusso solare) e ebms_main.ipynb (esperimenti sulle EBM), oltre alle relative utils.
- **Lab_XAI_tensorflow:** contenente il codice originale Tensorflow/Keras.

## Requirements
Il progetto sfrutta uv come environment manager. 
## 1. Clonare il repository

Per prima cosa, scaricare una copia locale del progetto clonando il repository e accedere alla cartella di lavoro principale. 

Aprire il terminale e digitare:

```bash
git clone https://github.com/Epot12/Lab_XAI.git
cd Lab_XAI/Lab_XAI_pytorch
```

---

## 2. Installazione di `uv` e configurazione dell'ambiente
Assicurarsi di trovarsi all'interno della cartella `Lab_XAI_pytorch`. Scegliere il comando di installazione in base al sistema operativo:

### macOS e Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (tramite PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Sincronizzazione delle dipendenze
Una volta installato `uv`, ricreare l'ambiente virtuale con le librerie esatte richieste dal progetto. Assicurandosi di essere ancora dentro la cartella `Lab_XAI_pytorch`, eseguire:

```bash
uv sync
```
Questo comando leggerà automaticamente il file `uv.lock` presente nel repository, creerà un ambiente virtuale isolato (nella cartella `.venv`) e installerà tutte le dipendenze con le versioni corrette.

---

## 3. Scaricare i dati
Per ottenere il dataset, scaricare i dati dal link presente nel file Data/MARSIS_historical_dataset.txt e inserire il file csv così ottenuto all'interno della cartella Data.

## 4. Eseguire i Notebook

Una volta completata la sincronizzazione, è possibile eseguire i notebook Jupyter. 

Utilizzando `uv`, non è necessario attivare manualmente l'ambiente virtuale; il gestore si occuperà di lanciare Jupyter utilizzando le librerie corrette. Il comando seguente è universale e funziona perfettamente da terminale su **Windows**, **Linux** e **macOS**:

```bash
uv run jupyter notebook nome_notebook.ipynb
```

*(Nota: se si preferisce l'interfaccia più moderna di JupyterLab, è possibile eseguire invece `uv run jupyter lab nome_notebook.ipynb`).*

Il comando aprirà automaticamente un pannello nel browser web predefinito. Da lì è possibile navigare nei file del progetto, aprire i file con estensione `.ipynb` ed eseguire l'analisi o l'addestramento dei modelli.
