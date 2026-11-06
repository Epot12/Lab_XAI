# Explainable Artificial Intelligence (XAI) Project Work

This project focuses on the application of Explainable Artificial Intelligence (XAI) techniques in the field of space observation. Specifically, the observations considered in this project are acquired by the MARSIS radar onboard the European Space Agency’s (ESA) Mars Express spacecraft, a mission dedicated to the exploration and analysis of Mars, with the aim of identifying conditions that may be suitable for human life, such as the presence of water.

To this end, the radar emits electromagnetic waves that are reflected by the Martian surface. By analyzing these reflected signals, it is possible to gain insights into the composition of the planet’s surface and subsurface.

## Repository Structure
- **Lab_XAI_REPORT.pdf:** project report.
- **Lab_XAI_pytorch:** the folder contains the main script (which includes the MLP experiments), flux_pred_main.ipynb (solar flux index prediction), and ebms_main.ipynb (experiments with Explainable Boosting Machines, or EBMs), along with their corresponding utility modules.
- **Lab_XAI_tensorflow:** contains the original TensorFlow/Keras code, intended to be run in Google Colab.

## Requirements
The project uses uv as its Python environment and package manager. 
## 1. Clone the Repository


First, create a local copy of the project by cloning the repository, then navigate to the main project directory.


Open the terminal and enter the following command:
```bash
git clone https://github.com/Epot12/Lab_XAI.git
cd Lab_XAI/Lab_XAI_pytorch
```

---

## 2. Installing `uv` and setting up the environment
Make sure you are inside the Lab_XAI_pytorch directory. Select the appropriate installation command based on your operating system:

### macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (using PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Synchronizing dependencies

Once `uv` has been installed, recreate the virtual environment with the exact package versions required by the project. Make sure you are still inside the `Lab_XAI_pytorch` directory, then run:

```bash
uv sync
```

This command will automatically read the `uv.lock` file included in the repository, create an isolated virtual environment (in the `.venv` directory), and install all the required dependencies with their exact specified versions.

---

## 3. Downloading the dataset

To obtain the dataset, download the data using the link provided in `Data/MARSIS_historical_dataset.txt`. Then, place the downloaded CSV file inside the `Data` directory.

## 4. Running the notebooks

Once the dependency synchronization is complete, the Jupyter notebooks can be executed. 

When using `uv`, there is no need to manually activate the virtual environment. The package manager automatically launches Jupyter using the correct dependencies. The following command is cross-platform and works seamlessly from the terminal on **Windows**, **Linux**, and **macOS**:

```bash
uv run jupyter notebook notebook_name.ipynb
```



The command will automatically open a Jupyter interface in your default web browser. From there, you can navigate through the project files, open `.ipynb` notebooks, and run the analyses or train the models.
