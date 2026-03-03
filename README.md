# Lung Cancer Risk Prediction with Deep Learning

Thesis project: deep learning (neural networks) for lung cancer risk prediction using the same survey dataset as the 2022 paper **"Lung Cancer Risk Prediction with Machine Learning"** (PDF in this folder). Data: `data/survey lung cancer.csv`. The 2022 paper reports **97.1% accuracy** and **99.3% AUC** with Rotation Forest (see **[REFERENCE_PAPER.md](REFERENCE_PAPER.md)**). This project uses **PyTorch** only (no TensorFlow).

For a clear map of folders and what to run, see **[STRUCTURE_OVERVIEW.md](STRUCTURE_OVERVIEW.md)**.

## Project structure (short)

```
Lung-Cancer-Risk-Prediction-with-Deep-Learning/
├── data/raw/             # survey lung cancer.csv
├── src/                  # data_preprocessing, model, train, compare_models
├── notebooks/            # notebook.ipynb (uses src/)
├── models/               # Saved .pth (after you run train or compare_models)
├── reports/              # Plots + model_comparison_results.csv
├── REFERENCE_PAPER.md    # 2022 paper baseline: 97.1% accuracy, 99.3% AUC
└── requirements.txt
```

## Installation

1. Create a virtual environment (recommended):
```bash
conda create -n lung_dl_env python=3.12
conda activate lung_dl_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

- **Train one model:** `python src/train.py` → saves to `models/`, plots in `reports/`.
- **Train and compare all architectures:** `python src/compare_models.py` → all models + comparison CSV and plots in `reports/`.
- **5-fold or 10-fold CV:** `python src/compare_models.py --cv 5` or `--cv 10` → results in `reports/cv5/` or `reports/cv10/`.
- **Generate PowerPoint:** `pip install python-pptx` then `python src/create_presentation.py` → `reports/Lung_Cancer_Deep_Learning_Presentation.pptx`.
- **Notebook:** open `notebooks/notebook.ipynb`, run cells top to bottom (it imports from `src/`).

## Model Architecture

The deep neural network (`LungCancerNet`) consists of:
- **Input Layer**: 15 features
- **Hidden Layers**: 128 → 64 → 32 neurons
- **Activation**: ReLU
- **Regularization**: Batch Normalization + Dropout (0.3)
- **Output Layer**: 2 classes (Binary classification)

## Key Features

- **Data Preprocessing**: Automatic normalization and encoding
- **PyTorch Implementation**: Professional deep learning framework
- **Model Comparison**: Compares ML (Logistic Regression) vs DL (Neural Network)
- **Visualizations**: Training history, confusion matrix, model comparison
- **Reproducibility**: Fixed random seeds and proper train/test splits

## Results

The model achieves competitive performance compared to traditional machine learning approaches. Detailed results are available in the notebook and generated reports.