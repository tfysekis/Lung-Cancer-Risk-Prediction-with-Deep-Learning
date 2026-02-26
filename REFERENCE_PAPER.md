# Reference Paper – Baseline for Comparison

**Paper:** Dritsas, E.; Trigka, M. *Lung Cancer Risk Prediction with Machine Learning Models*. Big Data Cogn. Comput. 2022, 6, 139.  
https://doi.org/10.3390/bdcc6040139

## What they did

- **Data:** Same type of lung cancer survey dataset (habits and symptoms as features).
- **Method:** Classical **machine learning** (Rotation Forest and other classifiers).
- **Best model:** **Rotation Forest**.

## Reported performance (baseline for your thesis)

| Metric    | Value   |
|----------|---------|
| Accuracy | **97.1%** |
| AUC      | **99.3%** |
| F-Measure | 97.1% |
| Precision | 97.1% |
| Recall   | 97.1%  |

So the 2022 paper reports **97.1% accuracy** and **99.3% AUC** — a strong baseline. Your neural networks can be compared fairly to this (e.g. similar or slightly lower accuracy with DL is still a valid result; you can discuss dataset size, overfitting, and why ML might reach high accuracy on this task).

## Framework note

- The **2022 paper** uses classical ML (Rotation Forest, etc.), not deep learning.
- **This project** uses **PyTorch** only (no TensorFlow/Keras). The teacher’s PDFs (in `pdfs/`) mention PyTorch, Keras, and TensorFlow for general deep learning; for this thesis, PyTorch is enough and you do not need to add TensorFlow.

---

## Why your accuracy might be a bit lower (and how to report it)

- **Single split vs cross-validation:** The 2022 paper likely reported **cross-validated** results (e.g. 10-fold), so 97.1% is an average over many train/test splits. With one fixed 80/20 split you have only 62 test samples, so a few predictions either way change accuracy a lot. For a fair comparison, run **5-fold or 10-fold CV** and report **mean ± std** (e.g. "Accuracy 92% ± 4%").
- **How to get CV numbers:** Run `python src/compare_models.py --cv 5`. Results are written to `reports/model_comparison_results_cv.csv` with mean and standard deviation per model. Use these numbers in the thesis and state that you used stratified k-fold CV like the reference.
- **Small dataset:** With ~250 training samples, classical ML (trees/ensembles) often does very well; deep learning tends to need more data. So reporting "our best DL model reached X% (vs 97.1% Rotation Forest)" is a valid result and you can discuss in the thesis: dataset size, risk of overfitting, and why ensemble ML can outperform DL on small tabular data.
