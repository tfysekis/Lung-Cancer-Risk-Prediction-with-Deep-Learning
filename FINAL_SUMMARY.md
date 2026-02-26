# Summary – What You Have (Thesis)

## Reference baseline (2022 paper)

**Dritsas & Trigka (2022)** – *Lung Cancer Risk Prediction with Machine Learning*  
- **Best model:** Rotation Forest (classical ML).  
- **Reported:** Accuracy **97.1%**, AUC **99.3%**, F-Measure / precision / recall 97.1%.  
- See **REFERENCE_PAPER.md** for details.

---

## 7 deep learning architectures (for thesis)

1. **Standard (128-64-32)** – 3-layer MLP, ReLU, BatchNorm, dropout 0.3  
2. **Simple (64-32)** – 2-layer, lightweight  
3. **Minimal (32)** – 1-layer, simplest baseline  
4. **Deep (5 layers)** – 5 hidden layers, 64 units each  
5. **ELU Activation** – Same as Standard with ELU  
6. **LayerNorm** – Same as Standard with Layer Normalization  
7. **Residual** – Skip connections  

---

## What gets saved when you run `python src/compare_models.py`

- **models/** – 7 `.pth` files (one per architecture).  
- **reports/** – For each model: confusion matrix, ROC curve, PR curve, training history (4 PNGs each).  
- **reports/model_comparison_results.csv** – Table with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, parameters, best epoch.  
- **reports/accuracy_comparison.png** – Bar chart with a **red line at 97.1%** (2022 baseline).  
- **reports/complexity_vs_performance.png** – Parameters vs accuracy.  
- **reports/comprehensive_metrics.png** – Four panels: accuracy, precision, recall, F1.  

---

## Hyperparameters (defaults)

- **Epochs:** 100 (with early stopping, patience 15).  
- **Batch size:** 32.  
- **Learning rate:** 0.001.  
- **Optimizer:** Adam.  
- **Loss:** Cross-entropy with class weights (imbalanced data).  

Defined in `src/compare_models.py` and `src/train.py`.

---

## Framework

- **PyTorch** only. TensorFlow/Keras are not used; the teacher’s PDFs mention them for general learning.
