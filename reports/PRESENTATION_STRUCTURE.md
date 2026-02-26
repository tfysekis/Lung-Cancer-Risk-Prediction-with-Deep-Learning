# Presentation structure (slide-by-slide outline)

Use this to build slides manually or to check what the auto-generated PowerPoint contains.

---

**Slide 1 — Title**  
- Title: Lung Cancer Risk Prediction with Deep Learning  
- Subtitle: Comparison with Dritsas & Trigka (2022) — Same dataset, neural networks vs classical ML  

---

**Slide 2 — Objective**  
- Same task as the 2022 paper: predict lung cancer risk (YES/NO) from survey data  
- Same dataset: lung cancer survey (~309 samples, 15 features)  
- Their approach: classical ML (Rotation Forest) — 97.1% accuracy, 99.3% AUC  
- Our approach: deep learning (7 MLP architectures) — compare performance  

---

**Slide 3 — Data**  
- Source: survey lung cancer.csv (data/raw/)  
- Features: age, gender, smoking, yellow fingers, anxiety, peer pressure, chronic disease, fatigue, allergy, wheezing, alcohol, coughing, shortness of breath, swallowing, chest pain  
- Target: LUNG_CANCER (YES/NO)  
- Preprocessing: standard scaling, stratified splits, class weights for imbalance  

---

**Slide 4 — Pipeline / Steps**  
- 1. Load data, encode labels  
- 2. Preprocess: StandardScaler, stratified split  
- 3. Train 7 architectures (same hyperparameters)  
- 4. Evaluate: single split, 5-fold CV, 10-fold CV  
- 5. Compare with 2022 paper  

---

**Slide 5 — Methods (7 architectures)**  
- Standard (128-64-32): 3-layer MLP, ReLU, BatchNorm, Dropout 0.3  
- Simple (64-32): 2-layer (64→32), Dropout 0.2  
- Minimal (32): 1 hidden layer (32 units), Dropout 0.2  
- Deep (5 layers): 5 hidden layers (64 each), Dropout 0.3  
- ELU Activation: same as Standard but ELU instead of ReLU  
- LayerNorm: same as Standard but LayerNorm instead of BatchNorm  
- Residual: skip connections, hidden size 128, Dropout 0.3  

---

**Slide 6 — Training setup (same for all)**  
- Learning rate 0.001 (Adam), batch size 32, max epochs 100  
- Early stopping: 15 epochs without improvement  
- Loss: weighted Cross-Entropy (class imbalance)  
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 10)  
- Random seed 42; input standardized  

---

**Slide 7 — Evaluation — what we report**  
- Single 80/20 split: one split — 80% train, 20% test; one number per model  
- 5-fold CV: 5 parts; 5 rounds (4 train, 1 test); report mean ± std  
- 10-fold CV: 10 parts; 10 rounds; more stable estimate  
- Metrics: accuracy, precision, recall, F1, ROC-AUC  

---

**Slide 8 — 5-fold vs 10-fold CV**  
- 5-fold: 5 rounds; 10-fold: 10 rounds; same idea, average over rounds  
- More folds → more stable mean; results usually similar  
- We run both for standard practice  

---

**Slide 9 — Results: Single 80/20 split**  
- **Image:** `reports/single_split/accuracy_comparison.png`  
- Caption: Test accuracy by model (red line = 2022 reference 97.1%)  

---

**Slide 10 — Results: Single split — complexity**  
- **Image:** `reports/single_split/complexity_vs_performance.png`  
- Caption: Parameters vs test accuracy  

---

**Slide 10b — Results: Metrics (Accuracy, Precision, Recall, F1)**  
- **Image:** `reports/single_split/comprehensive_metrics.png`  
- Caption: All four metrics per model (single split)  

---

**Slides 10c–10f — Confusion matrices (all 7 architectures)**  
- 2 per slide, then 1:  
  - Slide: Standard (128-64-32) & Simple (64-32) — `confusion_matrix_standard_128-64-32.png`, `confusion_matrix_simple_64-32.png`  
  - Slide: Minimal (32) & Deep (5 layers) — `confusion_matrix_minimal_32.png`, `confusion_matrix_deep_5_layers.png`  
  - Slide: ELU Activation & LayerNorm — `confusion_matrix_elu_activation.png`, `confusion_matrix_layernorm.png`  
  - Slide: Residual — `confusion_matrix_residual.png`  

---

**Slides 10g–10j — Training curves (all 7 architectures)**  
- Same grouping: Standard+Simple, Minimal+Deep, ELU+LayerNorm, Residual  
- Files: `training_history_*.png`  

---

**Slides 10k–10n — ROC curves (all 7 architectures)**  
- Same grouping; files: `roc_curve_*.png`  

---

**Slides 10o–10r — Precision-Recall curves (all 7 architectures)**  
- Same grouping; files: `pr_curve_*.png`  

---

**Slide 11 — Results: 5-fold CV**  
- **Image:** `reports/cv5/accuracy_comparison.png`  
- Caption: 5-fold CV mean ± std; comparable to how the 2022 paper likely reported results  

---

**Slide 12 — Results: 5-fold vs 10-fold CV (optional)**  
- **Images:** `reports/cv5/accuracy_comparison.png` and `reports/cv10/accuracy_comparison.png` side by side  

---

**Slide 13 — Comparison with 2022 paper**  
- Reference: Rotation Forest — 97.1% accuracy, 99.3% AUC  
- Our best (e.g. LayerNorm with 5-fold CV): ~92% accuracy, ~94% AUC  
- Gap is expected: on small tabular data, tree ensembles often outperform neural nets  
- Same dataset and task; we compare a different family of models (DL)  
- Conclusion: DL is viable; classical ML remains strong on this dataset  

---

**Slide 14 — Summary**  
- Same lung cancer survey data and classification task as the 2022 paper  
- Seven DL architectures compared (single split + 5-fold and 10-fold CV)  
- Best DL performance ~92% accuracy (vs 97.1% Rotation Forest)  
- All visuals and CSVs in reports/single_split, reports/cv5, reports/cv10  

---

**Slide 15 — Thank you**  
- Questions?  

---

## How to get the PowerPoint

From the project root:

```bash
pip install python-pptx
python src/create_presentation.py
```

Output: `reports/Lung_Cancer_Deep_Learning_Presentation.pptx`

If you prefer to build slides yourself, use this outline and the images in `reports/single_split/`, `reports/cv5/`, and `reports/cv10/`.
