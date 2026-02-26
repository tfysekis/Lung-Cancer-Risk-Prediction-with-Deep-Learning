# Thesis project structure and cleanup
_Exported on 2/5/2026 at 18:40:14 GMT+2 from Cursor (2.4.22)_

---

**User**

All right, so here we see a structure with some things that we were trying to do, but there are just so many files and things that we're not using. So first thing first, I kind of want to understand the structure, what we did and how to improve it. So So if you wanna have an idea, this is for my thesis. So what we know I have... To talk to my teacher almost a year, so I need to make him a really good representation. This is something that we'll do much later, so don't worry about it. The only thing that I know is that we have one PDF that you can find it by the name "Lung Cancer Risk Prediction with Machine Learning". So, this one is a research done in 2022 and they were taking some data from lung cancer and they made a machine learning model. So what we need to do is to take that and try with the same data that I have downloaded I guess it's inside the data Da da ro. I guess. And... They are. we need to create a neural network for that. So... When we fix the neural network and see the values and everything, then we can go on and write the paper. So what I want to do is for you to understand the structure, maybe remove so many MD files and don't think that we need so much. Also, there are so many folders that I'm not even sure what they are doing. So... Let's just see what's going on. And Yeah, that's it actually. Also, I have a folder that the name of the folder is PDFs, where... We have two PDF. that the Teacher, gave me so I can I know where to look for learning, you know, programming and stuff.

---

**Cursor**


Exploring the project structure so we can clean it up and prepare a clear picture for your thesis. Mapping the repository and identifying what's used versus unused.



Checking the core source code and the other .md files to identify redundancies:



Checking for an `experiments` folder and inspecting the raw data:



Creating a structure overview document and cleaning up redundant files.



Here’s a concise summary of the structure and what was changed.

---

## Structure (what you have and what’s used)

**Repo root: `lung_cancer_thesis/`**
- **`Lung-Cancer-Risk-Prediction-with-Deep-Learning/`** — main thesis project (code, data, models, reports).
- **`pdfs/`** — your teacher’s two PDFs (Dive into Deep Learning; Python Deep Learning). For learning only, not part of the code.

**Inside the project:**

| What | Role |
|------|------|
| **`data/raw/survey lung cancer.csv`** | The only real data (309 rows, 15 features + `LUNG_CANCER`). Same type of survey data as the 2022 paper. |
| **`data/processed/`** and **`data/external/`** | Empty (only `.gitkeep`). Placeholders; not used. |
| **`src/`** | All logic: load/preprocess data, define models, train one model, compare all models. The notebook **depends on this**; don’t remove it. |
| **`notebooks/notebook.ipynb`** | Exploration and running the same pipeline (imports from `src/`). |
| **`models/`** | 17 saved `.pth` models (one per architecture you trained). |
| **`reports/`** | Plots (confusion matrices, ROC, PR, training curves) and **`model_comparison_results.csv`** with metrics for all models. |
| **`Lung_Cancer_Risk_Prediction_with_Machine_Learning_ (1).pdf`** | The 2022 reference paper (inside the project folder). |

So: **data** → **`src`** (preprocessing + models + training/comparison) → **notebook** runs that; **models** and **reports** are outputs. The “experiments” folder in the old README doesn’t exist; it was removed from the docs.

---

## Cleanup done

**Removed 7 redundant Markdown files** (they repeated the same “notebook vs src”, “run order”, “what each file does”):
- `QUICK_START.md`, `HOW_IT_WORKS.md`, `SIMPLE_EXPLANATION.md`, `WORKFLOW_GUIDE.md`, `PROJECT_STRUCTURE.md`, `ARCHITECTURE_COVERAGE.md`, `IMPROVEMENTS_SUMMARY.md`

**Kept / added:**
- **`README.md`** — Updated: thesis goal, link to the 2022 paper, short structure, install, and how to run (`train.py` vs `compare_models.py` vs notebook). No more reference to `experiments/`.
- **`STRUCTURE_OVERVIEW.md`** — **New.** Single place for: goal (paper + same data + neural network), what each folder does, what each code file does, and how to run (one model vs all models vs notebook). Use this when you (or your teacher) need a quick map of the project.
- **`FINAL_SUMMARY.md`** — **Kept.** Short list of the 14 architectures and what gets saved (models + reports). Useful when writing the thesis and choosing which model(s) to fix on.

So you now have **3 Markdown files**: README (entry point), STRUCTURE_OVERVIEW (map of folders and code), FINAL_SUMMARY (what you have for the thesis).

---

## Next steps for the thesis

1. **Fix the neural network:** Pick one (or a few) best models from `reports/model_comparison_results.csv` (e.g. LayerNorm, Deep 5/6, ELU, Minimal all around 93–95% test accuracy). You can retrain with a fixed seed and document that as your “final” model(s).
2. **Write the paper:** Use the 2022 PDF as reference, your data in `data/raw/`, and the chosen model(s) + metrics and plots from `reports/`.

If you tell me your priority (e.g. “one final model only” or “keep comparison of 2–3”), I can suggest exactly which scripts to run and what to put in the thesis (methods + results).

---

**User**

Okay, I guess you can remove more folders like the data inside it has this in external and processed which they are empty. The experiment folder is still empty. Now about the models. Okay, we can leave that. Notebox we can leave that. Reports I see that I have too many images. So okay, now we go inside. The... the compare models.py, the data processing.py. I guess we don't need the data processing. We have model.py and train. So I need you to go inside there and see what's actually going on. Like I said to you, we want with the same data to train and neural network. I don't know what is the best way to do it. I mean, training like 30 neural networks I guess it's not something that we need to do. I think And I'm not sure you can always speak your mind. I guess we can fix like six or seven so I can represent that to my teacher and we need good visuals. Now, I'm not sure what the value should be, like the learning rate, the batch number, things like that. I don't even remember them. How many neurons we have to do? I don't know. So I need you to go inside those and delete the overkill because I remember that we've built like too many models. Let's just pick the most common one and the most, you know, right one for a thesis. And Also, I will remove all the models folder and the reports folder so we can have a clean start. And Yeah, I don't know fix those things set right the parameters I guess we need TensorFlow or it's something that we don't need I'm asking that because I see in the PDF that it has some guns by TorchGarage and TestOnFlow and I guess we are using some of them, right? So Yeah, go inside, just keep the best models. And I am going to run them and of course keep a lot of visuals so I can have them later. Also, It would be really nice to tell me in the Post that we're working, you know, the lung cancer risk prediction with machine learning. What was the prediction, the accuracy for that model? So, you know, we don't want to go with 95% accuracy. We don't want an overfitting. And the other guy with machine learning maybe did 60%. I don't know what percent he did. So, Let's go in there. and do those things.

---

**Cursor**


Checking the 2022 paper's reported accuracy, then simplifying the codebase to 6–7 models with sensible defaults and removing empty folders.



Found the 2022 paper: Dritsas & Trigka report 97.1% accuracy and 99.3% AUC with Rotation Forest (not 60%). Implementing the cleanup and model reduction.



Reducing `model.py` to seven architectures and updating `compare_models.py` and `train.py`.



Checking the notebook for imports and comparison cells that need updating:



Updating FINAL_SUMMARY to reflect the 7-model setup:



Summary of what’s done:

---

## 1. Removed empty folders

- **`data/processed/`** and **`data/external/`** are removed (only **`data/raw/`** with `survey lung cancer.csv` remains).
- There was no **experiments/** folder in the repo (only mentioned in the old README; that reference is removed).

---

## 2. Reference paper – actual numbers

In the 2022 paper (**Dritsas & Trigka**), the best model is **Rotation Forest** (classical ML), not 60%:

- **Accuracy: 97.1%**
- **AUC: 99.3%**
- F-Measure, precision, recall: 97.1%

So the baseline is strong. Your DL models can sit a bit below that and still be a valid comparison (you can discuss dataset size and overfitting). This is documented in **REFERENCE_PAPER.md** and the comparison plots now show a **red line at 97.1%** so you can compare at a glance.

---

## 3. Reduced to 7 models (thesis-ready)

**`src/model.py`** now defines only these 7 architectures:

| # | Model | Role |
|---|--------|------|
| 1 | **Standard (128-64-32)** | Main baseline MLP |
| 2 | **Simple (64-32)** | Lightweight |
| 3 | **Minimal (32)** | Simplest baseline |
| 4 | **Deep (5 layers)** | Depth comparison |
| 5 | **ELU Activation** | Activation variant |
| 6 | **LayerNorm** | Normalization variant |
| 7 | **Residual** | Skip connections |

Wide, LeakyReLU, Swish, Wide&Deep, Attention, High Dropout, and the extra Deep 4/6 variants are removed.

---

## 4. Single place for hyperparameters

- **`src/compare_models.py`** – at the top: `NUM_EPOCHS = 100`, `BATCH_SIZE = 32`, `LEARNING_RATE = 0.001`, `EARLY_STOPPING_PATIENCE = 15`.
- **`src/train.py`** – same defaults in `DEFAULT_*` constants so `train.py` and `compare_models.py` stay in sync.

So: **100 epochs**, **batch size 32**, **lr 0.001**, **early stopping 15**. You can change them in one place in each file.

---

## 5. Data preprocessing kept

**`data_preprocessing.py`** is required: it loads the CSV, encodes GENDER/LUNG_CANCER, normalizes, does the train/test split, and builds the DataLoaders. Both `train.py` and `compare_models.py` use it. Nothing was removed from it.

---

## 6. TensorFlow / framework

- The project uses **PyTorch only** (see `requirements.txt`: `torch`).
- The 2022 lung cancer paper uses classical ML (Rotation Forest), not DL.
- The teacher’s PDFs (PyTorch/Keras/TensorFlow) are for general learning. For this thesis, **PyTorch is enough**; TensorFlow is not used or required.

---

## 7. Docs updated

- **README.md** – Structure without processed/external, reference to 2022 baseline (97.1%), PyTorch-only note, REFERENCE_PAPER link.
- **STRUCTURE_OVERVIEW.md** – Only `data/raw/`, 7 models, comparison plots with baseline line.
- **REFERENCE_PAPER.md** – New: paper citation, 97.1% / 99.3% baseline, framework note.
- **FINAL_SUMMARY.md** – Updated for the 7-model setup and the reference baseline.

---

## What you can do next

1. **Delete** the contents of **`models/`** and **`reports/`** (or the whole folders) for a clean start.
2. From the project root run:
   ```bash
   python src/compare_models.py
   ```
   This will train the 7 models, save 7 `.pth` in **models/** and all visuals + **model_comparison_results.csv** in **reports/** (including the 97.1% baseline line in the accuracy and complexity plots).
3. Use **reports/** for the thesis (tables and figures) and **REFERENCE_PAPER.md** when you write the comparison with the 2022 paper.

If you tell me your preferred learning rate or batch size, I can suggest exact values to paste into the top of `compare_models.py` and `train.py`.

