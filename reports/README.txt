Lung Cancer Risk Prediction - Reports for presentation
======================================================

Run the comparison three ways. Each run writes to its own folder so you can
compare results and use the right set of figures for your presentation.

  python src/compare_models.py           -->  reports/single_split/
  python src/compare_models.py --cv 5    -->  reports/cv5/
  python src/compare_models.py --cv 10   -->  reports/cv10/

Folder contents
---------------
- single_split/   One 80/20 train-test split (seed 42).
                  Contains: model_comparison_results.csv, run_info.txt,
                  accuracy_comparison.png, complexity_vs_performance.png,
                  comprehensive_metrics.png, plus per-model plots
                  (confusion_matrix_*, roc_curve_*, pr_curve_*, training_history_*).

- cv5/            5-fold stratified cross-validation (mean +/- std).
                  Contains: model_comparison_results.csv, run_info.txt,
                  accuracy_comparison.png, complexity_vs_performance.png,
                  comprehensive_metrics.png.

- cv10/           Same as cv5 but 10-fold CV.

Each folder has run_info.txt with the exact command and run type.
Reference (Dritsas & Trigka 2022): 97.1% accuracy, 99.3% AUC.
