"""
Compare 7 deep learning architectures for lung cancer risk prediction.
Trains each model, saves to models/, and generates visuals + CSV in reports/.
Reference: Dritsas & Trigka (2022) reported 97.1% accuracy, 99.3% AUC with Rotation Forest (ML).
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
import os
import sys
import argparse
from sklearn.model_selection import StratifiedKFold

# Imports
try:
    from model import (
        LungCancerNet, LungCancerNetSimple, LungCancerNetMinimal,
        LungCancerNetDeep, LungCancerNetELU, LungCancerNetLayerNorm,
        LungCancerNetResidual
    )
    from data_preprocessing import load_data, prepare_data, prepare_fold, get_X_y, create_dataloaders
    from train import train_model
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import (
        LungCancerNet, LungCancerNetSimple, LungCancerNetMinimal,
        LungCancerNetDeep, LungCancerNetELU, LungCancerNetLayerNorm,
        LungCancerNetResidual
    )
    from src.data_preprocessing import load_data, prepare_data, prepare_fold, get_X_y, create_dataloaders
    from src.train import train_model

# ---------- Hyperparameters (thesis defaults) ----------
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED = 42

# Reference: Dritsas & Trigka (2022) - Rotation Forest on same dataset
REFERENCE_ACCURACY = 0.971   # 97.1%
REFERENCE_AUC = 0.993        # 99.3%


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_visualizations(model_name, model, test_loader, result, report_dir, device):
    """Save confusion matrix, ROC, PR curve, and training history for one model."""
    model.eval()
    model = model.to(device)
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probabilities[:, 1].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and').lower()

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f'confusion_matrix_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f'roc_curve_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f'pr_curve_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Training History
    history = result['history']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss - {model_name}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy - {model_name}', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f'training_history_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}


def create_comparison_plots(results_df, report_dir):
    """Accuracy bar chart, complexity vs accuracy, and 4-metric comparison."""
    # 1. Accuracy comparison
    plt.figure(figsize=(10, 6))
    bars = plt.barh(results_df['Model'], results_df['Test Accuracy'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(results_df))))
    plt.axvline(REFERENCE_ACCURACY, color='red', linestyle='--', linewidth=2,
                label=f'Reference (2022 ML): {REFERENCE_ACCURACY:.1%}')
    plt.xlabel('Test Accuracy')
    plt.title('Model Comparison: Test Accuracy (red line = 2022 paper baseline)', fontsize=14, fontweight='bold')
    plt.xlim([max(0, results_df['Test Accuracy'].min() - 0.05), 1.0])
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    for bar, acc in zip(bars, results_df['Test Accuracy']):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height() / 2, f'{acc:.2%}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Parameters vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Parameters'], results_df['Test Accuracy'], s=150, alpha=0.7)
    for _, row in results_df.iterrows():
        plt.annotate(row['Model'], (row['Parameters'], row['Test Accuracy']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.axhline(REFERENCE_ACCURACY, color='red', linestyle='--', alpha=0.7, label='2022 baseline')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')
    plt.title('Complexity vs Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'complexity_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Four metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, metric in enumerate(['Test Accuracy', 'Precision', 'Recall', 'F1-Score']):
        ax = axes[idx // 2, idx % 2]
        ax.barh(results_df['Model'], results_df[metric], color=plt.cm.Set3(np.linspace(0, 1, len(results_df))))
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.set_xlim([max(0, results_df[metric].min() - 0.05), 1.0])
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_cv_comparison_plots(results_df, report_dir, n_splits):
    """Comparison plots for CV results: mean +/- std, reference line."""
    acc_mean = results_df['Test Accuracy (mean)']
    acc_std = results_df['Test Accuracy (std)']
    auc_mean = results_df['ROC-AUC (mean)']
    auc_std = results_df['ROC-AUC (std)']

    # 1. Accuracy comparison (with error bars)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(results_df['Model'], acc_mean, xerr=acc_std, capsize=4,
                    color=plt.cm.viridis(np.linspace(0, 1, len(results_df))))
    plt.axvline(REFERENCE_ACCURACY, color='red', linestyle='--', linewidth=2,
                label=f'Reference (2022 ML): {REFERENCE_ACCURACY:.1%}')
    plt.xlabel('Test Accuracy (mean +/- std)')
    plt.title(f'Model Comparison ({n_splits}-fold CV): Test Accuracy', fontsize=14, fontweight='bold')
    plt.xlim([max(0, (acc_mean - acc_std).min() - 0.05), 1.0])
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    for i, (bar, m, s) in enumerate(zip(bars, acc_mean, acc_std)):
        plt.text(m + s + 0.02, bar.get_y() + bar.get_height() / 2, f'{m:.2%}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Parameters vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['Parameters'], acc_mean, yerr=acc_std, fmt='o', capsize=5, markersize=10)
    for _, row in results_df.iterrows():
        plt.annotate(row['Model'], (row['Parameters'], row['Test Accuracy (mean)']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.axhline(REFERENCE_ACCURACY, color='red', linestyle='--', alpha=0.7, label='2022 baseline')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy (mean +/- std)')
    plt.title(f'Complexity vs Performance ({n_splits}-fold CV)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'complexity_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Two-panel: Accuracy and ROC-AUC (mean +/- std)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(results_df['Model'], acc_mean, xerr=acc_std, capsize=3, color=plt.cm.Set3(np.linspace(0, 1, len(results_df))))
    ax1.axvline(REFERENCE_ACCURACY, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Test Accuracy')
    ax1.set_title(f'Test Accuracy ({n_splits}-fold CV)')
    ax1.set_xlim([0, 1.0])
    ax1.grid(axis='x', alpha=0.3)
    ax2.barh(results_df['Model'], auc_mean, xerr=auc_std, capsize=3, color=plt.cm.Set3(np.linspace(0, 1, len(results_df))))
    ax2.axvline(REFERENCE_AUC, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('ROC-AUC')
    ax2.set_title(f'ROC-AUC ({n_splits}-fold CV)')
    ax2.set_xlim([0, 1.05])
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def write_run_info(report_dir, run_type, command_hint):
    """Write a short README in the report folder so you know what run produced it."""
    path = os.path.join(report_dir, 'run_info.txt')
    with open(path, 'w') as f:
        f.write(f"Lung Cancer Risk Prediction - Report folder\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Run type: {run_type}\n\n")
        f.write(f"Command: {command_hint}\n\n")
        f.write(f"Reference (Dritsas & Trigka 2022): 97.1% accuracy, 99.3% AUC\n")
        f.write(f"Use this folder for presentation slides (single run vs 5-fold vs 10-fold).\n")


def compare_all_models(
    data_path,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device='cpu',
    save_dir='../models',
    report_dir='../reports',
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
):
    """Train and compare the 7 thesis models. Returns DataFrame of results."""
    print("=" * 70)
    print("LUNG CANCER RISK PREDICTION - DEEP LEARNING MODEL COMPARISON")
    print("=" * 70)
    print(f"Reference (Dritsas & Trigka 2022, Rotation Forest): "
          f"Accuracy {REFERENCE_ACCURACY:.1%}, AUC {REFERENCE_AUC:.1%}")
    print(f"Settings: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}, "
          f"early_stop={early_stopping_patience}")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    write_run_info(report_dir, "Single 80/20 split (no CV)", "python src/compare_models.py")

    print("\n[1/4] Loading data...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler, class_weights = prepare_data(df, normalize=True)
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=batch_size
    )
    input_size = X_train.shape[1]
    print(f"   Train: {len(y_train)}, Test: {len(y_test)}, Features: {input_size}")

    # 7 models for thesis
    models_config = [
        {'name': 'Standard (128-64-32)', 'model': LungCancerNet(input_size=input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3),
         'description': 'Standard 3-layer MLP'},
        {'name': 'Simple (64-32)', 'model': LungCancerNetSimple(input_size=input_size, hidden_size=64, dropout_rate=0.2),
         'description': 'Simple 2-layer'},
        {'name': 'Minimal (32)', 'model': LungCancerNetMinimal(input_size=input_size, hidden_size=32, dropout_rate=0.2),
         'description': 'Minimal 1-layer'},
        {'name': 'Deep (5 layers)', 'model': LungCancerNetDeep(input_size=input_size, hidden_size=64, num_layers=5, dropout_rate=0.3),
         'description': 'Deep 5-layer MLP'},
        {'name': 'ELU Activation', 'model': LungCancerNetELU(input_size=input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3),
         'description': 'Standard with ELU'},
        {'name': 'LayerNorm', 'model': LungCancerNetLayerNorm(input_size=input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3),
         'description': 'Layer Normalization'},
        {'name': 'Residual', 'model': LungCancerNetResidual(input_size=input_size, hidden_size=128, dropout_rate=0.3),
         'description': 'Residual connections'},
    ]

    results = []
    print(f"\n[2/4] Training {len(models_config)} models...")
    for i, config in enumerate(models_config, 1):
        model_name = config['name']
        model = config['model']
        description = config['description']
        print(f"\n[{i}/{len(models_config)}] {model_name} ({description}) — Params: {count_parameters(model):,}")

        model_save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_model.pth")
        try:
            result = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_path=model_save_path,
                class_weights=class_weights,
                early_stopping_patience=early_stopping_patience,
            )
            y_true, y_pred = result['y_true'], result['y_pred']
            viz = save_model_visualizations(model_name, result['model'], test_loader, result, report_dir, device)
            metrics = {
                'Model': model_name,
                'Description': description,
                'Parameters': count_parameters(result['model']),
                'Test Accuracy': result['best_test_acc'],
                'Precision': precision_score(y_true, y_pred, average='weighted'),
                'Recall': recall_score(y_true, y_pred, average='weighted'),
                'F1-Score': f1_score(y_true, y_pred, average='weighted'),
                'ROC-AUC': viz['roc_auc'],
                'PR-AUC': viz['pr_auc'],
                'Final Train Loss': result['history']['train_loss'][-1],
                'Final Test Loss': result['history']['test_loss'][-1],
                'Best Epoch': np.argmax(result['history']['test_acc']) + 1,
            }
            results.append(metrics)
            print(f"   Done — Test Accuracy: {metrics['Test Accuracy']:.4f}")
        except Exception as e:
            print(f"   Error: {e}")
            continue

    print("\n[3/4] Saving comparison table...")
    results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
    results_path = os.path.join(report_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"   {results_path} (single split)")

    print("\n[4/4] Comparison plots...")
    create_comparison_plots(results_df, report_dir)
    print(f"   Plots in {report_dir}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(results_df[['Model', 'Test Accuracy', 'F1-Score', 'ROC-AUC', 'Parameters']].to_string(index=False))
    return results_df


def _get_roc_auc(model, test_loader, device):
    """Compute ROC-AUC for a trained model on test_loader."""
    model.eval()
    model = model.to(device)
    y_true, y_proba = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return auc(fpr, tpr)


def compare_all_models_cv(
    data_path,
    n_splits=5,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device='cpu',
    report_dir='../reports',
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
):
    """
    Train each of the 7 models with stratified k-fold CV and report mean ± std.
    More comparable to the 2022 paper (they likely used CV). No models saved.
    """
    print("=" * 70)
    print("LUNG CANCER RISK PREDICTION - CROSS-VALIDATION COMPARISON")
    print("=" * 70)
    print(f"Reference (Dritsas & Trigka 2022): Accuracy {REFERENCE_ACCURACY:.1%}, AUC {REFERENCE_AUC:.1%}")
    print(f"Settings: {n_splits}-fold stratified CV, lr={learning_rate}, batch_size={batch_size}, "
          f"epochs={num_epochs}, early_stop={early_stopping_patience}")
    print("=" * 70)

    os.makedirs(report_dir, exist_ok=True)
    write_run_info(report_dir, f"{n_splits}-fold stratified CV",
                   f"python src/compare_models.py --cv {n_splits}")
    df = load_data(data_path)
    X, y = get_X_y(df)
    input_size = X.shape[1]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    models_config = [
        {'name': 'Standard (128-64-32)', 'model_factory': lambda: LungCancerNet(input_size=input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3)},
        {'name': 'Simple (64-32)', 'model_factory': lambda: LungCancerNetSimple(input_size=input_size, hidden_size=64, dropout_rate=0.2)},
        {'name': 'Minimal (32)', 'model_factory': lambda: LungCancerNetMinimal(input_size=input_size, hidden_size=32, dropout_rate=0.2)},
        {'name': 'Deep (5 layers)', 'model_factory': lambda: LungCancerNetDeep(input_size=input_size, hidden_size=64, num_layers=5, dropout_rate=0.3)},
        {'name': 'ELU Activation', 'model_factory': lambda: LungCancerNetELU(input_size=input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3)},
        {'name': 'LayerNorm', 'model_factory': lambda: LungCancerNetLayerNorm(input_size=input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3)},
        {'name': 'Residual', 'model_factory': lambda: LungCancerNetResidual(input_size=input_size, hidden_size=128, dropout_rate=0.3)},
    ]

    results = []
    for i, config in enumerate(models_config, 1):
        model_name = config['name']
        accs, aucs = [], []
        print(f"\n[{i}/7] {model_name} — {n_splits}-fold CV...")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test, y_train, y_test, class_weights = prepare_fold(X, y, train_idx, test_idx, normalize=True)
            train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=batch_size)
            model = config['model_factory']()
            result = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                save_path=None,
                class_weights=class_weights,
                early_stopping_patience=early_stopping_patience,
            )
            accs.append(result['best_test_acc'])
            roc_auc = _get_roc_auc(result['model'], test_loader, device)
            aucs.append(roc_auc)
        n_params = count_parameters(config['model_factory']())
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        results.append({
            'Model': model_name,
            'Parameters': n_params,
            'Test Accuracy (mean)': mean_acc,
            'Test Accuracy (std)': std_acc,
            'ROC-AUC (mean)': mean_auc,
            'ROC-AUC (std)': std_auc,
        })
        print(f"   Accuracy: {mean_acc:.2%} +/- {std_acc:.2%}  |  ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    results_df = pd.DataFrame(results).sort_values('Test Accuracy (mean)', ascending=False)
    results_path = os.path.join(report_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nCV results saved to {results_path}")

    create_cv_comparison_plots(results_df, report_dir, n_splits)
    print(f"   Comparison plots saved in {report_dir}")

    # Summary vs reference
    print("\n" + "=" * 70)
    print("CV SUMMARY vs REFERENCE (2022)")
    print("=" * 70)
    print(results_df[['Model', 'Test Accuracy (mean)', 'Test Accuracy (std)', 'ROC-AUC (mean)', 'ROC-AUC (std)', 'Parameters']].to_string(index=False))
    best_acc = results_df['Test Accuracy (mean)'].max()
    best_auc = results_df['ROC-AUC (mean)'].max()
    print(f"\nBest CV Accuracy: {best_acc:.2%}  (reference 2022: {REFERENCE_ACCURACY:.1%})")
    print(f"Best CV AUC:      {best_auc:.4f}  (reference: {REFERENCE_AUC:.3f})")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DL models for lung cancer risk prediction.")
    parser.add_argument("--cv", type=int, default=None, metavar="K",
                        help="Run K-fold stratified CV and report mean±std (e.g. --cv 5). No --cv = single 80/20 split.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds when using --cv (default: 5).")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(project_root, "data", "raw", "survey lung cancer.csv")
    SAVE_DIR = os.path.join(project_root, "models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.cv is not None:
        n_splits = max(2, args.cv)
        report_dir = os.path.join(project_root, "reports", f"cv{n_splits}")
        compare_all_models_cv(
            data_path=DATA_PATH,
            n_splits=n_splits,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device,
            report_dir=report_dir,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )
    else:
        report_dir = os.path.join(project_root, "reports", "single_split")
        compare_all_models(
            data_path=DATA_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device,
            save_dir=SAVE_DIR,
            report_dir=report_dir,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )
