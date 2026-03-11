"""
Training script for lung cancer risk prediction using deep learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import sys

# Handle imports - works both from notebook and command line
try:
    from model import LungCancerNet, LungCancerNetSimple
    from data_preprocessing import load_data, prepare_data, create_dataloaders
except ImportError:
    # If running from project root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import LungCancerNet, LungCancerNetSimple
    from src.data_preprocessing import load_data, prepare_data, create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Store predictions for metrics
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    learning_rate=0.001,
    device='cpu',
    save_path=None,
    class_weights=None,
    early_stopping_patience=15
):
    """
    Train the deep learning model.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save the best model
        class_weights: Tensor of class weights for imbalanced data (None for balanced)
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        dict: Training history and best model metrics
    """
    model = model.to(device)
    
    # Handle class imbalance with weighted loss
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss with class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        test_loss, test_acc, _, _ = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            print(f"  Best Test Acc: {best_test_acc:.4f}")
            if epochs_without_improvement > 0:
                print(f"  No improvement for {epochs_without_improvement} epochs")
            print("-" * 50)
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            print(f"Best test accuracy: {best_test_acc:.4f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_test_acc': best_test_acc,
            'history': history
        }, save_path)
        print(f"\nModel saved to {save_path}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_loss, test_acc, y_pred, y_true = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['NO', 'YES']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return {
        'model': model,
        'history': history,
        'best_test_acc': best_test_acc,
        'y_pred': y_pred,
        'y_true': y_true
    }


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    
    plt.show()


# ---------- Default hyperparameters (same as compare_models.py) ----------
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE = 15

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(project_root, "data", "survey lung cancer.csv")
    if not os.path.isfile(DATA_PATH):
        DATA_PATH = os.path.join(project_root, "data", "raw", "survey lung cancer.csv")
    MODEL_SAVE_PATH = os.path.join(project_root, "models", "lung_cancer_model.pth")
    PLOT_SAVE_PATH = os.path.join(project_root, "reports", "training_history.png")
    NUM_EPOCHS = DEFAULT_EPOCHS
    BATCH_SIZE = DEFAULT_BATCH_SIZE
    LEARNING_RATE = DEFAULT_LEARNING_RATE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, class_weights = prepare_data(df)
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
    )
    
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Input features: {X_train.shape[1]}")
    
    # Create model
    model = LungCancerNet(input_size=X_train.shape[1])
    
    # Train model
    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path=MODEL_SAVE_PATH,
        class_weights=class_weights,
        early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    
    # Plot training history
    plot_training_history(results['history'], save_path=PLOT_SAVE_PATH)

