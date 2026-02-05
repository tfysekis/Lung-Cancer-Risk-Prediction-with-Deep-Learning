"""
Data preprocessing module for lung cancer risk prediction.
Handles data loading, cleaning, and preparation for deep learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class LungCancerDataset(Dataset):
    """PyTorch Dataset for lung cancer data."""
    
    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of features
            labels: numpy array of labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(data_path):
    """
    Load and preprocess the lung cancer dataset.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(data_path)
    
    # Convert categorical variables to numeric
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    
    return df


def get_X_y(df):
    """
    Get feature matrix and target from preprocessed dataframe.
    Use with StratifiedKFold for cross-validation.
    """
    X = df.drop("LUNG_CANCER", axis=1).values
    y = df["LUNG_CANCER"].values
    return X, y


def prepare_fold(X, y, train_idx, test_idx, normalize=True):
    """
    Prepare one CV fold: split by indices, scale on train only, compute class weights.
    Returns (X_train, X_test, y_train, y_test, class_weights).
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    from collections import Counter
    import torch
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    class_weights = torch.FloatTensor([
        total / (n_classes * class_counts[i]) for i in range(n_classes)
    ])
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, class_weights


def prepare_data(df, test_size=0.2, random_state=42, normalize=True):
    """
    Prepare data for training: split and optionally normalize.
    
    Args:
        df: Preprocessed dataframe
        test_size: Proportion of test set
        random_state: Random seed
        normalize: Whether to normalize features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, class_weights)
    """
    # Separate features and target
    X = df.drop("LUNG_CANCER", axis=1).values
    y = df["LUNG_CANCER"].values
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate class weights for imbalanced data
    from collections import Counter
    import torch
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    class_weights = torch.FloatTensor([
        total / (n_classes * class_counts[i]) for i in range(n_classes)
    ])
    
    print(f"Class distribution in training set:")
    for i, count in class_counts.items():
        print(f"  Class {i}: {count} samples ({count/total*100:.1f}%)")
    print(f"Class weights: {class_weights.numpy()}")
    
    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, class_weights


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        batch_size: Batch size for DataLoader
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = LungCancerDataset(X_train, y_train)
    test_dataset = LungCancerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

