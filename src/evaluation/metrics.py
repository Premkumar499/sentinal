import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate standard classification metrics, including IoU for changed pixels."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate IoU for the positive class (changed pixels)
    intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    iou = intersection / union if union > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Unchanged', 'Changed'],
                yticklabels=['Unchanged', 'Changed'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix', fontweight='bold', pad=15)
    
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
