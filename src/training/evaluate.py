from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
accuracy_score, precision_score,
recall_score,  f1_score, confusion_matrix )

def pca_elbow_plot(pca, out_path: Path):
    exp = pca.explained_variance_ratio_
    fig = plt.figure()
    plt.plot(range(1, len(exp)+1), exp.cumsum(), marker="o")
    plt.xlabel("Number of Components"); plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Elbow")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def save_confusion(y_true, y_pred, out_path: Path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, square=True)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def write_metrics(records: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("F1_Score", ascending=False).to_csv(out_csv, index=False)



def get_metrics(name, y_test, y_pred, average="macro"):
    metrics = {
        "model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, average=average, zero_division=0),
    }
    return metrics  

