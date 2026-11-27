"""
utils/evaluation.py
---------------------
Đánh giá mô hình học máy:
  - Tính toán các chỉ số: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - In bảng Classification Report
  - Tự động vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix) vào output/confusion_matrix/
  - Tự động vẽ và lưu Đường cong ROC (ROC Curve) vào output/roc_curve/
  - Lưu bảng tổng hợp kết quả so sánh vào output/model_comparison.csv
  - Vẽ biểu đồ cột so sánh tổng thể các mô hình vào output/model_accuracy.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from utils.logger import print_header

CM_DIR = os.path.join("output", "confusion_matrix")
ROC_DIR = os.path.join("output", "roc_curve")
OUTPUT_DIR = "output"


def _ensure_dirs():
    os.makedirs(CM_DIR, exist_ok=True)
    os.makedirs(ROC_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_and_evaluate(name: str, train_fn, X_train, y_train, X_test, y_test):
    """Huấn luyện và đánh giá một mô hình học máy theo format chuẩn."""
    _ensure_dirs()

    print_header(f"TRAINING {name.upper()}", width=60)
    model = train_fn(X_train, y_train)

    print_header(name, width=60)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.5

    print(f"{'Accuracy':<12}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'Precision':<12}: {prec:.4f}")
    print(f"{'Recall':<12}: {rec:.4f}")
    print(f"{'F1 Score':<12}: {f1:.4f}")
    print(f"{'ROC AUC':<12}: {auc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {name}", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    cm_path = os.path.join(CM_DIR, f"{name}.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve - {name}", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(ROC_DIR, f"{name}.png")
    plt.savefig(roc_path, dpi=120)
    plt.close()
    print(f"Saved: {roc_path}")

    result = {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc,
    }
    return model, result


def print_model_comparison(results: list):
    """In bảng so sánh tổng hợp, lưu CSV và vẽ biểu đồ so sánh tổng thể."""
    _ensure_dirs()
    print_header("MODEL PERFORMANCE COMPARISON", width=70)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved : {csv_path}")

    # Biểu đồ so sánh Accuracy, F1 & ROC_AUC
    plt.figure(figsize=(9, 5))
    df_melt = df.melt(id_vars="Model", value_vars=["Accuracy", "F1", "ROC_AUC"], 
                      var_name="Metric", value_name="Score")
    ax = sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Model Performance Comparison (Cell2Cell Dataset)", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, "model_accuracy.png")
    plt.savefig(acc_path, dpi=120)
    plt.close()
    print(f"Saved : {acc_path}")

    return df
