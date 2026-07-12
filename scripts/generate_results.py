from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import WeldingDataset


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "welding-detection-challenge-dataset"
)

TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"

BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "baseline_resnet18.pth"
)

BALANCED_MODEL_PATH = (
    PROJECT_ROOT / "models" / "balanced_resnet18.pth"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GENERATING FINAL RESULT FIGURES")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# TEST DATASET
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


test_dataset = WeldingDataset(
    TEST_PATH,
    DATASET_ROOT,
    test_transform,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print(f"Test samples: {len(test_dataset)}")


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_path, model_name):

    print(f"\nEvaluating: {model_name}")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )


    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(DEVICE)

    model.eval()


    y_true = []

    y_pred = []

    y_score = []


    with torch.no_grad():

        for batch_number, (images, labels) in enumerate(
            test_loader,
            start=1,
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )


            outputs = model(images)


            probabilities = torch.softmax(
                outputs,
                dim=1,
            )


            predictions = outputs.argmax(
                dim=1,
            )


            y_true.extend(
                labels.numpy().tolist()
            )


            y_pred.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )


            y_score.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
                .tolist()
            )


            if batch_number % 50 == 0:

                print(
                    f"Processed batch: "
                    f"{batch_number}/{len(test_loader)}"
                )


    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_score),
    )


# ============================================================
# EVALUATE BOTH MODELS
# ============================================================

baseline_true, baseline_pred, baseline_score = evaluate_model(
    BASELINE_MODEL_PATH,
    "Baseline ResNet-18",
)


balanced_true, balanced_pred, balanced_score = evaluate_model(
    BALANCED_MODEL_PATH,
    "Balanced ResNet-18",
)


# ============================================================
# CONFUSION MATRIX FUNCTION
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )


    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)


    ax.set_title(title)

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("True Class")


    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(["OK", "KO"])

    ax.set_yticklabels(["OK", "KO"])


    for row in range(2):

        for column in range(2):

            ax.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
            )


    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

save_confusion_matrix(
    baseline_true,
    baseline_pred,
    "Baseline ResNet-18 Confusion Matrix",
    "baseline_confusion_matrix.png",
)


save_confusion_matrix(
    balanced_true,
    balanced_pred,
    "Balanced ResNet-18 Confusion Matrix",
    "balanced_confusion_matrix.png",
)


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

baseline_cm = confusion_matrix(
    baseline_true,
    baseline_pred,
).ravel()


balanced_cm = confusion_matrix(
    balanced_true,
    balanced_pred,
).ravel()


baseline_tn, baseline_fp, baseline_fn, baseline_tp = baseline_cm

balanced_tn, balanced_fp, balanced_fn, balanced_tp = balanced_cm


baseline_precision = (
    baseline_tp
    / (baseline_tp + baseline_fp)
)


baseline_recall = (
    baseline_tp
    / (baseline_tp + baseline_fn)
)


baseline_f1 = (
    2
    * baseline_precision
    * baseline_recall
    / (baseline_precision + baseline_recall)
)


balanced_precision = (
    balanced_tp
    / (balanced_tp + balanced_fp)
)


balanced_recall = (
    balanced_tp
    / (balanced_tp + balanced_fn)
)


balanced_f1 = (
    2
    * balanced_precision
    * balanced_recall
    / (balanced_precision + balanced_recall)
)


baseline_pr_auc = average_precision_score(
    baseline_true,
    baseline_score,
)


balanced_pr_auc = average_precision_score(
    balanced_true,
    balanced_score,
)


metrics = [
    "KO Precision",
    "KO Recall",
    "KO F1",
    "PR-AUC",
]


baseline_values = [
    baseline_precision,
    baseline_recall,
    baseline_f1,
    baseline_pr_auc,
]


balanced_values = [
    balanced_precision,
    balanced_recall,
    balanced_f1,
    balanced_pr_auc,
]


x = np.arange(len(metrics))

width = 0.35


fig, ax = plt.subplots(figsize=(9, 6))


ax.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline",
)


ax.bar(
    x + width / 2,
    balanced_values,
    width,
    label="Balanced",
)


ax.set_ylabel("Score")

ax.set_title(
    "Baseline vs Balanced ResNet-18"
)

ax.set_xticks(x)

ax.set_xticklabels(metrics)

ax.set_ylim(0, 1)

ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR / "model_comparison.png",
    dpi=300,
)


plt.close(fig)


# ============================================================
# BALANCED MODEL ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    balanced_true,
    balanced_score,
)


roc_auc = roc_auc_score(
    balanced_true,
    balanced_score,
)


fig, ax = plt.subplots(figsize=(7, 6))


ax.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}",
)


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)


ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title(
    "Balanced ResNet-18 ROC Curve"
)

ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR / "balanced_roc_curve.png",
    dpi=300,
)


plt.close(fig)


# ============================================================
# BALANCED MODEL PRECISION-RECALL CURVE
# ============================================================

precision, recall, _ = precision_recall_curve(
    balanced_true,
    balanced_score,
)


pr_auc = average_precision_score(
    balanced_true,
    balanced_score,
)


fig, ax = plt.subplots(figsize=(7, 6))


ax.plot(
    recall,
    precision,
    label=f"PR-AUC = {pr_auc:.4f}",
)


ax.set_xlabel("Recall")

ax.set_ylabel("Precision")

ax.set_title(
    "Balanced ResNet-18 Precision-Recall Curve"
)

ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURE_DIR / "balanced_precision_recall_curve.png",
    dpi=300,
)


plt.close(fig)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print("FINAL RESULT FIGURES GENERATED")

print("=" * 60)

print(f"Figures saved inside: {FIGURE_DIR}")

print("\nGenerated files:")

print("baseline_confusion_matrix.png")

print("balanced_confusion_matrix.png")

print("model_comparison.png")

print("balanced_roc_curve.png")

print("balanced_precision_recall_curve.png")