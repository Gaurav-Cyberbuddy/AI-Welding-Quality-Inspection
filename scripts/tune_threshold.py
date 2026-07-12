from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import WeldingDataset


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "welding-detection-challenge-dataset"
)

VAL_PATH = PROJECT_ROOT / "data" / "splits" / "val.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_resnet18.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Same preprocessing used during validation/training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_dataset = WeldingDataset(
    VAL_PATH,
    DATASET_ROOT,
    transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# Load model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# Collect KO probabilities
y_true = []
y_score = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        ko_probabilities = probabilities[:, 1]

        y_true.extend(labels.numpy().tolist())

        y_score.extend(
            ko_probabilities.cpu().numpy().tolist()
        )


y_true = np.array(y_true)
y_score = np.array(y_score)


print("=" * 80)
print("VALIDATION THRESHOLD TUNING")
print("=" * 80)

print(f"Validation samples: {len(y_true)}")
print(f"Actual KO samples: {y_true.sum()}")

print("\nThreshold | Precision | Recall | F1 | FP | FN | TP")
print("-" * 80)


results = []

# Test thresholds from 0.05 to 0.95
for threshold in np.arange(0.05, 1.00, 0.05):

    y_pred = (
        y_score >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    results.append(
        (
            threshold,
            precision,
            recall,
            f1,
            fp,
            fn,
            tp,
        )
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f} | "
        f"{fp:3d} | "
        f"{fn:2d} | "
        f"{tp:2d}"
    )


# Choose threshold with highest KO F1
best_result = max(
    results,
    key=lambda result: result[3],
)

best_threshold = best_result[0]


print("\n" + "=" * 80)
print("BEST VALIDATION THRESHOLD BY KO F1")
print("=" * 80)

print(f"Threshold: {best_result[0]:.2f}")
print(f"KO Precision: {best_result[1]:.4f}")
print(f"KO Recall:    {best_result[2]:.4f}")
print(f"KO F1:        {best_result[3]:.4f}")
print(f"False Positives: {best_result[4]}")
print(f"False Negatives: {best_result[5]}")
print(f"True Positives:  {best_result[6]}")