from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import WeldingDataset


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "welding-detection-challenge-dataset"
)

TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"

MODEL_PATH = PROJECT_ROOT / "models" / "balanced_resnet18.pth"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("BALANCED MODEL TEST EVALUATION")
print("=" * 60)

print(f"Device: {DEVICE}")


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


# Create same ResNet-18 architecture

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)


# Load balanced model checkpoint

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print(
    f"Loaded checkpoint saved at epoch: "
    f"{checkpoint['epoch']}"
)


all_labels = []
all_predictions = []
all_ko_probabilities = []


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


        all_labels.extend(
            labels.numpy().tolist()
        )


        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


        all_ko_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )


        if batch_number % 25 == 0:

            print(
                f"Evaluated batch: "
                f"{batch_number}/{len(test_loader)}"
            )


y_true = all_labels

y_pred = all_predictions

y_score = all_ko_probabilities


# ============================================================
# METRICS
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)


report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["OK", "KO"],
    digits=4,
    zero_division=0,
)


pr_auc = average_precision_score(
    y_true,
    y_score,
)


roc_auc = roc_auc_score(
    y_true,
    y_score,
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(cm)


print("\nInterpretation:")

print(
    f"True Negatives  (OK correctly predicted): {tn}"
)

print(
    f"False Positives (OK predicted as KO):      {fp}"
)

print(
    f"False Negatives (KO predicted as OK):      {fn}"
)

print(
    f"True Positives  (KO correctly predicted):  {tp}"
)


print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(report)


print("=" * 60)

print("RANKING METRICS")

print("=" * 60)

print(f"PR-AUC:  {pr_auc:.4f}")

print(f"ROC-AUC: {roc_auc:.4f}")


print("\nBALANCED MODEL TEST EVALUATION COMPLETED")