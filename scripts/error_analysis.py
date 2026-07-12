from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


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

VAL_PATH = PROJECT_ROOT / "data" / "splits" / "val.parquet"

MODEL_PATH = PROJECT_ROOT / "models" / "balanced_resnet18.pth"

REPORT_DIR = PROJECT_ROOT / "results" / "error_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("BALANCED MODEL VALIDATION ERROR ANALYSIS")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================

val_dataset = WeldingDataset(
    VAL_PATH,
    DATASET_ROOT,
    transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_df = pd.read_parquet(
    VAL_PATH
).reset_index(drop=True)


print(f"Validation samples: {len(val_dataset)}")


# ============================================================
# LOAD MODEL
# ============================================================

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


print(
    f"Loaded checkpoint saved at epoch: "
    f"{checkpoint['epoch']}"
)


# ============================================================
# RUN INFERENCE
# ============================================================

predictions = []

ko_probabilities = []


with torch.no_grad():

    for batch_number, (images, labels) in enumerate(
        val_loader,
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


        batch_predictions = outputs.argmax(
            dim=1,
        )


        predictions.extend(
            batch_predictions
            .cpu()
            .numpy()
            .tolist()
        )


        ko_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )


        if batch_number % 25 == 0:

            print(
                f"Analyzed batch: "
                f"{batch_number}/{len(val_loader)}"
            )


# ============================================================
# ADD PREDICTIONS TO METADATA
# ============================================================

analysis_df = val_df.copy()


analysis_df["true_label"] = analysis_df["class"].map({
    "OK": 0,
    "KO": 1,
})


analysis_df["predicted_label"] = predictions

analysis_df["ko_probability"] = ko_probabilities


analysis_df["predicted_class"] = (
    analysis_df["predicted_label"]
    .map({
        0: "OK",
        1: "KO",
    })
)


analysis_df["correct"] = (
    analysis_df["true_label"]
    == analysis_df["predicted_label"]
)


# ============================================================
# EXTRACT ERRORS
# ============================================================

false_negatives = analysis_df[
    (analysis_df["true_label"] == 1)
    &
    (analysis_df["predicted_label"] == 0)
].copy()


false_positives = analysis_df[
    (analysis_df["true_label"] == 0)
    &
    (analysis_df["predicted_label"] == 1)
].copy()


all_errors = analysis_df[
    analysis_df["correct"] == False
].copy()


# ============================================================
# SAVE REPORTS
# ============================================================

analysis_df.to_csv(
    REPORT_DIR / "validation_predictions.csv",
    index=False,
)


false_negatives.to_csv(
    REPORT_DIR / "false_negatives.csv",
    index=False,
)


false_positives.to_csv(
    REPORT_DIR / "false_positives.csv",
    index=False,
)


all_errors.to_csv(
    REPORT_DIR / "all_errors.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ERROR SUMMARY")
print("=" * 60)


print(f"Total validation samples: {len(analysis_df)}")

print(f"Correct predictions: {analysis_df['correct'].sum()}")

print(f"Total errors: {len(all_errors)}")

print(f"False negatives (missed KO): {len(false_negatives)}")

print(f"False positives (false alarms): {len(false_positives)}")


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FALSE NEGATIVES BY WELDING SEAM")
print("=" * 60)

if len(false_negatives) > 0:

    print(
        false_negatives[
            "welding-seams"
        ].value_counts()
    )

else:

    print("No false negatives.")


print("\n" + "=" * 60)
print("FALSE NEGATIVES BY LABELLING TYPE")
print("=" * 60)

if len(false_negatives) > 0:

    print(
        false_negatives[
            "labelling_type"
        ].value_counts()
    )

else:

    print("No false negatives.")


print("\n" + "=" * 60)
print("FALSE NEGATIVES BY BLUR CLASS")
print("=" * 60)

if len(false_negatives) > 0:

    print(
        false_negatives[
            "blur_class"
        ].value_counts()
    )

else:

    print("No false negatives.")


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FALSE POSITIVES BY WELDING SEAM")
print("=" * 60)

if len(false_positives) > 0:

    print(
        false_positives[
            "welding-seams"
        ].value_counts()
    )

else:

    print("No false positives.")


print("\n" + "=" * 60)
print("FALSE POSITIVES BY LABELLING TYPE")
print("=" * 60)

if len(false_positives) > 0:

    print(
        false_positives[
            "labelling_type"
        ].value_counts()
    )

else:

    print("No false positives.")


print("\n" + "=" * 60)
print("FALSE POSITIVES BY BLUR CLASS")
print("=" * 60)

if len(false_positives) > 0:

    print(
        false_positives[
            "blur_class"
        ].value_counts()
    )

else:

    print("No false positives.")


# ============================================================
# NUMERICAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("BLUR AND LUMINOSITY COMPARISON")
print("=" * 60)


comparison = analysis_df.groupby(
    ["true_label", "correct"]
)[
    [
        "blur_level",
        "luminosity_level",
    ]
].mean()


print(comparison)


# ============================================================
# MOST CONFIDENT ERRORS
# ============================================================

print("\n" + "=" * 60)
print("MOST CONFIDENT MISSED KO IMAGES")
print("=" * 60)


if len(false_negatives) > 0:

    columns = [
        "sample_id",
        "welding-seams",
        "labelling_type",
        "blur_class",
        "blur_level",
        "luminosity_level",
        "ko_probability",
        "path",
    ]

    print(
        false_negatives
        .sort_values(
            "ko_probability",
            ascending=True,
        )[columns]
        .head(10)
        .to_string(index=False)
    )


else:

    print("No false negatives.")


print("\n" + "=" * 60)
print("MOST CONFIDENT FALSE ALARMS")
print("=" * 60)


if len(false_positives) > 0:

    columns = [
        "sample_id",
        "welding-seams",
        "labelling_type",
        "blur_class",
        "blur_level",
        "luminosity_level",
        "ko_probability",
        "path",
    ]

    print(
        false_positives
        .sort_values(
            "ko_probability",
            ascending=False,
        )[columns]
        .head(10)
        .to_string(index=False)
    )


else:

    print("No false positives.")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print("ERROR ANALYSIS COMPLETED")

print("=" * 60)

print(f"Reports saved inside: {REPORT_DIR}")