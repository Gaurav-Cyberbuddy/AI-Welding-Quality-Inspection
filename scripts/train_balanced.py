from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "balanced_resnet18.pth"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("BALANCED RESNET-18 TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = WeldingDataset(
    SPLITS_DIR / "train.parquet",
    DATASET_ROOT,
    train_transform,
)

val_dataset = WeldingDataset(
    SPLITS_DIR / "val.parquet",
    DATASET_ROOT,
    eval_transform,
)


print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ============================================================
# CREATE BALANCED SAMPLER
# ============================================================

train_df = pd.read_parquet(
    SPLITS_DIR / "train.parquet"
)


ok_count = (
    train_df["class"] == "OK"
).sum()

ko_count = (
    train_df["class"] == "KO"
).sum()


print("\nOriginal training distribution:")

print(f"OK samples: {ok_count}")
print(f"KO samples: {ko_count}")


class_counts = train_df["class"].value_counts()


sample_weights = train_df["class"].map({

    "OK": 1.0 / class_counts["OK"],

    "KO": 1.0 / class_counts["KO"],

}).to_numpy()


sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double,
)


sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True,
)


print("\nWeightedRandomSampler created.")

print(
    "Rare KO images will be sampled more frequently."
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),
)


print(f"\nTraining batches per epoch: {len(train_loader)}")

print(f"Validation batches: {len(val_loader)}")


# ============================================================
# CREATE MODEL
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# Replace original ImageNet classifier.
#
# Original:
# 512 features -> 1000 classes
#
# New:
# 512 features -> 2 classes
#
# OK = 0
# KO = 1

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)


model = model.to(DEVICE)


# ============================================================
# LOSS FUNCTION
# ============================================================

# Normal CrossEntropyLoss is used here.
#
# We are NOT using class-weighted loss because
# WeightedRandomSampler is already balancing
# exposure to OK and KO samples.

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    seen_ok = 0

    seen_ko = 0


    for batch_number, (images, labels) in enumerate(
        train_loader,
        start=1,
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )


        # Count sampled classes for monitoring

        seen_ok += (labels == 0).sum().item()

        seen_ko += (labels == 1).sum().item()


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels,
        )


        # Backpropagation

        loss.backward()


        # Update model weights

        optimizer.step()


        # Accumulate loss

        running_loss += (
            loss.item()
            * images.size(0)
        )


        # Predictions

        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        if batch_number % 100 == 0:

            print(
                f"Training batch: "
                f"{batch_number}/"
                f"{len(train_loader)}"
            )


    epoch_loss = (
        running_loss / total
    )


    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy,
        seen_ok,
        seen_ko,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels,
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    val_loss = (
        running_loss / total
    )


    val_accuracy = (
        correct / total
    )


    return (
        val_loss,
        val_accuracy,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    print(f"\nEpoch {epoch}/{EPOCHS}")

    print("-" * 60)


    (
        train_loss,
        train_accuracy,
        sampled_ok,
        sampled_ko,

    ) = train_one_epoch()


    val_loss, val_accuracy = validate()


    print(f"\nTrain Loss: {train_loss:.4f}")

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    print("\nSampled training distribution:")

    print(f"OK samples seen: {sampled_ok}")

    print(f"KO samples seen: {sampled_ko}")


    print(f"\nValidation Loss: {val_loss:.4f}")

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # Save model with lowest validation loss

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(

            {

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss": val_loss,

                "sampling_strategy":
                    "WeightedRandomSampler",

                "class_mapping": {
                    "OK": 0,
                    "KO": 1,
                },

            },

            BEST_MODEL_PATH,
        )


        print(
            f"\nBest model saved: "
            f"{BEST_MODEL_PATH}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print("BALANCED RESNET-18 TRAINING COMPLETED")

print("=" * 60)

print(f"\nBest model location: {BEST_MODEL_PATH}")