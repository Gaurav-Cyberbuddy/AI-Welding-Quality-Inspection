from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "welding-detection-challenge-dataset"
)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


# ------------------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ------------------------------------------------------------
# CUSTOM PYTORCH DATASET
# ------------------------------------------------------------

class WeldingDataset(Dataset):

    def __init__(self, parquet_path, dataset_root, transform=None):

        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)

        self.dataset_root = dataset_root

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        metadata_path = Path(row["path"])

        # Remove original server path:
        # challenge-welding/datasets/welding-detection-challenge-dataset/

        relative_path = Path(*metadata_path.parts[3:])

        image_path = self.dataset_root / relative_path


        # Load actual image

        with Image.open(image_path) as image:

            image = image.convert("RGB")


        if self.transform is not None:

            image = self.transform(image)


        # Binary labels
        # OK = 0
        # KO = 1

        label = 0 if row["class"] == "OK" else 1


        return image, label


# ------------------------------------------------------------
# CREATE DATASETS
# ------------------------------------------------------------

train_dataset = WeldingDataset(
    SPLITS_DIR / "train.parquet",
    DATASET_ROOT,
    transform,
)

val_dataset = WeldingDataset(
    SPLITS_DIR / "val.parquet",
    DATASET_ROOT,
    transform,
)

test_dataset = WeldingDataset(
    SPLITS_DIR / "test.parquet",
    DATASET_ROOT,
    transform,
)


# ------------------------------------------------------------
# CREATE DATALOADERS
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("=" * 60)
print("PYTORCH DATA PIPELINE TEST")
print("=" * 60)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ------------------------------------------------------------
# LOAD ONE BATCH
# ------------------------------------------------------------

images, labels = next(iter(train_loader))


print("\nFIRST TRAINING BATCH")

print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

print(f"Image dtype: {images.dtype}")
print(f"Label dtype: {labels.dtype}")

print(f"Minimum pixel value: {images.min().item():.4f}")
print(f"Maximum pixel value: {images.max().item():.4f}")

print(f"Labels in batch: {labels.tolist()}")


print("\nClass mapping:")
print("OK -> 0")
print("KO -> 1")


print("\nPYTORCH DATA PIPELINE TEST SUCCESSFUL")