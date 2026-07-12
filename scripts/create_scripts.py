from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ds_meta.parquet"
)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# Load metadata
df = pd.read_parquet(METADATA_PATH)

print(f"Total samples: {len(df)}")


# First split:
# 70% train, 30% temporary set
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"],
)


# Second split:
# Divide temporary set equally into validation and test
# Final ratio = 70% / 15% / 15%
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"],
)


# Save splits
train_path = SPLITS_DIR / "train.parquet"
val_path = SPLITS_DIR / "val.parquet"
test_path = SPLITS_DIR / "test.parquet"

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)


# Print summary
def print_split_summary(name, split_df):

    print(f"\n{name.upper()} SPLIT")
    print("-" * 40)

    print(f"Samples: {len(split_df)}")

    print("\nClass counts:")
    print(split_df["class"].value_counts())

    print("\nClass percentages:")
    print(
        (
            split_df["class"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )


print_split_summary("train", train_df)
print_split_summary("validation", val_df)
print_split_summary("test", test_df)


print("\nSplit files created successfully:")
print(train_path)
print(val_path)
print(test_path)