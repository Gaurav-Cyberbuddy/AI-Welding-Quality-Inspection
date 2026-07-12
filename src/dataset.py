from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WeldingDataset(Dataset):

    def __init__(self, parquet_path, dataset_root, transform=None):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        metadata_path = Path(row["path"])

        # Convert original dataset path to local relative path.
        relative_path = Path(*metadata_path.parts[3:])

        image_path = self.dataset_root / relative_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Binary classification labels:
        # OK = 0
        # KO = 1
        label = 0 if row["class"] == "OK" else 1

        return image, label