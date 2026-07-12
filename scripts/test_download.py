from pathlib import Path
import pandas as pd
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ds_meta.parquet"
)

TEST_DIR = PROJECT_ROOT / "data" / "test_download"
TEST_DIR.mkdir(parents=True, exist_ok=True)


# Load metadata
df = pd.read_parquet(METADATA_PATH)

# Select first sample
row = df.iloc[0]

sample_id = row["sample_id"]
url = row["external_path"]

print("=" * 60)
print("SINGLE IMAGE DOWNLOAD TEST")
print("=" * 60)

print(f"Sample ID: {sample_id}")
print(f"Class: {row['class']}")
print(f"Downloading image...")


# Download image
response = requests.get(
    url,
    timeout=30
)

print(f"HTTP Status Code: {response.status_code}")

response.raise_for_status()


# Determine extension
content_type = response.headers.get("Content-Type", "").lower()

if "png" in content_type:
    extension = ".png"

elif "jpeg" in content_type or "jpg" in content_type:
    extension = ".jpg"

else:
    extension = ".img"


output_path = TEST_DIR / f"{sample_id}{extension}"

output_path.write_bytes(response.content)


# Verify downloaded image
with Image.open(output_path) as image:

    image.verify()


# Reopen to inspect properties
with Image.open(output_path) as image:

    print(f"Image format: {image.format}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")


print(f"Saved to: {output_path}")

print("\nSINGLE IMAGE DOWNLOAD TEST SUCCESSFUL")