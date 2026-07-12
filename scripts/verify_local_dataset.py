from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "ds_meta.parquet"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "welding-detection-challenge-dataset"
)

REPORT_DIR = PROJECT_ROOT / "data" / "verification_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_parquet(METADATA_PATH)


print("=" * 60)
print("LOCAL DATASET VERIFICATION")
print("=" * 60)

print(f"Metadata records: {len(df)}")


missing_images = []
corrupt_images = []
metadata_mismatches = []

valid_images = 0


for index, row in df.iterrows():

    # Example metadata path:
    #
    # challenge-welding/datasets/
    # welding-detection-challenge-dataset/
    # c33/OK/operator/sample_0.jpeg

    metadata_path = Path(row["path"])

    # Keep only:
    # c33/OK/operator/sample_0.jpeg

    relative_path = Path(*metadata_path.parts[3:])

    local_image_path = DATASET_ROOT / relative_path


    # Check file exists
    if not local_image_path.exists():

        missing_images.append({
            "sample_id": row["sample_id"],
            "metadata_path": row["path"],
            "expected_local_path": str(local_image_path),
        })

        continue


    # Verify image integrity
    try:

        with Image.open(local_image_path) as image:
            image.verify()

        valid_images += 1

    except Exception as error:

        corrupt_images.append({
            "sample_id": row["sample_id"],
            "path": str(local_image_path),
            "error": str(error),
        })

        continue


    # Verify folder metadata
    # relative path:
    # c33 / OK / operator / sample_0.jpeg

    parts = relative_path.parts

    local_seam = parts[0]
    local_class = parts[1]
    local_labelling_type = parts[2]


    if (
        local_seam != row["welding-seams"]
        or local_class != row["class"]
        or local_labelling_type != row["labelling_type"]
    ):

        metadata_mismatches.append({
            "sample_id": row["sample_id"],
            "metadata_seam": row["welding-seams"],
            "folder_seam": local_seam,
            "metadata_class": row["class"],
            "folder_class": local_class,
            "metadata_labelling": row["labelling_type"],
            "folder_labelling": local_labelling_type,
            "path": str(local_image_path),
        })


    completed = index + 1

    if completed % 1000 == 0:

        print(
            f"Verified: {completed}/{len(df)}"
        )


# Save reports

if missing_images:

    pd.DataFrame(missing_images).to_csv(
        REPORT_DIR / "missing_images.csv",
        index=False
    )


if corrupt_images:

    pd.DataFrame(corrupt_images).to_csv(
        REPORT_DIR / "corrupt_images.csv",
        index=False
    )


if metadata_mismatches:

    pd.DataFrame(metadata_mismatches).to_csv(
        REPORT_DIR / "metadata_mismatches.csv",
        index=False
    )


print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

print(f"Metadata records: {len(df)}")
print(f"Valid images: {valid_images}")
print(f"Missing images: {len(missing_images)}")
print(f"Corrupt images: {len(corrupt_images)}")
print(f"Metadata/folder mismatches: {len(metadata_mismatches)}")


if (
    len(missing_images) == 0
    and len(corrupt_images) == 0
    and len(metadata_mismatches) == 0
):

    print("\nLOCAL DATASET VERIFICATION SUCCESSFUL")

else:

    print("\nDATASET VERIFICATION FOUND PROBLEMS")
    print(f"Check reports inside: {REPORT_DIR}")