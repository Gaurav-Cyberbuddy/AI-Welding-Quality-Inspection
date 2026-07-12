from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import time

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

IMAGE_DIR = PROJECT_ROOT / "data" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

FAILURE_LOG = PROJECT_ROOT / "data" / "download_failures.csv"

MAX_WORKERS = 8
MAX_RETRIES = 3
TIMEOUT = 30


df = pd.read_parquet(METADATA_PATH)

print("=" * 60)
print("WELDING IMAGE DOWNLOADER")
print("=" * 60)

print(f"Total images: {len(df)}")
print(f"Download directory: {IMAGE_DIR}")


def download_image(row):

    sample_id = row["sample_id"]
    url = row["external_path"]

    output_path = IMAGE_DIR / f"{sample_id}.jpg"

    # Resume support: skip existing valid images
    if output_path.exists():

        try:
            with Image.open(output_path) as image:
                image.verify()

            return sample_id, "SKIPPED", None

        except Exception:
            output_path.unlink()


    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            image_bytes = response.content

            # Verify downloaded bytes and detect image format
            with Image.open(BytesIO(image_bytes)) as image:

                image.verify()

                image_format = image.format


            if image_format != "JPEG":

                return (
                    sample_id,
                    "FAILED",
                    f"Unexpected image format: {image_format}"
                )


            # Save only after successful verification
            output_path.write_bytes(image_bytes)

            return sample_id, "DOWNLOADED", None


        except Exception as error:

            if attempt == MAX_RETRIES:

                return sample_id, "FAILED", str(error)

            time.sleep(attempt * 2)


failures = []

downloaded = 0
skipped = 0
failed = 0


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = [

        executor.submit(download_image, row)

        for _, row in df.iterrows()

    ]


    for completed, future in enumerate(
        as_completed(futures),
        start=1
    ):

        sample_id, status, error = future.result()

        if status == "DOWNLOADED":
            downloaded += 1

        elif status == "SKIPPED":
            skipped += 1

        else:
            failed += 1

            failures.append(
                {
                    "sample_id": sample_id,
                    "error": error
                }
            )


        if completed % 100 == 0 or completed == len(futures):

            print(
                f"Progress: {completed}/{len(futures)} | "
                f"Downloaded: {downloaded} | "
                f"Skipped: {skipped} | "
                f"Failed: {failed}"
            )


# Save failures
if failures:

    pd.DataFrame(failures).to_csv(
        FAILURE_LOG,
        index=False
    )

    print(f"\nFailure log saved to: {FAILURE_LOG}")

else:

    if FAILURE_LOG.exists():
        FAILURE_LOG.unlink()

    print("\nNo download failures.")


print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

print(f"Downloaded: {downloaded}")
print(f"Skipped: {skipped}")
print(f"Failed: {failed}")

print("\nIMAGE DOWNLOAD PROCESS COMPLETED")