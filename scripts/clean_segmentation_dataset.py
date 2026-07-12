from pathlib import Path
from collections import Counter
import shutil
import csv

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "Weld quality inspection - Segmentation"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "welding_segmentation_clean"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation_cleaning"
)

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "Bad Welding",
    1: "Crack",
    2: "Excess Reinforcement",
    3: "Good Welding",
    4: "Porosity",
    5: "Spatters",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".heic",
    ".heif",
}


# ============================================================
# PREPARE OUTPUT
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Remove only the previous CLEANED COPY.
# The original dataset is never modified.

if OUTPUT_ROOT.exists():

    print(f"Removing previous cleaned dataset: {OUTPUT_ROOT}")

    shutil.rmtree(OUTPUT_ROOT)


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPORT STORAGE
# ============================================================

removed_records = []

summary_records = []


# ============================================================
# VALIDATE ONE SEGMENTATION LINE
# ============================================================

def validate_segmentation_line(line):

    parts = line.strip().split()

    # YOLO segmentation:
    # class_id x1 y1 x2 y2 x3 y3 ...
    #
    # Minimum:
    # 1 class ID + 6 polygon coordinates

    if len(parts) < 7:

        return False, None, "too_few_values"


    if (len(parts) - 1) % 2 != 0:

        return False, None, "odd_number_of_coordinates"


    try:

        class_id = int(parts[0])

    except ValueError:

        return False, None, "invalid_class_id_format"


    if class_id not in CLASS_NAMES:

        return False, None, "unknown_class_id"


    try:

        coordinates = [
            float(value)
            for value in parts[1:]
        ]

    except ValueError:

        return False, None, "non_numeric_coordinate"


    if any(
        value < 0.0 or value > 1.0
        for value in coordinates
    ):

        return False, None, "coordinate_out_of_range"


    return True, class_id, None


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(split_name):

    print("\n" + "=" * 70)
    print(f"CLEANING SPLIT: {split_name.upper()}")
    print("=" * 70)


    source_images = SOURCE_ROOT / split_name / "images"

    source_labels = SOURCE_ROOT / split_name / "labels"


    output_images = OUTPUT_ROOT / split_name / "images"

    output_labels = OUTPUT_ROOT / split_name / "labels"


    output_images.mkdir(parents=True, exist_ok=True)

    output_labels.mkdir(parents=True, exist_ok=True)


    image_files = {

        path.stem: path

        for path in source_images.iterdir()

        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS

    }


    label_files = {

        path.stem: path

        for path in source_labels.glob("*.txt")

    }


    matched_stems = sorted(

        set(image_files)
        & set(label_files)

    )


    images_without_labels = sorted(

        set(image_files)
        - set(label_files)

    )


    labels_without_images = sorted(

        set(label_files)
        - set(image_files)

    )


    # ========================================================
    # RECORD UNMATCHED FILES
    # ========================================================

    for stem in images_without_labels:

        removed_records.append({

            "split": split_name,

            "file": image_files[stem].name,

            "reason": "image_without_label",

        })


    for stem in labels_without_images:

        removed_records.append({

            "split": split_name,

            "file": label_files[stem].name,

            "reason": "label_without_image",

        })


    # ========================================================
    # PROCESS MATCHED IMAGE/LABEL PAIRS
    # ========================================================

    copied_images = 0

    copied_labels = 0

    removed_pairs = 0

    removed_lines = 0

    valid_instances = 0

    class_counts = Counter()


    for index, stem in enumerate(
        matched_stems,
        start=1,
    ):

        image_path = image_files[stem]

        label_path = label_files[stem]


        # ====================================================
        # READ LABEL
        # ====================================================

        try:

            lines = label_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

        except Exception as error:

            removed_records.append({

                "split": split_name,

                "file": label_path.name,

                "reason": f"label_read_error: {error}",

            })

            removed_pairs += 1

            continue


        valid_lines = []


        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            if not line.strip():

                continue


            is_valid, class_id, reason = (
                validate_segmentation_line(line)
            )


            if not is_valid:

                removed_lines += 1

                removed_records.append({

                    "split": split_name,

                    "file": label_path.name,

                    "reason": (
                        f"removed_line_{line_number}:"
                        f"{reason}"
                    ),

                })

                continue


            valid_lines.append(line.strip())

            class_counts[class_id] += 1

            valid_instances += 1


        # ====================================================
        # REMOVE PAIR IF NO VALID POLYGONS REMAIN
        # ====================================================

        if len(valid_lines) == 0:

            removed_pairs += 1

            removed_records.append({

                "split": split_name,

                "file": stem,

                "reason": "no_valid_segmentation_annotations",

            })

            continue


        # ====================================================
        # COPY IMAGE
        # ====================================================

        shutil.copy2(

            image_path,

            output_images / image_path.name,

        )


        # ====================================================
        # WRITE CLEAN LABEL
        # ====================================================

        cleaned_label_path = (

            output_labels / label_path.name

        )


        cleaned_label_path.write_text(

            "\n".join(valid_lines) + "\n",

            encoding="utf-8",

        )


        copied_images += 1

        copied_labels += 1


        if index % 500 == 0:

            print(

                f"Processed: "
                f"{index}/{len(matched_stems)}"

            )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nCLEANING SUMMARY")

    print("-" * 70)

    print(f"Source images: {len(image_files)}")

    print(f"Source labels: {len(label_files)}")

    print(f"Matched pairs: {len(matched_stems)}")

    print(
        f"Images without labels: "
        f"{len(images_without_labels)}"
    )

    print(
        f"Labels without images: "
        f"{len(labels_without_images)}"
    )

    print(f"Removed malformed lines: {removed_lines}")

    print(f"Removed unusable pairs: {removed_pairs}")

    print(f"Clean images copied: {copied_images}")

    print(f"Clean labels created: {copied_labels}")

    print(f"Valid instances: {valid_instances}")


    print("\nCLEAN CLASS DISTRIBUTION")

    print("-" * 70)


    for class_id, class_name in CLASS_NAMES.items():

        print(

            f"{class_id}: "
            f"{class_name:<25} "
            f"{class_counts.get(class_id, 0)}"

        )


    summary_records.append({

        "split": split_name,

        "source_images": len(image_files),

        "source_labels": len(label_files),

        "matched_pairs": len(matched_stems),

        "images_without_labels": len(images_without_labels),

        "labels_without_images": len(labels_without_images),

        "removed_malformed_lines": removed_lines,

        "removed_unusable_pairs": removed_pairs,

        "clean_images": copied_images,

        "clean_labels": copied_labels,

        "valid_instances": valid_instances,

    })


# ============================================================
# RUN CLEANING
# ============================================================

print("=" * 70)
print("YOLO SEGMENTATION DATASET CLEANING")
print("=" * 70)

print(f"Original dataset: {SOURCE_ROOT}")

print(f"Clean dataset:    {OUTPUT_ROOT}")


for split in SPLITS:

    process_split(split)


# ============================================================
# CREATE CLEAN DATA.YAML
# ============================================================

data_yaml = """path: {dataset_path}
train: train/images
val: valid/images
test: test/images

nc: 6

names:
  0: Bad Welding
  1: Crack
  2: Excess Reinforcement
  3: Good Welding
  4: Porosity
  5: Spatters
""".format(

    dataset_path=OUTPUT_ROOT.as_posix()

)


(OUTPUT_ROOT / "data.yaml").write_text(

    data_yaml,

    encoding="utf-8",

)


# ============================================================
# SAVE REMOVAL REPORT
# ============================================================

removal_report_path = (

    REPORT_DIR / "removed_items.csv"

)


with removal_report_path.open(

    "w",

    newline="",

    encoding="utf-8",

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "split",
            "file",
            "reason",
        ],

    )

    writer.writeheader()

    writer.writerows(removed_records)


# ============================================================
# SAVE SUMMARY REPORT
# ============================================================

summary_report_path = (

    REPORT_DIR / "cleaning_summary.csv"

)


with summary_report_path.open(

    "w",

    newline="",

    encoding="utf-8",

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "split",
            "source_images",
            "source_labels",
            "matched_pairs",
            "images_without_labels",
            "labels_without_images",
            "removed_malformed_lines",
            "removed_unusable_pairs",
            "clean_images",
            "clean_labels",
            "valid_instances",
        ],

    )

    writer.writeheader()

    writer.writerows(summary_records)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("SEGMENTATION DATASET CLEANING COMPLETED")

print("=" * 70)

print(f"Clean dataset: {OUTPUT_ROOT}")

print(f"Clean data.yaml: {OUTPUT_ROOT / 'data.yaml'}")

print(f"Removal report: {removal_report_path}")

print(f"Summary report: {summary_report_path}")

print("\nOriginal dataset was NOT modified.")