from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "Weld quality inspection - Segmentation"
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
# AUDIT FUNCTION
# ============================================================

def audit_split(split_name):

    print("\n" + "=" * 70)
    print(f"AUDITING SPLIT: {split_name.upper()}")
    print("=" * 70)

    images_dir = DATASET_ROOT / split_name / "images"
    labels_dir = DATASET_ROOT / split_name / "labels"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return None

    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return None

    image_files = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_files = list(labels_dir.glob("*.txt"))

    image_stems = {
        path.stem
        for path in image_files
    }

    label_stems = {
        path.stem
        for path in label_files
    }

    missing_labels = sorted(
        image_stems - label_stems
    )

    missing_images = sorted(
        label_stems - image_stems
    )

    corrupt_images = []

    empty_labels = []

    malformed_labels = []

    invalid_class_ids = []

    invalid_coordinates = []

    class_counts = Counter()

    total_instances = 0


    # ========================================================
    # VERIFY IMAGES
    # ========================================================

    print("\nVerifying image integrity...")

    for image_path in image_files:

        try:

            with Image.open(image_path) as image:

                image.verify()

        except Exception as error:

            corrupt_images.append(
                (
                    image_path.name,
                    str(error),
                )
            )


    # ========================================================
    # VERIFY LABELS
    # ========================================================

    print("Verifying segmentation labels...")

    for label_path in label_files:

        try:

            lines = label_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

        except Exception as error:

            malformed_labels.append(
                (
                    label_path.name,
                    0,
                    f"Could not read file: {error}",
                )
            )

            continue


        non_empty_lines = [

            line.strip()

            for line in lines

            if line.strip()

        ]


        if len(non_empty_lines) == 0:

            empty_labels.append(
                label_path.name
            )

            continue


        for line_number, line in enumerate(
            non_empty_lines,
            start=1,
        ):

            parts = line.split()


            # YOLO segmentation format:
            #
            # class_id x1 y1 x2 y2 x3 y3 ...
            #
            # Minimum polygon:
            # class_id + 3 points
            # = 1 class value + 6 coordinates
            # = 7 total values

            if len(parts) < 7:

                malformed_labels.append(
                    (
                        label_path.name,
                        line_number,
                        "Too few values for segmentation polygon",
                    )
                )

                continue


            if (len(parts) - 1) % 2 != 0:

                malformed_labels.append(
                    (
                        label_path.name,
                        line_number,
                        "Odd number of polygon coordinates",
                    )
                )

                continue


            # =================================================
            # CLASS ID
            # =================================================

            try:

                class_id = int(parts[0])

            except ValueError:

                malformed_labels.append(
                    (
                        label_path.name,
                        line_number,
                        "Class ID is not an integer",
                    )
                )

                continue


            if class_id not in CLASS_NAMES:

                invalid_class_ids.append(
                    (
                        label_path.name,
                        line_number,
                        class_id,
                    )
                )

                continue


            # =================================================
            # POLYGON COORDINATES
            # =================================================

            try:

                coordinates = [
                    float(value)
                    for value in parts[1:]
                ]

            except ValueError:

                malformed_labels.append(
                    (
                        label_path.name,
                        line_number,
                        "Polygon contains non-numeric coordinates",
                    )
                )

                continue


            bad_coordinates = [

                value

                for value in coordinates

                if value < 0.0 or value > 1.0

            ]


            if bad_coordinates:

                invalid_coordinates.append(
                    (
                        label_path.name,
                        line_number,
                        bad_coordinates,
                    )
                )

                continue


            class_counts[class_id] += 1

            total_instances += 1


    # ========================================================
    # RESULTS
    # ========================================================

    print("\nDATASET COUNTS")
    print("-" * 70)

    print(f"Images: {len(image_files)}")

    print(f"Label files: {len(label_files)}")

    print(f"Total valid instances: {total_instances}")


    print("\nIMAGE/LABEL MATCHING")
    print("-" * 70)

    print(f"Images without labels: {len(missing_labels)}")

    print(f"Labels without images: {len(missing_images)}")


    print("\nDATA QUALITY")
    print("-" * 70)

    print(f"Corrupt images: {len(corrupt_images)}")

    print(f"Empty label files: {len(empty_labels)}")

    print(f"Malformed labels: {len(malformed_labels)}")

    print(f"Invalid class IDs: {len(invalid_class_ids)}")

    print(
        f"Instances with invalid coordinates: "
        f"{len(invalid_coordinates)}"
    )


    print("\nCLASS DISTRIBUTION")
    print("-" * 70)

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counts.get(
            class_id,
            0,
        )

        print(
            f"{class_id}: "
            f"{class_name:<25} "
            f"{count}"
        )


    # ========================================================
    # PROBLEM DETAILS
    # ========================================================

    if missing_labels:

        print("\nFIRST IMAGES WITHOUT LABEL FILES")

        for name in missing_labels[:10]:

            print(name)


    if missing_images:

        print("\nFIRST LABELS WITHOUT IMAGE FILES")

        for name in missing_images[:10]:

            print(name)


    if corrupt_images:

        print("\nFIRST CORRUPT IMAGES")

        for item in corrupt_images[:10]:

            print(item)


    if malformed_labels:

        print("\nFIRST MALFORMED LABELS")

        for item in malformed_labels[:10]:

            print(item)


    if invalid_class_ids:

        print("\nFIRST INVALID CLASS IDs")

        for item in invalid_class_ids[:10]:

            print(item)


    if invalid_coordinates:

        print("\nFIRST INVALID COORDINATES")

        for item in invalid_coordinates[:10]:

            print(item)


    return {

        "images": len(image_files),

        "labels": len(label_files),

        "instances": total_instances,

        "class_counts": class_counts,

        "missing_labels": len(missing_labels),

        "missing_images": len(missing_images),

        "corrupt_images": len(corrupt_images),

        "empty_labels": len(empty_labels),

        "malformed_labels": len(malformed_labels),

        "invalid_class_ids": len(invalid_class_ids),

        "invalid_coordinates": len(invalid_coordinates),

    }


# ============================================================
# RUN AUDIT
# ============================================================

print("=" * 70)
print("YOLO SEGMENTATION DATASET AUDIT")
print("=" * 70)

print(f"Dataset: {DATASET_ROOT}")


all_results = {}


for split in SPLITS:

    result = audit_split(split)

    if result is not None:

        all_results[split] = result


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)


total_images = sum(

    result["images"]

    for result in all_results.values()

)


total_labels = sum(

    result["labels"]

    for result in all_results.values()

)


total_instances = sum(

    result["instances"]

    for result in all_results.values()

)


combined_class_counts = Counter()


for result in all_results.values():

    combined_class_counts.update(
        result["class_counts"]
    )


print(f"Total images: {total_images}")

print(f"Total label files: {total_labels}")

print(f"Total valid instances: {total_instances}")


print("\nOVERALL CLASS DISTRIBUTION")

print("-" * 70)


for class_id, class_name in CLASS_NAMES.items():

    count = combined_class_counts.get(
        class_id,
        0,
    )

    print(
        f"{class_id}: "
        f"{class_name:<25} "
        f"{count}"
    )


total_problems = sum(

    result["missing_labels"]
    + result["missing_images"]
    + result["corrupt_images"]
    + result["malformed_labels"]
    + result["invalid_class_ids"]
    + result["invalid_coordinates"]

    for result in all_results.values()

)


print("\n" + "=" * 70)


if total_problems == 0:

    print("SEGMENTATION DATASET AUDIT SUCCESSFUL")

else:

    print("SEGMENTATION DATASET AUDIT FOUND PROBLEMS")

    print(f"Total detected problems: {total_problems}")


print("=" * 70)