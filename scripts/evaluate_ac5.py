from pathlib import Path

import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    Path.home()
    / "runs"
    / "segment"
    / "runs"
    / "welding_segmentation"
    / "baseline_yolo11n_seg-2"
    / "weights"
    / "best.pt"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "welding_segmentation_clean"
    / "test"
    / "images"
)

TEST_LABELS = (
    PROJECT_ROOT
    / "data"
    / "welding_segmentation_clean"
    / "test"
    / "labels"
)

# This file was created by tune_ac5_validation.py.
# It contains the threshold selected using ONLY validation data.

SELECTED_THRESHOLD_FILE = (
    PROJECT_ROOT
    / "results"
    / "ac5_validation_tuning"
    / "selected_ac5_threshold.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ac5_final_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CLASSES
# ============================================================

# 0 = Bad Welding
# 1 = Crack
# 2 = Excess Reinforcement
# 3 = Good Welding
# 4 = Porosity
# 5 = Spatters

GOOD_WELD_CLASS_ID = 3

DEFECT_CLASS_IDS = {
    0,
    1,
    2,
    4,
    5,
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# LOAD FROZEN VALIDATION-SELECTED THRESHOLD
# ============================================================


def load_frozen_threshold():

    if not SELECTED_THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            "Selected AC5 threshold file was not found:\n"
            f"{SELECTED_THRESHOLD_FILE}\n\n"
            "Run scripts/tune_ac5_validation.py first."
        )

    threshold_text = (
        SELECTED_THRESHOLD_FILE
        .read_text(encoding="utf-8")
        .strip()
    )

    try:

        threshold = float(threshold_text)

    except ValueError as error:

        raise ValueError(
            "The selected threshold file does not "
            "contain a valid number.\n"
            f"Found: {threshold_text}"
        ) from error

    if not 0.0 <= threshold <= 1.0:

        raise ValueError(
            "Threshold must be between 0 and 1.\n"
            f"Found: {threshold}"
        )

    return threshold


# ============================================================
# FIND TEST IMAGES
# ============================================================


def get_image_files():

    return sorted(
        [
            path
            for path in TEST_IMAGES.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


# ============================================================
# READ IMAGE-LEVEL GROUND TRUTH
# ============================================================


def read_ground_truth(label_path):

    """
    Convert YOLO segmentation annotations into
    image-level AC5 ground truth.

    Rule:

    If ANY defect annotation exists:
        DEFECT

    Otherwise, if Good Welding exists:
        GOOD_WELD

    Otherwise:
        UNKNOWN
    """

    if not label_path.exists():

        return "UNKNOWN"

    class_ids = []

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:

                continue

            values = line.split()

            try:

                class_id = int(
                    float(values[0])
                )

            except (ValueError, IndexError):

                continue

            class_ids.append(class_id)

    if any(
        class_id in DEFECT_CLASS_IDS
        for class_id in class_ids
    ):

        return "DEFECT"

    if GOOD_WELD_CLASS_ID in class_ids:

        return "GOOD_WELD"

    return "UNKNOWN"


# ============================================================
# EXTRACT MODEL CONFIDENCES
# ============================================================


def get_prediction_summary(result):

    """
    Extract:

    1. Highest-confidence Good Welding prediction.

    2. Highest-confidence defect prediction.

    3. Name of highest-confidence defect class.
    """

    best_good_confidence = 0.0

    best_defect_confidence = 0.0

    best_defect_class = None

    if result.boxes is None:

        return (
            best_good_confidence,
            best_defect_confidence,
            best_defect_class,
        )

    class_ids = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    for class_id, confidence in zip(
        class_ids,
        confidences,
    ):

        confidence = float(confidence)

        if class_id == GOOD_WELD_CLASS_ID:

            if confidence > best_good_confidence:

                best_good_confidence = confidence

        elif class_id in DEFECT_CLASS_IDS:

            if confidence > best_defect_confidence:

                best_defect_confidence = confidence

                best_defect_class = result.names[
                    class_id
                ]

    return (
        best_good_confidence,
        best_defect_confidence,
        best_defect_class,
    )


# ============================================================
# AC5 DECISION LOGIC
# ============================================================


def make_decision(
    best_good_confidence,
    best_defect_confidence,
    threshold,
):

    """
    AC5 safety-oriented decision logic.

    Priority:

    1. DEFECT prediction reaches threshold:
       DEFECT.

    2. Otherwise, Good Welding reaches threshold:
       GOOD_WELD.

    3. Otherwise:
       MANUAL_REVIEW.
    """

    if best_defect_confidence >= threshold:

        return (
            "DEFECT",
            best_defect_confidence,
        )

    if best_good_confidence >= threshold:

        return (
            "GOOD_WELD",
            best_good_confidence,
        )

    return (
        "MANUAL_REVIEW",
        max(
            best_good_confidence,
            best_defect_confidence,
        ),
    )


# ============================================================
# SAFE DIVISION
# ============================================================


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:

        return 0.0

    return numerator / denominator


# ============================================================
# CALCULATE FINAL TEST METRICS
# ============================================================


def calculate_final_metrics(dataframe):

    total_samples = len(dataframe)

    accepted = dataframe[
        dataframe["decision"]
        != "MANUAL_REVIEW"
    ]

    manual_review = dataframe[
        dataframe["decision"]
        == "MANUAL_REVIEW"
    ]

    correct_accepted = accepted[
        accepted["decision"]
        == accepted["ground_truth"]
    ]

    incorrect_accepted = accepted[
        accepted["decision"]
        != accepted["ground_truth"]
    ]

    defect_samples = dataframe[
        dataframe["ground_truth"]
        == "DEFECT"
    ]

    good_samples = dataframe[
        dataframe["ground_truth"]
        == "GOOD_WELD"
    ]

    correctly_accepted_defects = defect_samples[
        defect_samples["decision"]
        == "DEFECT"
    ]

    correctly_accepted_good = good_samples[
        good_samples["decision"]
        == "GOOD_WELD"
    ]

    dangerous_false_good = defect_samples[
        defect_samples["decision"]
        == "GOOD_WELD"
    ]

    false_defect = good_samples[
        good_samples["decision"]
        == "DEFECT"
    ]

    defect_manual_review = defect_samples[
        defect_samples["decision"]
        == "MANUAL_REVIEW"
    ]

    good_manual_review = good_samples[
        good_samples["decision"]
        == "MANUAL_REVIEW"
    ]

    coverage = safe_divide(
        len(accepted),
        total_samples,
    )

    manual_review_rate = safe_divide(
        len(manual_review),
        total_samples,
    )

    accepted_accuracy = safe_divide(
        len(correct_accepted),
        len(accepted),
    )

    defect_recall_all = safe_divide(
        len(correctly_accepted_defects),
        len(defect_samples),
    )

    good_weld_recall_all = safe_divide(
        len(correctly_accepted_good),
        len(good_samples),
    )

    return {

        "total_samples":
            total_samples,

        "accepted_samples":
            len(accepted),

        "manual_review_samples":
            len(manual_review),

        "coverage":
            coverage,

        "manual_review_rate":
            manual_review_rate,

        "correct_accepted":
            len(correct_accepted),

        "incorrect_accepted":
            len(incorrect_accepted),

        "accepted_accuracy":
            accepted_accuracy,

        "total_defect_samples":
            len(defect_samples),

        "correctly_accepted_defects":
            len(correctly_accepted_defects),

        "defect_recall_all":
            defect_recall_all,

        "defect_manual_review":
            len(defect_manual_review),

        "total_good_weld_samples":
            len(good_samples),

        "correctly_accepted_good_welds":
            len(correctly_accepted_good),

        "good_weld_recall_all":
            good_weld_recall_all,

        "good_weld_manual_review":
            len(good_manual_review),

        "dangerous_false_good":
            len(dangerous_false_good),

        "false_defect":
            len(false_defect),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 80)

    print("FINAL AC5 TEST EVALUATION")

    print("=" * 80)

    print(f"Model: {MODEL_PATH}")

    print(f"Test images: {TEST_IMAGES}")

    print(f"Test labels: {TEST_LABELS}")

    print(
        f"Frozen threshold file: "
        f"{SELECTED_THRESHOLD_FILE}"
    )


    # --------------------------------------------------------
    # VERIFY PATHS
    # --------------------------------------------------------


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )


    if not TEST_IMAGES.exists():

        raise FileNotFoundError(
            f"Test images not found:\n"
            f"{TEST_IMAGES}"
        )


    if not TEST_LABELS.exists():

        raise FileNotFoundError(
            f"Test labels not found:\n"
            f"{TEST_LABELS}"
        )


    # --------------------------------------------------------
    # LOAD FROZEN THRESHOLD
    # --------------------------------------------------------


    frozen_threshold = load_frozen_threshold()


    print("\n" + "=" * 80)

    print("FROZEN AC5 THRESHOLD")

    print("=" * 80)

    print(
        f"Threshold selected on validation set: "
        f"{frozen_threshold:.2f}"
    )

    print(
        "The test set will NOT be used "
        "to select or modify the threshold."
    )


    # --------------------------------------------------------
    # GET TEST IMAGES
    # --------------------------------------------------------


    image_files = get_image_files()


    print(
        f"\nNumber of test images: "
        f"{len(image_files)}"
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------


    print("\nLoading trained YOLO model...")


    model = YOLO(
        str(MODEL_PATH)
    )


    print("Model loaded successfully.")


    # --------------------------------------------------------
    # RUN TEST INFERENCE
    # --------------------------------------------------------


    print(
        "\nRunning inference once on "
        "the unseen test dataset..."
    )


    predictions = model.predict(

        source=str(TEST_IMAGES),

        # Use a low inference threshold to collect
        # predictions first.
        #
        # The validation-selected AC5 threshold
        # is applied afterwards.

        conf=0.001,

        imgsz=640,

        device=0,

        stream=True,

        verbose=False,
    )


    records = []


    for index, result in enumerate(
        predictions,
        start=1,
    ):


        image_path = Path(
            result.path
        )


        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )


        ground_truth = read_ground_truth(
            label_path
        )


        (
            best_good_confidence,
            best_defect_confidence,
            best_defect_class,

        ) = get_prediction_summary(
            result
        )


        (
            decision,
            decision_confidence,

        ) = make_decision(

            best_good_confidence,

            best_defect_confidence,

            frozen_threshold,
        )


        records.append(
            {

                "image_name":
                    image_path.name,

                "ground_truth":
                    ground_truth,

                "best_good_confidence":
                    best_good_confidence,

                "best_defect_confidence":
                    best_defect_confidence,

                "best_defect_class":
                    best_defect_class,

                "frozen_threshold":
                    frozen_threshold,

                "decision":
                    decision,

                "decision_confidence":
                    decision_confidence,
            }
        )


        if index % 25 == 0:

            print(
                f"Processed: "
                f"{index}/"
                f"{len(image_files)}"
            )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------


    dataframe = pd.DataFrame(
        records
    )


    unknown_count = (
        dataframe["ground_truth"]
        == "UNKNOWN"
    ).sum()


    if unknown_count > 0:

        print(
            f"\nWARNING: "
            f"{unknown_count} UNKNOWN samples "
            f"will be excluded."
        )


    dataframe = dataframe[
        dataframe["ground_truth"]
        != "UNKNOWN"
    ].copy()


    # --------------------------------------------------------
    # PRINT TEST DISTRIBUTION
    # --------------------------------------------------------


    print("\n" + "=" * 80)

    print("TEST GROUND-TRUTH DISTRIBUTION")

    print("=" * 80)


    print(
        dataframe[
            "ground_truth"
        ]
        .value_counts()
        .to_string()
    )


    # --------------------------------------------------------
    # CALCULATE FINAL METRICS
    # --------------------------------------------------------


    metrics = calculate_final_metrics(
        dataframe
    )


    # --------------------------------------------------------
    # SAVE FINAL REPORTS
    # --------------------------------------------------------


    detailed_csv = (
        OUTPUT_DIR
        / "ac5_final_test_predictions.csv"
    )


    summary_csv = (
        OUTPUT_DIR
        / "ac5_final_test_summary.csv"
    )


    dataframe.to_csv(
        detailed_csv,
        index=False,
    )


    summary_dataframe = pd.DataFrame(
        [
            {

                "frozen_threshold":
                    frozen_threshold,

                **metrics,
            }
        ]
    )


    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )


    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------


    print("\n" + "=" * 80)

    print(
        "FINAL AC5 TEST EVALUATION: "
        f"FROZEN THRESHOLD "
        f"{frozen_threshold:.2f}"
    )

    print("=" * 80)


    print(
        f"Total test samples:           "
        f"{metrics['total_samples']}"
    )


    print(
        f"Accepted samples:             "
        f"{metrics['accepted_samples']}"
    )


    print(
        f"Manual-review samples:        "
        f"{metrics['manual_review_samples']}"
    )


    print(
        f"Coverage:                     "
        f"{metrics['coverage'] * 100:.2f}%"
    )


    print(
        f"Manual-review rate:           "
        f"{metrics['manual_review_rate'] * 100:.2f}%"
    )


    print(
        f"Correct accepted decisions:   "
        f"{metrics['correct_accepted']}"
    )


    print(
        f"Incorrect accepted decisions: "
        f"{metrics['incorrect_accepted']}"
    )


    print(
        f"Accuracy when accepted:       "
        f"{metrics['accepted_accuracy'] * 100:.2f}%"
    )


    print("\nDEFECT PERFORMANCE")

    print("-" * 80)


    print(
        f"Total defect samples:         "
        f"{metrics['total_defect_samples']}"
    )


    print(
        f"Correctly accepted defects:   "
        f"{metrics['correctly_accepted_defects']}"
    )


    print(
        f"Defect recall overall:        "
        f"{metrics['defect_recall_all'] * 100:.2f}%"
    )


    print(
        f"Defects sent to review:       "
        f"{metrics['defect_manual_review']}"
    )


    print("\nGOOD-WELD PERFORMANCE")

    print("-" * 80)


    print(
        f"Total good-weld samples:      "
        f"{metrics['total_good_weld_samples']}"
    )


    print(
        f"Correctly accepted good welds:"
        f" {metrics['correctly_accepted_good_welds']}"
    )


    print(
        f"Good-weld recall overall:     "
        f"{metrics['good_weld_recall_all'] * 100:.2f}%"
    )


    print(
        f"Good welds sent to review:    "
        f"{metrics['good_weld_manual_review']}"
    )


    print("\nSAFETY-CRITICAL ERRORS")

    print("-" * 80)


    print(
        f"Dangerous false-good:         "
        f"{metrics['dangerous_false_good']}"
    )


    print(
        f"False-defect decisions:       "
        f"{metrics['false_defect']}"
    )


    print("\nReports saved to:")


    print(summary_csv)

    print(detailed_csv)


    print("\n" + "=" * 80)

    print("FINAL AC5 EVALUATION COMPLETED")

    print("=" * 80)


if __name__ == "__main__":

    main()