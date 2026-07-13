from pathlib import Path

import pandas as pd
from ultralytics import YOLO


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

VALID_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "welding_segmentation_clean"
    / "valid"
    / "images"
)

VALID_LABELS = (
    PROJECT_ROOT
    / "data"
    / "welding_segmentation_clean"
    / "valid"
    / "labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ac5_validation_tuning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Dataset classes:
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


# AC5 decision thresholds to test.
THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_image_files():

    return sorted(
        [
            path
            for path in VALID_IMAGES.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def read_ground_truth(label_path):

    """
    Convert segmentation annotations into
    image-level AC5 ground truth.

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
                class_id = int(float(values[0]))

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


def get_prediction_summary(result):

    """
    Find:

    highest-confidence Good Welding prediction

    highest-confidence defect prediction
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


def make_decision(
    best_good_confidence,
    best_defect_confidence,
    threshold,
):

    """
    AC5 safety-oriented decision rule.

    Priority:

    1. DEFECT prediction meeting threshold.
    2. GOOD_WELD prediction meeting threshold.
    3. Otherwise MANUAL_REVIEW.
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


def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(dataframe, threshold):

    total = len(dataframe)

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

    coverage = safe_divide(
        len(accepted),
        total,
    )

    accepted_accuracy = safe_divide(
        len(correct_accepted),
        len(accepted),
    )

    defect_recall = safe_divide(
        len(correctly_accepted_defects),
        len(defect_samples),
    )

    good_weld_recall = safe_divide(
        len(correctly_accepted_good),
        len(good_samples),
    )

    return {

        "threshold": threshold,

        "total_samples": total,

        "accepted_samples": len(accepted),

        "manual_review_samples": len(
            manual_review
        ),

        "coverage": coverage,

        "accepted_accuracy":
            accepted_accuracy,

        "defect_recall_all":
            defect_recall,

        "good_weld_recall_all":
            good_weld_recall,

        "dangerous_false_good":
            len(dangerous_false_good),

        "false_defect":
            len(false_defect),
    }


def main():

    print("=" * 80)
    print("AC5 VALIDATION THRESHOLD TUNING")
    print("=" * 80)

    print(f"Model: {MODEL_PATH}")
    print(f"Validation images: {VALID_IMAGES}")
    print(f"Validation labels: {VALID_LABELS}")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VALID_IMAGES.exists():

        raise FileNotFoundError(
            f"Validation images not found:\n"
            f"{VALID_IMAGES}"
        )

    if not VALID_LABELS.exists():

        raise FileNotFoundError(
            f"Validation labels not found:\n"
            f"{VALID_LABELS}"
        )

    image_files = get_image_files()

    print(
        f"Validation images found: "
        f"{len(image_files)}"
    )

    print("\nLoading YOLO model...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("Model loaded successfully.")

    print(
        "\nRunning inference once on "
        "validation dataset..."
    )

    predictions = model.predict(
        source=str(VALID_IMAGES),
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
            VALID_LABELS
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
            }
        )

        if index % 25 == 0:

            print(
                f"Processed: "
                f"{index}/"
                f"{len(image_files)}"
            )

    base_dataframe = pd.DataFrame(
        records
    )

    unknown_count = (
        base_dataframe[
            "ground_truth"
        ]
        == "UNKNOWN"
    ).sum()

    if unknown_count > 0:

        print(
            f"\nWARNING: "
            f"{unknown_count} UNKNOWN samples "
            f"will be excluded."
        )

    base_dataframe = base_dataframe[
        base_dataframe["ground_truth"]
        != "UNKNOWN"
    ].copy()

    print("\n" + "=" * 80)
    print("VALIDATION GROUND-TRUTH DISTRIBUTION")
    print("=" * 80)

    print(
        base_dataframe[
            "ground_truth"
        ]
        .value_counts()
        .to_string()
    )

    summary_records = []

    detailed_records = []

    for threshold in THRESHOLDS:

        threshold_dataframe = (
            base_dataframe.copy()
        )

        decisions = []

        decision_confidences = []

        for _, row in (
            threshold_dataframe.iterrows()
        ):

            (
                decision,
                decision_confidence,

            ) = make_decision(

                row[
                    "best_good_confidence"
                ],

                row[
                    "best_defect_confidence"
                ],

                threshold,
            )

            decisions.append(
                decision
            )

            decision_confidences.append(
                decision_confidence
            )

        threshold_dataframe[
            "decision"
        ] = decisions

        threshold_dataframe[
            "decision_confidence"
        ] = decision_confidences

        threshold_dataframe[
            "threshold"
        ] = threshold

        metrics = calculate_metrics(
            threshold_dataframe,
            threshold,
        )

        summary_records.append(
            metrics
        )

        detailed_records.append(
            threshold_dataframe
        )

    summary_dataframe = pd.DataFrame(
        summary_records
    )

    details_dataframe = pd.concat(
        detailed_records,
        ignore_index=True,
    )

    summary_csv = (
        OUTPUT_DIR
        / "ac5_validation_threshold_summary.csv"
    )

    details_csv = (
        OUTPUT_DIR
        / "ac5_validation_detailed_predictions.csv"
    )

    summary_dataframe.to_csv(
        summary_csv,
        index=False,
    )

    details_dataframe.to_csv(
        details_csv,
        index=False,
    )

    print("\n" + "=" * 80)
    print("AC5 VALIDATION THRESHOLD SUMMARY")
    print("=" * 80)

    display_columns = [

        "threshold",

        "accepted_samples",

        "manual_review_samples",

        "coverage",

        "accepted_accuracy",

        "defect_recall_all",

        "good_weld_recall_all",

        "dangerous_false_good",

        "false_defect",
    ]

    print(
        summary_dataframe[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # SELECT THRESHOLD
    # -------------------------------------------------
    #
    # Selection policy:
    #
    # 1. Zero dangerous false-good decisions.
    # 2. At least 90% accuracy on accepted samples.
    # 3. Among eligible thresholds, maximize coverage.
    #
    # If no threshold satisfies the policy,
    # choose the threshold with:
    #
    # fewest dangerous false-good decisions,
    # then highest accepted accuracy,
    # then highest coverage.
    # -------------------------------------------------

    eligible = summary_dataframe[
        (
            summary_dataframe[
                "dangerous_false_good"
            ]
            == 0
        )
        &
        (
            summary_dataframe[
                "accepted_accuracy"
            ]
            >= 0.90
        )
    ].copy()

    if len(eligible) > 0:

        selected_row = (
            eligible
            .sort_values(
                by=[
                    "coverage",
                    "defect_recall_all",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        selection_reason = (
            "Zero dangerous false-good decisions, "
            "accepted accuracy >= 90%, "
            "maximum validation coverage."
        )

    else:

        selected_row = (
            summary_dataframe
            .sort_values(
                by=[
                    "dangerous_false_good",
                    "accepted_accuracy",
                    "coverage",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        selection_reason = (
            "No threshold satisfied the primary "
            "selection policy. Selected the safest "
            "available fallback threshold."
        )

    selected_threshold = float(
        selected_row["threshold"]
    )

    selected_threshold_file = (
        OUTPUT_DIR
        / "selected_ac5_threshold.txt"
    )

    with open(
        selected_threshold_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"{selected_threshold:.2f}\n"
        )

    print("\n" + "=" * 80)
    print("SELECTED AC5 VALIDATION THRESHOLD")
    print("=" * 80)

    print(
        f"Threshold:               "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Coverage:                "
        f"{selected_row['coverage'] * 100:.2f}%"
    )

    print(
        f"Accepted accuracy:       "
        f"{selected_row['accepted_accuracy'] * 100:.2f}%"
    )

    print(
        f"Defect recall overall:   "
        f"{selected_row['defect_recall_all'] * 100:.2f}%"
    )

    print(
        f"Good-weld recall overall:"
        f" {selected_row['good_weld_recall_all'] * 100:.2f}%"
    )

    print(
        f"Dangerous false-good:    "
        f"{int(selected_row['dangerous_false_good'])}"
    )

    print(
        f"False-defect decisions:  "
        f"{int(selected_row['false_defect'])}"
    )

    print(
        f"\nSelection reason:\n"
        f"{selection_reason}"
    )

    print("\nReports saved to:")

    print(summary_csv)

    print(details_csv)

    print(selected_threshold_file)

    print("\n" + "=" * 80)
    print("AC5 VALIDATION TUNING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()