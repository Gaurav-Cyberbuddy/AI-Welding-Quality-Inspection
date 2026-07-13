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

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "welding_segmentation_clean"
    / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "yolo_confidence_tuning"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]


def safe_f1(precision, recall):

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def main():

    print("=" * 80)
    print("YOLO SEGMENTATION CONFIDENCE THRESHOLD TUNING")
    print("=" * 80)

    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print("Split: validation")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found:\n{DATA_YAML}"
        )

    print("\nLoading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    results_summary = []

    for threshold in THRESHOLDS:

        print("\n" + "=" * 80)
        print(
            f"EVALUATING CONFIDENCE THRESHOLD: "
            f"{threshold:.2f}"
        )
        print("=" * 80)

        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            conf=threshold,
            imgsz=640,
            device=0,
            plots=False,
            save_json=False,
            verbose=False,
        )

        box_precision = float(
            metrics.box.mp
        )

        box_recall = float(
            metrics.box.mr
        )

        box_map50 = float(
            metrics.box.map50
        )

        box_map5095 = float(
            metrics.box.map
        )

        mask_precision = float(
            metrics.seg.mp
        )

        mask_recall = float(
            metrics.seg.mr
        )

        mask_map50 = float(
            metrics.seg.map50
        )

        mask_map5095 = float(
            metrics.seg.map
        )

        box_f1 = safe_f1(
            box_precision,
            box_recall,
        )

        mask_f1 = safe_f1(
            mask_precision,
            mask_recall,
        )

        results_summary.append(
            {
                "threshold": threshold,

                "box_precision": box_precision,
                "box_recall": box_recall,
                "box_f1": box_f1,
                "box_map50": box_map50,
                "box_map50_95": box_map5095,

                "mask_precision": mask_precision,
                "mask_recall": mask_recall,
                "mask_f1": mask_f1,
                "mask_map50": mask_map50,
                "mask_map50_95": mask_map5095,
            }
        )

        print("\nRESULTS")

        print("-" * 80)

        print(
            f"Box Precision:      "
            f"{box_precision:.4f}"
        )

        print(
            f"Box Recall:         "
            f"{box_recall:.4f}"
        )

        print(
            f"Box F1:             "
            f"{box_f1:.4f}"
        )

        print(
            f"Mask Precision:     "
            f"{mask_precision:.4f}"
        )

        print(
            f"Mask Recall:        "
            f"{mask_recall:.4f}"
        )

        print(
            f"Mask F1:            "
            f"{mask_f1:.4f}"
        )

    dataframe = pd.DataFrame(
        results_summary
    )

    output_csv = (
        OUTPUT_DIR
        / "validation_threshold_results.csv"
    )

    dataframe.to_csv(
        output_csv,
        index=False,
    )

    best_index = dataframe[
        "mask_f1"
    ].idxmax()

    best_result = dataframe.loc[
        best_index
    ]

    print("\n" + "=" * 80)
    print("CONFIDENCE THRESHOLD SUMMARY")
    print("=" * 80)

    display_columns = [
        "threshold",
        "box_precision",
        "box_recall",
        "box_f1",
        "mask_precision",
        "mask_recall",
        "mask_f1",
    ]

    print(
        dataframe[
            display_columns
        ].round(4).to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("BEST THRESHOLD BY MASK F1")
    print("=" * 80)

    print(
        f"Threshold:       "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Mask Precision:  "
        f"{best_result['mask_precision']:.4f}"
    )

    print(
        f"Mask Recall:     "
        f"{best_result['mask_recall']:.4f}"
    )

    print(
        f"Mask F1:         "
        f"{best_result['mask_f1']:.4f}"
    )

    print("\n" + "=" * 80)
    print("AC5 REFERENCE THRESHOLD")
    print("=" * 80)

    threshold_90 = dataframe[
        dataframe["threshold"] == 0.90
    ].iloc[0]

    print(
        f"At confidence threshold 0.90:"
    )

    print(
        f"Mask Precision: "
        f"{threshold_90['mask_precision']:.4f}"
    )

    print(
        f"Mask Recall:    "
        f"{threshold_90['mask_recall']:.4f}"
    )

    print(
        f"Mask F1:        "
        f"{threshold_90['mask_f1']:.4f}"
    )

    print("\nResults saved to:")

    print(output_csv)

    print("\n" + "=" * 80)
    print("YOLO CONFIDENCE TUNING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()