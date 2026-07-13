from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    Path.home()
    / "runs"
    / "segment"
    / "runs"
    / "welding_segmentation"
    / "baseline_yolo11n_seg-2"
    / "weights"
    / "best.pt"
)


def parse_source(source):
    """
    Convert webcam numbers such as "0" into integer 0.
    Otherwise, return the video path as a string.
    """
    if source.isdigit():
        return int(source)

    return source


def main():

    parser = argparse.ArgumentParser(
        description="Real-Time Welding Defect Inspection using YOLO Segmentation"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file path or webcam index. Example: 0",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained YOLO segmentation model",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum YOLO detection confidence",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )

    args = parser.parse_args()

    source = parse_source(args.source)

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print("=" * 70)
    print("REAL-TIME WELDING DEFECT INSPECTION")
    print("=" * 70)

    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Image size: {args.imgsz}")

    print("\nLoading YOLO segmentation model...")

    model = YOLO(str(model_path))

    print("Model loaded successfully.")

    print("\nOpening video source...")

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"Video FPS: {fps:.2f}")
    print(f"Frame resolution: {width} x {height}")

    print("\nInspection started.")
    print("Press Q to stop.")

    frame_number = 0

    try:

        while True:

            success, frame = capture.read()

            if not success:
                print("\nEnd of video or frame capture failed.")
                break

            frame_number += 1

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=0,
                verbose=False,
            )

            result = results[0]

            annotated_frame = result.plot()

            detection_count = 0

            if result.boxes is not None:
                detection_count = len(result.boxes)

            cv2.putText(
                annotated_frame,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Detections: {detection_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Welding Quality Inspection - AC4",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nInspection stopped by operator.")
                break

    finally:

        capture.release()

        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("VIDEO INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()