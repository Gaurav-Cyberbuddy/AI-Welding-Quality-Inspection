# AI-Based Welding Quality Inspection System

An end-to-end deep learning and computer vision project for automated welding quality inspection using **ResNet-18** for binary weld-quality classification and **YOLO11 Instance Segmentation** for welding-defect localization.

The project addresses severe class imbalance, defect detection, instance segmentation, confidence-threshold analysis, selective automated inspection, manual-review fallback, error analysis, subgroup analysis, and single-image/video inference.

## Project Overview

Manual welding inspection can be time-consuming and dependent on operator experience. This project explores the use of deep learning and computer vision to develop an automated welding quality inspection pipeline.

The system contains three main components:

1. **AC1 - Binary Welding Quality Classification**
   - Determines whether a welding image is `OK` or `KO`.
   - Uses a pretrained ResNet-18 model.
   - Handles severe class imbalance using `WeightedRandomSampler`.
   - Evaluates defect-detection performance using precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrices.

2. **AC4 - Welding Defect Localization and Segmentation**
   - Uses YOLO11 instance segmentation.
   - Detects welding-related classes.
   - Produces bounding boxes, segmentation masks, class labels, and confidence scores.
   - Supports image and video inference.

3. **AC5 - Confidence-Based Inspection Decision System**
   - Converts segmentation predictions into image-level inspection decisions.
   - Produces `DEFECT`, `GOOD_WELD`, or `MANUAL_REVIEW`.
   - Selects the operating confidence threshold using the validation dataset.
   - Freezes the selected threshold before final test-set evaluation.

## Project Workflow

```text
Classification Dataset
        |
        v
Dataset Audit and Verification
        |
        v
Train / Validation / Test Splitting
        |
        v
PyTorch Dataset and DataLoader
        |
        v
Baseline ResNet-18 Training
        |
        v
Baseline Test Evaluation
        |
        v
Threshold Analysis
        |
        v
Class-Imbalance Handling
        |
        v
Balanced ResNet-18 Training
        |
        v
Balanced Model Test Evaluation
        |
        v
Error Analysis and Subgroup Analysis
        |
        v
Single-Image Inference
        |
        v
AC1 Binary Weld-Quality Decision


Segmentation Dataset
        |
        v
Dataset Audit
        |
        v
Dataset Cleaning
        |
        v
YOLO11 Segmentation Training
        |
        v
Validation and Test Evaluation
        |
        v
Defect Localization and Segmentation
        |
        v
Prediction Visualization
        |
        v
AC4 Completed
        |
        v
Confidence Threshold Analysis
        |
        v
AC5 Validation Threshold Tuning
        |
        v
Freeze Selected Threshold
        |
        v
Final AC5 Test Evaluation
        |
        v
DEFECT / GOOD_WELD / MANUAL_REVIEW
```

## Dataset

### Binary Welding Quality Classification Dataset

The classification dataset contains **22,753 welding images**.

The binary labels are:

- `OK` - Acceptable welding sample.
- `KO` - Defective welding sample.

Dataset split:

| Split | Number of Images |
|---|---:|
| Training | 15,927 |
| Validation | 3,413 |
| Test | 3,413 |
| Total | 22,753 |

The training dataset is severely imbalanced:

| Class | Training Samples |
|---|---:|
| OK | 15,581 |
| KO | 346 |

Only approximately **2.17%** of the training samples belong to the defective `KO` class.

### Welding Instance Segmentation Dataset

A separate instance-segmentation dataset is used for welding-defect localization.

The segmentation classes are:

- Bad Welding
- Crack
- Excess Reinforcement
- Good Welding
- Porosity
- Spatters

The original segmentation dataset was audited and cleaned before model training.

The cleaning pipeline:

- Preserves the original dataset.
- Removes malformed annotation lines.
- Excludes unmatched image-label files.
- Preserves valid segmentation polygons.
- Generates cleaning reports.
- Creates a clean YOLO `data.yaml`.

After cleaning:

- 1,384 images were preserved.
- 27,636 valid segmentation instances were preserved.

The cleaned dataset was divided into training, validation, and test sets for YOLO11 instance-segmentation experiments.

## Classification Dataset Verification

Before model training, the local classification dataset was verified.

```text
Metadata records: 22753
Valid images: 22753
Missing images: 0
Corrupt images: 0
Metadata/folder mismatches: 0
```

The verification process confirmed that the local classification dataset was complete and consistent.

## PyTorch Data Pipeline

The reusable PyTorch dataset implementation is located in:

```text
src/dataset.py
```

The DataLoader pipeline can be verified using:

```bash
python scripts/test_dataloader.py
```

Example training batch:

```text
Image batch shape: [32, 3, 224, 224]
Label batch shape: [32]

OK -> 0
KO -> 1
```

## AC1 - Binary Welding Quality Classification

### Baseline ResNet-18

A pretrained ResNet-18 model was used as the baseline binary classifier.

The final fully connected layer was modified for two-class classification.

```text
Input Welding Image
        |
        v
Image Preprocessing
        |
        v
ResNet-18 Feature Extraction
        |
        v
512-Dimensional Feature Vector
        |
        v
Fully Connected Layer
        |
        v
OK / KO Prediction
```

### Baseline Test Results

The baseline confusion matrix was:

```text
[[3282, 57],
 [  43, 31]]
```

Baseline performance:

| Metric | Result |
|---|---:|
| Test Accuracy | 97.01% |
| KO Precision | 35.23% |
| KO Recall | 41.89% |
| KO F1-Score | 38.27% |

The baseline model achieved high overall accuracy but performed poorly on the minority `KO` class.

This demonstrated why accuracy alone is insufficient for evaluating a severely imbalanced welding-quality dataset.

### Threshold Analysis

Classification thresholds were evaluated using validation data.

A threshold of `0.70` was evaluated on the test set, but increasing the threshold reduced KO recall.

This demonstrated that threshold adjustment alone was insufficient to solve the class-imbalance problem.

### Handling Class Imbalance

The improved classifier uses PyTorch's `WeightedRandomSampler`.

Original training distribution:

```text
OK = 15,581
KO = 346
```

Weighted sampling increases the frequency with which rare KO samples are selected during training without deleting majority-class samples or generating synthetic images.

### Final Balanced ResNet-18 Test Results

The balanced model was evaluated on the untouched test dataset containing **3,413 images**.

Confusion matrix:

```text
[[3306, 33],
 [  22, 52]]
```

Interpretation:

```text
True Negatives  (OK correctly predicted): 3306
False Positives (OK predicted as KO):        33
False Negatives (KO predicted as OK):        22
True Positives  (KO correctly predicted):    52
```

Final performance:

| Metric | Result |
|---|---:|
| Test Accuracy | 98.39% |
| OK Precision | 99.34% |
| OK Recall | 99.01% |
| OK F1-Score | 99.18% |
| KO Precision | 61.18% |
| KO Recall | 70.27% |
| KO F1-Score | 65.41% |
| Macro F1-Score | 82.29% |
| PR-AUC | 0.6548 |
| ROC-AUC | 0.9352 |

Compared with the baseline classifier, the balanced model substantially improved defective-weld detection.

KO recall increased from **41.89% to 70.27%**, while KO F1-score increased from **38.27% to 65.41%**.

## Error Analysis

The balanced classifier was further analyzed using:

```text
scripts/error_analysis.py
```

The analysis includes:

- False positives.
- False negatives.
- Welding seam.
- Labelling type.
- Blur class.
- Blur level.
- Luminosity level.
- Most confident incorrect predictions.

The generated reports are stored under:

```text
results/error_analysis/
```

## Subgroup Performance Analysis

The classifier was evaluated across dataset subgroups using:

```text
scripts/subgroup_analysis.py
```

Analyzed groups include:

### Welding Seam

- `c20`
- `c33`
- `c102`

### Labelling Source

- Expert
- Operator

### Image Quality

- Clean
- Blur

Subgroup analysis helps identify model weaknesses that may not be visible from aggregate metrics.

## Single-Image Classification Inference

The project provides:

```text
scripts/predict_image.py
```

Example usage:

```bash
python scripts/predict_image.py "path/to/welding_image.jpeg"
```

The script outputs:

- Predicted class.
- OK probability.
- KO probability.

## AC4 - YOLO11 Welding Defect Localization

Binary classification determines whether an entire welding image is acceptable or defective but does not identify the exact location of a welding defect.

The project therefore uses YOLO11 instance segmentation for AC4.

```text
Welding Image
        |
        v
YOLO11 Segmentation Model
        |
        v
Detected Welding Class
        +
Bounding Region
        +
Segmentation Mask
        +
Confidence Score
```

### Segmentation Dataset Audit

The segmentation dataset is audited using:

```text
scripts/audit_segmentation_dataset.py
```

The audit checks:

- Image and label counts.
- Missing image-label pairs.
- Corrupt images.
- Empty label files.
- Malformed segmentation polygons.
- Invalid class IDs.
- Invalid normalized coordinates.
- Class distribution.

### Segmentation Dataset Cleaning

The cleaning pipeline is implemented in:

```text
scripts/clean_segmentation_dataset.py
```

The script preserves the original dataset and creates a separate cleaned YOLO dataset.

Cleaning reports are stored under:

```text
results/segmentation_cleaning/
```

### YOLO11 Segmentation Model

A YOLO11n segmentation model was trained on the cleaned welding segmentation dataset.

The best-performing training checkpoint is stored as:

```text
best.pt
```

The model was evaluated on the segmentation test set containing:

```text
89 images
522 annotated instances
```

### AC4 Test Results

Overall test-set performance:

| Metric | Result |
|---|---:|
| Box Precision | 0.476 |
| Box Recall | 0.496 |
| Box mAP50 | 0.450 |
| Box mAP50-95 | 0.227 |
| Mask Precision | 0.446 |
| Mask Recall | 0.476 |
| Mask mAP50 | 0.412 |
| Mask mAP50-95 | 0.174 |

Per-class segmentation performance:

| Class | Images | Instances | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Bad Welding | 7 | 8 | 0.575 | 0.849 | 0.711 | 0.298 |
| Crack | 41 | 49 | 0.538 | 0.551 | 0.529 | 0.182 |
| Excess Reinforcement | 7 | 22 | 0.208 | 0.091 | 0.085 | 0.054 |
| Good Welding | 21 | 22 | 0.373 | 0.455 | 0.385 | 0.208 |
| Porosity | 13 | 119 | 0.416 | 0.597 | 0.386 | 0.167 |
| Spatters | 10 | 302 | 0.565 | 0.315 | 0.379 | 0.137 |

The model successfully generates welding-class predictions, localization regions, segmentation masks, and confidence scores.

AC4 is implemented and evaluated, although segmentation performance can be improved with additional high-quality annotated data and further model optimization.

## Image and Video Segmentation Inference

YOLO prediction was run on the segmentation test images.

The generated outputs contain:

- Predicted welding classes.
- Confidence scores.
- Bounding regions.
- Segmentation masks.

A video inspection script is also included:

```text
scripts/video_inspection.py
```

The script supports camera or video-file inference when a valid video source is available.

Real-time physical-camera validation was not performed as part of the final evaluation because a camera/video source was not available during the experiment.

## AC5 - Confidence-Based Inspection System

AC5 extends the segmentation pipeline with confidence-based image-level decision logic.

The system produces one of three decisions:

```text
DEFECT
GOOD_WELD
MANUAL_REVIEW
```

The segmentation classes are converted into two image-level categories:

```text
Good Welding
        |
        v
GOOD_WELD


Bad Welding
Crack
Excess Reinforcement
Porosity
Spatters
        |
        v
DEFECT
```

The AC5 safety-oriented decision rule gives priority to defect predictions.

```text
YOLO Predictions
        |
        v
Highest Defect Confidence
Highest Good-Weld Confidence
        |
        v
Apply Defined Confidence Threshold
        |
        +-----------------------------+
        |                             |
        v                             v
Confidence Meets Threshold     Confidence Below Threshold
        |                             |
        v                             v
DEFECT / GOOD_WELD              MANUAL_REVIEW
```

## YOLO Confidence Threshold Analysis

Segmentation confidence thresholds were first analyzed on the validation dataset.

The best Mask F1-score among the evaluated thresholds was:

```text
Threshold = 0.10

Mask Precision = 0.4740
Mask Recall    = 0.4804
Mask F1        = 0.4772
```

A reference threshold of `0.90` was also evaluated:

```text
Mask Precision = 0.6250
Mask Recall    = 0.0239
Mask F1        = 0.0460
```

This showed that a very high confidence threshold caused extremely low segmentation recall.

The threshold-analysis results are stored under:

```text
results/yolo_confidence_tuning/
```

## AC5 Validation Threshold Selection

The final AC5 threshold was selected using the validation dataset and image-level `GOOD_WELD`, `DEFECT`, and `MANUAL_REVIEW` decisions.

The selection policy required:

1. Zero dangerous false-good decisions.
2. At least 90% accuracy among automatically accepted samples.
3. Maximum validation coverage among thresholds satisfying the first two conditions.

The selected threshold was:

```text
Frozen AC5 Threshold = 0.85
```

Validation performance at threshold `0.85`:

| Metric | Result |
|---|---:|
| Validation Samples | 188 |
| Accepted Samples | 21 |
| Manual-Review Samples | 167 |
| Coverage | 11.17% |
| Accepted Accuracy | 100.00% |
| Defect Recall Overall | 12.67% |
| Good-Weld Recall Overall | 5.26% |
| Dangerous False-Good Decisions | 0 |
| False-Defect Decisions | 0 |

The selected threshold is saved in:

```text
results/ac5_validation_tuning/selected_ac5_threshold.txt
```

## Final AC5 Test Evaluation

The threshold selected on the validation dataset was frozen before final test evaluation.

The final AC5 evaluation used the unseen segmentation test dataset.

Final test results:

| Metric | Result |
|---|---:|
| Frozen Threshold | 0.85 |
| Total Test Samples | 89 |
| Accepted Samples | 6 |
| Manual-Review Samples | 83 |
| Coverage | 6.74% |
| Manual-Review Rate | 93.26% |
| Correct Accepted Decisions | 6 |
| Incorrect Accepted Decisions | 0 |
| Accuracy When Accepted | 100.00% |
| Total Defect Samples | 68 |
| Correctly Accepted Defects | 6 |
| Defect Recall Overall | 8.82% |
| Total Good-Weld Samples | 21 |
| Correctly Accepted Good Welds | 0 |
| Good-Weld Recall Overall | 0.00% |
| Dangerous False-Good Decisions | 0 |
| False-Defect Decisions | 0 |

The final AC5 system made no incorrect automatically accepted decisions on the test set.

However, the system is highly conservative. Only **6.74%** of test images were automatically accepted, while **93.26%** were sent for manual review.

Therefore, AC5 demonstrates a working confidence-based selective inspection pipeline with a manual-review fallback, but further model improvement and confidence calibration are required to increase automation coverage.

## Result Visualization

The project generates:

- Baseline confusion matrix.
- Balanced-model confusion matrix.
- Baseline vs balanced model comparison.
- Balanced-model ROC curve.
- Balanced-model Precision-Recall curve.

The figures are stored under:

```text
results/figures/
```

The result generation pipeline is implemented in:

```text
scripts/generate_results.py
```

## Project Structure

```text
AI-Welding-Quality-Inspection/
|
|-- data/
|   |-- metadata/
|   |-- splits/
|   |-- verification_reports/
|   |-- welding-detection-challenge-dataset/
|   |-- Weld quality inspection - Segmentation/
|   `-- welding_segmentation_clean/
|
|-- models/
|   |-- baseline_resnet18.pth
|   `-- balanced_resnet18.pth
|
|-- results/
|   |-- error_analysis/
|   |-- figures/
|   |-- segmentation_cleaning/
|   |-- yolo_confidence_tuning/
|   |-- ac5_evaluation/
|   |-- ac5_validation_tuning/
|   `-- ac5_final_evaluation/
|
|-- scripts/
|   |-- dataset_audit.py
|   |-- download_images.py
|   |-- test_download.py
|   |-- verify_local_dataset.py
|   |-- test_dataloader.py
|   |-- train_baseline.py
|   |-- evaluate_baseline.py
|   |-- tune_threshold.py
|   |-- evaluate_threshold.py
|   |-- train_balanced.py
|   |-- evaluate_balanced.py
|   |-- error_analysis.py
|   |-- subgroup_analysis.py
|   |-- predict_image.py
|   |-- generate_results.py
|   |-- audit_segmentation_dataset.py
|   |-- clean_segmentation_dataset.py
|   |-- tune_yolo_confidence.py
|   |-- tune_ac5_validation.py
|   |-- evaluate_ac5.py
|   `-- video_inspection.py
|
|-- src/
|   `-- dataset.py
|
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Large datasets, trained model checkpoints, YOLO training runs, and local ZIP archives may be excluded from Git version control using `.gitignore`.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- Ultralytics YOLO11
- OpenCV
- Pillow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- CUDA

## Acceptance Criteria Status

| Acceptance Criterion | Implementation | Status |
|---|---|---|
| AC1 - Binary weld-quality decision | Balanced ResNet-18 classifier evaluated on untouched test data | Completed |
| AC4 - Defect localization and segmentation | YOLO11 instance segmentation with masks, classes, confidence scores, test evaluation, and prediction visualization | Completed |
| AC5 - Confidence-based inspection decision | Validation-selected frozen threshold with `DEFECT`, `GOOD_WELD`, and `MANUAL_REVIEW` decisions | Completed with limited automation coverage |

## Key Findings

1. High overall classification accuracy can hide poor minority-class defect detection.

2. Threshold tuning alone does not solve severe class imbalance.

3. Weighted sampling substantially improved KO recall and F1-score compared with the baseline classifier.

4. Error analysis and subgroup evaluation are important for identifying weaknesses hidden by aggregate metrics.

5. Binary classification can determine overall weld quality but cannot localize defects.

6. YOLO11 instance segmentation provides defect localization, masks, class predictions, and confidence scores.

7. Segmentation dataset auditing and cleaning are important because malformed annotations can negatively affect model training and evaluation.

8. Very high confidence thresholds significantly reduce model recall and automation coverage.

9. Confidence thresholds should be selected using validation data and frozen before final test evaluation.

10. A manual-review fallback can prevent low-confidence predictions from being automatically accepted, although the current AC5 system requires further improvement to increase automation coverage.

## Current Project Status

### Completed

- Classification dataset audit.
- Local dataset verification.
- Train/validation/test splitting.
- PyTorch Dataset and DataLoader pipeline.
- Baseline ResNet-18 training.
- Baseline test evaluation.
- Classification threshold analysis.
- Class-imbalance handling.
- Balanced ResNet-18 training.
- Balanced-model test evaluation.
- Error analysis.
- Subgroup analysis.
- Single-image classification inference.
- Final classification result visualization.
- Segmentation dataset audit.
- Segmentation dataset cleaning.
- YOLO11 segmentation training.
- YOLO11 test evaluation.
- Segmentation prediction visualization.
- Image segmentation inference.
- Video inspection pipeline implementation.
- YOLO confidence-threshold analysis.
- AC5 image-level decision logic.
- AC5 validation threshold selection.
- Frozen-threshold final AC5 test evaluation.
- AC1 implementation and evaluation.
- AC4 implementation and evaluation.
- AC5 implementation and evaluation.

## Limitations

- The binary classification dataset is severely imbalanced.
- The KO class contains substantially fewer samples than the OK class.
- The segmentation dataset is relatively small.
- Some segmentation classes contain limited training and test samples.
- Segmentation performance varies significantly across defect classes.
- Excess Reinforcement has particularly weak segmentation performance.
- The AC5 confidence-based system has low automation coverage.
- At the frozen threshold of `0.85`, 93.26% of test images require manual review.
- No good-weld test images were automatically accepted by the final AC5 system.
- Real-time physical-camera validation was not completed.
- Additional high-quality welding data and annotation improvements are required for stronger deployment performance.

## Future Improvements

- Collect additional high-quality defective-weld samples.
- Improve minority-class representation in the classification dataset.
- Increase the size and annotation quality of the segmentation dataset.
- Apply stronger and domain-specific data augmentation.
- Investigate focal loss and other imbalance-aware loss functions.
- Compare larger YOLO11 segmentation models.
- Tune YOLO training hyperparameters.
- Improve segmentation performance for underperforming classes.
- Investigate confidence calibration methods.
- Increase AC5 automation coverage while maintaining low safety-critical error rates.
- Evaluate the system using additional independent test datasets.
- Validate the video inspection pipeline using real welding video streams.
- Develop an operator-facing inspection dashboard.
- Optimize inference performance for real-time deployment.
- Export the final model for deployment.
