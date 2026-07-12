# AI-Based Welding Quality Inspection System

An AI-based computer vision system for automated welding quality inspection using **ResNet-18** and **YOLO11 Segmentation**.

The project focuses on weld quality classification, handling severe class imbalance, model evaluation, error analysis, subgroup performance analysis, single-image inference, and defect localization using instance segmentation.

## Project Overview

Manual welding inspection can be time-consuming and dependent on operator experience. This project explores the use of deep learning and computer vision to build an automated welding inspection pipeline.

The development is divided into two main stages:

1. **Binary Welding Quality Classification**
   - Classifies welding images as `OK` or `KO`.
   - Uses a pretrained ResNet-18 model.
   - Handles severe class imbalance using weighted sampling.
   - Performs threshold tuning, model evaluation, error analysis, and subgroup analysis.

2. **Welding Defect Localization**
   - Uses YOLO11 instance segmentation.
   - Detects and highlights welding defects.
   - Produces defect classes, segmentation masks, and confidence scores.
   - YOLO segmentation training and evaluation are currently in progress.

## Project Workflow

```text
Dataset Collection
        ↓
Dataset Audit
        ↓
Local Dataset Verification
        ↓
PyTorch Dataset and DataLoader
        ↓
Baseline ResNet-18 Training
        ↓
Baseline Evaluation
        ↓
Threshold Tuning
        ↓
Class Imbalance Handling
        ↓
Balanced ResNet-18 Training
        ↓
Balanced Model Evaluation
        ↓
Error Analysis
        ↓
Subgroup Analysis
        ↓
Single-Image Inference
        ↓
Result Visualization
        ↓
Requirement Gap Analysis
        ↓
YOLO Segmentation Dataset Audit
        ↓
Segmentation Dataset Cleaning
        ↓
YOLO11 Segmentation Training
        ↓
YOLO Evaluation
        ↓
Confidence-Based Inspection
        ↓
Real-Time Video Inspection UI
```

## Dataset

### Welding Quality Detection Dataset

The first dataset contains **22,753 welding images**.

The binary classification labels are:

- `OK` - Acceptable welding sample
- `KO` - Defective welding sample

Dataset split:

| Split | Number of Images |
|---|---:|
| Training | 15,927 |
| Validation | 3,413 |
| Test | 3,413 |
| Total | 22,753 |

The training dataset is highly imbalanced:

| Class | Training Samples |
|---|---:|
| OK | 15,581 |
| KO | 346 |

Only approximately **2.17% of the training samples belong to the KO class**.

### Welding Instance Segmentation Dataset

A separate instance-segmentation dataset is used for defect localization experiments.

The available classes are:

- Bad Welding
- Crack
- Excess Reinforcement
- Good Welding
- Porosity
- Spatters

The dataset was audited before training. Malformed annotations and unmatched label files were detected.

A cleaning pipeline was developed to create a separate cleaned dataset without modifying the original data.

After cleaning:

- 1,384 images were preserved.
- 27,636 valid segmentation instances were preserved.
- Invalid segmentation annotation lines were removed.
- Unmatched label files were excluded from the cleaned dataset.

## Dataset Verification

Before model training, the local classification dataset was verified.

Results:

```text
Metadata records: 22753
Valid images: 22753
Missing images: 0
Corrupt images: 0
Metadata/folder mismatches: 0
```

This confirmed that the classification dataset was complete and consistent.

## PyTorch Data Pipeline

A reusable PyTorch dataset pipeline was implemented in:

```text
src/dataset.py
```

The DataLoader pipeline was verified using:

```text
scripts/test_dataloader.py
```

Example training batch:

```text
Image batch shape: [32, 3, 224, 224]
Label batch shape: [32]

OK -> 0
KO -> 1
```

## Baseline ResNet-18 Model

A pretrained ResNet-18 model was used as the initial baseline.

The final fully connected layer was modified for binary classification:

```text
Input Welding Image
        ↓
Image Preprocessing
        ↓
ResNet-18 Feature Extraction
        ↓
512-Dimensional Feature Vector
        ↓
Fully Connected Layer
        ↓
OK / KO Prediction
```

## Baseline Model Results

The baseline model achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | 97.07% |
| KO Precision | 35.23% |
| KO Recall | 41.89% |
| KO F1-Score | 38.27% |
| PR-AUC | 0.2996 |
| ROC-AUC | 0.8807 |

Although overall accuracy was high, the model showed poor performance in detecting the minority KO class.

This demonstrated that accuracy alone was not an appropriate metric for evaluating the highly imbalanced dataset.

## Threshold Tuning

Different classification thresholds were evaluated using the validation dataset.

The best validation KO F1-score was observed around:

```text
Threshold = 0.70
```

However, applying this threshold to the unseen test dataset reduced KO recall.

This experiment demonstrated that threshold adjustment alone was insufficient to solve the underlying class-imbalance problem.

## Handling Class Imbalance

The improved model used PyTorch's `WeightedRandomSampler`.

Original training distribution:

```text
OK = 15,581
KO = 346
```

During balanced training, the model was exposed to approximately equal numbers of OK and KO samples per epoch.

The technique increased the frequency with which rare KO samples were selected without deleting majority-class samples or generating synthetic images.

## Balanced ResNet-18 Results

The balanced model achieved:

| Metric | Baseline | Balanced Model |
|---|---:|---:|
| Test Accuracy | 97.07% | 98.95% |
| KO Precision | 35.23% | 77.94% |
| KO Recall | 41.89% | 71.62% |
| KO F1-Score | 38.27% | 74.65% |
| PR-AUC | 0.2996 | 0.7195 |
| ROC-AUC | 0.8807 | 0.9356 |

The balanced training strategy significantly improved defective-weld detection.

## Error Analysis

The trained balanced model was further analyzed using:

```text
scripts/error_analysis.py
```

The analysis included:

- False positives
- False negatives
- Welding seam
- Labelling type
- Blur class
- Blur level
- Luminosity level
- Most confident incorrect predictions

This analysis helps identify systematic weaknesses of the trained model.

## Subgroup Performance Analysis

Model performance was evaluated across different dataset subgroups using:

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

This analysis demonstrates that high overall performance does not necessarily imply equal performance under every operating condition.

## Single-Image Inference

A prediction script was developed for individual welding images:

```text
scripts/predict_image.py
```

Example usage:

```bash
python scripts/predict_image.py "path/to/welding_image.jpeg"
```

The script outputs:

- Predicted class
- OK probability
- KO probability

## Result Visualization

The project generates visual results for model analysis and documentation.

Generated results include:

- Confusion matrices
- ROC curves
- Precision-Recall curves
- Baseline vs balanced model comparisons

The result generation pipeline is implemented in:

```text
scripts/generate_results.py
```

## YOLO11 Instance Segmentation

Image classification can determine whether an entire welding image is acceptable or defective, but it cannot identify the exact location of a defect.

Therefore, the project was extended using YOLO11 instance segmentation.

YOLO segmentation provides:

```text
Welding Image / Video Frame
            ↓
YOLO11 Segmentation Model
            ↓
Defect Classification
            +
Defect Localization
            +
Segmentation Mask
            +
Confidence Score
```

## Segmentation Dataset Audit

Before YOLO training, the segmentation dataset was analyzed using:

```text
scripts/audit_segmentation_dataset.py
```

The audit checks:

- Image and label counts
- Missing image-label pairs
- Corrupt images
- Empty label files
- Malformed segmentation polygons
- Invalid class IDs
- Invalid normalized coordinates
- Class distribution

## Segmentation Dataset Cleaning

The cleaning pipeline is implemented in:

```text
scripts/clean_segmentation_dataset.py
```

The script:

- Preserves the original dataset.
- Creates a separate cleaned dataset.
- Removes malformed annotation lines.
- Excludes unmatched files.
- Preserves valid segmentation polygons.
- Generates cleaning reports.
- Generates a clean YOLO `data.yaml`.

## Project Structure

```text
AI-Welding-Quality-Inspection/
│
├── data/
│   ├── metadata/
│   ├── splits/
│   ├── verification_reports/
│   ├── welding-detection-challenge-dataset/
│   ├── Weld quality inspection - Segmentation/
│   └── welding_segmentation_clean/
│
├── models/
│   ├── baseline_resnet18.pth
│   └── balanced_resnet18.pth
│
├── results/
│   ├── error_analysis/
│   └── segmentation_cleaning/
│
├── scripts/
│   ├── dataset_audit.py
│   ├── download_images.py
│   ├── test_download.py
│   ├── verify_local_dataset.py
│   ├── test_dataloader.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── tune_threshold.py
│   ├── evaluate_threshold.py
│   ├── train_balanced.py
│   ├── evaluate_balanced.py
│   ├── error_analysis.py
│   ├── subgroup_analysis.py
│   ├── predict_image.py
│   ├── generate_results.py
│   ├── audit_segmentation_dataset.py
│   └── clean_segmentation_dataset.py
│
├── src/
│   └── dataset.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

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

## Current Status

### Completed

- Classification dataset audit
- Local dataset verification
- PyTorch Dataset and DataLoader pipeline
- Baseline ResNet-18 training
- Baseline evaluation
- Threshold tuning
- Class-imbalance handling
- Balanced ResNet-18 training
- Balanced model evaluation
- Error analysis
- Subgroup analysis
- Single-image inference
- Result visualization
- YOLO segmentation environment setup
- Segmentation dataset audit
- Segmentation dataset cleaning

### In Progress

- YOLO11 segmentation baseline training and evaluation

### Planned Work

- Improve YOLO segmentation performance.
- Obtain or annotate data for all exact required welding defect classes.
- Implement confidence-based inspection logic.
- Develop real-time video inference.
- Build an inspection interface for operator verification.
- Perform final end-to-end system evaluation.

## Acceptance Criteria Status

| Acceptance Criterion | Current Status |
|---|---|
| Multi-class welding defect classification | Partially completed; exact required classes are not yet fully covered by the available datasets |
| Defect highlighting on image/video feed | YOLO segmentation pipeline in progress |
| Confidence-based prediction logic | Planned after segmentation model evaluation |

## Key Findings

1. High classification accuracy can hide poor minority-class defect detection.

2. Threshold tuning alone does not solve severe dataset imbalance.

3. Balanced sampling significantly improved KO precision, recall, F1-score, PR-AUC, and ROC-AUC.

4. Error analysis and subgroup evaluation are necessary to identify model weaknesses that aggregate metrics do not reveal.

5. Classification alone cannot satisfy defect-localization requirements, motivating the extension to YOLO instance segmentation.

6. Dataset auditing and cleaning are essential before segmentation model training.
## Future Improvements
- Acquire correctly annotated datasets for the exact required defect classes.
- Investigate data augmentation and imbalance-aware training strategies for YOLO.
- Compare different YOLO model sizes.
- Perform confidence-threshold calibration.
- Add real-time webcam and video inference.
- Develop an operator-facing inspection dashboard.
- Export the final model for deployment.

