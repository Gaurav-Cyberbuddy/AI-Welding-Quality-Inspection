from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "error_analysis"
    / "validation_predictions.csv"
)

df = pd.read_csv(PREDICTIONS)

print("=" * 70)
print("SUBGROUP PERFORMANCE ANALYSIS")
print("=" * 70)


def analyze(column):

    print(f"\n{column.upper()}")
    print("-" * 70)

    summary = []

    for value in sorted(df[column].unique()):

        subset = df[df[column] == value]

        total = len(subset)

        correct = subset["correct"].sum()

        accuracy = correct / total * 100

        ko = subset[subset["true_label"] == 1]

        if len(ko) > 0:

            recall = (
                (ko["predicted_label"] == 1).sum()
                / len(ko)
            )

        else:

            recall = float("nan")


        summary.append({

            column: value,

            "Samples": total,

            "Accuracy (%)": round(accuracy, 2),

            "KO Recall (%)": round(recall * 100, 2)

            if pd.notna(recall)
            else None

        })

    print(pd.DataFrame(summary).to_string(index=False))


analyze("welding-seams")

analyze("labelling_type")

analyze("blur_class")

print("\n" + "=" * 70)
print("SUBGROUP ANALYSIS COMPLETED")
print("=" * 70)