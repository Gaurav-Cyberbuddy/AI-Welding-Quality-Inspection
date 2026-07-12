from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ds_meta.parquet"
)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_parquet(METADATA_PATH)


print("=" * 60)
print("DATASET AUDIT")
print("=" * 60)


# ============================================================
# 1. DATASET SHAPE
# ============================================================

print("\n1. DATASET SHAPE")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. COLUMNS AND DATA TYPES
# ============================================================

print("\n2. COLUMNS AND DATA TYPES")

print(df.dtypes)


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\n3. CLASS DISTRIBUTION")

class_counts = df["class"].value_counts(dropna=False)

print(class_counts)


print("\nClass percentages:")

class_percentages = (
    df["class"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print(class_percentages)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n4. MISSING VALUES")

missing_values = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing_values = missing_values[missing_values > 0]


if missing_values.empty:

    print("No missing values found.")

else:

    print(missing_values)


# ============================================================
# 5. DUPLICATE SAMPLE IDs
# ============================================================

print("\n5. DUPLICATE SAMPLE IDs")

duplicate_sample_ids = df["sample_id"].duplicated().sum()

print(f"Duplicate sample IDs: {duplicate_sample_ids}")


# ============================================================
# 6. DUPLICATE IMAGE HASHES
# ============================================================

print("\n6. DUPLICATE IMAGE HASHES")


def make_hashable(value):

    if hasattr(value, "tolist"):

        value = value.tolist()

    if isinstance(value, list):

        return tuple(value)

    return value


hashable_sha256 = df["sha256"].apply(make_hashable)

duplicate_hashes = hashable_sha256.duplicated().sum()

print(f"Duplicate SHA256 values: {duplicate_hashes}")


# ============================================================
# 7. CLASS CONFLICTS FOR DUPLICATE HASHES
# ============================================================

print("\n7. DUPLICATE HASH CLASS CONFLICTS")

hash_class_df = pd.DataFrame({

    "sha256_hashable": hashable_sha256,

    "class": df["class"]

})


hash_class_counts = (

    hash_class_df

    .groupby("sha256_hashable")["class"]

    .nunique()

)


conflicting_hashes = (

    hash_class_counts > 1

).sum()


print(

    f"Image hashes assigned to multiple classes: "
    f"{conflicting_hashes}"

)


# ============================================================
# 8. CATEGORICAL COLUMN SUMMARY
# ============================================================

print("\n8. CATEGORICAL COLUMN SUMMARY")


categorical_columns = [

    "class",

    "welding-seams",

    "labelling_type",

    "storage_type",

    "data_origin",

    "blur_class"

]


for column in categorical_columns:

    if column in df.columns:

        print(f"\n--- {column} ---")

        print(

            df[column]

            .value_counts(dropna=False)

            .head(30)

        )


# ============================================================
# 9. RESOLUTION SUMMARY
# ============================================================

print("\n9. RESOLUTION SUMMARY")


if "resolution" in df.columns:

    resolution_values = (

        df["resolution"]

        .apply(make_hashable)

        .value_counts(dropna=False)

    )

    print(resolution_values.head(30))


# ============================================================
# 10. NUMERICAL COLUMN SUMMARY
# ============================================================

print("\n10. NUMERICAL COLUMN SUMMARY")


numerical_columns = [

    "blur_level",

    "luminosity_level"

]


existing_numerical_columns = [

    column

    for column in numerical_columns

    if column in df.columns

]


print(

    df[existing_numerical_columns]

    .describe()

    .T

)


# ============================================================
# 11. PATH INFORMATION
# ============================================================

print("\n11. PATH INFORMATION")


for column in ["path", "external_path"]:

    if column in df.columns:

        print(f"\n{column}")

        print(

            f"Missing values: "
            f"{df[column].isnull().sum()}"

        )

        print(

            f"Unique values: "
            f"{df[column].nunique()}"

        )


# ============================================================
# 12. SAMPLE RECORDS
# ============================================================

print("\n12. SAMPLE RECORDS")


display_columns = [

    "sample_id",

    "class",

    "welding-seams",

    "resolution",

    "blur_level",

    "blur_class",

    "luminosity_level",

    "data_origin"

]


existing_display_columns = [

    column

    for column in display_columns

    if column in df.columns

]


print(

    df[existing_display_columns]

    .head(10)

    .to_string(index=False)

)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("AUDIT SUMMARY")

print("=" * 60)


print(f"Total samples: {len(df)}")

print(f"OK samples: {(df['class'] == 'OK').sum()}")

print(f"KO samples: {(df['class'] == 'KO').sum()}")

print(

    f"KO percentage: "

    f"{((df['class'] == 'KO').mean() * 100):.2f}%"

)

print(f"Duplicate sample IDs: {duplicate_sample_ids}")

print(f"Duplicate image hashes: {duplicate_hashes}")

print(f"Class-conflicting image hashes: {conflicting_hashes}")


print("\nDATASET AUDIT COMPLETED SUCCESSFULLY")