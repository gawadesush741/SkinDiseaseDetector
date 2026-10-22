import pandas as pd

input_file = "data/raw/ham10000_metadata_2026-10-01.csv"
output_file = "data/raw/skin_dataset_clean.csv"

# Read metadata
df = pd.read_csv(input_file)

# Remove records without a diagnosis
clean_df = df.dropna(subset=["diagnosis_3"]).copy()

# Keep the columns we need
clean_df = clean_df[
    [
        "isic_id",
        "diagnosis_3",
        "age_approx",
        "anatom_site_1",
        "sex"
    ]
]

# Save cleaned dataset
clean_df.to_csv(output_file, index=False)

print("Original records:", len(df))
print("Clean records:", len(clean_df))
print("Removed records:", len(df) - len(clean_df))

print("\nClean dataset saved to:")
print(output_file)