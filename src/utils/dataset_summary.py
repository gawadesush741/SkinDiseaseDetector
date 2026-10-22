import pandas as pd

csv_file = "data/raw/ham10000_metadata_2026-10-01.csv"

df = pd.read_csv(csv_file)

print("Total images:", len(df))

print("\nDiagnosis counts:")
print(df["diagnosis_3"].value_counts())

print("\nMissing diagnosis values:")
print(df["diagnosis_3"].isna().sum())