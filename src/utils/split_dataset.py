import pandas as pd
from sklearn.model_selection import train_test_split


input_file = "data/raw/skin_dataset_clean.csv"

train_file = "data/raw/train.csv"
validation_file = "data/raw/validation.csv"
test_file = "data/raw/test.csv"


# Load clean dataset
df = pd.read_csv(input_file)

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["diagnosis_3"],
    random_state=42
)

# Second split: divide temporary set into 15% validation and 15% test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["diagnosis_3"],
    random_state=42
)


# Save the three datasets
train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)


print("Dataset split completed!")

print("\nTraining images:", len(train_df))
print("Validation images:", len(validation_df))
print("Test images:", len(test_df))

print("\nTotal:", len(train_df) + len(validation_df) + len(test_df))