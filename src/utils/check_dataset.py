import os
import pandas as pd

# Where our images are
image_folder = "data/raw/images"

# Our metadata CSV
csv_file = "data/raw/ham10000_metadata_2026-10-01.csv"

# Read metadata
df = pd.read_csv(csv_file)

print("Number of metadata records:", len(df))

# Get image filenames
image_files = os.listdir(image_folder)

# Remove file extension
image_ids = {
    os.path.splitext(filename)[0]
    for filename in image_files
}

# Check how many metadata IDs have matching images
metadata_ids = set(df["isic_id"].astype(str))

matching_ids = metadata_ids.intersection(image_ids)

print("Number of image files:", len(image_ids))
print("Matching image IDs:", len(matching_ids))

print("\nExample matching IDs:")
for image_id in list(matching_ids)[:5]:
    print(image_id)