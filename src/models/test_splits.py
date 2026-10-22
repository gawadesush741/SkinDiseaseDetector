import sys
sys.path.append("src")

from models.skin_dataset import SkinDataset
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms


train_dataset = SkinDataset(
    csv_file="data/raw/train.csv",
    image_folder="data/raw/images",
    label_map=LABEL_MAP,
    transform=image_transforms
)

validation_dataset = SkinDataset(
    csv_file="data/raw/validation.csv",
    image_folder="data/raw/images",
    label_map=LABEL_MAP,
    transform=image_transforms
)

test_dataset = SkinDataset(
    csv_file="data/raw/test.csv",
    image_folder="data/raw/images",
    label_map=LABEL_MAP,
    transform=image_transforms
)


print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))

print("\nLabel map:")
for name, number in LABEL_MAP.items():
    print(number, "=", name)


image, label = train_dataset[0]

print("\nTraining image loaded!")
print("Tensor shape:", image.shape)
print("Label:", label)