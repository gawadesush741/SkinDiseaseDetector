import sys
sys.path.append("src")
from skin_dataset import SkinDataset
from preprocessing.transforms import image_transforms


dataset = SkinDataset(
    csv_file="data/raw/skin_dataset_clean.csv",
    image_folder="data/raw/images",
    transform=image_transforms
)

print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.label_map))

print("\nClasses:")
for name, number in dataset.label_map.items():
    print(number, "=", name)


image, label = dataset[0]

print("\nFirst image loaded successfully!")
print("Image tensor shape:", image.shape)
print("Label number:", label)