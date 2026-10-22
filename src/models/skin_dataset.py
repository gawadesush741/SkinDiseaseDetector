import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SkinDataset(Dataset):

    def __init__(self, csv_file, image_folder, label_map, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_id = row["isic_id"]
        diagnosis = row["diagnosis_3"]

        image_path = os.path.join(
            self.image_folder,
            image_id + ".jpg"
        )

        image = Image.open(image_path).convert("RGB")

        label = self.label_map[diagnosis]

        if self.transform:
            image = self.transform(image)

        return image, label