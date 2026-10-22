import pandas as pd
import torch


def calculate_class_weights(csv_file, label_map):

    df = pd.read_csv(csv_file)

    counts = df["diagnosis_3"].value_counts()

    weights = []

    for diagnosis, label in label_map.items():
        count = counts[diagnosis]

        weight = len(df) / (len(label_map) * count)

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)