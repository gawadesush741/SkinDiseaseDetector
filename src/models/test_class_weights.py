import sys
sys.path.append("src")

from models.label_map import LABEL_MAP
from models.class_weights import calculate_class_weights


weights = calculate_class_weights(
    "data/raw/train.csv",
    LABEL_MAP
)

print("Class weights:")

for name, label in LABEL_MAP.items():
    print(name, "=", weights[label].item())