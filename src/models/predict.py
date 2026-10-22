import torch
from PIL import Image
import sys
from pymongo import MongoClient

sys.path.append("src")

from models.skin_model import create_model
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms


# -----------------------------
# 1. Setup
# -----------------------------

device = torch.device("cpu")


# -----------------------------
# 2. Load trained AI model
# -----------------------------

model = create_model(num_classes=len(LABEL_MAP))

model.load_state_dict(
    torch.load(
        "skin_disease_model.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# 3. Reverse label map
# -----------------------------

reverse_label_map = {
    value: key
    for key, value in LABEL_MAP.items()
}


# -----------------------------
# 4. Load test image
# -----------------------------

image_path = "data/raw/test_images/test1.jpg"

image = Image.open(image_path).convert("RGB")

image_tensor = image_transforms(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)


# -----------------------------
# 5. AI prediction
# -----------------------------

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        1
    )


predicted_label = reverse_label_map[
    predicted_class.item()
]

confidence_percentage = confidence.item() * 100


# -----------------------------
# 6. Connect to MongoDB
# -----------------------------

client = MongoClient(
    "mongodb://localhost:27017"
)

db = client["skin_disease_ai"]

collection = db["diseases"]


# -----------------------------
# 7. Get disease information
# -----------------------------

disease_info = collection.find_one(
    {"name": predicted_label}
)


# -----------------------------
# 8. Display result
# -----------------------------

print("\n==============================")
print("      MODEL PREDICTION")
print("==============================")

print(
    "Predicted condition:",
    predicted_label
)

print(
    "Confidence:",
    round(confidence_percentage, 2),
    "%"
)

print("==============================")

if disease_info:

    print("\nDescription:")
    print(disease_info["description"])

    print("\nCauses / associations:")

    for cause in disease_info["causes"]:
        print("-", cause)

    print("\nGeneral care:")

    for care in disease_info["general_care"]:
        print("-", care)

    print("\nWhen to seek medical attention:")
    print(disease_info["medical_attention"])

else:

    print("\nNo information found in MongoDB.")


# Close MongoDB connection
client.close()