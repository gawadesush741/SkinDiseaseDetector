from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

from PIL import Image

import torch
import io
import sys
from pathlib import Path

from pymongo import MongoClient


# ==========================================
# FIND PROJECT FILES
# ==========================================

sys.path.append("src")

from models.skin_model import create_model
from models.label_map import LABEL_MAP
from preprocessing.transforms import image_transforms


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Skin Disease AI",
    description="AI-based skin lesion classification prototype",
    version="1.0.0"
)


# ==========================================
# LOAD AI MODEL
# ==========================================

device = torch.device("cpu")

model = create_model(
    num_classes=len(LABEL_MAP),
    pretrained=False
)

model_path = Path(__file__).resolve().parents[1] / "skin_disease_model.pth"

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()


# Convert class numbers to disease names

reverse_label_map = {
    value: key
    for key, value in LABEL_MAP.items()
}


# ==========================================
# CONNECT TO MONGODB
# ==========================================

client = MongoClient(
    "mongodb://localhost:27017"
)

db = client["skin_disease_ai"]

disease_collection = db["diseases"]


# ==========================================
# PREDICT API
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Preprocess image
    image_tensor = image_transforms(image)

    image_tensor = image_tensor.unsqueeze(0)

    # AI prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            1
        )

    # Get disease name
    predicted_label = reverse_label_map[
        predicted_class.item()
    ]

    confidence_percentage = (
        confidence.item() * 100
    )

    # Get information from MongoDB
    disease_info = disease_collection.find_one(
        {
            "name": predicted_label
        },
        {
            "_id": 0
        }
    )

    # Return result
    if disease_info:

        return {
            "prediction": predicted_label,

            "confidence": round(
                confidence_percentage,
                2
            ),

            "description": disease_info.get(
                "description",
                ""
            ),

            "causes": disease_info.get(
                "causes",
                []
            ),

            "general_care": disease_info.get(
                "general_care",
                []
            ),

            "medical_attention": disease_info.get(
                "medical_attention",
                ""
            )
        }

    # Fallback
    return {
        "prediction": predicted_label,

        "confidence": round(
            confidence_percentage,
            2
        ),

        "description": "",

        "causes": [],

        "general_care": [],

        "medical_attention":
            "Please consult a dermatologist "
            "for professional evaluation."
    }


# ==========================================
# SERVE FRONTEND
# ==========================================

app.mount(
    "/",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)