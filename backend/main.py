from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
import io
import os
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

cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
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

mongo_uri = os.getenv("MONGODB_URI")
disease_collection = None

if mongo_uri:
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["skin_disease_ai"]
        disease_collection = db["diseases"]
    except Exception as e:
        print(f"Warning: Failed to connect to MongoDB: {e}")
        disease_collection = None


# ==========================================
# FALLBACK DISEASE KNOWLEDGE BASE
# ==========================================

DEFAULT_DISEASE_INFO = {
    "Nevus": {
        "description": "A nevus, commonly called a mole, is a common skin growth composed of melanocytes.",
        "causes": [
            "Genetic factors",
            "Normal development of pigment-producing cells",
            "Sun exposure can influence the number and appearance of moles"
        ],
        "general_care": [
            "Protect the skin from excessive sun exposure",
            "Monitor moles for changes in size, shape, or color",
            "Avoid picking or irritating a mole"
        ],
        "medical_attention": "A changing, itching, bleeding, or unusual mole should be examined by a dermatologist."
    },
    "Melanoma, NOS": {
        "description": "Melanoma is a type of skin cancer involving pigment-producing cells.",
        "causes": [
            "UV radiation exposure",
            "Previous sun damage",
            "Genetic and other risk factors"
        ],
        "general_care": [
            "Protect skin from UV exposure",
            "Avoid tanning beds",
            "Monitor skin for changing or unusual spots"
        ],
        "medical_attention": "A suspicious or changing pigmented lesion should be evaluated promptly by a dermatologist."
    },
    "Basal cell carcinoma": {
        "description": "Basal cell carcinoma is a common type of skin cancer that usually grows slowly.",
        "causes": [
            "Long-term UV exposure",
            "Sun damage",
            "Indoor tanning exposure"
        ],
        "general_care": [
            "Protect skin from excessive UV exposure",
            "Use sun protection",
            "Have suspicious or persistent skin growths examined"
        ],
        "medical_attention": "A sore that does not heal, repeatedly bleeds, or a suspicious new growth should be examined by a dermatologist."
    },
    "Squamous cell carcinoma, NOS": {
        "description": "Squamous cell carcinoma is a type of skin cancer that develops from squamous cells.",
        "causes": [
            "UV radiation exposure",
            "Cumulative sun damage",
            "Previous skin damage and certain risk factors"
        ],
        "general_care": [
            "Protect skin from UV exposure",
            "Avoid tanning beds",
            "Monitor persistent or changing skin lesions"
        ],
        "medical_attention": "A persistent, growing, bleeding, crusting, or non-healing lesion should be evaluated by a dermatologist."
    },
    "Solar or actinic keratosis": {
        "description": "Actinic keratosis is a precancerous skin growth associated with long-term UV damage.",
        "causes": [
            "Long-term sun exposure",
            "UV radiation",
            "Indoor tanning exposure"
        ],
        "general_care": [
            "Protect the skin from UV exposure",
            "Use sun protection",
            "Monitor rough or changing skin spots"
        ],
        "medical_attention": "Actinic keratoses should be evaluated by a healthcare professional because some can progress to squamous cell carcinoma."
    },
    "Pigmented benign keratosis": {
        "description": "A pigmented benign keratosis is a non-cancerous pigmented skin lesion.",
        "causes": [
            "Age-related skin changes",
            "Genetic factors",
            "Other factors affecting skin growth"
        ],
        "general_care": [
            "Protect skin from excessive sun exposure",
            "Avoid scratching or picking the lesion",
            "Monitor for noticeable changes"
        ],
        "medical_attention": "A lesion that changes significantly, bleeds, or becomes symptomatic should be examined by a dermatologist."
    },
    "Dermatofibroma": {
        "description": "A dermatofibroma is a common benign skin growth.",
        "causes": [
            "The exact cause is not always known",
            "Some may develop after minor skin injury or insect bites"
        ],
        "general_care": [
            "Avoid repeatedly irritating or scratching the lesion",
            "Monitor for significant changes"
        ],
        "medical_attention": "A rapidly changing, painful, bleeding, or unusual lesion should be evaluated by a dermatologist."
    }
}


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

    # Get information from MongoDB or fallback knowledge base
    disease_info = None

    if disease_collection is not None:
        try:
            disease_info = disease_collection.find_one(
                {
                    "name": predicted_label
                },
                {
                    "_id": 0
                }
            )
        except Exception as e:
            print(f"Warning: MongoDB lookup failed: {e}")
            disease_info = None

    if not disease_info:
        disease_info = DEFAULT_DISEASE_INFO.get(predicted_label, {})

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
            "Please consult a dermatologist for professional evaluation."
        )
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
