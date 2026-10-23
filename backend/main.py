from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
import numpy as np

# Decompression bomb prevention & upload constraints
Image.MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
CONFIDENCE_THRESHOLD = 45.0  # Alert if top prediction is below 45%


def validate_image_quality(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Heuristic check to reject blank, solid-color, or completely dark/flat images
    before passing them to the AI model.
    """
    thumb = np.array(pil_image.resize((64, 64)))
    std_dev = float(np.std(thumb))
    if std_dev < 15.0:
        return False, "The uploaded image appears blank, solid-color, or severely under-exposed. Please upload a clear, focused photo of the skin lesion."
    return True, ""

import torch
import io
import os
import sys
from pathlib import Path

from pymongo import MongoClient


# ==========================================
# FIND PROJECT FILES
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

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
# PREDICT API & INFERENCE HELPERS
# ==========================================

def run_model_inference(pil_image: Image.Image):
    """
    Synchronous inference worker function.
    Executed in a background thread to prevent blocking FastAPI's async event loop.
    """
    image_tensor = image_transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

        top_k = min(3, len(reverse_label_map))
        top_confidences, top_classes = torch.topk(probabilities, top_k, dim=1)

    top_predictions = [
        {
            "disease": reverse_label_map[top_classes[0][i].item()],
            "confidence": round(top_confidences[0][i].item() * 100, 2)
        }
        for i in range(top_k)
    ]

    return top_predictions


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Validate file size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file exceeds the maximum allowed limit of 10 MB."
        )

    # Validate image decoding
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Please upload a clear JPG, PNG, or WEBP image."
        )

    # Validate image quality (reject completely blank, uniform, or blacked-out files)
    is_valid_quality, quality_err = validate_image_quality(image)
    if not is_valid_quality:
        raise HTTPException(
            status_code=400,
            detail=quality_err
        )

    # Run non-blocking inference in threadpool
    top_predictions = await run_in_threadpool(run_model_inference, image)

    top_result = top_predictions[0]
    predicted_label = top_result["disease"]
    confidence_percentage = top_result["confidence"]
    is_inconclusive = confidence_percentage < CONFIDENCE_THRESHOLD

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

    inconclusive_notice = None
    if is_inconclusive:
        inconclusive_notice = (
            f"The AI model identified {predicted_label} with low confidence ({confidence_percentage}%). "
            "The image may be blurry, have harsh lighting, or not show distinctive skin lesion patterns. "
            "Please consult a dermatologist for a professional clinical examination."
        )

    return {
        "prediction": predicted_label,
        "confidence": confidence_percentage,
        "is_inconclusive": is_inconclusive,
        "inconclusive_notice": inconclusive_notice,
        "top_predictions": top_predictions,
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
# MULTI-PAGE & CATALOG API ENDPOINTS
# ==========================================

@app.get("/api/diseases")
async def get_diseases_catalog():
    """Returns dynamic catalog of all 7 skin diseases for the encyclopedia."""
    if disease_collection is not None:
        try:
            records = list(disease_collection.find({}, {"_id": 0}))
            if records:
                return {"diseases": records}
        except Exception as e:
            print(f"MongoDB catalog lookup failed: {e}")

    category_map = {
        "Melanoma, NOS": {"category": "malignant", "risk": "High (Malignant)", "color": "#ef4444"},
        "Basal cell carcinoma": {"category": "malignant", "risk": "Moderate-High (Malignant)", "color": "#ef4444"},
        "Squamous cell carcinoma, NOS": {"category": "malignant", "risk": "High (Malignant)", "color": "#ef4444"},
        "Solar or actinic keratosis": {"category": "precancerous", "risk": "Pre-cancerous", "color": "#f59e0b"},
        "Nevus": {"category": "benign", "risk": "Benign (Non-cancerous)", "color": "#10b981"},
        "Pigmented benign keratosis": {"category": "benign", "risk": "Benign (Non-cancerous)", "color": "#10b981"},
        "Dermatofibroma": {"category": "benign", "risk": "Benign (Non-cancerous)", "color": "#10b981"}
    }

    catalog = []
    for name, info in DEFAULT_DISEASE_INFO.items():
        cat = category_map.get(name, {"category": "benign", "risk": "Benign", "color": "#10b981"})
        catalog.append({
            "name": name,
            "category": cat["category"],
            "risk": cat["risk"],
            "color": cat["color"],
            "description": info.get("description", ""),
            "causes": info.get("causes", []),
            "general_care": info.get("general_care", []),
            "medical_attention": info.get("medical_attention", "")
        })

    return {"diseases": catalog}


frontend_dir = BASE_DIR / "frontend"

@app.api_route("/scan", methods=["GET", "HEAD"])
async def get_scan_page():
    return FileResponse(frontend_dir / "scan.html")

@app.api_route("/diseases", methods=["GET", "HEAD"])
async def get_diseases_page():
    return FileResponse(frontend_dir / "diseases.html")

@app.api_route("/history", methods=["GET", "HEAD"])
async def get_history_page():
    return FileResponse(frontend_dir / "history.html")

@app.api_route("/clinics", methods=["GET", "HEAD"])
async def get_clinics_page():
    return FileResponse(frontend_dir / "clinics.html")

@app.api_route("/about", methods=["GET", "HEAD"])
async def get_about_page():
    return FileResponse(frontend_dir / "about.html")


# ==========================================
# HEALTH CHECK & SERVE FRONTEND
# ==========================================

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Skin Disease AI"}

if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=str(frontend_dir),
            html=True
        ),
        name="frontend"
    )
