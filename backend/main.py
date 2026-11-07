from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import numpy as np
import json
import torch
import io
import os
import sys
from pathlib import Path
from typing import List, Optional

from pymongo import MongoClient

# Decompression bomb prevention & upload constraints
Image.MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
CONFIDENCE_THRESHOLD = 45.0  # Alert if top prediction is below 45%


def validate_image_quality(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Heuristic check to reject blank, solid-color, or completely dark/flat images
    before passing them to the AI model.
    """
    if pil_image.width < 48 or pil_image.height < 48:
        return False, "Image resolution is too low. Please provide a clear, higher-resolution skin lesion photo."

    thumb = np.array(pil_image.resize((64, 64)))
    std_dev = float(np.std(thumb))
    if std_dev < 15.0:
        return False, "The uploaded image appears blank, solid-color, or severely under-exposed. Please upload a clear, focused photo of the skin lesion."
    
    # Check dynamic range
    if float(np.max(thumb) - np.min(thumb)) < 25.0:
        return False, "The image lacks contrast or illumination. Please ensure adequate lighting on the skin lesion."

    return True, ""


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
    description="AI-based skin lesion classification prototype with realtime scanner and e-commerce care recommendations",
    version="1.2.0"
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
# PERSISTENT CUSTOM CATEGORIES STORAGE
# ==========================================

CUSTOM_DISEASES_FILE = BASE_DIR / "data" / "custom_diseases.json"

def load_custom_diseases() -> list[dict]:
    """Loads user-added disease categories from disk."""
    if not CUSTOM_DISEASES_FILE.exists():
        return []
    try:
        with open(CUSTOM_DISEASES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("diseases", [])
    except Exception as e:
        print(f"Warning: Failed to read custom diseases file: {e}")
        return []

def save_custom_diseases(diseases_list: list[dict]):
    """Persists user-added disease categories to disk."""
    CUSTOM_DISEASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_DISEASES_FILE, "w", encoding="utf-8") as f:
        json.dump({"diseases": diseases_list}, f, indent=2)


# ==========================================
# RECOMMENDED E-COMMERCE PRODUCTS REPOSITORY
# ==========================================

DEFAULT_PRODUCTS = {
    "Melanoma, NOS": [
        {
            "name": "La Roche-Posay Anthelios Mineral Ultra-Light Sunscreen SPF 50",
            "category": "Broad-Spectrum UV Defense",
            "reason": "High-potency mineral shield (zinc oxide/titanium dioxide) essential for stopping UV radiation that drives malignant cell proliferation.",
            "badge": "Derm Recommended",
            "price_range": "$28 - $36",
            "redirect_url": "https://www.amazon.com/s?k=la+roche+posay+anthelios+mineral+spf+50"
        },
        {
            "name": "DermLite DL100 / Smartphone Polarized Dermoscopy Lens",
            "category": "Clinical Monitoring Tool",
            "reason": "Medical-grade polarized dermatoscope attachment to help closely monitor pigment changes, asymmetry, and border irregularities.",
            "badge": "Clinical Tool",
            "price_range": "$120 - $250",
            "redirect_url": "https://www.amazon.com/s?k=smartphone+dermatoscope+lens"
        },
        {
            "name": "Coolibar UPF 50+ Sun Protection Long Sleeve Top",
            "category": "Physical UV Shield",
            "reason": "Dermatologist-endorsed UPF 50+ fabric that blocks 98% of UVA/UVB rays during prolonged outdoor exposure.",
            "badge": "UPF 50+ Defense",
            "price_range": "$45 - $65",
            "redirect_url": "https://www.amazon.com/s?k=coolibar+upf+50+clothing"
        }
    ],
    "Basal cell carcinoma": [
        {
            "name": "EltaMD UV Clear Broad-Spectrum Facial Sunscreen SPF 46",
            "category": "Daily Photoprotection",
            "reason": "Formulated with high-purity zinc oxide and niacinamide to calm sensitive, sun-damaged, or pre/post-procedure skin.",
            "badge": "Top Rated",
            "price_range": "$39 - $43",
            "redirect_url": "https://www.amazon.com/s?k=eltamd+uv+clear+spf+46"
        },
        {
            "name": "CeraVe Healing Ointment with Ceramides & Petrolatum",
            "category": "Skin Barrier Protection",
            "reason": "Non-comedogenic barrier balm that prevents dehydration and shields delicate lesion sites from environmental abrasion.",
            "badge": "Barrier Repair",
            "price_range": "$14 - $21",
            "redirect_url": "https://www.amazon.com/s?k=cerave+healing+ointment"
        },
        {
            "name": "Wide-Brimmed UPF 50+ Sun Protection Sun Hat",
            "category": "Physical Shade",
            "reason": "Protects high-risk facial, ear, and neck areas most frequently impacted by cumulative UV damage.",
            "badge": "Physical Defense",
            "price_range": "$22 - $35",
            "redirect_url": "https://www.amazon.com/s?k=upf+50+wide+brim+hat"
        }
    ],
    "Squamous cell carcinoma, NOS": [
        {
            "name": "ISDIN Eryfotona Actinica Mineral Sunscreen SPF 50+",
            "category": "Actinic DNA Defense",
            "reason": "Features 11% zinc oxide plus DNA Repairsomes to mitigate chronic sun-induced damage on photo-exposed skin.",
            "badge": "Derm Favorite",
            "price_range": "$55 - $70",
            "redirect_url": "https://www.amazon.com/s?k=isdin+eryfotona+actinica"
        },
        {
            "name": "Aquaphor Advanced Therapy Healing Ointment",
            "category": "Lesion Moisture Shield",
            "reason": "Protects irritated or scaly skin, promotes optimal moist wound-healing conditions, and prevents friction.",
            "badge": "Skin Protectant",
            "price_range": "$12 - $18",
            "redirect_url": "https://www.amazon.com/s?k=aquaphor+healing+ointment"
        },
        {
            "name": "UPF 50+ UV Protective Sun Sleeves & Driving Gloves",
            "category": "Targeted Sun Defense",
            "reason": "Essential for shielding sun-exposed forearms and dorsum of the hands from cumulative actinic damage.",
            "badge": "Daily Protection",
            "price_range": "$16 - $24",
            "redirect_url": "https://www.amazon.com/s?k=upf+50+sun+sleeves"
        }
    ],
    "Solar or actinic keratosis": [
        {
            "name": "Paula's Choice 10% Niacinamide Booster Serum",
            "category": "Cellular Repair",
            "reason": "Niacinamide (Vitamin B3) supports natural cellular energy replenishment and photo-damage recovery in actinic areas.",
            "badge": "Skin Energy",
            "price_range": "$36 - $48",
            "redirect_url": "https://www.amazon.com/s?k=paulas+choice+10+niacinamide"
        },
        {
            "name": "CeraVe SA Smoothing Cream with Salicylic Acid & Ceramides",
            "category": "Keratolytic Hydration",
            "reason": "Gently softens rough, scaly, sandpapery actinic spots without causing excessive peeling or irritation.",
            "badge": "Gentle Smoothing",
            "price_range": "$18 - $26",
            "redirect_url": "https://www.amazon.com/s?k=cerave+sa+cream"
        },
        {
            "name": "Neutrogena Ultra Sheer Dry-Touch Mineral Sunscreen SPF 70",
            "category": "Intense UV Shield",
            "reason": "Crucial preventative barrier to halt further actinic progression into invasive squamous cell lesions.",
            "badge": "High Broad Spectrum",
            "price_range": "$12 - $17",
            "redirect_url": "https://www.amazon.com/s?k=neutrogena+ultra+sheer+spf+70"
        }
    ],
    "Nevus": [
        {
            "name": "La Roche-Posay Anthelios Melt-in Milk Body & Face SPF 60",
            "category": "Daily Sun Defense",
            "reason": "Critical daily protection preventing benign melanocytic nevi from acquiring UV-induced cellular dysplasia.",
            "badge": "Consumer Choice",
            "price_range": "$25 - $38",
            "redirect_url": "https://www.amazon.com/s?k=la+roche+posay+anthelios+melt+in+milk"
        },
        {
            "name": "Cetaphil Gentle Skin Cleanser for Sensitive Skin",
            "category": "Gentle Cleanser",
            "reason": "Non-stripping, hypoallergenic formula that preserves natural skin barrier integrity around sensitive moles.",
            "badge": "Sensitive Skin",
            "price_range": "$11 - $16",
            "redirect_url": "https://www.amazon.com/s?k=cetaphil+gentle+skin+cleanser"
        },
        {
            "name": "Precision ABCDE Mole Measuring Ruler & 10x LED Magnifier",
            "category": "Self-Monitoring Tool",
            "reason": "Enables accurate monthly measurement of mole diameter (>6mm threshold) and asymmetry evaluation at home.",
            "badge": "Self Check Tool",
            "price_range": "$15 - $28",
            "redirect_url": "https://www.amazon.com/s?k=skin+mole+measuring+ruler+magnifier"
        }
    ],
    "Pigmented benign keratosis": [
        {
            "name": "AmLactin Daily Moisturizing Body Lotion (12% Lactic Acid)",
            "category": "AHA Exfoliation",
            "reason": "Lactic acid gently exfoliates thickened, stuck-on hyperkeratotic layers while deeply moisturizing.",
            "badge": "AHA Exfoliant",
            "price_range": "$16 - $24",
            "redirect_url": "https://www.amazon.com/s?k=amlactin+daily+moisturizing+lotion"
        },
        {
            "name": "CeraVe SA Cleanser with Salicylic Acid & Niacinamide",
            "category": "Smoothing Wash",
            "reason": "Helps gently smooth rough, waxy keratotic textures during regular daily showers without barrier disruption.",
            "badge": "Derm Favorite",
            "price_range": "$14 - $19",
            "redirect_url": "https://www.amazon.com/s?k=cerave+sa+cleanser"
        },
        {
            "name": "Aveeno Calm + Restore Oat Gel Moisturizer",
            "category": "Calming Moisture",
            "reason": "Prebiotic oat instantly calms itchiness and friction-induced inflammation around benign keratoses.",
            "badge": "Soothing Oat",
            "price_range": "$18 - $24",
            "redirect_url": "https://www.amazon.com/s?k=aveeno+calm+and+restore+oat+gel"
        }
    ],
    "Dermatofibroma": [
        {
            "name": "La Roche-Posay Lipikar Triple Repair AP+M Moisturizing Cream",
            "category": "Lipid Replenishing Balm",
            "reason": "Clinically proven to relieve dryness, tightness, and surface discomfort over firm nodular skin bumps.",
            "badge": "Barrier Soothing",
            "price_range": "$20 - $26",
            "redirect_url": "https://www.amazon.com/s?k=la+roche+posay+lipikar+triple+repair"
        },
        {
            "name": "Curad Hydrocolloid Cushioning Friction Relief Bandages",
            "category": "Friction Cushion",
            "reason": "Shields raised, firm dermatofibromas from rubbing against clothing waistbands, straps, or footwear.",
            "badge": "Friction Relief",
            "price_range": "$8 - $14",
            "redirect_url": "https://www.amazon.com/s?k=hydrocolloid+friction+cushion+bandages"
        },
        {
            "name": "Vanicream Gentle Body Wash for Sensitive Skin",
            "category": "Hypoallergenic Wash",
            "reason": "Free of dyes, fragrances, parabens, and botanical extracts to eliminate avoidable nodule irritation.",
            "badge": "Fragrance-Free",
            "price_range": "$12 - $17",
            "redirect_url": "https://www.amazon.com/s?k=vanicream+gentle+body+wash"
        }
    ]
}


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
# PREDICT API & INFERENCE HELPERS (WITH TTA)
# ==========================================

def run_model_inference(pil_image: Image.Image):
    """
    Synchronous inference worker function with Test-Time Augmentation (TTA).
    Runs multi-view TTA (original, horizontal flip, vertical flip, 90-degree rotation)
    to eliminate camera angle/rotational sensitivity and significantly boost real-world accuracy.
    Executed in a background threadpool to prevent blocking FastAPI's async event loop.
    """
    # Create 4 multi-view augmented representations
    views = [
        pil_image,
        pil_image.transpose(Image.FLIP_LEFT_RIGHT),
        pil_image.transpose(Image.FLIP_TOP_BOTTOM),
        pil_image.transpose(Image.ROTATE_90)
    ]
    tensors = [image_transforms(v) for v in views]
    batch_tensor = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities_batch = torch.softmax(outputs, dim=1)
        # Average probabilities across the 4 augmented views for smooth, stable predictions
        probabilities = probabilities_batch.mean(dim=0, keepdim=True)

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

    # Run non-blocking inference in threadpool with TTA
    top_predictions = await run_in_threadpool(run_model_inference, image)

    top_result = top_predictions[0]
    predicted_label = top_result["disease"]
    confidence_percentage = top_result["confidence"]
    is_inconclusive = confidence_percentage < CONFIDENCE_THRESHOLD

    # Get information from MongoDB, custom categories, or fallback knowledge base
    disease_info = None

    if disease_collection is not None:
        try:
            disease_info = disease_collection.find_one(
                {"name": predicted_label},
                {"_id": 0}
            )
        except Exception as e:
            print(f"Warning: MongoDB lookup failed: {e}")
            disease_info = None

    if not disease_info:
        # Check custom categories
        custom_diseases = load_custom_diseases()
        for cd in custom_diseases:
            if cd.get("name", "").lower() == predicted_label.lower():
                disease_info = cd
                break

    if not disease_info:
        disease_info = DEFAULT_DISEASE_INFO.get(predicted_label, {})

    # Retrieve tailored e-commerce product recommendations
    products = disease_info.get("products") or DEFAULT_PRODUCTS.get(predicted_label, [])
    if not products:
        # General soothing/sun protective fallback products
        products = [
            {
                "name": "Broad-Spectrum SPF 50+ Daily Mineral Sunscreen",
                "category": "Sun Protection",
                "reason": "Crucial barrier defense to protect skin lesions from UV-induced degradation.",
                "badge": "Essential Care",
                "price_range": "$18 - $28",
                "redirect_url": "https://www.amazon.com/s?k=mineral+sunscreen+spf+50"
            },
            {
                "name": "CeraVe Moisturizing Cream with Essential Ceramides",
                "category": "Barrier Repair",
                "reason": "Restores protective skin lipid barrier and minimizes dry irritation.",
                "badge": "Top Rated",
                "price_range": "$14 - $22",
                "redirect_url": "https://www.amazon.com/s?k=cerave+moisturizing+cream"
            }
        ]

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
        "description": disease_info.get("description", ""),
        "causes": disease_info.get("causes", []),
        "general_care": disease_info.get("general_care", []),
        "medical_attention": disease_info.get(
            "medical_attention",
            "Please consult a dermatologist for professional evaluation."
        ),
        "products": products
    }


# ==========================================
# CATEGORY MANAGEMENT & ENCYCLOPEDIA APIS
# ==========================================

class ProductItem(BaseModel):
    name: str
    category: str = "Skincare"
    reason: str = ""
    badge: str = "Recommended"
    price_range: str = "$15 - $25"
    redirect_url: str = ""

class DiseaseCategoryCreate(BaseModel):
    name: str
    category: str = "benign"  # malignant, precancerous, benign, inflammatory, infectious
    risk: str = "Moderate"
    color: str = "#10b981"
    description: str
    causes: List[str] = Field(default_factory=list)
    general_care: List[str] = Field(default_factory=list)
    medical_attention: str = "Consult a dermatologist for a professional clinical evaluation."
    products: List[ProductItem] = Field(default_factory=list)


@app.get("/api/diseases")
async def get_diseases_catalog():
    """Returns dynamic catalog of all skin diseases (built-in + user added)."""
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
    
    # 1. Base built-in categories
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
            "medical_attention": info.get("medical_attention", ""),
            "is_custom": False,
            "products": DEFAULT_PRODUCTS.get(name, [])
        })

    # 2. Append custom categories from persistent storage
    custom_diseases = load_custom_diseases()
    for cd in custom_diseases:
        catalog.append({
            "name": cd.get("name", "Untitled Condition"),
            "category": cd.get("category", "benign"),
            "risk": cd.get("risk", "Low (Manageable)"),
            "color": cd.get("color", "#3b82f6"),
            "description": cd.get("description", ""),
            "causes": cd.get("causes", []),
            "general_care": cd.get("general_care", []),
            "medical_attention": cd.get("medical_attention", "Consult a dermatologist for professional evaluation."),
            "is_custom": True,
            "products": cd.get("products", [])
        })

    # 3. If MongoDB has additional records, merge them
    if disease_collection is not None:
        try:
            records = list(disease_collection.find({}, {"_id": 0}))
            existing_names = {c["name"].lower() for c in catalog}
            for rec in records:
                if rec.get("name", "").lower() not in existing_names:
                    rec["is_custom"] = True
                    catalog.append(rec)
        except Exception as e:
            print(f"MongoDB catalog lookup failed: {e}")

    return {"diseases": catalog}


@app.post("/api/diseases", status_code=status.HTTP_201_CREATED)
async def create_disease_category(payload: DiseaseCategoryCreate):
    """
    Frontend Privilege Endpoint: Allows adding new disease categories with custom
    care guidelines, risk profiles, and e-commerce product recommendations.
    """
    clean_name = payload.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Disease name is required.")

    # Check built-in collision
    if clean_name in DEFAULT_DISEASE_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"'{clean_name}' is a built-in core model category and cannot be overwritten."
        )

    custom_diseases = load_custom_diseases()
    for existing in custom_diseases:
        if existing.get("name", "").lower() == clean_name.lower():
            raise HTTPException(
                status_code=400,
                detail=f"A disease category named '{clean_name}' already exists."
            )

    # Format products with default Amazon redirect if missing
    formatted_products = []
    for p in payload.products:
        p_dict = p.dict()
        if not p_dict.get("redirect_url"):
            search_query = p_dict.get("name", clean_name).replace(" ", "+")
            p_dict["redirect_url"] = f"https://www.amazon.com/s?k={search_query}"
        formatted_products.append(p_dict)

    new_record = {
        "name": clean_name,
        "category": payload.category.lower(),
        "risk": payload.risk,
        "color": payload.color,
        "description": payload.description.strip(),
        "causes": [c.strip() for c in payload.causes if c.strip()],
        "general_care": [g.strip() for g in payload.general_care if g.strip()],
        "medical_attention": payload.medical_attention.strip(),
        "is_custom": True,
        "products": formatted_products
    }

    # Save to persistent file
    custom_diseases.append(new_record)
    save_custom_diseases(custom_diseases)

    # Also persist to MongoDB if available
    if disease_collection is not None:
        try:
            disease_collection.update_one(
                {"name": clean_name},
                {"$set": new_record},
                upsert=True
            )
        except Exception as e:
            print(f"Warning: MongoDB upsert failed: {e}")

    return {
        "status": "success",
        "message": f"Disease category '{clean_name}' added successfully.",
        "category": new_record
    }


@app.delete("/api/diseases/{disease_name}")
async def delete_disease_category(disease_name: str):
    """Deletes a custom disease category from persistent storage."""
    clean_name = disease_name.strip()
    if clean_name in DEFAULT_DISEASE_INFO:
        raise HTTPException(
            status_code=400,
            detail="Built-in core model categories cannot be deleted."
        )

    custom_diseases = load_custom_diseases()
    initial_len = len(custom_diseases)
    custom_diseases = [d for d in custom_diseases if d.get("name", "").lower() != clean_name.lower()]

    if len(custom_diseases) == initial_len:
        raise HTTPException(status_code=404, detail=f"Custom category '{clean_name}' not found.")

    save_custom_diseases(custom_diseases)

    if disease_collection is not None:
        try:
            disease_collection.delete_one({"name": clean_name})
        except Exception as e:
            print(f"Warning: MongoDB delete failed: {e}")

    return {"status": "success", "message": f"Category '{clean_name}' deleted."}


# ==========================================
# PAGE ROUTES & STATIC SERVING
# ==========================================

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
    return {
        "status": "ok",
        "service": "Skin Disease AI",
        "tta_enabled": True,
        "classes_count": len(reverse_label_map) + len(load_custom_diseases())
    }

if frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=str(frontend_dir),
            html=True
        ),
        name="frontend"
    )
