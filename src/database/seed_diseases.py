from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["skin_disease_ai"]
collection = db["diseases"]

diseases = [
    {
        "name": "Nevus",
        "description": "A nevus, commonly called a mole, is a common skin growth.",
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

    {
        "name": "Melanoma, NOS",
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

    {
        "name": "Basal cell carcinoma",
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

    {
        "name": "Squamous cell carcinoma, NOS",
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

    {
        "name": "Solar or actinic keratosis",
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

    {
        "name": "Pigmented benign keratosis",
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

    {
        "name": "Dermatofibroma",
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
]

collection.delete_many({})
collection.insert_many(diseases)

print("Disease information inserted successfully!")
print("Number of records:", collection.count_documents({}))