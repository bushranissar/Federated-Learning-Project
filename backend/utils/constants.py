"""
Disease Metadata and Constants
Federated Learning Tomato Leaf Disease Detection
"""

# Disease classes in order (0-4)
DISEASE_CLASSES = [
    "Healthy",
    "Bacterial Spot",
    "Late Blight",
    "Septoria Leaf Spot",
    "Yellow Leaf"
]

DISEASE_INFO = {
    "Healthy": {
        "name": "Healthy",
        "description": "The tomato leaf appears healthy with no signs of disease infection. Normal chlorophyll production and tissue structure are intact.",
        "symptoms": [
            "Uniform green coloration",
            "No spots or lesions on leaf surface",
            "Normal leaf structure and turgor",
            "No wilting or discoloration"
        ],
        "severity": "None",
        "treatment": "No treatment required. Continue standard plant care practices including proper watering, fertilization, and pest management.",
        "prevention": [
            "Regular monitoring for early signs of disease",
            "Maintain proper plant spacing for air circulation",
            "Practice crop rotation",
            "Use disease-resistant varieties when available"
        ]
    },
    "Bacterial Spot": {
        "name": "Bacterial Spot",
        "description": "Bacterial spot is caused by Xanthomonas campestris pv. vesicatoria. It affects leaves, stems, and fruits, causing significant yield loss in tomato crops.",
        "symptoms": [
            "Small, water-soaked spots on leaves (2-5 mm)",
            "Spots turn dark brown to black with age",
            "Yellow halos surrounding lesions",
            "Leaf curling and defoliation in severe cases",
            "Raised, scabby lesions on fruits"
        ],
        "severity": "Moderate to High",
        "treatment": "Apply copper-based bactericides (copper hydroxide or copper sulfate) at 7-10 day intervals. Use streptomycin sulfate as a preventive measure during early growth stages.",
        "prevention": [
            "Use disease-free seeds and transplants",
            "Avoid overhead irrigation",
            "Apply copper-based sprays preventatively",
            "Remove and destroy infected plant debris",
            "Practice 3-4 year crop rotation with non-host crops"
        ]
    },
    "Late Blight": {
        "name": "Late Blight",
        "description": "Late blight is caused by Phytophthora infestans, a highly destructive oomycete. It was responsible for the Irish Potato Famine and remains a major threat to tomato production worldwide.",
        "symptoms": [
            "Large, irregular water-soaked lesions on leaves",
            "White cottony fungal growth on leaf undersides",
            "Dark brown to black lesions on stems",
            "Rapid wilting and collapse of foliage",
            "Firm, dark lesions on green fruits"
        ],
        "severity": "Very High (can destroy entire crop in days)",
        "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or metalaxyl immediately at first sign. For organic control, use copper-based fungicides. Remove and destroy infected plants to prevent spread.",
        "prevention": [
            "Plant resistant tomato varieties",
            "Ensure proper air circulation through pruning",
            "Avoid overhead irrigation",
            "Monitor weather conditions (cool, wet weather promotes disease)",
            "Apply fungicides preventatively during high-risk periods",
            "Destroy volunteer tomato plants and potato culls"
        ]
    },
    "Septoria Leaf Spot": {
        "name": "Septoria Leaf Spot",
        "description": "Septoria leaf spot is caused by the fungus Septoria lycopersici. It is one of the most common foliar diseases of tomatoes, typically appearing after fruit set.",
        "symptoms": [
            "Small circular spots (1/16 to 1/4 inch diameter)",
            "Gray centers with dark brown edges",
            "Tiny black fruiting bodies (pycnidia) in center of spots",
            "Lower leaves affected first",
            "Progressive defoliation from bottom to top"
        ],
        "severity": "Moderate",
        "treatment": "Apply fungicides containing chlorothalonil, copper, or mancozeb at 7-14 day intervals. Remove infected lower leaves to slow disease progression.",
        "prevention": [
            "Use disease-free seeds",
            "Mulch around plants to prevent soil splash",
            "Water at base of plants, keeping foliage dry",
            "Remove lower leaves as plants grow",
            "Practice 2-3 year crop rotation",
            "Clean up all plant debris at end of season"
        ]
    },
    "Yellow Leaf": {
        "name": "Yellow Leaf (Nutrient Deficiency / Viral)",
        "description": "Yellowing of tomato leaves can indicate nutrient deficiencies (nitrogen, iron, magnesium) or viral infections such as Tomato Yellow Leaf Curl Virus (TYLCV) transmitted by whiteflies.",
        "symptoms": [
            "Uniform or interveinal yellowing of leaves",
            "Leaf curling or cupping",
            "Stunted plant growth",
            "Reduced fruit set and yield",
            "Vein clearing in new growth"
        ],
        "severity": "Moderate to High",
        "treatment": "For nutrient deficiency: Apply balanced fertilizer with micronutrients. For viral infection: No cure available; remove and destroy infected plants to prevent spread to healthy plants.",
        "prevention": [
            "Maintain proper soil nutrition with regular testing",
            "Use reflective mulches to deter whiteflies",
            "Install insect netting to exclude whitefly vectors",
            "Use TYLCV-resistant tomato varieties",
            "Control whitefly populations with insecticidal soaps",
            "Remove weeds that may serve as virus reservoirs"
        ]
    }
}

# Model paths
MODEL_CONFIG = {
    "cnn_model_path": "models/trained/cnn_model.pth",
    "svm_model_path": "models/trained/svm_model.pkl",
    "input_size": (64, 64),  # Model input size
    "feature_dim": 128,
    "ensemble_weights": {
        "cnn": 0.6,
        "svm": 0.4
    }
}