
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest

# Dataset location
DATASET = Path("dataset/mvtec-ad/bottle")
TRAIN_DIR = DATASET / "train" / "good"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load normal bottle images
features = []

for image_path in TRAIN_DIR.glob("*.png"):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    features.append(pixels.flatten())

if not features:
    raise FileNotFoundError(
        f"No PNG images found in {TRAIN_DIR}"
    )

# Train anomaly detection model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)
model.fit(features)

# Save the trained model
joblib.dump(model, MODEL_DIR / "bottle_model.pkl")

print(f"Training images: {len(features)}")
print("Model saved to models/bottle_model.pkl")
