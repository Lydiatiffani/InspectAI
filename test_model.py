from pathlib import Path

import joblib
import numpy as np
from PIL import Image

DATASET = Path("dataset/mvtec-ad/bottle/test")
MODEL_PATH = Path("models/bottle_model.pkl")

model = joblib.load(MODEL_PATH)

for category in ["good", "broken_large", "broken_small", "contamination"]:
    folder = DATASET / category
    images = list(folder.glob("*.png"))

    if not images:
        print(f"{category}: No PNG images found")
        continue

    image = Image.open(images[0]).convert("RGB")
    image = image.resize((32, 32))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    prediction = model.predict([pixels.flatten()])[0]

    result = (
        "Potential anomaly"
        if prediction == -1
        else "No anomaly detected"
    )

    print(f"{category}: {result}")
