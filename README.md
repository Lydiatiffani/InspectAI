# InspectAI — AI-Based Product Quality Inspection System

## Overview
InspectAI is a beginner-friendly AI project that analyzes bottle images and flags potentially unusual products. It demonstrates how anomaly detection can support quality inspection in manufacturing.

## Problem Statement
Manual product inspection can be time-consuming and inconsistent. Some visible defects may be missed, leading to product waste and additional inspection costs.

## Objectives
- Analyze uploaded bottle images.
- Identify images that appear unusual compared with normal training images.
- Display inspection results through a web dashboard.
- Demonstrate the use of machine learning for quality control.

## Technologies Used
- Python
- Streamlit
- NumPy
- Pillow
- Scikit-learn
- Joblib
- Isolation Forest anomaly detection

## How It Works
1. Collect normal bottle images for training.
2. Resize images to 32 × 32 pixels.
3. Convert pixel values into numerical features.
4. Train an Isolation Forest model using normal images.
5. Upload a bottle image through the dashboard.
6. Display whether the model flags the image as potentially anomalous.

## Project Structure
```text
AI-PRODUCT-INSPECTION/
├── app.py
├── train_model.py
├── test_model.py
├── requirements.txt
├── .gitignore
└── models/
    └── bottle_model.pkl  (generated locally)
```

## Setup
Use Python 3.11 or 3.12 for the recommended compatibility range.

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install the required packages:
   `pip install -r requirements.txt`
4. Download the bottle subset of the MVTec Anomaly Detection dataset from an authorized source and follow its license.
5. Place the dataset under `dataset/mvtec-ad/bottle/`.
6. Train the model:
   `python train_model.py`
7. Start the dashboard:
   `python -m streamlit run app.py`

The expected training folder is `dataset/mvtec-ad/bottle/train/good/`.

## Testing
The prototype was tested with sample images from the `good`, `broken_large`, `broken_small`, and `contamination` categories. In an initial four-image check, three sample predictions matched their expected categories. This is only a preliminary check, not a reliable accuracy measurement.

## Limitations
- Uses small, resized RGB pixel values rather than advanced image features.
- Lighting, image backgrounds, and camera conditions may affect predictions.
- Small defects can be missed, and normal products may be flagged.
- The decision score is not a probability of a defect.
- Results should not be used as the sole basis for real industrial quality decisions.

## Dataset and License
The project was developed using the bottle subset of the MVTec Anomaly Detection dataset through a third-party dataset repository. Review the original dataset terms and the specific source repository's license before using or redistributing dataset files or derived artifacts. Dataset images are not included in this repository.

## Future Improvements
- Use more representative image features or a pretrained vision model.
- Evaluate on a larger test set using precision, recall, and other suitable metrics.
- Add defect localization and a more detailed inspection report.
- Test under different lighting and image conditions.

## Disclaimer
InspectAI is an educational prototype and is not a certified industrial inspection system.
