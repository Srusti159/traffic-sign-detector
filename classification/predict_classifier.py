import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification.preprocessing import preprocess_single_image
from classification.traffic_sign_info import GTSRB_CLASSES

def predict_single_image(
    image_path,
    model_path="models/traffic_sign_classifier.keras",
    mapping_path="models/class_indices.json"
):
    """
    Performs standalone classification on a single cropped traffic sign image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping not found at: {mapping_path}")

    # Load Model & Mapping
    model = tf.keras.models.load_model(model_path)
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    # Preprocess & Predict
    input_tensor = preprocess_single_image(image_path, target_size=(128, 128))
    probabilities = model.predict(input_tensor, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    
    # Map back to GTSRB Class ID
    dir_name = mapping["index_to_class"][str(predicted_idx)]
    gtsrb_id = int(dir_name)
    sign_info = GTSRB_CLASSES.get(gtsrb_id, {"name": "Unknown", "meaning": "N/A", "action": "N/A"})

    result = {
        "image_path": image_path,
        "gtsrb_class_id": gtsrb_id,
        "class_name": sign_info["name"],
        "confidence": confidence,
        "meaning": sign_info["meaning"],
        "action": sign_info["action"]
    }

    print("\n" + "=" * 65)
    print("🚦 TRAFFIC SIGN CLASSIFICATION RESULT")
    print("=" * 65)
    print(f"File:                   {image_path}")
    print(f"Predicted Sign:         {result['class_name']} (Class {result['gtsrb_class_id']})")
    print(f"Classification Conf:    {result['confidence'] * 100:.2f}%")
    print(f"Sign Meaning:           {result['meaning']}")
    print(f"Recommended Action:     {result['action']}")
    print("=" * 65 + "\n")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a cropped traffic sign image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/traffic_sign_classifier.keras", help="Path to .keras model")
    parser.add_argument("--mapping", type=str, default="models/class_indices.json", help="Path to class_indices.json")
    args = parser.parse_args()

    predict_single_image(args.image, args.model, args.mapping)
