"""
prediction.py

Loads the trained model and makes a prediction on ONE new image.
This is the function your API's /predict endpoint will call.
"""

import numpy as np
from src.preprocessing import preprocess_single_image, CLASS_NAMES
from src.model import load_model

# Loaded once when the module is imported, so we don't reload the model
# from disk on every single prediction request (that would be slow).
_model = None


def get_model(model_path="models/skin_cancer_model.keras"):
    """
    Returns the loaded model, loading it from disk the first time
    and reusing it (cached) on subsequent calls.
    """
    global _model
    if _model is None:
        _model = load_model(model_path)
    return _model


def reload_cached_model(model_path="models/skin_cancer_model.keras"):
    """
    Forces the cached model to be thrown away and reloaded from disk on the
    next prediction. Call this right after retraining, so /predict serves
    the freshly retrained weights instead of the old cached ones.
    """
    global _model
    _model = load_model(model_path)
    return _model


def predict_image(image_path: str, model_path="models/skin_cancer_model.keras"):
    """
    Predicts the skin lesion class for a single image.

    Args:
        image_path: path to the image file to classify
        model_path: path to the saved trained model

    Returns:
        dict with:
            - predicted_class: the class name with the highest probability
            - confidence: how confident the model is (0-1)
            - all_probabilities: probability for every one of the 9 classes
              (useful for showing a breakdown in the UI)
    """
    model = get_model(model_path)
    img_array = preprocess_single_image(image_path)

    predictions = model.predict(img_array)[0]  # shape: (9,) — one probability per class
    predicted_index = int(np.argmax(predictions))

    result = {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(predictions[predicted_index]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
        },
    }
    return result


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python src/prediction.py path/to/some_image.jpg
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/prediction.py <path_to_image>")
    else:
        result = predict_image(sys.argv[1])
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
