import numpy as np
import tensorflow as tf
from PIL import Image
from src.preprocessing import CLASS_NAMES

# Global memory cache so the model is loaded once on server startup
_MODEL_CACHE = {}

def get_loaded_model(model_path):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)
    return _MODEL_CACHE[model_path]

def reload_cached_model(model_path):
    """Call after retraining to refresh the memory cache with updated weights."""
    _MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)

def predict_image(image_file, model_path):
    """Accepts an image file path or file-like object and runs fast in-memory prediction."""
    model = get_loaded_model(model_path)
    
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    
    all_probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return {
        'predicted_class': CLASS_NAMES[predicted_idx],
        'confidence': float(predictions[predicted_idx]),
        'all_probabilities': all_probs
    }