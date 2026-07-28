import os
import shutil
import zipfile
from fastapi import FastAPI, UploadFile, File
from src.prediction import predict_image, reload_cached_model
from src.preprocessing import load_datasets, CLASS_NAMES
from src.model import build_model, compile_model, train_model, save_model

app = FastAPI(title="Skin Cancer ISIC Classifier API")
MODEL_PATH = "models/skin_cancer_model.tf"

@app.get("/health")
def health():
    return {"status": "online", "model_exists": os.path.exists(MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Predict directly from file memory stream without disk write overhead
    result = predict_image(file.file, MODEL_PATH)
    return result

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    temp_zip = f"temp_{file.filename}"
    with open(temp_zip, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
        zip_ref.extractall("data/train")
    
    os.remove(temp_zip)
    return {"message": "Bulk data extracted and merged into training set."}

@app.post("/retrain")
def retrain():
    train_ds, val_ds, _ = load_datasets("data/train", "data/test")
    model, _ = build_model(num_classes=len(CLASS_NAMES))
    model = compile_model(model)
    train_model(model, train_ds, val_ds, epochs=3, checkpoint_path=MODEL_PATH)
    
    # Reload model memory cache so /predict instantly uses newly retrained weights
    reload_cached_model(MODEL_PATH)
    return {"message": "Model retrained and reloaded into memory successfully."}