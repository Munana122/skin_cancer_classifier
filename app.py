import os
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.prediction import predict_image, reload_cached_model
from src.preprocessing import load_datasets, CLASS_NAMES
from src.model import build_model, compile_model, train_model, save_model

app = FastAPI(title="Skin Cancer ISIC Classifier API")

# Must exactly match what your notebook saves — check Colab's save_model()
# call / your models/ folder if you rename this file.
MODEL_PATH = "models/skin_cancer_model.keras"


@app.get("/health")
def health():
    return {
        "status": "online",
        "model_exists": os.path.exists(MODEL_PATH),
        "num_classes": len(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail=f"Model file not found at {MODEL_PATH}")
    try:
        result = predict_image(file.file, MODEL_PATH)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")
    temp_zip = f"/tmp/{file.filename}"
    try:
        with open(temp_zip, "wb") as buffer:
            buffer.write(await file.read())
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall("data/train")
        return {"message": "Bulk data extracted and merged into training set."}
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)


@app.post("/retrain")
def retrain():
    try:
        train_ds, val_ds, test_ds = load_datasets("data/train", "data/test")
        model, base_model = build_model()
        model = compile_model(model)
        train_model(model, train_ds, val_ds, epochs=1, checkpoint_path=MODEL_PATH)
        save_model(model, path=MODEL_PATH)

        # Reload model memory cache so /predict instantly uses newly retrained weights
        reload_cached_model(MODEL_PATH)
        return {"message": "Model retrained and reloaded into memory successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
