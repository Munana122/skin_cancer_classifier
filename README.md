# Skin Cancer ISIC Classifier

An end-to-end MLOps project that classifies dermoscopic skin lesion images into
9 diagnostic categories, using transfer learning with MobileNetV2. This covers
the full pipeline: data preprocessing, model training and evaluation, a live
API, a web UI, retraining triggered from new data, and load testing.

## Live links

- **API (Render):** https://skin-cancer-classifier-z0bg.onrender.com
- **API docs (Swagger):** https://skin-cancer-classifier-z0bg.onrender.com/docs
- **Web UI (Streamlit):** https://skincancerclassifier-olhj8dxmqrp6cfv5z4o6xr.streamlit.app
- **Video demo:** https://youtu.be/uc8i5jAUSec
- **GitHub repo:** https://github.com/Munana122/skin_cancer_classifier

Note: the API runs on Render's free tier, so it spins down after periods of
inactivity. The first request after a while can take 30-60 seconds to wake
back up — that's normal, not a bug.

## What this project does

I used the ISIC skin cancer dataset from Kaggle, which has 9 classes:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

The model is a MobileNetV2 backbone (pretrained on ImageNet) with a custom
classification head on top, trained in two phases — first with the base
frozen, then fine-tuned on the last 30 layers with a lower learning rate.

## Results, honestly

Test accuracy came out to around 42-44%, with a macro F1 score in the
0.36-0.40 range. That's not a great number on paper, and I want to be upfront
about why: the dataset is pretty imbalanced. Some classes like `pigmented
benign keratosis` have 450+ training images, while `seborrheic keratosis` has
under 80. The model's performance tracks that imbalance closely — it does
genuinely well on well-represented classes (`vascular lesion` hits close to
perfect precision/recall in most runs) and struggles badly on the smaller
ones (`seborrheic keratosis` scores 0 across the board pretty consistently).

Full breakdown, confusion matrix, and per-class metrics are in the notebook.

## Project structure

```
skin_cancer_classifier/
├── notebook/
│   └── skin_cancer_classifier.ipynb   # data prep, training, evaluation
├── src/
│   ├── preprocessing.py                # data loading + image preprocessing
│   ├── model.py                        # model architecture + training
│   └── prediction.py                   # single-image prediction logic
├── streamlit_app/
│   ├── ui.py                           # web UI (deployed separately on Streamlit Cloud)
│   └── requirements.txt                # lightweight deps for the UI only
├── data/
│   ├── train/                          # one folder per class
│   └── test/
├── models/
│   └── skin_cancer_model.keras
├── app.py                              # FastAPI backend
├── locustfile.py                       # load testing script
├── Dockerfile
└── requirements.txt                    # full deps for the API/backend
```

## Setting it up locally

**1. Clone the repo**
```bash
git clone https://github.com/Munana122/skin_cancer_classifier.git
cd skin_cancer_classifier
```

**2. Get the trained model**

The trained model file isn't tiny (~22MB), so it lives with the rest of the
repo at `models/skin_cancer_model.keras`. If you're running the notebook
yourself from scratch instead, training + saving it there will regenerate it.

**3. Install dependencies and run the API**
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to try the endpoints directly.

**4. Run the UI locally (optional)**
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run ui.py
```
By default it points at the live Render backend. To point it at your local
API instead, set:
```bash
export API_URL=http://localhost:8000
```

## API endpoints

| Method | Endpoint    | What it does                                              |
|--------|-------------|-------------------------------------------------------------|
| GET    | `/health`   | Confirms the API is up and whether the model file loaded  |
| POST   | `/predict`  | Upload one image, get back a predicted class + confidence |
| POST   | `/upload`   | Upload a `.zip` of new training images                    |
| POST   | `/retrain`  | Retrains the model on everything currently in `data/train`|

## A note on retraining

Retraining works, but Render's free tier has a request timeout that's
shorter than a full retrain sometimes takes. If a retrain request times out
in the UI, it may still be finishing on the server — check `/health` after a
minute or two to confirm. In a production setup, this would be handled as a
background job instead of a blocking request.

## Load testing

I used Locust to simulate concurrent traffic against the live `/predict`
endpoint:

```bash
locust -f locustfile.py --host https://skin-cancer-classifier-z0bg.onrender.com
```

Then opened `http://localhost:8089`, ran it with ~20 simulated users, and
recorded the response time and requests/second.

**Results:**
<img width="944" height="430" alt="image" src="https://github.com/user-attachments/assets/417ee17a-2b5f-4f0a-b374-0c6f28ac8566" />



## What I'd improve with more time

- More training data, especially for the underrepresented classes
- Class-weighted training to push back against the imbalance
- Moving retraining to a background task/queue instead of a blocking request
- A proper database for tracking uploaded images instead of just dumping
  them into the existing folder structure

## Tech stack

TensorFlow / Keras, FastAPI, Streamlit, Docker, Render, Locust
