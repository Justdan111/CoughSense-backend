from fastapi import FastAPI, UploadFile, Depends
from utils.audio import process_audio
from utils.auth import verify_user
import numpy as np
import joblib

app = FastAPI(title="CoughSense API")

# Load model ONCE
classifier = joblib.load("model/yamnet_random_forest.joblib")
yamnet_mean = np.load("model/yamnet_mean.npy")


def extract_features(audio: np.ndarray):
    # Simulated embedding (already trained)
    features = np.mean(audio)
    return np.full_like(yamnet_mean, features)

 
@app.post("/predict")
async def predict(
    file: UploadFile,
    user=Depends(verify_user)
):
    audio_bytes = await file.read()
    audio = process_audio(audio_bytes)

    features = extract_features(audio).reshape(1, -1)

    prob = classifier.predict_proba(features)[0][1]
    prediction = int(prob > 0.5)

    return {
        "prediction": prediction,
        "probability": float(prob)
    }
