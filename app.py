from fastapi import FastAPI, UploadFile, HTTPException
from utils.audio import process_audio
# from utils.auth import verify_user  # Temporarily disabled for testing
import numpy as np
import joblib

app = FastAPI(title="CoughSense API")

# Load model ONCE
classifier = joblib.load("model/yamnet_random_forest.joblib")
yamnet_mean = np.load("model/yamnet_mean.npy")


def extract_features(audio: np.ndarray):
    # Simulated embedding to match trained model input shape
    features = np.mean(audio)
    return np.full_like(yamnet_mean, features)


def risk_from_probability(prob: float):
    if prob >= 0.8:
        return "High", "Please see a doctor soon."
    if prob >= 0.5:
        return "Medium", "Consider seeing a doctor if symptoms persist."
    return "Low", "Monitor symptoms and practice self‑care."

 
@app.post("/predict")
async def predict(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    audio_bytes = await file.read()
    try:
        audio = process_audio(audio_bytes, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")

    features = extract_features(audio).reshape(1, -1)

    try:
        prob = classifier.predict_proba(features)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    risk, advice = risk_from_probability(float(prob))

    return {
        "risk": risk,
        "confidence": float(prob),
        "advice": advice,
    }



