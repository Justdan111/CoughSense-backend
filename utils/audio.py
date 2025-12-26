import librosa
import numpy as np
import soundfile as sf
import io

TARGET_SR = 16000

def process_audio(file_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(file_bytes))
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return audio.astype(np.float32)
