import librosa
import numpy as np
import soundfile as sf
import io
import os
import tempfile

TARGET_SR = 16000

def process_audio(file_bytes: bytes, filename: str | None = None) -> np.ndarray:
    """Robustly load audio bytes and resample to TARGET_SR.

    Tries soundfile first (best for WAV/FLAC), then falls back to librosa for
    formats like MP3. Returns mono float32 audio at TARGET_SR.
    """
    # Try soundfile on a byte stream
    try:
        audio, sr = sf.read(io.BytesIO(file_bytes))
    except Exception:
        # Fall back to librosa using a temporary file (handles MP3 via audioread)
        suffix = os.path.splitext(filename or "")[1] or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            audio, sr = librosa.load(tmp.name, sr=None, mono=False)

    # Ensure mono
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return np.asarray(audio, dtype=np.float32)
