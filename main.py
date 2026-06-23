import io
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from scipy import signal
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

from visualization import (
    apply_gradcam_colormap,
    apply_spectrogram_colormap,
    render_axis_panel,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLASS_INDEX_PATH = MODELS_DIR / "class_indicates.json"

SPECTROGRAM_DIR = BASE_DIR / "outputs" / "spectrograms"
GRADCAM_DIR = BASE_DIR / "outputs" / "gradcams"
OVERLAY_WAVEFORM_DIR = BASE_DIR / "outputs" / "overlay_waveforms"

SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 4000
LOW_CUT = 100
HIGH_CUT = 1800
DURATION_SECONDS = 4.0
IMG_SIZE = (224, 224)
DEFAULT_MODEL_NAME = os.getenv("PARUSCAN_DEFAULT_MODEL", "vgg16").lower()
PUBLIC_BASE_URL = os.getenv("PARUSCAN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://pulmosense-prod.vercel.app",
    ).split(",")
    if origin.strip()
]
OUTPUT_RETENTION_HOURS = float(os.getenv("PARUSCAN_OUTPUT_RETENTION_HOURS", "168"))
OUTPUT_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("PARUSCAN_OUTPUT_CLEANUP_INTERVAL_SECONDS", "3600")
)

FREQ_BANDS = [
    {
        "band": "Low",
        "range_hz": "100-400 Hz",
        "low": 100,
        "high": 400,
        "desc": "Low-frequency component",
    },
    {
        "band": "Mid",
        "range_hz": "400-900 Hz",
        "low": 400,
        "high": 900,
        "desc": "Mid-frequency component",
    },
    {
        "band": "High",
        "range_hz": "900-1800 Hz",
        "low": 900,
        "high": 1800,
        "desc": "High-frequency component",
    },
]


@dataclass(frozen=True)
class ModelSpec:
    artifact_dir: Path
    preprocess_input: Callable[[np.ndarray], np.ndarray]
    gradcam_layer: str


MODEL_SPECS = {
    "vgg16": ModelSpec(
        artifact_dir=MODELS_DIR / "best_vgg16_keras",
        preprocess_input=vgg16_preprocess,
        gradcam_layer="block5_conv3",
    ),
    "vgg19": ModelSpec(
        artifact_dir=MODELS_DIR / "best_vgg19_keras",
        preprocess_input=vgg19_preprocess,
        gradcam_layer="block5_conv4",
    ),
}

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

app = FastAPI(
    title="ParuScan AI Service",
    description="AI inference service for lung sound pneumonia screening",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")


loaded_models: dict[str, tf.keras.Model] = {}
class_names: list[str] = []
model_lock = Lock()
cleanup_lock = Lock()
last_cleanup_at = 0.0


def load_class_names() -> list[str]:
    if not CLASS_INDEX_PATH.exists():
        raise RuntimeError(f"Class index file not found: {CLASS_INDEX_PATH}")

    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    names: list[str | None] = [None] * len(class_indices)
    for label, idx in class_indices.items():
        names[int(idx)] = label

    if any(name is None for name in names):
        raise RuntimeError(f"Invalid class index mapping: {CLASS_INDEX_PATH}")

    return [str(name) for name in names]


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()
    if normalized not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_name '{model_name}'. Use one of: {supported}",
        )
    return normalized


def get_model(model_name: str) -> tf.keras.Model:
    normalized = normalize_model_name(model_name)
    cached_model = loaded_models.get(normalized)
    if cached_model is not None:
        return cached_model

    spec = MODEL_SPECS[normalized]
    if not spec.artifact_dir.exists():
        raise RuntimeError(f"Model artifact not found: {spec.artifact_dir}")

    with model_lock:
        cached_model = loaded_models.get(normalized)
        if cached_model is not None:
            return cached_model

        model = tf.keras.models.load_model(spec.artifact_dir, compile=False)
        model.get_layer(spec.gradcam_layer)
        loaded_models[normalized] = model
        print(f"Loaded {normalized} model from {spec.artifact_dir}")
        return model


def cleanup_old_outputs(force: bool = False) -> int:
    global last_cleanup_at

    if OUTPUT_RETENTION_HOURS <= 0:
        return 0

    now = time.time()
    if not force and now - last_cleanup_at < OUTPUT_CLEANUP_INTERVAL_SECONDS:
        return 0

    with cleanup_lock:
        now = time.time()
        if not force and now - last_cleanup_at < OUTPUT_CLEANUP_INTERVAL_SECONDS:
            return 0

        cutoff = now - OUTPUT_RETENTION_HOURS * 3600
        deleted_count = 0

        for output_dir in (SPECTROGRAM_DIR, GRADCAM_DIR, OVERLAY_WAVEFORM_DIR):
            for path in output_dir.glob("*.png"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        deleted_count += 1
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    print(f"Failed to delete old output {path}: {exc}")

        last_cleanup_at = now
        if deleted_count:
            print(f"Cleaned {deleted_count} old output file(s)")
        return deleted_count


@app.on_event("startup")
def load_ai_model():
    global class_names

    class_names = load_class_names()
    get_model(DEFAULT_MODEL_NAME)

    print("Class names:", class_names)
    print("Default model:", DEFAULT_MODEL_NAME)
    print("Output retention hours:", OUTPUT_RETENTION_HOURS)
    cleanup_old_outputs(force=True)


def read_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {exc}") from exc

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="WAV file is empty")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), int(sr)


def pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.size >= target_len:
        return audio[:target_len]

    pad_width = target_len - audio.size
    mode = "reflect" if audio.size > 1 else "edge"
    return np.pad(audio, (0, pad_width), mode=mode)


def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr <= 0:
        raise HTTPException(status_code=400, detail="Invalid WAV sample rate")

    if sr != TARGET_SR:
        divisor = math.gcd(sr, TARGET_SR)
        audio = signal.resample_poly(audio, TARGET_SR // divisor, sr // divisor)

    audio = audio.astype(np.float32)
    audio -= np.mean(audio)

    if audio.size < 256:
        audio = pad_or_trim(audio, 256)

    nyquist = TARGET_SR / 2.0
    sos = signal.butter(
        4,
        [LOW_CUT / nyquist, HIGH_CUT / nyquist],
        btype="band",
        output="sos",
    )
    audio = signal.sosfiltfilt(sos, audio)
    audio /= np.max(np.abs(audio)) + 1e-9

    target_len = int(DURATION_SECONDS * TARGET_SR)
    return pad_or_trim(audio.astype(np.float32), target_len)


def wav_bytes_to_stft_image(wav_bytes: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    raw_audio, sr = read_wav_bytes(wav_bytes)
    audio = preprocess_audio(raw_audio, sr)

    freqs, _, stft = signal.stft(
        audio,
        fs=TARGET_SR,
        nperseg=256,
        noverlap=192,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft)
    freq_mask = (freqs >= LOW_CUT) & (freqs <= HIGH_CUT)
    magnitude = magnitude[freq_mask]

    if magnitude.size == 0:
        raise RuntimeError("Failed to generate STFT image")

    db = 20 * np.log10(magnitude + 1e-8)
    low, high = np.percentile(db, [5, 95])
    normalized = np.clip((db - low) / (high - low + 1e-8), 0, 1)

    image = np.flipud((normalized * 255).astype(np.uint8))
    pil_image = Image.fromarray(image, mode="L").resize(IMG_SIZE, RESAMPLE_BILINEAR)
    rgb_image = apply_spectrogram_colormap(
        np.asarray(pil_image, dtype=np.float32) / 255.0
    )

    return np.asarray(rgb_image, dtype=np.uint8), audio, sr


def make_model_input(stft_image: np.ndarray, model_name: str) -> np.ndarray:
    spec = MODEL_SPECS[model_name]
    batch = np.expand_dims(stft_image.astype(np.float32), axis=0)
    return spec.preprocess_input(batch.copy())


def predict_image(
    model: tf.keras.Model,
    stft_image: np.ndarray,
    model_name: str,
) -> tuple[str, float, np.ndarray]:
    if not class_names:
        raise RuntimeError("Class names are not loaded")

    model_input = make_model_input(stft_image, model_name)
    preds = np.asarray(model.predict(model_input, verbose=0)[0], dtype=np.float32)
    pred_idx = int(np.argmax(preds))

    if pred_idx >= len(class_names):
        raise RuntimeError("Model output size does not match class index mapping")

    label = class_names[pred_idx]
    confidence = float(preds[pred_idx] * 100)
    return label, confidence, preds


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    return apply_gradcam_colormap(heatmap)


def describe_frequency_focus(heatmap: np.ndarray) -> str:
    hot_rows = np.where(heatmap > 0.6)[0]
    if hot_rows.size == 0:
        return "Perhatian model tersebar pada seluruh spektrum"

    row_mean = float(np.mean(hot_rows))
    freq_focus = HIGH_CUT - (row_mean / (IMG_SIZE[1] - 1)) * (HIGH_CUT - LOW_CUT)
    freq_low = max(LOW_CUT, freq_focus - 150)
    freq_high = min(HIGH_CUT, freq_focus + 150)
    return f"Fokus model sekitar {freq_low:.0f}-{freq_high:.0f} Hz"


def draw_waveform_overlay(image: np.ndarray, audio: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    width, height = pil_image.size
    x_positions = np.arange(width)
    audio_positions = np.linspace(0, audio.size - 1, width)
    waveform = np.interp(audio_positions, np.arange(audio.size), audio)
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-9)
    y_positions = (10 + waveform * (height - 20)).astype(int)

    points = [(int(x), int(y)) for x, y in zip(x_positions, y_positions)]
    if len(points) > 1:
        draw.line(points, fill=(255, 230, 0), width=2)

    return np.asarray(pil_image, dtype=np.uint8)


def gradcam_outputs(
    model: tf.keras.Model,
    stft_image: np.ndarray,
    audio: np.ndarray,
    model_name: str,
    pred_idx: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    spec = MODEL_SPECS[model_name]
    model_input = make_model_input(stft_image, model_name)
    input_tensor = tf.convert_to_tensor(model_input)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(spec.gradcam_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        class_channel = predictions[:, pred_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Failed to compute Grad-CAM gradients")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_heat = float(tf.reduce_max(heatmap).numpy())
    if max_heat > 0:
        heatmap = heatmap / max_heat

    heatmap_image = Image.fromarray(
        np.uint8(heatmap.numpy() * 255),
        mode="L",
    ).resize(IMG_SIZE, RESAMPLE_BILINEAR)
    heatmap_resized = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    heatmap_rgb = colorize_heatmap(heatmap_resized)

    overlay = stft_image.astype(np.float32) * 0.6 + heatmap_rgb.astype(np.float32) * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_with_waveform = draw_waveform_overlay(overlay, audio)

    return heatmap_rgb, overlay_with_waveform, describe_frequency_focus(heatmap_resized)


def energy_per_band(audio: np.ndarray) -> list[dict[str, float | str]]:
    energies: list[float] = []
    nyquist = TARGET_SR / 2.0

    for band in FREQ_BANDS:
        sos = signal.butter(
            4,
            [float(band["low"]) / nyquist, float(band["high"]) / nyquist],
            btype="band",
            output="sos",
        )
        filtered = signal.sosfiltfilt(sos, audio)
        energies.append(float(np.sum(filtered**2)))

    total_energy = sum(energies) + 1e-10
    results = []
    for band, energy in zip(FREQ_BANDS, energies):
        results.append(
            {
                "band": str(band["band"]),
                "range_hz": str(band["range_hz"]),
                "desc": str(band["desc"]),
                "percent": energy / total_energy * 100,
            }
        )
    return results


def get_risk_level(label: str, confidence: float) -> str:
    label_upper = label.upper()

    if confidence < 65:
        return "Confidence Rendah"

    if label_upper == "PNEUMONIA":
        if confidence >= 85:
            return "Tinggi"
        return "Sedang"

    return "Rendah"


def get_recommendation(label: str, confidence: float) -> str:
    label_upper = label.upper()

    if confidence < 65:
        return "Confidence rendah. Disarankan melakukan rekaman ulang dengan kualitas audio lebih baik."

    if label_upper == "PNEUMONIA":
        return "Terdapat indikasi pneumonia. Disarankan melakukan pemeriksaan klinis lanjutan."

    return "Tidak terdeteksi indikasi pneumonia. Konsultasikan dengan tenaga medis jika gejala tetap berlanjut."


@app.get("/")
def root():
    return {
        "service": "ParuScan AI Service",
        "status": "running",
        "default_model": DEFAULT_MODEL_NAME,
        "supported_models": sorted(MODEL_SPECS),
    }


@app.get("/models")
def list_models():
    return {
        "default_model": DEFAULT_MODEL_NAME,
        "loaded_models": sorted(loaded_models),
        "supported_models": [
            {
                "name": name,
                "artifact_dir": str(spec.artifact_dir.relative_to(BASE_DIR)),
                "gradcam_layer": spec.gradcam_layer,
            }
            for name, spec in sorted(MODEL_SPECS.items())
        ],
        "class_names": class_names,
    }


@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
    model_name: str = Query(DEFAULT_MODEL_NAME, description="Use vgg16 or vgg19"),
):
    selected_model_name = normalize_model_name(model_name)

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    try:
        wav_bytes = await file.read()
        selected_model = get_model(selected_model_name)

        stft_image, audio, sr = wav_bytes_to_stft_image(wav_bytes)
        label, confidence, preds = predict_image(
            model=selected_model,
            stft_image=stft_image,
            model_name=selected_model_name,
        )

        pred_idx = int(np.argmax(preds))
        gradcam_img, overlay_waveform_img, frequency_focus = gradcam_outputs(
            model=selected_model,
            stft_image=stft_image,
            audio=audio,
            model_name=selected_model_name,
            pred_idx=pred_idx,
        )

        bands = energy_per_band(audio)
        result_id = str(uuid.uuid4())

        spectrogram_filename = f"{result_id}_spectrogram.png"
        gradcam_filename = f"{result_id}_gradcam.png"
        overlay_waveform_filename = f"{result_id}_overlay_waveform.png"

        spectrogram_path = SPECTROGRAM_DIR / spectrogram_filename
        gradcam_path = GRADCAM_DIR / gradcam_filename
        overlay_waveform_path = OVERLAY_WAVEFORM_DIR / overlay_waveform_filename

        render_axis_panel(stft_image).save(spectrogram_path)
        render_axis_panel(
            gradcam_img,
            show_contribution_scale=True,
        ).save(gradcam_path)
        render_axis_panel(
            overlay_waveform_img,
            show_waveform_legend=True,
        ).save(overlay_waveform_path)
        cleanup_old_outputs()

        return {
            "model": selected_model_name,
            "prediction": label,
            "confidence": round(float(confidence), 2),
            "risk_level": get_risk_level(label, confidence),
            "recommendation": get_recommendation(label, confidence),
            "spectrogram_url": f"{PUBLIC_BASE_URL}/outputs/spectrograms/{spectrogram_filename}",
            "gradcam_url": f"{PUBLIC_BASE_URL}/outputs/gradcams/{gradcam_filename}",
            "overlay_waveform_url": (
                f"{PUBLIC_BASE_URL}/outputs/overlay_waveforms/{overlay_waveform_filename}"
            ),
            "frequency_focus": frequency_focus,
            "energy_bands": [
                {
                    "band": item["band"],
                    "range_hz": item["range_hz"],
                    "description": item["desc"],
                    "percentage": round(float(item["percent"]), 2),
                }
                for item in bands
            ],
            "raw_probabilities": {
                class_names[i]: round(float(preds[i]), 4)
                for i in range(len(class_names))
            },
            "audio": {
                "sample_rate": sr,
                "processed_sample_rate": TARGET_SR,
                "duration_seconds": DURATION_SECONDS,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
