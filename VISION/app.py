# app.py
import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="VISION - Upload Photo + Predict", page_icon="📷")
st.title("🧠 VISION Prediction")
st.write("Take a photo or upload one, then run it through the CNN model and optionally save it.")

# =========================
# Settings (edit if needed)
# =========================
MODEL_PATH = "face.keras"
CLASSES_PATH = "class_names.json"

# If confidence is below this value, return 'unknown'
UNKNOWN_THRESHOLD = 0.60

# Folder where images will be saved (on the computer/server running Streamlit)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load model + classes
# =========================
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def get_class_names():
    p = Path(CLASSES_PATH)
    if not p.exists():
        # If missing, we can still run in binary mode or show indices
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return data

def infer_img_size_and_channels(model):
    """
    Infer (H, W, C) from model.input_shape.
    Typical shape: (None, H, W, C)
    """
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    if len(ishape) != 4:
        raise ValueError(f"Unexpected model input shape: {ishape}")
    _, h, w, c = ishape
    return int(h), int(w), int(c)

def save_uploaded_image(file_obj, prefix: str = "img") -> Path:
    """
    Save the uploaded/captured image bytes to disk (server-side).
    """
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path

def preprocess_for_model(uploaded_file, target_hw, channels) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile into a model-ready numpy array:
    shape: (1, H, W, C), dtype float32, scaled to [0,1]
    """
    img = Image.open(uploaded_file)

    # Ensure correct number of channels
    if channels == 1:
        img = img.convert("L")   # grayscale
    else:
        img = img.convert("RGB") # 3-channel RGB

    # Resize to model expected input
    img = img.resize(target_hw)

    # PIL -> np array
    arr = np.array(img, dtype=np.float32)

    # If grayscale, ensure channel axis exists
    if channels == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)

    # Normalize to [0,1]
    arr /= 255.0

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr

def predict_label(model, x, class_names=None, unknown_threshold=0.60):
    """
    Supports:
    - Multi-class softmax output: shape (num_classes,)
    - Binary sigmoid output: shape (1,)
    Returns: (label_str, confidence_float, probs_array)
    """
    raw = model.predict(x, verbose=0)[0]

    # Binary sigmoid case: raw is (1,)
    if raw.shape[0] == 1:
        p1 = float(raw[0])
        probs = np.array([1.0 - p1, p1], dtype=np.float32)

        if class_names is not None and len(class_names) == 2:
            labels = class_names
        else:
            labels = ["class_0", "class_1"]

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = labels[idx]

        if conf < unknown_threshold:
            return "unknown", conf, probs

        return label, conf, probs

    # Multi-class softmax case
    probs = np.array(raw, dtype=np.float32)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    if class_names is not None:
        if len(class_names) != probs.shape[0]:
            raise ValueError(
                f"Model outputs {probs.shape[0]} classes but class_names.json has {len(class_names)}."
            )
        label = class_names[idx]
    else:
        label = f"class_{idx}"

    if conf < unknown_threshold:
        return "unknown", conf, probs

    return label, conf, probs

# =========================
# Startup checks
# =========================
try:
    model = get_model()
except Exception as e:
    st.error(f"Failed to load model: {MODEL_PATH}")
    st.exception(e)
    st.stop()

try:
    class_names = get_class_names()
except Exception as e:
    st.warning("class_names.json could not be loaded. Predictions will show class indices.")
    st.exception(e)
    class_names = None

try:
    H, W, C = infer_img_size_and_channels(model)
except Exception as e:
    st.error("Could not infer model input shape.")
    st.exception(e)
    st.stop()

st.caption(f"Model file: {Path(MODEL_PATH).resolve()}")
st.caption(f"Expected input: {H}x{W}x{C} | unknown threshold: {int(UNKNOWN_THRESHOLD*100)}%")
st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")

# =========================
# UI
# =========================
st.subheader("1) Take a picture (camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from computer")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

chosen = camera_photo if camera_photo is not None else gallery_photo

if chosen is not None:
    st.image(chosen, caption="Preview", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Predict"):
            try:
                x = preprocess_for_model(chosen, target_hw=(W, H), channels=C)
                label, conf, probs = predict_label(
                    model,
                    x,
                    class_names=class_names,
                    unknown_threshold=UNKNOWN_THRESHOLD
                )

                st.success(f"Prediction: **{label}**")
                st.write(f"Confidence: **{conf*100:.2f}%**")

                # Optional diagnostics
                st.caption(f"Sent to model with shape: {x.shape} (should be (1, {H}, {W}, {C}))")

                with st.expander("Show probabilities"):
                    if class_names is not None and len(class_names) == len(probs):
                        rows = [{"class": cn, "probability": float(p)} for cn, p in zip(class_names, probs)]
                    else:
                        rows = [{"class": f"class_{i}", "probability": float(p)} for i, p in enumerate(probs)]
                    st.dataframe(rows, use_container_width=True)

                if label == "unknown":
                    st.warning("Low confidence -> returned 'unknown'.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    with col2:
        if st.button("💾 Save image to computer"):
            try:
                saved_path = save_uploaded_image(chosen, prefix="vision")
                st.success(f"Saved ✅  {saved_path.resolve()}")
                st.info("The image is saved on the machine running Streamlit (your computer).")
            except Exception as e:
                st.error(f"Failed to save: {e}")
                st.exception(e)

else:
    st.info("Please capture or upload an image to continue.")
