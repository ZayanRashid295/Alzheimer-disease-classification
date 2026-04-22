"""
Flask app: prediction endpoints, Grad-CAM explainability, health check, and full web UI.
Run: python app.py  then open http://127.0.0.1:5000
"""
import base64
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.transforms import get_eval_transforms
from models import AlzheimerCNN
from utils.gradcam import GradCAM, overlay_heatmap

app = Flask(__name__, static_folder="static", static_url_path="")

CHECKPOINTS = {
    "model1": PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt",
    "model2": PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_20260307_141516.pt",
}
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Any image format PIL can open (we still validate by opening the file)
ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif",
    "jfif", "pjpeg", "pjp", "ico", "ppm", "pgm", "pbm", "tga",
}


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])
    state_dict = ckpt["model_state_dict"]

    # Prefer architecture metadata saved in checkpoint; fall back to current config.
    conv_filters = ckpt.get("conv_filters", config.CONV_FILTERS)
    use_se = ckpt.get("use_se", config.USE_SE_ATTENTION)

    # Infer fc_sizes from checkpoint weights so older/newer checkpoints both load.
    fc_weight_keys = sorted(
        (
            k for k, v in state_dict.items()
            if k.startswith("fc.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 2
        ),
        key=lambda k: int(k.split(".")[1]),
    )
    if len(fc_weight_keys) >= 2:
        # All FC layers except last classification layer are hidden sizes.
        fc_sizes = [int(state_dict[k].shape[0]) for k in fc_weight_keys[:-1]]
    else:
        fc_sizes = ckpt.get("fc_sizes", config.FC_SIZES)

    model = AlzheimerCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=conv_filters,
        fc_sizes=fc_sizes,
        dropout=(0.0, 0.0),
        use_se=use_se,
        he_init=False,
    )
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model, class_names, in_channels


def init_models():
    for key, path in CHECKPOINTS.items():
        if path.exists():
            MODELS[key] = load_model(path)
            print(f"Loaded {key}: {path.name}")
        else:
            print(f"Warning: checkpoint not found for {key}: {path}")


def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_from_image(image_bytes, model_key):
    if model_key not in MODELS:
        return None, "Model not loaded"
    model, class_names, in_channels = MODELS[model_key]
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception as e:
        return None, f"Invalid image: {e}"
    transform = get_eval_transforms(config.IMAGE_SIZE, channels=in_channels)
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())
    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "class_names": class_names,
    }, None


def gradcam_overlay_bytes(image_bytes, model_key, alpha=0.5):
    """Run Grad-CAM for the loaded model and return overlay image as PNG bytes, or (None, error)."""
    if model_key not in MODELS:
        return None, "Model not loaded"
    model, class_names, in_channels = MODELS[model_key]
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception as e:
        return None, f"Invalid image: {e}"
    transform = get_eval_transforms(config.IMAGE_SIZE, channels=in_channels)
    x = transform(img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    target_layer = model.conv_blocks[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(x, target_class=None)
    img_np = np.array(img.resize(config.IMAGE_SIZE))
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    overlay = overlay_heatmap(heatmap, img_np, alpha=alpha)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return buf.read(), None


@app.route("/api/health")
def health():
    """Health check and list of loaded models."""
    return jsonify({
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "device": str(DEVICE),
    })


@app.route("/api/models")
def list_models():
    """List available model keys and checkpoint names."""
    return jsonify({
        "models": [
            {"id": k, "checkpoint": v.name, "loaded": k in MODELS}
            for k, v in CHECKPOINTS.items()
        ]
    })


@app.route("/api/predict", methods=["POST"])
def predict_unified():
    """Predict using form field 'model' (model1 or model2; default model1)."""
    model_key = (request.form.get("model") or "model1").strip().lower()
    if model_key not in CHECKPOINTS:
        return jsonify({"error": f"Unknown model. Use one of: {', '.join(CHECKPOINTS)}"}), 400
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image provided. Use form field 'image' or 'file'."}), 400
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use an image (e.g. JPG, PNG, WebP, GIF, BMP, TIFF)."}), 400
    result, err = predict_from_image(file.read(), model_key)
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@app.route("/api/gradcam", methods=["POST"])
def gradcam():
    """Return Grad-CAM overlay image as base64 PNG. Form: image, model (model1 or model2)."""
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image provided."}), 400
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use an image (e.g. JPG, PNG, WebP, GIF, BMP, TIFF)."}), 400
    model_key = (request.form.get("model") or "model1").strip().lower()
    if model_key not in MODELS:
        return jsonify({"error": f"Model not loaded: {model_key}"}), 400
    data = file.read()
    png_bytes, err = gradcam_overlay_bytes(data, model_key)
    if err:
        return jsonify({"error": err}), 400
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return jsonify({"image_base64": b64})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict/model1", methods=["POST"])
def predict_model1():
    """Predict using Model 1 (best_model.pt)."""
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image provided. Use form field 'image' or 'file'."}), 400
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use an image (e.g. JPG, PNG, WebP, GIF, BMP, TIFF)."}), 400
    result, err = predict_from_image(file.read(), "model1")
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@app.route("/api/predict/model2", methods=["POST"])
def predict_model2():
    """Predict using Model 2 (best_model_20260307_141516.pt)."""
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image provided. Use form field 'image' or 'file'."}), 400
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use an image (e.g. JPG, PNG, WebP, GIF, BMP, TIFF)."}), 400
    result, err = predict_from_image(file.read(), "model2")
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


if __name__ == "__main__":
    init_models()
    if not MODELS:
        print("No models loaded. Exiting.")
        sys.exit(1)
    app.run(host="0.0.0.0", port=5000, debug=True)

