"""
Flask app: two prediction endpoints (one per model) and a simple UI.
Run: python app.py  then open http://127.0.0.1:5000
"""
import sys
from pathlib import Path

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

app = Flask(__name__, static_folder="static", static_url_path="")

CHECKPOINTS = {
    "model1": PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt",
    "model2": PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_20260307_141516.pt",
}
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    in_channels = int(ckpt["in_channels"])
    num_classes = int(ckpt["num_classes"])
    model = AlzheimerCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_filters=config.CONV_FILTERS,
        fc_sizes=config.FC_SIZES,
        dropout=(0.0, 0.0),
        use_se=config.USE_SE_ATTENTION,
        he_init=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
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
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        return jsonify({"error": "Allowed extensions: " + ", ".join(ALLOWED_EXTENSIONS)}), 400
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
        return jsonify({"error": "Allowed extensions: " + ", ".join(ALLOWED_EXTENSIONS)}), 400
    result, err = predict_from_image(file.read(), "model2")
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


if __name__ == "__main__":
    init_models()
    if not MODELS:
        print("No models loaded. Exiting.")
        sys.exit(1)
    app.run(host="0.0.0.0", port=5000, debug=False)
