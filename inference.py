"""
Inference script: predict class for a single MRI image from CLI.
Outputs: predicted class, probability distribution, confidence. Optional Grad-CAM overlay.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.transforms import get_eval_transforms, ToTensorMRI, NormalizeMRI
from models import AlzheimerCNN
from utils.gradcam import GradCAM, save_gradcam


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
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
    model = model.to(device)
    model.eval()
    return model, class_names, in_channels


def run_inference(
    image_path: Path,
    checkpoint: Path,
    save_gradcam_path: Optional[Path] = None,
) -> dict:
    device = get_device()
    model, class_names, in_channels = load_model(checkpoint, device)

    img_pil = Image.open(image_path)
    if img_pil.mode in ("RGBA", "P"):
        img_pil = img_pil.convert("RGB")
    elif img_pil.mode not in ("RGB", "L"):
        img_pil = img_pil.convert("RGB")

    transform = get_eval_transforms(config.IMAGE_SIZE, channels=in_channels)
    x = transform(img_pil)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())

    result = {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "class_names": class_names,
    }

    if save_gradcam_path is not None:
        target_layer = model.conv_blocks[-1]
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(x, target_class=pred_idx)
        save_gradcam(heatmap, x, save_gradcam_path, alpha=0.5)
        result["gradcam_path"] = str(save_gradcam_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Alzheimer MRI inference")
    parser.add_argument("image_path", type=Path, help="Path to MRI image")
    parser.add_argument("--checkpoint", type=Path, default=config.DEFAULT_CKPT)
    parser.add_argument("--gradcam", type=Path, default=None, help="Save Grad-CAM overlay to this path")
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Error: image not found: {args.image_path}")
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    result = run_inference(
        args.image_path,
        args.checkpoint,
        save_gradcam_path=args.gradcam,
    )

    print("Predicted class:", result["predicted_class"])
    print("Confidence:", f"{result['confidence']:.4f}")
    print("Probability distribution:")
    for k, v in result["probabilities"].items():
        print(f"  {k}: {v:.4f}")
    if result.get("gradcam_path"):
        print("Grad-CAM saved to:", result["gradcam_path"])


if __name__ == "__main__":
    main()
