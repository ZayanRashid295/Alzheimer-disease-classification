"""
Configuration for Alzheimer's MRI multiclass classification pipeline.
All paths and hyperparameters in one place; no hardcoded values in scripts.
"""
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths (relative to project root or overridable via env/CLI)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# After move_dataset_to_split.py: data lives in data/train, data/val, data/test
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
EVAL_PLOTS_DIR = OUTPUT_DIR / "eval_plots"
INFERENCE_OUTPUT_DIR = OUTPUT_DIR / "inference"

# Ensure output dirs exist at import (optional; also created in scripts)
for d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, EVAL_PLOTS_DIR, INFERENCE_OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Inferred from dataset; override if needed (1 for grayscale, 3 for RGB)
INPUT_CHANNELS: Optional[int] = None  # None = auto-detect from first image
IMAGE_SIZE = (224, 224)
# Supported extensions when scanning dataset
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# Train / val / test split (stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Augmentation (medical-safe)
# ---------------------------------------------------------------------------
AUGMENTATION = {
    "rotation_degrees": 15,
    "affine_translate": (0.1, 0.1),  # fraction
    "affine_scale": (0.9, 1.1),
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.0,  # often disabled for brain MRI
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CONV_FILTERS = [32, 64, 128, 256]
FC_SIZES = [256, 128, 64]  # before num_classes
DROPOUT = (0.5, 0.3)  # after first two FC layers
USE_SE_ATTENTION = True
HE_INIT = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_MAX_NORM = 1.0
USE_AMP = True
USE_FOCAL_LOSS = False
FOCAL_ALPHA = None  # set per-class if using focal
FOCAL_GAMMA = 2.0

# ReduceLROnPlateau
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-6

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_BATCH_SIZE = 64
ROC_OVR = True

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
DEFAULT_CKPT = CHECKPOINT_DIR / "best_model.pt"
