# Alzheimer MRI Classifier (Training + Web App)

PyTorch-based classifier for Alzheimer's stage prediction from MRI images, with a Flask web app for upload + prediction + Grad-CAM visualization.

## Current Scope

This repository currently keeps only:

- Training pipeline
- Core model/data/utils modules
- Web app (`app.py` + `static/index.html`)

Older evaluation/inference helper scripts were intentionally removed during cleanup.

## Dataset Layout

Training uses the split under `data/`:

- `data/train/<Class>/`
- `data/val/<Class>/`
- `data/test/<Class>/`

Current configured split ratios in `config.py`:

- `TRAIN_RATIO = 0.80`
- `VAL_RATIO = 0.20`
- `TEST_RATIO = 0.00`

Note: training still expects a validation set for checkpointing and early stopping.

## Project Structure

```text
alzheimer_classifier/
├── app.py
├── config.py
├── train.py
├── static/
│   └── index.html
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── splits.py
│   ├── transforms.py
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── __init__.py
│   └── alzheimer_cnn.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── gradcam.py
├── outputs/
└── requirements.txt
```

## Setup

```bash
cd "/Users/zayanrashidrana/Documents/Alzheimer disease/alzheimer_classifier"
python3 -m pip install -r requirements.txt
```

## Training

```bash
python3 train.py
```

Useful options:

- `--data-root` (default: `config.DATA_ROOT`)
- `--epochs`
- `--batch-size`
- `--lr`
- `--no-amp`
- `--seed`

Default key training settings:

- `BATCH_SIZE = 16`
- Class-weighted `CrossEntropyLoss`
- Early stopping enabled
- AMP enabled when CUDA is available

Outputs:

- Best checkpoint: `outputs/checkpoints/best_model.pt`
- Timestamped best checkpoint: `outputs/checkpoints/best_model_<timestamp>.pt`
- TensorBoard logs: `outputs/logs/`

## Web App

Run:

```bash
python3 app.py
```

Open:

- `http://127.0.0.1:5000`

Available API routes:

- `GET /api/health`
- `GET /api/models`
- `POST /api/predict`
- `POST /api/gradcam`

## Configuration Notes

Important knobs in `config.py`:

- Data/image: `IMAGE_SIZE`, `IMAGE_EXTENSIONS`, split ratios
- Model: `CONV_FILTERS`, `FC_SIZES`, `DROPOUT`, `USE_SE_ATTENTION`
- Training: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`
- Augmentation: random resized crop, affine, rotation, flips, jitter, blur, erasing

## License

Use and adapt for research/internal use.
