# Alzheimer MRI Multiclass Classifier

Research-grade PyTorch pipeline for classifying brain MRI scans using the Alzheimer's Disease Multiclass Images Dataset. The pipeline is **data-driven**: class names, channel count (grayscale vs RGB), and class distribution are inferred from the dataset.

## Dataset

The dataset lives **inside** the project under `data/` as a stratified 70/15/15 split:

- **`data/train/<Class>/`** — training images (~30,800)
- **`data/val/<Class>/`** — validation images (~6,600)
- **`data/test/<Class>/`** — test images (~6,600)

Classes: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`. Images are JPEG, RGB. The pipeline discovers classes and paths from these folders; no separate `combined_images` is needed.

To recreate the split from a single `combined_images/` folder (one subfolder per class), run once:

```bash
python3 alzheimer_classifier/scripts/move_dataset_to_split.py
```
(Expects `combined_images/` next to the `alzheimer_classifier` folder.)

## Project Structure

```
alzheimer_classifier/
├── data/           # Dataset, stratified splits, transforms
├── models/         # Custom CNN (4 conv blocks, optional SE, GAP, FC)
├── utils/          # Metrics, Grad-CAM, Focal Loss
├── train.py        # Training with AMP, early stopping, TensorBoard
├── evaluate.py     # Test-set metrics, confusion matrix, ROC-AUC
├── inference.py    # Single-image prediction + optional Grad-CAM
├── config.py       # Paths and hyperparameters
├── requirements.txt
└── README.md
```

## Setup

```bash
cd alzheimer_classifier
pip install -r requirements.txt
```

## Training

From the **project root** (parent of `alzheimer_classifier`):

```bash
python alzheimer_classifier/train.py --data-root /path/to/combined_images
```

Or from inside `alzheimer_classifier`:

```bash
python train.py
```

Options: `--epochs`, `--batch-size`, `--lr`, `--no-amp`, `--focal` (Focal Loss), `--seed`.

- Best checkpoint: `outputs/checkpoints/best_model.pt`
- TensorBoard: `tensorboard --logdir outputs/logs`

## Evaluation

```bash
python alzheimer_classifier/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
```

Writes to `outputs/eval_plots/`: confusion matrix, ROC (OvR), classification report.

## Inference

```bash
python alzheimer_classifier/inference.py path/to/mri.jpg --checkpoint outputs/checkpoints/best_model.pt
```

Optional Grad-CAM overlay:

```bash
python alzheimer_classifier/inference.py path/to/mri.jpg --gradcam outputs/inference/gradcam.png
```

## Configuration

Edit `config.py` for:

- Paths: `DATA_ROOT`, `CHECKPOINT_DIR`, etc.
- Data: `IMAGE_SIZE`, `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- Model: `CONV_FILTERS`, `USE_SE_ATTENTION`, `DROPOUT`
- Training: `BATCH_SIZE`, `LEARNING_RATE`, `USE_AMP`, `USE_FOCAL_LOSS`, `EARLY_STOPPING_PATIENCE`

## Design Notes

- **Channels**: Set `INPUT_CHANNELS` in config to `1` or `3`, or leave `None` to auto-detect from the first image.
- **Class weights**: CrossEntropyLoss (and optional Focal Loss) use inverse-frequency weights from the training set.
- **Augmentation**: Rotation ≤15°, small affine translate/scale, horizontal flip; no vertical flip by default (brain MRI).
- **Grad-CAM**: Uses the last conv block of the custom CNN; overlay saved as JPEG.

## License

Use and adapt as needed for research or internal use.
