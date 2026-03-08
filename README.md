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

**Compare both checkpoints on the test set:**

```bash
python alzheimer_classifier/evaluate_both_models.py
```

Prints accuracy, precision, recall, F1, ROC-AUC and per-class recall for Model 1 (`best_model.pt`) and Model 2 (`best_model_20260307_141516.pt`) and a side-by-side comparison.

**Bulk test (all images in `data/test/`, all classes):**

```bash
python alzheimer_classifier/bulk_test.py [--model model1|model2|both]
```

- Runs every test image through the chosen model(s). Default: `--model both`.
- Prints overall and per-class accuracy, recall, F1, classification report, and confusion matrix.
- Writes CSV(s) to `outputs/bulk_test/bulk_test_model1.csv` and/or `bulk_test_model2.csv` with columns: `path`, `true_class`, `predicted_class`, `correct`, `confidence`, `prob_<Class>` for each class.

**Test on archive dataset (Augmented + Original):**

If you have the archive at `alzheimer_classifier/archive (1)/` with subfolders `AugmentedAlzheimerDataset` and `OriginalDataset` (each with class subfolders: NonDemented, VeryMildDemented, MildDemented, ModerateDemented):

```bash
python alzheimer_classifier/test_archive.py [--archive-root "alzheimer_classifier/archive (1)"]
```

- Runs all images in **AugmentedAlzheimerDataset** and **OriginalDataset** through model1 and model2.
- Prints metrics (accuracy, precision, recall, F1, per-class recall, classification report, confusion matrix) for each (dataset, model).
- Writes CSVs to `outputs/archive_test/`: `AugmentedAlzheimerDataset_model1.csv`, `AugmentedAlzheimerDataset_model2.csv`, `OriginalDataset_model1.csv`, `OriginalDataset_model2.csv`.
- Use `--model model1` or `--model model2` to run only one model.

## Web UI and API

Run the Flask app to get a browser UI and two prediction endpoints (one per model):

```bash
cd alzheimer_classifier && python app.py
```

Then open **http://127.0.0.1:5000**. Upload an MRI image and click:

- **Model 1 (best_model.pt)** — calls `POST /api/predict/model1`
- **Model 2 (timestamped)** — calls `POST /api/predict/model2`

Both endpoints accept a multipart form with an image file (field name `image` or `file`). Response JSON: `predicted_class`, `confidence`, `probabilities`, `class_names`.

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
