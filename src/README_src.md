# Source Code (`src/`)

This folder contains all the core source code for the Urban Sound Classification project.  
The pipeline goes from **audio files → feature dataframes → model training → evaluation → prediction**.

---

## Folder Overview

### `extract_data.py`
- Reads `.wav` audio files from the `archive/` dataset.
- Extracts MFCC features using `utils.py`.
- Saves per-fold DataFrames into `archive_dataframes/`.

Run:
```bash
python -m src.extract_data
```

---

### `utils.py`
- Helper functions for feature extraction and DataFrame creation.
- `get_features(file_path)` → extracts MFCC features from a single audio file.
- `make_dataframe(folder_path, class_by_file_name)` → builds a labeled DataFrame from all `.wav` files in a folder.

---

### `data_io.py`
- Loads multiple per-fold CSVs from `archive_dataframes/` and concatenates them into one DataFrame.
- Function: `load_archive_dataframes(folder)`.

---

### `preprocess.py`
- Data preparation and preprocessing functions:
  - `split_xy(df, target_col)` → split features and labels.
  - `make_splits()` → train/val/test split.
  - `fit_transform_pca_scaler()` → apply PCA and scaling.
  - `label_encode()` → encode class labels.

---

### `models.py`
- Defines model candidates for training:
  - Multiple KNN variants with different distance metrics.
  - Decision Tree, Logistic Regression, Random Forest.

Function: `build_candidates(seed=42)`.

---

### `evaluate.py`
- Evaluation utilities:
  - `pca_elbow_plot()` → plots explained variance for PCA.
  - `save_confusion()` → saves confusion matrix heatmap.
  - `write_metrics()` → saves metrics table to CSV.
  - `get_metrics()` → computes Accuracy, Precision, Recall, F1.

---

### `train.py`
- Main training pipeline:
  1. Loads DataFrames (`data_io.py`).
  2. Preprocesses with PCA + scaling (`preprocess.py`).
  3. Trains multiple models (`models.py`).
  4. Selects best model by validation F1-score.
  5. Evaluates on test set, saves metrics + confusion matrix.
  6. Saves artifacts (`model.pkl`, `pca.pkl`, `scaler.pkl`, `label_encoder.pkl`).

Run:
```bash
python -m src.training.train \
  --archive_df_dir archive_dataframes \
  --target class \
  --models_dir models \
  --reports_dir reports
```

---

### `predict.py`
- Loads saved model + preprocessing artifacts.
- Reads features from a CSV.
- Runs prediction and saves results.

Run:
```bash
python -m src.training.predict \
  --models_dir models \
  --features_csv data/processed/features.csv \
  --out_csv reports/predictions.csv
```

---

## Typical Workflow
1. **Extract features**  
   `python -m src.extract_data`

2. **Train models**  
   `python -m src.training.train --archive_df_dir archive_dataframes --target class`

3. **Predict on new data**  
   `python -m src.training.predict --features_csv data/processed/features.csv`

---
