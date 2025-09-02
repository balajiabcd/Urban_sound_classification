# Urban Sound Classification 🎵🔊

A machine learning project for classifying **environmental sounds** from
the [UrbanSound8K
dataset](https://urbansounddataset.weebly.com/urbansound8k.html).\
The pipeline extracts audio features (MFCCs, spectral features, etc.),
builds feature dataframes, trains multiple ML models, and evaluates
their performance.

------------------------------------------------------------------------

## 📂 Project Structure

    UrbanSound-Classification/
    ├─ src/
    │  ├─ extract/                 # feature extraction utils
    │  │   ├─ utils.py
    │  │   └─ extract_data.py
    │  ├─ training
    │  │   ├─ preprocess.py            # splitting, scaling, PCA, encoding
    │  │   ├─ models.py                # candidate ML models
    │  │   ├─ train.py                 # training pipeline (PCA + ML models)
    │  │   ├─ evaluate.py              # metrics, plots, confusion matrix
    │  │   ├─ data_io.py               # load combined feature CSVs
    │  │   └─ predict.py               # run predictions on new features
    │
    ├─ archive/                    # (ignored) raw dataset
    ├─ archive_dataframes/         # (ignored) extracted features CSVs
    ├─ models/                     # (ignored) saved models, scalers, encoders
    ├─ reports/                    # (ignored) metrics & figures
    ├─ requirements.txt
    ├─ README.md
    └─ .gitignore

------------------------------------------------------------------------

## ⚙️ Workflow

### 1️⃣ Feature Extraction

Convert `.wav` files → numerical features using **MFCCs** and other
descriptors.

``` bash
python -m src.extract.extract_data
```

This creates feature CSVs in `archive_dataframes/`.

------------------------------------------------------------------------

### 2️⃣ Train Models

Train multiple ML models (KNN, Decision Trees, Random Forests, Logistic
Regression, SVM).\
Automatically picks the **best model** based on validation F1-score.

``` bash
python -m src.train --archive_df_dir archive_dataframes --target class
```

Artifacts saved in: - `models/` → `model.pkl`, `pca.pkl`, `scaler.pkl`,
`label_encoder.pkl`\
- `reports/` → metrics CSVs, confusion matrix, PCA elbow plot

------------------------------------------------------------------------

### 3️⃣ Evaluate

Generates evaluation metrics & plots: - Accuracy, Precision, Recall,
F1-score\
- Confusion Matrix\
- PCA Explained Variance plot

------------------------------------------------------------------------

### 4️⃣ Predict on New Data

Run predictions on a new features CSV:

``` bash
python -m src.predict     --models_dir models     --features_csv my_features.csv     --out_csv reports/predictions.csv
```

------------------------------------------------------------------------

## 📊 Results (example)

  Model            Accuracy   Precision   Recall   F1-score
  ---------------- ---------- ----------- -------- ----------
  KNN Cosine k=1   0.931      0.932       0.925    0.925
  Random Forest    0.88       0.87        0.88     0.87

✅ Best Model → **KNN (cosine, k=1)** with **F1-score = 0.932**

------------------------------------------------------------------------

## 📦 Dependencies

Main libraries (see `requirements.txt`): - Python 3.11+ - numpy (≤
1.24.4 for Numba compatibility) - pandas - librosa - scikit-learn -
seaborn, matplotlib - joblib, tqdm, pyyaml

Install:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Next Steps

-   Deploy trained model as a **web app** (Flask/FastAPI + Streamlit
    UI).\
-   Extend with **deep learning (CNNs on spectrograms)** for higher
    accuracy.\
-   Experiment with **data augmentation** (noise injection, pitch
    shifting).

------------------------------------------------------------------------

## 🙌 Acknowledgements

-   [UrbanSound8K
    dataset](https://urbansounddataset.weebly.com/urbansound8k.html)\
-   [Librosa](https://librosa.org/) for audio feature extraction\
-   [Scikit-learn](https://scikit-learn.org/) for ML models and
    preprocessing

------------------------------------------------------------------------

✨ With this repo, you can **extract features, train models, and
classify urban sounds** in a reproducible and modular way.
