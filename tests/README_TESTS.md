# 🧪 Test Suite for Urban Sound Classification

This document explains the **unit tests** and **integration tests** written for the project.

---

## 📂 Test Structure

All tests live in the `tests/` folder:

```
tests/
│── test_utils.py          # Feature extraction (MFCCs, dataframe creation)
│── test_extract_data.py   # Data extraction pipeline
│── test_models.py         # Candidate models builder
│── test_preprocess.py     # Data splitting, scaling, PCA, encoding
│── test_train.py          # Full training pipeline (end-to-end)
│── test_evaluate.py       # Metrics + plotting utilities
│── test_data_io.py        # Archive dataframe loader
│── test_predict.py        # Prediction pipeline with saved artifacts
```

---

## 🚀 Running the Tests

1. **Activate your environment**:
   ```bash
   conda activate env_sound_classification
   # or
   .\env_sound_classification\Scripts\activate
   ```

2. **Run the full suite**:
   ```bash
   pytest -v
   ```

3. **Run a specific test file**:
   ```bash
   pytest tests/test_train.py -v
   ```

4. **Run a single test function**:
   ```bash
   pytest tests/test_train.py::test_main_runs -v
   ```

---

## ⚠️ Warnings

Some external libraries raise deprecation or user warnings during tests:

- `pkg_resources` (deprecated in setuptools)
- `aifc`, `audioop`, `sunau` (deprecated in Python 3.13)
- `is_sparse` (deprecated in scikit-learn / pandas)
- `librosa` “empty filters detected” warning  

These do **not affect correctness**.  
You can silence them by extending `pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
filterwarnings =
    ignore:pkg_resources is deprecated:DeprecationWarning
    ignore:'aifc' is deprecated:DeprecationWarning
    ignore:'audioop' is deprecated:DeprecationWarning
    ignore:'sunau' is deprecated:DeprecationWarning
    ignore:is_sparse is deprecated:DeprecationWarning
    ignore:Empty filters detected in mel frequency basis:UserWarning
```

---

## ✅ What is Tested

- **Feature extraction**: ensures MFCCs have correct shape and are stored in DataFrames.  
- **Data extraction**: verifies archive CSVs + metadata mapping works.  
- **Preprocessing**: confirms splits, PCA, scaling, and label encoding.  
- **Model building**: checks candidate models dictionary is built.  
- **Training pipeline**: runs end-to-end, producing artifacts (`model.pkl`, `pca.pkl`, `scaler.pkl`, `label_encoder.pkl`, `metrics.csv`).  
- **Evaluation**: metrics dictionary + confusion matrix plot.  
- **Prediction pipeline**: loads artifacts, transforms features, saves predictions.  
- **Data I/O**: concatenates multiple CSVs correctly.

---

## 📊 Interpreting Results

- `PASSED`: functionality is working as expected.  
- `FAILED`: test detected a bug (check traceback).  
- `warnings`: safe to ignore, but can be silenced as shown above.  

---

✨ With all tests passing, you can be confident the pipeline works end-to-end.  
