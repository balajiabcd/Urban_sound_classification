# Test Suite (pytest)

## Layout
- `conftest.py` – shared fixtures (RNG, tiny dataset, mock librosa)
- `test_utils.py` – file/seed helpers
- `test_extract_data.py` – feature extraction & dataframe build (uses mocked `librosa`)
- `test_data_io.py` – CSV roundtrip
- `test_preprocess.py` – splits and transformers
- `test_models.py` – model factory & fit/predict
- `test_train.py` – end-to-end training pipeline (saves artifacts to tmp dir)
- `test_evaluate.py` – metric dictionary presence
- `test_predict.py` – single-sample prediction from saved artifacts

## How to run
```bash
pytest -q
```

> Tip: If your module paths differ, adjust the `from extract ...` / `from training ...` imports.
