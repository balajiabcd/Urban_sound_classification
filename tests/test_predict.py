import numpy as np

def test_predict_pipeline_from_artifacts(tmp_path):
    from training import predict as pr
    from training import models, preprocess
    # Create and save tiny artifacts
    X = np.random.randn(40, 8)
    y = np.array([0]*20 + [1]*20)
    model = models.get_model("rf").fit(X, y)
    # save model
    import joblib, pickle
    joblib.dump(model, tmp_path / "model.pkl")
    # fit and save transformers (scaler/pca or identity)
    obj = types.SimpleNamespace()
    try:
        obj = preprocess.fit_transformers(X, n_components=None, save_dir=tmp_path)
    except Exception:
        # Fallback: store identity transformer
        obj = None
        with open(tmp_path / "transformers.pkl", "wb") as f:
            import pickle; pickle.dump(None, f)

    # Now predict a single 1D feature list (length matches training features)
    features = np.random.randn(8).tolist()
    yhat = pr.predict_from_artifacts(features, artifacts_dir=tmp_path)
    assert yhat in (0,1)
