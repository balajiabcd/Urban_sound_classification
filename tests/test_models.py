import numpy as np

def test_get_model_known_names():
    from training import models
    for name in ["logreg", "svm", "rf", "knn", "xgb", "gb"]:
        m = models.get_model(name)
        assert hasattr(m, "fit") and hasattr(m, "predict")

def test_model_fit_predict(tiny_classification_data):
    from training import models
    X, y = tiny_classification_data
    model = models.get_model("rf")
    model.fit(X, y)
    yhat = model.predict(X[:5])
    assert yhat.shape == (5,)
