from src.training import models

def test_build_candidates():
    all_models = models.build_candidates()
    assert "knn_euclidean_k1" in all_models
    assert any("rf_" in k for k in all_models.keys())
