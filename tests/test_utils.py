import os
import shutil
import pytest

utils = pytest.importorskip("src.extract.utils", reason="src.extract.utils not found")

def test_create_folder_if_not_exists(tmp_path):
    target = tmp_path / "nested" / "dir"
    if target.exists():
        shutil.rmtree(target)
    assert hasattr(utils, "create_folder_if_not_exists"), "utils.create_folder_if_not_exists missing"
    utils.create_folder_if_not_exists(target)
    assert target.exists()

def test_set_seed_idempotent():
    assert hasattr(utils, "set_seed"), "utils.set_seed missing"
    utils.set_seed(123)
    import numpy as np, random
    a = np.random.rand(3); b = random.random()
    utils.set_seed(123)
    a2 = np.random.rand(3); b2 = random.random()
    assert (a == a2).all() and b == b2
