import os
from pathlib import Path
import pytest

def test_create_folder_if_not_exists(tmp_path):
    # import lazily to avoid import errors if path differs
    from extract import utils
    out = tmp_path / "newdir"
    assert not out.exists()
    utils.create_folder_if_not_exists(out)
    assert out.exists() and out.is_dir()

def test_set_seed_idempotent():
    from extract import utils
    utils.set_seed(42)
    a = utils.np_random_rand()
    utils.set_seed(42)
    b = utils.np_random_rand()
    assert a == b
