import torch

from rocket2d.config import Config, set_seed


def test_set_seed_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_config_dataset_dirs():
    config = Config(base_data_dir="/tmp/data")
    dirs = config.dataset_dirs
    assert dirs["neu"] == "/tmp/data/neu"
    assert dirs["xray"] == "/tmp/data/xray"
    assert dirs["dtd"] == "/tmp/data/dtd"
    assert set(dirs) == {"neu", "xray", "lc", "dtd"}
