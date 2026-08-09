from rocket2d.data.datasets import ImageDataset, get_images


def test_get_images_neu(neu_dataset_dir):
    paths, labels = get_images(neu_dataset_dir / "train")
    assert len(paths) == 12
    assert set(labels) == {"crazing", "scratches"}


def test_image_dataset_neu(neu_dataset_dir):
    ds = ImageDataset(root=neu_dataset_dir, dataset="neu", size=16, grayscale=True)
    assert set(ds.class_to_idx) == {"crazing", "scratches"}

    X, y = ds.prepare()
    assert X.shape == (24, 16, 16)
    assert y.shape == (24,)
    assert X.min() >= 0.0 and X.max() <= 1.0

    assert ds.validate() is True


def test_image_dataset_xray(xray_dataset_dir):
    ds = ImageDataset(root=xray_dataset_dir, dataset="xray", size=16, grayscale=True)
    assert ds.class_to_idx == {"NORMAL": 0, "PNEUMONIA": 1}
    X, y = ds.prepare()
    assert X.shape[0] == 36


def test_image_dataset_dtd(dtd_dataset_dir):
    ds = ImageDataset(root=dtd_dataset_dir, dataset="dtd", size=16, grayscale=False)
    X, y = ds.prepare()
    assert X.shape == (12, 16, 16, 3)


def test_image_dataset_torch_channel_order(neu_dataset_dir):
    ds = ImageDataset(root=neu_dataset_dir, dataset="neu", size=16, grayscale=True)
    X, y = ds.torch()
    assert X.shape == (24, 1, 16, 16)
    assert y.shape == (24,)


def test_unknown_dataset_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ImageDataset(root=tmp_path, dataset="not-a-dataset")


def test_image_dataset_skips_unknown_class_in_split(neu_dataset_dir):
    # A class present only in "validation" (not in "train") must be skipped,
    # not crash the loader with a KeyError.
    import numpy as np
    from PIL import Image

    extra_dir = neu_dataset_dir / "validation" / "unseen_class"
    extra_dir.mkdir()
    arr = np.zeros((16, 16), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(extra_dir / "img_0.png")

    ds = ImageDataset(root=neu_dataset_dir, dataset="neu", size=16, grayscale=True)
    assert "unseen_class" not in ds.class_to_idx
    X, y = ds.prepare()
    assert X.shape[0] == 24  # unseen_class images are excluded, not counted
