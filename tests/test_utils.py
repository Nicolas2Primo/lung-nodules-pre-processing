import numpy as np
from lidc_preprocessing_pipeline import resample, resample_mask, lung_mask


def test_resample_shapes():
    vol = np.ones((2, 2, 2))
    spacing = np.array([2.0, 2.0, 2.0])
    out = resample(vol, spacing, (1.0, 1.0, 1.0))
    assert out.shape == (4, 4, 4)


def test_resample_mask():
    mask = np.zeros((1, 1, 1), dtype=np.uint8)
    mask[0, 0, 0] = 1
    spacing = np.array([2.0, 2.0, 2.0])
    out = resample_mask(mask, spacing, (1.0, 1.0, 1.0))
    assert out.shape == (2, 2, 2)
    assert np.allclose(out.sum(), 8)


def test_lung_mask():
    vol = np.full((2, 2, 2), -400)
    mask = lung_mask(vol)
    assert mask.shape == vol.shape
    assert mask.dtype == np.uint8

