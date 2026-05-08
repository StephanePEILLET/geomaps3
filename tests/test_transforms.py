import numpy as np
import torch

from geomaps.data.transforms import Compose, Rotation90, ToDoubleTensor


def test_rotation90_preserves_shape(numpy_sample):
    out = Rotation90()(**numpy_sample)
    assert out["image"].shape == numpy_sample["image"].shape
    assert out["mask"].shape == numpy_sample["mask"].shape


def test_to_double_tensor_returns_torch_tensors(numpy_sample):
    out = ToDoubleTensor()(**numpy_sample)
    assert isinstance(out["image"], torch.Tensor)
    assert isinstance(out["mask"], torch.Tensor)
    assert out["image"].dtype == torch.float32
    assert out["image"].shape[0] == 3  # CHW
    assert out["mask"].shape[0] == 3


def test_compose_chains_transforms(numpy_sample):
    pipeline = Compose([Rotation90(), ToDoubleTensor()])
    out = pipeline(**numpy_sample)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 32, 32)
