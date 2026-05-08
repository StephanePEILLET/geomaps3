import numpy as np
import pytest
import torch


@pytest.fixture
def numpy_sample():
    """A {'image': HxWxC, 'mask': HxWxC} sample with C=3 classes."""
    rng = np.random.default_rng(42)
    image = rng.random((32, 32, 3)).astype(np.float32)
    mask = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=(32, 32))]
    return {"image": image, "mask": mask}


@pytest.fixture
def torch_logits():
    """Random logits tensor (B=2, C=3, H=16, W=16)."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 16, 16)


@pytest.fixture
def torch_targets():
    """Random integer target tensor (B=2, H=16, W=16) in [0, 3)."""
    torch.manual_seed(1)
    return torch.randint(0, 3, (2, 16, 16))
