import torch

from geomaps.models.unet import LightUNet, UNet


def test_light_unet_forward_shape():
    model = LightUNet(in_channels=3, classes=4)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 4, 64, 64)


def test_unet_forward_shape():
    model = UNet(in_channels=3, classes=2)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 2, 32, 32)
