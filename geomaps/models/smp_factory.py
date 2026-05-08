from typing import Optional

import segmentation_models_pytorch as smp
import torch


class SegmentationModelFactory:
    def __init__(
        self,
        in_chans: int = 3,
        classes: int = 15,
        model_name: Optional[str] = "unet",
        encoder_name: Optional[str] = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
    ) -> None:

        self.model_name: str = model_name
        self.encoder_name: str = encoder_name
        self.encoder_weights: str = encoder_weights
        self.in_chans: int = in_chans
        self.classes: int = classes

    def get(self) -> torch.nn.Module:
        model: smp.base.model = smp.create_model(
            arch=self.model_name,
            encoder_name=self.encoder_name,
            classes=self.classes,
            in_channels=self.in_chans,
            encoder_weights=self.encoder_weights,
        )
        return model
