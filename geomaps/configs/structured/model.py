from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
from omegaconf import MISSING


# Models Geomaps
@dataclass
class BaseModelConf:
    in_channels: Optional[int] = None
    classes: Optional[int] = None


@dataclass
class UnetConf(BaseModelConf):
    _target_: str = "geomaps.models.unet.UNet"
    in_channels: Optional[int] = 4
    classes: Optional[int] = 4
    bilinear: bool = True
    batch_norm: bool = True


@dataclass
class LightUnetGeomapsConf(BaseModelConf):
    _target_: str = "geomaps.models.unet.LightUNet"
    bilinear: bool = True
    batch_norm: bool = False


@dataclass
class MediumUnetGeomapsConf(BaseModelConf):
    _target_: str = "geomaps.models.unet.MediumUNet"
    bilinear: bool = True
    batch_norm: bool = False


@dataclass
class LargeUnetGeomapsConf(BaseModelConf):
    _target_: str = "geomaps.models.unet.LargeUNet"
    bilinear: bool = True
    batch_norm: bool = False


@dataclass
class ResBlockNetGeomapsConf(BaseModelConf):
    _target_: str = "geomaps.models.resnet.ResBlockNet"
    model_name: str = "rnet_16_basicblock_8"
    bilinear: bool = True


@dataclass
class DeepLabGeomapsConf(BaseModelConf):
    _target_: str = "geomaps.models.deeplabv3p.DeeplabV3p"


# Models from Segmentation Models Pytorch (SMP) package
@dataclass
class UnetSmpConf:
    _target_: str = "segmentation_models_pytorch.Unet"
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    decoder_use_batchnorm: bool = True
    decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16)
    decoder_attention_type: Optional[str] = None
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[Any] = None  # Union[str, callable, NoneType]
    aux_params: Optional[Any] = None


@dataclass
class DeepLabV3SmpConf:
    _target_: str = "segmentation_models_pytorch.DeepLabV3"
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    decoder_channels: int = 256
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[str] = None
    upsampling: int = 8
    aux_params: Optional[Any] = None


@dataclass
class DeepLabV3PlusSmpConf:
    _target_: str = "segmentation_models_pytorch.DeepLabV3Plus"
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    encoder_output_stride: int = 16
    decoder_channels: int = 256
    decoder_atrous_rates: Tuple[int, int, int] = (12, 24, 36)
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[str] = None
    upsampling: int = 4
    aux_params: Optional[Any] = None


@dataclass
class FPNSmpConf:
    _target_: str = "segmentation_models_pytorch.FPN"
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    decoder_pyramid_channels: int = 256
    decoder_segmentation_channels: int = 128
    decoder_merge_policy: str = "add"
    decoder_dropout: float = 0.2
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[str] = None
    upsampling: int = 4
    aux_params: Optional[Any] = None


@dataclass
class LinknetSmpConf:
    _target_: str = "segmentation_models_pytorch.Linknet"
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    decoder_use_batchnorm: bool = True
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[Any] = None  # Union[str, callable, NoneType]
    aux_params: Optional[Any] = None


@dataclass
class MAnetSmpConf:
    _target_: str = "segmentation_models_pytorch.MAnet"
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    decoder_use_batchnorm: bool = True
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    decoder_pab_channels: int = 64
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[Any] = None
    aux_params: Optional[Any] = None


@dataclass
class PANSmpConf:
    _target_: str = "segmentation_models_pytorch.PAN"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    encoder_dilation: bool = True
    decoder_channels: int = 32
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[Any] = None  # Union[str, callable, NoneType]
    upsampling: int = 4
    aux_params: Optional[Any] = None


@dataclass
class PSPNetSmpConf:
    _target_: str = "segmentation_models_pytorch.PSPNet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    encoder_depth: int = 3
    psp_out_channels: int = 512
    psp_use_batchnorm: bool = True
    psp_dropout: float = 0.2
    in_channels: Optional[int] = None
    classes: Optional[int] = None
    activation: Optional[Any] = None  # Union[str, callable, NoneType]
    upsampling: int = 8
    aux_params: Optional[Any] = None
