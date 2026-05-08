from dataclasses import dataclass
from typing import Any, Optional
from omegaconf import MISSING


# Data augmentation transformation from Albumentations package
@dataclass
class BlurConf:
    _target_: str = "albumentations.Blur"
    blur_limit: int = 7
    always_apply: bool = False
    p: float = 0.5


@dataclass
class CLAHEConf:
    _target_: str = "albumentations.CLAHE"
    clip_limit: float = 4.0
    tile_grid_size: Any = (8, 8)
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ChannelDropoutConf:
    _target_: str = "albumentations.ChannelDropout"
    channel_drop_range: Any = (1, 1)
    fill_value: int = 0
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ChannelShuffleConf:
    _target_: str = "albumentations.ChannelShuffle"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ColorJitterConf:
    _target_: str = "albumentations.ColorJitter"
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.2
    always_apply: bool = False
    p: float = 0.5


@dataclass
class DownscaleConf:
    _target_: str = "albumentations.Downscale"
    scale_min: float = 0.25
    scale_max: float = 0.25
    interpolation: int = 0
    always_apply: bool = False
    p: float = 0.5


@dataclass
class EmbossConf:
    _target_: str = "albumentations.Emboss"
    alpha: Any = (0.2, 0.5)
    strength: Any = (0.2, 0.7)
    always_apply: bool = False
    p: float = 0.5


@dataclass
class EqualizeConf:
    _target_: str = "albumentations.Equalize"
    mode: str = "cv"
    by_channels: bool = True
    mask: Optional[Any] = None
    mask_params: Any = ()
    always_apply: bool = False
    p: float = 0.5


@dataclass
class FDAConf:
    _target_: str = "albumentations.FDA"
    reference_images: Any = MISSING  # List[Union[str, ndarray]]
    beta_limit: float = 0.1
    read_fn: Any = MISSING  # function
    always_apply: bool = False
    p: float = 0.5


@dataclass
class FancyPCAConf:
    _target_: str = "albumentations.FancyPCA"
    alpha: float = 0.1
    always_apply: bool = False
    p: float = 0.5


@dataclass
class FromFloatConf:
    _target_: str = "albumentations.FromFloat"
    dtype: str = "uint16"
    max_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 1.0


@dataclass
class GaussNoiseConf:
    _target_: str = "albumentations.GaussNoise"
    var_limit: Any = (10.0, 50.0)
    mean: int = 0
    per_channel: bool = True
    always_apply: bool = False
    p: float = 0.5


@dataclass
class GaussianBlurConf:
    _target_: str = "albumentations.GaussianBlur"
    blur_limit: Any = (3, 7)
    sigma_limit: int = 0
    always_apply: bool = False
    p: float = 0.5


@dataclass
class GlassBlurConf:
    _target_: str = "albumentations.GlassBlur"
    sigma: float = 0.7
    max_delta: int = 4
    iterations: int = 2
    always_apply: bool = False
    mode: str = "fast"
    p: float = 0.5


@dataclass
class HistogramMatchingConf:
    _target_: str = "albumentations.HistogramMatching"
    reference_images: Any = MISSING  # List[Union[str, ndarray]]
    blend_ratio: Any = (0.5, 1.0)
    read_fn: Any = MISSING  # function
    always_apply: bool = False
    p: float = 0.5


@dataclass
class HueSaturationValueConf:
    _target_: str = "albumentations.HueSaturationValue"
    hue_shift_limit: int = 20
    sat_shift_limit: int = 30
    val_shift_limit: int = 20
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ISONoiseConf:
    _target_: str = "albumentations.ISONoise"
    color_shift: Any = (0.01, 0.05)
    intensity: Any = (0.1, 0.5)
    always_apply: bool = False
    p: float = 0.5


@dataclass
class InvertImgConf:
    _target_: str = "albumentations.InvertImg"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class MedianBlurConf:
    _target_: str = "albumentations.MedianBlur"
    blur_limit: int = 7
    always_apply: bool = False
    p: float = 0.5


@dataclass
class MotionBlurConf:
    _target_: str = "albumentations.MotionBlur"
    blur_limit: int = 7
    always_apply: bool = False
    p: float = 0.5


@dataclass
class MultiplicativeNoiseConf:
    _target_: str = "albumentations.MultiplicativeNoise"
    multiplier: Any = (0.9, 1.1)
    per_channel: bool = False
    elementwise: bool = False
    always_apply: bool = False
    p: float = 0.5


@dataclass
class NormalizeConf:
    _target_: str = "albumentations.Normalize"
    mean: Any = (0.485, 0.456, 0.406)
    std: Any = (0.229, 0.224, 0.225)
    max_pixel_value: float = 255.0
    always_apply: bool = False
    p: float = 1.0


@dataclass
class PosterizeConf:
    _target_: str = "albumentations.Posterize"
    num_bits: int = 4
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RGBShiftConf:
    _target_: str = "albumentations.RGBShift"
    r_shift_limit: int = 20
    g_shift_limit: int = 20
    b_shift_limit: int = 20
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomBrightnessContrastConf:
    _target_: str = "albumentations.RandomBrightnessContrast"
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    brightness_by_max: bool = True
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomFogConf:
    _target_: str = "albumentations.RandomFog"
    fog_coef_lower: float = 0.3
    fog_coef_upper: int = 1
    alpha_coef: float = 0.08
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomGammaConf:
    _target_: str = "albumentations.RandomGamma"
    gamma_limit: Any = (80, 120)
    eps: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomRainConf:
    _target_: str = "albumentations.RandomRain"
    slant_lower: int = -10
    slant_upper: int = 10
    drop_length: int = 20
    drop_width: int = 1
    drop_color: Any = (200, 200, 200)
    blur_value: int = 7
    brightness_coefficient: float = 0.7
    rain_type: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomShadowConf:
    _target_: str = "albumentations.RandomShadow"
    shadow_roi: Any = (0, 0.5, 1, 1)
    num_shadows_lower: int = 1
    num_shadows_upper: int = 2
    shadow_dimension: int = 5
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomSnowConf:
    _target_: str = "albumentations.RandomSnow"
    snow_point_lower: float = 0.1
    snow_point_upper: float = 0.3
    brightness_coeff: float = 2.5
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomSunFlareConf:
    _target_: str = "albumentations.RandomSunFlare"
    flare_roi: Any = (0, 0, 1, 0.5)
    angle_lower: int = 0
    angle_upper: int = 1
    num_flare_circles_lower: int = 6
    num_flare_circles_upper: int = 10
    src_radius: int = 400
    src_color: Any = (255, 255, 255)
    always_apply: bool = False
    p: float = 0.5


@dataclass
class SharpenConf:
    _target_: str = "albumentations.Sharpen"
    alpha: Any = (0.2, 0.5)
    lightness: Any = (0.5, 1.0)
    always_apply: bool = False
    p: float = 0.5


@dataclass
class SolarizeConf:
    _target_: str = "albumentations.Solarize"
    threshold: int = 128
    always_apply: bool = False
    p: float = 0.5


@dataclass
class SuperpixelsConf:
    _target_: str = "albumentations.Superpixels"
    p_replace: float = 0.1
    n_segments: int = 100
    max_size: Optional[int] = 128
    interpolation: int = 1
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ToFloatConf:
    _target_: str = "albumentations.ToFloat"
    max_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 1.0


@dataclass
class ToGrayConf:
    _target_: str = "albumentations.ToGray"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ToSepiaConf:
    _target_: str = "albumentations.ToSepia"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class AffineConf:
    _target_: str = "albumentations.Affine"
    scale: Optional[Any] = None
    translate_percent: Optional[Any] = None
    translate_px: Optional[Any] = None
    rotate: Optional[Any] = None
    shear: Optional[Any] = None
    interpolation: int = 1
    cval: int = 0
    cval_mask: int = 0
    mode: str = "constant"
    fit_output: bool = False
    always_apply: bool = False
    p: float = 0.5


@dataclass
class CenterCropConf:
    _target_: str = "albumentations.CenterCrop"
    height: Any = MISSING
    width: Any = MISSING
    always_apply: bool = False
    p: float = 1.0


@dataclass
class CoarseDropoutConf:
    _target_: str = "albumentations.CoarseDropout"
    max_holes: int = 8
    max_height: int = 8
    max_width: int = 8
    min_holes: Optional[Any] = None
    min_height: Optional[Any] = None
    min_width: Optional[Any] = None
    fill_value: int = 0
    mask_fill_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class CropAndPadConf:
    _target_: str = "albumentations.CropAndPad"
    px: Optional[Any] = None
    percent: Optional[Any] = None
    pad_mode: str = "constant"
    pad_cval: int = 0
    pad_cval_mask: int = 0
    keep_size: bool = True
    sample_independently: bool = True
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1


@dataclass
class CropConf:
    _target_: str = "albumentations.Crop"
    x_min: int = 0
    y_min: int = 0
    x_max: int = 1024
    y_max: int = 1024
    always_apply: bool = False
    p: float = 1.0


@dataclass
class CropNonEmptyMaskIfExistsConf:
    _target_: str = "albumentations.CropNonEmptyMaskIfExists"
    height: Any = MISSING
    width: Any = MISSING
    ignore_values: Optional[Any] = None
    ignore_channels: Optional[Any] = None
    always_apply: bool = False
    p: float = 1.0


@dataclass
class ElasticTransformConf:
    _target_: str = "albumentations.ElasticTransform"
    alpha: int = 1
    sigma: int = 50
    alpha_affine: int = 50
    interpolation: int = 1
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    always_apply: bool = False
    approximate: bool = False
    p: float = 0.5


@dataclass
class FlipConf:
    _target_: str = "albumentations.Flip"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class GridDistortionConf:
    _target_: str = "albumentations.GridDistortion"
    num_steps: int = 5
    distort_limit: float = 0.3
    interpolation: int = 1
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class GridDropoutConf:
    _target_: str = "albumentations.GridDropout"
    ratio: float = 0.5
    unit_size_min: Optional[int] = None
    unit_size_max: Optional[int] = None
    holes_number_x: Optional[int] = None
    holes_number_y: Optional[int] = None
    shift_x: int = 0
    shift_y: int = 0
    random_offset: bool = False
    fill_value: int = 0
    mask_fill_value: Optional[int] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class HorizontalFlipConf:
    _target_: str = "albumentations.HorizontalFlip"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class LambdaConf:
    _target_: str = "albumentations.Lambda"
    image: Optional[Any] = None
    mask: Optional[Any] = None
    keypoint: Optional[Any] = None
    bbox: Optional[Any] = None
    name: Optional[Any] = None
    always_apply: bool = False
    p: float = 1.0


@dataclass
class LongestMaxSizeConf:
    _target_: str = "albumentations.LongestMaxSize"
    max_size: int = 1024
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1.0


@dataclass
class MaskDropoutConf:
    _target_: str = "albumentations.MaskDropout"
    max_objects: int = 1
    image_fill_value: int = 0
    mask_fill_value: int = 0
    always_apply: bool = False
    p: float = 0.5


@dataclass
class NoOpConf:
    _target_: str = "albumentations.NoOp"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class OpticalDistortionConf:
    _target_: str = "albumentations.OpticalDistortion"
    distort_limit: float = 0.05
    shift_limit: float = 0.05
    interpolation: int = 1
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class PadIfNeededConf:
    _target_: str = "albumentations.PadIfNeeded"
    min_height: Optional[int] = 1024
    min_width: Optional[int] = 1024
    pad_height_divisor: Optional[int] = None
    pad_width_divisor: Optional[int] = None
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 1.0


@dataclass
class PerspectiveConf:
    _target_: str = "albumentations.Perspective"
    scale: Any = (0.05, 0.1)
    keep_size: bool = True
    pad_mode: str = 'constant'
    pad_val: int = 0
    mask_pad_val: int = 0
    fit_output: bool = False
    interpolation: int = 1
    always_apply: bool = False
    p: float = 0.5


@dataclass
class PiecewiseAffineConf:
    _target_: str = "albumentations.PiecewiseAffine"
    scale: Any = (0.03, 0.05)
    nb_rows: int = 4
    nb_cols: int = 4
    interpolation: int = 1
    mask_interpolation: int = 1
    cval: int = 0
    cval_mask: int = 0
    mode: str = "constant"
    absolute_scale: bool = False
    keypoints_threshold: float = 0.01
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomCropConf:
    _target_: str = "albumentations.RandomCrop"
    height: Any = MISSING
    width: Any = MISSING
    always_apply: bool = False
    p: float = 1.0


@dataclass
class RandomCropNearBBoxConf:
    _target_: str = "albumentations.RandomCropNearBBox"
    max_part_shift: float = 0.3
    always_apply: bool = False
    p: float = 1.0


@dataclass
class RandomGridShuffleConf:
    _target_: str = "albumentations.RandomGridShuffle"
    grid: Any = (3, 3)
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomResizedCropConf:
    _target_: str = "albumentations.RandomResizedCrop"
    height: Any = MISSING
    width: Any = MISSING
    scale: Any = (0.08, 1.0)
    ratio: Any = (0.75, 1.3333333333333333)
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1.0


@dataclass
class RandomRotate90Conf:
    _target_: str = "albumentations.RandomRotate90"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomScaleConf:
    _target_: str = "albumentations.RandomScale"
    scale_limit: float = 0.1
    interpolation: int = 1
    always_apply: bool = False
    p: float = 0.5


@dataclass
class RandomSizedBBoxSafeCropConf:
    _target_: str = "albumentations.RandomSizedBBoxSafeCrop"
    height: Any = MISSING
    width: Any = MISSING
    erosion_rate: float = 0.0
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1.0


@dataclass
class RandomSizedCropConf:
    _target_: str = "albumentations.RandomSizedCrop"
    min_max_height: Any = MISSING
    height: Any = MISSING
    width: Any = MISSING
    w2h_ratio: float = 1.0
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1.0


@dataclass
class ResizeConf:
    _target_: str = "albumentations.Resize"
    height: Any = MISSING
    width: Any = MISSING
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1


@dataclass
class RotateConf:
    _target_: str = "albumentations.Rotate"
    limit: int = 90
    interpolation: int = 1
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class SafeRotateConf:
    _target_: str = "albumentations.SafeRotate"
    limit: int = 90
    interpolation: int = 1
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class ShiftScaleRotateConf:
    _target_: str = "albumentations.ShiftScaleRotate"
    shift_limit: float = 0.0625
    scale_limit: float = 0.1
    rotate_limit: int = 45
    interpolation: int = 1
    border_mode: int = 4
    value: Optional[Any] = None
    mask_value: Optional[Any] = None
    shift_limit_x: Optional[Any] = None
    shift_limit_y: Optional[Any] = None
    always_apply: bool = False
    p: float = 0.5


@dataclass
class SmallestMaxSizeConf:
    _target_: str = "albumentations.SmallestMaxSize"
    max_size: int = 1024
    interpolation: int = 1
    always_apply: bool = False
    p: float = 1


@dataclass
class TransposeConf:
    _target_: str = "albumentations.Transpose"
    always_apply: bool = False
    p: float = 0.5


@dataclass
class VerticalFlipConf:
    _target_: str = "albumentations.VerticalFlip"
    always_apply: bool = False
    p: float = 0.5
