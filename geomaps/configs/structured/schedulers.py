from dataclasses import dataclass
from typing import Any, List, Optional
from omegaconf import MISSING


@dataclass
class ReduceLROnPlateauConf:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-08
    verbose: bool = False


@dataclass
class CosineAnnealingLRConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = MISSING
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class CosineAnnealingWarmRestartsConf:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    T_0: int = MISSING
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class CyclicLRConf:
    _target_: str = "torch.optim.lr_scheduler.CyclicLR"
    base_lr: float = MISSING
    max_lr: float = MISSING
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = "triangular"
    gamma: float = 1.0
    scale_fn: Optional[str] = None
    scale_mode: str = "cycle"
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.9
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class ExponentialLRConf:
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = MISSING
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class LambdaLRConf:
    _target_: str = "torch.optim.lr_scheduler.LambdaLR"
    lr_lambda: Any = MISSING
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class MultiplicativeLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiplicativeLR"
    lr_lambda: Any = MISSING
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class MultiStepLRConf:
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: List[int] = MISSING
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class OneCycleLRConf:
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    max_lr: float = MISSING
    total_steps: Optional[int] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = MISSING
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False
