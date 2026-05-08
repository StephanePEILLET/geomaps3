from dataclasses import dataclass
from typing import Optional, Tuple
from omegaconf import MISSING


@dataclass
class SGDConf:
    _target_: str = "torch.optim.SGD"
    lr: Optional[float] = None  # _RequiredParameter
    momentum: int = 0
    dampening: int = 0
    weight_decay: int = 0
    nesterov: bool = False


@dataclass
class AdadeltaConf:
    _target_: str = "torch.optim.Adadelta"
    lr: Optional[float] = None
    rho: float = 0.9
    eps: float = 1e-06
    weight_decay: int = 0


@dataclass
class AdagradConf:
    _target_: str = "torch.optim.Adagrad"
    lr: Optional[float] = None
    lr_decay: int = 0
    weight_decay: int = 0
    initial_accumulator_value: int = 0
    eps: float = 1e-10


@dataclass
class AdamConf:
    _target_: str = "torch.optim.Adam"
    lr: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: int = 0
    amsgrad: bool = False


@dataclass
class AdamaxConf:
    _target_: str = "torch.optim.Adamax"
    lr: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: int = 0


@dataclass
class AdamWConf:
    _target_: str = "torch.optim.AdamW"
    lr: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class ASGDConf:
    _target_: str = "torch.optim.ASGD"
    lr: Optional[float] = None
    lambd: float = 0.0001
    alpha: float = 0.75
    t0: float = 1000000.0
    weight_decay: int = 0


@dataclass
class LBFGSConf:
    _target_: str = "torch.optim.LBFGS"
    lr: Optional[float] = None
    max_iter: int = 20
    max_eval: Optional[int] = None
    tolerance_grad: float = 1e-07
    tolerance_change: float = 1e-09
    history_size: int = 100
    line_search_fn: Optional[str] = None


@dataclass
class RMSpropConf:
    _target_: str = "torch.optim.RMSprop"
    lr: Optional[float] = None
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: int = 0
    momentum: int = 0
    centered: bool = False


@dataclass
class RpropConf:
    _target_: str = "torch.optim.Rprop"
    lr: Optional[float] = None
    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-06, 50)


@dataclass
class SparseAdamConf:
    _target_: str = "torch.optim.SparseAdam"
    lr: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
