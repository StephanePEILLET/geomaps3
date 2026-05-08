from typing import (
    Dict,
    Tuple,
    Optional,
    List,
    Any
)
from dataclasses import field
from dataclasses import dataclass
from omegaconf import MISSING
from geomaps.configs.trainer import LightningTrainerConf
from geomaps.constants.enums import TaskType


@dataclass
class Files:
    output_folder: str = MISSING
    name_exp_log: str = MISSING
    version_name: str = MISSING
    model_filename: str = MISSING
    model_out_ext: Optional[str] = ".ckpt"


@dataclass
class DataModuleConf:
    # Files pointing to the data
    train_file: str = MISSING
    val_file: str = MISSING
    test_file: Optional[str] = None
    sampling_file: Optional[str] = None

    # Data features
    resolution: Optional[float] = 0.2
    class_labels: Optional[List[str]] = field(default_factory=lambda: [])

    # Dataset definition
    image_bands: Optional[List[int]] = field(default_factory=lambda: []) 
    mask_bands: Optional[List[int]] = field(default_factory=lambda: [])
    width: Optional[int] = None
    height: Optional[int] = None
    percentage_val: Optional[float] = 0.5
    deterministic: Optional[bool] = False

    # Dataloaders definition
    batch_size: Optional[int] = 1
    num_workers: Optional[int] = 4
    pin_memory: Optional[bool] = True
    subset: Optional[bool] = True
    seed: Optional[int] = 42
    drop_last: Optional[bool] = False
    get_sample_info: Optional[bool] = False


@dataclass
class TransformsConf:
    train: List = field(default_factory=lambda: [])
    val: List = field(default_factory=lambda: [])
    test: Optional[List] = None


@dataclass
class Config:
    files: Files = MISSING
    datamodule: DataModuleConf = MISSING
    transforms: TransformsConf = TransformsConf()
    model: Any = MISSING
    callbacks: Any = MISSING # field(default_factory=lambda: [])
    logger: Optional[Any] = None
    optimizer: Any = MISSING
    loss: Any = MISSING
    lr: float = 0.001
    scheduler: Optional[Any] = None
    profiler: Optional[Any] = None
    trainer: LightningTrainerConf = LightningTrainerConf()
    seed: Optional[int] = 42
    run_test: Optional[bool] = False
    run_pred: Optional[bool] = False
    print_config: Optional[bool] = False
    debug: Optional[bool] = False
    ignore_warnings: Optional[bool] = True
    deterministic: Optional[bool] = True
    checkpoint: Optional[str] = None
    task: TaskType = MISSING
