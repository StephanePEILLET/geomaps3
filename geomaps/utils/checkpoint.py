import os
import numpy as np
from typing import Union, Optional
from pathlib import Path


def get_path_best_ckpt(
    ckpt_folder: Union[Path, str],
    monitor: Optional[str] = "val_loss",
    mode: Optional[str] = "min"
    )-> str:
    """
    Function to pick the best checkpoint in an input folder according to a chosen metric.
    This function is used during the training phase when multiple checkpoints are saved.
    Here we supposed that the metric name and value which led to the checkpoint creation are 
    in the name of the file (eg. name_file-epoch=48-val_miou=12.08.ckpt) 

    Parameters
    ----------
    ckpt_folder : str
        Path to the directory where the checkpoints are stored.
    monitor: str
        Metric to use for checkpoint selection. Default "val_loss" (Loss during the validation phase)
    mode: str
        Mode to do the determination between checkpoint metric values. Default "min".
        (With val_loss as monitored metric we want the min of the val_loss retrieved in the nameof each checpoint)

    Returns
    -------
    str
        Path of the best checkpoint according to the monitored metric.
    """

    best_ckpt_path = None
    list_ckpt = os.listdir(ckpt_folder)

    if len(list_ckpt) == 1:
        best_ckpt_path = list_ckpt[0]

    else:
        list_ckpt = [x for x in list_ckpt if monitor in x]

        get_value_monitor = lambda x : float(x.split(monitor)[-1][1: 5])

        value_ckpt = np.array([get_value_monitor(x) for x in list_ckpt ])

        if mode == "min":
            best_ckpt_path = list_ckpt[np.argmin(value_ckpt)]
        else:
            best_ckpt_path = list_ckpt[np.argmax(value_ckpt)]

    return os.path.join(ckpt_folder, best_ckpt_path)
