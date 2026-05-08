import warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl

import hydra
from omegaconf import OmegaConf
from geomaps.utils.print import print_config

from geomaps import LOGGER
from geomaps.utils.exception import GeomapsError, ErrorCodes
from geomaps.configs.core import Config
from geomaps.configs import database_lib
from geomaps.cli import train, test, predict
from geomaps.constants.enums import TaskType

CONFIG_PATH = "configs/conf"  # Path of the directory where the config files are stored
CONFIG_NAME = "config" # basename of the file used for the config (here config.yaml which is in ../configs/conf/)

database_lib.register_configs()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.1")
def main(config: Config)-> None:

    try:

        if config.print_config is True:
            print_config(config)

        if config.deterministic is True:
            pl.seed_everything(config.seed, workers=True)

        if config.task == TaskType.TRAIN:
            train.train(config)
        elif config.task == TaskType.TEST:
            test.test(config)
        elif config.task == TaskType.PRED:
            predict.pred(config)
        else:
            LOGGER.error("ERROR: Wrong task type value from the input config")
            raise GeomapsError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Input configuration values are incorrect..")

    except GeomapsError as error:
        raise GeomapsError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training",
                            stack_trace=error)

    return 0


if __name__ == "__main__":
    main()
