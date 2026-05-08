import os
import torch
import pytorch_lightning as pl
from geomaps import LOGGER
from geomaps.utils.exception import GeomapsError, ErrorCodes
from geomaps.configs.core import Config
from geomaps.utils.instantiate import (
    instantiate_datamodule,
    instantiate_module,
    instantiate_trainer
)
from geomaps.utils.checkpoint import get_path_best_ckpt
from geomaps.configs.core import Config


def test(config: Config)-> None:
    try:
        if config.datamodule.test_file is None:
            raise GeomapsError(ErrorCodes.ERR_CONFIG_HYDRA_ERROR, "Test task need test_file parameter to be defined.")

        datamodule = instantiate_datamodule(config=config.datamodule,
                                            transform_config=config.transforms)

        module = instantiate_module(config=config, datamodule=datamodule)

        trainer = instantiate_trainer(config=config)

        if os.path.splitext(config.checkpoint)[-1] == ".pth":
            # Load model weights into the model of the seg module
            path_saved_pth = os.path.join(config.checkpoint)
            best_model_state_dict = torch.load(path_saved_pth)
            module.model.load_state_dict(state_dict=best_model_state_dict)
            LOGGER.info(f"Test with .pth file :{path_saved_pth}")

        trainer.test(model=module,
                     datamodule=datamodule,
                     ckpt_path=config.checkpoint)

    except GeomapsError as error:
        LOGGER.error("ERROR: Something went wrong during the fit step of the training")
        raise GeomapsError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training",
                            stack_trace=error)

    return 0