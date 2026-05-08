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


def train(config: Config)-> None:

    try:

        datamodule = instantiate_datamodule(config=config.datamodule,
                                            transform_config=config.transforms)

        module = instantiate_module(config=config, datamodule=datamodule)

        trainer = instantiate_trainer(config=config)

        trainer.fit(model=module, datamodule=datamodule)

        if config.datamodule.test_file is not None and (config.run_test or config.run_pred):

            path_best_monitor_ckpt = None
            # Vérification de l'existence de la callback ckpt saving
            if config.files.model_out_ext == ".ckpt":
                if config.callbacks.custom_ckpt.dirpath is None:
                    raise GeomapsError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training")
                path_ckpt = os.path.join(config.callbacks.custom_ckpt.dirpath, config.callbacks.custom_ckpt.version)
                path_best_monitor_ckpt = get_path_best_ckpt(ckpt_folder=path_ckpt,
                                                            monitor=config.callbacks.custom_ckpt.monitor,
                                                            mode=config.callbacks.custom_ckpt.mode)

            elif config.files.model_out_ext == ".pth":
                # Load model weights into the model of the seg module
                path_saved_pth = os.path.join(config.files.output_folder,
                                              config.files.name_exp_log,
                                              config.files.version_name,
                                              config.files.model_filename)

                best_model_state_dict = torch.load(path_saved_pth)
                module.model.load_state_dict(state_dict=best_model_state_dict)
                LOGGER.info(f"Test with .pth file :{path_saved_pth}")

            if config.run_test is True:
                trainer.test(model=module,
                             datamodule=datamodule,
                             ckpt_path=path_best_monitor_ckpt)

            elif config.run_pred is True:
                trainer.predict(model=module,
                                datamodule=datamodule,
                                ckpt_path=path_best_monitor_ckpt)

    except GeomapsError as error:
        LOGGER.error("ERROR: Something went wrong during the fit step of the training")
        raise GeomapsError(ErrorCodes.ERR_TRAINING_ERROR,
                            "ERROR: Something went wrong during the fit step of the training",
                            stack_trace=error)

    return 0

