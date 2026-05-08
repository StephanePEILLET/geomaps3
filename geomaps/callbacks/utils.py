import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import rasterio
from rasterio.warp import aligned_target
from typing import Optional, Union, Any
from pathlib import Path
from geomaps.constants.enums import WriteIntervals
from geomaps.data.image import TypeConverter
from geomaps.data.raster import ndarray_to_affine

THRESHOLD = 0.5
SAVE_TOP_K = 3


class LightningCheckpoint(ModelCheckpoint):
    """
        Custom version of the built-in lightning callback ModelCheckpoint. In addition to the original one,
        it allows to create automatically an output directory with a new version name at each training in order
        to differentiate easily checkpoints between experiences.  
    """
    def __init__(
        self,
        monitor: str,
        dirpath: Union[Path, str],
        version: str,
        save_top_k: Optional[int] = SAVE_TOP_K,
        mode: Optional[str] = "min",
        filename: Optional[str]=None,
        **kwargs: Any,
        )-> None:

        self.monitor = monitor
        self.dirpath = dirpath
        self.version = version

        if filename is None:
            filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
        elif save_top_k > 1:
            filename = os.path.splitext(filename)[0] + "-{epoch:02d}-{" + monitor + ":.2f}"
        else:
            filename = os.path.splitext(filename)[0]

        self.output_path = self.check_dirpath()

        super().__init__(monitor=self.monitor,
                         dirpath=self.output_path,
                         filename=filename,
                         save_top_k=save_top_k,
                         mode=mode,
                         **kwargs)

    def check_dirpath(self):
        """
            Check if the input dirpath exists 
        """
        path = os.path.join(self.dirpath, self.version)

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        return super().on_load_checkpoint(trainer, pl_module, callback_state)

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


class HistorySaver(pl.Callback):
    
    def __init__(self):
        super().__init__()
        self.idx_loggers = None

    def on_fit_start(self, trainer, pl_module):
        if pl_module.logger is not None:
            idx_csv_loggers = [idx for idx, logger in enumerate(pl_module.logger.experiment)\
                if isinstance(logger, pl.loggers.csv_logs.ExperimentWriter)]
            if idx_csv_loggers :
                self.idx_loggers = {'val': idx_csv_loggers[0], 'test': idx_csv_loggers[-1]}
            else:
                self.idx_loggers = {'val': None, 'test': None}

    def on_test_start(self, trainer, pl_module):
        if self.idx_loggers is None:
            self.on_fit_start(trainer, pl_module)
        return super().on_test_start(trainer, pl_module)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger_idx = self.idx_loggers['val']
        metric_collection = {key: value.cpu().numpy() for key, value in pl_module.val_epoch_metrics.items()}
        metric_collection['loss'] = pl_module.val_epoch_loss.cpu().numpy()
        metric_collection['learning rate'] = pl_module.hparams.lr  # TODO Add learning rate not from hparams
        pl_module.logger[logger_idx].experiment.log_metrics(metric_collection, pl_module.current_epoch)
        pl_module.logger[logger_idx].experiment.save()

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        logger_idx = self.idx_loggers['test']
        metric_collection = {key: value.cpu().numpy() for key, value in pl_module.test_epoch_metrics.items()}
        metric_collection['loss'] = pl_module.test_epoch_loss.cpu().numpy()
        pl_module.logger[logger_idx].experiment.log_metrics(metric_collection, pl_module.current_epoch)
        pl_module.logger[logger_idx].experiment.save()


class CustomPredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        output_dir: Union[Path, str],
        output_type: str,
        write_interval: WriteIntervals,
        threshold: Optional[float] = THRESHOLD,
        img_size_pixel: Optional[int] = None,
        driver: Optional[str] = "GTiff",
        compress: Optional[str] = "LZW",
        tiled: Optional[bool] = True
        ):

        super().__init__(write_interval)

        self.output_dir = output_dir
        self.output_type = output_type
        self.threshold = threshold
        self.driver = driver
        self.compress = compress
        self.tiled = tiled
        self.meta = None
        self.img_size_pixel = img_size_pixel

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_predict_start(self, trainer, pl_module):
        if self.img_size_pixel is None:
            self.img_size_pixel = min(trainer.datamodule.sample_dims['image'][0],
                                      trainer.datamodule.sample_dims['image'][1])

        self.gdal_options = {"compress": self.compress,
                             "tiled": self.tiled,
                             "blockxsize": self.img_size_pixel,
                             "blockysize": self.img_size_pixel}

        self.meta = trainer.datamodule.meta["test"]
        self.meta["driver"] = self.driver
        self.meta["dtype"] = "uint8" if self.output_type in ["uint8", "bit"] else "float32"
        self.meta["count"] = trainer.datamodule.num_classes
        self.meta["width"] = self.img_size_pixel
        self.meta["height"] = self.img_size_pixel
        
        if self.output_type == "bit":
            self.gdal_options["bit"] = 1
        return super().on_predict_start(trainer, pl_module)  

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        probas, filenames, affines = prediction["proba"], prediction["filename"], prediction["affine"]
    
        # Pass prediction and their transformations on CPU
        probas = probas.cpu().numpy()
        affines = affines.cpu().numpy()

        for proba, filename, affine in zip(probas, filenames, affines):
            output_file = os.path.join(self.output_dir, filename)
            self.meta["transform"] = ndarray_to_affine(affine)
            self.meta["transform"], _, _ = aligned_target(self.meta["transform"],
                                                          self.meta["width"],
                                                          self.meta["height"],
                                                          trainer.datamodule.resolution["test"])

            with rasterio.open(output_file, "w", **self.meta, **self.gdal_options) as src:
                converter = TypeConverter()
                pred = converter.from_type("float32").to_type(self.output_type).convert(proba,
                                                                                        threshold=self.threshold)
                src.write(pred)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)


# Check size of tensors in forward pass
class CheckBatchGradient(pl.Callback):

    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()

        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)

        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")
