"""Filesystem and raster-band guards used by the DL pipeline."""
import os
from typing import List

import rasterio
from omegaconf.listconfig import ListConfig

from geomaps import LOGGER
from geomaps.utils.exception import ErrorCodes, GeomapsError


def files_exist(list_of_file):
    for element in list_of_file:
        if isinstance(element, str):
            if not os.path.isfile(element):
                raise GeomapsError(ErrorCodes.ERR_FILE_NOT_EXIST,
                                 f"the file {element} doesn't exists")
        else:
            for sub_element in element:
                if not os.path.isfile(sub_element):
                    raise GeomapsError(ErrorCodes.ERR_FILE_NOT_EXIST,
                                     f"the file {sub_element} doesn't exists")


def check_files(list_files):
    try:
        files_exist(list_files)
    except GeomapsError as error:
        raise GeomapsError(ErrorCodes.ERR_TRAINING_ERROR,
                         "something went wrong during training configuration",
                         stack_trace=error)


def check_raster_bands(raster_band, proposed_bands):
    """Check that proposed bands all exist among the raster bands."""
    if isinstance(proposed_bands, (List, ListConfig)) and len(proposed_bands) >= 1:
        if not all(band in raster_band for band in proposed_bands):
            LOGGER.error('ERROR: the bands in the configuration file do not correspond '
                         'to the available bands in the image.')
            raise GeomapsError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                             "The input parameters mask_bands and pred_bands are incorrect.")
    else:
        LOGGER.error('ERROR: bands must be a list with a length greater than 1.')
        raise GeomapsError(ErrorCodes.ERR_JSON_SCHEMA_ERROR,
                         "The input parameters mask_bands and pred_bands are incorrect.")


def raster_bands_exist(raster, list_of_band):
    try:
        rasters = [raster] if isinstance(raster, str) else raster
        for r in rasters:
            with rasterio.open(r) as src:
                bands_count = src.count
                for band in list_of_band:
                    if band > bands_count:
                        raise GeomapsError(ErrorCodes.ERR_RASTER_BAND_NOT_EXIST,
                                         f"the band {band} from raster {r} doesn't exists")
    except rasterio.errors.RasterioIOError as rioe:
        raise GeomapsError(ErrorCodes.ERR_IO,
                         f"Geomaps encountered an error while opening raster {raster}", stack_trace=rioe)
