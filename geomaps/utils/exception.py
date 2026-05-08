"""
Module to define custom exception
"""
from enum import Enum, unique, auto
import traceback


class GeomapsError(Exception):
    """
    Custom Exception for Geomaps project
    """

    def __init__(self, error_code, message='', stack_trace=None, *args, **kwargs):
        if not isinstance(error_code, ErrorCodes):
            msg = 'Error code passed in the error_code param must be of type {0}'
            raise GeomapsError(ErrorCodes.ERR_INCORRECT_ERRCODE, msg, args=[ErrorCodes.__class__.__name__])

        self.error_code = error_code
        self.traceback = traceback.format_exc()
        self.stack_trace = stack_trace if stack_trace is not None else ""

        try:
            msg = f"{str(message)} \n error code : {str(self.error_code)} \n " \
                  f"trace back: {str(self.traceback)} \n" \
                  f" stack trace: {str(self.stack_trace)} \n" \
                  f" {str(args)} \n str{str(kwargs)}"
        except (IndexError, KeyError):
            msg = f"{error_code.name},  {message}"

        super().__init__(msg)


@unique
class ErrorCodes(Enum):
    """Error codes for Geomaps DL pipeline exceptions."""

    ERR_INCORRECT_ERRCODE = auto()
    ERR_COORDINATE_REFERENCE_SYSTEM = auto()
    ERR_DRIVER_COMPATIBILITY = auto()
    ERR_RASTER_BAND_NOT_EXIST = auto()
    ERR_FILE_NOT_EXIST = auto()
    ERR_DIR_NOT_EXIST = auto()
    ERR_IO = auto()
    ERR_JSON_SCHEMA_ERROR = auto()
    INVALID_DATASET_PATH = auto()
    ERR_TRAINING_ERROR = auto()
    ERR_CALLBACK_ERROR = auto()
    ERR_CONFIG_HYDRA_ERROR = auto()

    def __str__(self):
        return f"name of error: {self.name}, code value of error: {self.value}"
