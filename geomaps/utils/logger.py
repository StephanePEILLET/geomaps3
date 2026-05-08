"""Logger helpers for the Geomaps DL pipeline."""
import logging


class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    magenta = "\x1b[35;1m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "'%(asctime)s \t %(name)s  \t [%(levelname)s | ''%(filename)s:%(lineno)s] > %(message)s'"

    FORMATS = {
        logging.DEBUG: magenta + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_new_logger(name):
    if name in logging.root.manager.loggerDict:
        raise Exception(f"{name} exists already")
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    return log


def get_simple_handler(level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    return ch


def get_stream_handler(level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter())
    return ch
