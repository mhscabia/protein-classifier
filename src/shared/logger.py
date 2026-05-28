import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(show_time=True, show_path=False, markup=True)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
