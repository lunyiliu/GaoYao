import logging
import sys

def _setup():
    logger = logging.getLogger('eval_logger')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = _setup()
