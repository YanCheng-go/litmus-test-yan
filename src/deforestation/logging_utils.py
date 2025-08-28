from __future__ import annotations
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(logs_dir: str = "./logs", name: str = "deforestation") -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = RotatingFileHandler(os.path.join(logs_dir, f"{name}.log"), maxBytes=2_000_000, backupCount=3)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
