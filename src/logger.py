"""
logger.py

Centralized logging configuration.
"""

import logging
import os
import sys

from src.config import config


def setup_logger(name: str = "wav2vec2") -> logging.Logger:
    """
    Create and configure the application logger.

    Returns
    -------
    logging.Logger
    """

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(config.OUTPUT_DIR, "pipeline.log"),
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()