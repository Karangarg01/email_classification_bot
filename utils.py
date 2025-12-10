import logging
import os
from typing import Dict, Tuple

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "email_bot.log")


def get_logger(name: str = "email_bot") -> logging.Logger:
    """
    Returns a configured logger that writes to logs/email_bot.log
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def validate_prediction(true_label: str, predicted_label: str) -> Tuple[str, str]:
    """
    Returns (status, comment) based on whether prediction matches true label.
    Status: 'OK' if correct, 'WARNING' if different.
    """
    if true_label == "unknown":
        return "WARNING", "Unknown true label (from filename)."

    if predicted_label == true_label:
        return "OK", "Prediction matches true label."
    else:
        return "WARNING", f"Predicted '{predicted_label}' but true label is '{true_label}'."