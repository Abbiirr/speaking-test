"""IELTS Speaking Practice — centralized logging setup."""

import logging
import os

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

# Root logger for the package
_root_logger = logging.getLogger("speaking_test")
_root_logger.setLevel(logging.DEBUG)

# File handler — everything (DEBUG+), UTF-8 to handle phonetic symbols etc.
_file_handler = logging.FileHandler(
    os.path.join(_LOG_DIR, "app.log"),
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
_root_logger.addHandler(_file_handler)

# Console handler — errors only
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.ERROR)
_console_handler.setFormatter(
    logging.Formatter("%(levelname)s %(name)s: %(message)s")
)
_root_logger.addHandler(_console_handler)

# Prevent propagation to the root logger (avoids duplicate console output)
_root_logger.propagate = False
