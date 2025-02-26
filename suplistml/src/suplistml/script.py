__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import importlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ipython():
    ipython = importlib.import_module("IPython")
    return ipython is not None and ipython.get_ipython() is not None


def resolve_module_name(name):
    if name == "__main__":
        return sys.modules[name].__spec__.name
    return name


def make_output_path(module_name):
    run_id = int(time.time())
    user = os.getenv("USER", "user")
    output_path = Path(tempfile.gettempdir()) / f"{user}_{module_name}" / f"run+{run_id}"
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using {output_path=}")
    return output_path


def setup_logging(output_path: Path):
    log_path = output_path / "main.log"
    file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler()

    formatter = logging.Formatter(fmt=logging.BASIC_FORMAT)
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger.setLevel(logging.WARNING)
    logging.getLogger("suplistml").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
