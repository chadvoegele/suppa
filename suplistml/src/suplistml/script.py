__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import argparse
import importlib
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from argparse import Action
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ipython():
    ipython = importlib.import_module("IPython")
    return ipython is not None and ipython.get_ipython() is not None


def resolve_module_name(name):
    if name == "__main__":
        return sys.modules[name].__spec__.name
    return name


def make_output_path(dir: Path):
    run_id = int(time.time())
    user = os.getenv("USER", "user")
    output_path = Path(tempfile.gettempdir()) / user / f"run+{run_id}" / dir
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using {output_path=}")
    return output_path


def setup_logging(output_path: Path = None):
    stdout_handler = logging.StreamHandler()

    formatter = logging.Formatter(fmt=logging.BASIC_FORMAT)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(stdout_handler)

    if output_path is not None:
        log_path = output_path / "main.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.WARNING)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger("suplistml").setLevel(log_level)
    logging.getLogger("__main__").setLevel(log_level)


class JsonAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        value = json.loads(values)
        setattr(namespace, self.dest, value)


def _add_argument(parser, parameter):
    primitive_types = (int, str, Path)

    add_argument_args = {}

    if parameter.default is parameter.empty:
        add_argument_args["required"] = True
    else:
        add_argument_args["default"] = parameter.default

    if parameter.annotation in primitive_types:
        add_argument_args["type"] = parameter.annotation
    else:
        add_argument_args["action"] = JsonAction

    parser.add_argument(f"--{parameter.name}", **add_argument_args)


def _build_parser_for_function(function, args=None):
    parser = argparse.ArgumentParser(exit_on_error=False)
    signature = inspect.signature(function)
    parameters = signature.parameters

    for parameter in parameters.values():
        _add_argument(parser, parameter)

    return parser


def _run_function(function, args=None):
    parser = _build_parser_for_function(function, args)
    args = parser.parse_args(args)
    args = _replace_auto_paths(args, function)
    return function(**vars(args))


def _replace_auto_paths(args, function):
    for name, value in vars(args).items():
        if isinstance(value, Path) and str(value) == "__AUTO__":
            resolved_path = make_output_path(Path(function.__module__) / function.__name__ / name)
            setattr(args, name, resolved_path)
    return args


def script_main(globals, args=None):
    module_name = globals["__spec__"].name

    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument("--name", required=True)

    script_args, rest_args = parser.parse_known_args(args)
    name = script_args.name

    module = importlib.import_module(module_name)

    if not hasattr(module, name):
        raise ValueError(f"{name} not found in {module_name}")

    obj = getattr(module, name)

    if not inspect.isfunction(obj):
        raise ValueError(f"Expected {name} to be a function but was {type(obj)}")

    return _run_function(function=obj, args=rest_args)
