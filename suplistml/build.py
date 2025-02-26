#!/usr/bin/env python
__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import argparse
import logging
import subprocess
import unittest

logger = logging.getLogger(__name__)


def test():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test*.py")

    test_runner = unittest.TextTestRunner(verbosity=2)

    test_result = test_runner.run(test_suite)
    print(test_result)


def format():
    subprocess.run("ruff format".split(), check=True)
    subprocess.run("ruff check --fix".split(), check=True)


def wheel():
    subprocess.run("hatch build".split(), check=True)


def setup():
    logging.basicConfig(level=logging.INFO)


def main():
    setup()

    parser = argparse.ArgumentParser(description="Build script")
    subparsers = parser.add_subparsers(help="Subcommands")

    parser_test = subparsers.add_parser("test", help="Run tests")
    parser_test.set_defaults(func=test)

    parser_format = subparsers.add_parser("format", help="Run code formatters")
    parser_format.set_defaults(func=format)

    parser_wheel = subparsers.add_parser("wheel", help="Build the wheel")
    parser_wheel.set_defaults(func=wheel)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func()
    else:
        format()
        test()
        wheel()


if __name__ == "__main__":
    main()
