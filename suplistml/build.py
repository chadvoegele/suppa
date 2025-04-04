#!/usr/bin/env python
__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import argparse
import logging
import re
import subprocess
import unittest
from unittest import TestCase
from unittest.suite import TestSuite

logger = logging.getLogger(__name__)


def filter_test_suite(test_suite: TestSuite, pattern: str):
    filtered = TestSuite()
    for test in test_suite._tests:
        if isinstance(test, TestSuite):
            filtered.addTests(filter_test_suite(test, pattern))
        elif isinstance(test, TestCase):
            has_method_name_match = re.search(pattern, test._testMethodName, flags=re.IGNORECASE) is not None
            has_class_name_match = re.search(pattern, test.__class__.__name__, flags=re.IGNORECASE) is not None
            if has_method_name_match or has_class_name_match:
                filtered.addTest(test)
        else:
            raise ValueError(f"Unknown test of {type(test)=}")
    return filtered


def test(pattern: str = None):
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test*.py")

    if pattern is not None:
        test_suite = filter_test_suite(test_suite, pattern)

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
    parser_test.add_argument("--pattern", required=False, default=None)

    parser_format = subparsers.add_parser("format", help="Run code formatters")
    parser_format.set_defaults(func=format)

    parser_wheel = subparsers.add_parser("wheel", help="Build the wheel")
    parser_wheel.set_defaults(func=wheel)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(**{k: v for k, v in vars(args).items() if k != "func"})
    else:
        format()
        test()
        wheel()


if __name__ == "__main__":
    main()
