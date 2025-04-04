__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from suplistml.script import script_main


def test_case1(a: int = 5, b: str = "hello"):
    return f"{b}{a}"


def test_auto_output(output: Path = "__AUTO__"):
    return str(output)


class TestScriptMain(TestCase):
    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
            script_main(fake_globals, args=["--name=not_a_test_case"])

    def test_case1_defaults(self):
        fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
        result = script_main(fake_globals, args=["--name=test_case1"])
        self.assertEqual(result, "hello5")

    def test_case1_int(self):
        fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
        result = script_main(fake_globals, args=["--name=test_case1", "--a=0"])
        self.assertEqual(result, "hello0")

    def test_case1_str(self):
        fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
        result = script_main(fake_globals, args=["--name=test_case1", "--b=world"])
        self.assertEqual(result, "world5")

    def test_case1_all(self):
        fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
        result = script_main(fake_globals, args=["--name=test_case1", "--a=8", "--b=world"])
        self.assertEqual(result, "world8")

    def test_case1_invalid(self):
        with self.assertRaises(argparse.ArgumentError) as cm:
            fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
            script_main(fake_globals, args=["--name=test_case1", "--a=hello", "--b=world"])
        self.assertEqual(cm.exception.message, "invalid int value: 'hello'")

    def test_auto_output(self):
        with tempfile.TemporaryDirectory(prefix="test_suplistml") as tmpdir,\
            patch("tempfile.gettempdir") as gettempdir:
            gettempdir.return_value = tmpdir
            fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))
            path = script_main(fake_globals, args=["--name=test_auto_output"])
            self.assertIn("test_auto_output/output", path)
