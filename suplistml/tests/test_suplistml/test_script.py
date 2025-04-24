__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest import TestCase
from unittest.mock import patch

from suplistml.script import script_main


def test_case1(a: int = 5, b: str = "hello"):
    return f"{b}{a}"


def test_auto_output(output: Path = "__AUTO__"):
    return str(output)


def test_boolean(b: bool = False):
    return b


def test_complex_obj(o: List[Dict[str, str]]):
    return o


def test_required(a: str):
    return a


class TestScriptMain(TestCase):
    fake_globals = dict(__spec__=SimpleNamespace(name="test_suplistml.test_script"))

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            script_main(self.fake_globals, args=["--name=not_a_test_case"])

    def test_case1_defaults(self):
        result = script_main(self.fake_globals, args=["--name=test_case1"])
        self.assertEqual(result, "hello5")

    def test_case1_int(self):
        result = script_main(self.fake_globals, args=["--name=test_case1", "--a=0"])
        self.assertEqual(result, "hello0")

    def test_case1_str(self):
        result = script_main(self.fake_globals, args=["--name=test_case1", "--b=world"])
        self.assertEqual(result, "world5")

    def test_case1_all(self):
        result = script_main(self.fake_globals, args=["--name=test_case1", "--a=8", "--b=world"])
        self.assertEqual(result, "world8")

    def test_case1_invalid(self):
        with self.assertRaises(argparse.ArgumentError) as cm:
            script_main(self.fake_globals, args=["--name=test_case1", "--a=hello", "--b=world"])
        self.assertEqual(cm.exception.message, "invalid int value: 'hello'")

    def test_auto_output(self):
        with tempfile.TemporaryDirectory(prefix="test_suplistml") as tmpdir, patch("tempfile.gettempdir") as gettempdir:
            gettempdir.return_value = tmpdir
            path = script_main(self.fake_globals, args=["--name=test_auto_output"])
            self.assertIn("test_auto_output/output", path)

    def test_boolean_default(self):
        result = script_main(self.fake_globals, args=["--name=test_boolean"])
        self.assertFalse(result)

    def test_boolean_true(self):
        result = script_main(self.fake_globals, args=["--name=test_boolean", "--b=true"])
        self.assertTrue(result)

    def test_boolean_false(self):
        result = script_main(self.fake_globals, args=["--name=test_boolean", "--b=false"])
        self.assertFalse(result)

    def test_complex_obj(self):
        result = script_main(self.fake_globals, args=["--name=test_complex_obj", '--o=[{"a": 5}, {"b": "c"}]'])
        self.assertEqual(result, [dict(a=5), dict(b="c")])

    def test_required(self):
        with self.assertRaises(argparse.ArgumentError) as cm:
            script_main(self.fake_globals, args=["--name=test_required"])
        self.assertEqual(cm.exception.message, "the following arguments are required: --a")
