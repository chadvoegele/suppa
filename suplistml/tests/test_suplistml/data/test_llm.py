__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import TestCase

from suplistml.data.llm import get_llm_aisles_data


class TestLlm(TestCase):
    def setUp(self):
        self.exit_stack = ExitStack()
        temp_file = tempfile.NamedTemporaryFile(mode="w", prefix="suplistml_tests")
        self.temp_file = self.exit_stack.enter_context(temp_file)

    def tearDown(self):
        self.exit_stack.close()

    def test_aisle_fix(self):
        data = """
{"food": "flour", "aisle": "baking"}
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_llm_aisles_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["food"], "flour")
        self.assertEqual(df.iloc[0]["aisle"], "bakery")

    def test_space_comma_fix(self):
        data = """
{"food": "lemon , sliced", "aisle": "produce"}
{"food": "chopped tomatoes ( canned are fine; drain them first )", "aisle": "produce"}
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_llm_aisles_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["food"], "lemon, sliced")
        self.assertEqual(df.iloc[0]["aisle"], "produce")
        self.assertEqual(df.iloc[1]["food"], "chopped tomatoes (canned are fine; drain them first)")
        self.assertEqual(df.iloc[1]["aisle"], "produce")
