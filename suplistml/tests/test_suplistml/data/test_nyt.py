__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import TestCase

from suplistml.data.nyt import get_nyt_data


class TestNyt(TestCase):
    def setUp(self):
        self.exit_stack = ExitStack()
        temp_file = tempfile.NamedTemporaryFile(mode="w", prefix="suplistml_tests")
        self.temp_file = self.exit_stack.enter_context(temp_file)

    def tearDown(self):
        self.exit_stack.close()

    def test_text_cleaning(self):
        data = """
index,input,name,qty,range_end,unit,comment
0,"1/4 cup confectioners’ sugar","confectioners’ sugar",0.25,0.0,cup,
1,1/4 cup whole milk ,whole milk ,0.25,0.0,cup,
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_nyt_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["qty"], "1/4")
        self.assertEqual(df.iloc[0]["unit"], "cup")
        self.assertEqual(df.iloc[0]["name"], "confectioners' sugar")
        self.assertEqual(df.iloc[0]["input"], "1/4 cup confectioners' sugar")
        self.assertEqual(df.iloc[1]["qty"], "1/4")
        self.assertEqual(df.iloc[1]["unit"], "cup")
        self.assertEqual(df.iloc[1]["name"], "whole milk")
        self.assertEqual(df.iloc[1]["input"], "1/4 cup whole milk")

    def test_fractions(self):
        data = """
index,input,name,qty,range_end,unit,comment
0,"1 1/4 cups cooked butternut squash",butternut squash,1.25,0.0,cup,"cooked"
0,"1 cup broccoli",broccoli,1,0.0,cup,
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_nyt_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["qty"], "1 1/4")
        self.assertEqual(df.iloc[0]["unit"], "cups")
        self.assertEqual(df.iloc[0]["name"], "butternut squash")
        self.assertEqual(df.iloc[1]["qty"], "1")
        self.assertEqual(df.iloc[1]["unit"], "cup")
        self.assertEqual(df.iloc[1]["name"], "broccoli")

    def test_unicode(self):
        data = """
index,input,name,qty,range_end,unit,comment
0,"1 \xbc cups cooked macaroni",macaroni,1.25,0.0,cups,"cooked"
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_nyt_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["qty"], "1 1/4")
        self.assertEqual(df.iloc[0]["unit"], "cups")
        self.assertEqual(df.iloc[0]["name"], "macaroni")

    def test_case(self):
        data = """
index,input,name,qty,range_end,unit,comment
0,"1 cup cooked Macaroni",macaroni,1,0.0,cup,cooked
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_nyt_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["qty"], "1")
        self.assertEqual(df.iloc[0]["unit"], "cup")
        self.assertEqual(df.iloc[0]["name"], "Macaroni")

    def test_plural(self):
        data = """
index,input,name,qty,range_end,unit,comment
0,"2 to 3 tablespoons jalapeño",jalapeños,2,0.0,tablespoons,
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_nyt_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["qty"], "2")
        self.assertEqual(df.iloc[0]["unit"], "tablespoons")
        self.assertEqual(df.iloc[0]["name"], "jalapeño")

    def test_inversion(self):
        data = """
index,input,name,qty,range_end,unit,comment
0,"1/2 cup finely chopped fennel","fennel, finely chopped",0.5,0.0,cup,
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_nyt_data(Path(self.temp_file.name))
        self.assertEqual(df.iloc[0]["qty"], "1/2")
        self.assertEqual(df.iloc[0]["unit"], "cup")
        self.assertEqual(df.iloc[0]["name"], "finely chopped fennel")
