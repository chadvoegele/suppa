__copyright__ = "Copyright (C) 2026 Chad Voegele"
__license__ = "GNU GPLv2"

import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import TestCase

from suplistml.data.recipe_nlg import (
    DIRECTION_CLASS,
    TITLE_CLASS,
    Recipe,
    get_recipe_nlg_data,
    get_recipe_nlg_dataframe,
    get_recipe_nlg_training_df,
)


class TestRecipeNlg(TestCase):
    def setUp(self):
        self.exit_stack = ExitStack()
        temp_file = tempfile.NamedTemporaryFile(mode="w", prefix="suplistml_tests")
        self.temp_file = self.exit_stack.enter_context(temp_file)

    def tearDown(self):
        self.exit_stack.close()

    def test_get_recipe_nlg_dataframe(self):
        data = """
title,ingredients,directions,link,source,NER
"Test Recipe","[""ingredient1"", ""ingredient2""]","[""step1"", ""step2""]",http://example.com,Test Source,"[""ner1"", ""ner2""]"
"Another Recipe","[""flour"", ""sugar""]","[""mix"", ""bake""]",http://test.com,Another Source,"[""flour"", ""sugar""]"
""".strip()  # noqa
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_recipe_nlg_dataframe(Path(self.temp_file.name))
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["title"], "Test Recipe")
        self.assertEqual(df.iloc[0]["ingredients"], ["ingredient1", "ingredient2"])
        self.assertEqual(df.iloc[0]["directions"], ["step1", "step2"])
        self.assertEqual(df.iloc[0]["link"], "http://example.com")
        self.assertEqual(df.iloc[0]["source"], "Test Source")
        self.assertEqual(df.iloc[0]["NER"], ["ner1", "ner2"])
        self.assertEqual(df.iloc[1]["title"], "Another Recipe")
        self.assertEqual(df.iloc[1]["ingredients"], ["flour", "sugar"])
        self.assertEqual(df.iloc[1]["directions"], ["mix", "bake"])
        self.assertEqual(df.iloc[1]["link"], "http://test.com")
        self.assertEqual(df.iloc[1]["source"], "Another Source")
        self.assertEqual(df.iloc[1]["NER"], ["flour", "sugar"])

    def test_get_recipe_nlg_data(self):
        data = """
title,ingredients,directions,link,source,NER
"Test Recipe","[""ingredient1"", ""ingredient2""]","[""step1"", ""step2""]",http://example.com,Test Source,"[""ner1"", ""ner2""]"
""".strip()  # noqa
        self.temp_file.write(data)
        self.temp_file.flush()
        recipes = list(get_recipe_nlg_data(Path(self.temp_file.name)))
        self.assertEqual(len(recipes), 1)
        recipe = recipes[0]
        self.assertEqual(recipe.title, "Test Recipe")
        self.assertEqual(recipe.ingredients, ["ingredient1", "ingredient2"])
        self.assertEqual(recipe.directions, ["step1", "step2"])
        self.assertEqual(recipe.link, "http://example.com")
        self.assertEqual(recipe.source, "Test Source")
        self.assertEqual(recipe.NER, ["ner1", "ner2"])

    def test_get_recipe_nlg_dataframe_with_nrows(self):
        data = """
title,ingredients,directions,link,source,NER
"Recipe 1","[""a""]","[""b""]",http://a.com,Source1,"[""c""]"
"Recipe 2","[""d""]","[""e""]",http://b.com,Source2,"[""f""]"
"Recipe 3","[""g""]","[""h""]",http://c.com,Source3,"[""i""]"
""".strip()
        self.temp_file.write(data)
        self.temp_file.flush()
        df = get_recipe_nlg_dataframe(Path(self.temp_file.name), nrows=2)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["title"], "Recipe 1")
        self.assertEqual(df.iloc[1]["title"], "Recipe 2")

    def test_get_recipe_nlg_training_df(self):
        recipes = [
            Recipe(
                title="Pasta",
                ingredients=["noodles", "sauce"],
                directions=["boil noodles", "add sauce"],
                link="http://example.com",
                source="Test",
                NER=["noodles", "sauce"],
            ),
            Recipe(
                title="Salad",
                ingredients=["lettuce"],
                directions=["chop lettuce"],
                link="http://example.com",
                source="Test",
                NER=["lettuce"],
            ),
        ]
        df = get_recipe_nlg_training_df(recipes=recipes)
        title_rows = df[df["aisle"] == TITLE_CLASS]
        direction_rows = df[df["aisle"] == DIRECTION_CLASS]
        self.assertEqual(len(title_rows), 6)  # 3 per recipe
        self.assertIn("Pasta", title_rows["input"].values)
        self.assertIn("# Pasta", title_rows["input"].values)
        self.assertIn("## Pasta", title_rows["input"].values)
        self.assertEqual(len(direction_rows), 3)  # 2 + 1
        self.assertIn("boil noodles", direction_rows["input"].values)
        self.assertIn("add sauce", direction_rows["input"].values)
        self.assertIn("chop lettuce", direction_rows["input"].values)
        self.assertTrue((df["name"] == "").all())
        self.assertTrue((df["qty"] == "").all())
        self.assertTrue((df["unit"] == "").all())

    def test_get_recipe_nlg_training_df_nrows(self):
        recipes = [
            Recipe(
                title="Pasta",
                ingredients=["noodles"],
                directions=["boil noodles", "drain", "serve"],
                link="http://example.com",
                source="Test",
                NER=["noodles"],
            ),
        ]
        df = get_recipe_nlg_training_df(recipes=recipes, nrows=4)
        self.assertEqual(len(df), 4)
