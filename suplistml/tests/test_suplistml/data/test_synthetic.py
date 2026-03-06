__copyright__ = "Copyright (C) 2026 Chad Voegele"
__license__ = "GNU GPLv2"

from unittest import TestCase

import pandas as pd

from suplistml.data.synthetic import augment_with_list_prefixes


class TestAugmentWithListPrefixes(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "input": ["2 stalks celery, chopped coarse", "1 cup flour"],
                "name": ["celery", "flour"],
                "qty": ["2", "1"],
                "unit": ["stalks", "cup"],
                "aisle": ["fresh vegetables", "pantry"],
            }
        )

    def test_output_length(self):
        result = augment_with_list_prefixes(self.df, n=3)
        self.assertEqual(len(result), len(self.df) * 4)  # original + 3 augmented

    def test_original_rows_preserved(self):
        result = augment_with_list_prefixes(self.df, n=3)
        original_inputs = set(self.df["input"])
        result_inputs = set(result["input"])
        self.assertTrue(original_inputs.issubset(result_inputs))

    def test_prefixes_applied(self):
        result = augment_with_list_prefixes(self.df, n=3, seed=42)
        inputs = result["input"].tolist()
        self.assertIn("8. 2 stalks celery, chopped coarse", inputs)
        self.assertIn("f. 2 stalks celery, chopped coarse", inputs)
        self.assertIn("5. 1 cup flour", inputs)

    def test_non_input_columns_unchanged(self):
        result = augment_with_list_prefixes(self.df, n=3)
        augmented = result[~result["input"].isin(self.df["input"])]
        self.assertTrue((augmented["aisle"].isin(self.df["aisle"])).all())
        self.assertTrue((augmented["name"].isin(self.df["name"])).all())

    def test_other_column(self):
        result = augment_with_list_prefixes(self.df, n=3, seed=42)
        original = result[result["input"].isin(self.df["input"])]
        augmented = result[~result["input"].isin(self.df["input"])]
        self.assertTrue((original["other"] == "").all())
        self.assertTrue((augmented["other"] != "").all())
        self.assertIn("8.", result["other"].values)
        self.assertIn("f.", result["other"].values)

    def test_reproducible_with_seed(self):
        result1 = augment_with_list_prefixes(self.df, n=3, seed=0)
        result2 = augment_with_list_prefixes(self.df, n=3, seed=0)
        pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))

    def test_different_seeds_differ(self):
        result1 = augment_with_list_prefixes(self.df, n=3, seed=0)
        result2 = augment_with_list_prefixes(self.df, n=3, seed=99)
        self.assertFalse(result1["input"].equals(result2["input"]))
