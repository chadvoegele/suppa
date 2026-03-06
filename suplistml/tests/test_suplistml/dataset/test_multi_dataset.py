__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import logging
from io import StringIO
from unittest import TestCase

import pandas as pd
import torch

from suplistml.dataset.multi_dataset import MultiDataset, _find_sequence_mask, _get_joined_nyt_aisles_dfs
from suplistml.model.tokenizer import (
    get_class_tokenizer,
    get_tag_tokenizer,
    get_tokenizer,
)

logger = logging.getLogger(__name__)


class TestFindSequenceMask(TestCase):
    def test_finds_first_match(self):
        seq = torch.tensor([5, 6, 1, 2, 1])
        pattern = torch.tensor([1, 2])
        mask = _find_sequence_mask(seq, pattern)
        expected = torch.tensor([False, False, True, True, False])
        torch.testing.assert_close(mask, expected)

    def test_no_match(self):
        seq = torch.tensor([5, 6, 7])
        pattern = torch.tensor([1, 2])
        mask = _find_sequence_mask(seq, pattern)
        torch.testing.assert_close(mask, torch.zeros(3, dtype=torch.bool))

    def test_single_token_first_match_only(self):
        seq = torch.tensor([1, 2, 1, 3])
        pattern = torch.tensor([1])
        mask = _find_sequence_mask(seq, pattern)
        expected = torch.tensor([True, False, False, False])
        torch.testing.assert_close(mask, expected)

    def test_exclude_mask_skips_to_next_match(self):
        seq = torch.tensor([1, 2, 1, 2])
        pattern = torch.tensor([1, 2])
        exclude_mask = torch.tensor([True, True, False, False])
        mask = _find_sequence_mask(seq, pattern, exclude_mask=exclude_mask)
        expected = torch.tensor([False, False, True, True])
        torch.testing.assert_close(mask, expected)

    def test_exclude_mask_partial_overlap_skips(self):
        # pattern [1, 2] at position 1 overlaps with excluded position 2
        seq = torch.tensor([3, 1, 2, 1, 2])
        pattern = torch.tensor([1, 2])
        exclude_mask = torch.tensor([False, False, True, False, False])
        mask = _find_sequence_mask(seq, pattern, exclude_mask=exclude_mask)
        expected = torch.tensor([False, False, False, True, True])
        torch.testing.assert_close(mask, expected)

    def test_no_match_when_all_excluded(self):
        seq = torch.tensor([1, 2, 1, 2])
        pattern = torch.tensor([1, 2])
        exclude_mask = torch.ones(4, dtype=torch.bool)
        mask = _find_sequence_mask(seq, pattern, exclude_mask=exclude_mask)
        torch.testing.assert_close(mask, torch.zeros(4, dtype=torch.bool))


class TestDFJoin(TestCase):
    def test_join_nyt_aisle_dfs(self):
        nyt_csv = """
input,name,qty,unit
1/2 lb ground beef,ground beef,1/2,lb
2 carrots,carrots,2,
3 pears,pears,3,
""".strip()
        aisles_json_lines = """
{"food": "ground beef", "aisle": "meat"}
{"food": "carrots", "aisle": "produce"}
{"food": "pecans", "aisle": "nuts"}
""".strip()
        joined_json_lines = """
{"input":"2 carrots","name":"carrots","qty":"2","unit":null,"aisle":"produce"}
{"input":"1/2 lb ground beef","name":"ground beef","qty":"1/2","unit":"lb","aisle":"meat"}
{"input":"3 pears","name":"pears","qty":"3","unit":null,"aisle":null}
{"input":"pecans","name":null,"qty":null,"unit":null,"aisle":"nuts"}
""".strip()
        nyt_df = pd.read_csv(StringIO(nyt_csv))
        aisles_df = pd.read_json(StringIO(aisles_json_lines), lines=True)
        expected_df = pd.read_json(
            StringIO(joined_json_lines),
            lines=True,
        )
        df = _get_joined_nyt_aisles_dfs(nyt_df, aisles_df)
        pd.testing.assert_frame_equal(df, expected_df)


class TestMultiDataset(TestCase):
    def setUp(self):
        self.tokenizer = get_tokenizer()
        self.tag_tokenizer = get_tag_tokenizer()
        self.class_tokenizer = get_class_tokenizer()
        self.seqlen = 16

    def _assert_row(self, row, item, tags):
        n_pad_tokens = (item["attention_mask"] == 0).sum()

        class_id = self.tokenizer("[CLS]", add_special_tokens=False, return_tensors="pt")["input_ids"].item()
        pad_id = self.tokenizer("[PAD]", add_special_tokens=False, return_tensors="pt")["input_ids"].item()

        expected_ids = [
            class_id,
            *self.tokenizer(row["input"], add_special_tokens=False)["input_ids"],
            *([pad_id] * n_pad_tokens),
        ]
        self.assertEqual(item["input_ids"].tolist(), expected_ids)

        if pd.isna(row["aisle"]):
            self.assertEqual(item["class_labels"], torch.tensor([-100]))
        else:
            self.assertEqual(self.class_tokenizer.decode(item["class_labels"].item()), row["aisle"])

        if tags is None:
            torch.testing.assert_close(item["tag_labels"], torch.tensor([-100] * (self.seqlen - 1)))
        else:
            expected_tag_ids = [
                *self.tag_tokenizer(tags.split(), return_tensors="pt")["input_ids"].view(-1).tolist(),
                *([-100] * n_pad_tokens),
            ]
            self.assertEqual(item["tag_labels"].tolist(), expected_tag_ids)

    def test_multi_dataset(self):
        json_lines = """
{"input":"2 carrots","name":"carrots","qty":"2","unit":null,"aisle":"produce"}
{"input":"1/2 lb ground beef","name":"ground beef","qty":"1/2","unit":"lb","aisle":"meat"}
{"input":"3 pears","name":"pears","qty":"3","unit":null,"aisle":null}
{"input":"pecans","name":null,"qty":null,"unit":null,"aisle":"nuts"}
""".strip()
        df = pd.read_json(StringIO(json_lines), lines=True)
        dataset = MultiDataset(
            tokenizer=self.tokenizer,
            tag_tokenizer=self.tag_tokenizer,
            class_tokenizer=self.class_tokenizer,
            df=df,
            seqlen=self.seqlen,
        )
        self.assertEqual(len(dataset), 4)

        self._assert_row(row=df.iloc[0], item=dataset[0], tags="QTY NAME NAME")
        self._assert_row(row=df.iloc[1], item=dataset[1], tags="QTY QTY QTY UNIT NAME NAME")
        self._assert_row(row=df.iloc[2], item=dataset[2], tags="QTY NAME NAME")
        self._assert_row(row=df.iloc[3], item=dataset[3], tags=None)

    def test_multi_dataset_other_column_prevents_qty_override(self):
        # "other" marks the prefix tokens first so qty matches the correct "1"
        json_lines = """
{"input":"1. 1 banana","name":"banana","qty":"1","unit":null,"aisle":"produce","other":"1."}
{"input":"2. 1/2 cup frozen mango","name":"frozen mango","qty":"1/2","unit":"cup","aisle":"frozen foods","other":"a."}
""".strip()
        df = pd.read_json(StringIO(json_lines), lines=True)
        dataset = MultiDataset(
            tokenizer=self.tokenizer,
            tag_tokenizer=self.tag_tokenizer,
            class_tokenizer=self.class_tokenizer,
            df=df,
            seqlen=self.seqlen,
        )
        self._assert_row(row=df.iloc[0], item=dataset[0], tags="OTHER OTHER QTY NAME")
        self._assert_row(row=df.iloc[1], item=dataset[1], tags="OTHER OTHER QTY QTY QTY UNIT NAME NAME")
