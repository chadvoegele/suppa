__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import logging
from io import StringIO
from unittest import TestCase

import pandas as pd
import torch

from suplistml.dataset.multi_dataset import MultiDataset, _get_joined_nyt_aisles_dfs
from suplistml.model.tokenizer import (
    get_class_tokenizer,
    get_tag_tokenizer,
    get_tokenizer,
)

logger = logging.getLogger(__name__)


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
    def test_multi_dataset(self):
        json_lines = """
{"input":"2 carrots","name":"carrots","qty":"2","unit":null,"aisle":"produce"}
{"input":"1/2 lb ground beef","name":"ground beef","qty":"1/2","unit":"lb","aisle":"meat"}
{"input":"3 pears","name":"pears","qty":"3","unit":null,"aisle":null}
{"input":"pecans","name":null,"qty":null,"unit":null,"aisle":"nuts"}
""".strip()
        df = pd.read_json(StringIO(json_lines), lines=True)
        tokenizer = get_tokenizer()
        tag_tokenizer = get_tag_tokenizer()
        class_tokenizer = get_class_tokenizer()
        seqlen = 16
        dataset = MultiDataset(
            tokenizer=tokenizer,
            tag_tokenizer=tag_tokenizer,
            class_tokenizer=class_tokenizer,
            df=df,
            seqlen=seqlen,
        )
        self.assertEqual(len(dataset), 4)
        class_id = tokenizer("[CLS]", add_special_tokens=False, return_tensors="pt")["input_ids"].item()
        pad_id = tokenizer("[PAD]", add_special_tokens=False, return_tensors="pt")["input_ids"].item()

        def _assert_row(row, item, tags):
            n_pad_tokens = (item["attention_mask"] == 0).sum()

            expected_ids = [
                class_id,
                *tokenizer(row["input"], add_special_tokens=False)["input_ids"],
                *([pad_id] * n_pad_tokens),
            ]
            self.assertEqual(item["input_ids"].tolist(), expected_ids)

            if row["aisle"] is None:
                self.assertEqual(item["class_labels"], torch.tensor([-100]))
            else:
                self.assertEqual(class_tokenizer.decode(item["class_labels"].item()), row["aisle"])

            if tags is None:
                torch.testing.assert_close(item["tag_labels"], torch.tensor([-100] * (seqlen - 1)))
            else:
                expected_tag_ids = [
                    *tag_tokenizer(tags.split(), return_tensors="pt")["input_ids"].view(-1).tolist(),
                    *([-100] * n_pad_tokens),
                ]
                self.assertEqual(item["tag_labels"].tolist(), expected_tag_ids)

        _assert_row(row=df.iloc[0], item=dataset[0], tags="QTY NAME NAME")
        _assert_row(row=df.iloc[1], item=dataset[1], tags="QTY QTY QTY UNIT NAME NAME")
        _assert_row(row=df.iloc[2], item=dataset[2], tags="QTY NAME NAME")
        _assert_row(row=df.iloc[3], item=dataset[3], tags=None)
