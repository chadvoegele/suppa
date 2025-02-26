__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from suplistml.data.llm import get_llm_aisles_data
from suplistml.data.nyt import get_nyt_data


def _get_joined_nyt_aisles_dfs(nyt_df=None, aisles_df=None, nrows=None):
    if nyt_df is None:
        nyt_df = get_nyt_data(nrows=nrows)

    if aisles_df is None:
        aisles_df = get_llm_aisles_data()

    _df = pd.merge(nyt_df, aisles_df, how="outer", left_on="name", right_on="food")

    no_name_index = np.argwhere(pd.isna(_df["name"])).flatten()
    _df.loc[no_name_index, "input"] = _df.iloc[no_name_index]["food"]

    _df = _df.drop(["food"], axis=1)

    if nrows is not None:
        _df = _df.iloc[:nrows]

    return _df


class MultiDataset(Dataset):
    def __init__(self, tokenizer, tag_tokenizer, class_tokenizer, df=None, seqlen=64, nrows=None):
        self._tokenizer = tokenizer
        self._tag_tokenizer = tag_tokenizer
        self._class_tokenizer = class_tokenizer
        self._seqlen = seqlen

        if df is None:
            df = _get_joined_nyt_aisles_dfs(nrows=nrows)
        self._df = df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, i):
        row = self._df.iloc[i]

        class_token = "[CLS]"

        in_ids = self._tokenizer(
            f"{class_token} {row['input']}",
            max_length=self._seqlen,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )

        if not pd.isna(row["aisle"]) and len(row["aisle"]) > 0:
            class_labels = self._class_tokenizer(row["aisle"], return_tensors="pt")["input_ids"][:, 0]
        else:
            class_labels = torch.tensor([-100])

        batch_size = in_ids["input_ids"].size(0)
        assert batch_size == 1
        n_tokens = in_ids["input_ids"].size(1) - 1
        tag_labels = torch.full((batch_size, n_tokens), -100)

        if not pd.isna(row["name"]) or not pd.isna(row["qty"]) or not pd.isna(row["unit"]):

            def _set_tag_labels_(column_name, tag_name):
                value = row[column_name]

                if not pd.isna(value) and len(str(value)) > 0:
                    tag_id = self._tag_tokenizer(tag_name, return_tensors="pt")["input_ids"]
                    assert tag_id.size(0) == 1
                    assert tag_id.size(1) == 1
                    assert tag_id[0, 0] > 1
                    ids = self._tokenizer(value, add_special_tokens=False, return_tensors="pt")["input_ids"]
                    tag_labels[torch.isin(in_ids["input_ids"][:, 1:], ids)] = tag_id[0, 0]

            _set_tag_labels_(column_name="name", tag_name="NAME")
            _set_tag_labels_(column_name="qty", tag_name="QTY")
            _set_tag_labels_(column_name="unit", tag_name="UNIT")

            other_tag_id = self._tag_tokenizer("OTHER", return_tensors="pt")["input_ids"][0, 0]
            tag_labels[torch.logical_and(in_ids["input_ids"][:, 1:] != 0, tag_labels == -100)] = other_tag_id

        item = {**in_ids, "class_labels": class_labels, "tag_labels": tag_labels}

        for k, v in item.items():
            item[k] = v.squeeze(0)

        return item
