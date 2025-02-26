__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import logging
from unittest import TestCase

from suplistml.model.tokenizer import get_tag_tokenizer

logger = logging.getLogger(__name__)


class TestTagTokenizer(TestCase):
    def test_pad_unk(self):
        tags = ["NAME", "QTY", "OTHER"]
        tokenizer = get_tag_tokenizer(tags)
        pad_ids = tokenizer("<pad>", return_tensors="pt")
        unk_ids = tokenizer("<unk>", return_tensors="pt")
        self.assertEqual(pad_ids["input_ids"].tolist(), [[0]])
        self.assertEqual(unk_ids["input_ids"].tolist(), [[1]])

    def test_tags(self):
        tags = ["NAME", "QTY", "OTHER"]
        tokenizer = get_tag_tokenizer(tags)
        tag_ids = tokenizer(["NAME", "QTY", "OTHER"], return_tensors="pt")
        self.assertEqual(tag_ids["input_ids"].tolist(), [[2], [3], [4]])

    def test_oov(self):
        tags = ["NAME", "QTY", "OTHER"]
        tokenizer = get_tag_tokenizer(tags)
        oov_ids = tokenizer("NOTINVOCAB", return_tensors="pt")
        self.assertEqual(oov_ids["input_ids"].tolist(), [[1]])
