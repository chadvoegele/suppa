__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import logging
from unittest import TestCase

import numpy as np
import torch

from suplistml.model.multi_bert import MultiBert, MultiBertConfig

logger = logging.getLogger(__name__)


class TestBertForSequenceTokenClassification(TestCase):
    def _get_model(self):
        config = MultiBertConfig(
            vocab_size=16,
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=32,
            n_class_labels=4,
            n_tag_labels=3,
            dropout=0.0,
            cls_token_id=0,
        )
        model = MultiBert(config)
        return model

    def test_overfit(self):
        torch.manual_seed(41)

        model = self._get_model()

        dataset = [
            {"input_ids": [1, 5, 6, 2], "tag_labels": [1, 1, 2], "class_label": 0},
            {"input_ids": [1, 2, 5, 6], "tag_labels": [2, 1, 1], "class_label": 1},
            {"input_ids": [1, 2, 7, 8], "tag_labels": [2, 1, 1], "class_label": 2},
        ]

        input_ids = torch.tensor([r["input_ids"] for r in dataset])
        attention_mask = torch.ones_like(input_ids)
        class_labels = torch.tensor([r["class_label"] for r in dataset])
        tag_labels = torch.tensor([r["tag_labels"] for r in dataset])

        max_epochs = 120
        n_to_log = 10
        target_loss = 0.05

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0, betas=(0.8, 0.9))

        losses = []
        model.train()
        for epoch in range(max_epochs):
            model_outputs = model.forward(
                input_ids,
                attention_mask=attention_mask,
                class_labels=class_labels,
                tag_labels=tag_labels,
            )
            loss = model_outputs.loss
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % n_to_log == 0:
                logger.info(f"{epoch=} loss={loss.item()} norm={norm.item()}")
                losses.append(loss)

            if loss < target_loss:
                break

        avg_down = np.average([ploss > loss for ploss, loss in zip(losses[:-1], losses[1:])])
        self.assertGreater(avg_down, 0.85)
        self.assertLess(loss.item(), target_loss)

        for row in dataset:
            input_ids = torch.tensor([row["input_ids"]])
            attention_mask = torch.ones_like(input_ids)
            model_outputs = model.forward(input_ids, attention_mask=attention_mask)
            predicted_class_label = model_outputs.class_logits.argmax(dim=-1)[0].item()
            self.assertEqual(row["class_label"], predicted_class_label)
            predicted_tag_labels = model_outputs.tag_logits.argmax(dim=-1)[0]
            torch.testing.assert_close(torch.tensor(row["tag_labels"]), predicted_tag_labels)

    def test_none_labels(self):
        model = self._get_model()

        input_ids = torch.tensor([[1, 4, 6], [1, 6, 7]])
        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=None,
            tag_labels=None,
        )
        self.assertEqual(model_outputs.loss, None)

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=torch.tensor([-100, -100]),
            tag_labels=None,
        )
        self.assertTrue(model_outputs.loss.isnan())

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=None,
            tag_labels=torch.tensor([[-100, -100], [-100, -100]]),
        )
        self.assertTrue(model_outputs.loss.isnan())

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=torch.tensor([-100, -100]),
            tag_labels=torch.tensor([[-100, -100], [-100, -100]]),
        )
        self.assertTrue(model_outputs.loss.isnan())

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=torch.tensor([2, -100]),
            tag_labels=torch.tensor([[-100, -100], [-100, -100]]),
        )
        self.assertTrue(model_outputs.loss > 0)

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=torch.tensor([-100, -100]),
            tag_labels=torch.tensor([[-100, 2], [-100, -100]]),
        )
        self.assertTrue(model_outputs.loss > 0)

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=torch.tensor([2, -100]),
            tag_labels=None,
        )
        self.assertTrue(model_outputs.loss > 0)

        model_outputs = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            class_labels=None,
            tag_labels=torch.tensor([[-100, 2], [-100, -100]]),
        )
        self.assertTrue(model_outputs.loss > 0)
