__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel

logger = logging.getLogger(__name__)


@dataclass
class MultiBertConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=32,
        n_class_labels=4,
        n_tag_labels=3,
        dropout=0.01,
        cls_token_id=101,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.n_class_labels = n_class_labels
        self.n_tag_labels = n_tag_labels
        self.dropout = dropout
        self.cls_token_id = cls_token_id


@dataclass
class MultiBertOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    class_logits: torch.FloatTensor = None
    tag_logits: torch.FloatTensor = None


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiBert(PreTrainedModel):
    def __init__(self, config, alpha=0.5):
        super().__init__(config)
        self.config = config
        self.alpha = alpha
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            attn_implementation="eager",
        )
        self.bert = BertModel(bert_config, add_pooling_layer=False)

        self.classifier_head = MLP(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            output_size=self.config.n_class_labels,
            dropout_prob=config.dropout,
        )
        self.tag_head = MLP(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            output_size=self.config.n_tag_labels,
            dropout_prob=config.dropout,
        )

        self.classifier_head.apply(self._init_weights)
        self.tag_head.apply(self._init_weights)
        self.bert.apply(self.bert._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, class_labels=None, tag_labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        hiddens = bert_outputs.last_hidden_state
        class_logits = self.classifier_head(hiddens[:, 0])
        tag_logits = self.tag_head(hiddens[:, 1:])

        class_loss = torch.tensor(0.0)
        if class_labels is not None:
            class_loss = F.cross_entropy(class_logits.view(-1, self.config.n_class_labels), class_labels.view(-1))

        tag_loss = torch.tensor(0.0)
        if tag_labels is not None:
            tag_loss = F.cross_entropy(tag_logits.view(-1, self.config.n_tag_labels), tag_labels.view(-1))

        if (
            (class_loss.isnan() and tag_loss.isnan())
            or (class_loss.isnan() and tag_labels is None)
            or (class_labels is None and tag_loss.isnan())
        ):
            logger.warning("Found nan loss!")
            loss = torch.tensor(torch.nan)

        elif class_labels is None and tag_labels is None:
            logger.warning("Found None loss!")
            loss = None

        else:
            loss = torch.tensor(0.0).to(class_loss.device)
            if not class_loss.isnan():
                loss += self.alpha * class_loss
            if not tag_loss.isnan():
                loss += (1 - self.alpha) * tag_loss

        outputs = MultiBertOutputs(class_logits=class_logits, tag_logits=tag_logits, loss=loss)
        return outputs
