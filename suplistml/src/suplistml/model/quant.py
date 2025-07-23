__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import json
import logging
from pathlib import Path

import safetensors.torch
import torch
import torch.nn as nn
from torch.utils.data import Subset

from suplistml.data.lfs import lfs_open
from suplistml.data.synthetic import get_aisles, get_synthetic_df
from suplistml.dataset.multi_dataset import MultiDataset
from suplistml.model.tokenizer import (
    get_class_tokenizer,
    get_tag_tokenizer,
    get_tokenizer,
)
from suplistml.model.train import (
    get_joint_model,
    seed_everything,
)
from suplistml.script import (
    is_ipython,
    script_main,
    setup_logging,
)

logger = logging.getLogger(__name__)


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__(in_features, out_features, bias)
        self.bits = bits
        self.qmax = 2 ** (self.bits - 1) - 1
        self.qmin = -self.qmax

    def qdq(self, weights):
        max_val = weights.abs().max(dim=0, keepdim=True)[0]
        scale = max_val / self.qmax
        return (weights / scale).round().clamp_(self.qmin, self.qmax) * scale

    def forward(self, x):
        w_q = self.weight + (self.qdq(self.weight) - self.weight).detach()
        return nn.functional.linear(x, w_q, self.bias)


def replace_linear_with_quant(module, bits=8):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            quant_layer = QuantLinear(child.in_features, child.out_features, child.bias is not None, bits)
            quant_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quant_layer.bias.data = child.bias.data.clone()
            setattr(module, name, quant_layer)
        else:
            replace_linear_with_quant(child, bits)
    return module


def _get_loss(model, dataset):
    device = next(model.parameters()).device
    losses = []
    for i in range(len(dataset)):
        sample = dataset[i]
        outputs = model(
            input_ids=sample["input_ids"].unsqueeze(0).to(device),
            attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
            class_labels=sample["class_labels"].unsqueeze(0).to(device),
            tag_labels=sample["tag_labels"].unsqueeze(0).to(device),
        )
        losses.append(outputs.loss.item())
    losses = torch.tensor(losses)
    avg_loss = losses.mean().item()
    return avg_loss


def run_quantize(
    output_path: Path = "__AUTO__",
    nrows: int = 1024,
    bits: int = 8,
    n_test_samples: int = 1024,
):
    setup_logging(output_path)
    seed_everything()
    logger.info(json.dumps({k: str(v) for k, v in locals().items()}))

    df = get_synthetic_df(nrows=nrows)
    classes = get_aisles()

    logger.info("Getting tokenizer")
    tokenizer = get_tokenizer()
    tag_tokenizer = get_tag_tokenizer()
    class_tokenizer = get_class_tokenizer(classes=classes)

    logger.info("Getting data")
    dataset = MultiDataset(tokenizer, tag_tokenizer, class_tokenizer, nrows=nrows, df=df, seqlen=256)
    model = get_joint_model(tokenizer, tag_tokenizer, class_tokenizer)
    model = model.to("cuda")

    model_root = Path("src/suplistml/models/run+1748084792")
    trained_model_path = model_root / "model.safetensors"
    logger.info(f"Loading model from {trained_model_path}")
    with lfs_open(trained_model_path, "rb") as g:
        safetensors_bytes = g.read()
        state_dict = safetensors.torch.load(safetensors_bytes)
    model.load_state_dict(state_dict, strict=True)

    split_rng = torch.Generator().manual_seed(8385)
    random_indices = torch.randperm(len(dataset), generator=split_rng)[:n_test_samples].tolist()
    test_dataset = Subset(dataset, random_indices)

    loss = _get_loss(model, test_dataset)
    logger.info(f"Average loss on test set: {loss}")

    logger.info(f"Replacing linear layers with quantized versions (bits={bits})")
    model = replace_linear_with_quant(model, bits=bits)
    loss = _get_loss(model, test_dataset)
    logger.info(f"Average loss with quantized model on test set: {loss}")


if __name__ == "__main__" and not is_ipython():
    script_main(globals())
