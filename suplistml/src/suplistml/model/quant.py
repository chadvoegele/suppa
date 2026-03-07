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
from suplistml.data.recipe_nlg import (
    DIRECTION_CLASS,
    TITLE_CLASS,
)
from suplistml.data.synthetic import get_aisles, get_synthetic_df
from suplistml.dataset.multi_dataset import MultiDataset
from suplistml.model.tokenizer import (
    get_class_tokenizer,
    get_tag_tokenizer,
    get_tokenizer,
)
from suplistml.model.train import (
    calculate_accuracy,
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

    def _quantize(self, weight):
        max_val = weight.abs().max(dim=0, keepdim=True)[0]
        scale = max_val / self.qmax
        qweight = (weight / scale).round().clamp_(self.qmin, self.qmax)
        return qweight, scale

    def _quantize_dequantize(self, weight):
        qweight, scale = self._quantize(weight)
        return qweight * scale

    def forward(self, x):
        w_q = self.weight + (self._quantize_dequantize(self.weight) - self.weight).detach()
        return nn.functional.linear(x, w_q, self.bias)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        qweight, scale = self._quantize(self.weight)

        state_dict[prefix + "weight"] = qweight.to(torch.int8)
        state_dict[prefix + "scale"] = scale

        return state_dict


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


def _get_accuracies(model, dataset):
    device = next(model.parameters()).device
    class_accuracies = []
    tag_accuracies = []
    for i in range(len(dataset)):
        sample = dataset[i]
        outputs = model(
            input_ids=sample["input_ids"].unsqueeze(0).to(device),
            attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
        )
        class_labels = sample["class_labels"]
        class_accuracy = calculate_accuracy(
            outputs.class_logits.detach().cpu().numpy(), class_labels.detach().cpu().numpy()
        )
        class_accuracies.append(class_accuracy)

        tag_labels = sample["tag_labels"]
        tag_accuracy = calculate_accuracy(outputs.tag_logits.detach().cpu().numpy(), tag_labels.detach().cpu().numpy())
        tag_accuracies.append(tag_accuracy)

    class_accuracies = torch.tensor(class_accuracies)
    avg_class_accuracy = class_accuracies.mean().item()

    tag_accuracies = torch.tensor(tag_accuracies)
    avg_tag_accuracy = tag_accuracies.mean().item()

    return avg_class_accuracy, avg_tag_accuracy


def run_quantize(
    output_path: Path = "__AUTO__",
    model_root: Path = None,
    nrows: int = 102400,
    bits: int = 8,
    n_test_samples: int = 1024,
):
    setup_logging(output_path)
    seed_everything()

    if model_root is None:
        model_root = Path("src/suplistml/models/run+1772630896")

    logger.info(json.dumps({k: str(v) for k, v in locals().items()}))

    df = get_synthetic_df(nrows=nrows)
    classes = get_aisles() + [TITLE_CLASS, DIRECTION_CLASS]

    logger.info("Getting tokenizer")
    tokenizer = get_tokenizer()
    tag_tokenizer = get_tag_tokenizer()
    class_tokenizer = get_class_tokenizer(classes=classes)

    logger.info("Getting data")
    dataset = MultiDataset(tokenizer, tag_tokenizer, class_tokenizer, nrows=nrows, df=df, seqlen=256)
    model = get_joint_model(tokenizer, tag_tokenizer, class_tokenizer)
    model = model.to("cuda")

    trained_model_path = model_root / "model.safetensors"
    logger.info(f"Loading model from {trained_model_path}")
    with lfs_open(trained_model_path, "rb") as g:
        safetensors_bytes = g.read()
        state_dict = safetensors.torch.load(safetensors_bytes)
    model.load_state_dict(state_dict, strict=True)

    state_dict = model.state_dict()
    state_dict_fp16 = {k: v.half() for k, v in state_dict.items()}
    fp16_output_path = Path(output_path) / "model_fp16.safetensors"
    safetensors.torch.save_file(state_dict_fp16, fp16_output_path)
    logger.info(f"Saved model in fp16 to {str(output_path)}")

    split_rng = torch.Generator().manual_seed(8386)
    random_indices = torch.randperm(len(dataset), generator=split_rng)[:n_test_samples].tolist()
    test_dataset = Subset(dataset, random_indices)

    loss = _get_loss(model, test_dataset)
    logger.info(f"{loss=}")

    class_accuracy, tag_accuracy = _get_accuracies(model, test_dataset)
    logger.info(f"{class_accuracy=}")
    logger.info(f"{tag_accuracy=}")

    logger.info(f"Replacing linear layers with quantized versions ({bits=})")
    model = model.half()
    model = replace_linear_with_quant(model, bits=bits)

    state_dict_quant = model.state_dict()
    quant_output_path = Path(output_path) / f"model_w{bits}.safetensors"
    safetensors.torch.save_file(state_dict_quant, quant_output_path)
    logger.info(f"Saved model in quant({bits=}) to {str(output_path)}")

    loss = _get_loss(model, test_dataset)
    logger.info(f"Quantized model loss {loss=}")

    class_accuracy, tag_accuracy = _get_accuracies(model, test_dataset)
    logger.info(f"Quantized model {class_accuracy=}")
    logger.info(f"Quantized model {tag_accuracy=}")


if __name__ == "__main__" and not is_ipython():
    script_main(globals())
