__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import json
import logging
import random
from pathlib import Path
from typing import Optional

import evaluate
import numpy as np
import safetensors.torch
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.trainer import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.utils import TRANSFORMERS_CACHE

from suplistml.data.lfs import lfs_open
from suplistml.dataset.multi_dataset import MultiDataset
from suplistml.model.multi_bert import MultiBert, MultiBertConfig
from suplistml.model.tokenizer import (
    export_tokenizer,
    get_class_tokenizer,
    get_tag_tokenizer,
    get_tokenizer,
)
from suplistml.script import (
    is_ipython,
    script_main,
    setup_logging,
)

logger = logging.getLogger(__name__)


def seed_everything(seed=8385):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(json.dumps({"message": "Seeding random number generators", "seed": seed}))


def get_joint_model(tokenizer, tag_tokenizer, class_tokenizer):
    pretrained_model = get_model()

    config = MultiBertConfig(
        vocab_size=len(tokenizer),
        n_class_labels=len(class_tokenizer),
        n_tag_labels=len(tag_tokenizer),
        hidden_size=pretrained_model.config.hidden_size,
        num_hidden_layers=pretrained_model.config.num_hidden_layers,
        num_attention_heads=pretrained_model.config.num_attention_heads,
        intermediate_size=pretrained_model.config.intermediate_size,
    )
    model = MultiBert(config)
    model.bert.encoder.load_state_dict(pretrained_model.bert.encoder.state_dict(), strict=True)
    model.bert.embeddings.load_state_dict(pretrained_model.bert.embeddings.state_dict(), strict=True)
    return model


def run_training(
    output_path: Path = "__AUTO__",
    nrows: int = None,
    eval_steps: int = 16,
    global_batch_size: int = 256,
    debug: bool = False,
    dataset: str = "nyt_full_synthetic+model=gemini25flash0417.2025apr26",
):
    setup_logging(output_path)
    seed_everything()
    logger.info(json.dumps({k: str(v) for k, v in locals().items()}))

    if debug:
        nrows = 100
        eval_steps = 2
        global_batch_size = 1

    if dataset == "nyt_full_synthetic+model=gemini25flash0417.2025apr26":
        from suplistml.data.synthetic import get_aisles, get_synthetic_df

        df = get_synthetic_df(nrows=nrows)
        classes = get_aisles()

    elif dataset == "nyt_llm_categories+model=mistral7b+prompt=icl.2024jan29":
        from suplistml.dataset.multi_dataset import _get_joined_nyt_aisles_dfs
        from suplistml.model.tokenizer import _get_classes

        classes = _get_classes()
        df = _get_joined_nyt_aisles_dfs(nrows=nrows)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info("Getting tokenizer")
    tokenizer = get_tokenizer()
    tag_tokenizer = get_tag_tokenizer()
    class_tokenizer = get_class_tokenizer(classes=classes)

    logger.info("Getting data")
    dataset = MultiDataset(tokenizer, tag_tokenizer, class_tokenizer, nrows=nrows, df=df, seqlen=256)
    model = get_joint_model(tokenizer, tag_tokenizer, class_tokenizer)

    model.config.save_pretrained(output_path / "config")
    export_tokenizer(output_path=output_path / "tokenizer", tokenizer=tokenizer)
    export_tokenizer(output_path=output_path / "tag_tokenizer", tokenizer=tag_tokenizer)
    export_tokenizer(output_path=output_path / "class_tokenizer", tokenizer=class_tokenizer)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        (class_logits, tag_logits), (class_labels, tag_labels) = eval_pred
        class_predictions = class_logits.argmax(axis=-1).flatten()
        tag_predictions = tag_logits.argmax(axis=-1).flatten()
        class_labels = class_labels.flatten()
        tag_labels = tag_labels.flatten()
        tag_predictions = tag_predictions[tag_labels != -100]
        tag_labels = tag_labels[tag_labels != -100]

        class_accuracy = metric.compute(predictions=class_predictions.tolist(), references=class_labels.tolist())
        tag_accuracy = metric.compute(predictions=tag_predictions.tolist(), references=tag_labels.tolist())
        metrics = {"class": class_accuracy, "tag": tag_accuracy}
        return metrics

    n_samples = len(dataset)
    n_test_samples = min(int(0.1 * n_samples), 1024)
    n_train_samples = n_samples - n_test_samples
    split_rng = torch.Generator().manual_seed(8385)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train_samples, n_test_samples], generator=split_rng
    )

    rng = np.random.default_rng(seed=8385)
    subset_indices = rng.choice(np.arange(n_train_samples), n_test_samples, replace=False)
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices.tolist())

    per_device_train_batch_size = 20
    gradient_accumulation_steps = max(1, global_batch_size // per_device_train_batch_size)
    logger.info(f"Per device train batch size: {per_device_train_batch_size}")
    logger.info(f"Global batch size: {global_batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=20,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=128,
        save_total_limit=3,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset,
        eval_dataset={"train": train_subset, "test": test_dataset},
        compute_metrics=compute_metrics,
    )

    class LoggerCallback(TrainerCallback):
        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs,
            **kwargs,
        ):
            logger.info(json.dumps(logs))

    trainer.add_callback(LoggerCallback())
    trainer.train()


def predict(output_path: Path = "__AUTO__"):
    from suplistml.data.synthetic import get_aisles

    logger.info("Getting tokenizer")
    tokenizer = get_tokenizer()
    tag_tokenizer = get_tag_tokenizer()
    classes = get_aisles()
    class_tokenizer = get_class_tokenizer(classes=classes)

    model_root = Path("src/suplistml/models/run+1748084792")
    trained_model_path = model_root / "model.safetensors"
    with lfs_open(trained_model_path, "rb") as g:
        safetensors_bytes = g.read()
        state_dict = safetensors.torch.load(safetensors_bytes)

    config_path = model_root / "config.json"
    with lfs_open(config_path, "r") as f:
        config_dict = json.load(f)
        config = MultiBertConfig.from_dict(config_dict)

    model = MultiBert(config)
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    data = [
        {"input": "1 cup dried beans"},
        {"input": "red apples"},
        {"input": "2 green apples"},
        {"input": "2 pears"},
        {"input": "10 green beans"},
        {"input": "1 pound ground beef"},
        {"input": "1 tablespoon flour"},
        {"input": "6 radishes"},
        {"input": "133g masa harina"},
        {"input": "187g water"},
        {"input": "1 avocado"},
        {"input": "Salt"},
        {"input": "1 clove garlic, minced"},
        {"input": "Juice from 1 lime"},
        {"input": "2 cups basil leaves"},
        {"input": "1 cup Parmesan cheese"},
        {"input": "¼ cup pine nuts"},
        {"input": "2 garlic cloves"},
        {"input": "½ teaspoon fine sea salt"},
        {"input": "1 cup extra-virgin olive oil"},
        {"input": "2 tablespoons mayo"},
        {"input": "4-6 dates"},
        {"input": "1 tablespoon honey"},
    ]
    in_ids = tokenizer([row["input"] for row in data], return_tensors="pt", padding=True)
    outputs = model(in_ids["input_ids"], in_ids["attention_mask"])
    class_preds = outputs.class_logits.argmax(dim=-1)
    tag_preds = outputs.tag_logits.argmax(dim=-1)

    for i, row in enumerate(data):
        input = row["input"]
        class_pred = class_tokenizer.decode(class_preds[i : (i + 1)].tolist())
        tag_pred = [tag_tokenizer.decode(t) for t in tag_preds[i, in_ids["attention_mask"][i, 1:] == 1].tolist()]
        tokens = [
            tokenizer.decode(t) for t in in_ids["input_ids"][i : (i + 1), in_ids["attention_mask"][i, :] == 1][0, 1:]
        ]
        tokens_tag = " ".join("|".join((token, tag)) for token, tag in zip(tokens, tag_pred))
        print(
            f"""
{input=}
{class_pred=}
{tokens_tag=}
""".strip()
        )
        print()


def get_model():
    config = BertConfig.from_pretrained(
        Path(TRANSFORMERS_CACHE)
        / "models--intfloat--e5-small-v2/snapshots/dca8b1a9dae0d4575df2bf423a5edb485a431236/config.json",
    )
    model = BertForSequenceClassification(config)
    state_dict = safetensors.torch.load_file(
        Path(TRANSFORMERS_CACHE)
        / "models--intfloat--e5-small-v2/snapshots/dca8b1a9dae0d4575df2bf423a5edb485a431236/model.safetensors"
    )
    state_dict = {k: v for k, v in state_dict.items() if k != "embeddings.position_ids"}
    model.bert.load_state_dict(state_dict, strict=True)
    return model


def test_candle():
    tokenizer = get_tokenizer()
    tag_tokenizer = get_tag_tokenizer()
    class_tokenizer = get_class_tokenizer()

    model_root = Path("suplistml/models/run+1733494653")
    trained_model_path = model_root / "model.safetensors"
    with lfs_open(trained_model_path, "rb") as g:
        safetensors_bytes = g.read()
        state_dict = safetensors.torch.load(safetensors_bytes)

    config_path = model_root / "config.json"
    with lfs_open(config_path, "r") as f:
        config_dict = json.load(f)
        config = MultiBertConfig.from_dict(config_dict)

    model = MultiBert(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    input_ids = tokenizer("[CLS] 2 apples", return_tensors="pt", add_special_tokens=False)
    print(input_ids["input_ids"])
    print(input_ids["attention_mask"])
    outputs = model(input_ids["input_ids"], input_ids["attention_mask"])
    print(f"{outputs.class_logits.tolist()=}")
    print(f"{outputs.tag_logits.tolist()=}")
    class_preds = outputs.class_logits.argmax(dim=-1)
    tag_preds = outputs.tag_logits.argmax(dim=-1)
    class_pred = class_tokenizer.batch_decode(class_preds)
    tag_pred = tag_tokenizer.batch_decode(tag_preds)
    print(f"{class_pred=}")
    print(f"{tag_pred=}")


def dataset_checker(output_path: Path = "__AUTO__", nrows: Optional[int] = None):
    setup_logging(output_path)
    seed_everything()
    logger.info(json.dumps({k: str(v) for k, v in locals().items()}))

    from suplistml.data.synthetic import get_aisles, get_synthetic_df

    tokenizer = get_tokenizer()
    tag_tokenizer = get_tag_tokenizer()
    class_tokenizer = get_class_tokenizer(classes=get_aisles())

    df = get_synthetic_df(nrows=None)
    logger.info("Getting data")
    dataset = MultiDataset(tokenizer, tag_tokenizer, class_tokenizer, nrows=nrows, seqlen=256, df=df)
    n = len(dataset)
    n_nothing = 0
    n_no_class = 0
    n_no_tags = 0
    for row in dataset:
        input_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        if (row["tag_labels"] == -100).all() and (row["class_labels"] == -100).all():
            logger.info(f"{input_text} has no labels at all!!!!")
            n_nothing += 1
        elif (row["class_labels"] == -100).all():
            logger.info(f"{input_text} has no class labels")
            n_no_class += 1
        elif (row["tag_labels"] == -100).all():
            logger.info(f"{input_text} has no tag labels")
            n_no_tags += 1
    logger.info(f"{n=} {n_nothing=} {n_no_class=} {n_no_tags=}")


def reshard_model(output_path: Path):
    model_root = Path("suplistml/models/run+1733494653")
    trained_model_path = model_root / "model.safetensors"
    with lfs_open(trained_model_path, "rb") as g:
        safetensors_bytes = g.read()
        state_dict = safetensors.torch.load(safetensors_bytes)

    config_path = model_root / "config.json"
    with lfs_open(config_path, "r") as f:
        config_dict = json.load(f)
        config = MultiBertConfig.from_dict(config_dict)

    model = MultiBert(config)
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(save_directory=output_path / "shards", max_shard_size="20MB")


def convert_to_16(output_path: Path):
    model_root = Path("src/suplistml/models/run+1733494653")
    trained_model_path = model_root / "model.safetensors"
    with lfs_open(trained_model_path, "rb") as g:
        safetensors_bytes = g.read()
        state_dict = safetensors.torch.load(safetensors_bytes)

    state_dict16 = {k: v.half() for k, v in state_dict.items()}
    safetensors16_file = output_path / "model.safetensors"
    safetensors.torch.save_file(state_dict16, safetensors16_file)
    logger.info(f"Saved 16-bit model to {safetensors16_file}")


if __name__ == "__main__" and not is_ipython():
    script_main(globals())
