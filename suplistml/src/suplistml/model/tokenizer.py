__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers.utils import TRANSFORMERS_CACHE

from suplistml.script import (
    is_ipython,
    make_output_path,
    resolve_module_name,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _get_classes():
    CLASSES = [
        "bakery",
        "beverages",
        "condiments",
        "dairy",
        "deli",
        "frozen foods",
        "grains",
        "meat",
        "nuts",
        "pasta",
        "produce",
        "seafood",
        "spices",
        "vegetables",
    ]
    return CLASSES


def get_class_tokenizer(classes=None):
    if classes is None:
        classes = _get_classes()
    tokenizer = _get_tokenizer_from_tokens(classes)
    return tokenizer


def _get_tokenizer_from_tokens(tokens):
    all_tokens = ["<pad>", "<unk>", *tokens]
    vocab = {t: i for i, t in enumerate(all_tokens)}
    base_tokenizer = Tokenizer(WordLevel(vocab))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
    return tokenizer


def _get_tags():
    TAGS = ["NAME", "OTHER", "QTY", "UNIT"]
    return TAGS


def get_tag_tokenizer(tags=None):
    if tags is None:
        tags = _get_tags()
    tokenizer = _get_tokenizer_from_tokens(tags)
    return tokenizer


def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        Path(TRANSFORMERS_CACHE) / "models--intfloat--e5-small-v2/snapshots/dca8b1a9dae0d4575df2bf423a5edb485a431236"
    )
    return tokenizer


def export_tokenizers(output_path: Path, tokenizer):
    tag_tokenizer = get_tag_tokenizer()
    export_tokenizer(output_path / "tag_tokenizer", tag_tokenizer)

    class_tokenizer = get_class_tokenizer()
    export_tokenizer(output_path / "class_tokenizer", class_tokenizer)


def export_tokenizer(output_path: Path, tokenizer):
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)


def main():
    module_name = resolve_module_name(__name__)
    output_path = make_output_path(module_name)
    setup_logging(output_path)

    export_tokenizers(output_path)


if __name__ == "__main__" and not is_ipython():
    main()
