__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import json
import logging
from pathlib import Path
from string import Template

import pandas as pd
from llama_cpp import Llama, LlamaGrammar

from suplistml.data.lfs import LfsObject, lfs_open
from suplistml.data.nyt import get_nyt_data
from suplistml.script import is_ipython

logger = logging.getLogger(__name__)


def get_aisles():
    aisles = [
        "bakery",
        "beverages",
        "dairy",
        "deli",
        "frozen foods",
        "fruits",
        "meat",
        "baking",
        "condiments",
        "grains",
        "nuts",
        "pasta",
        "seafood",
        "spices",
        "vegetables",
        "produce",
    ]
    return aisles


def load_llm(model_path: Path):
    logger.debug(f"loading model at {model_path=}")
    model_lfs = LfsObject.try_from_path(model_path)
    if model_lfs is None:
        raise RuntimeError(f"{model_lfs=} not an lfs file!")
    logger.debug(f"loading model at {model_lfs.data_path=}")
    llm = Llama(model_path=str(model_lfs.data_path), verbose=False)
    return llm


def get_llm_grammar():
    aisles = get_aisles()
    aisles_gbnf = " | ".join(f'"{a}"' for a in aisles)
    grammar_gbnf = f"""root ::= item
item ::= {aisles_gbnf}"""
    grammar = LlamaGrammar.from_string(grammar_gbnf)
    return grammar


def categorize_with_llm(llm, grammar, food):
    prompt_template = Template(
        "Which aisle are frozen mango in? frozen foods\n Which aisle are shelled pistachios in? nuts\n Which aisle is ginger in? produce\n Which aisle is ${food} in?"  # noqa
    )

    prompt = prompt_template.substitute({"food": food})
    response = llm(prompt, max_tokens=4, grammar=grammar, temperature=0)
    return response["choices"][0]["text"]


def run():
    model_path = Path(__file__).parent / "mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf"
    llm = load_llm(model_path)
    grammar = get_llm_grammar()
    food = "white or light brown sugar"
    categorize_with_llm(llm, grammar, food)


def process_nyt_data(nrows=None):
    model_path = Path(__file__).parent / "mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf"
    llm = load_llm(model_path)
    grammar = get_llm_grammar()

    logger.debug("Loading nyt data")
    nyt_data = get_nyt_data(nrows=nrows)
    logger.debug("Extracting food parts")
    food_rows = nyt_data["name"].to_list()
    food_rows = set(food_rows)

    with open("nyt_llm_categories.json", "w") as f:
        for food_i, food in enumerate(food_rows):
            logger.debug(f"Processing row {food_i} of {len(food_rows)}: {food}")
            aisle = categorize_with_llm(llm, grammar, food)
            result = {"food": food, "aisle": aisle}
            result_json = json.dumps(result)
            logger.debug(result_json)
            f.write(result_json + "\n")


def get_llm_aisles_path():
    if is_ipython():
        path = Path("suplistml") / "data" / "nyt_llm_categories+model=mistral7b+prompt=icl.2024jan29.json"
    else:
        path = Path(__file__).parent / "nyt_llm_categories+model=mistral7b+prompt=icl.2024jan29.json"
    return path


def _read_data(path, nrows=None):
    with lfs_open(path, "r") as f:
        df = pd.read_json(f, lines=True, nrows=nrows)
    return df


def _fix_space_punctuation(row, pattern):
    assert len(pattern) == 2
    assert " " in pattern
    food = row["food"]

    if pattern not in food:
        return row

    row = row.copy()
    no_space_pattern = pattern.replace(" ", "")
    fixed_food = food.replace(pattern, no_space_pattern)
    row["food"] = fixed_food
    return row


def _fix_bad_aisles(row):
    aisle = row["aisle"]

    bad_to_good_aisle = {
        "baking": "bakery",
        "cond": "condiments",
        "condiment": "condiments",
    }

    if aisle not in bad_to_good_aisle.keys():
        return row

    good_aisle = bad_to_good_aisle[aisle]
    row = row.copy()
    row["aisle"] = good_aisle
    return row


def _process_row(row):
    row = _fix_bad_aisles(row)
    row = _fix_space_punctuation(row, " ,")
    row = _fix_space_punctuation(row, "( ")
    row = _fix_space_punctuation(row, " )")
    return row


def get_llm_aisles_data(path=None, nrows=None):
    if path is None:
        path = get_llm_aisles_path()
    df = _read_data(path, nrows=nrows)
    df = df.apply(_process_row, axis=1)
    return df


def main():
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logging.getLogger("suplistml").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    process_nyt_data()


if __name__ == "__main__" and not is_ipython():
    main()
