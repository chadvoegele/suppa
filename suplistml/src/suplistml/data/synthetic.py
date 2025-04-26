__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"


import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests
from google import genai

from suplistml.script import is_ipython, script_main, setup_logging

logger = logging.getLogger(__name__)


def get_aisles():
    aisles = [
        "bakery",
        "beverages",
        "dairy and eggs",
        "deli",
        "fresh fruits",
        "fresh vegetables",
        "frozen",
        "meat",
        "pantry",
        "seafood",
        "spices",
    ]
    return aisles


def get_nyt_inputs(nrows=None):
    from suplistml.data.nyt import _read_data, get_nyt_path

    def remove_hrefs(text):
        if pd.isna(text):
            return text
        text = re.sub(r"\((see )?<a? href=.*>(see )?recipe</a>\)", "", text)
        text = re.sub(r"<a href=.*>(.*?)</a>", r"\1", text)
        return text

    path = get_nyt_path()
    df = _read_data(path, nrows=nrows)
    df = df[["index", "input"]]
    df["input"] = df["input"].apply(remove_hrefs)
    df = df.drop_duplicates("input")
    df = df.dropna()

    return df


def dump_nyt_inputs(output_path: Path = "__AUTO__", nrows: int = None):
    setup_logging(output_path)
    df = get_nyt_inputs(nrows=nrows)
    df_path = output_path / "nyt_inputs.csv"
    df.to_csv(df_path, index=False)
    logger.info(f"Dumped NYT inputs to {df_path=}")


def get_schema(aisles: Optional[List[str]] = None):
    if aisles is None:
        aisles = get_aisles()

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "unit": {"type": "string"},
                "quantity": {"type": "string"},
                "aisle": {"type": "string", "enum": aisles},
            },
        },
    }
    return schema


def prompt_ingredients(ingredients: List[str], aisles: Optional[List[str]] = None):
    if aisles is None:
        aisles = get_aisles()

    list_ingredients = "\n".join([f"- {ing}" for ing in ingredients])
    prompt = f"""Extract the name, quantity, and unit from the ingredients, and choose an aisle from: {", ".join(aisles)}. If there isn't a unit or quantity, just leave them empty. Return only json row for every ingredient row. If there are multiple ingredients in a row, just return the first one.

Ingredients:
{list_ingredients}
"""  # noqa
    return prompt


@dataclass
class IngredientRow:
    input: str
    index: str


class GoogleGenAIExtractor:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-04-17", aisles: Optional[List[str]] = None):
        assert model in ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25"]
        self.model = model
        self.client = genai.Client(api_key=api_key)

        self.aisles = aisles if aisles is not None else get_aisles()
        logger.info(json.dumps({"message": "GoogleGenAIExtractor using", "aisles": self.aisles, "model": self.model}))

    @staticmethod
    def is_retriable(exception: Exception) -> bool:
        if isinstance(exception, genai.errors.APIError):
            return exception.code in [429, 502, 503, 504]
        if isinstance(exception, requests.exceptions.ConnectionError):
            return True
        return False

    def retry_generate_content(self, retry_counter: int = 0, backoff_base_s: int = 2, max_retries: int = 5, **kwargs):
        try:
            response = self.client.models.generate_content(**kwargs)
            return response
        except Exception as e:
            if self.is_retriable(e) and retry_counter < max_retries:
                logger.warning(
                    json.dumps(
                        {
                            "message": "Retrying generate_content",
                            "retry_counter": retry_counter,
                            "exception": str(e),
                        }
                    )
                )
                time.sleep(backoff_base_s**retry_counter)
                return self.retry_generate_content(**kwargs, retry_counter=retry_counter + 1)
            else:
                logger.error(
                    json.dumps(
                        {
                            "message": "Failed to generate content",
                            "retry_counter": retry_counter,
                            "exception": str(e),
                        }
                    )
                )
                raise e

    def extract(self, rows: List[IngredientRow]):
        logger.debug(f"Extracting {len(rows)} ingredients")
        ingredients = [row.input for row in rows]
        prompt = prompt_ingredients(ingredients, aisles=self.aisles)
        schema = get_schema(aisles=self.aisles)

        response_length_threshold = 20000
        response = self.retry_generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "max_output_tokens": response_length_threshold,
            },
        )

        if response.text is None:
            logger.error(
                {
                    "message": "response text is None",
                    "prompt": prompt,
                    "indices": [row.index for row in rows],
                }
            )
            yield from []
            return

        logger.debug(
            json.dumps(
                {
                    "message": "lm response length",
                    "response length": len(response.text),
                    "prompt length": len(prompt),
                    "ingredients length": len(rows),
                }
            )
        )

        try:
            extracted = json.loads(response.text)

        except json.JSONDecodeError:
            logger.error(
                {
                    "message": "response json parse failed",
                    "prompt": prompt,
                    "text": response.text,
                    "indices": [row.index for row in rows],
                }
            )
            yield from []
            return

        if len(extracted) != len(rows):
            logger.error(
                {
                    "message": "response length mismatch",
                    "prompt": prompt,
                    "text": response.text,
                    "extracted length": len(extracted),
                    "rows length": len(rows),
                    "indices": [row.index for row in rows],
                }
            )
            yield from []
            return

        input_output = [{**asdict(row), **e} for row, e in zip(rows, extracted)]
        yield from input_output


class BatchExtractor:
    def __init__(self, extractor, batch_size: int = 10):
        self.extractor = extractor
        self.batch_size = batch_size

    def extract(self, ingredients: List[IngredientRow]):
        for i in range(0, len(ingredients), self.batch_size):
            batch = ingredients[i : i + self.batch_size]
            extracted = self.extractor.extract(batch)
            yield from extracted


class CachedExtractor:
    def __init__(self, extractor, data):
        self.extractor = extractor
        self.data = data
        self.data_indices = [row["index"] for row in self.data]

    def extract(self, rows: List[IngredientRow]):
        index_to_extract = [row.index for row in rows]

        if all(index in self.data_indices for index in index_to_extract):
            logger.info(json.dumps({"message": "Using cached data", "indices": index_to_extract}))
            extracted = [row for row in self.data if row["index"] in index_to_extract]
            yield from extracted
            return

        extracted = self.extractor.extract(rows)
        yield from extracted


def timed_enumerate(iterable: Iterable):
    iterator = iter(iterable)
    index = 0

    while True:
        try:
            start = time.perf_counter()
            value = next(iterator)
            end = time.perf_counter()
            duration_s = end - start
            yield index, duration_s, value
            index += 1
        except StopIteration:
            break


def test_timed_enumerate():
    def my_list():
        for i in range(10):
            time.sleep(0.5)  # Simulate some processing time
            yield i

    for index, duration_s, value in timed_enumerate(my_list()):
        print(f"Index: {index}, Time taken: {duration_s:.6f} seconds, Value: {value}")


def run_extract(
    api_key_file: str,
    output_path: Path = "__AUTO__",
    nrows: Optional[int] = None,
    batch_size: int = 4,
    data_file: str = None,
):
    setup_logging(output_path)
    logger.info(f"{output_path=}")

    api_key = Path(api_key_file).read_text().strip()
    extractor = GoogleGenAIExtractor(api_key, model="gemini-2.5-flash-preview-04-17", aisles=get_aisles())

    if data_file:
        logger.info(f"Using cached data from {data_file=}")
        data = [json.loads(row.strip()) for row in Path(data_file).read_text().splitlines()]
        extractor = CachedExtractor(extractor, data)

    extractor = BatchExtractor(extractor, batch_size=batch_size)

    logger.debug("Loading nyt data")
    df = get_nyt_inputs(nrows=nrows)
    ingredients_rows = [IngredientRow(**row) for row in df.to_dict(orient="records")]

    extracted_path = output_path / "extracted.json"
    extracted_path.write_text("")
    with extracted_path.open("a") as f:
        for i, duration_s, extracted in timed_enumerate(extractor.extract(ingredients_rows)):
            logger.info(
                json.dumps({"message": "progress", "i": i, "total": len(ingredients_rows), "duration_s": duration_s})
            )
            logger.info(json.dumps({"message": "extracted result", "extracted": extracted}))
            f.write(json.dumps(extracted) + "\n")
    logger.info(f"Dumped extracted data to {extracted_path=}")


if __name__ == "__main__" and not is_ipython():
    script_main(globals())
