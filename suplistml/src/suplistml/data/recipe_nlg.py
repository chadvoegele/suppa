"""
RecipeNLG Dataset License

I (the "Researcher") have requested permission to use the RecipeNLG dataset (the "Dataset") at Poznań University of Technology (PUT). In exchange for such permission, Researcher hereby agrees to the following terms and conditions:

    Researcher shall use the Dataset only for non-commercial research and educational purposes.
    PUT makes no representations or warranties regarding the Dataset, including but not limited to warranties of non-infringement or fitness for a particular purpose.
    Researcher accepts full responsibility for his or her use of the Dataset and shall defend and indemnify PUT, including its employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the Dataset including but not limited to Researcher's use of any copies of copyrighted images or text that he or she may create from the Dataset.
    Researcher may provide research associates and colleagues with access to the Dataset provided that they first agree to be bound by these terms and conditions.
    If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.
"""  # noqa

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from suplistml.data.lfs import lfs_open
from suplistml.script import is_ipython


def get_recipe_nlg_path():
    if is_ipython():
        path = Path("src/suplistml") / "data/recipe_nlg/full_dataset.csv"
    else:
        path = Path(__file__).parent / "recipe_nlg/full_dataset.csv"
    return path


@dataclass
class Recipe:
    title: str
    ingredients: list[str]
    directions: list[str]
    link: str
    source: str
    NER: list[str]


def get_recipe_nlg_data(path=None) -> Recipe:
    """Generator function that reads recipe_nlg CSV line by line into Recipe dataclass."""
    if path is None:
        path = get_recipe_nlg_path()

    with lfs_open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield Recipe(
                title=row["title"],
                ingredients=json.loads(row["ingredients"]),
                directions=json.loads(row["directions"]),
                link=row["link"],
                source=row["source"],
                NER=json.loads(row["NER"]),
            )


def get_recipe_nlg_dataframe(path=None, nrows=None) -> pd.DataFrame:
    """Read recipe_nlg data into a pandas DataFrame.

    Args:
        path: Path to the CSV file. If None, uses the default path.
        nrows: Number of rows to read. If None, reads all rows.

    Returns:
        DataFrame with columns: title, ingredients, directions, link, source, NER
    """
    if path is None:
        path = get_recipe_nlg_path()

    with lfs_open(path, "r") as f:
        df = pd.read_csv(f, nrows=nrows)
    df["ingredients"] = df["ingredients"].apply(json.loads)
    df["directions"] = df["directions"].apply(json.loads)
    df["NER"] = df["NER"].apply(json.loads)
    return df
