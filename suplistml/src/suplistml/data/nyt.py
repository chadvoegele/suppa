__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import decimal
import re
from pathlib import Path

import pandas as pd

import suplistml.data
from suplistml.data.lfs import lfs_open


def _tokenize(s):
    """
    Tokenize on parenthesis, punctuation, spaces and American units followed by a slash.
    """

    s = _split_imperial_metric(s)
    s = _clump_fractions(s)
    tokens = [f for f in re.split(r"([,\(\)])+|\s+", s) if f]
    tokens = [_unclump(t) for t in tokens]
    return tokens


def _split_imperial_metric(s):
    """
    We sometimes give American units and metric units for baking recipes. For example:
        * 2 tablespoons/30 mililiters milk or cream
        * 2 1/2 cups/300 grams all-purpose flour

    The recipe database only allows for one unit, and we want to use the American one.
    But we must split the text on "cups/" etc. in order to pick it up.
    """
    american_units = [
        "cup",
        "tablespoon",
        "teaspoon",
        "pound",
        "ounce",
        "quart",
        "pint",
    ]
    for unit in american_units:
        s = s.replace(unit + "/", unit + " / ")
        s = s.replace(unit + "s/", unit + "s / ")

    return s


def _clump_fractions(s):
    """
    Replaces the whitespace between the integer and fractional part of a quantity
    with a dollar sign, so it's interpreted as a single token. The rest of the
    string is left alone.

        clump_fractions("aaa 1 2/3 bbb")
        # => "aaa 1$2/3 bbb"
    """
    return re.sub(r"(\d+)\s+(\d)/(\d)", r"\1$\2/\3", s)


def _convertUnicodeFractionsToAscii(s):
    """
    Replace unicode fractions with ascii representation, preceded by a
    space.

    "1\x215e" => "1 7/8"
    """

    fractions = {
        "\x215b": "1/8",
        "\x215c": "3/8",
        "\x215d": "5/8",
        "\x215e": "7/8",
        "\x2159": "1/6",
        "\x215a": "5/6",
        "\x2155": "1/5",
        "\x2156": "2/5",
        "\x2157": "3/5",
        "\x2158": "4/5",
        "\xbc": " 1/4",
        "\xbe": "3/4",
        "\x2153": "1/3",
        "\x2154": "2/3",
        "\xbd": "1/2",
    }

    for f_unicode, f_ascii in list(fractions.items()):
        s = s.replace(f_unicode, " " + f_ascii)

    return s


def _unclump(s):
    """
    Replacess $'s with spaces. The reverse of clump_fractions.
    """
    return re.sub(r"\$", " ", s)


def _singularize(word):
    """
    A poor replacement for the pattern.en singularize function, but ok for now.
    """

    units = {
        "cups": "cup",
        "tablespoons": "tablespoon",
        "teaspoons": "teaspoon",
        "pounds": "pound",
        "ounces": "ounce",
        "cloves": "clove",
        "sprigs": "sprig",
        "pinches": "pinch",
        "bunches": "bunch",
        "slices": "slice",
        "grams": "gram",
        "heads": "head",
        "quarts": "quart",
        "stalks": "stalk",
        "pints": "pint",
        "pieces": "piece",
        "sticks": "stick",
        "dashes": "dash",
        "fillets": "fillet",
        "cans": "can",
        "ears": "ear",
        "packages": "package",
        "strips": "strip",
        "bulbs": "bulb",
        "bottles": "bottle",
    }

    if word in list(units.keys()):
        return units[word]
    else:
        return word


def _parseNumbers(s):
    """
    Parses a string that represents a number into a decimal data type so that
    we can match the quantity field in the db with the quantity that appears
    in the display name. Rounds the result to 2 places.
    """
    ss = _unclump(s)

    m3 = re.match(r"^\d+$", ss)
    if m3 is not None:
        return decimal.Decimal(round(float(ss), 2))

    m1 = re.match(r"(\d+)\s+(\d)/(\d)", ss)
    if m1 is not None:
        num = int(m1.group(1)) + (float(m1.group(2)) / float(m1.group(3)))
        return decimal.Decimal(str(round(num, 2)))

    m2 = re.match(r"^(\d)/(\d)$", ss)
    if m2 is not None:
        num = float(m2.group(1)) / float(m2.group(2))
        return decimal.Decimal(str(round(num, 2)))

    return None


def get_nyt_path():
    path = Path(suplistml.data.__file__).parent / "nyt-ingredients-snapshot-2015.csv"
    return path


def _read_data(path, nrows=None):
    with lfs_open(path, "r") as f:
        df = pd.read_csv(f, nrows=nrows, dtype=str)
    return df


def _convert_unicode(row):
    ascii_row = _convertUnicodeFractionsToAscii(row["input"])
    row = row.copy()
    row["input"] = ascii_row
    return row


def _denormalize_qty(row):
    clumped = _clump_fractions(row["input"])
    tokens = _tokenize(clumped)
    maybe_qty = list(
        set(t for t in tokens if _parseNumbers(t) is not None and float(_parseNumbers(t)) == float(row["qty"]))
    )
    if len(maybe_qty) == 1:
        qty = maybe_qty[0]
        row = row.copy()
        row["qty"] = qty
    return row


def _denormalize_unit(row):
    tokens = _tokenize(row["input"])
    maybe_unit = list(set(t for t in tokens if _singularize(t) == row["unit"]))
    if len(maybe_unit) == 1:
        unit = maybe_unit[0]
        row = row.copy()
        row["unit"] = unit
    return row


def _clean_up_text(text):
    cleaned_text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("⁄", "/")
        .replace("—", "-")
        .replace("–", "-")
        .replace(" ", " ")
        .replace("“", "'")
        .replace("”", "'")
        .replace("\\t", " ")
        .replace("Fj", "")
        .replace("\x90", "")
    )
    cleaned_text = re.sub(r"<a.*?>(.*?)<\/a>", r"\1", cleaned_text)
    return cleaned_text


def _clean_up_text_in_row(row):
    row = row.copy()
    row["input"] = _clean_up_text(row["input"]).strip()
    row["name"] = _clean_up_text(row["name"]).strip()
    return row


def _case_check_name(row):
    input = row["input"]
    name = row["name"]
    if name in input:
        return row
    name_pattern = name.replace(")", "\\)").replace("(", "\\(")
    result = re.search(name_pattern, input, re.IGNORECASE)

    row = row.copy()
    if result is not None:
        matched_name = result.group(0)
        row["name"] = matched_name
    return row


def _plural_check_name(row):
    input = row["input"]
    name = row["name"]
    if name in input:
        return row

    plural_to_singular = {"jalapeños": "jalapeño"}
    singular_name = plural_to_singular.get(name, name)
    row = row.copy()
    if singular_name in input:
        row["name"] = singular_name

    return row


def _try_invert_modifier(row):
    input = row["input"]
    name = row["name"]
    if name in input:
        return row

    if ", " in name:
        result = re.match("(.*), (.*)", name)
        assert result is not None
        inverted_name = f"{result.group(2)} {result.group(1)}"
        if inverted_name in input:
            row = row.copy()
            row["name"] = inverted_name

    return row


def _process_row(row):
    row = _clean_up_text_in_row(row)
    row = _convert_unicode(row)
    row = _denormalize_qty(row)
    row = _denormalize_unit(row)
    row = _case_check_name(row)
    row = _plural_check_name(row)
    row = _try_invert_modifier(row)
    return row


def _process_data(df):
    df = df.fillna("")
    df = df[df["input"].str.len() != 0]
    df = df.apply(_process_row, axis=1)
    df = df.drop(["range_end"], axis=1)
    return df


def get_nyt_data(path=None, nrows=None):
    if path is None:
        path = get_nyt_path()
    df = _read_data(path, nrows=nrows)
    df = _process_data(df)
    return df
