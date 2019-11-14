"""
FEATURES:

raw_text:               The raw text

processed_text:         The raw text of the last word in the N-gram

text_pattern:           The raw text, after replacing
                            uppercase characters with X
                            lowercase with x
                            numbers with 0
                            repeating whitespace with a single whitespace
                            the rest with ?

bottom_margin:          Vertical coordinate of the bottom margin of the
                        N-gram normalized to the page height

top_margin:             Vertical coordinate of the top margin of the
                        N-gram normalized to the page height

right_margin:           Horizontal coordinate of the right margin of the
                        N-gram normalized to the page width

left_margin:            Horizontal coordinate of the left margin of the
                        N-gram normalized to the page width

has_digits:             Whether there are any digits 0-9 in the N-gram

length:                 Number of characters in the N-gram

position_on_line:       Count of words to the left of this N-gram normalized
                        to the count of total words on this line

line_size:              The number of words on this line

parses_as_date:         Whether the N-gram parses as a date

parses_as_number:       Whether the N-gram parses as a fractional amount
"""

import argparse
import copy
import logging
import re
import sys

import datefinder
from nltk import ngrams
import pandas as pd
from tqdm import tqdm

LOG = logging.getLogger(__name__)


EMPTY_SINGLE_GRAM = {
    "raw_text": "",
    "processed_text": "",
    "text_pattern": "",
    "length": 0,
    "line_size": 0,
    "position_on_line": 0,
    "has_digits": False,
    "bottom_margin": 1,
    "top_margin": 0,
    "left_margin": 0,
    "right_margin": 1,
    "parses_as_date": False,
    "parses_as_number": False,
    "labels": [],
    "closest_ngrams": [-1, -1, -1, -1],  # left, top, right, bottom
}


LABEL_DICT = {0: 0, 1: 1, 2: 2, 18: 3}


def _calculate_distance_between_grams(first, second):
    """Calculates distance vectors between two grams.

    Direction is from from first to second.
    """
    above = first["top_margin"] - second["bottom_margin"]
    below = second["top_margin"] - first["bottom_margin"]
    right = second["left_margin"] - first["right_margin"]
    left = first["left_margin"] - second["right_margin"]
    left_margin_offset = abs(second["left_margin"] - first["left_margin"])
    return left, above, right, below, left_margin_offset


def _parses_as_number(text):
    currencies = [r"\$", "USD", "€", "EUR"]
    currencies_pattern = "|".join(currencies)
    amount_pattern = (
        r"(?:^|\s|"
        + currencies_pattern
        + r")\d+\.\d+(?:$|\s|"
        + currencies_pattern
        + r")"
    )
    try:
        return re.search(amount_pattern, text)[0]
    except TypeError:  # no matches
        return None


def _process_text(ngram):
    """Returns proccessed text and what does it parse as."""
    # TODO check if preserving titles changes anything
    processed_text = []
    as_date = False
    as_number = False
    for word in ngram:
        try:
            as_date = bool(next(datefinder.find_dates(word)))
        except (StopIteration, OverflowError):
            as_date = False

        word_is_number = _parses_as_number(word)
        as_number = as_number or word_is_number

        if as_date:
            processed_text.append("date")
        elif word_is_number:
            processed_text.append("number")
        else:
            alphanum_only = "".join(filter(str.isalnum, word))
            if alphanum_only:
                processed_text.append(alphanum_only.lower())
            else:  # keep the same amount of words in raw and processed text
                # for word-wise labels would correspond to same words
                processed_text.append(word)
    as_number = as_number or as_date
    return " ".join(processed_text), as_date, as_number


def _text_pattern(raw_text):
    text = re.sub("\s+", " ", raw_text)
    text_pattern = ["?"] * len(text)
    for i, char in enumerate(text):
        if char.isnumeric():
            text_pattern[i] = "0"
        elif char.isspace():
            text_pattern[i] = " "
        elif char.islower():  # works for unicode chars too
            text_pattern[i] = "x"
        elif char.isupper():  # works for unicode chars too
            text_pattern[i] = "X"
        # non-letter, non-number, npn-whitespace left as ?
    return "".join(text_pattern)


def _group_by_file(df):
    """Filter data into individual files.

    Also estimates:
        width and height of each file.
        x coordinate for each token in each line for every file.
    """
    try:
        files = {name: {"rows": rows} for name, rows in df.groupby("file_name")}
    except KeyError:
        LOG.warning(
            "Couldn't find file names. "
            "Assuming everything comes from a single file."
        )
        files = {"untitled": {"rows": df}}
    for filename, file_info in files.items():
        # Assuming all pages of invoice have the same width/height
        files[filename]["xmin"] = min(xmin for xmin in file_info["rows"].x1)
        files[filename]["xmax"] = max(xmax for xmax in file_info["rows"].x2)
        files[filename]["width"] = (
            files[filename]["xmax"] - files[filename]["xmin"]
        )
        files[filename]["ymin"] = min(ymin for ymin in file_info["rows"].y1)
        files[filename]["ymax"] = max(ymax for ymax in file_info["rows"].y2)
        files[filename]["page_height"] = (
            files[filename]["ymax"] - files[filename]["ymin"]
        )
        files[filename]["height"] = (
            max(file_info["rows"].page_number) + 1
        ) * files[filename]["page_height"]
        files[filename]["rows"].words.map(lambda x: x.strip().split())
        # using dict instead of list to match row index
        token_coords = {}
        words = {}
        for row_num, row in file_info["rows"].iterrows():
            words[row_num] = row.words.strip().split()
            avg_token_width = (row.x2 - row.x1) / len(words[row_num])
            token_coords[row_num] = []
            for idx in range(len(words[row_num])):
                left_offset = row.x1 + idx * avg_token_width
                token_coords[row_num].append(
                    {"xmin": left_offset, "xmax": left_offset + avg_token_width}
                )
        files[filename]["rows"].words = pd.Series(words)
        files[filename]["rows"]["token_coords"] = pd.Series(token_coords)
    return files


def _find_ngram_labels(ngram, line):
    left_index = line.words.index(ngram[0])
    right_index = line.words.index(ngram[-1], left_index)
    labels = line.labels.split()[left_index : right_index + 1]
    return [LABEL_DICT[int(l)] for l in labels]


def _fill_gram_features(ngram, file_info, line):
    gram = copy.deepcopy(EMPTY_SINGLE_GRAM)
    (
        gram["processed_text"],
        gram["parses_as_date"],
        gram["parses_as_number"],
    ) = _process_text(ngram)
    raw_text = " ".join(ngram)
    gram["raw_text"] = raw_text
    gram["text_pattern"] = _text_pattern(raw_text)
    gram["length"] = len(" ".join(ngram))
    gram["line_size"] = len(line.words)
    gram["position_on_line"] = line.words.index(ngram[0]) / len(line.words)
    gram["has_digits"] = bool(re.search(r"\d", raw_text))
    leftmost_coord = line.token_coords[line.words.index(ngram[0])]["xmin"]
    gram["left_margin"] = (leftmost_coord - file_info["xmin"]) / file_info[
        "width"
    ]
    gram["top_margin"] = (
        line.y1
        - file_info["ymin"]
        + line.page_number * file_info["page_height"]
    ) / file_info["height"]
    rightmost_coord = line.token_coords[line.words.index(ngram[-1])]["xmax"]
    gram["right_margin"] = (rightmost_coord - file_info["xmin"]) / file_info[
        "width"
    ]
    gram["bottom_margin"] = (
        line.y2
        - file_info["ymin"]
        + line.page_number * file_info["page_height"]
    ) / file_info["height"]
    if "labels" in line:
        gram["labels"] = _find_ngram_labels(ngram, line)
    return gram


def _find_closest_grams(grams, start=0):
    """Find closest ngrams to the left/right and top/down.

    Modifies grams in-place.
    """
    for outer_gram in grams[start:]:
        distance = {
            "top": sys.maxsize,
            "bottom": sys.maxsize,
            "left": sys.maxsize,
            "right": sys.maxsize,
            "top_left": sys.maxsize,
            "bottom_left": sys.maxsize,
        }
        for inner_gram_id, inner_gram in enumerate(grams[start:]):
            inner_gram_id += start  # so id in loop matches id in sequence
            if id(outer_gram) == id(inner_gram):
                continue
            (
                left,
                above,
                right,
                below,
                left_margin_offset,
            ) = _calculate_distance_between_grams(outer_gram, inner_gram)
            # If in the same line, check for closest ngram to left and right
            if above == below:
                if distance["left"] > left > 0:
                    distance["left"] = left
                    outer_gram["closest_ngrams"][0] = inner_gram_id
                if distance["right"] > right > 0:
                    distance["right"] = right
                    outer_gram["closest_ngrams"][2] = inner_gram_id
            # If inner ngram is above outer gram
            elif (
                distance["top"] >= above >= 0
                and distance["top_left"] > left_margin_offset
            ):
                distance["top"] = above
                distance["top_left"] = left_margin_offset
                outer_gram["closest_ngrams"][1] = inner_gram_id
            # If inner ngram is below outer gram
            elif (
                distance["bottom"] >= below >= 0
                and distance["bottom_left"] > left_margin_offset
            ):
                distance["bottom"] = below
                distance["bottom_left"] = left_margin_offset
                outer_gram["closest_ngrams"][3] = inner_gram_id


def ngrammer(tokens, length=4):
    """
    Generates n-grams from the given tokens
    :param tokens: list of tokens in the text
    :param length: n-grams of up to this length
    :return: n-grams as tuples
    """
    for n in range(1, min(len(tokens) + 1, length + 1)):
        for gram in ngrams(tokens, n):
            yield gram


def extract_features(dataframe):
    """
    Create n-grams and extract features.

    :param path: path to pickled dataframe
    :return: dataframe containing n-grams and corresponding features
    """
    LOG.info("\nExtracting features...\n")
    # Avoid overwriting the original
    dataframe = copy.deepcopy(dataframe)
    files = _group_by_file(dataframe)
    del dataframe

    grams = []
    # Calculates N-grams of lengths ranging from 1-4 for each line in each
    # file and calculates 17 features for each N-gram.
    with tqdm(total=len(files)) as progress_bar:
        for filename, file_info in files.items():
            if "labels" not in file_info["rows"]:
                LOG.warning(
                    "File %s is unlabeled. Continuing to extract features.",
                    filename,
                )
            old_num_grams = len(grams)
            for _line_num, line in file_info["rows"].iterrows():
                for ngram in ngrammer(line.words):
                    grams.append(_fill_gram_features(ngram, file_info, line))
            _find_closest_grams(grams, start=old_num_grams)
            progress_bar.update(1)

    return pd.DataFrame(data=grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default="data/dftrain.pk", help="path to training data"
    )
    ap.add_argument(
        "--save_as",
        default="data/features.pk",
        help="save extracted features with this name",
    )
    args = ap.parse_args()
    features = extract_features(args.data)
    features.to_pickle(args.save_as, protocol=3)
    LOG.info("\nSaved features as {}".format(args.save_as))


if __name__ == "__main__":
    main()
