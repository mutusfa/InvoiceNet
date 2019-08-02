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

page_height:            The height of the page of this N-gram

page_width:             The width of the page of this N-gram

parses_as_amount:       Whether the N-gram parses as a fractional amount

parses_as_date:         Whether the N-gram parses as a date

parses_as_number:       Whether the N-gram parses as an integer
"""

import argparse
import copy
import logging
import pickle
import re
import sys

import datefinder
from nltk import ngrams
import pandas as pd
from tqdm import tqdm


EMPTY_SINGLE_GRAM = {
    'raw_text': "",
    'processed_text': "",
    'text_pattern': "",
    'length': 0,
    'line_size': 0,
    'position_on_line': 0,
    'has_digits': False,
    'bottom_margin': 1,
    'top_margin': 0,
    'left_margin': 0,
    'right_margin': 1,
    'page_width': sys.maxsize,
    'page_height': sys.maxsize,
    'parses_as_amount': False,
    'parses_as_date': False,
    'parses_as_number': False,
    'label': 0,
    'closest_ngrams': [-1, -1, -1, -1],  # left, top, right, bottom
}


def _calculate_distance_between_grams(first, second):
    """Calculates distance vectors between two grams.

    Direction is from from first to second.
    """
    above = first['top_margin'] - second['bottom_margin']
    below = second['top_margin'] - first['bottom_margin']
    right = second['left_margin'] - first['right_margin']
    left = first['left_margin'] - second['right_margin']
    left_margin_offset = abs(
        second['left_margin'] - first['left_margin'])
    return left, above, right, below, left_margin_offset


def _parses_as_amount(text):
    amount_pattern = r'(?:^|\s)\d+\.\d+(?:$|\s)'
    try:
        return re.search(amount_pattern, text)[0]
    except TypeError:  # no matches
        return None


def _process_text(ngram):
    """Returns proccessed text and what does it parse as."""
    processed_text = []
    as_date = False
    as_amount = False
    as_number = False
    for word in ngram:
        try:
            as_date = bool(
                list(datefinder.find_dates(word)))
        except OverflowError:
            as_date = False

        word_is_number = word.isnumeric()
        as_number = as_number or word_is_number

        if _parses_as_amount(word):
            processed_text.append('amount')
            as_amount = True
        elif as_date:
            processed_text.append('date')
        elif word_is_number:
            processed_text.append('number')
        else:
            processed_text.append(word.lower())
    as_number = as_number or as_date or as_amount
    return ' '.join(processed_text), as_date, as_amount, as_number


def _text_pattern(raw_text):
    text = re.sub('\s+', ' ', raw_text)
    text_pattern = ['?'] * len(text)
    for i, char in enumerate(text):
        if char.isnumeric():
            text_pattern[i] = '0'
        elif char.isspace():
            text_pattern[i] = ' '
        elif char.islower():  # works for unicode chars too
            text_pattern[i] = 'x'
        elif char.isupper():  # works for unicode chars too
            text_pattern[i] = 'X'
        # non-letter, non-number, npn-whitespace left as ?
    return ''.join(text_pattern)


def _group_by_file(df):
    """Filters data into individual files

    Also estimates:
        width and height of each file.
        x coordinate for each token in each line for every file.
    """
    files = {}
    for _i, row in df.iterrows():
        filename = row['files']
        if filename not in files:
            files[filename] = {
                'lines': {'words': [], 'labels': [], 'ymin': [], 'ymax': []},
                'xmin': sys.maxsize,
                'ymin': sys.maxsize,
                'xmax': 0,
                'ymax': 0
            }
        tokens = row['words'].strip().split(' ')
        char_length = (row['coords'][2] - row['coords']
                       [0]) / len(row['words'].strip())
        token_coords = [{'xmin': row['coords'][0],
                         'xmax': row['coords'][0] + (char_length * len(tokens[0]))}]
        for idx in range(1, len(tokens)):
            token_coords.append({
                'xmin': token_coords[-1]['xmax'] + char_length,
                'xmax': token_coords[-1]['xmax'] + (char_length * (len(tokens[idx])+1))
            })
        files[filename]['lines']['words'].append(
            {'tokens': tokens, 'coords': token_coords})
        files[filename]['lines']['labels'].append(row['labels'])
        files[filename]['lines']['ymin'].append(row['coords'][1])
        files[filename]['lines']['ymax'].append(row['coords'][3])
        files[filename]['xmin'] = min(
            files[filename]['xmin'], row['coords'][0])
        files[filename]['ymin'] = min(
            files[filename]['ymin'], row['coords'][1])
        files[filename]['xmax'] = max(
            files[filename]['xmax'], row['coords'][2])
        files[filename]['ymax'] = max(
            files[filename]['ymax'], row['coords'][3])
        files[filename]['page_width'] = \
            files[filename]['xmax'] - files[filename]['xmin']
        files[filename]['page_height'] = \
            files[filename]['ymax'] - files[filename]['ymin']

    return files


def _fill_gram_features(
    ngram,
    tokens,
    token_coords,
    file_info,
    ymin,
    ymax,
    label
):
    gram = copy.deepcopy(EMPTY_SINGLE_GRAM)
    label_dict = {0: 0, 1: 1, 2: 2, 18: 3}
    (
        gram['processed_text'],
        gram['parses_as_date'],
        gram['parses_as_amount'],
        gram['parses_as_number']
    ) = _process_text(ngram)
    raw_text = ' '.join(ngram)
    gram['raw_text'] = raw_text
    gram['text_pattern'] = _text_pattern(raw_text)
    gram['length'] = len(' '.join(ngram))
    gram['line_size'] = len(tokens)
    gram['position_on_line'] = tokens.index(ngram[0])/len(tokens)
    gram['has_digits'] = bool(re.search(r'\d', raw_text))
    gram['left_margin'] = (
        (token_coords[tokens.index(ngram[0])]['xmin'] - file_info['xmin']) /
        file_info['page_width']
    )
    gram['top_margin'] = (ymin - file_info['ymin']) / file_info['page_height']
    gram['right_margin'] = (
        (token_coords[tokens.index(ngram[-1])]['xmax'] - file_info['xmin']) /
        file_info['page_width']
    )
    gram['bottom_margin'] = \
        (ymax - file_info['ymin']) / file_info['page_height']
    gram['label'] = label
    return gram


def _find_closest_grams(grams, start=0):
    """Finds closest ngrams to the left/right and top/down.

    Modifies grams in-place.
    """
    for outer_gram in grams[start:]:
        distance = {
            'top': sys.maxsize,
            'bottom': sys.maxsize,
            'left': sys.maxsize,
            'right': sys.maxsize,
            'top_left': sys.maxsize,
            'bottom_left': sys.maxsize,
        }
        for inner_gram_id, inner_gram in enumerate(grams[start:]):
            inner_gram_id += start  # so id in loop matches id in sequence
            if id(outer_gram) == id(inner_gram):
                continue
            left, above, right, below, left_margin_offset = \
                _calculate_distance_between_grams(outer_gram, inner_gram)
            # If in the same line, check for closest ngram to left and right
            if above == below:
                if distance['left'] > left > 0:
                    distance['left'] = left
                    outer_gram['closest_ngrams'][0] = inner_gram_id
                if distance['right'] > right > 0:
                    distance['right'] = right
                    outer_gram['closest_ngrams'][2] = inner_gram_id
            # If inner ngram is above outer gram
            elif distance['top'] >= above >= 0 and \
                    distance['top_left'] > left_margin_offset:
                distance['top'] = above
                distance['top_left'] = left_margin_offset
                outer_gram['closest_ngrams'][1] = inner_gram_id
            # If inner ngram is below outer gram
            elif distance['bottom'] >= below >= 0 and \
                    distance['bottom_left'] > left_margin_offset:
                distance['bottom'] = below
                distance['bottom_left'] = left_margin_offset
                outer_gram['closest_ngrams'][3] = inner_gram_id


def ngrammer(tokens, length=4):
    """
    Generates n-grams from the given tokens
    :param tokens: list of tokens in the text
    :param length: n-grams of up to this length
    :return: n-grams as tuples
    """
    for n in range(1, min(len(tokens) + 1, length+1)):
        for gram in ngrams(tokens, n):
            yield gram


def extract_features(path):
    """
    Loads a pickled dataframe from the given path, creates n-grams and extracts features
    :param path: path to pickled dataframe
    :return: dataframe containing n-grams and corresponding features
    """

    with open(path, 'rb') as pklfile:
        df = pickle.load(pklfile)

    logging.info("\nExtracting features...\n")
    files = _group_by_file(df)
    del df

    grams = []
    # Calculates N-grams of lengths ranging from 1-4 for each line in each
    # file and calculates 17 features for each N-gram.
    with tqdm(total=len(files)) as progress_bar:
        for file_info in files.values():
            old_num_grams = len(grams)
            for line_num in range(len(file_info['lines']['words'])):
                words = file_info['lines']['words'][line_num]
                tokens = words['tokens']
                token_coords = words['coords']
                for ngram in ngrammer(tokens):
                    grams.append(_fill_gram_features(
                        ngram,
                        tokens,
                        token_coords,
                        file_info,
                        file_info['lines']['ymin'][line_num],
                        file_info['lines']['ymax'][line_num],
                        file_info['lines']['labels'][line_num],
                    ))
            _find_closest_grams(grams, start=old_num_grams)
            progress_bar.update(1)

    return pd.DataFrame(data=grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dftrain.pk",
                    help="path to training data")
    ap.add_argument("--save_as", default="data/features.pk",
                    help="save extracted features with this name")
    args = ap.parse_args()
    features = extract_features(args.data)
    features.to_pickle(args.save_as, protocol=3)
    logging.info("\nSaved features as {}".format(args.save_as))


if __name__ == '__main__':
    main()
