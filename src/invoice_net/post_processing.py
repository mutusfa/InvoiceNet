from collections import defaultdict
import dateutil.parser
from typing import Any, Callable, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.optimize

from invoice_net.extract_features import (
    _parses_as_number,
    _parses_as_serial_number,
)
from invoice_net.data_handler import DataHandler
from invoice_net.metrics import convert_to_classes


def __inner_filter_out_mistakes(
    tokens: Iterable[str],
    filter_func: Callable[[str], Any],
    ignore_exceptions: bool = True,
) -> np.ndarray:
    mask = []
    for token in tokens:
        try:
            mask.append(bool(filter_func(token)))
        except Exception:
            if ignore_exceptions:
                mask.append(False)
            else:
                raise
    return np.array(mask)


def _filter_out_mistakes(token_predictions: pd.DataFrame) -> pd.DataFrame:
    """Filter out obvious mistakes, like Foo bar -> date prediction"""
    filters_table: Dict[str, Callable[[str], Any]] = defaultdict(
        lambda: lambda x: x
    )
    filters_table["document_date"] = dateutil.parser.parse
    filters_table["document_id"] = _parses_as_serial_number
    filters_table["amount_total"] = _parses_as_number
    groups = []
    for prediction, group in token_predictions.groupby("pred"):
        groups.append(
            group[
                __inner_filter_out_mistakes(
                    group.word, filters_table[prediction]
                )
            ]
        )
    return pd.concat(groups)


def _get_token_predictions(
    predictions: np.ndarray, raw_text: Sequence[str], file_names: Sequence[str]
) -> pd.DataFrame:
    """Take model predictions and flatten to prediction per token."""
    predicted_classes = convert_to_classes(predictions)
    confidence = predictions.max(axis=-1)
    assert predictions.shape[0] == len(raw_text) == len(file_names), (
        f"Number of samples does not match; ({predictions.shape[0]}, "
        f"{len(raw_text)}, {len(file_names)})"
    )

    tmp = []
    for line_num, line in enumerate(raw_text):
        for word_idx, word in enumerate(line.split()):
            tmp.append(
                {
                    "word": word,
                    "pred": predicted_classes[line_num, word_idx],
                    "confidence": confidence[line_num, word_idx],
                    "file_name": file_names[line_num],
                }
            )
    return pd.DataFrame.from_records(tmp)


def hungarian_prediction(token_predictions):
    predictions = defaultdict(dict)
    for file_name, df in token_predictions.groupby("file_name"):
        hungarian_table = pd.pivot_table(
            df,
            values=["cost"],
            index=["word"],
            columns=["pred"],
            aggfunc=np.min,
            fill_value=1,
        )
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(
            hungarian_table
        )
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            col_name = hungarian_table.columns[col_idx][1]
            predictions[file_name][col_name] = (
                hungarian_table.iloc[row_idx].name,
                1 - hungarian_table.iloc[row_idx, col_idx],
            )
    predictions_df = pd.DataFrame(predictions).transpose()
    return predictions_df.reindex(columns=sorted(predictions_df.columns))


def get_predicted_classes(
    predictions: np.ndarray, data_handler: DataHandler
) -> pd.DataFrame:
    """Get one predicted label per one file"""
    token_predictions = _get_token_predictions(
        predictions, data_handler.data.raw_text, data_handler.data.file_name
    )
    token_predictions["cost"] = 1 - token_predictions["confidence"]
    token_predictions["pred"] = data_handler.to_human_readable_classes(
        token_predictions.pred
    )
    token_predictions.drop(
        token_predictions[token_predictions.pred == "unclassified"].index,
        inplace=True,
    )
    token_predictions = _filter_out_mistakes(token_predictions)
    return hungarian_prediction(token_predictions)