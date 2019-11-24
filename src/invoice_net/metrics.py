"""Custom metrics/callbacks for invoice_net."""
import functools
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.callbacks import Callback


def convert_to_labels(
    predictions: np.ndarray, threshold: float = 0.7
) -> np.ndarray:
    """Convert one-hot encoded predictions to labels.

    Assumes label 0 is for uncategorized and assigns 0 for any predictions
    that did not meet the threshold.
    """
    predicted_labels = np.argmax(predictions, axis=-1)
    mask = np.max(predictions, axis=-1) >= threshold
    return predicted_labels * mask


def labeled_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    human_readable_labels: Dict[int, str] = None,
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred)
    human_readable_labels = human_readable_labels or {
        i: str(i) for i in range(matrix.shape[0])
    }

    matrix = pd.DataFrame(
        matrix,
        columns=[
            f"pred_{human_readable_labels[i]}" for i in range(matrix.shape[0])
        ],
        index=[
            f"true_{human_readable_labels[i]}" for i in range(matrix.shape[0])
        ],
    )
    return matrix


class BaseCallback(Callback):
    def __init__(self, validation_data, period=1, **kwds):
        super().__init__(**kwds)
        self.validation_features = validation_data[0]
        self.validation_labels = validation_data[1]
        self.period = period


class ValPredictionsCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs):
        if self.period and (epoch + 1) % self.period == 0:
            predictions = self.model.predict(self.validation_features)
            predicted_labels = convert_to_labels(predictions)
            logs["predictions"] = predictions
            logs["predicted_labels"] = predicted_labels


class F1ScoreCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs):
        if self.period and (epoch + 1) % self.period == 0:
            macrof1 = f1_score(
                logs["predicted_labels"],
                self.validation_labels,
                average="macro",
            )
            print(f" - val_macro_f1: {macrof1}")
            logs["val_macro_f1"] = macrof1


class ConfusionMatrixCallback(BaseCallback):
    def __init__(self, *args, human_readable_labels=None, **kwds):
        super().__init__(*args, **kwds)
        self.human_readable_labels = human_readable_labels

    def on_epoch_end(self, epoch, logs):
        if self.period and (epoch + 1) % self.period == 0:
            matrix = labeled_confusion_matrix(
                self.validation_labels,
                logs["predicted_labels"],
                self.human_readable_labels,
            )
            print(f"\nConfusion matrix for validation data:\n{matrix}")
