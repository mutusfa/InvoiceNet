"""Custom metrics/callbacks for invoice_net."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.callbacks import Callback


def convert_to_labels(
    predictions: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Convert one-hot encoded predictions to labels.

    Assumes label 0 is for uncategorized and assigns 0 for any predictions
    that did not meet the threshold.
    """
    predicted_labels = np.argmax(predictions, axis=-1)
    mask = np.max(predictions, axis=-1) >= threshold
    return predicted_labels * mask


def labeled_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred)
    matrix = pd.DataFrame(
        matrix,
        columns=[f"pred_{i}" for i in range(matrix.shape[0])],
        index=[f"true_{i}" for i in range(matrix.shape[0])],
    )
    return matrix


class ValPredictionsCallback(Callback):
    def __init__(self, validation_data, **kwds):
        super().__init__(**kwds)
        self.validation_features = validation_data[0]
        self.validation_labels = validation_data[1]

    def on_epoch_end(self, epoch, logs):
        predictions = self.model.predict(self.validation_features)
        predicted_labels = convert_to_labels(predictions)
        logs["predictions"] = predictions
        logs["predicted_labels"] = predicted_labels


class F1ScoreCallback(Callback):
    def __init__(self, validation_data, **kwds):
        super().__init__(**kwds)
        self.validation_features = validation_data[0]
        self.validation_labels = validation_data[1]

    def on_epoch_end(self, epoch, logs):
        macrof1 = f1_score(
            logs["predicted_labels"], self.validation_labels, average="macro"
        )
        print(f" - val_macro_f1: {macrof1}")
        logs["val_macro_f1"] = macrof1


class ConfusionMatrixCallback(Callback):
    def __init__(self, validation_data, period=1, **kwds):
        super().__init__(**kwds)
        self.validation_features = validation_data[0]
        self.validation_labels = validation_data[1]
        self.period = period

    def on_epoch_end(self, epoch, logs):
        if not (epoch and epoch % self.period == 0):
            return
        matrix = labeled_confusion_matrix(
            self.validation_labels, logs["predicted_labels"]
        )
        print(f"\nConfusion matrix for validation data:\n{matrix}")
