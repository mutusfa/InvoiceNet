"""Custom metrics/callbacks for invoice_net."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.callbacks import Callback

from invoice_net.utils import convert_to_classes


def false_positives_false_negatives(y_true, y_pred):
    y_true_classes = convert_to_classes(y_true)
    y_pred_classes = convert_to_classes(y_pred)
    mistakes = y_pred_classes != y_true_classes

    y_true_any = y_true_classes.astype(bool)  # any nonzero class
    y_pred_any = y_pred_classes.astype(bool)  # any nonzero class
    false_positives = mistakes & y_pred_any & ~y_true_any
    false_negatives = mistakes & ~y_pred_any & y_true_any
    other_mistakes = mistakes & ~false_positives & ~false_negatives
    return false_positives, false_negatives, other_mistakes


def labeled_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    human_readable_labels: Dict[int, str] = None,
) -> pd.DataFrame:
    assert (
        y_true.shape == y_pred.shape
    ), "Target and predictions shapes do not match"
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
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

    diagonal = np.diag(matrix)
    recall = diagonal / matrix.apply(lambda x: x.sum(), axis="columns")
    precision = diagonal / matrix.apply(lambda x: x.sum())

    row = pd.Series(data=precision, index=matrix.columns, name="precision")
    matrix = matrix.append(row)
    matrix["recall"] = np.append(recall, None)

    return matrix


class BaseCallback(Callback):
    def __init__(
        self, training_data, validation_data, num_classes, period=1, **kwds
    ):
        super().__init__(**kwds)
        self.training_features = training_data[0]
        self.training_labels = convert_to_classes(training_data[1], num_classes)
        self.validation_features = validation_data[0]
        self.validation_labels = convert_to_classes(
            validation_data[1], num_classes
        )
        self.num_classes = num_classes
        self.period = period


class PredictionsCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs):
        if self.period and (epoch + 1) % self.period == 0:
            predictions = self.model.predict(self.training_features)
            predicted_labels = convert_to_classes(predictions, self.num_classes)
            logs["predicted_labels"] = predicted_labels

            val_predictions = self.model.predict(self.validation_features)
            val_predicted_labels = convert_to_classes(
                val_predictions, self.num_classes
            )
            logs["val_predicted_labels"] = val_predicted_labels


class F1ScoreCallback(BaseCallback):
    def on_epoch_end(self, epoch, logs):
        if self.period and (epoch + 1) % self.period == 0:
            y_true = self.training_labels.reshape(-1)
            y_pred = logs["predicted_labels"].reshape(-1)
            macrof1 = f1_score(y_true, y_pred, average="macro")
            print(f" - macro_f1: {macrof1}")
            logs["macro_f1"] = macrof1

            y_true = self.validation_labels.reshape(-1)
            y_pred = logs["val_predicted_labels"].reshape(-1)
            macrof1 = f1_score(y_true, y_pred, average="macro")
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
                logs["val_predicted_labels"],
                self.human_readable_labels,
            )
            with pd.option_context(
                "display.width",
                0,
                "display.max_columns",
                None,
                "display.max_rows",
                None,
            ):
                print(f"\nConfusion matrix for validation data:\n{matrix}\n")
