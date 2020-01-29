import numpy as np


def convert_to_classes(
    predictions: np.ndarray, num_classes=None, threshold: float = 0.5,
) -> np.ndarray:
    """Convert one-hot encoded predictions to labels.

    Assumes label 0 is for uncategorized and assigns 0 for any predictions
    that did not meet the threshold.
    """
    predicted_labels = np.argmax(predictions, axis=-1)
    mask = np.max(predictions, axis=-1) >= threshold
    return predicted_labels * mask
