import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class LabelEncoder:
    def __init__(self):
        self._encoder = {}
        self._decoder = {}

    def add(self, value) -> int:
        try:
            return self._encoder[value]
        except KeyError:
            label = len(self._encoder)
            self._encoder[value] = label
            self._decoder[label] = value
            return label

    def update(self, values: Iterable):
        unique = set(values)
        for klass in unique:
            self.add(klass)
        return self

    def encode(self, values: Sequence) -> np.ndarray:
        values = np.asarray(values)
        encoded = np.zeros(values.shape, dtype=np.int)
        for label, encoding in self._encoder.items():
            encoded[values == label] = encoding
        return encoded

    def decode(self, values: Sequence) -> np.ndarray:
        values = np.asarray(values)
        decoded = np.zeros(values.shape, dtype=object)
        for label, decoding in self._decoder.items():
            decoded[values == label] = decoding
        return decoded

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self._encoder, f)

    def load(self, path: Path) -> None:
        with open(path, "r") as f:
            self._encoder = json.load(f)
            self._decoder = {v: k for k, v in self._encoder.items()}

    def __len__(self):
        return len(self._encoder)

    def __repr__(self):
        return f"{self.__class__.__name__}(): {self._encoder}"
