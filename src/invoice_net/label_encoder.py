from __future__ import annotations
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

    def fit(self, values: Iterable) -> LabelEncoder:
        unique = set(values)
        for klass in unique:
            self.add(klass)
        return self

    def encode(self, values: Sequence):
        return np.vectorize(self._encoder.__getitem__)(values)

    def decode(self, values: Sequence[int]) -> np.ndarray:
        return np.vectorize(self._decoder.__getitem__)(values)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self._encoder, f)

    def load(self, path: Path) -> None:
        with open(path, "r") as f:
            self._encoder = json.load(f)
            self._decoder = {v: k for k, v in self._encoder.items()}
