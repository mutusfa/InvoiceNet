"""Handles data for nn models."""

import fasttext
import numpy as np


def lazy_property(fn):
    """Make a property lazy-evaluated."""
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def split_data(data, train_frac=1, val_frac=0, test_frac=0):
    def inner(array):
        return np.split(
            array,
            [
                int(train_frac * len(array)),
                int((train_frac + val_frac) * len(array)),
            ],
        )

    assert train_frac + val_frac + test_frac == 1, (
        f"Fractions should sum up to 1; got train_frac={train_frac}, "
        f"val_frac={val_frac} and test_frac={test_frac}"
    )
    # Keras allows passing multiple inputs/outputs as dict
    if isinstance(data, dict):
        train_dict = dict.fromkeys(data)
        val_dict = dict.fromkeys(data)
        test_dict = dict.fromkeys(data)
        for key, value in data.items():
            train_dict[key], val_dict[key], test_dict[key] = inner(value)
        return train_dict, val_dict, test_dict
    elif isinstance(data, list):
        raise ValueError(
            f"Ambigous data type {type(data)}.\n"
            "Use np.array for single input and dict for multiple inputs instead"
        )
    else:
        return inner(data)


class DataHandler:
    coordinates_features = [
        "bottom_margin",
        "top_margin",
        "left_margin",
        "right_margin",
    ]
    auxillary_features = [
        "length",
        "line_size",
        "position_on_line",
        "has_digits",
        "parses_as_amount",
        "parses_as_date",
        "parses_as_number",
    ]

    def __init__(self, data=None, max_len=10, validation_split=0.125):
        print("Initializing data handler")
        self.data = data
        self.max_length = max_len
        self.word2idx = {}
        self.embed_size = None
        self.fasttext = None
        self.label_dict = {0: 0, 1: 1, 2: 2, 8: 3, 14: 4, 18: 5}
        self.num_classes = len(self.label_dict)
        self.validation_split = validation_split
        self.train_data = {}
        self.validation_data = {}

    def prepare_data(self):
        """Prepare data for training."""
        print("Preparing data")

        sentences_embeddings = np.array(
            [
                self.fasttext.get_sentence_vector(s)
                for s in self.data.processed_text
            ]
        )

        features = {}
        features["sentences_embeddings"] = sentences_embeddings
        features["coordinates"] = self.data.loc[
            :, self.coordinates_features
        ].values
        features["aux_features"] = self.data.loc[
            :, self.auxillary_features
        ].values.astype(float)
        self.train_data, self.validation_data, _ = split_data(
            features, 1 - self.validation_split, self.validation_split
        )

        try:
            labels = self.data.apply(
                lambda x: max(*x.labels, 0), axis="columns"
            ).values
        except AttributeError:
            print("Found no labels; continuing to work without labels")
            self.train_data["labels"] = None
            self.validation_data["labels"] = None
        else:
            (
                self.train_data["labels"],
                self.validation_data["labels"],
                _,
            ) = split_data(
                labels, 1 - self.validation_split, self.validation_split
            )

    @property
    def features(self):
        # only keys are copied so this is cheap
        features = self.train_data.copy()
        features.pop("labels", None)
        return features

    @property
    def labels(self):
        return self.train_data["labels"]

    @property
    def validation_features(self):
        # only keys are copied so this is cheap
        features = self.validation_data.copy()
        features.pop("labels", None)
        return features

    @property
    def validation_labels(self):
        return self.validation_data["labels"]

    def load_embeddings(self, model_path):
        """Load pre-trained gensim model."""
        print("\nLoading pre-trained embeddings...")

        self.fasttext = fasttext.load_model(model_path)
        self.embed_size = self.fasttext.get_dimension()

        print("\nSuccessfully loaded pre-trained embeddings!")
