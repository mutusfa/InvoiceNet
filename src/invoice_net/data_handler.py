"""Handles data for nn models."""

import fasttext
import numpy as np
import pandas as pd


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
        "parses_as_date",
        "parses_as_number",
    ]
    debugging_features = ["raw_text", "processed_text"]
    human_readable_labels = {
        0: "unclassified",
        1: "invoice_date",
        2: "document_id",
        3: "amount_total",
    }

    def __init__(self, data=None, validation_split=0.125, test_split=0.1):
        print("Initializing data handler")
        self.data = data
        self.word2idx = {}
        self.embed_size = None
        self.fasttext = None
        self.label_dict = {0: 0, 1: 1, 2: 2, 8: 3, 14: 4, 18: 5}
        self.num_classes = len(self.label_dict)
        self.validation_split = validation_split
        self.test_split = test_split
        self.train_data = {}
        self.validation_data = {}
        self.test_data = {}

    def prepare_data(self):
        """Prepare data for training."""
        print("Preparing data")

        sentences_embeddings = np.array(
            [
                self.fasttext.get_sentence_vector(s)
                for s in self.data.processed_text
            ]
        )

        closest_ngrams = pd.DataFrame(
            self.data.closest_ngrams.values.tolist(),
            columns=["left", "top", "right", "bottom"],
            index=self.data.index
        )

        def get_df_by_indices(column):
            idx = closest_ngrams[closest_ngrams[column] != -1][column]
            df = closest_ngrams.iloc[idx, :]
            df.set_index(idx.index, inplace=True)
            df.columns = [f"{column}_{c}" for c in df.columns]
            return df

        left_df = get_df_by_indices("left")
        top_df = get_df_by_indices("top")
        right_df = get_df_by_indices("right")
        bottom_df = get_df_by_indices("bottom")

        df = pd.concat(
            [self.data, left_df, top_df, right_df, bottom_df], axis=1
        )

        features = {}
        features["sentences_embeddings"] = sentences_embeddings
        features["coordinates"] = self.data.loc[
            :, self.coordinates_features
        ].values
        features["aux_features"] = self.data.loc[
            :, self.auxillary_features
        ].values.astype(float)
        for key in self.debugging_features:
            features[key] = self.data.loc[:, key].values
        self.train_data, self.validation_data, self.test_data = split_data(
            features,
            1 - self.validation_split - self.test_split,
            self.validation_split,
            self.test_split,
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
                self.test_data["labels"],
            ) = split_data(
                labels,
                1 - self.validation_split - self.test_split,
                self.validation_split,
                self.test_split,
            )

    def _features(self, data_dict):
        """Return features for nn use."""
        # only keys are copied so this is cheap
        features = data_dict.copy()
        for key in ("labels", *self.debugging_features):
            features.pop(key, None)
        return features

    @property
    def features(self):
        return self._features(self.train_data)

    @property
    def labels(self):
        return self.train_data["labels"]

    @property
    def validation_features(self):
        return self._features(self.validation_data)

    @property
    def validation_labels(self):
        return self.validation_data["labels"]

    @property
    def test_features(self):
        return self._features(self.test_data)

    @property
    def test_labels(self):
        return self.test_data["labels"]

    def load_embeddings(self, model_path):
        """Load pre-trained fasttext model."""
        print("\nLoading pre-trained embeddings...")

        self.fasttext = fasttext.load_model(model_path)
        self.embed_size = self.fasttext.get_dimension()

        print("\nSuccessfully loaded pre-trained embeddings!")
