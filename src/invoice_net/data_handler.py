"""Handles data for nn models."""

import gzip
import pickle

import fasttext
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
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
        self.vocab_size = 0
        self.word2idx = {}
        self.embeddings = None
        self.embed_size = None
        self.PAD = "<pad>"
        self.UNKNOWN = "<unk>"
        self.label_dict = {0: 0, 1: 1, 2: 2, 8: 3, 14: 4, 18: 5}
        self.num_classes = len(self.label_dict)
        self.validation_split = validation_split
        self.train_data = {}
        self.validation_data = {}

    def prepare_data(self):
        """Prepare data for training."""
        print("Preparing data")

        sequences = [text_to_word_sequence(t) for t in self.data.processed_text]
        encoded = []
        for sequence in sequences:
            sequence_input = []
            for word in sequence:
                sequence_input.append(
                    self.word2idx.get(word, self.word2idx[self.UNKNOWN])
                )
            encoded.append(sequence_input)
        encoded = np.array(encoded)
        padded = pad_sequences(
            encoded, maxlen=self.max_length, padding="post", truncating="post"
        )

        features = {}
        features["words_input"] = padded
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

    @lazy_property
    def idx2word(self):
        value = {v: k for k, v in self.word2idx.items()}
        return value

    def load_embeddings(self, model_path, use_model="word2vec"):
        """Load pre-trained gensim model."""
        print("\nLoading pre-trained embeddings...")

        if use_model == "word2vec":
            model = Word2Vec.load(model_path)
            words = list(model.wv.vocab)
            embed_size = model.layer1_size
            get_vector = lambda word: model.wv[word]  # noqa
        elif use_model == "fasttext":
            model = fasttext.load_model(model_path)
            words = model.words
            embed_size = model.get_dimension()
            get_vector = lambda word: model[word]  # noqa
        else:
            raise ValueError(f"Unknown model type {use_model}.")

        embed = []
        word2idx = {self.PAD: 0, self.UNKNOWN: 1}

        embed.append(np.zeros(embed_size, dtype=np.float32))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))

        for word in words:
            vector = get_vector(word)
            embed.append(vector)
            word2idx[word] = len(word2idx)

        self.vocab_size = len(word2idx)
        self.word2idx = word2idx
        self.embeddings = np.array(embed, dtype=np.float32)

        print("\nSuccessfully loaded pre-trained embeddings!")

    def get_word_id(self, token):
        """Return the id of a token."""
        token = token.lower()
        return self.word2idx.get(token, self.word2idx[self.UNKNOWN])

    def load_data(self, path):
        """Load embeddings and vocab from a zipped pickle file."""
        with gzip.open(path, "rb") as in_file:
            pkl = pickle.load(in_file)
        self.embeddings = pkl["embeddings"]
        self.embed_size = self.embeddings.shape[1]
        self.word2idx = pkl["word2idx"]
        self.vocab_size = len(self.word2idx)
        print("\nSuccessfully loaded data from {}".format(path))
