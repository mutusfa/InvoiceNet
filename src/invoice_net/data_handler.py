import gzip
import pickle

import fasttext
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import pandas as pd


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

    def __init__(self, data=None, max_len=10):
        print("Initializing data handler")
        self.data = data
        self.max_length = max_len
        self.vocab_size = 0
        self.word2idx = {}
        self.embeddings = None
        self.embed_size = 300
        self.PAD = "<pad>"
        self.UNKNOWN = "<unk>"
        self.START = "<start>"
        self.END = "<end>"
        self.label_dict = {0: 0, 1: 1, 2: 2, 8: 3, 14: 4, 18: 5}
        self.num_classes = len(self.label_dict)
        self.train_data = {}

    def prepare_data(self):
        """Prepares data for training"""
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

        self.train_data["words_input"] = padded
        self.train_data["labels"] = self.data.apply(
            lambda x: max(*x.labels, 0), axis="columns"
        ).values
        self.train_data["coordinates"] = self.data.loc[
            :, self.coordinates_features
        ].values
        self.train_data["aux_features"] = self.data.loc[
            :, self.auxillary_features
        ].values.astype(float)

    @property
    def features(self):
        # only keys are copied so this is cheap
        features = self.train_data.copy()
        features.pop("labels")
        return features

    @property
    def idx2word(self):
        value = {v: k for k, v in self.word2idx.items()}
        self.idx2word = value
        return value

    @property
    def labels(self):
        return self.train_data["labels"]

    def load_embeddings(self, model_path, use_model="word2vec"):
        """Loads pre-trained gensim model"""
        print("\nLoading pre-trained embeddings...")

        if use_model == "word2vec":
            model = Word2Vec.load(model_path)
            words = list(model.wv.vocab)
            embed_size = model.layer1_size
            get_vector = lambda word: model.wv[word]
        elif use_model == "fasttext":
            model = fasttext.load_model(model_path)
            words = model.words
            embed_size = model.get_dimension()
            get_vector = lambda word: model[word]
        else:
            raise ValueError(f"Unknown model type {use_model}.")

        embed = []
        word2idx = {self.PAD: 0, self.UNKNOWN: 1, self.START: 2, self.END: 3}

        embed.append(np.zeros(embed_size, dtype=np.float32))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))
        embed.append(np.random.uniform(-0.1, 0.1, embed_size))
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
        """Returns the id of a token"""
        token = token.lower()
        return self.word2idx.get(token, self.word2idx[self.UNKNOWN])

    def load_data(self, path):
        """Loads embeddings and vocab from a zipped pickle file"""
        with gzip.open(path, "rb") as in_file:
            pkl = pickle.load(in_file)
        self.embeddings = pkl["embeddings"]
        self.embed_size = self.embeddings.shape[1]
        self.word2idx = pkl["word2idx"]
        self.vocab_size = len(self.word2idx)
        print("\nSuccessfully loaded data from {}".format(path))
