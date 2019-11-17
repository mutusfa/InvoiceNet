import argparse

import pandas as pd

from invoice_net.extract_features import extract_features

parent_parser = argparse.ArgumentParser()

parent_parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "test"],
    default="test",
    help="train|test",
)
parent_parser.add_argument(
    "--data", default="data/dftrain.pk", help="path to training data"
)
parent_parser.add_argument("--raw_data", help="path to unprocessed data")
parent_parser.add_argument("--load_weights", help="path to load weights")
parent_parser.add_argument(
    "--embedding_model",
    default="data/model.bin",
    help="path to word -> vector embedding model",
)
parent_parser.add_argument(
    "--checkpoint_dir",
    default="./checkpoints",
    help="path to directory where checkpoints should be stored",
)
parent_parser.add_argument(
    "--log_dir",
    default="./logs",
    help="path to directory where tensorboard logs should be stored",
)
parent_parser.add_argument(
    "--size_hidden", type=int, default=16, help="size of hidden layer"
)
parent_parser.add_argument(
    "--num_epochs", type=int, default=20, help="number of epochs"
)
parent_parser.add_argument(
    "--batch_size", type=int, default=16000, help="size of mini-batch"
)
parent_parser.add_argument(
    "--shuffle", action="store_true", help="shuffle dataset"
)


def get_data_for_nn(config):
    if config.raw_data:
        raw = pd.read_pickle(config.raw_data)
        return extract_features(raw)
    else:
        return pd.read_pickle(config.data)
