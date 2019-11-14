import argparse

import pandas as pd

from invoice_net.data_handler import DataHandler
from invoice_net.model import InvoiceNetCloudScan
from invoice_net.extract_features import extract_features


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="train|test",
    )
    ap.add_argument(
        "--data", default="data/dftrain.pk", help="path to training data"
    )
    ap.add_argument("--raw_data", help="path to unprocessed data")
    ap.add_argument(
        "--model_path",
        default="./model",
        help="path to directory where trained model should be stored",
    )
    ap.add_argument(
        "--load_weights",
        default="./model/InvoiceNetCloudScan.model",
        help="path to load weights",
    )
    ap.add_argument(
        "--embedding_model",
        default="model.bin",
        help="path to word -> vector embedding model"
    )
    ap.add_argument(
        "--embedding_type",
        default="word2vec",
        choices=["word2vec", "fasttext"],
        help="type of embedding model"
    )
    ap.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="path to directory where checkpoints should be stored",
    )
    ap.add_argument(
        "--log_dir",
        default="./logs",
        help="path to directory where tensorboard logs should be stored",
    )
    ap.add_argument(
        "--num_hidden", type=int, default=16, help="size of hidden layer"
    )
    ap.add_argument(
        "--num_epochs", type=int, default=20, help="number of epochs"
    )
    ap.add_argument(
        "--batch_size", type=int, default=16000, help="size of mini-batch"
    )
    ap.add_argument(
        "--num_layers", type=int, default=1, help="number of layers"
    )
    ap.add_argument(
        "--num_input", type=int, default=17, help="size of input layer"
    )
    ap.add_argument(
        "--num_output", type=int, default=4, help="size of output layer"
    )
    ap.add_argument("--shuffle", action="store_true", help="shuffle dataset")
    ap.add_argument(
        "--oversample",
        type=int,
        default=0,
        help="oversample minority classes to prevent class imbalance",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.raw_data:
        features = pd.read_pickle(args.raw_data)
        features = extract_features(features)
    else:
        features = pd.read_pickle(args.data)

    data = DataHandler(features, max_len=12)
    data.load_embeddings(args.embedding_model, use_model=args.embedding_type)
    data.prepare_data()
    net = InvoiceNetCloudScan(data_handler=data, config=args)

    if args.mode == "train":
        net.train()
    else:
        net.load_weights(args.load_weights)
        predictions = net.evaluate()
        net.f1_score(predictions, features.label.values)
        # for i in range(predictions.shape[0]):
        #     print(predictions[i], features.label.values[i], features.iloc[i])
