import argparse

import pandas as pd

from invoice_net.data_handler import DataHandler
from invoice_net.model import InvoiceNetCloudScan
from invoice_net.extract_features import extract_features
from invoice_net._config import get_data_for_nn, parent_parser


def parse_args():
    return parent_parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    data = DataHandler(get_data_for_nn(config), max_len=12)
    data.load_embeddings(config.embedding_model)
    data.prepare_data()
    net = InvoiceNetCloudScan(data_handler=data, config=config)

    if config.mode == "train":
        net.train()
    else:
        net.load_weights(config.load_weights)
        predictions = net.evaluate()
        net.f1_score(predictions, features.label.values)
