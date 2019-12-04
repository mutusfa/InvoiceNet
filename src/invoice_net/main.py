from invoice_net.data_handler import DataHandler
from invoice_net.model import InvoiceNet
from invoice_net._config import parent_parser, get_data_for_nn


parent_parser.set_defaults(
    load_weights="./model/InvoiceNet.hdf5", model_path="./model/InvoiceNet.hdf5"
)


def parse_args():
    return parent_parser.parse_args()


def main(config=None):
    config = config or parse_args()
    if config.mode == "train":
        data_handler = DataHandler(get_data_for_nn(config))
    else:
        data_handler = DataHandler(
            get_data_for_nn(config), validation_split=0, test_split=1
        )
    data_handler.load_embeddings(config.embedding_model)
    data_handler.prepare_data()
    net = InvoiceNet(data_handler=data_handler, config=config)

    if config.mode == "train":
        net.compile_model()
        net.train()
    else:
        net.load_weights(config.load_weights)
        net.evaluate()
    return net


if __name__ == "__main__":
    main()
