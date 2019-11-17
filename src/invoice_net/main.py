import argparse
import pickle

from invoice_net.data_handler import DataHandler
from invoice_net.model import InvoiceNet


ap = argparse.ArgumentParser()

ap.add_argument(
    "--mode",
    type=str,
    help="train|test",
    choices=["train", "test"],
    required=True,
)
ap.add_argument(
    "--data", help="path to training data", default="data/train_api.pk"
)
ap.add_argument(
    "--load_weights",
    default="./checkpoints/InvoiceNet_.157-0.53-0.48.hdf5",
    help="path to load weights",
)
ap.add_argument(
    "--embedding_model",
    default="model.bin",
    help="path to word -> vector embedding model",
)
ap.add_argument(
    "--embedding_type",
    default="word2vec",
    choices=["word2vec", "fasttext"],
    help="type of embedding model",
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
ap.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
ap.add_argument(
    "--size_hidden", type=int, default=128, help="size of hidden layer"
)
ap.add_argument(
    "--batch_size", type=int, default=16000, help="size of mini-batch"
)
ap.add_argument("--shuffle", action="store_true", help="shuffle dataset")

args = ap.parse_args()

with open(args.data, "rb") as pklfile:
    df = pickle.load(pklfile)

data = DataHandler(df, max_len=12)
data.load_embeddings(args.embedding_model, use_model=args.embedding_type)
data.prepare_data()

net = InvoiceNet(data_handler=data, config=args)

if args.mode == "train":
    net.train()
else:
    net.load_weights(args.load_weights)
    predictions = net.evaluate()
    net.f1_score(predictions)
    for i in range(predictions.shape[0]):
        print(
            predictions[i], net.data_handler.train_data["labels"][i], df.iloc[i]
        )
