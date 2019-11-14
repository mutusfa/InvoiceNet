import os
from typing import Any

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    concatenate,
    Convolution1D,
    Dense,
    Dropout,
    Embedding,
    Input,
    GRU,
)
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(1337)


def get_recall(predicted_labels, true_labels, target_label):
    mask = true_labels == target_label
    return np.mean(predicted_labels[mask] == true_labels[mask])


def get_precision(predicted_labels, true_labels, target_label):
    mask = true_labels == target_label
    return (
        np.sum(predicted_labels[mask] == true_labels[mask]) / true_labels.size
    )


def get_f1_score(predicted_labels, true_labels, target_label):
    precision = get_precision(predicted_labels, true_labels, target_label)
    recall = get_recall(predicted_labels, true_labels, target_label)
    return 2 * (precision * recall) / (precision + recall)


def get_f1_scores(predictions, true_labels):
    f1_scores = {}
    for target_label in true_labels:
        f1 = get_f1_score(prediction, true_labels, target_label)
        f1_scores[target_label] = f1
    macrof1 = sum(f1_scores.values()) / len(true_labels)
    return f1_scores, macrof1


def f1(y_true, y_pred, num_classes=6):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())

    macrof1 = 0
    for target_label in range(num_classes):
        y_target = K.cast(K.equal(y_true, target_label), K.floatx())
        y_target_pred = K.cast(K.equal(y_pred_labels, target_label), K.floatx())
        prec = precision(y_target, y_target_pred)
        rec = recall(y_target, y_target_pred)
        f1_score = 2 * ((prec * rec) / (prec + rec + K.epsilon()))
        macrof1 += f1_score / num_classes
    return macrof1


class InvoiceNetInterface:
    def __init__(self, data_handler, config):
        print("Initializing model...")
        self.data_handler = data_handler
        self.config = config
        print("Defining model graph...")
        self.model = self.create_model(data_handler, config)
        print("Compiling model...")
        self.compile_model()

    def _create_needed_dirs(self):
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)

    @property
    def tensorboard_callback(self):
        return TensorBoard(
            log_dir=self.config.log_dir, histogram_freq=1, write_graph=True
        )

    @property
    def modelcheckpoints_callback(self):
        filename_format = ".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"
        return ModelCheckpoint(
            os.path.join(
                self.config.checkpoint_dir,
                self.__class__.__name__ + filename_format,
            ),
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
        )

    def get_class_weights(self, true_labels):
        class_weights = compute_class_weight(
            "balanced",
            np.unique(true_labels),
            self.data_handler.train_data["labels"],
        )
        return dict(enumerate(class_weights))

    def prepare_data(self, features, labels):
        """A hook for subclasses to modify data for their own needs"""
        return features, labels

    def train(self):
        print("\nInitializing training...")
        self._create_needed_dirs()
        features, labels = self.prepare_data(
            self.data_handler.features, self.data_handler.labels
        )

        self.model.fit(
            features,
            labels,
            batch_size=self.config.batch_size,
            verbose=True,
            epochs=self.config.num_epochs,
            callbacks=[self.tensorboard_callback],
            validation_split=0.125,
            shuffle=self.config.shuffle,
            class_weight=self.get_class_weights(self.data_handler.labels),
        )

        self.model.save_weights(
            os.path.join(
                self.config.model_path, self.__class__.__name__ + ".model"
            )
        )

    def create_model(self, data_handler, config) -> Any:
        raise NotImplementedError(
            "Model should be defined in create_model method"
        )

    def compile_model(self):
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy", f1],
        )


class InvoiceNet(InvoiceNetInterface):
    def create_model(self, data_handler, config):
        coordinates = Input(
            shape=(data_handler.train_data["coordinates"].shape[1],),
            dtype="float32",
            name="coordinates",
        )
        aux_features = Input(
            shape=(data_handler.train_data["aux_features"].shape[1],),
            dtype="float32",
            name="aux_features",
        )
        words_input = Input(
            shape=(data_handler.max_length,), dtype="int32", name="words_input"
        )
        words = Embedding(
            data_handler.vocab_size,
            data_handler.embeddings.shape[1],
            weights=[data_handler.embeddings],
            trainable=False,
        )(words_input)
        output = GRU(
            config.num_hidden,
            dropout=0.5,
            recurrent_dropout=0.5,
            go_backwards=True,
        )(words)
        output = concatenate([output, coordinates, aux_features])
        output = Dense(config.num_hidden, activation="relu")(output)
        output = Dense(config.num_hidden, activation="relu")(output)
        output = Dropout(0.5)(output)
        output = Dense(config.num_hidden, activation="relu")(output)
        output = Dense(data_handler.num_classes, activation="softmax")(output)

        return Model(
            inputs=[words_input, coordinates, aux_features], outputs=[output]
        )
        # self.model.summary()

    def load_weights(self, path):
        """Loads weights from the given model file"""
        self.model.load_weights(path)
        print("\nSuccessfully loaded weights from {}".format(path))

    def evaluate(self):
        predictions = self.model.predict(
            [
                self.data_handler.train_data["inputs"],
                self.data_handler.train_data["coordinates"],
            ],
            verbose=True,
        )
        predictions = predictions.argmax(axis=-1)
        acc = np.sum(
            predictions
            == (self.data_handler.train_data["labels"])
            / float(len(self.data_handler.train_data["labels"]))
        )
        print("\nTest Accuracy: {}".format(acc))
        return predictions

    def f1_score(self, predictions):
        true_labels = self.data_handler.train_data["labels"]
        f1_scores, macrof1 = get_f1_scores(predictions, true_labels)
        print("\nMacro-Averaged F1: %.4f\n" % macrof1)
        return f1_scores, macrof1


class InvoiceNetCloudScan(InvoiceNetInterface):
    def create_model(self, data_handler, config):
        coordinates = Input(
            shape=(data_handler.train_data["coordinates"].shape[1],),
            dtype="float32",
            name="coordinates",
        )
        aux_features = Input(
            shape=(data_handler.train_data["aux_features"].shape[1],),
            dtype="float32",
            name="aux_features",
        )
        words_input = Input(
            shape=(data_handler.max_length,), dtype="int32", name="words_input"
        )
        words = Embedding(
            data_handler.vocab_size,
            data_handler.embeddings.shape[1],
            weights=[data_handler.embeddings],
            trainable=False,
        )(words_input)
        words = GRU(
            config.num_hidden, dropout=0, recurrent_dropout=0, go_backwards=True
        )(words)
        output = concatenate([words, coordinates, aux_features])
        output = Dense(
            data_handler.num_classes, activation="softmax", name="output"
        )(output)

        return Model(
            inputs=[words_input, coordinates, aux_features], outputs=[output]
        )

    def load_weights(self, path):
        """Loads weights from the given model file"""
        self.model.load_weights(path)
        print("\nSuccessfully loaded weights from {}".format(path))

    def evaluate(self, data):
        x_test, y_test = self.prepare_data(data)
        predictions = self.model.predict([x_test], verbose=True)
        predictions = predictions.argmax(axis=-1)
        acc = np.sum(predictions == y_test) / float(len(y_test))
        print("\nTest Accuracy: {}".format(acc))
        return predictions

    def f1_score(self, predictions, ground_truth):
        f1_scores, macrof1 = get_f1_scores(predictions, ground_truth)
        print("\nMacro-Averaged F1: %.4f\n" % macrof1)
        return macrof1
