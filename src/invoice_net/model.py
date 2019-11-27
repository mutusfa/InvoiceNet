import os
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras.callbacks import ModelCheckpoint  # specifically not tf.keras version
from sklearn.utils.class_weight import compute_class_weight

from invoice_net.metrics import (
    convert_to_classes,
    labeled_confusion_matrix,
    ConfusionMatrixCallback,
    F1ScoreCallback,
    ValPredictionsCallback,
)


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

    def tensorboard_callback(self, period=5):
        return TensorBoard(
            log_dir=self.config.log_dir,
            histogram_freq=period,
            write_graph=True,
            write_images=True,
        )

    def metrics_callbacks(self, validation_data, period=5):
        return [
            ValPredictionsCallback(
                validation_data=validation_data,
                num_classes=self.data_handler.num_classes,
                period=period,
            ),
            F1ScoreCallback(
                validation_data=validation_data,
                num_classes=self.data_handler.num_classes,
                period=period,
            ),
            ConfusionMatrixCallback(
                validation_data=validation_data,
                num_classes=self.data_handler.num_classes,
                human_readable_labels=self.data_handler.human_readable_labels,
                period=period,
            ),
        ]

    def modelcheckpoints_callback(self, period=5):
        filename_format = ".{epoch:02d}-{val_loss:.2f}.hdf5"
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
            period=period,
        )

    def get_class_weights(self):
        true_labels = convert_to_classes(
            self.data_handler.labels, num_classes=self.data_handler.num_classes
        ).reshape(-1)
        class_weights = compute_class_weight(
            "balanced", np.unique(true_labels), true_labels
        )
        return dict(enumerate(class_weights))

    def train(self, callback_period=10):
        print("\nInitializing training...")
        self._create_needed_dirs()
        validation_data = (
            self.data_handler.validation_features,
            self.data_handler.validation_labels,
        )

        self.model.fit(
            self.data_handler.features,
            self.data_handler.labels,
            batch_size=self.config.batch_size,
            verbose=True,
            epochs=self.config.num_epochs,
            callbacks=[
                *self.metrics_callbacks(
                    validation_data=validation_data, period=callback_period
                ),
                self.tensorboard_callback(period=callback_period),
                self.modelcheckpoints_callback(period=callback_period),
                EarlyStopping(patience=50, monitor="loss"),
            ],
            validation_data=(validation_data),
            shuffle=self.config.shuffle,
            class_weight=self.get_class_weights(),
        )
        self.model.save_weights(os.path.join(self.config.model_path))

    def evaluate(self):
        predictions = self.model.predict(self.data_handler.test_features)
        predicted_labels = convert_to_classes(
            predictions, num_classe=self.data_handler.num_classes
        )
        raw_text_comparison_df = pd.DataFrame(
            {
                "raw_text": self.data_handler.test_data["raw_text"],
                "true": [
                    self.data_handler.human_readable_labels[i]
                    for i in self.data_handler.test_labels
                ],
                "pred": [
                    self.data_handler.human_readable_labels[i]
                    for i in predicted_labels
                ],
            }
        )
        matrix = labeled_confusion_matrix(
            self.data_handler.test_labels,
            predicted_labels,
            self.data_handler.human_readable_labels,
        )
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            0,
        ):
            print(raw_text_comparison_df)
            print(matrix)
        return raw_text_comparison_df, matrix

    def create_model(self, data_handler, config) -> Any:
        raise NotImplementedError(
            "Model should be defined in create_model method"
        )

    def compile_model(self):
        self.model.compile(
            loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )

    def load_weights(self, path):
        """Load weights from the given model file."""
        self.model.load_weights(path)
        print("\nSuccessfully loaded weights from {}".format(path))


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
        sentences_embeddings = Input(
            shape=(self.data_handler.embed_size),
            dtype="float32",
            name="sentences_embeddings",
        )
        embeddings = Flatten()(sentences_embeddings)
        output = concatenate([embeddings, coordinates, aux_features])
        output = Dense(config.size_hidden, activation="relu")(output)
        output = Dense(config.size_hidden, activation="relu")(output)
        output = Dropout(0.5)(output)
        output = Dense(config.size_hidden, activation="relu")(output)
        output = Dense(
            data_handler.num_classes * data_handler.max_ngram_size,
            activation="sigmoid",
        )(output)

        return Model(
            inputs=[sentences_embeddings, coordinates, aux_features],
            outputs=[output],
        )


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
        sentences_embeddings = Input(
            shape=(self.data_handler.embed_size),
            dtype="float32",
            name="sentences_embeddings",
        )
        embeddings = Flatten()(sentences_embeddings)
        output = concatenate([embeddings, coordinates, aux_features])
        output = Dense(
            data_handler.num_classes, activation="softmax", name="output"
        )(output)

        return Model(
            inputs=[sentences_embeddings, coordinates, aux_features],
            outputs=[output],
        )
