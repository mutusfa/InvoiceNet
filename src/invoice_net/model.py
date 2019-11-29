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
        filename_format = ".{epoch:02d}-{val_loss:.2f}-{val_macro_f1}.hdf5"
        return ModelCheckpoint(
            os.path.join(
                self.config.checkpoint_dir,
                self.__class__.__name__ + filename_format,
            ),
            monitor="val_macro_f1",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
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

    def evaluate(self, skip_correctly_uncategorized=True):
        predictions = self.model.predict(self.data_handler.test_features)
        predicted_labels = convert_to_classes(
            predictions, num_classes=self.data_handler.num_classes
        )
        test_labels = convert_to_classes(
            self.data_handler.test_labels,
            num_classes=self.data_handler.num_classes,
        )

        true_df = pd.DataFrame(
            self.data_handler.to_human_readable_classes(test_labels)
        )
        pred_df = pd.DataFrame(
            self.data_handler.to_human_readable_classes(predicted_labels)
        )
        raw_text_comparison_df = pd.DataFrame(
            {
                "raw_text": self.data_handler.test_data["raw_text"],
                "processed_text": self.data_handler.test_data["processed_text"],
            }
        )
        raw_text_comparison_df = raw_text_comparison_df.merge(
            true_df, left_index=True, right_index=True
        ).merge(pred_df, left_index=True, right_index=True)

        if skip_correctly_uncategorized:
            correct_predictions_mask = (test_labels == predicted_labels).all(
                axis=-1
            )
            only_uncategorized_mask = ~test_labels.any(axis=-1)
            correctly_uncategorized_mask = np.logical_and(
                correct_predictions_mask, only_uncategorized_mask
            )
            raw_text_comparison_df = raw_text_comparison_df[
                ~correctly_uncategorized_mask
            ]

        matrix = labeled_confusion_matrix(
            test_labels,
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
        left_sentences_embeddings = Input(
            shape=(self.data_handler.embed_size),
            dtype="float32",
            name="left_sentences_embeddings",
        )
        top_sentences_embeddings = Input(
            shape=(self.data_handler.embed_size),
            dtype="float32",
            name="top_sentences_embeddings",
        )
        right_sentences_embeddings = Input(
            shape=(self.data_handler.embed_size),
            dtype="float32",
            name="right_sentences_embeddings",
        )
        bottom_sentences_embeddings = Input(
            shape=(self.data_handler.embed_size),
            dtype="float32",
            name="bottom_sentences_embeddings",
        )

        output = concatenate(
            [
                Flatten()(sentences_embeddings),
                Flatten()(left_sentences_embeddings),
                Flatten()(top_sentences_embeddings),
                Flatten()(right_sentences_embeddings),
                Flatten()(bottom_sentences_embeddings),
                coordinates,
                aux_features,
            ]
        )
        output = Dense(config.size_hidden, activation="relu")(output)
        output = Dense(config.size_hidden, activation="relu")(output)
        output = Dropout(0.5)(output)
        output = Dense(config.size_hidden, activation="relu")(output)
        output = Dense(
            data_handler.num_classes * data_handler.max_ngram_size,
            activation="sigmoid",
        )(output)

        return Model(
            inputs=[
                sentences_embeddings,
                left_sentences_embeddings,
                top_sentences_embeddings,
                right_sentences_embeddings,
                bottom_sentences_embeddings,
                coordinates,
                aux_features,
            ],
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
