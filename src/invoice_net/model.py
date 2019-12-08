import math
import os
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    concatenate,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
)
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint  # specifically not tf.keras version
from keras_contrib.callbacks.cyclical_learning_rate import CyclicLR
from tensorflow.python.framework import ops
from sklearn.utils.class_weight import compute_class_weight

from invoice_net.metrics import (
    convert_to_classes,
    labeled_confusion_matrix,
    ConfusionMatrixCallback,
    F1ScoreCallback,
    false_positives_false_negatives,
    ValPredictionsCallback,
)


class OneCycleLearning(CyclicLR):
    def __init__(self, num_epochs, num_batches_per_epoch, *args, **kwds):
        total_steps = num_epochs * num_batches_per_epoch
        self.num_trn_steps = int(math.ceil(total_steps / 2.1)) * 2
        super().__init__(*args, step_size=self.num_trn_steps / 2, **kwds)
        self.num_epochs = num_epochs
        self._end_lr = self.base_lr / 100
        self._end_step_size = (total_steps - self.num_trn_steps) * 2
        self._converging = False

    def _start_converge(self, batch, logs=None):
        self._converging = True
        super()._reset(new_base_lr=self._end_lr, new_max_lr=self.base_lr)
        self.clr_iterations = self.step_size
        self.step_size = self._end_step_size

    def on_batch_end(self, batch, logs=None):
        if not self._converging and self.trn_iterations > self.num_trn_steps:
            self._start_converge(batch, logs=logs)
        super().on_batch_end(batch, logs=logs)


def weighted_binary_crossentropy(class_weights,):
    weights = ops.convert_to_tensor(class_weights, dtype="float32")

    def bce(*args, weights=weights, **kwds):
        orig_bce = K.binary_crossentropy(*args, **kwds)
        return K.mean(orig_bce * weights)

    return bce


class InvoiceNetInterface:
    def __init__(self, data_handler, config):
        print("Initializing model...")
        self.data_handler = data_handler
        self.config = config
        print("Defining model graph...")
        self.model = self.create_model(data_handler, config)

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

    def one_cycle_learning_callback(self):
        num_training_iterations_per_epoch = math.ceil(
            len(self.data_handler.labels) / self.config.batch_size
        )
        return OneCycleLearning(
            num_epochs=self.config.num_epochs,
            num_batches_per_epoch=num_training_iterations_per_epoch,
            base_lr=0.001,
            max_lr=0.01,
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
            mode="max",
            period=period,
        )

    def get_class_weights(self, labels=None):
        labels = labels or self.data_handler.labels
        true_labels_by_word = convert_to_classes(
            labels, num_classes=self.data_handler.num_classes
        )
        class_weights = []
        for word_idx in range(true_labels_by_word.shape[1]):
            class_weights.append(
                compute_class_weight(
                    "balanced",
                    np.unique(true_labels_by_word),
                    true_labels_by_word[:, word_idx],
                )
            )
        class_weights = np.array(class_weights)
        print(f"Using class weights:\n{class_weights}")
        return class_weights

    def train(self, callback_period=10, validation_freq=None):
        callback_period = callback_period or self.config.checkpoint_period
        validation_freq = validation_freq or callback_period
        print("\nInitializing training...")
        self._create_needed_dirs()
        validation_data = (
            self.data_handler.validation_features,
            self.data_handler.validation_labels,
        )

        history = self.model.fit(
            self.data_handler.features,
            self.data_handler.labels,
            batch_size=self.config.batch_size,
            verbose=True,
            epochs=self.config.num_epochs,
            callbacks=[
                *self.metrics_callbacks(
                    validation_data=validation_data, period=callback_period
                ),
                self.one_cycle_learning_callback(),
                self.tensorboard_callback(period=callback_period),
                self.modelcheckpoints_callback(period=callback_period),
            ],
            validation_data=(validation_data),
            validation_freq=validation_freq,
            shuffle=self.config.shuffle,
        )
        self.model.save_weights(os.path.join(self.config.model_path))
        return history

    def evaluate(self, print_tables=False, skip_correctly_uncategorized=True):
        predictions = self.model.predict(
            self.data_handler.test_features, batch_size=self.config.batch_size
        )
        predicted_labels = convert_to_classes(
            predictions, num_classes=self.data_handler.num_classes
        )
        test_labels = convert_to_classes(
            self.data_handler.test_labels,
            num_classes=self.data_handler.num_classes,
        )

        (
            false_positives,
            false_negatives,
            other_mistakes,
        ) = false_positives_false_negatives(
            self.data_handler.test_labels, predictions
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
        ).merge(
            pred_df,
            left_index=True,
            right_index=True,
            suffixes=("true", "pred"),
        )
        raw_text_comparison_df["fp"] = false_positives.reshape(
            false_positives.shape[0], -1
        ).any(axis=-1)
        raw_text_comparison_df["fn"] = false_negatives.reshape(
            false_negatives.shape[0], -1
        ).any(axis=-1)
        raw_text_comparison_df["other"] = other_mistakes.reshape(
            other_mistakes.shape[0], -1
        ).any(axis=-1)

        if skip_correctly_uncategorized:
            # all masks here work on whole prediction/text line
            # i.e., whether all ngrams were predicted correctly
            # and whether whole line is full of uninteresting fluff
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
        if print_tables:
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
        print("Compiling model...")
        self.model.compile(
            loss=weighted_binary_crossentropy(self.get_class_weights()),
            optimizer="Adam",
            metrics=["accuracy"],
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
        words_embeddings = Input(
            shape=(
                self.data_handler.max_ngram_size,
                self.data_handler.embed_size,
            ),
            dtype="float32",
            name="words_embeddings",
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
                Flatten()(words_embeddings),
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
        output = Reshape(
            (data_handler.num_classes, data_handler.max_ngram_size)
        )(output)

        return Model(
            inputs=[
                words_embeddings,
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
