import os
import pickle

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    concatenate,
    Convolution1D,
    Dense,
    Dropout,
    Embedding,
    Input,
    GlobalMaxPooling1D
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
    return np.sum(predicted_labels[mask] == true_labels[mask]) / true_labels.size


def get_f1_score(predicted_labels, true_labels, target_label):
    precision = get_precision(predicted_labels, true_labels, target_label)
    recall = get_recall(predicted_labels, true_labels, target_label)
    return 2 * (precision * recall) / (precision + recall)


def get_f1_scores(predictions, true_labels):
    f1_scores = {}
    for target_label in true_labels:
        f1 = get_precision(prediction, self.data_handler['labels'], target_label)
        f1_scores[target_label] = f1
    macrof1 = sum(f1_scores.values()) / len(true_labels)
    return f1_scores, macrof1


class InvoiceNetInterface:
    def __init__(self, data_handler, config):
        raise NotImplementedError("Model should be defined in __init__ method")

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
            log_dir=self.config.log_dir, histogram_freq=1, write_graph=True)

    @property
    def modelcheckpoints_callback(
            self,
            filename_format=".{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5",
        ):
        return ModelCheckpoint(
            os.path.join(
                self.config.checkpoint_dir,
                self.__class__.__name__ + filename_format,
            ),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        )

    def get_class_weights(self, true_labels):
        class_weights = compute_class_weight(
                'balanced',
                np.unique(true_labels),
                self.data_handler.train_data['labels'])
        d_class_weights = dict(enumerate(class_weights))

    def prepare_data(self, features, labels):
        """A hook for subclasses to modify data for their own needs"""
        return features, labels

    def train(self):
        print("\nInitializing training...")
        self._create_needed_dirs()
        features, labels = self.prepare_data(
            self.data_handler.features,
            self.data_handler.labels
        )

        self.model.fit(
            features,
            labels,
            batch_size=self.config.batch_size,
            verbose=True,
            epochs=self.config.num_epochs,
            callbacks=[self.tensorboard_callback, self.modelcheckpoints_callback],
            validation_split=0.125,
            shuffle=self.config.shuffle,
            class_weight=self.get_class_weights(self.data_handler.labels)
        )

        self.model.save_weights(os.path.join(
            self.config.model_path, self.__class__.__name__ + ".model"))


class InvoiceNet(InvoiceNetInterface):

    def __init__(self, data_handler, config):
        coordinates = Input(shape=(data_handler.train_data['coordinates'].shape[1],),
                            dtype='float32', name='coordinates')
        words_input = Input(shape=(data_handler.max_length,),
                            dtype='int32', name='words_input')
        aux_features = Input(shape=(data_handler.train_data['aux_features'].shape[1],),
                dtype='float32', name='aux_features')
        words = Embedding(data_handler.embeddings.shape[0], data_handler.embeddings.shape[1],
                          weights=[data_handler.embeddings],
                          trainable=False)(words_input)

        output = Dropout(0.5)(words)
        output = layers.GRU(config.num_hidden, dropout=0.5, recurrent_dropout=0.5, go_backwards=True)(output)
        output = concatenate([output, coordinates, aux_features])
        output = Dense(config.num_hidden, activation='relu')(output)
        output = Dense(config.num_hidden, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(config.num_hidden, activation='relu')(output)
        output = Dense(data_handler.num_classes, activation='softmax')(output)

        self.model = Model(
            inputs=[words_input, coordinates, aux_features],
            outputs=[output]
        )
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='Adam', metrics=['accuracy'])
        # self.model.summary()
        self.data_handler = data_handler
        self.config = config

    def load_weights(self, path):
        """Loads weights from the given model file"""
        self.model.load_weights(path)
        print("\nSuccessfully loaded weights from {}".format(path))

    def predict(self, tokens, coordinates):
        """Performs inference on the given tokens and coordinates"""
        inp, coords = self.data_handler.process_data(tokens, coordinates)
        pred = self.model.predict([inp, coords], verbose=True)
        pred = pred.argmax(axis=-1)
        return pred

    def evaluate(self):
        predictions = self.model.predict([self.data_handler.train_data['inputs'],
                                          self.data_handler.train_data['coordinates']], verbose=True)
        predictions = predictions.argmax(axis=-1)
        acc = np.sum(predictions == (self.data_handler.train_data['labels']) /
                     float(len(self.data_handler.train_data['labels'])))
        print("\nTest Accuracy: {}".format(acc))
        return predictions

    def f1_score(self, predictions):
        true_labels = self.data_handler.train_data['labels']
        f1_scores, macrof1 = get_f1_scores(predictions, true_labels)
        print("\nMacro-Averaged F1: %.4f\n" % macrof1)
        return f1_scores, macrof1


class InvoiceNetCloudScan(InvoiceNetInterface):

    def __init__(self, data_handler, config):
        num_features = sum(v.shape[1] for v in data_handler.features.values())
        features = Input(shape=(num_features*5,),
                         dtype='float32', name='features')

        if config.num_layers == 2:
            output = Dense(config.num_hidden, activation='relu')(features)
            output = Dropout(0.5)(output)
        else:
            output = features
        output = Dense(config.num_output,
                       activation='softmax',
                       kernel_regularizer=L1L2(l1=0.0, l2=0.1))(output)
        self.model = Model(inputs=[features], outputs=[output])
        self.model.compile(optimizer='Adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())
        self.data_handler = data_handler
        self.config = config

    def prepare_data(self, features, labels):
        features_array = np.concatenate([v for v in features.values()], axis=1)

        spatial_features = np.zeros(
            [features_array.shape[0], features_array.shape[1]*4], dtype=np.float32)

        zero_vec = np.zeros(features_array.shape[1], dtype=np.float32)
        for i in range(features_array.shape[0]):
            vectors = [zero_vec if j == -1 else features_array[j]
                       # TODO fix this hardcoded reference
                       for j in self.data_handler.data.at[i, 'closest_ngrams']]
            spatial_features[i, :] = np.concatenate(vectors)

        features_array = np.concatenate((features_array, spatial_features), axis=1)
        # TODO add oversampling back
        return features_array, labels

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
