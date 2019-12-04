import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class RangeLRSearch(Callback):
    def __init__(self, *args, initial_lr=1e-4, max_lr=1e10, **kwds):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.increase_qoutient = 1.1
        super().__init__(*args, **kwds)

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        K.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        logs["lr"] = lr
        if lr > self.max_lr:
            return
        print(f"Setting lr to {self.increase_qoutient * lr}")
        K.set_value(self.model.optimizer.lr, self.increase_qoutient * lr)
