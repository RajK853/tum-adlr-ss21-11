import os
import numpy as np
import tensorflow.compat.v1 as tf

from .progressbar import ProgressBar

Callback = tf.keras.callbacks.Callback


class ImageSaverCallback(Callback):
    def __init__(self, data_gen, log_dir, callback):
        self.data_gen = data_gen
        self.log_dir = log_dir
        self.callback = callback
        os.makedirs(log_dir, exist_ok=True)
        N = len(self.data_gen.indexes)
        self.pbar = ProgressBar(total_iter=N, title=f"Saving {N} images:")
        self.index = 0
    
    def on_predict_batch_end(self, batch, logs=None):
        batch_size = logs["outputs"].shape[0]
        batch_inputs, batch_outputs = self.data_gen[batch]
        logs["inputs"] = batch_inputs[0]
        logs["true_outputs"] = batch_outputs[0]
        self.callback(self.index, logs, self.log_dir)
        self.pbar.step(batch_size)
        self.index += batch_size
        