import os
import numpy as np
import tensorflow.compat.v1 as tf

from .progressbar import ProgressBar

Callback = tf.keras.callbacks.Callback


class ImageSaverCallback(Callback):
    
    def __init__(self, data_gen, log_dir, callback, data_gen_size=100):
        self.data_gen = data_gen
        self.data_iter = iter(data_gen)
        self.log_dir = log_dir
        self.callback = callback
        self.pbar = ProgressBar(total_iter=data_gen_size, title=f"Saving images:")
        self.index = 0
        os.makedirs(self.log_dir, exist_ok=True)
    
    def on_predict_begin(self, logs=None):
        self.index = 0
        self.data_iter = iter(self.data_gen)
    
    def on_predict_batch_end(self, batch, logs=None):
        batch_inputs, batch_outputs = next(self.data_iter)
        logs["inputs"] = batch_inputs.numpy()
        if isinstance(batch_outputs, dict):
            logs["true_outputs"] = [v.numpy() for v in batch_outputs.values()]
            batch_size = logs["true_outputs"][0].shape[0]
        else:
            logs["true_outputs"] = batch_outputs.numpy()
            batch_size = logs["true_outputs"].shape[0]
        self.callback(self.index, logs, self.log_dir)
        self.pbar.step(batch_size)
        self.index += batch_size
        