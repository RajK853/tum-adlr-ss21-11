import numpy as np
from tensorflow.compat.v1.keras.utils import Sequence

class DataGen(Sequence):
    def __init__(self, data_dict, batch_size=32, callback=None):
        self.data_dict = data_dict
        self.batch_size = batch_size
        num_items = len(self.data_dict["goal_imgs"])
        self.indexes = np.arange(num_items, dtype="uint16")
        self.gen_len = np.math.ceil(num_items/self.batch_size)
        self.callback = self.default_callback if callback is None else callback
        assert callable(self.callback)
    
    def __len__(self):
        return self.gen_len
    
    @staticmethod
    def default_callback(batch_indexes, data_dict):
        return {key: val[batch_indexes] for key, val in data_dict.items()}
    
    def __getitem__(self, index):
        lower_index = index * self.batch_size
        higher_index = lower_index + self.batch_size
        batch_indexes = self.indexes[lower_index:higher_index]
        batch_data = self.callback(batch_indexes, self.data_dict)
        return batch_data
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
