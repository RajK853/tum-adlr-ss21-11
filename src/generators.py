import numpy as np
from tensorflow.compat.v1.keras.utils import Sequence

class DataGen(Sequence):
    def __init__(self, data_dict, callback, batch_size=32, length_key="path_rows"):
        self.data_dict = data_dict
        self.batch_size = batch_size
        num_items = len(self.data_dict[length_key])
        self.indexes = np.arange(num_items, dtype="uint16")
        self.gen_len = np.math.ceil(num_items/self.batch_size)
        self.callback = callback
        assert callable(self.callback)
    
    def __len__(self):
        return self.gen_len
    
    def __getitem__(self, index):
        lower_index = index*self.batch_size
        higher_index = lower_index + self.batch_size
        batch_indexes = self.indexes[lower_index:higher_index]
        batch_data = self.callback(batch_indexes, self.data_dict)
        return batch_data
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
