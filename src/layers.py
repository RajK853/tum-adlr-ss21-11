import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Layer, Conv2D, ReLU, Concatenate, BatchNormalization, AvgPool2D, UpSampling2D


class ConvBlock(Layer):
    def __init__(self, filters, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.bn = BatchNormalization() 
        self.conv2d = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")
        self.relu = ReLU()
        self.concat = Concatenate(axis=-1)
        
    def call(self, x, training=False):
        x_in = x
        x = self.bn(x, training=training) 
        x = self.conv2d(x)
        x = self.relu(x)
        x_out = self.concat([x_in, x])
        return x_out
    
    def get_config(self):
        base_configs = super().get_config()
        return {"filters": self.filters, **base_configs}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseBlock(Layer):    
    def __init__(self, num_layers, filters, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.filters = filters
        self.num_layers = num_layers
        self.layers = [ConvBlock(self.filters) for i in range(self.num_layers)]
    
    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x
    
    def get_config(self):
        base_configs = super().get_config()
        return {"filters": self.filters, "num_layers": self.num_layers, **base_configs}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransitionBlock(Layer):
    def __init__(self, filters, trans_down=True, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.trans_down = trans_down
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.conv2d = Conv2D(filters=filters, kernel_size=(1, 1), padding="same")
        if trans_down:
            self.pool2d = AvgPool2D(pool_size=2, strides=2)
        else:
            self.pool2d = UpSampling2D(size=(2, 2))
    
    def call(self, x, training=False):
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.conv2d(x)
        x = self.pool2d(x)
        return x
    
    def get_config(self):
        base_configs = super().get_config()
        return {"filters": self.filters, "trans_down": self.trans_down, **base_configs}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config
