from tensorflow.compat.v1.keras import Model
from tensorflow.compat.v1.keras.layers import Input, Conv1D, Conv2D, Concatenate, Flatten, Dense, Reshape, MaxPool2D

from .layers import DenseBlock, TransitionBlock, LayerNormBlock


def u_dense_net(input_shape, num_db, num_channels=64, growth_rate=32, convs_per_db=3):
    assert len(input_shape) == 3, f"Input shape must have 3 dimension! Received '{input_shape}'!"
    assert (num_db > 1) and (num_db % 2 == 1), f"Number of DenseBlocks must be an odd number more than 1! Received '{num_db}'!"
    # In a U-shaped DenseNet with N DenseBlocks, each side has floor(N/2) DenseBlocks  
    num_trans_down = num_trans_up = num_db//2
    assert (input_shape[0] % (2**num_trans_down) == 0) and (input_shape[1] % (2**num_trans_down) == 0), f"Dimension of the input shape {input_shape[:2]} must be a multiple of {2**num_trans_down} to preserve the tensor shape after down-scaling and up-scaling"
    assert (num_channels > 0) and (num_channels % 2 == 0), f"Number of channels for TransitionBlock must be an even number more than 0! Received '{num_channels}'!"    
    
    _num_channels = num_channels 
    img_in = Input(dtype="float32", shape=input_shape, name="image_input")
    x = Conv2D(_num_channels, kernel_size=(5, 5), activation="relu", padding="same")(img_in)
    ############################### Transition down section ###############################
    db_outputs = []
    for i in range(num_trans_down):
        x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
        db_outputs.insert(0, x)
        num_channels += growth_rate*i
        num_channels //= 2
        x = TransitionBlock(filters=num_channels, trans_down=True)(x)
    #################################### Mid DenseBlock ###################################
    x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
    ################################ Transition up section ################################
    for i in range(num_trans_up):
        num_channels += growth_rate*(i+1)
        num_channels //= 2
        x = TransitionBlock(filters=num_channels, trans_down=False)(x)
        x = Concatenate(axis=-1)([x, db_outputs[i]])
        x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
    # x = Conv2D(_num_channels, kernel_size=(5, 5), activation="relu", padding="same")(x)
    # TODOs: Configure output layer arguments 
    x_out = Conv2D(1, kernel_size=(5, 5), activation="sigmoid", padding="same", name="image_output")(x)
    model = Model(inputs=[img_in], outputs=[x_out], name="DenseNet")
    return model

def dense_net(input_shape, output_shape, num_db, num_channels=32, growth_rate=32, convs_per_db=3):
    # TODO: Get output_shape instead 
    assert len(input_shape) == 3, f"Input shape must have 3 dimension! Received '{input_shape}'!"
    assert (num_db > 1), f"Number of DenseBlocks must be more than 1! Received '{num_db}'!"
    # In a U-shaped DenseNet with N DenseBlocks, each side has floor(N/2) DenseBlocks  
    assert (num_channels > 0), f"Number of channels for TransitionBlock must be more than 0! Received '{num_channels}'!"    
    
    _num_channels = num_channels 
    img_in = Input(dtype="float32", shape=input_shape, name="image_input")
    x = Conv2D(_num_channels, kernel_size=(5, 5), activation="relu", padding="same")(img_in)
    for i in range(num_db):
        x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
        num_channels += growth_rate*i
        num_channels //= 2
        x = TransitionBlock(filters=num_channels, trans_down=True)(x)
    x = Conv2D(_num_channels, kernel_size=(5, 5), activation="relu", padding="same")(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)    
    x = Dense(output_shape[0]*output_shape[1], activation=None)(x)
    path_out = Reshape(output_shape)(x)
    model = Model(inputs=[img_in], outputs=[path_out], name="DenseNet")
    return model

def u_dense_net_2(input_shape, output_shape, num_db, num_channels=64, growth_rate=32, convs_per_db=3):
    assert len(input_shape) == 3, f"Input shape must have 3 dimension! Received '{input_shape}'!"
    assert (num_db > 1) and (num_db % 2 == 1), f"Number of DenseBlocks must be an odd number more than 1! Received '{num_db}'!"
    # In a U-shaped DenseNet with N DenseBlocks, each side has floor(N/2) DenseBlocks  
    num_trans_down = num_trans_up = num_db//2
    assert (input_shape[0] % (2**num_trans_down) == 0) and (input_shape[1] % (2**num_trans_down) == 0), f"Dimension of the input shape {input_shape[:2]} must be a multiple of {2**num_trans_down} to preserve the tensor shape after down-scaling and up-scaling"
    assert (num_channels > 0) and (num_channels % 2 == 0), f"Number of channels for TransitionBlock must be an even number more than 0! Received '{num_channels}'!"    
    
    _num_channels = num_channels 
    img_in = Input(dtype="float32", shape=input_shape, name="image_input")
    x = Conv2D(_num_channels, kernel_size=(5, 5), activation="relu", padding="same")(img_in)
    ############################### Transition down section ###############################
    db_outputs = []
    for i in range(num_trans_down):
        x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
        db_outputs.insert(0, x)
        num_channels += growth_rate*i
        num_channels //= 2
        x = TransitionBlock(filters=num_channels, trans_down=True)(x)
    #################################### Mid DenseBlock ###################################
    x_mid = x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
    ################################ Transition up section ################################
    for i in range(num_trans_up):
        num_channels += growth_rate*(i+1)
        num_channels //= 2
        x = TransitionBlock(filters=num_channels, trans_down=False)(x)
        x = Concatenate(axis=-1)([x, db_outputs[i]])
        x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
    img_out = Conv2D(1, kernel_size=(5, 5), activation="sigmoid", padding="same", name="img_out")(x)
    
    x = TransitionBlock(filters=_num_channels, trans_down=True)(x)
    x = DenseBlock(num_layers=convs_per_db, filters=growth_rate)(x)
    x = TransitionBlock(filters=_num_channels, trans_down=True)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(output_shape[0]*output_shape[1])(x)
    path_out = Reshape(output_shape, name="path_out")(x)
    model = Model(inputs=[img_in], outputs=[img_out, path_out], name="DenseNet")
    return model