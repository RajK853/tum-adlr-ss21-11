import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from src import losses                      # Change
from src.models import dense_net            # Change
from src.utils import exec_from_yaml, dump_yaml
from src.callbacks import ImageSaverCallback
from src.load import get_values_sql, compressed2img, object2numeric_array 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

N_DIM = 2
N_VOXELS = 64
VOXEL_SIZE = 10 / 64     # in m
EXTENT = [0, 10, 0, 10]  # in m
N_WAYPOINTS = 22  # start + 20 inner points + end
N_PATHS_PER_WORLD = 1000 # Load database path


def decode_data_tf(raw_data_dict, img_shape=(64, 64, 1), vector_shape=(22, 2), dtype=tf.uint8):
    def decode_img(img_bytes):
        img_str_tf = tf.io.decode_compressed(img_bytes, compression_type="ZLIB")
        img_tf = tf.io.decode_raw(img_str_tf, out_type=dtype)
        img_tf = tf.reshape(img_tf, (tf.shape(img_bytes)[0], *img_shape))
        return img_tf
    
    def decode_vector(vector_bytes):
        vector_tf = tf.reshape(vector_bytes, (tf.shape(vector_bytes)[0], *vector_shape))
        return vector_tf
    
    def decode(key, value):
        if key.endswith("_img_cmp"):
            return decode_img(value)
        return decode_vector(value)
    
    data_dict = {key: decode(key, value) for key, value in raw_data_dict.items()}
    return data_dict


def image2vector_callback(data_dict):
    input_data = tf.concat((data_dict["obst_img_cmp"], data_dict["start_img_cmp"], data_dict["end_img_cmp"]), axis=-1)
    output_data = data_dict["q_path"]
    return input_data, output_data


def get_data_gen(data_df, batch_size, callback, epochs=1, img_shape=(64, 64, 1), vector_shape=(22, 2), shuffle=True):
    def preprocess(key, value):
        if key.endswith("_img_cmp"):
            return value
        return np.stack(value)
    
    def decoder(x):
        return decode_data_tf(x, img_shape=img_shape, vector_shape=vector_shape)

    data_dict = {k: preprocess(k, v) for k, v in data_df.items()}
    # Normalize q_path from 0.0-10.0 range to 0.0-1.0.
    # data_dict["q_path"] = data_dict["q_path"]/EXTENT[1]
    data_gen = tf.data.Dataset.from_tensor_slices(data_dict)
    if shuffle:
        data_gen = data_gen.shuffle(len(data_df)//10)
    data_gen = data_gen.batch(batch_size)
    data_gen = data_gen.map(decoder, num_parallel_calls=tf.data.AUTOTUNE)
    data_gen = data_gen.cache()
    data_gen = data_gen.map(callback, num_parallel_calls=tf.data.AUTOTUNE)
    if epochs > 1:
        data_gen = data_gen.repeat(epochs)
    data_gen = data_gen.prefetch(tf.data.AUTOTUNE)
    return data_gen


def image2vector_saver(index, logs, log_dir):   # TODO: Reworking like image2image_saver needed?
    batch_inputs = logs["inputs"]
    batch_outputs = logs["true_outputs"]
    batch_predictions = logs["outputs"]
    batch_size, img_w, img_h, *_ = batch_inputs.shape
    img_shape = (img_w, img_h, 3)          # RGB image shape
    for i in range(batch_size):
        # Prepare plot image
        img = np.full(img_shape, 1-batch_inputs[i, :, :, 0:1], dtype=np.float32)
        img[:, :, [1, 2]] -= batch_inputs[i, :, :, 1:2]                    # Start pos in red channel
        img[:, :, [0, 2]] -= batch_inputs[i, :, :, 2:3]                    # End pos in green channel
        np.clip(img, 0.0, 1.0, out=img)
        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(img, origin="lower", extent=EXTENT)
        # Plot vector lines
        py, px = batch_outputs[i, :, :].T
        ax.plot(px, py, color="k", marker="o")    # True path
        py, px = batch_predictions[i].T
        ax.plot(px, py, color="darkred", marker="o")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{log_dir}/Image_{index+i}.png")
        plt.close(fig=fig)


def get_loss_func(loss_config):
    loss_name = loss_config.pop("name")
    loss_func = getattr(losses, loss_name, None)
    assert loss_func is not None, f"'{loss_name}' is not a valid loss function!"
    return loss_func(**loss_config)


def loss_func(y_true, y_pred):
    def get_length(points):
        point_separations = tf.norm(points[:, 1:]-points[:, :-1], ord=1, axis=-1)
        vector_length = tf.reduce_sum(point_separations, axis=-1)
        return vector_length

    def length_loss(y_true, y_pred):
        y_true_length = get_length(y_true)
        y_pred_length = get_length(y_pred)
        return (y_true_length - y_pred_length)**2

    loss = tf.reduce_mean(tf.norm(y_true-y_pred, ord=1, axis=-1))
    # loss += tf.reduce_mean(length_loss(y_true, y_pred))
    return loss


def main(*, epochs, log_dir, batch_size, path_row_config, model_config, loss_config):
    # Save experiment parameters to dump it later
    exp_config = {"Config": locals()}

    # Create log directory based on current timestamp
    timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M")
    log_path = os.path.join(log_dir, f"{loss_config['name']}_{timestamp}")
    
    # Dump experiment parameters to a YAML file
    os.makedirs(log_path, exist_ok=True)
    config_dump_path = os.path.join(log_path, "config.yaml")
    dump_yaml(exp_config, file_path=config_dump_path)
    print(f"# Experiment configuration saved at '{config_dump_path}'")


    db_path = os.environ.get("DB_PATH")
    assert db_path is not None, "No database path found! Set the path using the command 'export DB_PATH=path/to/the/db/file'."

    # Load obstacle data
    print("# Loading data")
    world_df = get_values_sql(file=db_path, table="worlds", columns=["obst_img_cmp"])

    # Load train, validation and test data generators
    steps = {}
    data_gens = {}
    cmp_names = ["start_img_cmp", "end_img_cmp", "q_path"]
    for data_type, path_row_range in path_row_config.items():
        path_rows = np.arange(*path_row_range)
        data_df = get_values_sql(file=db_path, table="paths", rows=path_rows, columns=cmp_names)
        data_df["obst_img_cmp"] = world_df.iloc[data_df.index//N_PATHS_PER_WORLD].values
        if data_type == "test":
            data_gens[data_type] = get_data_gen(data_df, batch_size=batch_size, callback=image2vector_callback, epochs=1, shuffle=False)
            steps[data_type] = len(path_rows)
        else:
            data_gens[data_type] = get_data_gen(data_df, batch_size=batch_size, callback=image2vector_callback, epochs=epochs, shuffle=True)
            steps[data_type] = np.math.ceil(len(path_rows)/batch_size)
    # Load model
    print("# Loading DenseNet model")
    tf.keras.backend.clear_session()
    lr = model_config.pop("lr", 1e-3) # TODO: Pass as main() function parameter?
    denseNet = dense_net(**model_config)   # TODO: Get output size from config
    # loss_func = get_loss_func(loss_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    denseNet.compile(optimizer=optimizer, loss=loss_func)
    model_img_path = os.path.join(log_path, 'model.png')
    tf.keras.utils.plot_model(denseNet, to_file=model_img_path, show_layer_names=False, show_shapes=True)
    print(f"# Model graph saved at '{model_img_path}'")

    # Train model
    print("# Training the model")
    print("  Training information are available via Tensorboard with the given command:")
    print(f"  tensorboard --host localhost --logdir {log_dir}\n")
    tb_log_path = os.path.join(log_path, "tb_logs")
    train_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tb_log_path), 
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5),
    ]
    train_kwargs = {
        "x": data_gens["train"], 
        "validation_data": data_gens["validation"],
        "steps_per_epoch": steps["train"],
        "validation_steps": steps["validation"],
        "epochs": epochs, 
        "callbacks": train_callbacks
    }
    history = denseNet.fit(**train_kwargs)

    # Save model
    model_path = os.path.join(log_path, "model.tf")
    denseNet.save(model_path)
    print(f"# Model saved at '{model_path}'")

    # Predict on test data set
    print("# Predicting on test data set")
    img_dump_path = os.path.join(log_path, "test_images")
    test_callbacks = [ImageSaverCallback(data_gens["test"], img_dump_path, callback=image2vector_saver, data_gen_size=steps["test"])]
    denseNet.predict(data_gens["test"], callbacks=test_callbacks)

if __name__ == "__main__":
    config_path = sys.argv[1]
    exec_from_yaml(config_path, exec_func=main, safe_load=True)
