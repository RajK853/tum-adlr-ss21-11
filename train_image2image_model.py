import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from src import losses
from src.models import u_dense_net
from src.generators import DataGen
from src.utils import exec_from_yaml, dump_yaml
from src.callbacks import ImageSaverCallback
from src.load import get_values_sql, compressed2img, object2numeric_array

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

N_DIM = 2
N_VOXELS = 64
VOXEL_SIZE = 10 / 64     # in m
EXTENT = [0, 10, 0, 10]  # in m
N_WAYPOINTS = 22  # start + 20 inner points + end
N_PATHS_PER_WORLD = 1000  # Load database path


def image2image_saver(index, logs, log_dir):
    batch_inputs = logs["inputs"]
    batch_outputs = logs["true_outputs"]
    batch_predictions = logs["outputs"]
    # TODO: Save only Nth images
    for i in range(batch_predictions.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(batch_inputs[i, :, :, 0], origin="lower", cmap="binary", extent=EXTENT)               # Obstacle 
        ax.imshow(batch_inputs[i, :, :, 1], origin="lower", cmap="Greens", extent=EXTENT, alpha=0.3)    # Goal
        ax.imshow(batch_outputs[i, :, :, 0], origin="lower", cmap="Blues", extent=EXTENT, alpha=0.5)    # True path
        ax.imshow(batch_predictions[i, :, :, 0], origin="lower", cmap="Reds", extent=EXTENT, alpha=0.5) # Predicted path
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


def compressed2img_tf(bytes_dict, shape=None, dtype="uint8"):
    def decode_img(img_bytes):
        img_str_tf = tf.io.decode_compressed(img_bytes, compression_type="ZLIB")
        img_tf = tf.io.decode_raw(img_str_tf, out_type=dtype)
        if shape is not None:
            img_tf = tf.reshape(img_tf, (tf.shape(img_bytes)[0], *shape))
        return img_tf
    
    return {key: decode_img(img_bytes) for key, img_bytes in bytes_dict.items()}
    

def image2image_callback_tf(data_dict):
    goal_imgs = data_dict["start_img_cmp"] + data_dict["end_img_cmp"]
    obst_imgs = data_dict["obst_img_cmp"]
    input_data = tf.concat((obst_imgs, goal_imgs), axis=-1)
    output_data = data_dict["path_img_cmp"]
    return input_data, output_data


def get_data_gen(data_df, batch_size, epochs=1, shuffle=True):
    data_gen = tf.data.Dataset.from_tensor_slices(dict(data_df))
    if shuffle:
        data_gen = data_gen.shuffle(len(data_df)//10)
    data_gen = data_gen.batch(batch_size)
    data_gen = data_gen.map(lambda x: compressed2img_tf(x, shape=(64, 64, 1)), num_parallel_calls=tf.data.AUTOTUNE)
    data_gen = data_gen.map(image2image_callback, num_parallel_calls=tf.data.AUTOTUNE)
    data_gen = data_gen.cache()
    if epochs > 1:
        data_gen = data_gen.repeat(epochs)
    data_gen = data_gen.prefetch(tf.data.AUTOTUNE)
    return data_gen


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
    cmp_names = ["start_img_cmp", "end_img_cmp", "path_img_cmp"]
    data_gens = {}
    steps = {}
    for data_type, path_row_range in path_row_config.items():
        path_rows = np.arange(*path_row_range)
        data_df = get_values_sql(file=db_path, table="paths", rows=path_rows, columns=cmp_names)
        data_df["obst_img_cmp"] = world_df.iloc[data_df.index//N_PATHS_PER_WORLD].values
        if data_type == "test":
            data_gens[data_type] = get_data_gen(data_df, batch_size=batch_size, epochs=1, shuffle=False)
            steps[data_type] = len(path_rows)
        else:
            data_gens[data_type] = get_data_gen(data_df, batch_size=batch_size, epochs=epochs, shuffle=True)
            steps[data_type] = np.math.ceil(len(path_rows)/batch_size)
        
    # Load model
    print("# Loading U-DenseNet model")
    tf.keras.backend.clear_session()
    lr = model_config.pop("lr", 1e-3)
    denseNet = u_dense_net(**model_config)
    loss_func = get_loss_func(loss_config)
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
    test_callbacks = [ImageSaverCallback(data_gens["test"], img_dump_path, callback=image2image_saver, data_gen_size=steps["test"])]
    denseNet.predict(data_gens["test"], callbacks=test_callbacks)

if __name__ == "__main__":
    config_path = sys.argv[1]
    exec_from_yaml(config_path, exec_func=main, safe_load=True)
