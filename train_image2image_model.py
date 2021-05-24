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
N_PATHS_PER_WORLD = 1000# Load database path


def load_data_from_sql(db_path, table, cmp_names, rows=-1):
    table_df = get_values_sql(file=db_path, table=table, rows=rows, columns=cmp_names)
    for cmp_name in cmp_names:
        if "_img_" in cmp_name:
            images = compressed2img(img_cmp=table_df[cmp_name].values, n_voxels=N_VOXELS, n_dim=N_DIM)
            images = np.expand_dims(images, axis=-1)
            yield images
            del images
        else:
            values = object2numeric_array(table_df[cmp_name].values)
            values = values.reshape(-1, N_WAYPOINTS, N_DIM)
            yield values
            del values


def image2image_callback(batch_indexes, data_dict):
    path_indexes = data_dict["path_rows"][batch_indexes]
    obst_indexes = path_indexes//N_PATHS_PER_WORLD
    obst_batch_data = data_dict["obst_imgs"][obst_indexes]
    goal_batch_data = data_dict["goal_imgs"][batch_indexes]
    path_batch_data = data_dict["path_imgs"][batch_indexes]
    input_batch_data = [np.concatenate([obst_batch_data, goal_batch_data], axis=-1)]
    output_batch_data = [path_batch_data]
    return input_batch_data, output_batch_data


def image2image_saver(index, logs, log_dir):
    batch_inputs = logs["inputs"]
    batch_outputs = logs["true_outputs"]
    batch_predictions = logs["outputs"]
    for i in range(batch_predictions.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(batch_inputs[i, :, :, 0], origin="lower", cmap="binary", extent=EXTENT)               # Obstacle 
        ax.imshow(batch_inputs[i, :, :, 1], origin="lower", cmap="Greens", extent=EXTENT, alpha=0.3)    # Goal
        ax.imshow(batch_outputs[i, :, :, 0], origin="lower", cmap="Blues", extent=EXTENT, alpha=0.5)    # True path
        ax.imshow(batch_predictions[i, :, :, 0], origin="lower", cmap="Reds", extent=EXTENT, alpha=0.5) # Predicted path
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(f"{log_dir}/Image_{index+i}.png")
        plt.close(fig=fig)


def get_loss_func(loss_config):
    loss_name = loss_config.pop("name")
    loss_func = getattr(losses, loss_name, None)
    assert loss_func is not None, f"'{loss_name}' is not a valid loss function!"
    return loss_func(**loss_config)


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
    cmp_names = ["obst_img_cmp"]
    obst_imgs, *_ = load_data_from_sql(db_path, table="worlds", cmp_names=cmp_names)

    # Load train, validation and test data generators
    cmp_names = ["start_img_cmp", "end_img_cmp", "path_img_cmp"]
    data_gens = {}
    for data_type, path_row_range in path_row_config.items():
        path_rows = np.arange(*path_row_range)
        start_imgs, end_imgs, path_imgs = load_data_from_sql(db_path, table="paths", rows=path_rows, cmp_names=cmp_names)
        # Numpy.add reduces the memory spike during addition operation compared to the + operator
        goal_imgs = np.add(start_imgs, end_imgs, out=start_imgs)
        data_dict = {
            "path_rows": path_rows,
            "obst_imgs": obst_imgs,
            "path_imgs": path_imgs,
            "goal_imgs": goal_imgs,
        }
        data_gens[data_type] = DataGen(data_dict, callback=image2image_callback, batch_size=batch_size, length_key="path_rows")
        del path_rows, start_imgs, end_imgs, path_imgs, goal_imgs
        
    # Load model
    print("# Loading U-DenseNet model")
    tf.keras.backend.clear_session()
    lr = model_config.pop("lr", 1e-3)
    denseNet = u_dense_net(**model_config)
    loss_func = get_loss_func(loss_config)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    denseNet.compile(optimizer=optimizer, loss=loss_func)
    model_img_path = os.path.join(log_path, 'model.png')
    tf.keras.utils.plot_model(denseNet, to_file=model_img_path, show_shapes=True)
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
    history = denseNet.fit(data_gens["train"], validation_data=data_gens["validation"], epochs=epochs, callbacks=train_callbacks)

    # Save model
    model_path = os.path.join(log_path, "model.tf")
    denseNet.save(model_path)
    print(f"# Model saved at '{model_path}'")

    # Predict on test data set
    print("# Predicting on test data set")
    img_dump_path = os.path.join(log_path, "test_images")
    test_callbacks = [ImageSaverCallback(data_gens["test"], img_dump_path, callback=image2image_saver)]
    denseNet.predict(data_gens["test"], callbacks=test_callbacks)

if __name__ == "__main__":
    config_path = sys.argv[1]
    exec_from_yaml(config_path, exec_func=main, safe_load=True)
