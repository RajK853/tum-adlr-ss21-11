import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from src import losses
from src.models import u_dense_net
from src.generators import DataGen
from src.utils import exec_from_yaml
from src.progressbar import ProgressBar
from src.callbacks import ImageSaverCallback
from src.load import get_values_sql, compressed2img, object2numeric_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N_PATHS_PER_WORLD = 1000


def load_world_data(db_path, n_voxels, n_dim):
    worlds = get_values_sql(file=db_path, table="worlds")
    obstacle_images = compressed2img(img_cmp=worlds.obst_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    obstacle_images = np.expand_dims(obstacle_images, axis=-1)
    return obstacle_images


def load_data(db_path, path_indexes, n_voxels, n_dim):
    # TODO: Create a generic load function later
    paths = get_values_sql(file=db_path, table='paths', rows=path_indexes)
    path_images = compressed2img(img_cmp=paths.path_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    start_images = compressed2img(img_cmp=paths.start_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    end_images = compressed2img(img_cmp=paths.end_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    # Process images
    start_images = np.expand_dims(start_images, axis=-1)
    end_images = np.expand_dims(end_images, axis=-1)
    path_images = np.expand_dims(path_images, axis=-1)
    return start_images, end_images, path_images


def image2image_callback(batch_indexes, data_dict):
    path_indexes = data_dict["path_rows"][batch_indexes]
    obst_indexes = path_indexes//N_PATHS_PER_WORLD
    obst_batch_data = data_dict["obst_imgs"][obst_indexes]
    goal_batch_data = data_dict["goal_imgs"][batch_indexes]
    path_batch_data = data_dict["path_imgs"][batch_indexes]
    input_batch_data = [np.concatenate([obst_batch_data, goal_batch_data], axis=-1)]
    output_batch_data = [path_batch_data]
    return input_batch_data, output_batch_data


def save_images(index, logs, log_dir):
    batch_inputs = logs["inputs"]
    batch_outputs = logs["true_outputs"]
    batch_predictions = logs["outputs"]
    for i in range(batch_predictions.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(batch_inputs[i, :, :, 0], origin='lower', cmap='binary')              # Obstacle 
        ax.imshow(batch_inputs[i, :, :, 1], origin='lower', cmap='Greens', alpha=0.3)   # Goal
        ax.imshow(batch_outputs[i, :, :, 0], origin='lower', cmap='Blues', alpha=0.5)   # True path
        ax.imshow(batch_predictions[i, :, :, 0], origin='lower', cmap='Reds', alpha=0.3)# Predicted path
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(f"{log_dir}/Image_{index+i}.png")
        plt.close(fig=fig)


def get_loss_func(loss_config):
    loss_name = loss_config.pop("name")
    loss_func = getattr(losses, loss_name, None)
    assert loss_func is not None, f"'{loss_name}' is not a valid loss function!"
    return loss_func(**loss_config)


def main(*, epochs, log_dir, batch_size, path_row_config, data_config, model_config, loss_config):
    timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M")
    log_path = os.path.join(log_dir, f"{loss_config['name']}_{timestamp}")
    # Generate path rows
    path_rows = {
        key: np.arange(*value) 
        for key, value in path_row_config.items()
    }
    # Load data
    # TODO: Concise data loading procedure
    print("# Loading data: ", end="")
    db_path = os.environ.get("DB_PATH")
    assert db_path is not None, "No database path found! Set the path using the command 'export DB_PATH=path/to/your/db/file'."
    print("train", end=", ")

    obst_imgs = load_world_data(db_path, **data_config)
    train_start_imgs, train_end_imgs, train_path_imgs = load_data(db_path, path_rows["train"], **data_config)
    train_path_imgs = train_path_imgs.astype("float32")
    train_goal_imgs = train_start_imgs + train_end_imgs   # Add start and end images together
    print("validation", end=", ")
    validation_start_imgs, validation_end_imgs, validation_path_imgs = load_data(db_path, path_rows["validation"], **data_config)
    validation_path_imgs = validation_path_imgs.astype("float32")
    validation_goal_imgs = validation_start_imgs + validation_end_imgs
    print("test")
    test_start_imgs, test_end_imgs, test_path_imgs = load_data(db_path, path_rows["test"], **data_config)
    test_path_imgs = test_path_imgs.astype("float32")
    test_goal_imgs = test_start_imgs + test_end_imgs
    del train_start_imgs, train_end_imgs, validation_start_imgs, validation_end_imgs, test_start_imgs, test_end_imgs
    # Prepare data generators
    print("# Creating data generators")
    data_dicts = {
        "train": {
            "path_rows": path_rows["train"],
            "obst_imgs": obst_imgs,
            "goal_imgs": train_goal_imgs,
            "path_imgs": train_path_imgs
        },
        "validation": {
            "path_rows": path_rows["validation"],
            "obst_imgs": obst_imgs,
            "goal_imgs": validation_goal_imgs,
            "path_imgs": validation_path_imgs
        },
        "test": {
            "path_rows": path_rows["test"],
            "obst_imgs": obst_imgs,
            "goal_imgs": test_goal_imgs,
            "path_imgs": test_path_imgs
        }
    }
    data_gens = {
        key: DataGen(data_dict, callback=image2image_callback, batch_size=batch_size)
        for key, data_dict in data_dicts.items()
    }
    # Load model
    print("# Loading U-DenseNet model")
    tf.keras.backend.clear_session()
    lr = model_config.pop("lr", 1e-3)
    denseNet = u_dense_net(**model_config)
    loss_func = get_loss_func(loss_config)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    denseNet.compile(optimizer=optimizer, loss=loss_func)
    os.makedirs(log_path, exist_ok=True)
    model_img_path = os.path.join(log_path, 'model.png')
    tf.keras.utils.plot_model(denseNet, to_file=model_img_path, show_shapes=True)
    print(f"# Model graph saved at '{model_img_path}'")
    # Train model
    train_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "tb_logs")), 
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5)
    ]
    print("# Training the model")
    print("  Training information are available via Tensorboard with the given command:")
    print(f"  tensorboard --host localhost --logdir {log_dir}\n")
    history = denseNet.fit(data_gens["train"], validation_data=data_gens["validation"], epochs=epochs, callbacks=train_callbacks)
    # Save model
    model_path = os.path.join(log_path, "model.tf")
    denseNet.save(model_path)
    print(f"# Model saved at '{model_path}'")
    # Predict on test data set
    print("# Predicting on test data set")
    test_callbacks = [
        ImageSaverCallback(data_gens["test"], os.path.join(log_path, "test_images"), callback=save_images)
    ]
    denseNet.predict(data_gens["test"], callbacks=test_callbacks)

if __name__ == "__main__":
    config_path = sys.argv[1]
    exec_from_yaml(config_path, exec_func=main, safe_load=True)
