import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf


from src.load import get_values_sql
from src.models import u_dense_net_2
from src.callbacks import ImageSaverCallback
from src.utils import exec_from_yaml, dump_yaml
from train_image2image_model import decode_data_tf, get_data_gen, get_loss_func


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

N_DIM = 2
N_VOXELS = 64
VOXEL_SIZE = 10 / 64     # in m
EXTENT = [0, 10, 0, 10]  # in m
N_WAYPOINTS = 22         # start + 20 inner points + end
N_PATHS_PER_WORLD = 1000 # Load database path
VECTOR_SHAPE = (22, 2)


def image2vector_callback(data_dict):
    """
    Callback function to preprocess input data for image-to-vector approach
    """
    input_data = tf.concat((data_dict["obst_img_cmp"], data_dict["start_img_cmp"], data_dict["end_img_cmp"]), axis=-1)
    output_data = {"img_out": data_dict["path_img_cmp"], "path_out": data_dict["q_path"]}
    return input_data, output_data


def image2vector_saver(index, logs, log_dir):
    """
    Callback function to save results during evaluation
    """
    batch_inputs = logs["inputs"]
    batch_img_outputs, batch_vec_outputs = logs["true_outputs"]
    batch_img_predictions, batch_vec_predictions = logs["outputs"]
    batch_size, img_w, img_h, *_ = batch_inputs.shape
    img_shape = (img_w, img_h, 3)                                          # RGB image shape
    for i in range(batch_size):
        # Prepare plot image
        img = np.full(img_shape, 1-batch_inputs[i, :, :, 0:1], dtype=np.float32)
        img[:, :, [1, 2]] -= batch_inputs[i, :, :, 1:2]                    # Start pos in red channel
        img[:, :, [0, 2]] -= batch_inputs[i, :, :, 2:3]                    # End pos in green channel
        # img -= 0.5*batch_img_outputs[i]                                    # True path in grey color (partially all channels)
        img[:, :, [1, 2]] -= 0.75*batch_img_predictions[i]                 # Predicted path in red channel
        np.clip(img, 0.0, 1.0, out=img)
        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(img, origin="lower", extent=EXTENT)
        # Plot vector lines
        py, px = batch_vec_outputs[i, :, :].T
        ax.plot(px, py, color="k", marker="o")                             # True path
        py, px = batch_vec_predictions[i].T
        ax.plot(px, py, color="y", marker="o")                             # Predicted path
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{log_dir}/Image_{index+i}.png")
        plt.close(fig=fig)


def l2_loss_func(y_true, y_pred):
    """
    L2-loss function
    """
    loss = tf.reduce_mean(tf.norm(y_true-y_pred, ord=1, axis=-1))
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
    cmp_names = ["start_img_cmp", "end_img_cmp", "path_img_cmp", "q_path"]
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
    lr = model_config.pop("lr", 1e-3)
    model_config["input_shape"] = (*model_config["input_shape"][:2], 3)
    denseNet = u_dense_net(output_shape=VECTOR_SHAPE, **model_config)
    img_loss_func = get_loss_func(loss_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    denseNet.compile(optimizer=optimizer, loss=[img_loss_func, l2_loss_func], loss_weights=[1.0, 2.0])
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
