import sys
import numpy as np
import matplotlib.pyplot as plt
from src.load import get_values_sql, compressed2img, object2numeric_array

n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 22  # start + 20 inner points + end
n_dim = 2
n_paths_per_world = 1000
path_rows = [0, 1, 1000, 2000, 3500]

def main(db_path):
    worlds = get_values_sql(file=db_path, table='worlds')
    obstacle_images = compressed2img(img_cmp=worlds.obst_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    print(f"Obstacle images dimension: {obstacle_images.shape}")

    paths = get_values_sql(file=db_path, table='paths', rows=path_rows)
    path_images = compressed2img(img_cmp=paths.path_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    start_images = compressed2img(img_cmp=paths.start_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
    end_images = compressed2img(img_cmp=paths.end_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

    q_paths = object2numeric_array(paths.q_path.values)
    q_paths = q_paths.reshape(-1, n_waypoints, n_dim)

    nrows = min(5, len(path_rows))             # Only plot the first N graphs
    _, axs = plt.subplots(nrows=nrows, ncols=5, figsize=(3*nrows, 20))
    for i in range(nrows):
        world_i = path_rows[i]//n_paths_per_world
        axs[i, 0].imshow(obstacle_images[world_i].T, origin='lower', extent=extent, cmap='binary')
        axs[i, 1].imshow(start_images[i].T, origin='lower', extent=extent, cmap='Greens')
        axs[i, 2].imshow(end_images[i].T, origin='lower', extent=extent, cmap='Reds')
        axs[i, 3].imshow(path_images[i].T, origin='lower', extent=extent, cmap='binary')
        # Combined plot
        axs[i, 4].imshow(obstacle_images[world_i].T, origin='lower', extent=extent, cmap='binary')
        axs[i, 4].imshow(start_images[i].T, origin='lower', extent=extent, cmap='Greens', alpha=0.4)
        axs[i, 4].imshow(end_images[i].T, origin='lower', extent=extent, cmap='Reds', alpha=0.4)
        axs[i, 4].imshow(path_images[i].T, origin='lower', extent=extent, cmap='Blues', alpha=0.2)
        axs[i, 4].plot(*q_paths[i].T, color='k', marker='o')
        # Set titles and remove x/y ticks
        for (ax, title) in zip(axs[i], ("Obstacles", "Start point", "End point", "Path", "Combined")) :
            if i == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    # database_path = Path of SingleSphere02.db database file
    database_path = sys.argv[1]
    assert database_path.endswith(".db"), f"Invalid database file path detected! Received '{database_path}'"
    main(database_path)
