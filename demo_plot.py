import sys
import numpy as np
from src.load import get_values_sql, compressed2img, object2numeric_array

n_voxels = 64
voxel_size = 10 / 64     # in m
extent = [0, 10, 0, 10]  # in m
n_waypoints = 22  # start + 20 inner points + end
n_dim = 2
n_paths_per_world = 1000
n_worlds = 5000
path_rows = [0, 1, 2, 1000, 2000]

def main(db_path):
	worlds = get_values_sql(file=db_path, table='worlds')
	obstacle_images = compressed2img(img_cmp=worlds.obst_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

	# always 1000 paths belong to one world
	# 0...999     -> world 0
	# 1000...1999 -> world 1
	# 2000...2999 -> world 2
	paths = get_values_sql(file=db_path, table='paths', rows=path_rows)
	path_images = compressed2img(img_cmp=paths.path_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
	start_images = compressed2img(img_cmp=paths.start_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)
	end_images = compressed2img(img_cmp=paths.end_img_cmp.values, n_voxels=n_voxels, n_dim=n_dim)

	q_paths = object2numeric_array(paths.q_path.values)
	q_paths = q_paths.reshape(-1, n_waypoints, n_dim)

	# Plot an example
	i = 0
	i_world = paths.i_world[path_rows[i]].item()

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()

	ax.imshow(obstacle_images[i_world].T, origin='lower', extent=extent, cmap='binary',)
	ax.imshow(start_images[i].T, origin='lower', extent=extent, cmap='Greens', alpha=0.4)
	ax.imshow(end_images[i].T, origin='lower', extent=extent, cmap='Reds', alpha=0.4)
	ax.imshow(path_images[i].T, origin='lower', extent=extent, cmap='Blues', alpha=0.2)

	ax.plot(*q_paths[i].T, color='k', marker='o')
	plt.show()


if __name__ == "__main__":
	# database_path = Path of SingleSphere02.db database file
	database_path = sys.argv[1]
	assert database_path.endswith(".db"), f"Invalid database file path detected! Received '{database_path}'"
	main(database_path)