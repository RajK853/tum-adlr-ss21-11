import os
import cv2
import yaml
import numpy as np


def resize_imgs(imgs, size, dtype=np.bool_):
    N = imgs.shape[0]
    new_imgs = np.zeros((N, *size), dtype=dtype)
    for i in range(N):
        new_imgs[i] = cv2.resize(imgs[i].astype("uint8"), size)
    return new_imgs


def pipe_funcs(data, funcs):
    for func in funcs:
        data = func(data)
    return data


def load_files(dir_path, file_types=None):
    """
    Recursively searches and returns the file paths of given types located in the given directory path
    :param dir_path: (str) Root directory path to search
    :param file_types: (iter) Collection of file extension to search. Defaults to None
    :returns: (list) List of found file paths 
    """
    
    def has_type(file_name, types):
        if types is None:
            return True
        _, file_type = os.path.splitext(file_name)
        return file_type in types

    file_paths = []
    for root_dir, _, files in os.walk(dir_path):
        if files:
            matched_files = [os.path.join(root_dir, file_path) for file_path in files if has_type(file_path, file_types)]
            file_paths.extend(matched_files)
    return file_paths


def load_yaml(file_path, safe_load=True):
    """
    Loads a YAML file from the given path
    :param file_path: (str) YAML file path
    :param safe_load: (bool) If True, uses yaml.safe_load() instead of yaml.load()
    :returns: (dict) Loaded YAML file as a dictionary
    """
    load_func = yaml.safe_load if safe_load else yaml.load
    with open(file_path, "r") as fp:
        return load_func(fp)


def exec_from_yaml(config_path, exec_func, title="Experiment", safe_load=True, skip_prefix="ignore"):
    """
    Executes the given function by loading parameters from a YAML file with given structure:
    NOTE: The argument names in the YAML file should match the argument names of the given execution function.
    
    :Example:

    - Experiment_1 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    - Experiment_2 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    
    :param config_path: (str) YAML file path
    :param exec_func: (callable) Function to execute with the loaded parameters
    :param title: (str) Label for each experiment
    :param safe_load: (bool) When True, uses yaml.safe_load. Otherwise uses yaml.load
    :param skip_prefix: (str) Experiment names with given prefix will not be executed 
    :returns: (dict) Dictionary with results received from each experiment execution
    """
    # Process YAML file names
    if os.path.isdir(config_path):
        yaml_paths = load_files(config_path, file_types=[".yaml"])
        assert len(yaml_paths) > 0, f"No YAML config files found in '{config_path}'"
    elif config_path.endswith(".yaml"):
        yaml_paths = [config_path]
    else:
        raise TypeError("Invalid config_path value. Must be either a path to a .yaml file or a path to directory with .yaml files!")
    
    i = 1
    result_dict = {}
    for yaml_path in yaml_paths:
        config_dict = load_yaml(yaml_path, safe_load=safe_load)
        for exp_name, exp_kwargs in config_dict.items():
            if exp_name.lower().startswith(skip_prefix):
                print(f"# Skipped {exp_name}")
                continue
            print(f"\n{i}. {title}: {exp_name}")
            # Execute the exec_func function by unpacking the experiment's keyword-arguments
            result = exec_func(**exp_kwargs)
            result_dict[exp_name] = result
            i += 1
    return result_dict
