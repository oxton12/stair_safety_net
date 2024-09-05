import numpy as np
import mxnet as mx
import os
from pathlib import Path


def get_data_paths(path, context):
    labeled_paths = {}
    for root, dirs, files in os.walk(path):
        parent = Path(root).name
        action = mx.nd.array([[0, 1]]) if parent == "not_holding" else mx.nd.array([[1, 0]])
        mx_action = mx.nd.array(action).as_in_context(context)
        for file in files:
            labeled_paths[os.path.join(path, parent, file)] = mx_action
    return labeled_paths


def get_mean_std(data_paths):
    means = np.empty((0, 17))
    stds = np.empty((0, 17))

    for data_path in data_paths.keys():
        data = mx.nd.load(data_path)[0]
        for sample in data:
            sample_np = sample.asnumpy()
            sample_mean = np.expand_dims(sample_np.mean(axis=(1,2)), 0)
            sample_std = np.expand_dims(sample_np.std(axis=(1,2)), 0)
            means = np.concatenate((means, sample_mean), axis=0)
            stds = np.concatenate((stds, sample_std), axis=0)

    return list(means.mean(axis=0)), list(stds.mean(axis=0))
