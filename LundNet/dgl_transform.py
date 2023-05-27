"""Module for graph transformation utilities."""

from collections.abc import Iterable, Mapping
from collections import defaultdict
import numpy as np
from scipy import sparse


from dgl.backend import backend as F


def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + F.swapaxes(x2s, -1, -2) - 2 * x @ F.swapaxes(x, -1, -2)
