from dataclasses import dataclass

import numpy as np
from .grid import Grid

def cylinder(grid: Grid, center, axes, radius):
    """Creates an axis align cylinder implicit surface function

    Args:
        grid (Grid): Grid object
        ignore_dims (List): List specifing axis where cylinder is aligned (0-indexed)
        center (List): List specifying the center of cylinder
        radius (float): Radius of cylinder

    Returns:
        np.ndarray: implicit surface function of the cylinder
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndim):
        if i not in axes:
            # This works because of broadcasting
            data = data + np.power(grid.states[..., i] - center[i], 2)
    data = np.sqrt(data) - radius
    return data


def rectangle(grid: Grid, target_min, target_max):
    data = np.maximum(+grid.states[..., 0] - target_max[0],
                      -grid.states[..., 0] + target_min[0])

    for i in range(grid.ndim):
        data = np.maximum(data,  grid.states[..., i] - target_max[i])
        data = np.maximum(data, -grid.states[..., i] + target_min[i])

    return data

def lower_half_space(grid: Grid, axis, value):
    """Creates an axis aligned lower half space 

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V < value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndim):
        if i == axis:
            data += grid.states[..., i] - value
    return data


def upper_half_space(grid: Grid, axis, value):
    """Creates an axis aligned upper half space 

    Args:
        grid (Grid): Grid object
        axis (int): Dimension of the half space (0-indexed)
        value (float): Used in the implicit surface function for V > value

    Returns:
        np.ndarray: implicit surface function of the lower half space
                    of size grid.pts_each_dim
    """
    data = np.zeros(grid.shape)
    for i in range(grid.ndim):
        if i == axis:
            data += -grid.states[..., i] + value
    return data


def union(shape, *shapes):
    """ Calculates the union of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    result = shape 
    for shape in shapes: 
        result = np.minimum(result, shape)
    return result


def intersection(shape, *shapes):
    """ Calculates the intersection of two shapes

    Args:
        shape1 (np.ndarray): implicit surface representation of a shape
        shape2 (np.ndarray): implicit surface representation of a shape

    Returns:
        np.ndarray: the element-wise minimum of two shapes
    """
    result = shape
    for shape in shapes:
        result = np.maximum(result, shape)
    return result

def setminus(a, *bs):
    result = a
    for b in bs:
        result = np.maximum(result, -b)
    return result

def make_tube(times, vf):
    return np.concatenate([vf[np.newaxis, ...]] * len(times))

def project_onto(vf, *idxs, keepdims=False):
    idxs = [len(vf.shape) + i if i < 0 else i for i in idxs]
    dims = [i for i in range(len(vf.shape)) if i not in idxs]
    return vf.min(axis=tuple(dims), keepdims=keepdims)

def is_invariant(grid, times, a):
    return a is None or len(a.shape) != len(times.shape + grid.shape)