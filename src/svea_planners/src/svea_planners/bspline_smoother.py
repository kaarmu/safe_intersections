#! /usr/bin/env python3

import numpy as np
import scipy.interpolate as interpolate


class BSpline(object):
    """
    PathSmoother class, based on PythonRobotics Package
    """

    def __init__(self, path):
        """
        Init method for class PathSmoother

        :param path: path to be smoothed
        :type path: list[tuple[float]]
        """
        self.path = np.array(path)
        self.n_points = np.shape(self.path)[0]

    def approximate_b_spline_path(self, degree: int = 3, s=None, ) -> tuple:
        """
        Approximate points with a B-Spline path

        :param degree: B Spline curve degree. Must be 2<= k <= 5, defaults to 3
        :type degree: int, optional
        :param s: smoothing parameter. If this value is bigger, the path will be smoother, but it will be less accurate.
        If this value is smaller, the path will be more accurate, but it will be less smooth. When `s` is 0, it is 
        equivalent to the interpolation. Default is None, in this case `s` will be `len(x)`, defaults to None
        :type s: int, optional
        :return: x, y, heading, curvature of resultant path
        :rtype: tuple
        """
        distances = self._calc_distance_vector(self.path[:, 0], self.path[:, 1])

        spl_i_x = interpolate.UnivariateSpline(distances, self.path[:, 0], k=degree, s=s)
        spl_i_y = interpolate.UnivariateSpline(distances, self.path[:, 1], k=degree, s=s)

        sampled = np.linspace(0.0, distances[-1], self.n_points)
        return self._evaluate_spline(sampled, spl_i_x, spl_i_y)
    
    def interpolate_b_spline_path(self, degree: int = 3) -> tuple:
        """
        Interpolate x-y points with a B-Spline path

        :param degree: B Spline curve degree. Must be 2<= k <= 5, defaults to 3
        :type degree: int, optional
        :param s: smoothing parameter. If this value is bigger, the path will be smoother, but it will be less accurate.
        If this value is smaller, the path will be more accurate, but it will be less smooth. When `s` is 0, it is 
        equivalent to the interpolation. Default is None, in this case `s` will be `len(x)`, defaults to None
        :type s: int, optional
        :return: x, y, heading, curvature of resultant path
        :rtype: tuple
        """
        return self.approximate_b_spline_path(degree, s=0.0)
    
    def _calc_distance_vector(self, x, y):
        dx, dy = np.diff(x), np.diff(y)
        distances = np.cumsum([np.hypot(idx, idy) for idx, idy in zip(dx, dy)])
        distances = np.concatenate(([0.0], distances))
        distances /= distances[-1]
        return distances
    
    def _evaluate_spline(self, sampled, spl_i_x, spl_i_y):
        x = spl_i_x(sampled)
        y = spl_i_y(sampled)
        dx = spl_i_x.derivative(1)(sampled)
        dy = spl_i_y.derivative(1)(sampled)
        heading = np.arctan2(dy, dx)
        ddx = spl_i_x.derivative(2)(sampled)
        ddy = spl_i_y.derivative(2)(sampled)
        curvature = (ddy * dx - ddx * dy) / np.power(dx * dx + dy * dy, 2.0 / 3.0)
        return np.array(x), y, heading, curvature,

