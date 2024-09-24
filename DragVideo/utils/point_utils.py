import numpy as np
import torch

# This function aims to transform the points in normal video, into the encoded super resolution data's format. (e.g.,
# one video with (512, 512) may have super resolution with (256, 256)
#
# The points input should follow the format (n, 2) or (f, n, 2) or (f, n, k, 2). First 2 means handle/target point, second 2 is h and w location
def process_points_to_sup_res(points,
                              res_shape,
                              sup_res_shape):
    full_h, full_w = res_shape
    sup_h, sup_w = sup_res_shape
    if len(points.shape) == 4:
        points_1 = (points[:, :, :, 0] / full_w) * sup_w
        points_0 = (points[:, :, :, 1] / full_h) * sup_h
        points[:, :, :, 0] = points_0
        points[:, :, :, 1] = points_1
    elif len(points.shape) == 3:
        points_1 = (points[:, :, 0] / full_w) * sup_w
        points_0 = (points[:, :, 1] / full_h) * sup_h
        points[:, :, 0] = points_0
        points[:, :, 1] = points_1
    elif len(points.shape) == 2:
        points_1 = (points[:, 0] / full_w) * sup_w
        points_0 = (points[:, 1] / full_h) * sup_h
        points[:, 0] = points_0
        points[:, 1] = points_1
    else:
        raise ValueError("The input points shape should be either (n, 2) or (f, n, 2) or (f, n, k, 2)")
    return torch.round(points)

# The input poits of this function should be (n, 2) or (f, n, 2) or (f, n, k, 2) it should process handle/target points separately
def reconstruct_points_from_sup_res(points,
                                    res_shape,
                                    sup_res_shape):
    full_h, full_w = res_shape
    sup_h, sup_w = sup_res_shape

    if len(points.shape) == 3:
        points_0 = (points[:, :, 1] / sup_w) * full_w
        points_1 = (points[:, :, 0] / sup_h) * full_h

        points[:, :, 0] = points_0
        points[:, :, 1] = points_1
    elif len(points.shape) == 4:
        points_0 = (points[:, :, :, 1] / sup_w) * full_w
        points_1 = (points[:, :, :, 0] / sup_h) * full_h

        points[:, :, :, 0] = points_0
        points[:, :, :, 1] = points_1
    elif len(points.shape) == 2:
        points_0 = (points[:, 1] / sup_w) * full_w
        points_1 = (points[:, 0] / sup_h) * full_h

        points[:, 0] = points_0
        points[:, 1] = points_1
    else:
        raise ValueError("The input points shape should be either (n, 2) or (f, n, 2) or (f, n, k, 2)")
    return torch.round(points)

