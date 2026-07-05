"""
Geometric utilities for 3D Bounding Box Estimation.

Based on Mousavian et al. (CVPR 2017):
  "3D Bounding Box Estimation Using Deep Learning and Geometry"

Reference implementation: https://github.com/ruhyadi/YOLO3D
"""
import math
import torch
import numpy as np


def generate_bins(bins=2):
    """
    Generate MultiBin angle bin centers (in radians).

    Divides [0, 2π] into `bins` overlapping bins.
    Returns list of bin center angles.

    Args:
        bins: number of bins (default 2 → [0°, 180°])
    """
    interval = 2 * np.pi / bins
    angle_bins = []
    for i in range(bins):
        angle_bins.append(i * interval)
    return angle_bins  # e.g. [0.0, 3.14159...]


def theta_ray_from_bbox(box_2d_center_x, img_width, fx):
    """
    Compute the ray angle θ_ray from the camera to the object center.

    θ_ray = arctan((u - c_u) / f_u)

    Args:
        box_2d_center_x: x-coordinate of 2D box center (in image pixels)
        img_width: image width in pixels
        fx: camera focal length (pixels)

    Returns:
        theta_ray (float): ray angle in radians
    """
    fovx = 2 * np.arctan(img_width / (2 * fx))
    dx = box_2d_center_x - (img_width / 2)
    mult = 1.0 if dx >= 0 else -1.0
    dx = abs(dx)
    angle = np.arctan((2 * dx * np.tan(fovx / 2)) / img_width)
    return angle * mult


def calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray):
    """
    Solve for 3D object translation (X, Y, Z) using the geometric constraint
    that the projected 3D bounding box should tightly fit the 2D bounding box.

    Ported from YOLO3D's library/Math.py calc_location().

    Args:
        dimensions: (3,) [height, width, length] in meters (KITTI order)
        proj_matrix: (3, 4) projection matrix P = K[R|t]
        box_2d: [(xmin, ymin), (xmax, ymax)] in image pixels
        alpha: local observation angle (radians)
        theta_ray: ray angle to box center (radians)

    Returns:
        location: (3,) [X, Y, Z] in camera coordinates
    """
    # global orientation
    orient = alpha + theta_ray
    R = _rotation_matrix_np(orient)

    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    box_corners = [xmin, ymin, xmax, ymax]

    # KITTI dimensions: height, width, length
    dx = dimensions[2] / 2  # half length
    dy = dimensions[0] / 2  # half height
    dz = dimensions[1] / 2  # half width

    # Determine left/right multipliers based on alpha
    left_mult = 1
    right_mult = -1

    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # Build constraint sets (left, top, right, bottom)
    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    for i in (-1, 1):
        left_constraints.append([left_mult * dx, i * dy, -switch_mult * dz])
    for i in (-1, 1):
        right_constraints.append([right_mult * dx, i * dy, switch_mult * dz])
    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i * dx, -dy, j * dz])
    for i in (-1, 1):
        for j in (-1, 1):
            bottom_constraints.append([i * dx, dy, j * dz])

    # 64 combinations (4 × 4 × 2 × 2)
    constraints = []
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # Filter duplicates
    constraints = [c for c in constraints if len(c) == len(set(tuple(x) for x in c))]

    # Identity matrix base
    pre_M = np.eye(4, dtype=np.float64)

    best_loc = None
    best_error = [1e09]

    for constraint in constraints:
        Xa, Xb, Xc, Xd = constraint
        X_array = [Xa, Xb, Xc, Xd]

        Ma, Mb, Mc, Md = pre_M.copy(), pre_M.copy(), pre_M.copy(), pre_M.copy()
        M_array = [Ma, Mb, Mc, Md]

        A = np.zeros((4, 3), dtype=np.float64)
        b = np.zeros((4, 1), dtype=np.float64)

        indices = [0, 1, 0, 1]  # x, y, x, y
        for row, index in enumerate(indices):
            X = X_array[row]
            M = M_array[row]

            RX = np.dot(R, X)
            M[:3, 3] = RX.reshape(3)
            M = np.dot(proj_matrix, M)

            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        # Least squares solve
        try:
            loc, error, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if error.size > 0 and error[0] < best_error[0]:
            best_loc = loc
            best_error = error

    if best_loc is None:
        return np.array([0.0, 0.0, 10.0])

    return np.array([best_loc[0][0], best_loc[1][0], best_loc[2][0]])


def _rotation_matrix_np(yaw):
    """Rotation around Y axis (yaw)."""
    cos_t = np.cos(yaw)
    sin_t = np.sin(yaw)
    R = np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t],
    ])
    return R


# ============================================================================
# PyTorch batched version of orientation utilities
# ============================================================================


def generate_bins_tensor(bins=2, device=None):
    """Generate bin centers as torch tensor."""
    interval = 2 * torch.pi / bins
    return torch.arange(bins, device=device).float() * interval


def decode_orientation(orient_tensor, conf_tensor, angle_bins):
    """
    Decode MultiBin orientation predictions.

    Args:
        orient_tensor: (N, bins, 2) normalized sin/cos per bin
        conf_tensor:   (N, bins) confidence logits per bin
        angle_bins:    (bins,) bin center angles

    Returns:
        alpha: (N,) local observation angle in radians
    """
    argmax = conf_tensor.argmax(dim=1)  # (N,)
    orient_selected = orient_tensor[torch.arange(len(argmax)), argmax]  # (N, 2)
    sin_a = orient_selected[:, 0]
    cos_a = orient_selected[:, 1]
    alpha = torch.atan2(sin_a, cos_a)  # (N,)
    alpha = alpha + angle_bins[argmax]  # (N,)
    alpha = alpha - torch.pi  # shift back to [-π, π]
    return alpha


def theta_ray_from_bbox_tensor(box_2d_center_x, img_width, fx):
    """
    Batched PyTorch version of theta_ray computation.

    Args:
        box_2d_center_x: (N,) x-coordinate of 2D box centers
        img_width: scalar, image width
        fx: scalar, focal length

    Returns:
        theta_ray: (N,) ray angles
    """
    fovx = 2 * torch.arctan(img_width / (2 * fx))
    dx = box_2d_center_x - (img_width / 2)
    mult = torch.where(dx >= 0, torch.tensor(1.0, device=dx.device),
                       torch.tensor(-1.0, device=dx.device))
    angle = torch.arctan((2 * dx.abs() * torch.tan(fovx / 2)) / img_width)
    return angle * mult


# ============================================================================
# KITTI Class Dimension Averages (H, W, L in meters)
# Computed from KITTI training set — same concept as YOLO3D's ClassAverages
# ============================================================================

KITTI_DIM_AVERAGES = {
    "Car":        [1.5256, 1.6292, 3.8840],   # H, W, L
    "Pedestrian": [1.7552, 0.6352, 0.8359],
    "Cyclist":    [1.7219, 0.5994, 1.7632],
}

# class index → (H, W, L) average dimensions
# index 0 = Car, 1 = Pedestrian, 2 = Cyclist
KITTI_CLASS_DIMS = [
    KITTI_DIM_AVERAGES["Car"],         # [H=1.53, W=1.63, L=3.88]
    KITTI_DIM_AVERAGES["Pedestrian"],  # [H=1.76, W=0.64, L=0.84]
    KITTI_DIM_AVERAGES["Cyclist"],     # [H=1.72, W=0.60, L=1.76]
]
