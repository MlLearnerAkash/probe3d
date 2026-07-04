"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn


def depth_si_loss(depth_pr, depth_gt, alpha=10, lambda_scale=0.85, eps=1e-5):
    """
    Based on the loss proposed by Eigen et al (NeurIPS 2014). This differs from the
    implementation used by PixelFormer in that the sqrt is applied per image before
    mean as opposed to compute the mean loss before square-root.
    """
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    valid = (depth_gt > 0).detach().float()
    num_valid = valid.sum(dim=(-1, -2)).clamp(min=1)

    depth_pr = depth_pr.clamp(min=eps).log()
    depth_gt = depth_gt.clamp(min=eps).log()
    diff = (depth_pr - depth_gt) * valid
    diff_mean = diff.pow(2).sum(dim=(-2, -1)) / num_valid
    diff_var = diff.sum(dim=(-2, -1)).pow(2) / num_valid.pow(2)
    loss = alpha * (diff_mean - lambda_scale * diff_var).sqrt().mean()

    return loss


def sig_loss(depth_pr, depth_gt, sigma=0.85, eps=0.001, only_mean=False):
    """
    SigLoss
        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.
        adapated from DINOv2 code

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    # ignore invalid depth pixels
    valid = depth_gt > 0
    depth_pr = depth_pr[valid]
    depth_gt = depth_gt[valid]

    g = torch.log(depth_pr + eps) - torch.log(depth_gt + eps)

    loss = g.pow(2).mean() - sigma * g.mean().pow(2)
    loss = loss.sqrt()
    return loss


class DepthLoss(nn.Module):
    def __init__(self, weight_sig=10.0, weight_grad=0.5, max_depth=10):
        # TODO based on DINOv2 code
        super().__init__()
        self.sig_w = weight_sig
        self.grad_w = weight_grad
        self.max_depth = max_depth

    def forward(self, pred, target):
        # 0 out max depth so it gets ignored
        target[target > self.max_depth] = 0

        loss_s = self.sig_w * sig_loss(pred, target)
        loss_g = self.grad_w * gradient_loss(pred, target)
        return loss_s + loss_g


def gradient_loss(depth_pr, depth_gt, eps=0.001):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/ and DINOv2 repo

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    depth_pr_downscaled = [depth_pr] + [
        depth_pr[..., :: 2 * i, :: 2 * i] for i in range(1, 4)
    ]
    depth_gt_downscaled = [depth_gt] + [
        depth_gt[..., :: 2 * i, :: 2 * i] for i in range(1, 4)
    ]

    gradient_loss = 0
    for depth_pr, depth_gt in zip(depth_pr_downscaled, depth_gt_downscaled):

        # ignore invalid depth pixels
        valid = depth_gt > 0
        N = torch.sum(valid)

        depth_pr_log = torch.log(depth_pr + eps)
        depth_gt_log = torch.log(depth_gt + eps)
        log_d_diff = depth_pr_log - depth_gt_log

        log_d_diff = torch.mul(log_d_diff, valid)

        v_gradient = torch.abs(log_d_diff[..., 0:-2, :] - log_d_diff[..., 2:, :])
        v_valid = torch.mul(valid[..., 0:-2, :], valid[..., 2:, :])
        v_gradient = torch.mul(v_gradient, v_valid)

        h_gradient = torch.abs(log_d_diff[..., :, 0:-2] - log_d_diff[..., :, 2:])
        h_valid = torch.mul(valid[..., :, 0:-2], valid[..., :, 2:])
        h_gradient = torch.mul(h_gradient, h_valid)

        gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

    return gradient_loss


def angular_loss(snorm_pr, snorm_gt, mask, uncertainty_aware=False, eps=1e-4):
    """
    Angular loss with uncertainty aware component based on Bae et al.
    """
    # ensure mask is float and batch x height x width
    assert mask.ndim == 4, f"mask should be (batch x height x width) not {mask.shape}"
    mask = mask.squeeze(1).float()

    # compute correct loss
    if uncertainty_aware:
        assert snorm_pr.shape[1] == 4
        loss_ang = torch.cosine_similarity(snorm_pr[:, :3], snorm_gt, dim=1)
        loss_ang = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

        # apply elu and add 1.01 to have a min kappa of 0.01 (similar to paper)
        kappa = torch.nn.functional.elu(snorm_pr[:, 3]) + 1.01
        kappa_reg = (1 + (-kappa * torch.pi).exp()).log() - (kappa.pow(2) + 1).log()

        loss = kappa_reg + kappa * loss_ang
    else:
        assert snorm_pr.shape[1] == 3
        loss_ang = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
        loss = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

    # compute loss over valid position
    loss_mean = loss[mask.bool()].mean()
    if loss_mean != loss_mean:
        breakpoint()
    return loss_mean


def snorm_l1_loss(snorm_pr, snorm_gt, mask, eps=1e-4):
    """
    Angular loss with uncertainty aware component based on Bae et al.
    """
    # ensure mask is float and batch x height x width
    assert mask.ndim == 4, f"mask should be (batch x height x width) not {mask.shape}"
    mask = mask.squeeze(1).float()

    assert snorm_pr.shape[1] == 3
    loss = torch.nn.functional.l1_loss(snorm_pr, snorm_gt, reduction="none")
    loss = loss.mean(dim=1)

    # compute loss over valid position
    loss_mean = loss[mask.bool()].mean()
    if loss_mean != loss_mean:
        breakpoint()
    return loss_mean


# ============================================================================
# FCOS3D-style 3D Bounding Box Losses
# ============================================================================


def box3d_loss(
    pred_dense,
    gt_boxes,
    gt_labels,
    intrinsics,
    stride=16,
    loss_weights=None,
):
    """
    FCOS3D-style 3D box loss.

    Args:
        pred_dense:  (B, 10, H, W) raw dense predictions from Box3DHead
        gt_boxes:    list of (N_i, 7) tensors [X, Y, Z, W, H, L, θ] per image
        gt_labels:   list of (N_i,) class label tensors per image
        intrinsics:  (B, 3, 3) camera intrinsics
        stride:      feature stride
        loss_weights: dict with keys:
            'center':  weight for 2D center offset L1 loss (default 1.0)
            'depth':   weight for depth L1 loss (default 1.0)
            'dims':    weight for dimension L1 loss (default 1.0)
            'yaw':     weight for yaw loss (default 1.0)
            'corner':  weight for corner loss (default 0.5)
            'obj':     weight for objectness focal loss (default 1.0)

    Returns:
        total_loss, loss_dict
    """
    B, C, H, W = pred_dense.shape

    if loss_weights is None:
        loss_weights = {}

    w_center = loss_weights.get("center", 1.0)
    w_depth  = loss_weights.get("depth",  1.0)
    w_dims   = loss_weights.get("dims",   1.0)
    w_yaw    = loss_weights.get("yaw",    1.0)
    w_corner = loss_weights.get("corner", 0.5)
    w_obj    = loss_weights.get("obj",    1.0)

    # split channels
    obj_logits  = pred_dense[:, 0:1]    # (B, 1, H, W)
    offset_2d   = pred_dense[:, 1:3]    # (B, 2, H, W)
    depth_raw   = pred_dense[:, 3:4]    # (B, 1, H, W)
    dims_raw    = pred_dense[:, 4:7]    # (B, 3, H, W)
    yaw_sincos  = pred_dense[:, 7:9]    # (B, 2, H, W)
    dir_logits  = pred_dense[:, 9:10]   # (B, 1, H, W)

    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    # build grid
    device = pred_dense.device
    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    total_center_loss = torch.tensor(0.0, device=device)
    total_depth_loss  = torch.tensor(0.0, device=device)
    total_dims_loss   = torch.tensor(0.0, device=device)
    total_yaw_loss    = torch.tensor(0.0, device=device)
    total_corner_loss = torch.tensor(0.0, device=device)
    total_obj_loss    = torch.tensor(0.0, device=device)
    num_objects = 0

    for b in range(B):
        if len(gt_boxes[b]) == 0:
            # no objects: objectness loss only (all negative)
            target_obj = torch.zeros(1, H, W, device=device)
            total_obj_loss += focal_loss(
                obj_logits[b:b+1].sigmoid(), target_obj, alpha=0.25, gamma=2.0
            )
            continue

        boxes = gt_boxes[b]  # (N, 7): [X, Y, Z, W, H, L, θ]
        N = boxes.shape[0]

        for n in range(N):
            X_gt, Y_gt, Z_gt, W_gt, H_gt, L_gt, theta_gt = boxes[n]

            # --- project 3D center to image ---
            u_proj = fx[b] * X_gt / Z_gt + cx[b]
            v_proj = fy[b] * Y_gt / Z_gt + cy[b]

            # map to feature grid
            u_f = u_proj / stride
            v_f = v_proj / stride

            u_idx = int(u_f.floor().clamp(0, W - 1))
            v_idx = int(v_f.floor().clamp(0, H - 1))

            # --- compute target values at this grid location ---
            # offset: from grid cell center to projected center
            tgt_off_u = u_f - (u_idx + 0.5)
            tgt_off_v = v_f - (v_idx + 0.5)

            # depth
            tgt_depth = Z_gt.log()  # log-space depth

            # dimensions (log-space)
            tgt_dW = W_gt.log()
            tgt_dH = H_gt.log()
            tgt_dL = L_gt.log()

            # yaw: compute local observation angle α
            alpha_gt = theta_gt - torch.atan2(u_proj - cx[b], fx[b])

            # normalize alpha to [-π, π]
            alpha_gt = torch.atan2(
                torch.sin(alpha_gt), torch.cos(alpha_gt)
            )

            tgt_sin = torch.sin(alpha_gt)
            tgt_cos = torch.cos(alpha_gt)
            tgt_dir = (alpha_gt.abs() > torch.pi / 2).float()  # 0=front, 1=back

            # --- L1 losses at the assigned grid cell ---
            # center offset
            pred_off_u = offset_2d[b, 0, v_idx, u_idx]
            pred_off_v = offset_2d[b, 1, v_idx, u_idx]
            center_loss = (pred_off_u - tgt_off_u).abs() + (pred_off_v - tgt_off_v).abs()

            # depth
            pred_z = depth_raw[b, 0, v_idx, u_idx]
            depth_loss = (pred_z - tgt_depth).abs()

            # dimensions
            pred_dW = dims_raw[b, 0, v_idx, u_idx]
            pred_dH = dims_raw[b, 1, v_idx, u_idx]
            pred_dL = dims_raw[b, 2, v_idx, u_idx]
            dims_loss = (pred_dW - tgt_dW).abs() + (pred_dH - tgt_dH).abs() + (pred_dL - tgt_dL).abs()

            # yaw (sin/cos L1 + dir BCE)
            pred_sin = yaw_sincos[b, 0, v_idx, u_idx]
            pred_cos = yaw_sincos[b, 1, v_idx, u_idx]
            yaw_l1 = (pred_sin - tgt_sin).abs() + (pred_cos - tgt_cos).abs()
            dir_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                dir_logits[b, 0, v_idx, u_idx].unsqueeze(0),
                tgt_dir.unsqueeze(0),
            )
            yaw_loss = yaw_l1 + dir_bce

            # --- corner loss (project 8 corners and compute L1) ---
            # reconstruct predicted box at this location
            pred_Z = pred_z.exp().clamp(min=0.1)
            pred_X = (u_idx + 0.5 + pred_off_u) * stride
            pred_X = (pred_X - cx[b]) * pred_Z / fx[b]
            pred_Y = (v_idx + 0.5 + pred_off_v) * stride
            pred_Y = (pred_Y - cy[b]) * pred_Z / fy[b]

            pred_W = pred_dW.exp()
            pred_H = pred_dH.exp()
            pred_L = pred_dL.exp()

            pred_alpha = torch.atan2(pred_sin, pred_cos)
            pred_alpha = pred_alpha + (dir_logits[b, 0, v_idx, u_idx] > 0).float() * torch.pi
            pred_theta = pred_alpha + torch.atan2(
                (u_idx + 0.5 + pred_off_u) * stride - cx[b], fx[b]
            )

            corners_pred = _get_3d_corners(
                pred_X, pred_Y, pred_Z, pred_W, pred_H, pred_L, pred_theta
            )  # (8, 3)
            corners_gt = _get_3d_corners(
                X_gt, Y_gt, Z_gt, W_gt, H_gt, L_gt, theta_gt
            )  # (8, 3)

            # project to image
            corners_pred_2d = _project_points(corners_pred, intrinsics[b])  # (8, 2)
            corners_gt_2d   = _project_points(corners_gt,   intrinsics[b])  # (8, 2)

            corner_loss = (corners_pred_2d - corners_gt_2d).abs().sum() / 16.0

            total_center_loss += center_loss
            total_depth_loss  += depth_loss
            total_dims_loss   += dims_loss
            total_yaw_loss    += yaw_loss
            total_corner_loss += corner_loss
            num_objects += 1

        # --- objectness target (Gaussian around each projected center) ---
        target_obj = torch.zeros(H, W, device=device)
        for n in range(N):
            X_gt, Y_gt, Z_gt = boxes[n, 0], boxes[n, 1], boxes[n, 2]
            u_proj = fx[b] * X_gt / Z_gt + cx[b]
            v_proj = fy[b] * Y_gt / Z_gt + cy[b]
            u_f = u_proj / stride
            v_f = v_proj / stride

            sigma = 1.5  # Gaussian sigma in grid units
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    vi = int(v_f) + dy
                    ui = int(u_f) + dx
                    if 0 <= ui < W and 0 <= vi < H:
                        dist2 = (ui + 0.5 - u_f)**2 + (vi + 0.5 - v_f)**2
                        weight = torch.exp(-dist2 / (2 * sigma**2))
                        target_obj[vi, ui] = max(target_obj[vi, ui], weight)

        obj_loss = focal_loss(
            obj_logits[b:b+1].sigmoid(),
            target_obj.unsqueeze(0),
            alpha=0.25,
            gamma=2.0,
        )
        total_obj_loss += obj_loss

    # normalize by number of objects
    if num_objects > 0:
        total_center_loss = total_center_loss / num_objects
        total_depth_loss  = total_depth_loss  / num_objects
        total_dims_loss   = total_dims_loss   / num_objects
        total_yaw_loss    = total_yaw_loss    / num_objects
        total_corner_loss = total_corner_loss / num_objects

    total_obj_loss = total_obj_loss / B

    loss_dict = {
        "center": total_center_loss.item(),
        "depth":  total_depth_loss.item(),
        "dims":   total_dims_loss.item(),
        "yaw":    total_yaw_loss.item(),
        "corner": total_corner_loss.item(),
        "obj":    total_obj_loss.item(),
    }

    total_loss = (
        w_center * total_center_loss
        + w_depth * total_depth_loss
        + w_dims * total_dims_loss
        + w_yaw * total_yaw_loss
        + w_corner * total_corner_loss
        + w_obj * total_obj_loss
    )

    return total_loss, loss_dict


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for binary classification."""
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)

    pos_mask = (target > 0.5).float()
    neg_mask = (target <= 0.5).float()

    pos_loss = -alpha * pos_mask * (1 - pred).pow(gamma) * pred.log()
    neg_loss = -(1 - alpha) * neg_mask * pred.pow(gamma) * (1 - pred).log()

    # weight positive samples more
    num_pos = pos_mask.sum().clamp(min=1)
    num_neg = neg_mask.sum().clamp(min=1)

    return pos_loss.sum() / num_pos + neg_loss.sum() / num_neg


def _get_3d_corners(x, y, z, w, h, l, theta):
    """Get 8 corners of a 3D box in camera coordinates."""
    corners = torch.tensor([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], device=x.device, dtype=x.dtype)  # (8, 3)

    corners = corners * torch.tensor([l / 2, h / 2, w / 2], device=x.device, dtype=x.dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.tensor([
        [ cos_t, 0, sin_t],
        [ 0,     1, 0    ],
        [-sin_t, 0, cos_t],
    ], device=x.device, dtype=x.dtype)

    corners = corners @ R.T  # (8, 3)
    corners = corners + torch.stack([x, y, z])

    return corners


def _project_points(points_3d, K):
    """Project 3D points to 2D image plane."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2].clamp(min=1e-6)

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    return torch.stack([u, v], dim=-1)
