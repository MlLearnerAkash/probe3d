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
from loguru import logger


def depth_rmse(depth_pr, depth_gt, image_average=False):
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    # compute RMSE for each image and then average
    valid = (depth_gt > 0).detach().float()

    # clamp to 1 for empty depth images
    num_valid = valid.sum(dim=(1, 2))
    if (num_valid == 0).any():
        num_valid = num_valid.clamp(min=1)
        logger.warning("GT depth is empty. Clamping to avoid error.")

    # compute pixelwise squared error
    sq_error = (depth_gt - depth_pr).pow(2)
    sum_masked_sqe = (sq_error * valid).sum(dim=(1, 2))
    rmse_image = (sum_masked_sqe / num_valid).sqrt()

    return rmse_image.mean() if image_average else rmse_image


def evaluate_depth(
    depth_pr, depth_gt, image_average=False, scale_invariant=False, nyu_crop=False
):
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    if nyu_crop:
        # apply NYU crop --- commonly used in many repos for some reason
        assert depth_pr.shape[-2] == 480
        assert depth_pr.shape[-1] == 640
        depth_pr = depth_pr[..., 45:471, 41:601]
        depth_gt = depth_gt[..., 45:471, 41:601]

    if scale_invariant:
        depth_pr = match_scale_and_shift(depth_pr, depth_gt)

    # zero out invalid pixels
    valid = (depth_gt > 0).detach().float()
    depth_pr = depth_pr * valid

    # get num valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # get recall @ thresholds
    thresh = torch.maximum(
        depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
    )
    d1 = ((thresh < 1.25 ** 1).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((thresh < 1.25 ** 2).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((thresh < 1.25 ** 3).float() * valid).sum(dim=(1, 2)) / num_valid

    # compute RMSE
    sse = (depth_gt - depth_pr).pow(2)
    mse = (sse * valid).sum(dim=(1, 2)) / num_valid
    rmse = mse.sqrt()
    metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def evaluate_surface_norm(snorm_pr, snorm_gt, valid, image_average=False):
    """
    Metrics to evaluate surface norm based on iDISC (and probably Fouhey et al. 2016).
    """
    snorm_pr = snorm_pr[:, :3]
    assert snorm_pr.shape == snorm_gt.shape, f"{snorm_pr.shape} != {snorm_gt.shape}"

    # compute angular error
    cos_sim = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
    cos_sim = cos_sim.clamp(min=-1, max=1.0)
    err_deg = torch.acos(cos_sim) * 180.0 / torch.pi

    # zero out invalid errors
    assert len(valid.shape) == 4
    valid = valid.squeeze(1).float()
    err_deg = err_deg * valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # compute rmse
    rmse = (err_deg.pow(2).sum(dim=(1, 2)) / num_valid).sqrt()

    # compute recall at thresholds
    thresh = [11.25, 22.5, 30]
    d1 = ((err_deg < thresh[0]).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((err_deg < thresh[1]).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((err_deg < thresh[2]).float() * valid).sum(dim=(1, 2)) / num_valid

    metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def match_scale_and_shift(prediction, target):
    # based on implementation from
    # https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

    assert len(target.shape) == len(prediction.shape)
    if len(target.shape) == 4:
        four_chan = True
        target = target.squeeze(dim=1)
        prediction = prediction.squeeze(dim=1)
    else:
        four_chan = False

    mask = (target > 0).float()

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 *
    # a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    # compute scale and shift
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    scale = scale.view(-1, 1, 1).detach()
    shift = shift.view(-1, 1, 1).detach()
    prediction = prediction * scale + shift

    return prediction[:, None, :, :] if four_chan else prediction


# ============================================================================
# 3D Bounding Box Metrics
# ============================================================================


def evaluate_box3d(pred_boxes, pred_scores, gt_boxes, iou_thresholds=None):
    """
    Compute 3D detection metrics: AP, translation error, dimension error, yaw error.

    Args:
        pred_boxes:  (N_pred, 7) tensor [X, Y, Z, W, H, L, θ]
        pred_scores: (N_pred,) confidence scores
        gt_boxes:    (N_gt, 7) ground truth boxes
        iou_thresholds: list of IoU thresholds for AP (default [0.5, 0.7])

    Returns:
        dict with: ap_3d_xx, center_err, dim_err, yaw_err
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.7]

    N_pred = pred_boxes.shape[0]
    N_gt = gt_boxes.shape[0]

    if N_pred == 0 or N_gt == 0:
        metrics = {}
        for t in iou_thresholds:
            metrics[f"ap_3d_{t}"] = 0.0
        metrics["center_err"] = 0.0
        metrics["dim_err"] = 0.0
        metrics["yaw_err"] = 0.0
        return metrics

    # sort by score descending
    sorted_idx = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    # compute IoU matrix
    iou_matrix = _compute_iou3d(pred_boxes, gt_boxes)  # (N_pred, N_gt)

    matched_gt = set()
    tp = {t: torch.zeros(N_pred, device=pred_boxes.device) for t in iou_thresholds}
    fp = {t: torch.zeros(N_pred, device=pred_boxes.device) for t in iou_thresholds}

    for i in range(N_pred):
        best_iou, best_j = iou_matrix[i].max(dim=0)
        for t in iou_thresholds:
            if best_iou >= t and best_j.item() not in matched_gt:
                tp[t][i] = 1
                matched_gt.add(best_j.item())
            else:
                fp[t][i] = 1

    metrics = {}
    for t in iou_thresholds:
        tp_cum = tp[t].cumsum(dim=0)
        fp_cum = fp[t].cumsum(dim=0)
        recall = tp_cum / max(N_gt, 1)
        precision = tp_cum / (tp_cum + fp_cum).clamp(min=1)

        # 11-point interpolated AP
        ap = 0.0
        for r in torch.linspace(0, 1, 11, device=pred_boxes.device):
            mask = recall >= r
            if mask.any():
                ap += precision[mask].max() / 11.0
        metrics[f"ap_3d_{t}"] = ap.item()

    # best-match errors
    if N_gt > 0:
        best_iou, best_match = iou_matrix.max(dim=0)  # (N_gt,)
        # only use matches with IoU > 0.1
        valid = best_iou > 0.1

        if valid.any():
            matched_pred = pred_boxes[best_match[valid]]
            matched_gt   = gt_boxes[valid]

            center_err = (matched_pred[:, :3] - matched_gt[:, :3]).norm(dim=1).mean()
            dim_err    = (matched_pred[:, 3:6] - matched_gt[:, 3:6]).abs().mean()
            yaw_diff   = matched_pred[:, 6] - matched_gt[:, 6]
            yaw_diff   = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            yaw_err    = yaw_diff.abs().mean()

            metrics["center_err"] = center_err.item()
            metrics["dim_err"]    = dim_err.item()
            metrics["yaw_err"]    = yaw_err.item()
        else:
            metrics["center_err"] = float("nan")
            metrics["dim_err"]    = float("nan")
            metrics["yaw_err"]    = float("nan")
    else:
        metrics["center_err"] = float("nan")
        metrics["dim_err"]    = float("nan")
        metrics["yaw_err"]    = float("nan")

    return metrics


def _compute_iou3d(boxes_a, boxes_b):
    """
    Compute BEV (bird's-eye-view) IoU between two sets of 3D boxes.
    Approximates 3D IoU as BEV IoU × min height ratio.

    Args:
        boxes_a: (M, 7) [X, Y, Z, W, H, L, θ]
        boxes_b: (N, 7)

    Returns:
        iou: (M, N) tensor
    """
    M, N = boxes_a.shape[0], boxes_b.shape[0]
    device = boxes_a.device

    iou = torch.zeros(M, N, device=device)

    for i in range(M):
        for j in range(N):
            iou[i, j] = _bev_iou_single(
                boxes_a[i, 0], boxes_a[i, 2], boxes_a[i, 3], boxes_a[i, 5], boxes_a[i, 6],  # X, Z, W, L, θ
                boxes_b[j, 0], boxes_b[j, 2], boxes_b[j, 3], boxes_b[j, 5], boxes_b[j, 6],
            )

    return iou


def _bev_iou_single(x1, z1, w1, l1, yaw1, x2, z2, w2, l2, yaw2):
    """
    Compute BEV IoU between two oriented 2D boxes (X-Z plane).
    Uses triangle-area-based intersection (no shapely dependency).
    """
    import math

    # get corners in BEV (X, Z)
    def get_bev_corners(x, z, w, l, yaw):
        cos_t = math.cos(yaw)
        sin_t = math.sin(yaw)
        # corners: front-left, front-right, back-right, back-left
        corners_local = [
            (-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)
        ]
        corners = []
        for dx, dz in corners_local:
            cx = x + dx * cos_t - dz * sin_t
            cz = z + dx * sin_t + dz * cos_t
            corners.append((cx, cz))
        return corners

    corners1 = get_bev_corners(x1, z1, w1, l1, yaw1)
    corners2 = get_bev_corners(x2, z2, w2, l2, yaw2)

    area1 = w1 * l1
    area2 = w2 * l2

    # convex polygon intersection via Sutherland-Hodgman
    def poly_area(poly):
        if len(poly) < 3:
            return 0.0
        a = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            a += x1 * y2 - x2 * y1
        return abs(a) / 2.0

    def clip_polygon(subject, clip):
        def inside(p, cp1, cp2):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

        def intersection(s, e, cp1, cp2):
            dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
            dp = (s[0] - e[0], s[1] - e[1])
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0] + 1e-10)
            x = (n1 * dp[0] - n2 * dc[0]) * n3
            y = (n1 * dp[1] - n2 * dc[1]) * n3
            return (x, y)

        output = subject
        for i in range(len(clip)):
            cp1 = clip[i]
            cp2 = clip[(i + 1) % len(clip)]
            input_list = output
            output = []
            if len(input_list) == 0:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        output.append(intersection(s, e, cp1, cp2))
                    output.append(e)
                elif inside(s, cp1, cp2):
                    output.append(intersection(s, e, cp1, cp2))
                s = e
        return output

    inter_poly = clip_polygon(corners1, corners2)
    inter_area = poly_area(inter_poly)

    union_area = area1 + area2 - inter_area
    if union_area < 1e-6:
        return torch.tensor(0.0, device=x1.device if isinstance(x1, torch.Tensor) else torch.device("cpu"))

    return torch.tensor(inter_area / union_area)
