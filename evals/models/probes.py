import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F

class SurfaceNormalHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        output_dim = 4 if uncertainty_aware else 3

        self.kernel_size = kernel_size

        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        return self.head(feats)


class DepthHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        min_depth=0.001,
        max_depth=10,
        prediction_type="bindepth",
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.name = f"{prediction_type}_{head_type}_k{kernel_size}"

        if prediction_type == "bindepth":
            output_dim = 256
            self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)
        elif prediction_type == "sigdepth":
            output_dim = 1
            self.predict = DepthSigmoidPrediction(min_depth, max_depth)
        else:
            raise ValueError()

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        """Prediction each pixel."""
        feats = self.head(feats)
        depth = self.predict(feats)
        return depth


class DepthBinPrediction(nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=10,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_strategy = norm_strategy
        self.bins_strategy = bins_strategy

    def forward(self, prob):
        if self.bins_strategy == "UD":
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )
        elif self.bins_strategy == "SID":
            bins = torch.logspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            prob = torch.relu(prob)
            eps = 0.1
            prob = prob + eps
            prob = prob / prob.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            prob = torch.softmax(prob, dim=1)
        elif self.norm_strategy == "sigmoid":
            prob = torch.sigmoid(prob)
            prob = prob / prob.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikhw,k->ihw", [prob, bins])
        depth = depth.unsqueeze(dim=1)
        return depth


class DepthSigmoidPrediction(nn.Module):
    def __init__(self, min_depth=0.001, max_depth=10):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred):
        depth = pred.sigmoid()
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        return depth


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4

        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [
            interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            for x in feats
        ]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        out = interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        out = self.out_conv(out)
        out = interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size)
    else:
        assert num_layers > 1
        modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
        conv = nn.Sequential(*modules)

    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear", align_corners=True)
        return self.conv(feats)


class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [
            interpolate(feat, (h, w), mode="bilinear", align_corners=True)
            for feat in feats
        ]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear", align_corners=True)
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear", align_corners=True)
        return self.conv_out(feats)


# ============================================================================
# Detection Probe: Cube R-CNN / Omni3D style
# Frozen backbone → Feature Extractor (DPT/Multiscale/Linear) →
# Feature Pyramid → RPN → 2D Head + Cube Head
# ============================================================================

class FeaturePyramid(nn.Module):
    """Build an FPN from a single dense feature map.

    Takes a feature map at backbone stride and builds P2–P5 pyramid levels
    via strided downsampling + top-down lateral fusion.
    """

    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        # bottom-up pathway: P2 (input stride) → P3 → P4 → P5
        self.p2_conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.p3_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.p4_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.p5_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1)

        # top-down laterals
        self.lat_p5 = nn.Conv2d(out_dim, out_dim, 1)
        self.lat_p4 = nn.Conv2d(out_dim, out_dim, 1)
        self.lat_p3 = nn.Conv2d(out_dim, out_dim, 1)
        self.lat_p2 = nn.Conv2d(out_dim, out_dim, 1)

        # post-merge smoothing
        self.smooth_p5 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def forward(self, x):
        """
        Args:
            x: (B, in_dim, H, W) dense features at backbone stride

        Returns:
            list of [P2, P3, P4, P5] each (B, out_dim, H_i, W_i)
                P2: same as input
                P3: 2x downsampled
                P4: 4x downsampled
                P5: 8x downsampled
        """
        # bottom-up
        p2 = self.p2_conv(x).relu()   # stride = input_stride
        p3 = self.p3_conv(F.max_pool2d(p2, 2, 2)).relu()
        p4 = self.p4_conv(F.max_pool2d(p3, 2, 2)).relu()
        p5 = self.p5_conv(F.max_pool2d(p4, 2, 2)).relu()

        # top-down with lateral
        p5_out = self.smooth_p5(self.lat_p5(p5))
        p4_out = self.smooth_p4(self.lat_p4(p4) + interpolate(p5_out, scale_factor=2, mode="nearest"))
        p3_out = self.smooth_p3(self.lat_p3(p3) + interpolate(p4_out, scale_factor=2, mode="nearest"))
        p2_out = self.smooth_p2(self.lat_p2(p2) + interpolate(p3_out, scale_factor=2, mode="nearest"))

        return [p2_out, p3_out, p4_out, p5_out]


class RPNHead(nn.Module):
    """Lightweight Region Proposal Network.

    Shares convs across all FPN levels, predicts objectness + bbox deltas
    for a single anchor scale per level.
    """

    def __init__(self, in_dim=256, num_anchors=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # fixed anchor sizes per pyramid level (area in pixels at input resolution)
        self.anchor_areas = [32 * 32, 64 * 64, 128 * 128, 256 * 256]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = num_anchors * len(self.anchor_ratios)

        self.obj_pred = nn.Conv2d(in_dim, self.num_anchors, 1)       # objectness logits
        self.box_pred = nn.Conv2d(in_dim, self.num_anchors * 4, 1)   # (dx, dy, dw, dh)

    def forward(self, feats):
        """Return objectness logits and bbox deltas per pyramid level."""
        obj_logits, box_deltas = [], []
        for feat in feats:
            f = self.conv(feat)
            obj_logits.append(self.obj_pred(f))
            box_deltas.append(self.box_pred(f))
        return obj_logits, box_deltas

    def generate_anchors(self, feats, image_size, feat_strides):
        """
        Generate anchor boxes for all pyramid levels.

        Args:
            feats: list of [P2, P3, P4, P5] feature maps
            image_size: (H, W) input image size
            feat_strides: list of strides for each level relative to image

        Returns:
            anchors: (N_total, 4) [x1, y1, x2, y2] in image coordinates
        """
        device = feats[0].device
        anchors = []
        for lvl, (feat, stride, area) in enumerate(zip(feats, feat_strides, self.anchor_areas)):
            _, _, H_f, W_f = feat.shape
            ys = torch.arange(H_f, device=device).float() * stride + stride / 2.0
            xs = torch.arange(W_f, device=device).float() * stride + stride / 2.0
            y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

            for ratio in self.anchor_ratios:
                w = (area / ratio) ** 0.5
                h = w * ratio
                x1 = x_grid - w / 2
                y1 = y_grid - h / 2
                x2 = x_grid + w / 2
                y2 = y_grid + h / 2
                anchors.append(torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4))

        return torch.cat(anchors, dim=0)

    def decode_proposals(self, obj_logits, box_deltas, feats,
                         image_size, feat_strides,
                         pre_nms_topk=1000, post_nms_topk=100,
                         nms_thresh=0.7, score_thresh=0.05):
        """Decode RPN outputs into proposal boxes via NMS."""
        device = feats[0].device
        img_h, img_w = image_size
        proposals, scores_all = [], []

        for lvl, (obj, delta, feat, stride) in enumerate(
            zip(obj_logits, box_deltas, feats, feat_strides)
        ):
            _, _, H_f, W_f = feat.shape
            obj_scores = obj.sigmoid().permute(0, 2, 3, 1).reshape(H_f, W_f, -1)  # (H_f, W_f, nA)
            deltas = delta.permute(0, 2, 3, 1).reshape(H_f, W_f, -1, 4)  # (H_f, W_f, nA, 4)

            ys = torch.arange(H_f, device=device).float() * stride + stride / 2.0
            xs = torch.arange(W_f, device=device).float() * stride + stride / 2.0
            y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

            for a, ratio in enumerate(self.anchor_ratios):
                area = self.anchor_areas[lvl]
                aw = (area / ratio) ** 0.5
                ah = aw * ratio
                dx = deltas[:, :, a, 0]
                dy = deltas[:, :, a, 1]
                dw = deltas[:, :, a, 2]
                dh = deltas[:, :, a, 3]
                cx = x_grid + dx * aw
                cy = y_grid + dy * ah
                w = aw * dw.exp()
                h = ah * dh.exp()

                x1 = (cx - w / 2).clamp(0, img_w)
                y1 = (cy - h / 2).clamp(0, img_h)
                x2 = (cx + w / 2).clamp(0, img_w)
                y2 = (cy + h / 2).clamp(0, img_h)

                valid = (x2 > x1) & (y2 > y1)
                proposals.append(torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)[valid.reshape(-1)])
                scores_all.append(obj_scores[:, :, a].reshape(-1)[valid.reshape(-1)])

        if len(proposals) == 0:
            return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

        proposals = torch.cat(proposals, dim=0)
        scores_all = torch.cat(scores_all, dim=0)

        # pre-NMS top-k
        keep = scores_all.argsort(descending=True)[:pre_nms_topk]
        proposals = proposals[keep]
        scores_all = scores_all[keep]

        # NMS
        keep = torchvision_nms(proposals, scores_all, nms_thresh)
        keep = keep[:post_nms_topk]

        return proposals[keep], scores_all[keep]

    def match_anchors_to_gt(self, anchors, gt_boxes_list, pos_iou=0.7, neg_iou=0.3):
        """Match anchors to ground-truth boxes. Returns labels and regression targets."""
        device = anchors.device
        N = len(anchors)
        # flatten GT boxes across batch
        gt_all = []
        batch_indices = []
        for b, boxes in enumerate(gt_boxes_list):
            if boxes.numel() > 0:
                gt_all.append(boxes)
                batch_indices.extend([b] * len(boxes))

        if len(gt_all) == 0:
            labels = torch.zeros(N, device=device, dtype=torch.long)
            bbox_targets = torch.zeros(N, 4, device=device)
            return labels, bbox_targets

        gt_all = torch.cat(gt_all, dim=0)
        batch_indices_t = torch.tensor(batch_indices, device=device)

        # compute IoU between anchors and all GT boxes
        # anchors: (N, 4) [x1, y1, x2, y2]; gt_all: (M, 4)
        ious = box_iou(anchors, gt_all)  # (N, M)
        max_iou, max_idx = ious.max(dim=1)  # (N,)

        labels = torch.zeros(N, device=device, dtype=torch.long)
        labels[max_iou >= pos_iou] = 1

        # also mark the best anchor per GT as positive
        best_anchor_per_gt = ious.argmax(dim=0)
        labels[best_anchor_per_gt] = 1

        # regression targets for positive anchors
        bbox_targets = torch.zeros(N, 4, device=device)
        pos_mask = labels == 1
        if pos_mask.any():
            matched_gt = gt_all[max_idx[pos_mask]]
            anc = anchors[pos_mask]
            anc_cx = (anc[:, 0] + anc[:, 2]) / 2.0
            anc_cy = (anc[:, 1] + anc[:, 3]) / 2.0
            anc_w = anc[:, 2] - anc[:, 0]
            anc_h = anc[:, 3] - anc[:, 1]
            gt_cx = (matched_gt[:, 0] + matched_gt[:, 2]) / 2.0
            gt_cy = (matched_gt[:, 1] + matched_gt[:, 3]) / 2.0
            gt_w = matched_gt[:, 2] - matched_gt[:, 0]
            gt_h = matched_gt[:, 3] - matched_gt[:, 1]
            bbox_targets[pos_mask, 0] = (gt_cx - anc_cx) / anc_w.clamp(min=1)
            bbox_targets[pos_mask, 1] = (gt_cy - anc_cy) / anc_h.clamp(min=1)
            bbox_targets[pos_mask, 2] = (gt_w / anc_w.clamp(min=1)).log()
            bbox_targets[pos_mask, 3] = (gt_h / anc_h.clamp(min=1)).log()

        return labels, bbox_targets


def torchvision_nms(boxes, scores, iou_threshold):
    """Simple NMS implementation (no torchvision dependency)."""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # sort by score descending
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[order[1:]], boxes[idx:idx + 1]).squeeze(1)
        mask = ious <= iou_threshold
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU between two sets of boxes [x1, y1, x2, y2]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou


class Box2DHead(nn.Module):
    """Fast R-CNN style 2D box head (per-class box regression + classification)."""

    def __init__(self, in_dim=256, num_classes=3, hidden_dim=1024):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)  # +1 background
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

    def forward(self, x):
        """x: (N, in_dim, H_roi, W_roi) RoIAlign features."""
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        scores = self.cls_score(x)      # (N, num_classes+1)
        bbox_deltas = self.bbox_pred(x)  # (N, num_classes*4)
        return scores, bbox_deltas


class CubeHead3D(nn.Module):
    """Cube R-CNN style 3D cuboid prediction head.

    Predicts per-class: 2D center offset, depth Z, dimensions (W, H, L),
    and 6D continuous pose representation.
    """

    def __init__(self, in_dim=256, num_classes=3, hidden_dim=1024,
                 z_type="log", pose_type="6d", dims_prior=None):
        super().__init__()
        self.num_classes = num_classes
        self.z_type = z_type
        self.pose_type = pose_type

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )

        # per-class predictions
        self.xy_pred = nn.Linear(hidden_dim, num_classes * 2)     # 2D center offset
        self.z_pred = nn.Linear(hidden_dim, num_classes * 1)      # depth Z
        self.dims_pred = nn.Linear(hidden_dim, num_classes * 3)   # (W, H, L) log offsets
        self.pose_pred = nn.Linear(hidden_dim, num_classes * 6)   # 6D rotation

        # optional dims prior [num_classes, 3]: (W, H, L) in meters
        self.has_dims_prior = dims_prior is not None
        if self.has_dims_prior:
            self.register_buffer("dims_prior", torch.tensor(dims_prior, dtype=torch.float32))

    def forward(self, x):
        """x: (N, in_dim, H_roi, W_roi) RoIAlign features."""
        n = x.size(0)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)

        xy = self.xy_pred(x).view(n, self.num_classes, 2)
        z = self.z_pred(x).view(n, self.num_classes, 1)
        dims = self.dims_pred(x).view(n, self.num_classes, 3)
        pose = self.pose_pred(x).view(n, self.num_classes, 6)

        return xy, z, dims, pose

    def decode_3d(self, xy, z, dims, pose, proposals, K, class_ids, image_size):
        """
        Decode cube head outputs into 3D boxes.

        Args:
            xy: (N, 2) predicted 2D center offset in proposal coords
            z:  (N,) predicted depth
            dims: (N, 3) predicted dimensions (W, H, L)
            pose: (N, 6) 6D rotation
            proposals: (N, 4) [x1, y1, x2, y2] proposal boxes (image pixels)
            K: (3, 3) camera intrinsics
            class_ids: (N,) class indices
            image_size: (H, W) of input image

        Returns:
            boxes_3d: (N, 7) [X, Y, Z, W, H, L, theta] camera coordinates
        """
        device = xy.device
        N = xy.size(0)
        img_h, img_w = image_size

        # 2D center in image coords
        prop_cx = (proposals[:, 0] + proposals[:, 2]) / 2.0
        prop_cy = (proposals[:, 1] + proposals[:, 3]) / 2.0
        prop_w = proposals[:, 2] - proposals[:, 0]
        prop_h = proposals[:, 3] - proposals[:, 1]

        cx_pred = prop_cx + xy[:, 0] * prop_w
        cy_pred = prop_cy + xy[:, 1] * prop_h

        # decode depth Z
        if self.z_type == "log":
            z_pred = z.exp()
        elif self.z_type == "sigmoid":
            z_pred = z.sigmoid() * 100
        else:
            z_pred = z

        z_pred = z_pred.clamp(min=0.1, max=80.0)

        # back-project to 3D
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X = (cx_pred - cx) * z_pred / fx
        Y = (cy_pred - cy) * z_pred / fy

        # decode dimensions
        if self.has_dims_prior:
            prior = self.dims_prior[class_ids]  # (N, 3)
            W_box = dims[:, 0].exp() * prior[:, 0]
            H_box = dims[:, 1].exp() * prior[:, 1]
            L_box = dims[:, 2].exp() * prior[:, 2]
        else:
            W_box = dims[:, 0].exp()
            H_box = dims[:, 1].exp()
            L_box = dims[:, 2].exp()

        # decode 6D rotation → global yaw
        R = rotation_6d_to_matrix(pose)  # (N, 3, 3)

        # Compute yaw (rotation around Y-axis in camera frame)
        # R[:, 0, 0] = cos(θ), R[:, 0, 2] = sin(θ)
        theta_global = torch.atan2(R[:, 0, 2], R[:, 0, 0])

        boxes_3d = torch.stack([X, Y, z_pred, W_box, H_box, L_box, theta_global], dim=-1)
        return boxes_3d


# ---------------------------------------------------------------------------
# 6D rotation → rotation matrix (pytorch3d port, self-contained)
# ---------------------------------------------------------------------------
@torch.jit.script
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    From Zhou et al., "On the Continuity of Rotation Representations in Neural Networks",
    CVPR 2019.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class DetectionProbe(nn.Module):
    """Full detection probe: Feature Extractor → FPN → RPN → 2D Head + Cube Head.

    Follows the Cube R-CNN / Omni3D paradigm:
    1. Frozen backbone produces multi-block features
    2. DPT/MultiscaleHead/Linear projects features into a dense feature map
    3. Feature Pyramid (P2–P5) is built from that map
    4. RPN generates region proposals
    5. RoIAlign pools features per proposal
    6. 2D head predicts class + refined 2D box
    7. Cube head predicts 3D cuboid (center_2d_offset, Z, dims, pose)

    Args:
        feat_dim:    backbone feature dimensions (list of 4 ints or single int)
        num_classes: number of object classes (default 3 for KITTI: Car, Ped, Cyclist)
        head_type:   feature extraction head ("dpt", "multiscale", or "linear")
        hidden_dim:  hidden dimension for feature extraction head
        fpn_dim:     FPN output dimension
        kernel_size: conv kernel size for feature extractor
        image_size:  (H, W) input image size for stride computation
        dims_prior:  [num_classes, 3] class-average dimensions (W, H, L)
    """

    def __init__(
        self,
        feat_dim,
        num_classes=3,
        head_type="dpt",
        hidden_dim=512,
        fpn_dim=256,
        kernel_size=1,
        image_size=None,
        dims_prior=None,
        # unused; kept for config compatibility
        min_depth=None,
        max_depth=None,
        base_depth=None,
        base_dims=None,
        bins=None,
        conv_dim=None,
        checkpoint_path=None,
    ):
        super().__init__()

        self.name = f"detection_{head_type}"
        self.num_classes = num_classes
        self.image_size = image_size or (384, 1280)
        self.fpn_dim = fpn_dim

        # Normalize feat_dim: DPT/MultiscaleHead expect a list of 4 ints.
        # Backbones like DINO (dense-cls) provide a list; TIPSv2 provides a single int.
        if isinstance(feat_dim, int):
            feat_dim = [feat_dim] * 4

        # 1. Feature extractor: DPT / MultiscaleHead / Linear
        if head_type == "dpt":
            self.feat_extractor = DPT(feat_dim, fpn_dim, hidden_dim, kernel_size)
        elif head_type == "multiscale":
            self.feat_extractor = MultiscaleHead(feat_dim, fpn_dim, hidden_dim, kernel_size)
        elif head_type == "linear":
            self.feat_extractor = Linear(feat_dim, fpn_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # 2. Feature Pyramid
        self.fpn = FeaturePyramid(fpn_dim, fpn_dim)

        # 3. RPN
        self.rpn = RPNHead(fpn_dim, num_anchors=1)

        # 4. RoIAlign pooler (parameter-free, constructed per call)
        self.roi_output_size = 7

        # 5. 2D Box Head
        self.box_head = Box2DHead(fpn_dim, num_classes)

        # 6. Cube Head — use KITTI dims prior if none provided
        if dims_prior is None:
            dims_prior = [
                [1.60, 1.53, 3.90],   # Car    (W, H, L)
                [0.60, 1.75, 0.80],   # Pedestrian
                [0.60, 1.75, 1.80],   # Cyclist
            ]
        self.cube_head = CubeHead3D(fpn_dim, num_classes, dims_prior=dims_prior)

        # image-level strides for each FPN level (relative to image_size)
        feat_stride = self._compute_stride(feat_dim)
        self.feat_strides = [
            feat_stride,          # P2
            feat_stride * 2,      # P3
            feat_stride * 4,      # P4
            feat_stride * 8,      # P5
        ]

    def _compute_stride(self, feat_dim):
        """Estimate backbone stride from image size and feature dimensions."""
        # For ViT backbones, stride ≈ patch_size
        # We approximate: stride = image_h / feature_h
        # But since we don't know feature_h at init, use a heuristic
        return 14  # default ViT patch size; overridden during forward

    @property
    def device(self):
        return next(self.parameters()).device

    def _roi_align(self, features, proposals, output_size=7):
        """Simple RoIAlign: crop + interpolate from the feature map.

        Args:
            features: (B, C, H_f, W_f) or single feature map
            proposals: list of (N_i, 4) tensors [x1, y1, x2, y2] in image coords

        Returns:
            (N_total, C, output_size, output_size)
        """
        from torchvision.ops import RoIAlign

        if isinstance(features, list):
            features = features[0]  # use P2 for RoIAlign

        B, C, H_f, W_f = features.shape
        img_h, img_w = self.image_size
        spatial_scale = H_f / img_h

        roi_boxes = []
        for b in range(B if B > 0 else 1):
            if b < len(proposals):
                for box in proposals[b]:
                    x1, y1, x2, y2 = box
                    roi_boxes.append([
                        float(b),
                        float(x1) * spatial_scale,
                        float(y1) * spatial_scale,
                        float(x2) * spatial_scale,
                        float(y2) * spatial_scale,
                    ])

        if len(roi_boxes) == 0:
            return torch.empty(0, C, output_size, output_size, device=features.device)

        roi = torch.tensor(roi_boxes, device=features.device, dtype=torch.float32)
        roi_align = RoIAlign(output_size=(output_size, output_size),
                             spatial_scale=1.0, sampling_ratio=2)
        return roi_align(features, roi)

    def forward(self, feats, targets=None, K=None):
        """
        Args:
            feats: backbone features (single tensor or list)
            targets: list of dicts with 'boxes_2d', 'labels', 'boxes_3d' per image
            K: (B, 3, 3) camera intrinsics

        Returns (training):
            loss_dict: dict of scalar losses

        Returns (inference):
            results: list of dicts with 'boxes_3d', 'scores', 'labels' per image
        """
        # Normalize: DPT/MultiscaleHead expect a list of 4 feature maps.
        # Backbones like TIPSv2 return a single tensor; DINO dense-cls returns a list.
        if isinstance(feats, torch.Tensor):
            feats = [feats] * 4

        # 1. Feature extraction
        extracted = self.feat_extractor(feats)  # (B, fpn_dim, H, W)

        # 2. Feature Pyramid
        pyramid = self.fpn(extracted)  # [P2, P3, P4, P5]

        # 3. RPN forward
        obj_logits, box_deltas = self.rpn(pyramid)
        anchors = self.rpn.generate_anchors(pyramid, self.image_size, self.feat_strides)

        if self.training and targets is not None:
            return self._forward_train(pyramid, obj_logits, box_deltas, anchors,
                                       targets, K)
        else:
            return self._forward_inference(pyramid, obj_logits, box_deltas, anchors, K)

    def _forward_train(self, pyramid, obj_logits, box_deltas, anchors, targets, K):
        """Training forward pass with RPN + 2D head + cube head losses."""
        device = self.device
        B = len(targets)

        losses = {}

        # --- RPN loss ---
        gt_boxes_2d_list = [t["boxes_2d"].to(device) for t in targets]
        rpn_labels, rpn_bbox_targets = self.rpn.match_anchors_to_gt(anchors, gt_boxes_2d_list)

        # flatten all RPN outputs for loss
        na = self.rpn.num_anchors
        obj_logits_flat = torch.cat([o.permute(0, 2, 3, 1).reshape(-1) for o in obj_logits], dim=0)
        box_deltas_flat = torch.cat([b.permute(0, 2, 3, 1).reshape(-1, 4) for b in box_deltas], dim=0)

        # objectness loss (binary cross entropy)
        num_pos = rpn_labels.sum().clamp(min=1)
        num_neg = (rpn_labels == 0).sum().clamp(min=1)
        pos_mask = rpn_labels == 1
        neg_mask = rpn_labels == 0

        # sample negatives for balance
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) > num_pos * 3:
            neg_indices = neg_indices[torch.randperm(len(neg_indices), device=device)[:num_pos * 3]]
            neg_mask = torch.zeros_like(neg_mask)
            neg_mask[neg_indices] = True

        obj_loss_pos = F.binary_cross_entropy_with_logits(
            obj_logits_flat[pos_mask],
            torch.ones(pos_mask.sum(), device=device)
        )
        obj_loss_neg = F.binary_cross_entropy_with_logits(
            obj_logits_flat[neg_mask],
            torch.zeros(neg_mask.sum(), device=device)
        )
        losses["loss_rpn_obj"] = (obj_loss_pos + obj_loss_neg)

        # box regression loss (smooth L1 on positive anchors)
        if pos_mask.any():
            rpn_box_loss = F.smooth_l1_loss(
                box_deltas_flat[pos_mask],
                rpn_bbox_targets[pos_mask],
                beta=0.11,
            )
            losses["loss_rpn_box"] = rpn_box_loss
        else:
            losses["loss_rpn_box"] = torch.tensor(0.0, device=device)

        # --- Generate proposals for 2D head training ---
        # Use GT boxes as "proposals" during training (simpler, more stable)
        # Collect all GT 2D boxes
        all_proposals = []
        all_gt_labels = []
        all_gt_boxes_2d = []
        all_gt_boxes_3d = []
        all_K = []
        batch_indices_for_boxes = []

        for b, t in enumerate(targets):
            gt2d = t["boxes_2d"].to(device)
            gt_labels_b = t["labels"].to(device)
            gt3d = t["boxes_3d"].to(device) if "boxes_3d" in t else None

            if gt2d.numel() == 0:
                continue

            all_proposals.append(gt2d)
            all_gt_labels.append(gt_labels_b)
            all_gt_boxes_2d.append(gt2d)
            if gt3d is not None and gt3d.numel() > 0:
                all_gt_boxes_3d.append(gt3d)
            all_K.append(K[b] if K is not None else torch.eye(3, device=device))

            batch_indices_for_boxes.extend([b] * len(gt2d))

        if len(all_proposals) == 0:
            losses["loss_box_cls"] = torch.tensor(0.0, device=device)
            losses["loss_box_reg"] = torch.tensor(0.0, device=device)
            losses["loss_cube_xy"] = torch.tensor(0.0, device=device)
            losses["loss_cube_z"] = torch.tensor(0.0, device=device)
            losses["loss_cube_dims"] = torch.tensor(0.0, device=device)
            losses["loss_cube_pose"] = torch.tensor(0.0, device=device)
            return losses

        all_proposals_t = torch.cat(all_proposals, dim=0)
        all_gt_labels_t = torch.cat(all_gt_labels, dim=0)

        # RoIAlign features
        roi_feats = self._roi_align(pyramid,
                                     [all_proposals_t],  # single batch with all boxes
                                     self.roi_output_size)
        # Actually, we need per-image handling. Let's simplify:
        # Use pyramid[0] (P2) directly and RoIAlign with all proposals in one batch
        roi_proposals = []
        for b in range(B):
            if b < len(targets) and targets[b]["boxes_2d"].numel() > 0:
                roi_proposals.append(targets[b]["boxes_2d"].to(device))
            else:
                roi_proposals.append(torch.empty((0, 4), device=device))

        roi_feats = self._roi_align(pyramid, roi_proposals, self.roi_output_size)
        if roi_feats.numel() == 0:
            losses["loss_box_cls"] = torch.tensor(0.0, device=device)
            losses["loss_box_reg"] = torch.tensor(0.0, device=device)
            losses["loss_cube_xy"] = torch.tensor(0.0, device=device)
            losses["loss_cube_z"] = torch.tensor(0.0, device=device)
            losses["loss_cube_dims"] = torch.tensor(0.0, device=device)
            losses["loss_cube_pose"] = torch.tensor(0.0, device=device)
            return losses

        # 2D Box Head
        cls_scores, bbox_deltas = self.box_head(roi_feats)

        # 2D box loss
        cls_targets = all_gt_labels_t.clamp(max=self.num_classes - 1)
        losses["loss_box_cls"] = F.cross_entropy(cls_scores, cls_targets)

        # Box regression targets (per-class, select GT class)
        gt_boxes_2d_all = torch.cat(all_gt_boxes_2d, dim=0)
        prop_boxes_2d_all = gt_boxes_2d_all.clone()  # proposals == GT boxes during training
        prop_cx = (prop_boxes_2d_all[:, 0] + prop_boxes_2d_all[:, 2]) / 2
        prop_cy = (prop_boxes_2d_all[:, 1] + prop_boxes_2d_all[:, 3]) / 2
        prop_w = prop_boxes_2d_all[:, 2] - prop_boxes_2d_all[:, 0]
        prop_h = prop_boxes_2d_all[:, 3] - prop_boxes_2d_all[:, 1]
        gt_cx = (gt_boxes_2d_all[:, 0] + gt_boxes_2d_all[:, 2]) / 2
        gt_cy = (gt_boxes_2d_all[:, 1] + gt_boxes_2d_all[:, 3]) / 2
        gt_w = gt_boxes_2d_all[:, 2] - gt_boxes_2d_all[:, 0]
        gt_h = gt_boxes_2d_all[:, 3] - gt_boxes_2d_all[:, 1]

        box_targets = torch.stack([
            (gt_cx - prop_cx) / prop_w.clamp(min=1),
            (gt_cy - prop_cy) / prop_h.clamp(min=1),
            (gt_w / prop_w.clamp(min=1)).log(),
            (gt_h / prop_h.clamp(min=1)).log(),
        ], dim=1)  # (N, 4)

        # select per-class predictions
        N_roi = cls_targets.size(0)
        bbox_deltas_per_cls = bbox_deltas.view(N_roi, self.num_classes, 4)
        bbox_pred_sel = bbox_deltas_per_cls[torch.arange(N_roi), cls_targets]
        losses["loss_box_reg"] = F.smooth_l1_loss(bbox_pred_sel, box_targets, beta=0.11)

        # Cube Head
        xy, z, dims, pose = self.cube_head(roi_feats)

        # select per-class cube predictions
        xy_sel = xy[torch.arange(N_roi), cls_targets]    # (N, 2)
        z_sel = z[torch.arange(N_roi), cls_targets].squeeze(-1)  # (N,)
        dims_sel = dims[torch.arange(N_roi), cls_targets]  # (N, 3)
        pose_sel = pose[torch.arange(N_roi), cls_targets]  # (N, 6)

        # Cube losses
        # Z loss: log-depth
        if len(all_gt_boxes_3d) > 0:
            gt_boxes_3d_all = torch.cat(all_gt_boxes_3d, dim=0)
            gt_z = gt_boxes_3d_all[:, 2]
            gt_dims_3d = gt_boxes_3d_all[:, 3:6]  # (W, H, L)
            gt_theta = gt_boxes_3d_all[:, 6]

            losses["loss_cube_z"] = F.smooth_l1_loss(z_sel, gt_z.log(), beta=0.11)

            # Dims loss: log offsets
            if self.cube_head.has_dims_prior:
                prior = self.cube_head.dims_prior[cls_targets]
                dims_target = (gt_dims_3d / prior.clamp(min=1e-6)).log()
            else:
                dims_target = gt_dims_3d.log()
            losses["loss_cube_dims"] = F.smooth_l1_loss(dims_sel, dims_target, beta=0.11)

            # Pose loss: 6D rotation → 3x3 → geodesic angle
            # Compute predicted rotation
            R_pred = rotation_6d_to_matrix(pose_sel)  # (N, 3, 3)
            # GT rotation: yaw around Y-axis → 3x3 matrix
            cos_t, sin_t = gt_theta.cos(), gt_theta.sin()
            zeros = torch.zeros_like(cos_t)
            ones = torch.ones_like(cos_t)
            R_gt = torch.stack([
                torch.stack([cos_t, zeros, sin_t], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sin_t, zeros, cos_t], dim=-1),
            ], dim=-2)  # (N, 3, 3)

            # geodesic distance: arccos((trace(R_pred^T R_gt) - 1) / 2)
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = ((trace - 1) / 2).clamp(-1, 1)
            angle = torch.acos(cos_angle)
            losses["loss_cube_pose"] = angle.mean()

            # XY loss: 2D center offset (same as 2D box head regression)
            losses["loss_cube_xy"] = F.smooth_l1_loss(xy_sel, box_targets[:, :2], beta=0.11)
        else:
            losses["loss_cube_z"] = torch.tensor(0.0, device=device)
            losses["loss_cube_dims"] = torch.tensor(0.0, device=device)
            losses["loss_cube_pose"] = torch.tensor(0.0, device=device)
            losses["loss_cube_xy"] = torch.tensor(0.0, device=device)

        return losses

    def _forward_inference(self, pyramid, obj_logits, box_deltas, anchors, K):
        """Inference forward pass: RPN proposals → 2D head → cube head → 3D boxes."""
        device = self.device

        # Generate proposals from RPN
        proposals, rpn_scores = self.rpn.decode_proposals(
            obj_logits, box_deltas, pyramid,
            self.image_size, self.feat_strides,
            pre_nms_topk=1000, post_nms_topk=50,
            nms_thresh=0.7, score_thresh=0.05,
        )

        B = pyramid[0].size(0) if K is not None else 1
        if K is None:
            K = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)

        results = []
        for b in range(B):
            if proposals.numel() == 0:
                results.append({
                    "boxes_3d": torch.empty((0, 7), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
                continue

            # RoIAlign for all proposals
            roi_feats = self._roi_align(
                [pyramid[0][b:b+1]],  # single image
                [proposals],
                self.roi_output_size,
            )

            if roi_feats.numel() == 0:
                results.append({
                    "boxes_3d": torch.empty((0, 7), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
                continue

            # 2D head
            cls_scores, bbox_deltas = self.box_head(roi_feats)
            cls_probs = cls_scores.softmax(dim=-1)
            scores, class_ids = cls_probs[:, 1:].max(dim=1)  # skip background
            class_ids = class_ids + 1  # back to 0-indexed class

            # filter background
            fg_mask = class_ids < self.num_classes  # 0 is bg, so valid classes are 1-3
            # but we skipped bg above, so everything is fg. Just filter low scores.
            keep_mask = scores > 0.1
            if keep_mask.sum() == 0:
                results.append({
                    "boxes_3d": torch.empty((0, 7), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
                continue

            scores = scores[keep_mask]
            class_ids = class_ids[keep_mask]
            prop_kept = proposals[keep_mask]
            roi_feats_kept = roi_feats[keep_mask]

            # Refine 2D boxes
            bbox_d = bbox_deltas[keep_mask].view(-1, self.num_classes, 4)
            bbox_d_sel = bbox_d[torch.arange(keep_mask.sum()), class_ids]

            prop_cx = (prop_kept[:, 0] + prop_kept[:, 2]) / 2
            prop_cy = (prop_kept[:, 1] + prop_kept[:, 3]) / 2
            prop_w = prop_kept[:, 2] - prop_kept[:, 0]
            prop_h = prop_kept[:, 3] - prop_kept[:, 1]

            cx_refined = prop_cx + bbox_d_sel[:, 0] * prop_w
            cy_refined = prop_cy + bbox_d_sel[:, 1] * prop_h
            w_refined = prop_w * bbox_d_sel[:, 2].exp()
            h_refined = prop_h * bbox_d_sel[:, 3].exp()

            refined_boxes = torch.stack([
                cx_refined - w_refined / 2,
                cy_refined - h_refined / 2,
                cx_refined + w_refined / 2,
                cy_refined + h_refined / 2,
            ], dim=1)

            # Cube head
            xy, z, dims, pose = self.cube_head(roi_feats_kept)
            xy_sel = xy[torch.arange(keep_mask.sum()), class_ids]
            z_sel = z[torch.arange(keep_mask.sum()), class_ids].squeeze(-1)
            dims_sel = dims[torch.arange(keep_mask.sum()), class_ids]
            pose_sel = pose[torch.arange(keep_mask.sum()), class_ids]

            boxes_3d = self.cube_head.decode_3d(
                xy_sel, z_sel, dims_sel, pose_sel,
                refined_boxes, K[b], class_ids, self.image_size,
            )

            # Per-class NMS on final detections (using refined 2D boxes)
            keep_all = []
            for c in range(self.num_classes):
                c_mask = class_ids == c
                if c_mask.sum() == 0:
                    continue
                idx_c = c_mask.nonzero(as_tuple=True)[0]
                keep_c = torchvision_nms(refined_boxes[idx_c], scores[idx_c], 0.5)
                keep_all.append(idx_c[keep_c])

            if keep_all:
                keep = torch.cat(keep_all, dim=0)
                boxes_3d = boxes_3d[keep]
                scores = scores[keep]
                class_ids = class_ids[keep]

            results.append({
                "boxes_3d": boxes_3d,
                "scores": scores,
                "labels": class_ids,
            })

        return results


# ============================================================================
# FCOS3D-style 3D Bounding Box Probe (mmdet3d anchor_free_mono3d_head)
# ============================================================================

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )


class Box3DHead(nn.Module):

    def __init__(
        self,
        feat_dim,
        feat_channels=256,
        stacked_convs=4,
        base_depth=None,
        base_dims=None,
        min_depth=0.1,
        max_depth=80.0,
        head_type=None,       # ignored; kept for config compatibility
        hidden_dim=None,      # ignored
        kernel_size=None,     # ignored
    ):
        super().__init__()

        self.name = "box3d_fcos"

        self.cls_convs = nn.ModuleList()
        for i in range(stacked_convs):
            in_ch = feat_dim if i == 0 else feat_channels
            self.cls_convs.append(ConvBlock(in_ch, feat_channels))

        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            in_ch = feat_dim if i == 0 else feat_channels
            self.reg_convs.append(ConvBlock(in_ch, feat_channels))

        self.obj_branch = nn.Sequential(
            ConvBlock(feat_channels, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 1, 1),
        )
        self.offset_branch = nn.Sequential(
            ConvBlock(feat_channels, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 2, 1),
        )
        self.depth_branch = nn.Sequential(
            ConvBlock(feat_channels, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 1, 1),
        )
        self.dims_branch = nn.Sequential(
            ConvBlock(feat_channels, 64),
            nn.Conv2d(64, 3, 1),
        )
        self.yaw_branch = nn.Sequential(
            ConvBlock(feat_channels, 64),
            nn.Conv2d(64, 2, 1),
        )
        self.dir_branch = nn.Sequential(
            ConvBlock(feat_channels, 64),
            nn.Conv2d(64, 1, 1),
        )

        self.base_depth = base_depth
        self.base_dims = base_dims
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_convs, self.reg_convs,
                        self.obj_branch, self.offset_branch,
                        self.depth_branch, self.dims_branch,
                        self.yaw_branch, self.dir_branch]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cls_feat = x
        reg_feat = x
        for conv in self.cls_convs:
            cls_feat = conv(cls_feat)
        for conv in self.reg_convs:
            reg_feat = conv(reg_feat)

        obj = self.obj_branch(cls_feat)
        offset = self.offset_branch(reg_feat)
        depth = self.depth_branch(reg_feat)
        dims = self.dims_branch(reg_feat)
        yaw = self.yaw_branch(reg_feat)
        dir_cls = self.dir_branch(reg_feat)

        return torch.cat([obj, offset, depth, dims, yaw, dir_cls], dim=1)

    def decode_boxes(
        self,
        pred_dense,
        intrinsics,
        stride=16,
        topk=100,
        score_thresh=0.05,
        class_priors=None,  # (B,) class indices for multi-class priors
    ):
        """
        Decode dense predictions into 3D boxes (FCOS3D-style).

        Args:
            pred_dense:   (B, 10, H, W) raw output from forward()
            intrinsics:   (B, 3, 3) camera intrinsics K
            stride:       feature stride relative to input image
            topk:         max detections to return
            score_thresh: minimum objectness score

        Returns:
            boxes_3d: (B, topk, 7) — [X, Y, Z, W, H, L, θ] in camera coords
            scores:   (B, topk)    — objectness scores
            labels:   (B, topk)    — class labels (from dir_cls for now)
        """
        B, C, H, W = pred_dense.shape
        assert C == 10, f"Expected 10 channels, got {C}"

        # --- split channels ---
        obj_logits  = pred_dense[:, 0:1, :, :]     # (B, 1, H, W)
        offset_2d   = pred_dense[:, 1:3, :, :]     # (B, 2, H, W)
        depth_raw   = pred_dense[:, 3:4, :, :]     # (B, 1, H, W)
        dims_raw    = pred_dense[:, 4:7, :, :]     # (B, 3, H, W)
        yaw_sincos  = pred_dense[:, 7:9, :, :]     # (B, 2, H, W)
        dir_logits  = pred_dense[:, 9:10, :, :]    # (B, 1, H, W)

        # --- objectness: sigmoid → flatten → top-k ---
        obj_scores = obj_logits.sigmoid().squeeze(1)  # (B, H, W)
        obj_flat = obj_scores.view(B, -1)              # (B, H*W)

        if topk > H * W:
            topk = H * W

        topk_scores, topk_indices = obj_flat.topk(topk, dim=1)  # (B, K)

        # build grid of pixel coords
        device = pred_dense.device
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)          # (B, H, W)
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

        # --- gather predictions at top-k locations ---
        topk_x = x_grid.reshape(B, -1).gather(1, topk_indices)  # (B, K)
        topk_y = y_grid.reshape(B, -1).gather(1, topk_indices)  # (B, K)

        def gather(feat):
            return feat.view(B, feat.shape[1], -1).gather(2,
                topk_indices.unsqueeze(1).expand(-1, feat.shape[1], -1)
            )  # (B, C, K)

        topk_offset = gather(offset_2d)       # (B, 2, K)
        topk_depth  = gather(depth_raw)       # (B, 1, K)
        topk_dims   = gather(dims_raw)        # (B, 3, K)
        topk_yaw    = gather(yaw_sincos)      # (B, 2, K)
        topk_dir    = gather(dir_logits)      # (B, 1, K)

        # --- 1. Projected 3D center on image ---
        u_3d = (topk_x + topk_offset[:, 0, :]) * stride  # (B, K)
        v_3d = (topk_y + topk_offset[:, 1, :]) * stride  # (B, K)

        # --- 2. Decode depth Z ---
        depth = topk_depth[:, 0, :]  # (B, K)
        if self.base_depth is not None:
            if isinstance(self.base_depth, list) and class_priors is not None:
                # multi-class priors
                mu = torch.tensor(
                    [self.base_depth[c][0] for c in class_priors],
                    device=device
                ).view(B, 1)
                std = torch.tensor(
                    [self.base_depth[c][1] for c in class_priors],
                    device=device
                ).view(B, 1)
                z = mu + depth * std
            else:
                mu, std = self.base_depth
                z = mu + depth * std
        else:
            z = depth.exp()  # log-space

        z = z.clamp(min=self.min_depth, max=self.max_depth)

        # --- 3. Back-project to 3D camera coords ---
        fx = intrinsics[:, 0, 0].view(B, 1)  # (B, 1)
        fy = intrinsics[:, 1, 1].view(B, 1)
        cx = intrinsics[:, 0, 2].view(B, 1)
        cy = intrinsics[:, 1, 2].view(B, 1)

        X = (u_3d - cx) * z / fx  # (B, K)
        Y = (v_3d - cy) * z / fy  # (B, K)
        Z = z                      # (B, K)

        # --- 4. Decode dimensions ---
        dims = topk_dims.permute(0, 2, 1)  # (B, K, 3)
        W_box = dims[..., 0].exp()
        H_box = dims[..., 1].exp()
        L_box = dims[..., 2].exp()

        if self.base_dims is not None:
            if isinstance(self.base_dims, list) and class_priors is not None:
                base = torch.tensor(
                    [self.base_dims[c] for c in class_priors], device=device
                )  # (B, 3)
                W_box = W_box * base[:, 0:1]
                H_box = H_box * base[:, 1:2]
                L_box = L_box * base[:, 2:3]
            else:
                bw, bh, bl = self.base_dims
                W_box = W_box * bw
                H_box = H_box * bh
                L_box = L_box * bl

        # --- 5. Decode yaw ---
        sin_a = topk_yaw[:, 0, :]  # (B, K)
        cos_a = topk_yaw[:, 1, :]  # (B, K)
        # normalize
        norm = torch.sqrt(sin_a**2 + cos_a**2 + 1e-6)
        sin_a = sin_a / norm
        cos_a = cos_a / norm
        alpha = torch.atan2(sin_a, cos_a)  # local observation angle

        # direction classifier: if dir_logits > 0 then back (add pi)
        dir_cls = (topk_dir[:, 0, :] > 0).float()  # (B, K)
        alpha = alpha + dir_cls * torch.pi

        # global yaw: θ = α + arctan2(u_3d - c_u, f_u)
        theta = alpha + torch.atan2(u_3d - cx, fx)

        # --- assemble ---
        boxes_3d = torch.stack([X, Y, Z, W_box, H_box, L_box, theta], dim=-1)  # (B, K, 7)

        return boxes_3d, topk_scores, dir_cls.long()


# ============================================================================
# Geometric 3D Box Probe (Mousavian et al., CVPR 2017)
# Probes frozen backbone features — same paradigm as depth/snorm probes
# ============================================================================

class Geometric3DProbe(nn.Module):
    """
    3D Box Probe using Mousavian et al. approach.

    Takes dense features from a frozen backbone (TIPSv2, DINOv2, etc.),
    crops regions at 2D object boxes via RoIAlign, then predicts:
      - Orientation:  MultiBin sin/cos + bin confidence
      - Dimensions:   offsets from class averages

    Args:
        feat_dim:   backbone feature channels (e.g. 1024 for TIPSv2-L dense-cls)
        bins:       MultiBin orientation bins (default 4)
    """

    def __init__(self, feat_dim, bins=4,
                 hidden_dim=512, conv_dim=256,
                 image_size=None,
                 min_depth=None, max_depth=None,
                 head_type=None, kernel_size=None,
                 base_depth=None, base_dims=None,
                 checkpoint_path=None,
                 ):
        super().__init__()

        self.name = "geometric_box3d"
        self.bins = bins
        self.feat_dim = feat_dim
        self.image_size = image_size or (384, 1280)  # (H, W) for spatial_scale

        # --- Conv head: C → 256 → 512 → flatten ---
        self.conv_head = nn.Sequential(
            nn.Conv2d(feat_dim, conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # --- Orientation head: hidden_dim→256→256→bins*2 ---
        self.orientation = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, bins * 2),
        )

        # --- Confidence head: hidden_dim→256→bins ---
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, bins),
        )

        # --- Dimension head: hidden_dim→512→512→3 ---
        self.dimension = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(True), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
            nn.Linear(512, 3),
        )

        from evals.utils.geometric import KITTI_CLASS_DIMS, generate_bins
        self.class_dims = torch.tensor(KITTI_CLASS_DIMS, dtype=torch.float32)
        self.angle_bins = generate_bins(bins)

        self.base_depth = base_depth
        self.base_dims = base_dims

    def forward(self, feats, boxes_2d_list):
        """
        Args:
            feats:         (B, C, H_f, W_f) backbone dense features
            boxes_2d_list: list of (N_i, 4) tensors [x1, y1, x2, y2] in IMAGE pixels

        Returns:
            orient: (N_total, bins, 2), conf: (N_total, bins), dim: (N_total, 3)
        """
        from torchvision.ops import RoIAlign

        B, C, H_f, W_f = feats.shape
        img_h, img_w = self.image_size
        spatial_scale = H_f / img_h  # feature / image ratio

        roi_boxes = []
        for b in range(B):
            for n in range(len(boxes_2d_list[b])):
                x1, y1, x2, y2 = boxes_2d_list[b][n]
                roi_boxes.append([
                    float(b),
                    float(x1) * spatial_scale,
                    float(y1) * spatial_scale,
                    float(x2) * spatial_scale,
                    float(y2) * spatial_scale,
                ])

        if len(roi_boxes) == 0:
            return (
                torch.empty(0, self.bins, 2, device=feats.device),
                torch.empty(0, self.bins, device=feats.device),
                torch.empty(0, 3, device=feats.device),
            )

        roi = torch.tensor(roi_boxes, device=feats.device, dtype=torch.float32)
        roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        rois_feat = roi_align(feats, roi)  # (N, C, 7, 7)

        x = self.conv_head(rois_feat)
        x = self.pool(x).view(x.size(0), -1)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = torch.nn.functional.normalize(orientation, dim=2)

        return orientation, self.confidence(x), self.dimension(x)

    def decode_boxes(self, orient, conf, dim, box_2d_list, K_list, labels=None,
                     true_argmax=None):
        """
        Decode predictions into 3D boxes — exact YOLO3D inference logic.

        Args:
            true_argmax: (N,) optional ground-truth bin indices.
                         When provided, uses GT bin instead of model's argmax.
                         Useful for overfit/debug experiments.
        """
        import numpy as np
        from evals.utils.geometric import calc_location, generate_bins

        device = orient.device
        N = orient.size(0)
        angle_bins = generate_bins(self.bins)

        argmax = conf.argmax(dim=1) if true_argmax is None else true_argmax
        orient_sel = orient[torch.arange(N), argmax]
        cos_a, sin_a = orient_sel[:, 0], orient_sel[:, 1]
        alpha_pred = torch.atan2(sin_a, cos_a)
        alpha_pred = alpha_pred + torch.tensor(
            [angle_bins[i] for i in argmax.tolist()], device=device
        )
        alpha_pred = alpha_pred - torch.pi

        if labels is not None:
            class_avg = self.class_dims.to(device)[labels]
        else:
            class_avg = self.class_dims.to(device).mean(dim=0, keepdim=True).expand(N, -1)
        dims_pred = dim + class_avg

        boxes_3d, scores = [], []

        for i in range(N):
            dim_np = dims_pred[i].cpu().numpy()
            box_2d = box_2d_list[i]
            if isinstance(box_2d, torch.Tensor):
                box_2d = box_2d.cpu().tolist()
            K = K_list[i]
            if isinstance(K, torch.Tensor):
                K = K.cpu().numpy()

            fx, cx = float(K[0, 0]), float(K[0, 2])

            P = np.zeros((3, 4), dtype=np.float64)
            P[:3, :3] = K

            box_2d_pairs = [(float(box_2d[0]), float(box_2d[1])),
                           (float(box_2d[2]), float(box_2d[3]))]
            u_center = (box_2d[0] + box_2d[2]) / 2.0
            theta_ray = float(np.arctan2(u_center - cx, fx))

            loc = calc_location(dim_np, P, box_2d_pairs, alpha_pred[i].item(), theta_ray)
            orient_global = alpha_pred[i].item() + theta_ray

            # calc_location returns center; get_3d_corners expects Y_bottom
            Y_bottom = float(loc[1]) + float(dims_pred[i, 0]) / 2.0  # H = dims_pred[0]

            box = torch.tensor([
                float(loc[0]), Y_bottom, float(loc[2]),
                float(dims_pred[i, 1]), float(dims_pred[i, 0]), float(dims_pred[i, 2]),
                orient_global,
            ], device=device)
            boxes_3d.append(box)
            scores.append(conf[i, argmax[i]])

        if len(boxes_3d) == 0:
            return (torch.empty((0, 7), device=device),
                    torch.empty((0,), device=device),
                    torch.empty((0,), dtype=torch.long, device=device))

        return torch.stack(boxes_3d), torch.stack(scores), argmax