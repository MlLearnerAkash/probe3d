import torch
import torch.nn as nn
from torch.nn.functional import interpolate


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
