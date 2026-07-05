"""
YOLO3D-style Geometric 3D Bounding Box Probe Training
======================================================
Probes a frozen 2D backbone to test whether its features encode object-level
3D pose (orientation, dimensions). Uses RoIAlign on dense features.

Follows same structure as train_box3d.py / train_depth.py / train_snorm.py.

Usage:
    python train_mousavian3d.py --config-dir configs --config-name mousavian3d_training
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.datasets.kitti import collate_kitti
from evals.utils.losses import OrientationLoss, build_orientation_target
from evals.utils.geometric import KITTI_CLASS_DIMS
from evals.utils.metrics import evaluate_box3d, evaluate_aos


def train(
    model,
    probe,
    train_loader,
    optimizer,
    n_epochs,
    alpha,
    w,
    bins,
    conf_loss_func,
    dim_loss_func,
    orient_loss_func,
    rank=0,
    world_size=1,
    valid_loader=None,
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0.0
        pbar = tqdm(train_loader) if rank == 0 else train_loader

        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)
            boxes_3d_list = batch["boxes_3d"]
            boxes_2d_list = batch["boxes_2d"]
            labels_list = batch["labels"]
            K_batch = batch["K"].to(rank)

            optimizer.zero_grad()

            # frozen backbone → dense features
            with torch.no_grad():
                feats = model(images)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0] if len(feats) == 1 else feats[-1]

            # probe: RoIAlign crops → orientation, conf, dim
            orient, conf, dim = probe(feats, boxes_2d_list)

            # --- Build targets (exact YOLO3D) ---
            all_orient_gt, all_conf_gt, all_dim_gt = [], [], []
            for b in range(images.size(0)):
                for n in range(len(boxes_3d_list[b])):
                    box_3d, box_2d, K = boxes_3d_list[b][n], boxes_2d_list[b][n], K_batch[b]
                    theta = box_3d[6]
                    u_center = (box_2d[0] + box_2d[2]) / 2.0
                    theta_ray = torch.atan2(u_center - K[0, 2], K[0, 0])
                    alpha_gt = theta - theta_ray
                    alpha_gt = torch.atan2(torch.sin(alpha_gt), torch.cos(alpha_gt))
                    og, cg = build_orientation_target(alpha_gt.unsqueeze(0), bins=bins)
                    all_orient_gt.append(og.squeeze(0))
                    all_conf_gt.append(cg.squeeze(0))
                    label = labels_list[b][n].item()
                    class_avg = torch.tensor(KITTI_CLASS_DIMS[label], device=rank)
                    dim_gt = torch.tensor([box_3d[4], box_3d[3], box_3d[5]], device=rank)
                    all_dim_gt.append(dim_gt - class_avg)

            if len(all_orient_gt) == 0:
                continue

            orient_gt = torch.stack(all_orient_gt).to(rank)
            conf_gt = torch.stack(all_conf_gt).to(rank)
            dim_gt = torch.stack(all_dim_gt).to(rank)

            # --- Loss (exact YOLO3D) ---
            orient_loss = orient_loss_func(orient, orient_gt, conf_gt)
            dim_loss = dim_loss_func(dim, dim_gt)
            truth_conf = torch.max(conf_gt, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)
            loss = alpha * dim_loss + conf_loss + w * orient_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if rank == 0:
                pbar.set_description(
                    f"ep {ep} | loss: {train_loss / (i + 1):.4f} | "
                    f"orient: {orient_loss.item():.3f} dim: {dim_loss.item():.4f} "
                    f"conf: {conf_loss.item():.3f}"
                )

        train_loss /= len(train_loader)
        if rank == 0:
            logger.info(f"train loss ep {ep} | {train_loss:.4f}")
            if valid_loader is not None:
                val_metrics = validate(model, probe, valid_loader)
                logger.info(f"valid ep {ep} | {val_metrics}")


def validate(model, probe, loader, verbose=True):
    probe.eval()
    all_box3d, all_aos = [], []

    with torch.inference_mode():
        pbar = tqdm(loader, desc="Validation") if verbose else loader
        for batch in pbar:
            images = batch["image"].cuda()
            boxes_3d_list = batch["boxes_3d"]
            boxes_2d_list = batch["boxes_2d"]
            labels_list = batch["labels"]
            K_batch = batch["K"].cuda()

            feats = model(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0] if len(feats) == 1 else feats[-1]

            orient, conf, dim = probe(feats, boxes_2d_list)

            # per-image eval: collect all preds for this image
            for b in range(images.size(0)):
                if len(boxes_3d_list[b]) == 0:
                    continue

                # indices of this image's predictions in the flattened output
                N_prev = sum(len(boxes_2d_list[j]) for j in range(b))
                N_cur = len(boxes_2d_list[b])
                idx = slice(N_prev, N_prev + N_cur)

                orient_b = orient[idx]
                conf_b = conf[idx]
                dim_b = dim[idx]

                box_2d_tensors = [boxes_2d_list[b][n] for n in range(N_cur)]
                K_tensors = [K_batch[b]] * N_cur
                obj_labels_t = labels_list[b].to(orient.device)

                # Compute GT bins for overfit/debug: use GT argmax instead of model's
                true_bins = []
                for n in range(N_cur):
                    box_3d = boxes_3d_list[b][n]
                    box_2d = boxes_2d_list[b][n]
                    K = K_batch[b]
                    theta = box_3d[6]
                    u_center = (box_2d[0] + box_2d[2]) / 2.0
                    theta_ray = torch.atan2(u_center - K[0, 2], K[0, 0])
                    alpha_gt = theta - theta_ray
                    alpha_gt = torch.atan2(torch.sin(alpha_gt), torch.cos(alpha_gt))
                    _, cg = build_orientation_target(alpha_gt.unsqueeze(0), bins=probe.bins)
                    true_bins.append(cg.argmax(dim=1).item())
                true_argmax = torch.tensor(true_bins, device=orient.device)

                boxes_3d_pred, scores, _ = probe.decode_boxes(
                    orient_b, conf_b, dim_b, box_2d_tensors, K_tensors, labels=obj_labels_t,
                    true_argmax=true_argmax)
                gt_boxes = boxes_3d_list[b].cuda()

                all_aos.append(evaluate_aos(boxes_3d_pred, scores, gt_boxes))
                for n in range(len(gt_boxes)):
                    all_box3d.append(evaluate_box3d(boxes_3d_pred[n:n+1], scores[n:n+1], gt_boxes[n:n+1]))

    probe.train()

    agg = {}
    if all_box3d:
        for key in all_box3d[0]:
            vals = [m.get(key, float('nan')) for m in all_box3d]
            vals = [v for v in vals if v == v]
            agg[key] = sum(vals) / len(vals) if vals else 0.0
    if all_aos:
        aos_vals = [m["aos"] for m in all_aos if m["aos"] == m["aos"]]
        agg["aos"] = sum(aos_vals) / len(aos_vals) if aos_vals else 0.0

    return agg


def train_model(rank, world_size, cfg):
    # ===== DATA LOADERS =====
    collate_fn = collate_kitti
    trainval_loader = build_loader(
        cfg.dataset, "train", cfg.batch_size, world_size, collate_fn=collate_fn
    )
    test_loader = build_loader(
        cfg.dataset, "valid", cfg.batch_size, 1, collate_fn=collate_fn
    )

    # ===== MODELS =====
    model = instantiate(cfg.backbone)
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim)

    # ===== EXP NAME =====
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{cfg.optimizer.probe_lr:4.2e}",
        f"{cfg.optimizer.model_lr:4.2e}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info)
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")

    if rank == 0:
        exp_path = Path(__file__).parent / f"box3d_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    # SAM / MAE patch
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.__getitem__(0)["image"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    # ===== OPTIMIZER (exact YOLO3D: SGD + momentum) =====
    optimizer = torch.optim.SGD(probe.parameters(), lr=cfg.optimizer.probe_lr, momentum=0.9)

    # ===== LOSS FUNCTIONS (exact YOLO3D) =====
    conf_loss_func = nn.CrossEntropyLoss().to(rank)
    dim_loss_func = nn.MSELoss().to(rank)
    orient_loss_func = OrientationLoss

    train(
        model, probe,
        trainval_loader, optimizer,
        cfg.optimizer.n_epochs,
        cfg.get("alpha", 0.6), cfg.get("w", 0.4),
        cfg.probe.get("bins", 4),
        conf_loss_func, dim_loss_func, orient_loss_func,
        rank=rank, world_size=world_size,
        valid_loader=test_loader,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")
        test_metrics = validate(model, probe, test_loader)
        for k, v in test_metrics.items():
            logger.info(f"Final test {k:15s} | {v:.4f}")

        torch.save(
            {"probe": probe.state_dict(), "cfg": cfg},
            exp_path / "checkpoint.pth",
        )
        logger.info(f"Saved checkpoint to {exp_path / 'checkpoint.pth'}")


@hydra.main(version_base=None, config_path="configs", config_name="mousavian3d_training")
def main(cfg: DictConfig):
    train_model(0, cfg.system.num_gpus, cfg)


if __name__ == "__main__":
    main()
