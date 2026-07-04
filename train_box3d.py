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

FCOS3D-style 3D Bounding Box Probe Training
============================================
Probes a frozen 2D backbone to test whether its features encode object-level
3D pose (center, dimensions, orientation).

Usage:
    python train_box3d.py --config-dir configs --config-name box3d_training
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.datasets.kitti import collate_kitti
from evals.utils.losses import box3d_loss
from evals.utils.metrics import evaluate_box3d
from evals.utils.optim import cosine_decay_linear_warmup


def ddp_setup(rank: int, world_size: int, port: int):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    stride,
    rank=0,
    world_size=1,
    valid_loader=None,
):
    """FCOS3D-style training loop."""
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        running_losses = {}
        pbar = tqdm(train_loader) if rank == 0 else train_loader

        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)

            # get ground truth 3D boxes and intrinsics
            # dataset should provide: "boxes_3d" (list of Nx7 tensors),
            # "labels" (list of N tensors), "K" (Bx3x3 intrinsics)
            gt_boxes  = [b.to(rank) for b in batch["boxes_3d"]]
            gt_labels = [l.to(rank) for l in batch["labels"]]
            K = batch["K"].to(rank)

            optimizer.zero_grad()

            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
            else:
                feats = model(images)

            # FCOS3D dense prediction
            pred_dense = probe(feats)  # (B, 10, H, W)

            # compute loss
            total_loss, loss_dict = box3d_loss(
                pred_dense, gt_boxes, gt_labels, K, stride=stride
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            train_loss += total_loss.item()

            # accumulate running losses
            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v

            if rank == 0:
                avg_loss = train_loss / (i + 1)
                loss_str = " ".join(
                    f"{k}: {running_losses[k] / (i + 1):.3f}"
                    for k in running_losses
                )
                pbar.set_description(
                    f"ep {ep} | total: {avg_loss:.4f} | {loss_str} | lr: {pr_lr:.2e}"
                )

        train_loss /= len(train_loader)

        if rank == 0:
            logger.info(f"train loss ep {ep} | {train_loss:.4f}")
            if valid_loader is not None:
                val_metrics = validate(model, probe, valid_loader, stride)
                logger.info(f"valid ep {ep} | {val_metrics}")


def validate(model, probe, loader, stride, verbose=True):
    """Evaluate 3D box predictions."""
    all_metrics = []
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Validation") if verbose else loader
        for batch in pbar:
            images = batch["image"].cuda()
            gt_boxes  = [b.cuda() for b in batch["boxes_3d"]]
            K = batch["K"].cuda()

            feats = model(images)
            pred_dense = probe(feats)

            # decode boxes
            boxes_3d, scores, labels = probe.decode_boxes(
                pred_dense, K, stride=stride, topk=50, score_thresh=0.05
            )

            # compute metrics per sample
            B = images.shape[0]
            for b in range(B):
                metrics = evaluate_box3d(boxes_3d[b], scores[b], gt_boxes[b])
                all_metrics.append(metrics)

    # aggregate across samples
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if not (isinstance(m[key], float) and m[key] != m[key])]  # filter NaN
        agg[key] = sum(vals) / len(vals) if vals else 0.0

    return agg


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== GET DATA LOADERS =====
    # detect if using KITTI dataset (needs custom collate for variable boxes)
    dataset_name = cfg.dataset.get("_target_", "")
    use_kitti = "kitti" in dataset_name.lower()
    collate_fn = collate_kitti if use_kitti else None

    trainval_loader = build_loader(
        cfg.dataset, "train", cfg.batch_size, world_size, collate_fn=collate_fn
    )
    test_loader = build_loader(
        cfg.dataset, "valid", cfg.batch_size, 1, collate_fn=collate_fn
    )
    trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim)

    # setup experiment name
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

    # ===== SETUP LOGGING =====
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

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    # ===== OPTIMIZER =====
    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
        epoch,
        cfg.optimizer.n_epochs * len(trainval_loader),
        cfg.optimizer.warmup_epochs * len(trainval_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)

    stride = cfg.get("stride", 16)

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        stride=stride,
        rank=rank,
        world_size=world_size,
        valid_loader=test_loader,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")
        test_metrics = validate(model, probe, test_loader, stride)
        for k, v in test_metrics.items():
            logger.info(f"Final test {k:15s} | {v:.4f}")

        # save model
        torch.save(
            {"model": model.state_dict(), "probe": probe.state_dict(), "cfg": cfg},
            exp_path / "checkpoint.pth",
        )
        logger.info(f"Saved checkpoint to {exp_path / 'checkpoint.pth'}")

    if world_size > 1:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "configs"),
    config_name="box3d_training",
)
def main(cfg: DictConfig):
    logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
