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

KITTI 3D Object Detection Dataset Loader (HuggingFace Streaming)
=================================================================
Streams KITTI from HuggingFace Hub (nateraw/kitti) -- no local download needed.

Calibration: standard KITTI P2 intrinsics from
https://github.com/ruhyadi/YOLO3D/blob/main/dataset/KITTI/training/calib_cam_to_cam.txt

Label schema (from datasets):
    alpha:       observation angle (radians)
    bbox:        [x1, y1, x2, y2] in image pixels
    dimensions:  [height, width, length] in meters (KITTI order)
    location:    [x, y, z] camera coordinates in meters
    rotation_y:  global yaw around Y-axis (radians)
    type:        class name string
"""
import torch
import torchvision.transforms as tv_transforms
from datasets import load_dataset


# Standard KITTI P2 intrinsics (left color camera, cam2)
# P_rect_02: fx=718.856, fy=718.856, cx=607.1928, cy=185.2157
KITTI_P2 = torch.tensor([
    [718.856,   0.0,     607.1928],
    [  0.0,   718.856,   185.2157],
    [  0.0,     0.0,       1.0   ],
], dtype=torch.float32)

KITTI_ORIG_W = 1242
KITTI_ORIG_H = 375

KITTI_CLASS_MAP = {
    "Car": 0, "Van": 0, "Truck": 0, "Tram": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2,
}
KITTI_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def KITTI(
    split="train",
    image_mean="imagenet",
    image_size=None,
    augment_train=False,
    classes=None,
    max_depth=80.0,
    **_kwargs,
):
    """Factory for KITTI dataset streamed from HuggingFace Hub."""
    assert split in ["train", "valid", "trainval", "test"]
    if image_size is None:
        image_size = (384, 1280)
    if classes is None:
        classes = KITTI_CLASS_NAMES
    return KITTIDataset(split, image_mean, image_size, augment_train, classes, max_depth)


class KITTIDataset(torch.utils.data.Dataset):
    """Streaming KITTI 3D detection from HuggingFace (nateraw/kitti).

    Returns:
        image:    (3, H, W) normalized tensor
        boxes_3d: (N, 7) [X, Y, Z, W, H, L, theta] camera coords
        labels:   (N,) class indices
        K:        (3, 3) intrinsics scaled for image_size
        boxes_2d: (N, 4) [x1, y1, x2, y2] scaled
    """

    def __init__(self, split, image_mean, image_size, augment_train, classes, max_depth):
        super().__init__()
        self.name = "KITTI3D"
        self.image_size = image_size
        self.max_depth = max_depth
        self.class_to_idx = {c: KITTI_CLASS_MAP[c] for c in classes}

        # transforms
        if image_mean == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std  = [0.26862954, 0.26130258, 0.27577711]
        elif image_mean == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
        else:
            mean = [0.0, 0.0, 0.0]
            std  = [1.0, 1.0, 1.0]

        t_list = [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=mean, std=std),
        ]
        if augment_train and "train" in split:
            t_list.insert(-1, tv_transforms.RandomApply(
                [tv_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.8
            ))
        t_list.append(tv_transforms.Resize(
            image_size, interpolation=tv_transforms.InterpolationMode.BILINEAR
        ))
        self.transform = tv_transforms.Compose(t_list)

        # HuggingFace splits
        if split == "test":
            self._hf_split = "test"
            self._has_labels = False
            self._start, self._end = 0, 7518
        else:
            self._hf_split = "train"
            self._has_labels = True
            if split == "train":
                self._start, self._end = 0, 3712
            elif split == "valid":
                self._start, self._end = 3712, 7481
            else:
                self._start, self._end = 0, 7481

        self._ds = load_dataset("nateraw/kitti", split=self._hf_split)
        print(f"KITTI {split}: {self._end - self._start} images (cached)")

    def __len__(self):
        return self._end - self._start

    def _parse_sample(self, sample):
        image = sample["image"]  # PIL, RGB, 1242x375
        image = self.transform(image)

        scale_w = self.image_size[1] / KITTI_ORIG_W
        scale_h = self.image_size[0] / KITTI_ORIG_H
        K = KITTI_P2.clone()
        K[0, 0] *= scale_w; K[1, 1] *= scale_h
        K[0, 2] *= scale_w; K[1, 2] *= scale_h

        boxes_3d, boxes_2d, labels = [], [], []
        if self._has_labels:
            for obj in sample.get("label", []):
                cls_name = obj["type"]
                if cls_name not in self.class_to_idx:
                    continue
                if obj["truncated"] > 0.9:
                    continue
                h, w, l = obj["dimensions"]  # KITTI: height, width, length
                x, y, z = obj["location"]
                if z > self.max_depth or z < 0.1:
                    continue
                x1, y1, x2, y2 = obj["bbox"]
                x1 *= scale_w; x2 *= scale_w
                y1 *= scale_h; y2 *= scale_h
                boxes_3d.append([x, y, z, w, h, l, obj["rotation_y"]])
                boxes_2d.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[cls_name])

        if not boxes_3d:
            return {
                "image": image,
                "boxes_3d": torch.empty((0, 7)),
                "labels": torch.empty((0,), dtype=torch.long),
                "K": K,
                "boxes_2d": torch.empty((0, 4)),
            }
        return {
            "image": image,
            "boxes_3d": torch.tensor(boxes_3d, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "K": K,
            "boxes_2d": torch.tensor(boxes_2d, dtype=torch.float32),
        }

    def __getitem__(self, index):
        idx = self._start + index
        sample = self._ds[idx]
        return self._parse_sample(sample)


def collate_kitti(batch):
    """Custom collate: stacks images/Ks, keeps variable-size boxes as lists."""
    return {
        "image":    torch.stack([b["image"]    for b in batch]),
        "boxes_3d": [b["boxes_3d"] for b in batch],
        "labels":   [b["labels"]   for b in batch],
        "K":        torch.stack([b["K"]        for b in batch]),
        "boxes_2d": [b["boxes_2d"] for b in batch],
    }
