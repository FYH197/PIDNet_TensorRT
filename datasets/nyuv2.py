# ------------------------------------------------------------------------------
# NYUv2 dataset support for PIDNet training
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import torch
from PIL import Image

from configs import config
from .base_dataset import BaseDataset


INDOOR_15_CLASS_WEIGHTS = [
    0.500000,  # floor
    0.500000,  # wall
    1.452578,  # door
    4.000000,  # stairs
    0.500000,  # chair
    0.500000,  # table
    1.274291,  # sofa
    1.000000,  # bed
    0.500000,  # storage
    1.254441,  # small_obstacle
    4.000000,  # person
    4.000000,  # sanitary
    0.858074,  # window
    0.985338,  # other_obstacle
    4.000000,  # cup
]


class NYUv2(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        num_classes=895,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=640,
        crop_size=(480, 640),
        scale_factor=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bd_dilate_size=4,
    ):
        self.in_channels = int(getattr(config.MODEL, "IN_CHANNELS", 3))
        if self.in_channels == 4:
            # RGB-only normalization kept above as the constructor defaults.
            mean = list(mean) + [float(config.DATASET.DEPTH_MEAN)]
            std = list(std) + [float(config.DATASET.DEPTH_STD)]
        super(NYUv2, self).__init__(
            ignore_label, base_size, crop_size, scale_factor, mean, std
        )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.bd_dilate_size = bd_dilate_size
        self.class_weights = self._build_class_weights()
        self.use_boundary = getattr(config.LOSS, "USE_BOUNDARY", True)
        self.edge_root = os.path.join(root, "edges")
        self.use_precomputed_edges = self.use_boundary and os.path.isdir(self.edge_root)

        list_file = os.path.join(root, list_path)
        self.img_list = [line.strip().split() for line in open(list_file) if line.strip()]
        self.files = self.read_files()

    def _build_class_weights(self):
        if not getattr(config.LOSS, "CLASS_BALANCE", False):
            return None

        if self.num_classes != len(INDOOR_15_CLASS_WEIGHTS):
            raise ValueError(
                "CLASS_BALANCE is only configured for the 15-class indoor setup; "
                f"got num_classes={self.num_classes}."
            )

        return torch.tensor(INDOOR_15_CLASS_WEIGHTS, dtype=torch.float32)

    def read_files(self):
        files = []
        for item in self.img_list:
            # RGB-only two-column parsing kept for compatibility:
            # image_path, label_path = item
            if self.in_channels == 4:
                if len(item) != 3:
                    raise ValueError(
                        f"RGB-D split rows must contain image, depth, and label; got {item}"
                    )
                image_path, depth_path, label_path = item
            else:
                if len(item) != 2:
                    raise ValueError(
                        f"RGB split rows must contain image and label; got {item}"
                    )
                image_path, label_path = item
                depth_path = None
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append(
                {
                    "img": image_path,
                    "depth": depth_path,
                    "label": label_path,
                    "name": name,
                }
            )
        return files

    def label_transform(self, label):
        # NYUv2 extracted labels can exceed 255, so keep int32 instead of uint8.
        return np.asarray(label).astype(np.int32)

    def _build_edge(self, label, edge_size=4):
        label = np.asarray(label).astype(np.int32)
        valid = label != self.ignore_label
        edge = np.zeros(label.shape, dtype=np.uint8)

        diff_h = (label[1:, :] != label[:-1, :]) & valid[1:, :] & valid[:-1, :]
        diff_w = (label[:, 1:] != label[:, :-1]) & valid[:, 1:] & valid[:, :-1]

        edge[1:, :] |= diff_h
        edge[:-1, :] |= diff_h
        edge[:, 1:] |= diff_w
        edge[:, :-1] |= diff_w

        kernel = np.ones((edge_size, edge_size), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1) > 0).astype(np.float32)
        return edge

    def multi_scale_aug(self, image, label=None, edge=None, rand_scale=1, rand_crop=True):
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        if image.shape[2] == 4:
            rgb = cv2.resize(
                image[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            depth = cv2.resize(
                image[:, :, 3], (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
            image = np.concatenate([rgb, depth[:, :, None]], axis=2)
        else:
            image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        if label is not None:
            label = cv2.resize(
                label.astype(np.float32),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
            if edge is not None:
                edge = cv2.resize(
                    edge.astype(np.float32),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge

    def _load_edge(self, label_path):
        edge_path = os.path.join(self.edge_root, os.path.basename(label_path))
        if not os.path.exists(edge_path):
            return None

        edge = np.asarray(Image.open(edge_path)).astype(np.float32)
        if edge.max() > 1:
            edge = edge / 255.0
        return edge

    def _pad_to_stride(self, image, label, edge, stride=8):
        h, w = image.shape[:2]
        target_h = ((h + stride - 1) // stride) * stride
        target_w = ((w + stride - 1) // stride) * stride
        if target_h == h and target_w == w:
            return image, label, edge

        # RGB-only padding kept for reference:
        # image = self.pad_image(image, h, w, (target_h, target_w), (0.0, 0.0, 0.0))
        pad_value = (0.0, 0.0, 0.0, 0.0) if image.shape[2] == 4 else (0.0, 0.0, 0.0)
        image = self.pad_image(image, h, w, (target_h, target_w), pad_value)
        label = self.pad_image(label, h, w, (target_h, target_w), (self.ignore_label,))
        edge = self.pad_image(edge, h, w, (target_h, target_w), (0.0,))
        return image, label, edge

    def gen_sample(self, image, label, edge=None, multi_scale=True, is_flip=True, edge_size=4):
        if edge is None:
            edge = self._build_edge(label, edge_size=edge_size)

        if multi_scale:
            rand_scale = 0.5 + np.random.randint(0, self.scale_factor + 1) / 10.0
            image, label, edge = self.multi_scale_aug(
                image, label, edge, rand_scale=rand_scale
            )
        else:
            # Use the same fixed 640x480 evaluation resolution as S2Net.
            target_h, target_w = self.crop_size
            if image.shape[2] == 4:
                rgb = cv2.resize(
                    image[:, :, :3],
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                depth = cv2.resize(
                    image[:, :, 3],
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                image = np.concatenate([rgb, depth[:, :, None]], axis=2)
            else:
                image = cv2.resize(
                    image,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            label = cv2.resize(
                label.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
            edge = cv2.resize(
                edge.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)

        image = self.input_transform(image, city=False)
        label = self.label_transform(label)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root, item["img"])).convert("RGB")
        image = np.array(image)
        size = image.shape
        if self.in_channels == 4:
            depth_path = os.path.join(self.root, item["depth"])
            depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                raise FileNotFoundError(f"Unable to read depth map: {depth_path}")
            if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
                raise ValueError(
                    f"Depth must be single-channel uint16: {depth_path}; "
                    f"got dtype={depth_mm.dtype}, shape={depth_mm.shape}"
                )
            if depth_mm.shape != image.shape[:2]:
                raise ValueError(
                    f"RGB/depth size mismatch for {name}: "
                    f"RGB={image.shape[:2]}, depth={depth_mm.shape}"
                )
            depth_255 = (
                np.clip(
                    depth_mm.astype(np.float32),
                    0.0,
                    float(config.DATASET.DEPTH_SCALE_MM),
                )
                / float(config.DATASET.DEPTH_SCALE_MM)
                * 255.0
            )
            image = np.concatenate([image.astype(np.float32), depth_255[:, :, None]], axis=2)

        label = Image.open(os.path.join(self.root, item["label"]))
        label = np.array(label).astype(np.int32)
        if self.use_boundary:
            edge = self._load_edge(item["label"]) if self.use_precomputed_edges else None
        else:
            edge = np.zeros(label.shape, dtype=np.float32)

        image, label, edge = self.gen_sample(
            image,
            label,
            edge,
            self.multi_scale,
            self.flip,
            edge_size=self.bd_dilate_size,
        )

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint16)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i] + ".png"))
