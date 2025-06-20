# utils/dataloader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy.spatial import ConvexHull

class BBoxDataset(Dataset):
    def __init__(self, root_dir=r"D:\data", split='train'):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.samples = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.target_size = (1024, 1024)

    def __len__(self):
        return len(self.samples)

    def convert_corners_to_bbox_params(self, corners):
        corners = np.array(corners)
        hull = ConvexHull(corners)

        center = np.mean(corners, axis=0)

        mins = np.min(corners, axis=0)
        maxs = np.max(corners, axis=0)
        dims = maxs - mins  # [w, h, d]

        max_len = 0
        best_vec = None
        for simplex in hull.simplices:
            p1, p2 = corners[simplex[0]], corners[simplex[1]]
            vec = p2 - p1
            xy_len = np.linalg.norm(vec[:2])
            if xy_len > max_len:
                max_len = xy_len
                best_vec = vec

        yaw = np.arctan2(best_vec[1], best_vec[0]) if best_vec is not None else 0.0

        yaw_bin_size = np.pi
        yaw_bin = int((yaw + np.pi) // yaw_bin_size)
        bin_center = -np.pi + yaw_bin_size * (yaw_bin + 0.5)
        yaw_res = yaw - bin_center

        return np.array([*center, *dims, yaw_bin, yaw_res], dtype=np.float32)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        # --- Load paths ---
        rgb_path = os.path.join(sample_path, "rgb.jpg")
        pc_path = os.path.join(sample_path, "pc.npy")
        mask_path = os.path.join(sample_path, "mask.npy")
        bbox_path = os.path.join(sample_path, "bbox3d.npy")

        # --- Load and preprocess RGB ---
        rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        rgb = np.transpose(rgb, (2, 0, 1))  # [3, 1024, 1024]

        # --- Load and preprocess Point Cloud ---
        pc = np.load(pc_path).astype(np.float32)
        pc = np.nan_to_num(pc, nan=0.0)
        pc = np.transpose(pc, (1, 2, 0))  # [H, W, 3]
        pc = cv2.resize(pc, self.target_size, interpolation=cv2.INTER_NEAREST)
        pc = np.transpose(pc, (2, 0, 1))  # [3, 1024, 1024]

        # --- Stack RGB + PC ---
        full_tensor = np.concatenate([rgb, pc], axis=0)  # [6, 1024, 1024]
        full_tensor = torch.tensor(full_tensor, dtype=torch.float32)

        # --- Load and resize instance masks ---
        masks = np.load(mask_path).astype(np.uint8)  # [N, H, W]
        masks_resized = []
        for i in range(masks.shape[0]):
            resized = cv2.resize(masks[i], self.target_size, interpolation=cv2.INTER_NEAREST)
            masks_resized.append(resized)
        masks_resized = np.stack(masks_resized, axis=0)  # [N, 1024, 1024]
        masks_tensor = torch.tensor(masks_resized, dtype=torch.uint8)
        rgb = torch.tensor(rgb,dtype=torch.float32)
        pc = torch.tensor(pc,dtype=torch.float32)

        # --- Load 3D BBox ---
        # bbox = np.load(bbox_path).astype(np.float32)
        # gt = torch.tensor(bbox, dtype=torch.float32)

        bboxes_corners = np.load(bbox_path).astype(np.float32)        # [N, 8, 3]
        bbox_params = [self.convert_corners_to_bbox_params(c) for c in bboxes_corners]
        gt = torch.tensor(bbox_params)                                 # [N, 8]

        return full_tensor,rgb,pc, masks_tensor, gt
