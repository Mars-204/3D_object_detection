import torch
from torch.utils.data import DataLoader
from models.bbox_model import BBoxModel
from utils.dataloader import BBoxDataset
from utils.losses import bbox_loss,bbox3d_loss
from utils.visualize import create_video_from_frames,get_3d_box_corners, get_yaw_from_bin, project_to_image, draw_3d_box
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 1  # one frame at a time for sequential video
    num_instances = 10
    yaw_bins = 2

    test_dataset = BBoxDataset(split='test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: custom_collate_fn_fixed_maskes(b, num_instances)
    )
    model = BBoxModel(num_instances=num_instances).to(device)
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()  # Set to evaluation mode
   
   # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'bbox_visualization_direct.mp4'
    out_video = None

    with torch.no_grad():
        for batch_idx, (rgbpc, rgb, pc, masks, bboxes) in enumerate(test_loader):
            rgbpc = rgbpc.to(device)
            output = model(rgbpc)

            bbox_pred, yaw_logits, yaw_res = output
            preds = torch.cat([bbox_pred, yaw_logits.argmax(dim=-1, keepdim=True).float(), yaw_res.unsqueeze(-1)], dim=-1)

            for i in range(preds.shape[0]):
                rgb_img = rgb[i].detach().cpu().numpy()
                rgb_img = np.transpose(rgb_img, (1, 2, 0))  # [H, W, C]
                rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8).copy()
                for j in range(preds.shape[1]):  # instances
                    box = preds[i, j].cpu().numpy()
                    if np.all(box == 0): continue  # padded

                    x, y, z, w, h, d, yaw_bin, yaw_residual = box
                    yaw = get_yaw_from_bin(int(yaw_bin), yaw_residual, num_bins=2)
                    corners_3d = get_3d_box_corners([x, y, z], [w, h, d], yaw)
                    corners_2d = project_to_image(corners_3d, fx=1024, fy=1024, cx=512, cy=512)
                    rgb_img = draw_3d_box(rgb_img, corners_2d)

                # Init video writer if not yet
                if out_video is None:
                    height, width, _ = rgb_img.shape
                    out_video = cv2.VideoWriter(output_path, fourcc, 1, (width, height))

                out_video.write(rgb_img)
                print(f"Frame {batch_idx} written to video.")

    if out_video:
        out_video.release()
        print(f"🎥 Video saved to: {output_path}")
        


def pad_or_truncate(tensor, target_len, pad_value=0):
    
    n = tensor.shape[0]
    if n == target_len:
        return tensor
    elif n > target_len:
        return tensor[:target_len]
    else:
        pad_shape = (target_len - n,) + tensor.shape[1:]
        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)

def custom_collate_fn_fixed_maskes(batch, num_instances=10):
    full_tensors, rgbs, pcs, masks, bboxes = zip(*batch)

    full_tensors = torch.stack(full_tensors)
    rgbs = torch.stack(rgbs)
    pcs = torch.stack(pcs)

    masks_padded = []
    bboxes_padded = []
    for m, b in zip(masks, bboxes):
        m_pad = pad_or_truncate(m, num_instances, pad_value=0)
        b_pad = pad_or_truncate(b, num_instances, pad_value=0)
        masks_padded.append(m_pad)
        bboxes_padded.append(b_pad)

    masks_padded = torch.stack(masks_padded)  # [B, num_instances, H, W]
    bboxes_padded = torch.stack(bboxes_padded)  # [B, num_instances, 8]

    return full_tensors, rgbs, pcs, masks_padded, bboxes_padded

if __name__ == '__main__':
    inference()
