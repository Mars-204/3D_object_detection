import torch
from torch.utils.data import DataLoader
from models.bbox_model import BBoxModel
from utils.dataloader import BBoxDataset
from utils.losses import bbox_loss
from utils.metrics import iou_3d
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    num_instances = 10

    train_dataset = BBoxDataset(split='train')
    val_dataset = BBoxDataset(split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: custom_collate_fn_fixed_maskes(b, num_instances)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: custom_collate_fn_fixed_maskes(b, num_instances)
    )

    model = BBoxModel(num_instances=num_instances).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for rgbpc, rgb, pc, masks, bboxes in train_loader:
            rgbpc = rgbpc.to(device)
            # masks = masks.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            pred_bboxes = model(rgbpc)

            # # Resize gt masks to mask_size (64x64)
            bbox_pred, yaw_logits, yaw_res = model(rgbpc)
            loss = bbox3d_loss((bbox_pred, yaw_logits, yaw_res), bboxes)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")

def bbox3d_loss(pred, target, yaw_bins=2):
    """
    pred = (bbox: [B, N, 6], yaw_logits: [B, N, yaw_bins], yaw_res: [B, N])
    target: [B, N, 8] = [x, y, z, w, h, d, yaw_bin, yaw_res]
    """
    bbox_pred, yaw_logits, yaw_residual = pred

    bbox_target = target[:, :, :6]
    yaw_bin_target = target[:, :, 6].long()       # class index
    yaw_res_target = target[:, :, 7]

    loss_bbox = nn.functional.l1_loss(bbox_pred, bbox_target)

    # Reshape yaw_logits to [B*N, yaw_bins] for CE
    B, N = yaw_logits.shape[:2]
    loss_yaw_cls = nn.functional.cross_entropy(
        yaw_logits.view(B * N, yaw_bins),
        yaw_bin_target.view(-1)
    )

    # L1 loss on residual
    loss_yaw_res = nn.functional.l1_loss(yaw_residual, yaw_res_target)

    return loss_bbox + loss_yaw_cls + loss_yaw_res


def custom_collate_fn(batch):
    full_tensors = []
    rgbs = []
    pcs = []
    masks = []
    bboxes = []

    for full_tensor, rgb, pc, mask, bbox in batch:
        full_tensors.append(full_tensor)
        rgbs.append(rgb)
        pcs.append(pc)
        masks.append(mask)
        bboxes.append(bbox)

    full_tensors = torch.stack(full_tensors)  # [B, 6, 1024, 1024]
    rgbs = torch.stack(rgbs)                  # [B, 3, 1024, 1024]
    pcs = torch.stack(pcs)                    # [B, 3, 1024, 1024]

    # masks and bboxes are lists because their shape varies per sample
    return full_tensors, rgbs, pcs, masks, bboxes

def pad_or_truncate(tensor, target_len, pad_value=0):
    """
    Pads or truncates a tensor along the first dimension (instance dimension).
    tensor: [N, ...]
    Returns tensor of shape [target_len, ...]
    """
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

    # Pad/truncate masks and bboxes to fixed length
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
    train()
