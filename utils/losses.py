import torch
import torch.nn.functional as F
import torch.nn as nn

def bin_residual_loss(pred, target, num_bins=4):
    pred_bin = pred[:, 6]
    pred_res = pred[:, 7]
    target_bin = target[:, 6]
    target_res = target[:, 7]

    bin_loss = F.cross_entropy(pred_bin.unsqueeze(1), target_bin.long())
    res_loss = F.smooth_l1_loss(pred_res, target_res)
    return bin_loss + res_loss

def bbox_loss(pred, target):
    loc_loss = F.smooth_l1_loss(pred[:, :6], target[:, :6])
    angle_loss = bin_residual_loss(pred, target)
    return loc_loss + angle_loss

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
