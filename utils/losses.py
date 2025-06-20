import torch
import torch.nn.functional as F

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
