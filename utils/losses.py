import torch.nn.functional as F

def bbox_loss(pred, gt):
    # Split predictions
    loc_loss = F.l1_loss(pred[:, :6], gt[:, :6])  # x,y,z,w,h,d
    angle_loss = F.mse_loss(pred[:, 6], gt[:, 6])
    return loc_loss + angle_loss