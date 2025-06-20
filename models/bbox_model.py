import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BBoxModel(nn.Module):
    def __init__(self, num_instances=10, yaw_bins=2):
        super().__init__()
        self.num_instances = num_instances
        self.yaw_bins = yaw_bins

        # Use ResNet-18 and modify input channels
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Predict xyz, w, h, d → 6 values per instance
        self.box_reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_instances * 6)
        )

        # Predict yaw bin (classification)
        self.yaw_cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_instances * yaw_bins)
        )

        # Predict yaw residual (regression)
        self.yaw_res_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_instances)
        )

    def forward(self, x):
        # x: [B, 6, 1024, 1024]
        features = self.backbone(x)  # [B, 512]

        B = x.size(0)
        N = self.num_instances

        # 6 regression values (x, y, z, w, h, d)
        bbox_flat = self.box_reg_head(features)  # [B, N*6]
        bbox = bbox_flat.view(B, N, 6)

        # Yaw bin classification logits
        yaw_logits_flat = self.yaw_cls_head(features)  # [B, N*yaw_bins]
        yaw_logits = yaw_logits_flat.view(B, N, self.yaw_bins)

        # Yaw residual regression
        yaw_residual = self.yaw_res_head(features)  # [B, N]

        return bbox, yaw_logits, yaw_residual

