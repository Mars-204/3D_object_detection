import torch
import torch.nn as nn
import torchvision.models as models

class BBoxModel(nn.Module):
    def __init__(self):
        super(BBoxModel, self).__init__()

        # Pretrained CNN backbone (e.g., ResNet18)
        self.rgb_backbone = models.resnet18(pretrained=True)
        self.rgb_backbone.fc = nn.Identity()

        # Simple MLP for point cloud input
        self.pc_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # Final fusion and regression
        self.fc = nn.Sequential(
            nn.Linear(512 + 256 + 1, 256),  # RGB + PC + mask avg
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # (x, y, z, w, h, d, yaw)
        )

    def forward(self, rgb, pc, mask):
        b = rgb.shape[0]
        rgb_feat = self.rgb_backbone(rgb)
        pc_feat = self.pc_encoder(pc.view(b, -1, 3).mean(dim=1))
        mask_feat = mask.view(b, -1).float().mean(dim=1, keepdim=True)

        x = torch.cat([rgb_feat, pc_feat, mask_feat], dim=1)
        return self.fc(x)
