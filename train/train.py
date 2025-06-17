# train.py
import torch
from torch.utils.data import DataLoader
from models.bbox_model import BBoxModel
from utils.dataloader import BBoxDataset
from utils.losses import bbox_loss
from utils.metrics import iou_3d

import os
from torch.utils.tensorboard import SummaryWriter

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    batch_size = 8
    lr = 1e-4
    num_epochs = 30

    # Dataset & Dataloader
    train_dataset = BBoxDataset(split='train')
    val_dataset = BBoxDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = BBoxModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Logging
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            rgb, pc, mask, gt = [x.to(device) for x in batch]
            pred = model(rgb, pc, mask)
            loss = bbox_loss(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation
        model.eval()
        with torch.no_grad():
            total_iou = 0
            for batch in val_loader:
                rgb, pc, mask, gt = [x.to(device) for x in batch]
                pred = model(rgb, pc, mask)
                total_iou += iou_3d(pred, gt)
            avg_iou = total_iou / len(val_loader)
            writer.add_scalar("IoU/val", avg_iou, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - IoU: {avg_iou:.4f}")

if __name__ == '__main__':
    train()