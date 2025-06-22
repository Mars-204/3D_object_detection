import torch
from torch.utils.data import DataLoader
from models.bbox_model import BBoxModel
from utils.dataloader import BBoxDataset
from utils.losses import bbox_loss,bbox3d_loss
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

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
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)

        for rgbpc, rgb, pc, masks, bboxes in train_bar:
            rgbpc = rgbpc.to(device)
            # masks = masks.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            bbox_pred, yaw_logits, yaw_res = model(rgbpc)
            loss = bbox3d_loss((bbox_pred, yaw_logits, yaw_res), bboxes)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * rgbpc.size(0)
            avg_loss = running_train_loss / ((train_bar.n + 1) * batch_size)
            train_bar.set_postfix(loss=avg_loss)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")

        model.eval()
        running_val_loss = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        with torch.no_grad():
            for rgbpc, rgb, pc, masks, bboxes in val_bar:
                rgbpc = rgbpc.to(device)
                bboxes = bboxes.to(device)

                bbox_pred, yaw_logits, yaw_res = model(rgbpc)
                val_loss = bbox3d_loss((bbox_pred, yaw_logits, yaw_res), bboxes)

                running_val_loss += val_loss.item() * rgbpc.size(0)
                avg_val_loss = running_val_loss / ((val_bar.n + 1) * batch_size)
                val_bar.set_postfix(val_loss=avg_val_loss)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")


        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {epoch_train_loss:.4f} Val Loss: {epoch_val_loss:.4f}")

    
     # Plotting loss curves
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_loss.png')
    # plt.show()

    # Save loss values
    results = {'train_losses': train_losses, 'val_losses': val_losses}
    with open('loss_curves.json', 'w') as f:
        json.dump(results, f)

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
    train()
    # eval
    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()
