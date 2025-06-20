import torch
import os
from models.bbox_model import BBoxModel

def export_onnx(model_ckpt_path, save_path="exported_model.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BBoxModel().to(device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    dummy_rgb = torch.randn(1, 3, 224, 224).to(device)
    dummy_pc = torch.randn(1, 1024, 3).to(device)  # assuming 1024 points
    dummy_mask = torch.randn(1, 1, 224, 224).to(device)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_pc, dummy_mask),
        save_path,
        input_names=['rgb', 'pointcloud', 'mask'],
        output_names=['bbox_output'],
        opset_version=11,
        dynamic_axes={
            'rgb': {0: 'batch_size'},
            'pointcloud': {0: 'batch_size', 1: 'num_points'},
            'mask': {0: 'batch_size'},
            'bbox_output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to {save_path}")

if __name__ == '__main__':
    export_onnx("checkpoints/bbox_model.pth")
