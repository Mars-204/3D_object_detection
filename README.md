# 3D Object Detection  
**End-to-end pipeline for 3D bounding box estimation from RGB images and point cloud data**

---

## Overview

This project implements a pipeline to generate **3D bounding boxes** for objects using RGB images and point cloud data. The goal is to localize and describe objects in 3D space by leveraging both visual and depth information.

---
## Dataset

Each sample consists of:  
- **RGB image:** High-resolution color image  
- **Point cloud projection:** Depth or point cloud data aligned to RGB image  
- **Instance masks:** (For approach 2) Segmentation masks identifying individual objects  
- **3D bounding boxes:** Ground truth labels, represented as 8 corner points per object  

---


## Approach

Two complementary approaches are explored:

## 1) RGB-D Fusion Approach  
- **Input:** Concatenate RGB images with point cloud projections (depth maps) to form a multi-channel input (RGB-D).  
- **Model:** Train a deep neural network (e.g., modified ResNet) on the combined RGB-D input to directly regress 3D bounding box parameters.  
- **Advantage:** Simple end-to-end learning that integrates color and geometric cues.  

### Model Architecture

- Backbone: ResNet-18 modified to accept 6-channel RGB-D input  
- Output Heads:  
  - 3D bounding box regression (center, dimensions, orientation) for multiple object instances  
- Loss: L1 regression loss on bounding box parameters


### Training

- Data loading integrates RGB and point cloud images into a single tensor input  
- Ground truth 3D bounding boxes converted from 8 corner points to parameterized format (center, size, yaw bin + residual)  
- Training performed with batch loading, optimizer setup, and learning rate scheduling

## 2) Instance Segmentation + 3D Clustering Approach (Improved)

- Use a state-of-the-art instance segmentation model (e.g., YOLOv8 segmentation) on RGB images to get per-instance masks.
- For each mask:
  - Extract the corresponding 3D points from the aligned point cloud.
  - Filter out invalid points (NaNs or zeros).
  - Perform **PCA-based oriented 3D bounding box extraction** on the clustered points.
    - PCA finds the main object axes.
    - Creates an oriented bounding box tightly fitting the cluster.
- This results in more accurate, orientation-aware 3D bounding boxes compared to axis-aligned boxes.

### Benefits of PCA-based 3D Bounding Boxes

- Captures object orientation naturally.
- Provides tighter fits around objects, improving downstream tasks like tracking or pose estimation.
- Handles arbitrary object rotations in 3D space.

### Usage

1. Generate instance segmentation masks using YOLOv8 segmentation model on RGB images.
2. For each mask, cluster points from the corresponding aligned point cloud.
3. Compute oriented 3D bounding boxes using PCA (as implemented in `get_oriented_3d_bbox`).
4. Use these 3D bounding boxes for training or evaluation.

### Future Work & Improvements

- Enhance instance segmentation accuracy using state-of-the-art models  
- Refine 3D bounding box fitting using advanced geometric methods beyond convex hull  

---

## Requirements

- Python 3.7+  
- PyTorch  
- OpenCV  
- Albumentations  
- Scipy  
- Torchvision
- Cuda 11.6
- Ultralytics (Approach 2)

- Install Cuda from official website. Then use the following command to install dependencies 
```
pip install torch torchvision opencv-python albumentations scipy ultralytics scikit-learn matplotlib
```


