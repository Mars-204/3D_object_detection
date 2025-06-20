# 3D Object Detection  
**End-to-end pipeline for 3D bounding box estimation from RGB images and point cloud data**

---

## Overview

This project implements a pipeline to generate **3D bounding boxes** for objects using RGB images and point cloud data. The goal is to localize and describe objects in 3D space by leveraging both visual and depth information.

---

## Approach

Two complementary approaches are explored:

### 1) RGB-D Fusion Approach  
- **Input:** Concatenate RGB images with point cloud projections (depth maps) to form a multi-channel input (RGB-D).  
- **Model:** Train a deep neural network (e.g., modified ResNet) on the combined RGB-D input to directly regress 3D bounding box parameters.  
- **Advantage:** Simple end-to-end learning that integrates color and geometric cues.  

### 2) Instance Segmentation + Point Cloud Clustering  
- **Step 1:** Use RGB images to perform **instance segmentation**, generating masks that delineate individual objects.  
- **Step 2:** Calculate the centroids of each instance mask to localize clusters in the 2D image plane.  
- **Step 3:** Extract corresponding 3D point clusters from the point cloud using these centroids as guidance.  
- **Step 4:** Compute tight 3D bounding boxes around each extracted cluster.  
- **Advantage:** Utilizes precise instance masks to separate objects and produce accurate 3D bounding boxes via clustering.

---

## Dataset

Each sample consists of:  
- **RGB image:** High-resolution color image  
- **Point cloud projection:** Depth or point cloud data aligned to RGB image  
- **Instance masks:** (For approach 2) Segmentation masks identifying individual objects  
- **3D bounding boxes:** Ground truth labels, represented as 8 corner points per object  

---

## Model Architecture (for Approach 1)

- Backbone: ResNet-18 modified to accept 6-channel RGB-D input  
- Output Heads:  
  - 3D bounding box regression (center, dimensions, orientation) for multiple object instances  
- Loss: L1 regression loss on bounding box parameters

---

## Training

- Data loading integrates RGB and point cloud images into a single tensor input  
- Ground truth 3D bounding boxes converted from 8 corner points to parameterized format (center, size, yaw bin + residual)  
- Training performed with batch loading, optimizer setup, and learning rate scheduling

---

## Future Work & Improvements

- Enhance instance segmentation accuracy using state-of-the-art models (Mask R-CNN, etc.)  
- Refine 3D bounding box fitting using advanced geometric methods beyond convex hull  
- Incorporate temporal information for video sequences  
- Implement evaluation metrics such as 3D IoU and average precision for 3D detection

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

```
pip install torch torchvision opencv-python albumentations scipy ultralytics
```
---

## Usage

1. Prepare your dataset with the expected folder structure and file naming.  
2. Configure training parameters in the training script.  
3. Run training to produce a model capable of predicting 3D bounding boxes from RGB-D input.  
4. Use inference scripts to evaluate or visualize predictions.

---


