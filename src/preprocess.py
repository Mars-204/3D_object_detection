import numpy as np
import cv2
import open3d as o3d

def clean_pointcloud_and_rgb(rgb_img_path, pointcloud, box_z_percentile_range=(0, 15)):
    """
    Args:
        rgb_img_path (str): Path to the RGB image.
        pointcloud (np.ndarray): 3 x H x W NumPy array.
        box_z_percentile_range (tuple): Z percentiles (min%, max%) representing the box to remove.
    
    Returns:
        o3d.geometry.PointCloud: Cleaned colored point cloud.
        np.ndarray: Masked RGB image.
    """

    img = cv2.imread(rgb_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    
    z = pointcloud[2]
    z_flat = z.flatten()
    
    # ---- Step 1: Remove ground (top z%)
    ground_thresh = np.percentile(z_flat, 99.5)
    ground_mask = z < ground_thresh

    # ---- Step 2: Box edge mask (via contours)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_mask = np.zeros((H, W), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(box_mask, [largest], -1, 255, thickness=cv2.FILLED)
    box_mask = box_mask.astype(bool)

    # ---- Step 3: Remove box structure using Z filtering ----
    z_min = np.percentile(z_flat, box_z_percentile_range[0])
    z_max = np.percentile(z_flat, box_z_percentile_range[1])
    structure_mask = ~((z >= z_min) & (z <= z_max))  # Keep points NOT in this box band

    # ---- Final mask ----
    combined_mask = ground_mask & box_mask & structure_mask

    # Apply final mask
    points = pointcloud[:, combined_mask].T
    colors = img_rgb[combined_mask] / 255.0

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Masked RGB
    masked_img = img_rgb.copy()
    masked_img[~combined_mask] = 0

    return pcd, masked_img
