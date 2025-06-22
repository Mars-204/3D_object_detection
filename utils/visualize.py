import open3d as o3d
import numpy as np
import cv2
import os
import math

def draw_bboxes_in_open3d(pcd, bbox_array, mask=None, mask_color=[1, 0, 0], bbox_color=[0, 1, 0]):
    lines_idx = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    geometries = []

    # Full point cloud (gray)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(pcd)

    # Optional masked points
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.any(axis=0)
        pc_np = np.asarray(pcd.points)
        masked_points = pc_np[mask.flatten()]

        if len(masked_points) > 0:
            pcd_masked = o3d.geometry.PointCloud()
            pcd_masked.points = o3d.utility.Vector3dVector(masked_points)
            pcd_masked.paint_uniform_color(mask_color)
            geometries.append(pcd_masked)

    # Bounding boxes
    for corners in bbox_array:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines_idx)
        line_set.colors = o3d.utility.Vector3dVector([bbox_color for _ in lines_idx])
        geometries.append(line_set)

    return geometries

def create_video_from_frames(frames_dir, output_path='output_video.mp4', fps=10):
    images = sorted(
        [img for img in os.listdir(frames_dir) if img.startswith('vis_output_') and img.endswith('.jpg')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    if not images:
        print("No frames found.")
        return

    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, _ = first_frame.shape

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for img_file in images:
        frame = cv2.imread(os.path.join(frames_dir, img_file))
        video_writer.write(frame)
    video_writer.release()
    print(f" Video saved at {output_path}")

def draw_3d_box(image, corners_2d, color=(0, 255, 0)):
    if len(corners_2d) != 8:
        return image
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top square
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom square
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    for i, j in connections:
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[j])
        cv2.line(image, pt1, pt2, color, 2)
    return image

def project_to_image(corners_3d, fx, fy, cx, cy):
    points_2d = []
    for x, y, z in corners_3d:
        if z <= 0: continue  # skip invalid depth
        u = fx * x / z + cx
        v = fy * y / z + cy
        points_2d.append([int(u), int(v)])
    return np.array(points_2d)

def get_yaw_from_bin(bin_idx, residual, num_bins=2):
    angle_per_bin = 2 * math.pi / num_bins
    yaw_center = bin_idx * angle_per_bin + angle_per_bin / 2 - math.pi
    return yaw_center + residual

def get_3d_box_corners(center, size, yaw):
    x, y, z = center
    w, h, d = size

    # 8 corners in local frame (before rotation)
    corners = np.array([
        [w/2, h/2, d/2],
        [-w/2, h/2, d/2],
        [-w/2, -h/2, d/2],
        [w/2, -h/2, d/2],
        [w/2, h/2, -d/2],
        [-w/2, h/2, -d/2],
        [-w/2, -h/2, -d/2],
        [w/2, -h/2, -d/2],
    ])

    # Rotation matrix around yaw axis (Y-axis)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    R = np.array([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0],
        [-sin_yaw, 0, cos_yaw]
    ])

    rotated = (R @ corners.T).T
    translated = rotated + np.array(center)

    return translated
