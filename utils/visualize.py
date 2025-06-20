import open3d as o3d
import numpy as np

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
