import open3d as o3d

def draw_bboxes_in_open3d(pcd, bbox_array, color=[1, 0, 0]):
    lines_idx = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    geometries = [pcd]
    for corners in bbox_array:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines_idx)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines_idx])
        geometries.append(line_set)

    return geometries
