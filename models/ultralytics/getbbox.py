import numpy as np
from sklearn.decomposition import PCA

def get_oriented_3d_bbox(points_3d):
    """
    Given Nx3 points, return an oriented bounding box:
    - center: 3D centroid
    - extents: length along PCA axes (3,)
    - rotation: 3x3 PCA components matrix (columns are principal axes)
    """
    pca = PCA(n_components=3)
    pca.fit(points_3d)
    
    center = points_3d.mean(axis=0)
    points_centered = points_3d - center
    
    # Project points into PCA space
    points_pca = pca.transform(points_centered)
    
    # Find bounds in PCA space
    min_pca = points_pca.min(axis=0)
    max_pca = points_pca.max(axis=0)
    extents = max_pca - min_pca
    
    # Compute box corners in PCA space
    corners_pca = np.array([
        [min_pca[0], min_pca[1], min_pca[2]],
        [min_pca[0], min_pca[1], max_pca[2]],
        [min_pca[0], max_pca[1], min_pca[2]],
        [min_pca[0], max_pca[1], max_pca[2]],
        [max_pca[0], min_pca[1], min_pca[2]],
        [max_pca[0], min_pca[1], max_pca[2]],
        [max_pca[0], max_pca[1], min_pca[2]],
        [max_pca[0], max_pca[1], max_pca[2]],
    ])
    
    # Transform corners back to original space
    corners_3d = pca.inverse_transform(corners_pca) + center
    
    return {
        "center": center,
        "extents": extents,
        "rotation": pca.components_,
        "corners_3d": corners_3d
    }

def get_3d_bbox_from_mask(pc, mask):
    coords = np.argwhere(mask > 0)
    points_3d = pc[coords[:, 0], coords[:, 1], :]
    valid_mask = ~np.isnan(points_3d).any(axis=1) & (np.linalg.norm(points_3d, axis=1) > 0)
    points_3d = points_3d[valid_mask]

    if points_3d.shape[0] < 10:
        return None

    # Use PCA-based oriented bounding box
    bbox = get_oriented_3d_bbox(points_3d)
    return bbox
