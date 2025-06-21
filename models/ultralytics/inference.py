import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import ConvexHull
import json

def polygon_to_mask(polygon, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def get_3d_bbox_from_mask(pc, mask):
    coords = np.argwhere(mask > 0)
    points_3d = pc[coords[:, 0], coords[:, 1], :]
    valid_mask = ~np.isnan(points_3d).any(axis=1) & (np.linalg.norm(points_3d, axis=1) > 0)
    points_3d = points_3d[valid_mask]

    if points_3d.shape[0] < 10:
        return None

    min_corner = points_3d.min(axis=0)
    max_corner = points_3d.max(axis=0)
    hull = ConvexHull(points_3d)
    hull_points = points_3d[hull.vertices]

    return {
        "min_corner": min_corner.tolist(),
        "max_corner": max_corner.tolist(),
        "convex_hull_points": hull_points.tolist()
    }

def run_yolo_segmentation_and_get_bboxes(model, rgb_image, pc):
    results = model(rgb_image)
    print(results)
    bboxes_3d = []
    if results.masks is None:
        return bboxes_3d
    for mask_obj in results.masks:
        if hasattr(mask_obj, 'xy'):
            polygon = mask_obj.xy[0].cpu().numpy()
            mask = polygon_to_mask(polygon, rgb_image.shape)
        else:
            mask = mask_obj.data.cpu().numpy().astype(np.uint8)

        bbox_3d = get_3d_bbox_from_mask(pc, mask)
        if bbox_3d is not None:
            bboxes_3d.append(bbox_3d)

    return bboxes_3d

def process_dataset(data_root):
    model = YOLO(r'C:\Users\mp01\Documents\Personal\task\3D_object_detection\models\ultralytics\yolo11n-seg.pt')  # Load your model once
    all_results = {}

    for sample_folder in sorted(os.listdir(data_root)):
        sample_path = os.path.join(data_root, sample_folder)
        if not os.path.isdir(sample_path):
            continue

        rgb_path = os.path.join(sample_path, 'rgb.jpg')
        pc_path = os.path.join(sample_path, 'pc.npy')

        if not (os.path.exists(rgb_path) and os.path.exists(pc_path)):
            print(f"Skipping {sample_folder}, missing files.")
            continue

        rgb_image = cv2.imread(rgb_path)
        pc = np.load(pc_path)  # shape (H, W, 3)

        bboxes_3d = run_yolo_segmentation_and_get_bboxes(model, rgb_image, pc)
        
        all_results[sample_folder] = bboxes_3d
        print(f"Processed {sample_folder}: Found {len(bboxes_3d)} 3D bboxes.")

    # Save all results to JSON
    with open(os.path.join(data_root, '3d_bboxes_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("3D bounding box extraction completed and saved.")

if __name__ == "__main__":
    data_root = r"D:\data\test"  # Change to your root dataset path
    process_dataset(data_root)
