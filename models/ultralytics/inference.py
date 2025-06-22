import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
from ultralytics.data.augment import LetterBox
from ultralytics.utils.plotting import colors
from scipy.spatial import ConvexHull
import json
import torch
s_mode = 1  # 0: semantic, 1: instance

def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    """Project 3D points to 2D pixel coordinates."""
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    x_proj = (x * fx / z) + cx
    y_proj = (y * fy / z) + cy
    return np.stack([x_proj, y_proj], axis=1).astype(int)

def polygon_to_mask(polygon, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def get_3d_bbox_from_mask(pc, mask):
    coords = np.argwhere(mask > 0)
    pc = np.transpose(pc, (1, 2, 0)) 
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
    fx = fy = 525  # or use actual values
    cx = rgb_image.shape[1] // 2
    cy = rgb_image.shape[0] // 2
    results = model(rgb_image)[0]
    bboxes_3d = []

    masks = results.masks
    if masks is None:
        return [], rgb_image

    clss = [1] 
    im0 = rgb_image.copy()
    annotator = Annotator(im0)

    # Preprocess mask image
    img = LetterBox(masks.shape[1:])(image=annotator.result())
    im_gpu = (torch.as_tensor(img, dtype=torch.float16, device=masks.data.device)
                .permute(2, 0, 1).flip(0).contiguous() / 255)

    s_mode = 0
    
    annotator.masks(
        masks.data,
        colors=[colors(x, True) for x in clss],
        im_gpu=im_gpu
    )

    cv2.imwrite("segmentation_output.jpg", im0)

    for mask_obj in masks:
        if hasattr(mask_obj, 'xy') and mask_obj.xy[0] is not None:
            polygon = mask_obj.xy[0]
            if len(polygon) < 3:
                continue
            mask = polygon_to_mask(polygon, rgb_image.shape)
            bbox_3d = get_3d_bbox_from_mask(pc, mask)
            if bbox_3d is not None:
                bboxes_3d.append(bbox_3d)

                hull = np.array(bbox_3d['convex_hull_points'])
                pts_2d = project_3d_to_2d(hull, fx, fy, cx, cy)
                for pt in pts_2d:
                    cv2.circle(rgb_image, tuple(pt), 2, (0, 255, 0), -1)
                for i in range(len(pts_2d)):
                    cv2.line(rgb_image, tuple(pts_2d[i]), tuple(pts_2d[(i+1)%len(pts_2d)]), (255, 0, 0), 1)

    return bboxes_3d, im0

def process_dataset(data_root):
    model = YOLO(r'C:\Users\mp01\Documents\Personal\task\3D_object_detection\models\ultralytics\runs\segment\yolo_generic_objects_segmentation5\weights\best.pt')  # Load your model once
    all_results = {}
    video_path = os.path.join(data_root, "3d_overlay_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (1024, 1024)  # as all images are of varying size
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, frame_size)  

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
        pc = np.load(pc_path)  

        bboxes_3d,vis_image  = run_yolo_segmentation_and_get_bboxes(model, rgb_image, pc)
        
        resized = cv2.resize(vis_image, frame_size)
        video_writer.write(resized)
        all_results[sample_folder] = bboxes_3d
        print(f"Processed {sample_folder}: Found {len(bboxes_3d)} 3D bboxes.")

    video_writer.release()
    print("Video saved to", video_path)     
    
    with open(os.path.join(data_root, '3d_bboxes_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("3D bounding box extraction completed and saved.")

if __name__ == "__main__":
    data_root = r"D:\data\test"
    process_dataset(data_root)
