import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

# Conversion to yolo dataset

# --- Configuration ---
RAW_DATA_ROOT = r'D:\dl_challenge'  # e.g., 'data/my_items_dataset'
OUTPUT_YOLO_ROOT = 'yolo_dataset_seg'   # e.g., 'data/yolo_items_segmentation'

# Define your class names and their corresponding integer IDs (0-indexed for YOLO)
# IMPORTANT: Adjust this based on your actual classes.
# If all your masks represent the same type of object (e.g., "item"), use one class.
# If you have "box" and "item", define them here:
CLASS_NAMES = ['item'] # Example: If all segmented objects are just "items"
# CLASS_NAMES = ['box', 'item_A', 'item_B'] # Example: If you have multiple distinct classes

TEST_SPLIT_RATIO = 0.2  # 20% for validation/test
RANDOM_SEED = 42

# --- Create Output Directories ---
os.makedirs(os.path.join(OUTPUT_YOLO_ROOT, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_YOLO_ROOT, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_YOLO_ROOT, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_YOLO_ROOT, 'labels', 'val'), exist_ok=True)

# --- Collect All Data Point Folders ---
data_point_folders = [os.path.join(RAW_DATA_ROOT, d) for d in os.listdir(RAW_DATA_ROOT)
                      if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))]

if not data_point_folders:
    print(f"Error: No data point folders found in {RAW_DATA_ROOT}")
    exit()

# --- Split Data into Train and Validation ---
train_folders, val_folders = train_test_split(data_point_folders, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED)

print(f"Found {len(data_point_folders)} data points.")
print(f"Training set: {len(train_folders)} data points")
print(f"Validation set: {len(val_folders)} data points")

# --- Conversion Function ---
def convert_to_yolo_segmentation(data_folders, split_name):
    print(f"\nProcessing {split_name} set...")
    for folder_path in tqdm(data_folders, desc=f"Converting {split_name}"):
        folder_name = os.path.basename(folder_path)
        rgb_path = os.path.join(folder_path, 'rgb.jpg')
        mask_path = os.path.join(folder_path, 'mask.npy')

        if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
            print(f"Warning: Skipping {folder_path} as rgb.jpg or mask.npy is missing.")
            continue

        try:
            img = cv2.imread(rgb_path)
            # masks_data should be (N, H, W) where N is number of instances
            masks_data = np.load(mask_path)

            if img is None:
                print(f"Warning: Could not read image {rgb_path}. Skipping.")
                continue

            # Ensure image dimensions are consistent with mask dimensions if they differ
            # If your masks are generated for a specific resolution, make sure the input images match,
            # or handle resizing here. Assuming they are consistent for now.
            # If your masks are smaller/larger, you'd need to resize them before finding contours.
            img_H, img_W, _ = img.shape
            
            # --- IMPORTANT: Class Assignment for Each Instance ---
            # If you have multiple classes and their order/mapping is available, load it here.
            # E.g., if you have a `classes.txt` in each folder:
            # instance_classes = []
            # with open(os.path.join(folder_path, 'classes.txt'), 'r') as f:
            #     for line in f:
            #         instance_classes.append(line.strip())
            
            # For simplicity, if all masks are of the same type ('item'), use this:
            yolo_class_id_for_all_instances = 0 # Assuming 'item' is CLASS_NAMES[0]

            # If you have multiple classes and know the order of masks corresponds to a class list,
            # you'd replace the above line with something like:
            # yolo_class_ids_for_instances = [CLASS_NAMES.index(c) for c in instance_classes]


            # Copy image to YOLO dataset structure
            shutil.copy(rgb_path, os.path.join(OUTPUT_YOLO_ROOT, 'images', split_name, f"{folder_name}.jpg"))

            # Prepare label file
            label_file_path = os.path.join(OUTPUT_YOLO_ROOT, 'labels', split_name, f"{folder_name}.txt")
            with open(label_file_path, 'w') as f:
                # Iterate through each instance mask
                num_instances = masks_data.shape[0]

                if num_instances == 0:
                    # print(f"Warning: No instances found in {mask_path}. Label file will be empty.")
                    continue

                for i in range(num_instances):
                    instance_mask = masks_data[i, :, :] # Get the i-th binary mask (H, W)

                    # Ensure mask is binary (0 or 255) for contour finding
                    instance_mask = (instance_mask > 0).astype(np.uint8) * 255
                    
                    # Find contours for the current instance mask
                    # RETR_EXTERNAL to get only outer contours, CHAIN_APPROX_SIMPLE to compress points
                    contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    # YOLO expects one polygon per line. Choose the largest contour if multiple are found.
                    # This happens if a mask is disconnected.
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Flatten the contour points and normalize
                    # Convert to float first to avoid integer division issues
                    points = largest_contour.flatten().astype(float).tolist()
                    
                    # Normalize points relative to image width and height
                    normalized_points = []
                    for j, p in enumerate(points):
                        if j % 2 == 0: # X coordinate
                            normalized_points.append(f"{p / img_W:.6f}")
                        else: # Y coordinate
                            normalized_points.append(f"{p / img_H:.6f}")

                    # Determine the class for this instance
                    # If you have a list of classes for each mask based on external file:
                    # current_yolo_class_id = yolo_class_ids_for_instances[i]
                    # Else (single class for all masks):
                    current_yolo_class_id = yolo_class_id_for_all_instances

                    # Write to label file: class_id poly_x1 poly_y1 poly_x2 poly_y2 ...
                    f.write(f"{current_yolo_class_id} " + " ".join(normalized_points) + "\n")

        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            continue

# --- Run the Conversion ---
convert_to_yolo_segmentation(train_folders, 'train')
convert_to_yolo_segmentation(val_folders, 'val')

# --- Generate data.yaml ---
data_yaml_content = f"""
path: {os.path.abspath(OUTPUT_YOLO_ROOT)}
train: images/train
val: images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
with open(os.path.join(OUTPUT_YOLO_ROOT, 'data.yaml'), 'w') as f:
    f.write(data_yaml_content)

print(f"\nDataset conversion complete! YOLO dataset ready at: {os.path.abspath(OUTPUT_YOLO_ROOT)}")
print(f"Remember to verify the `CLASS_NAMES` in the script, and adjust class assignment if you have multiple classes per image.")