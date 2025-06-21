from ultralytics import SAM
import cv2
import os

# --- Configuration ---
# Path to your input image for inference
INPUT_IMAGE_PATH = r'D:\data\test\9e1ed846-9915-11ee-9103-bbb8eae05561\rgb.jpg' # Replace with an actual path to one of your rgb.jpg files

# Choose a SAM model variant.
# 'sam_b.pt': Base model, good balance.
# 'sam_l.pt': Large model, potentially better results, larger.
# 'sam_h.pt': Huge model, best results, largest and slowest.
# 'mobile_sam.pt': Much smaller and faster, but less accurate.
SAM_MODEL_NAME = "sam_b.pt" 

# --- Load the SAM Model ---
print(f"Loading SAM model: {SAM_MODEL_NAME}")
model = SAM(SAM_MODEL_NAME)

# --- Perform Inference ---
print(f"Performing inference on: {INPUT_IMAGE_PATH}")

# Method 1: Automatic segmentation (no prompts needed)
# This tries to find and segment all distinct objects in the image.
results = model(INPUT_IMAGE_PATH) 

# Method 2 (Optional): Inference with prompts (e.g., a point)
# If you want to segment a *specific* object by providing a point (x, y)
# You'd typically open the image, click on it, and get the coordinates.
# This is more interactive for specific objects.
# Example: Segment object at pixel (100, 200)
# results = model(INPUT_IMAGE_PATH, points=[100, 200], labels=[1]) # labels=[1] means foreground point

# Method 3 (Optional): Inference with bounding box prompt
# If you want to segment within a specific region (x1, y1, x2, y2)
# results = model(INPUT_IMAGE_PATH, bboxes=[0, 0, 640, 640]) # Example: entire image

# --- Visualize Results ---
if results:
    for r in results:
        # r.plot() returns an annotated image (NumPy array)
        annotated_image = r.plot() 
        cv2.imshow("SAM Inference Result", annotated_image)
        cv2.waitKey(0) # Press any key to close the window
        cv2.destroyAllWindows()

        print(f"\nResults for {INPUT_IMAGE_PATH}:")
        if r.masks is not None:
            print(f"  Detected {len(r.masks)} instances.")
            # r.masks.data is the raw binary masks tensor (num_masks, H, W)
            # r.boxes.data contains bounding boxes if detected (num_masks, 6) -> x1, y1, x2, y2, conf, class_id
            
            # Since SAM is zero-shot, the 'class_id' in r.boxes.data will typically be 0 
            # or some generic identifier, as it doesn't classify into specific categories.
            
            # Example: Iterate through masks and save them
            # for i, mask_tensor in enumerate(r.masks.data):
            #     # Convert tensor to numpy array and scale to 255 for visualization/saving
            #     mask_np = mask_tensor.cpu().numpy() * 255 
            #     cv2.imwrite(f"mask_{i}.png", mask_np)
        else:
            print("  No masks detected.")
else:
    print("No results object returned.")