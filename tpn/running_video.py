import cv2
import os
import numpy as np

# Path to the folder containing the image frames
folder_path = '../dataset_1vid/turtle/img'

# Path to the ground truth file containing bounding boxes
groundtruth_path = '../dataset_1vid/turtle/groundtruth.txt'

## Get a list of all files in the folder, and sort them
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')])

# Check if there are any image files in the folder
if not image_files:
    print(f"No image files found in {folder_path}")
    exit()

# Read the bounding boxes from the groundtruth file
with open(groundtruth_path, 'r') as f:
    bounding_boxes = f.readlines()

# Ensure there are as many bounding boxes as there are image files
if len(bounding_boxes) != len(image_files):
    print("The number of bounding boxes does not match the number of image files.")
    exit()

# Define the frame rate (frames per second)
fps = 40

# Create a named window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

 #Set up video writer
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = 1920  # Adjust to your combined frame width
frame_height = 1080  # Adjust to your combined frame height
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# Play the video with bounding boxes and smoothed masks
for image_file, bbox in zip(image_files, bounding_boxes):
    # Parse the bounding box coordinates
    x, y, width, height = map(int, bbox.strip().split(','))

    # Convert to x_min, y_min, x_max, y_max
    x_min = x
    y_min = y
    x_max = x + width
    y_max = y + height

    # Construct the full path to the image file
    image_path = os.path.join(folder_path, image_file)
    
    # Read the image
    frame = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if frame is None:
        print(f"Error loading image: {image_path}")
        continue
    
    # Create a mask of zeros with the same size as the image
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Set the values within the bounding box to ones
    mask[y_min:y_max, x_min:x_max] = 1
    
    # Apply Gaussian blur to the mask to smooth the edges
    smoothed_mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Apply morphological operations to further smooth the mask
    kernel = np.ones((25, 25), np.uint8)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
    
    # Optionally apply additional Gaussian blur for further smoothing
    smoothed_mask = cv2.GaussianBlur(smoothed_mask, (51, 51), 0)
    
    # Create a copy of the frame to draw the bounding box on
    frame_with_bbox = frame.copy()
    cv2.rectangle(frame_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Convert the mask to a 3-channel image for display purposes
    mask_display = cv2.merge([smoothed_mask * 255] * 3)
    
    # Combine the frames side by side
    combined_frame = cv2.hconcat([frame, mask_display, frame_with_bbox])
    
    # Resize the combined frame to a smaller size
    scale_percent = 50  # percent of original size
    new_width = int(combined_frame.shape[1] * scale_percent / 100)
    new_height = int(combined_frame.shape[0] * scale_percent / 100)
    dim = (new_width, new_height)
    resized_combined_frame = cv2.resize(combined_frame, dim)

    # Set the window size
    cv2.resizeWindow('Video', new_width, new_height)

    # Display the combined frame
    cv2.imshow('Video', resized_combined_frame)

    # Write the frame to the video file
    out.write(resized_combined_frame)

    
    # Wait for the specified time before displaying the next frame
    key = cv2.waitKey(int(1000 / fps))
    
    # Exit the loop if the 'q' key is pressed
    if key == ord('q'):
        break

# Release the video window
cv2.destroyAllWindows()