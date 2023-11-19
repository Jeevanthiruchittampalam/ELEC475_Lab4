import cv2
import os
from KittiAnchors import Anchors

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Instantiate the Anchors class
    anchors = Anchors()

    # Calculate anchor centers based on the image size and predefined grid
    anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

    # Get ROIs and their bounding boxes from the image
    ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

    return ROIs, boxes

def process_directory(input_directory_path, output_directory_path):
    # Ensure the output directory exists
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    # Iterate over each file in the directory
    for filename in os.listdir(input_directory_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Construct the full file path
            file_path = os.path.join(input_directory_path, filename)

            # Process the image to get ROIs and bounding boxes
            ROIs, boxes = process_image(file_path)

            # Save each ROI as a new image in the output directory
            for i, roi in enumerate(ROIs):
                roi_filename = f"{filename.split('.')[0]}_roi_{i}.png"
                roi_path = os.path.join(output_directory_path, roi_filename)
                cv2.imwrite(roi_path, roi)

            print(f"Processed {filename}: {len(boxes)} ROIs found.")

input_directory_path = './data/Kitti8/test/image'
output_directory_path = './data/2_4_1/test'
process_directory(input_directory_path, output_directory_path)
