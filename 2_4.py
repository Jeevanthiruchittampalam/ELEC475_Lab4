import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from KittiAnchors import Anchors  # You should import this module if available

# Convert label function
def convert_label(label):
    parts = label.split('_')
    number_part = parts[1].split('.')[0]
    formatted_number = '{:06d}'.format(int(number_part))
    new_label = f'{formatted_number}.png'
    return new_label

# ROI Dataset Class
class ROIDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.png') or img.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Datasets
roi_dataset = ROIDataset(image_dir='./data/2_4_1/train', transform=transform)
roi_loader = DataLoader(roi_dataset, batch_size=4, shuffle=False)

# Initialize and Load Model
model = resnet18(pretrained=True)  # Load a pre-trained ResNet18 model
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming binary classification

model.load_state_dict(torch.load('yoda_resnet18.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Classify the ROIs
roi_predictions = {}
total_images = len(roi_loader)  # Total number of images in the dataset
processed_images = 0

with torch.no_grad():
    for images, img_names in roi_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for img_name, pred in zip(img_names, predicted):
            roi_predictions[img_name] = pred.item()

        # Update progress
        processed_images += len(images)
        print(f"Processed {processed_images}/{total_images} images")

# Helper Functions - Implement these based on your project's specifics
def get_roi_coordinates(roi_filename, coords_directory_path):
    # Construct the path to the coordinates file
    coords_file_path = os.path.join(coords_directory_path, roi_filename.replace('.png', '.txt'))

    # Read the bounding box coordinates from the file
    with open(coords_file_path, 'r') as file:
        line = file.readline().strip()
        x1, y1, x2, y2 = map(int, line.split())

    return x1, y1, x2, y2

def map_roi_to_original_image(roi_name):
    # Extract the base part of the ROI filename (e.g., '000000' from '000000_roi_0.png')
    base_part = roi_name.split('_roi_')[0]
    # Construct the corresponding original image filename
    original_image_name = f'{base_part}.png'
    return original_image_name

def load_gt_boxes(image_name, label_dir):
    # Construct the label filename based on the image name
    label_filename = convert_label(image_name)
    label_filepath = os.path.join(label_dir, label_filename)

    # Read the bounding box coordinates from the label file
    gt_boxes = []
    with open(label_filepath, 'r') as file:
        for line in file:
            # Assuming each line in the file is a bounding box coordinate
            x1, y1, x2, y2 = map(int, line.strip().split())
            gt_boxes.append((x1, y1, x2, y2))

    return gt_boxes

# IoU Calculation (You should have the Anchors module available for this)
def calculate_iou(pred_box, gt_box):
    anchors = Anchors()
    return anchors.calc_IoU(pred_box, gt_box)

# Calculate IoU for Each ROI Classified as 'Car'
iou_scores = []
for roi_name, pred_label in roi_predictions.items():
    if pred_label == 1:  # Assuming '1' is the label for 'Car'
        roi_coords = get_roi_coordinates(roi_name, './data/2_4_1/traincoords')
        original_image_name = map_roi_to_original_image(roi_name)
        gt_boxes = load_gt_boxes(original_image_name, './data/Kitti8/train/label')

        for gt_box in gt_boxes:
            iou_score = calculate_iou(roi_coords, gt_box)
            iou_scores.append(iou_score)

# Calculate Mean IoU
mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
print(f"Mean IoU: {mean_iou}")

# Visualization and further processing as required for your project
# Visualization
def visualize_predictions(image_path, predictions, coords_directory_path):
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes on the image
    for roi_name, pred_label in predictions.items():
        if pred_label == 1:  # If the ROI is classified as 'Car'
            x1, y1, x2, y2 = get_roi_coordinates(roi_name, coords_directory_path)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Display the image with bounding boxes
    plt.imshow(image)
    plt.show()

# Run visualization on a few sample images
sample_image_names = ['000000.png', '000029.png']
for img_name in sample_image_names:
    original_image_path = os.path.join('./data/Kitti8/train/image', img_name)
    visualize_predictions(original_image_path, roi_predictions, './data/2_4_1/traincoords')
