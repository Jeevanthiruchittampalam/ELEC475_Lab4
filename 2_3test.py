import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os


model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


def convert_label(label):
    # Split the label into parts using underscore
    parts = label.split('_')

    # Extract the second number part (e.g., '0' from '0_0.png')
    second_number_part = parts[1].split('.')[0]

    # Convert it to an integer and add 6000 to it
    adjusted_number = int(second_number_part) + 6000

    # Format it as a 6-digit string
    formatted_number = '{:06d}'.format(adjusted_number)

    # Create the new label (e.g., '0_0.png' becomes '006000.png')
    new_label = f'{formatted_number}.png'
    return new_label


# Custom Dataset Class
class YodaDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = []
        with open(label_file, 'r') as f:
            for line in f:
                items = line.split()
                new_label = convert_label(items[0])
                self.labels.append((new_label, int(items[1])))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the Test Dataset
test_dataset = YodaDataset(image_dir='./data/Kitti8/test/image', label_file='./data/Kitti8_ROIs/test/label/labels.txt', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the Trained Model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Adjust for your number of classes

model.load_state_dict(torch.load('yoda_resnet18.pth'))

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Test the Model
total, correct = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate Accuracy
accuracy = 100 * correct / total
print(f'Accuracy on test images: {accuracy:.2f}%')
