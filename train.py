import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt


def convert_label(label):
    # Split the label into parts using underscore
    parts = label.split('_')
    # Extract the number part (e.g., '1' from '0_1.png')
    number_part = parts[1].split('.')[0]
    # Convert it to an integer and format it as a 6-digit string
    formatted_number = '{:06d}'.format(int(number_part))
    # Create the new label (e.g., '0_1.png' becomes '000001.png')
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
                # Convert the label format
                new_label = convert_label(items[0])
                # Assuming the image filenames are correctly listed in labels.txt
                self.labels.append((new_label, int(items[1])))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        # Update the path to where your images are located
        img_path = os.path.join(self.image_dir, 'image', img_name)  # Added 'image' subfolder
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

# Load Datasets
train_dataset = YodaDataset(image_dir='./data/Kitti8/train', label_file='./data/Kitti8_ROIs/train/label/labels.txt', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize Model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Binary classification

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
num_epochs = 1000
train_loss_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_loss_list.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the Model
torch.save(model.state_dict(), 'yoda_resnet18.pth')

# Plot Training Loss
plt.plot(train_loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
