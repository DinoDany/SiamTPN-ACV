import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import json 
import logging

# Set up logging configuration
logging.basicConfig(level=logging.CRITICAL + 1, format='%(asctime)s - %(levelname)s - %(message)s')

# Uncomment to see the debuging messages
#logging.getLogger().setLevel(logging.DEBUG)



# Define a new model to extract specific stages
class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Load pre-trained ShuffleNetV2 model
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

        #print(model)
        # Switch to evaluation mode for inference
        model.eval()        

        self.initial = torch.nn.Sequential(
            model.conv1,
            model.maxpool
        )
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.conv5 = model.conv5

        self.conv1 = nn.Conv2d(in_channels=232, out_channels=464, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=464, out_channels=464, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=464, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        logging.info("Staring the backbone")
        x = self.initial(x)
        x = self.stage2(x)

        x = self.stage3(x)
        stage3_output = x
        logging.debug("Stage 3/1 %s", stage3_output.shape)
        stage3_output = self.conv1(stage3_output)
        logging.debug("Stage 3/2 %s", stage3_output.shape)
        stage3_output = stage3_output.view(stage3_output.shape[0], stage3_output.shape[1], -1)  
        logging.debug("Stage 3/3 %s", stage3_output.shape)

        x = self.stage4(x)
        stage4_output = x
        logging.debug("Stage 4/1 %s", stage4_output.shape)
        stage4_output = self.conv2(stage4_output)
        logging.debug("Stage 4/2 %s", stage4_output.shape)
        stage4_output = stage4_output.view(stage4_output.shape[0], stage4_output.shape[1], -1) 
        logging.debug("Stage 4/3 %s", stage4_output.shape)

        x = self.conv5(x)
        stage5_output = x
        logging.debug("Stage 5/1 %s", stage5_output.shape)
        stage5_output = self.conv3(stage5_output)
        logging.debug("Stage 5/1 %s", stage5_output.shape)
        stage5_output = stage5_output.view(stage5_output.shape[0], stage5_output.shape[1], -1) 
        logging.debug("Stage 5/1 %s", stage5_output.shape)

        logging.info("Backbone finished")

        return stage3_output, stage4_output, stage5_output





"""
# Make a prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()

print(f'Predicted class: {predicted.item()}')

# Load ImageNet class labels
with open('../imagenet-simple-labels.json') as f:
    labels = json.load(f)

# Print the corresponding label
predicted_label = labels[predicted_class]
print(f'Predicted label: {predicted_label}')

# Fine-tuning the model (Optional)
# Modify the final layer for fine-tuning
num_classes = 10  # Replace with the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define your dataset
# Assuming your dataset is organized in folders by class in `data/train` and `data/val`
train_dataset = ImageFolder(root='data/train', transform=transform)
val_dataset = ImageFolder(root='data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
  
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Validation step
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy:.4f}')

print('Training complete')
"""
